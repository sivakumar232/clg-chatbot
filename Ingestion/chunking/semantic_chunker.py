import re
import numpy as np
import tiktoken
import pysbd
from pathlib import Path
from typing import List, Optional, Union
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter


# Lazy-loaded local embedding model for fast sentence similarity calculation
_LOCAL_MODEL = None


def get_local_embedder():
    """
    Returns a fast local SentenceTransformer model instance (all-MiniLM-L6-v2).
    Cached as a singleton to avoid re-loading weights across chunks.
    Runs locally on CPU in milliseconds with zero network latency and no API limits.
    """
    global _LOCAL_MODEL
    if _LOCAL_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _LOCAL_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _LOCAL_MODEL


class StructureAwareSemanticChunker:
    """
    Production-grade structure-aware semantic chunker.
    
    Addresses key RAG chunking issues:
    1. Token-Aware: Uses tiktoken (cl100k_base) to enforce token bounds rather than character counts.
    2. Robust SBD: Uses pysbd (Python Sentence Boundary Disambiguation) to prevent false breaks
       on abbreviations (Dr., e.g., Ph.D., i.e.), URLs, and numbers.
    3. Windowed Context: Compares sliding sentence buffers ([S_{i-1}, S_i] vs [S_{i+1}, S_{i+2}])
       with percentile distance thresholds to eliminate single-sentence noise.
    4. Structure Preservation: Protects Markdown tables and code blocks as indivisible atomic units.
    """

    def __init__(
        self,
        target_tokens: int = 500,
        max_tokens: int = 1000,
        buffer_size: int = 2,
        percentile_threshold: float = 88.0,
        embedder=None,
    ):
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        self.buffer_size = buffer_size
        self.percentile_threshold = percentile_threshold
        self.embedder = embedder

        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.segmenter = pysbd.Segmenter(language="en", clean=False)

    def count_tokens(self, text: str) -> int:
        """Counts exact tokens using tiktoken cl100k_base."""
        if not text:
            return 0
        return len(self.tokenizer.encode(text, disallowed_special=()))

    def _split_into_sentences(self, text: str) -> List[str]:
        """Splits text into sentences using pysbd with regex fallback."""
        if not text.strip():
            return []
        try:
            sentences = self.segmenter.segment(text)
            cleaned = [s.strip() for s in sentences if s and s.strip()]
            if cleaned:
                return cleaned
        except Exception:
            pass

        # Fallback regex that avoids splitting common abbreviations
        fallback_regex = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9])')
        return [s.strip() for s in fallback_regex.split(text) if s.strip()]

    def _cosine_distance(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Calculates cosine distance between two vectors."""
        dot = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0.0 or norm_b == 0.0:
            return 1.0
        similarity = dot / (norm_a * norm_b)
        return float(1.0 - np.clip(similarity, -1.0, 1.0))

    def _get_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Computes embeddings using the configured or local model."""
        if not texts:
            return []

        if self.embedder is not None:
            if hasattr(self.embedder, "embed_documents"):
                raw_vectors = self.embedder.embed_documents(texts)
            elif hasattr(self.embedder, "encode"):
                raw_vectors = self.embedder.encode(texts)
            else:
                raise ValueError("Configured embedder must have 'embed_documents' or 'encode'.")
        else:
            model = get_local_embedder()
            raw_vectors = model.encode(texts, show_progress_bar=False)

        return [np.array(v, dtype=np.float32) for v in raw_vectors]

    def split_text_semantically(self, text: str) -> List[str]:
        """
        Splits a text section semantically:
        - Segments into robust sentences.
        - Constructs sliding pre/post window buffers.
        - Identifies topic shift breakpoints via percentile distance thresholding.
        - Respects target_tokens and hard max_tokens ceilings.
        """
        total_tokens = self.count_tokens(text)
        if total_tokens <= self.target_tokens:
            return [text.strip()] if text.strip() else []

        sentences = self._split_into_sentences(text)
        if len(sentences) <= 1:
            # Single giant sentence: chunk by token window
            return self._chunk_by_tokens(text, self.max_tokens)


        # Build sliding window pairs: [S_{i-k+1..i}] vs [S_{i+1..i+k}]
        window_texts = []
        for i in range(len(sentences) - 1):
            start_idx = max(0, i - self.buffer_size + 1)
            end_idx = min(len(sentences), i + 1 + self.buffer_size)

            pre_window = " ".join(sentences[start_idx : i + 1])
            post_window = " ".join(sentences[i + 1 : end_idx])

            window_texts.append(pre_window)
            window_texts.append(post_window)

        embeddings = self._get_embeddings(window_texts)

        distances = []
        for i in range(0, len(embeddings), 2):
            dist = self._cosine_distance(embeddings[i], embeddings[i + 1])
            distances.append(dist)

        threshold = (
            float(np.percentile(distances, self.percentile_threshold))
            if distances
            else 0.5
        )

        # Assemble chunks
        chunks = []
        current_sentences = [sentences[0]]
        current_tokens = self.count_tokens(sentences[0])

        for i, dist in enumerate(distances):
            next_sent = sentences[i + 1]
            next_tokens = self.count_tokens(next_sent)

            is_semantic_shift = dist >= threshold
            is_large_enough = current_tokens >= self.target_tokens
            exceeds_max = (current_tokens + next_tokens) > self.max_tokens

            if (is_semantic_shift and is_large_enough) or exceeds_max:
                chunk_str = " ".join(current_sentences).strip()
                if chunk_str:
                    chunks.append(chunk_str)
                current_sentences = [next_sent]
                current_tokens = next_tokens
            else:
                current_sentences.append(next_sent)
                current_tokens += next_tokens

        if current_sentences:
            final_chunk = " ".join(current_sentences).strip()
            if final_chunk:
                chunks.append(final_chunk)

        return chunks

    def _chunk_by_tokens(self, text: str, max_tokens: int) -> List[str]:
        """Fallback token-based chunking for text blocks that cannot be split by sentences."""
        tokens = self.tokenizer.encode(text, disallowed_special=())
        chunks = []
        for start in range(0, len(tokens), max_tokens):
            sub_tokens = tokens[start : start + max_tokens]
            chunks.append(self.tokenizer.decode(sub_tokens))
        return chunks


# =========================================================
# Atomic Block Extraction (Tables & Code Blocks)
# =========================================================

def extract_atomic_blocks(text: str):
    """
    Splits content into alternating regular text blocks and atomic blocks (tables and code blocks).
    Atomic blocks should never be severed mid-table or mid-code.
    
    Returns a list of tuples: (content: str, is_atomic: bool)
    """
    table_pattern = r'(?:^[ \t]*\|[^\n]+\|[ \t]*\n)+(?:^[ \t]*\|[-:\s|]+\|[ \t]*\n)(?:^[ \t]*\|[^\n]+\|[ \t]*(?:\n|$))+'
    code_pattern = r'```[\s\S]*?```'
    combined = re.compile(f'({code_pattern}|{table_pattern})', re.MULTILINE)

    blocks = []
    last_idx = 0
    for match in combined.finditer(text):
        start, end = match.span()
        if start > last_idx:
            pre_text = text[last_idx:start].strip()
            if pre_text:
                blocks.append((pre_text, False))
        atomic_content = match.group(0).strip()
        if atomic_content:
            blocks.append((atomic_content, True))
        last_idx = end

    if last_idx < len(text):
        remaining = text[last_idx:].strip()
        if remaining:
            blocks.append((remaining, False))

    return blocks if blocks else [(text.strip(), False)]



# =========================================================
# Markdown Semantic Chunking Strategy
# =========================================================

def chunk_markdown_semantic(
    documents: List[Document],
    target_tokens: int = 500,
    max_tokens: int = 1000,
    embedder=None,
) -> List[Document]:
    """
    Splits markdown documents using Structure-Aware Semantic Chunking:
    - Extracts header hierarchy (#, ##, ###).
    - Preserves markdown tables and code blocks as atomic units.
    - Applies sliding context window semantic splitting to long sections.
    - Enforces target_tokens=500 and max_tokens=1000 (+200 increase).
    - Injects hierarchical breadcrumb headers for LLM / vector retrieval context.
    """
    print(f"Applying Structure-Aware Semantic Chunking to {len(documents)} Markdown documents...")

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )

    chunker = StructureAwareSemanticChunker(
        target_tokens=target_tokens,
        max_tokens=max_tokens,
        embedder=embedder,
    )

    final_chunks: List[Document] = []

    for doc in documents:
        # 1. Structural split by headers
        header_chunks = markdown_splitter.split_text(doc.page_content)
        if not header_chunks:
            header_chunks = [doc]

        source_ref = (
            doc.metadata.get("source_url")
            or doc.metadata.get("source", "Webpage")
        )
        page_title = Path(str(source_ref)).stem.replace("-", " ").replace("_", " ").title()

        for hc in header_chunks:
            combined_metadata = doc.metadata.copy()
            combined_metadata.update(hc.metadata)

            # Build breadcrumb hierarchy
            breadcrumbs = [page_title]
            for h in ["Header 1", "Header 2", "Header 3"]:
                val = combined_metadata.get(h)
                if val and val not in breadcrumbs:
                    breadcrumbs.append(val)

            breadcrumb_str = " > ".join(breadcrumbs)
            prefix = f"[Source: SRKR {breadcrumb_str}]\n\n"

            # 2. Extract atomic blocks (tables, code) from regular text
            blocks = extract_atomic_blocks(hc.page_content)

            for block_content, is_atomic in blocks:
                if is_atomic:
                    # Keep atomic table/code intact
                    chunk_text = f"{prefix}{block_content}"
                    final_chunks.append(Document(page_content=chunk_text, metadata=combined_metadata))
                else:
                    # Semantic chunking on long narrative text
                    sub_chunks = chunker.split_text_semantically(block_content)
                    for sc in sub_chunks:
                        chunk_text = f"{prefix}{sc}"
                        final_chunks.append(Document(page_content=chunk_text, metadata=combined_metadata))

    print(f" -> Generated {len(final_chunks)} Structure-Aware Semantic Markdown chunks.")
    return final_chunks


# =========================================================
# PDF Semantic Chunking Strategy
# =========================================================

def chunk_pdf_semantic(
    documents: List[Document],
    syllabus_target: int = 600,
    syllabus_max: int = 1100,
    standard_target: int = 500,
    standard_max: int = 1000,
    embedder=None,
) -> List[Document]:
    """
    Splits PDF documents using Structure-Aware Semantic Chunking:
    - Differentiates Syllabus PDFs (target=600, max=1100) from Standard PDFs (target=500, max=1000).
    - Preserves tables and tabular structures.
    - Segments narrative text at semantic topic transitions via sliding window cosine distances.
    - Injects document filename and page breadcrumbs for retrieval context.
    """
    print(f"Applying Structure-Aware Semantic Chunking to {len(documents)} PDF pages...")

    syllabus_chunker = StructureAwareSemanticChunker(
        target_tokens=syllabus_target,
        max_tokens=syllabus_max,
        embedder=embedder,
    )

    standard_chunker = StructureAwareSemanticChunker(
        target_tokens=standard_target,
        max_tokens=standard_max,
        embedder=embedder,
    )

    final_chunks: List[Document] = []

    for doc in documents:
        source_path = str(doc.metadata.get("source", ""))
        filename = Path(source_path).name or "PDF Document"
        page_num = doc.metadata.get("page", doc.metadata.get("page_number", None))
        
        page_label = f" (Page {page_num + 1})" if isinstance(page_num, int) else ""
        prefix = f"[Source PDF: {filename}{page_label}]\n\n"

        is_syllabus = "syllabus" in source_path.lower()
        chunker = syllabus_chunker if is_syllabus else standard_chunker

        # Preserve atomic blocks (e.g. extracted tables in markdown format)
        blocks = extract_atomic_blocks(doc.page_content)

        for block_content, is_atomic in blocks:
            if is_atomic:
                chunk_text = f"{prefix}{block_content}"
                final_chunks.append(Document(page_content=chunk_text, metadata=doc.metadata.copy()))
            else:
                sub_chunks = chunker.split_text_semantically(block_content)
                for sc in sub_chunks:
                    chunk_text = f"{prefix}{sc}"
                    final_chunks.append(Document(page_content=chunk_text, metadata=doc.metadata.copy()))

    print(f" -> Generated {len(final_chunks)} Structure-Aware Semantic PDF chunks.")
    return final_chunks
