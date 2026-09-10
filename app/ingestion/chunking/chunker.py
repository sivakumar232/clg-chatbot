from langchain_core.documents import Document
from .semantic_chunker import chunk_markdown_semantic, chunk_pdf_semantic
from .strategies.md_strategy import chunk_markdown as chunk_markdown_legacy
from .strategies.pdf_strategy import chunk_pdfs as chunk_pdfs_legacy


def chunk_documents(
    documents: list[Document],
    doc_type: str,
    use_semantic: bool = True,
) -> list[Document]:
    """
    Main entry point for chunking in the ingestion pipeline.
    Routes to Structure-Aware Semantic Chunking by default.
    """
    if doc_type == "markdown":
        if use_semantic:
            return chunk_markdown_semantic(documents)
        return chunk_markdown_legacy(documents)
    elif doc_type == "pdf":
        if use_semantic:
            return chunk_pdf_semantic(documents)
        return chunk_pdfs_legacy(documents)
    else:
        raise ValueError(f"Unknown document type for chunking: {doc_type}")

