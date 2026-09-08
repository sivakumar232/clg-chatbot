from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def chunk_pdfs(documents: list[Document]) -> list[Document]:
    """
    Splits PDF documents using adaptive chunk sizes:
    - Syllabus PDFs: chunk_size=1500, chunk_overlap=300 (preserves course unit coherence)
    - Other PDFs: chunk_size=1200, chunk_overlap=200
    Also prepends source PDF title into chunk text for LLM context.
    """
    print(f"Applying Adaptive Chunking to {len(documents)} PDF pages...")

    syllabus_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
    )

    standard_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
    )

    final_chunks = []

    for doc in documents:
        source_path = str(doc.metadata.get("source", ""))
        filename = Path(source_path).name or "PDF Document"

        is_syllabus = "syllabus" in source_path.lower()
        splitter = syllabus_splitter if is_syllabus else standard_splitter

        split_docs = splitter.split_documents([doc])

        for chunk in split_docs:
            # Prepend source title into chunk text for LLM RAG context
            chunk.page_content = f"[Source PDF: {filename}]\n\n{chunk.page_content}"
            final_chunks.append(chunk)

    print(f" -> Generated {len(final_chunks)} PDF chunks.")
    return final_chunks
