from langchain_core.documents import Document
from .strategies.md_strategy import chunk_markdown
from .strategies.pdf_strategy import chunk_pdfs

def chunk_documents(documents: list[Document], doc_type: str) -> list[Document]:
    """
    Main entry point for chunking. 
    Routes to the specific strategy based on the document type.
    """
    if doc_type == "markdown":
        return chunk_markdown(documents)
    elif doc_type == "pdf":
        return chunk_pdfs(documents)
    else:
        raise ValueError(f"Unknown document type for chunking: {doc_type}")
