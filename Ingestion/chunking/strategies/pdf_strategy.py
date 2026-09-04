from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def chunk_pdfs(documents: list[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> list[Document]:
    """
    Splits PDF documents using a standard recursive text splitter,
    prioritizing paragraphs, then sentences, then words.
    """
    print(f"Applying Recursive Chunking to {len(documents)} PDF pages...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    final_chunks = text_splitter.split_documents(documents)
    print(f" -> Generated {len(final_chunks)} PDF chunks.")
    return final_chunks
