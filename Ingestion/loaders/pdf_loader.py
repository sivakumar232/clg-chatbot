from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents import Document

def load_pdf_files(data_dir: str) -> list[Document]:
    """
    Scans the given directory and its subdirectories for all .pdf files.
    Returns a list of LangChain Document objects.
    """
    print(f"Scanning for PDF files in {data_dir}...")
    
    loader = PyPDFDirectoryLoader(
        data_dir,
        glob="**/*.pdf"
    )
    
    docs = loader.load()
    print(f"Successfully loaded {len(docs)} PDF pages.")
    return docs
