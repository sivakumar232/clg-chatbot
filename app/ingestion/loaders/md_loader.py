from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document

def load_markdown_files(data_dir: str) -> list[Document]:
    """
    Scans the given directory and its subdirectories for all .md files.
    Returns a list of LangChain Document objects.
    """
    print(f"Scanning for Markdown files in {data_dir}...")
    
    loader = DirectoryLoader(
        data_dir,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={'encoding': 'utf-8'}
    )
    
    docs = loader.load()
    print(f"Successfully loaded {len(docs)} Markdown documents.")
    return docs
