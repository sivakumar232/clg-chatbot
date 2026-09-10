from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def chunk_markdown(documents: list[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> list[Document]:
    """
    Splits markdown documents smartly based on headers.
    If a header section is still too large, it breaks it down using the recursive splitter.
    """
    print(f"Applying Semantic Markdown Chunking to {len(documents)} documents...")
    
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    
    final_chunks = []
    
    for doc in documents:
        # Split the text by headers
        header_chunks = markdown_splitter.split_text(doc.page_content)
        
        # Combine the original metadata with the new header metadata
        for hc in header_chunks:
            combined_metadata = doc.metadata.copy()
            combined_metadata.update(hc.metadata)
            
            source_ref = (
                doc.metadata.get("source_url")
                or doc.metadata.get("source", "Webpage")
            )
            page_title = source_ref.split("/")[-1].replace(".md", "").replace("-", " ").title()

            # Ensure no single chunk is larger than our chunk_size limit
            smaller_chunks = text_splitter.split_text(hc.page_content)
            for sc in smaller_chunks:
                chunk_text = f"[Source: SRKR {page_title}]\n\n{sc}"
                final_chunks.append(Document(page_content=chunk_text, metadata=combined_metadata))
                
    print(f" -> Generated {len(final_chunks)} perfectly sized Markdown chunks.")
    return final_chunks
