import re
from langchain_core.documents import Document

def process_documents(documents: list[Document], doc_type: str) -> list[Document]:
    """
    Cleans raw documents and strictly formats their metadata for the Vector Database.
    """
    print(f"Processing {len(documents)} {doc_type} documents...")
    processed_docs = []
    
    for doc in documents:
        clean_text = doc.page_content
        metadata = doc.metadata.copy()
        
        # 1. Enforce Content Type metadata
        metadata['content_type'] = doc_type
        
        # 2. Markdown Specific Processing (URL Extraction & Noise Removal)
        if doc_type == "markdown":
            lines = clean_text.split('\n')
            
            # The crawler saves the URL on line 1, like "Source URL: https://..."
            if lines and lines[0].startswith("Source URL:"):
                metadata['source_url'] = lines[0].replace("Source URL:", "").strip()
                # Drop the first two lines (The URL line and the "---" separator)
                clean_text = '\n'.join(lines[2:]).strip()
            else:
                metadata['source_url'] = "Unknown Webpage"
                
            # Remove repeating website noise that the crawler missed
            noisy_phrases = [
                r"Theme Presets.*",
                r"Copyright.*",
                r"Institution's Innovation Council \(IIC\) \| SRKR Engineering College"
            ]
            for phrase in noisy_phrases:
                clean_text = re.sub(phrase, "", clean_text, flags=re.IGNORECASE).strip()
                
        # 3. PDF Specific Processing
        elif doc_type == "pdf":
            # PDFs don't have a website URL, so we set the source to the file name
            file_name = metadata.get('source', 'Unknown PDF').split('/')[-1]
            metadata['source_url'] = f"PDF: {file_name}"
            
        # 4. Extract Category (Department) for Hybrid Search filtering
        search_string = metadata.get('source_url', '').lower() + metadata.get('source', '').lower()
        department_match = re.search(r'/(cse|it|ece|mech|eee|civil|ai&ds)/', search_string)
        
        if department_match:
            metadata['category'] = department_match.group(1)
        else:
            metadata['category'] = "general"
            
        # 5. Save the clean document
        if clean_text.strip():  # Only save if there is actually text left!
            processed_docs.append(Document(page_content=clean_text, metadata=metadata))
            
    print(f" -> Successfully processed {len(processed_docs)} documents.")
    return processed_docs
