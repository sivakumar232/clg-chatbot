import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import deque
import markdownify

BASE_URL = "https://www.srkrec.ac.in/"
DOMAIN = "www.srkrec.ac.in"
# Create Data folder at the root of your chat_bot project
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "Data", "raw")

def is_internal_link(url):
    """Check if the link belongs to the same domain."""
    parsed = urlparse(url)
    return parsed.netloc == DOMAIN or parsed.netloc == ""

def get_file_path_from_url(url, is_pdf=False):
    """Generate a valid file path inside the Data directory based on the URL structure."""
    parsed = urlparse(url)
    path = parsed.path
    
    # Remove leading slash
    if path.startswith('/'):
        path = path[1:]
        
    if is_pdf:
        # Just use the path, it should already end in .pdf
        if not path.lower().endswith('.pdf'):
            path += '.pdf'
        file_path = os.path.join(DATA_DIR, "pdfs", path)
        return file_path
    
    if path.endswith('/'):
        path += "index"
    if not path or path == "/":
        path = "index"
        
    # Append .md extension
    file_path = os.path.join(DATA_DIR, path + ".md")
    return file_path

def clean_text(html_content):
    """Extract and clean text from HTML content, ignoring scripts and styles."""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove script, style, nav, and footer elements to focus on main content
    for script_or_style in soup(['script', 'style', 'nav', 'footer', 'header']):
        script_or_style.decompose()
    # Convert the remaining HTML into Markdown
    md_text = markdownify.markdownify(str(soup), heading_style="ATX")
    
    # Clean up whitespace
    lines = (line.strip() for line in md_text.splitlines())
    text = '\n'.join(line for line in lines if line)
    
    return text

def crawl(max_pages=100):
    """Crawl the website using Breadth-First Search (BFS)."""
    print(f"Starting crawl of {BASE_URL}")
    print(f"Data will be saved to: {DATA_DIR}")
    
    os.makedirs(DATA_DIR, exist_ok=True)
    
    visited = set()
    queue = deque([BASE_URL])
    
    while queue and len(visited) < max_pages:
        url = queue.popleft()
        
        if url in visited:
            continue
            
        print(f"Crawling ({len(visited) + 1}/{max_pages}): {url}")
        visited.add(url)
        
        try:
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                print(f"  -> Failed to fetch. Status code: {response.status_code}")
                continue
                
            # Only process HTML pages and PDFs
            content_type = response.headers.get('content-type', '')
            if 'application/pdf' in content_type or url.lower().endswith('.pdf'):
                file_path = get_file_path_from_url(url, is_pdf=True)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                print(f"  -> Saved PDF to: {os.path.relpath(file_path, PROJECT_ROOT)}")
                continue
            elif 'text/html' not in content_type:
                continue
                
            # Extract text
            text_content = clean_text(response.text)
            
            if not text_content.strip():
                continue # Skip empty pages
            
            # Save to file mirroring the URL path
            file_path = get_file_path_from_url(url)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"Source URL: {url}\n")
                f.write("-" * 50 + "\n\n")
                f.write(text_content)
                
            print(f"  -> Saved to: {os.path.relpath(file_path, PROJECT_ROOT)}")
            
            # Find more links
            soup = BeautifulSoup(response.text, 'html.parser')
            for a_tag in soup.find_all('a', href=True):
                href_attr = a_tag.get('href')
                if isinstance(href_attr, list):
                    href = str(href_attr[0])
                else:
                    href = str(href_attr)
                
                # Ignore fragment identifiers, mailto, and javascript
                if href.startswith('#') or href.startswith('javascript:') or href.startswith('mailto:'):
                    continue
                    
                full_url = urljoin(url, href)
                full_url = full_url.split('#')[0] # Remove fragment
                
                # Check if it's internal and not visited
                if is_internal_link(full_url):
                    if full_url not in visited and full_url not in queue:
                        queue.append(full_url)
                        
        except Exception as e:
            import traceback
            print(f"  -> Error crawling {url}: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    # You can change max_pages to crawl more or fewer pages.
    crawl(max_pages=2200)
    print("\nCrawling completed successfully!")
