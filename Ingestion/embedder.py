import os
import sys

# Add the project root to the path so we can import config
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from config import settings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def get_embedding_model():
    """
    Initializes and returns the Gemini embedding model based on settings.
    This model turns text chunks into 768-dimensional math vectors.
    """
    print(f"Initializing Embedding Model: {settings.GEMINI_EMB_MODEL}")
    
    # Initialize the Google Gemini Embedding model
    embeddings = GoogleGenerativeAIEmbeddings(
        model=settings.GEMINI_EMB_MODEL,
        google_api_key=settings.GEMINI_API_KEY
    )
    
    return embeddings
