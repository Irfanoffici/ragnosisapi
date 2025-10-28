import os

class Config:
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 100
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    GROK_API_KEY = os.getenv("GROK_API_KEY")
