from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import os
import chromadb
from sentence_transformers import SentenceTransformer
import requests
import json

app = FastAPI(title="RAGnosis", version="1.0.0")

# Configuration
CONFIG = {
    "chunk_size": 800,
    "chunk_overlap": 100,
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "groq_api_key": os.getenv("GROQ_API_KEY")  # SECURE: No hardcoded key!
}

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class QueryResponse(BaseModel):
    answer: str
    contexts: List[str]

class SimpleRAG:
    def __init__(self):
        if not CONFIG["groq_api_key"]:
            raise ValueError("GROQ_API_KEY environment variable is required")
        
        self.embedder = SentenceTransformer(CONFIG["embedding_model"])
        self.client = chromadb.Client()
        self.collection = self.client.create_collection("medical_books")
        self.setup_database()
    
    def load_all_text_files(self):
        """Load text from all 9 medical text files"""
        books_path = "./books"
        all_chunks = []
        
        expected_files = [
            "Anatomy&Physiology.txt", "Cardiology.txt", "Dentistry.txt",
            "EmergencyMedicine.txt", "Gastrology.txt", "General.txt",
            "InfectiousDisease.txt", "InternalMedicine.txt", "Nephrology.txt"
        ]
        
        if not os.path.exists(books_path):
            print("⚠️ No 'books' folder found")
            return []
        
        print("📚 Loading medical textbooks...")
        
        for filename in expected_files:
            file_path = os.path.join(books_path, filename)
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        text = file.read()
                    
                    if text.strip():
                        file_chunks = self.chunk_text(text, filename.replace('.txt', ''))
                        all_chunks.extend(file_chunks)
                        print(f"✅ {filename} -> {len(file_chunks)} chunks")
                    
                except Exception as e:
                    print(f"❌ {filename}: {e}")
            else:
                print(f"❌ Missing: {filename}")
        
        return all_chunks
    
    def chunk_text(self, text, subject):
        """Split text into chunks"""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip() and len(p.strip()) > 20]
        chunks = []
        
        current_chunk = ""
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) > CONFIG["chunk_size"]:
                if current_chunk:
                    chunks.append(f"[{subject}] {current_chunk}".strip())
                current_chunk = paragraph
            else:
                current_chunk = current_chunk + " " + paragraph if current_chunk else paragraph
        
        if current_chunk:
            chunks.append(f"[{subject}] {current_chunk}".strip())
            
        return chunks
    
    def setup_database(self):
        """Setup vector database"""
        print("🔄 Building medical knowledge base...")
        
        chunks = self.load_all_text_files()
        
        if chunks:
            self.collection.add(documents=chunks, ids=[f"chunk_{i}" for i in range(len(chunks))])
            print(f"✅ Loaded {len(chunks)} chunks from medical textbooks")
        else:
            raise Exception("No medical data found. Please add text files to books/ folder")
    
    def search_contexts(self, query, top_k):
        """Search for relevant contexts"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=min(top_k, 10)
            )
            return results['documents'][0] if results['documents'] else []
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def generate_with_groq(self, query, contexts):
        """Generate answer using Groq API"""
        if not contexts:
            return "I couldn't find relevant information in the medical textbooks."
        
        context_text = "\n".join([f"- {ctx}" for ctx in contexts])
        
        prompt = f"""Based ONLY on this medical context, provide a concise answer:

Question: {query}

Context:
{context_text}

Answer using ONLY the provided context. Be concise (1-2 sentences). If no answer in context, say "I cannot find this information"."""

        try:
            headers = {
                "Authorization": f"Bearer {CONFIG['groq_api_key']}",
                "Content-Type": "application/json"
            }
            
            data = {
                "messages": [{"role": "user", "content": prompt}],
                "model": "llama3-8b-8192",
                "temperature": 0.1,
                "max_tokens": 150
            }
            
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content'].strip()
            else:
                return self.simple_fallback(contexts)
                
        except Exception as e:
            print(f"Groq error: {e}")
            return self.simple_fallback(contexts)
    
    def simple_fallback(self, contexts):
        return contexts[0][:200] + "..." if contexts else "Information not found."
    
    def query(self, query, top_k=5):
        contexts = self.search_contexts(query, top_k)
        answer = self.generate_with_groq(query, contexts)
        return answer, contexts

# Initialize RAG system
try:
    rag = SimpleRAG()
    print("🎉 RAGnosis API started successfully!")
except Exception as e:
    print(f"❌ Failed to initialize: {e}")
    rag = None

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    if not rag:
        return QueryResponse(
            answer="Service unavailable - initialization failed",
            contexts=[]
        )
    
    answer, contexts = rag.query(request.query, request.top_k)
    return QueryResponse(answer=answer, contexts=contexts)

@app.get("/")
async def root():
    return {"message": "RAGnosis Medical API", "status": "running"}

@app.get("/health")
async def health():
    status = "healthy" if rag else "unhealthy"
    return {"status": status}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
