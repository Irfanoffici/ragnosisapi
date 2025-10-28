from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import os
import chromadb
from sentence_transformers import SentenceTransformer
import requests
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter

app = FastAPI(title="RAGnosis", version="1.0.0")

# Configuration
CONFIG = {
    "chunk_size": 800,
    "chunk_overlap": 100,
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "groq_api_key": os.getenv("GROQ_API_KEY")
}

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class QueryResponse(BaseModel):
    answer: str
    contexts: List[str]

class SimpleRAG:
    def __init__(self):
        self.embedder = SentenceTransformer(CONFIG["embedding_model"])
        # Persistent client for deployment
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.client.get_or_create_collection("medical_books")
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
            print("⚠️ No 'books' folder found.")
            return []
        
        print("📚 Loading medical textbooks...")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CONFIG["chunk_size"],
            chunk_overlap=CONFIG["chunk_overlap"],
            length_function=len,
        )
        
        for filename in expected_files:
            file_path = os.path.join(books_path, filename)
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        text = file.read()
                    
                    if text.strip():
                        # Use LangChain for better chunking
                        chunks = text_splitter.split_text(text)
                        subject = filename.replace('.txt', '')
                        chunks_with_ref = [f"[{subject}] {chunk}" for chunk in chunks]
                        all_chunks.extend(chunks_with_ref)
                        print(f"✅ {filename}: {len(chunks)} chunks")
                    
                except Exception as e:
                    print(f"❌ Error processing {filename}: {e}")
            else:
                print(f"❌ Missing: {filename}")
        
        return all_chunks
    
    def setup_database(self):
        """Setup vector database - only if empty"""
        print("🔄 Checking medical knowledge base...")
        
        # Check if collection already has data
        if self.collection.count() > 0:
            print(f"✅ Database already loaded with {self.collection.count()} chunks")
            return
        
        chunks = self.load_all_text_files()
        
        if chunks:
            # Add in batches to avoid timeouts
            batch_size = 100
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                self.collection.add(
                    documents=batch,
                    ids=[f"chunk_{i + j}" for j in range(len(batch))]
                )
            print(f"🎉 Loaded {len(chunks)} chunks from medical textbooks")
        else:
            print("❌ No medical data found. Please check books folder.")
    
    def search_contexts(self, query, top_k):
        """Search for relevant medical contexts"""
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
        """Generate medical answer using Groq API"""
        if not contexts:
            return "I couldn't find relevant information in the medical textbooks."
        
        context_text = "\n".join([f"- {ctx}" for ctx in contexts])
        
        prompt = f"""Based ONLY on the following medical context, provide a concise answer.

Question: {query}

Context:
{context_text}

Answer using ONLY the provided context. Be concise (1-2 sentences). If unsure, say "I cannot find this information"."""

        try:
            headers = {
                "Authorization": f"Bearer {CONFIG['groq_api_key']}",
                "Content-Type": "application/json"
            }
            
            data = {
                "messages": [
                    {"role": "system", "content": "You are a medical assistant."},
                    {"role": "user", "content": prompt}
                ],
                "model": "llama3-8b-8192",
                "temperature": 0.1,
                "max_tokens": 150,
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
                return self.simple_fallback(query, contexts)
                
        except Exception as e:
            return self.simple_fallback(query, contexts)
    
    def simple_fallback(self, query, contexts):
        if not contexts:
            return "I cannot find this information in the medical textbooks."
        return contexts[0][:200] + "..." if len(contexts[0]) > 200 else contexts[0]
    
    def query(self, query, top_k=5):
        contexts = self.search_contexts(query, top_k)
        answer = self.generate_with_groq(query, contexts)
        return answer, contexts

# Initialize RAG system
rag = SimpleRAG()

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    answer, contexts = rag.query(request.query, request.top_k)
    return QueryResponse(answer=answer, contexts=contexts)

@app.get("/")
async def root():
    return {"message": "RAGnosis Medical API is running"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
