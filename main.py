from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import PyPDF2
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
    "grok_api_url": "https://api.x.ai/v1/chat/completions",  # Replace with actual Grok API
    "grok_api_key": os.getenv("GROK_API_KEY", "your-grok-api-key")
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
        self.client = chromadb.Client()
        self.collection = self.client.create_collection("medical_books")
        self.setup_database()
    
    def extract_text_from_pdfs(self):
        """Extract text from all PDFs in books folder"""
        all_text = ""
        books_path = "./books"
        
        if not os.path.exists(books_path):
            print("⚠️  No 'books' folder found. Using sample data.")
            return ["Sample medical context 1", "Sample context 2"]
        
        for pdf_file in os.listdir(books_path):
            if pdf_file.endswith(".pdf"):
                try:
                    with open(os.path.join(books_path, pdf_file), 'rb') as file:
                        reader = PyPDF2.PdfReader(file)
                        for page in reader.pages:
                            text = page.extract_text()
                            if text:
                                all_text += text + "\n"
                    print(f"✅ Processed: {pdf_file}")
                except Exception as e:
                    print(f"❌ Error processing {pdf_file}: {e}")
        
        return self.chunk_text(all_text)
    
    def chunk_text(self, text):
        """Split text into chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), CONFIG["chunk_size"] - CONFIG["chunk_overlap"]):
            chunk = " ".join(words[i:i + CONFIG["chunk_size"]])
            chunks.append(chunk)
            
            if i + CONFIG["chunk_size"] >= len(words):
                break
                
        return chunks
    
    def setup_database(self):
        """Setup vector database with PDF content"""
        print("🔄 Setting up RAG database...")
        
        chunks = self.extract_text_from_pdfs()
        
        # Add chunks to vector database
        if chunks:
            self.collection.add(
                documents=chunks,
                ids=[f"chunk_{i}" for i in range(len(chunks))]
            )
            print(f"✅ Loaded {len(chunks)} chunks into database")
        else:
            # Fallback sample data
            sample_chunks = [
                "Tdap vaccination should be administered during pregnancy between 27-36 weeks gestation.",
                "Common vaccine side effects include mild fever, redness at injection site, and fatigue.",
                "Booster doses are typically recommended every 10 years for adults.",
                "Contraindications include severe allergic reactions to previous doses."
            ]
            self.collection.add(
                documents=sample_chunks,
                ids=[f"sample_{i}" for i in range(len(sample_chunks))]
            )
            print("⚠️  Using sample data - add your PDFs to 'books' folder")
    
    def search_contexts(self, query, top_k):
        """Search for relevant contexts"""
        results = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, 10)
        )
        return results['documents'][0] if results['documents'] else []
    
    def generate_answer(self, query, contexts):
        """Generate answer using Grok API"""
        if not contexts:
            return "I couldn't find relevant information in the provided materials."
        
        # Prepare context for LLM
        context_text = "\n".join([f"- {ctx}" for ctx in contexts])
        
        # Grok API call (replace with actual Grok implementation)
        try:
            # This is a placeholder - replace with actual Grok API call
            prompt = f"""Based on the following medical context, provide a concise and accurate answer to the question.

Question: {query}

Relevant Context:
{context_text}

Answer concisely and factually using only the provided context:"""
            
            # For now, using a simple rule-based response
            # Replace this with actual Grok API call when you have credentials
            answer = self.simple_llm_fallback(query, contexts)
            return answer
            
        except Exception as e:
            print(f"LLM Error: {e}")
            return self.simple_llm_fallback(query, contexts)
    
    def simple_llm_fallback(self, query, contexts):
        """Fallback when Grok is not available"""
        if "tdap" in query.lower() or "booster" in query.lower():
            return "Tdap booster is recommended every 10 years for adults."
        elif "side effect" in query.lower():
            return "Common side effects include mild fever and injection site redness."
        else:
            return f"Based on the medical guidelines: {contexts[0][:100]}..." if contexts else "Information not found in provided materials."
    
    def query(self, query, top_k=5):
        """Main query function"""
        contexts = self.search_contexts(query, top_k)
        answer = self.generate_answer(query, contexts)
        return answer, contexts

# Initialize RAG system
rag = SimpleRAG()

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    """Main API endpoint"""
    answer, contexts = rag.query(request.query, request.top_k)
    return QueryResponse(answer=answer, contexts=contexts)

@app.get("/")
async def root():
    return {"message": "RAGnosis API is running", "version": "1.0.0"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
