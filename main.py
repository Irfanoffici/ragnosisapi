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
    "groq_api_key": os.getenv("GROQ_API_KEY", "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")
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
    
    def load_all_text_files(self):
        """Load text from all 9 medical text files with exact names"""
        books_path = "./books"
        all_chunks = []
        
        # Exact file names as they appear
        expected_files = [
            "Anatomy&Physiology.txt",  # Capital P
            "Cardiology.txt", 
            "Dentistry.txt",
            "EmergencyMedicine.txt",   # Capital M
            "Gastrology.txt",
            "General.txt",
            "InfectiousDisease.txt",
            "InternalMedicine.txt",
            "Nephrology.txt"
        ]
        
        if not os.path.exists(books_path):
            print("⚠️  No 'books' folder found. Using sample data.")
            return ["Sample medical context 1", "Sample context 2"]
        
        print("📚 Loading medical textbooks...")
        
        found_files = 0
        for filename in expected_files:
            file_path = os.path.join(books_path, filename)
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        text = file.read()
                    
                    if text.strip():  # Only process non-empty files
                        # Chunk each file individually
                        file_chunks = self.chunk_text(text, filename.replace('.txt', ''))
                        all_chunks.extend(file_chunks)
                        print(f"✅ Loaded: {filename} -> {len(file_chunks)} chunks")
                        found_files += 1
                    else:
                        print(f"⚠️  Empty file: {filename}")
                    
                except Exception as e:
                    print(f"❌ Error processing {filename}: {e}")
            else:
                print(f"❌ Missing: {filename}")
        
        print(f"📊 Found {found_files}/9 medical textbooks")
        return all_chunks
    
    def chunk_text(self, text, subject):
        """Split text into chunks with subject reference"""
        # Clean and split text
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip() and len(p.strip()) > 20]
        chunks = []
        
        current_chunk = ""
        for paragraph in paragraphs:
            # If adding this paragraph exceeds chunk size, save current chunk
            if len(current_chunk) + len(paragraph) > CONFIG["chunk_size"]:
                if current_chunk:
                    chunk_with_ref = f"[{subject}] {current_chunk}"
                    chunks.append(chunk_with_ref.strip())
                current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += " " + paragraph
                else:
                    current_chunk = paragraph
        
        # Add the last chunk
        if current_chunk:
            chunk_with_ref = f"[{subject}] {current_chunk}"
            chunks.append(chunk_with_ref.strip())
            
        return chunks
    
    def setup_database(self):
        """Setup vector database with all medical textbooks"""
        print("🔄 Building medical knowledge base...")
        
        chunks = self.load_all_text_files()
        
        if chunks and len(chunks) > 0:
            self.collection.add(
                documents=chunks,
                ids=[f"chunk_{i}" for i in range(len(chunks))]
            )
            print(f"🎉 Medical RAG system ready! Loaded {len(chunks)} chunks from medical textbooks")
            print("💊 Specialties: Anatomy, Cardiology, Dentistry, Emergency Medicine, Gastrology, General, Infectious Disease, Internal Medicine, Nephrology")
        else:
            # Fallback sample data
            sample_chunks = [
                "[Anatomy&Physiology] The human body consists of various systems including skeletal, muscular, and nervous systems.",
                "[Cardiology] The cardiovascular system includes the heart and blood vessels, responsible for circulating blood.",
                "[EmergencyMedicine] Emergency care focuses on immediate treatment of acute illnesses and injuries.",
                "[InfectiousDisease] Infectious diseases are caused by pathogenic microorganisms and can be transmitted between individuals.",
            ]
            self.collection.add(
                documents=sample_chunks,
                ids=[f"sample_{i}" for i in range(len(sample_chunks))]
            )
            print("⚠️  Using sample data - check that all 9 medical files are in books/ folder")
    
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
        
        prompt = f"""You are a medical expert. Based ONLY on the following context from medical textbooks, provide a concise and accurate answer to the question.

Question: {query}

Relevant Context from Medical Textbooks:
{context_text}

Instructions:
- Answer using ONLY the provided medical context
- Be concise and factual (1-2 sentences maximum)
- If context doesn't contain answer, say "I cannot find this information in the provided medical textbooks"
- Do not add any external knowledge or personal opinions
- Provide only evidence-based medical information

Medical Answer:"""
        
        try:
            headers = {
                "Authorization": f"Bearer {CONFIG['groq_api_key']}",
                "Content-Type": "application/json"
            }
            
            data = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a medical expert that provides accurate, evidence-based information using only the provided medical textbook context. Be concise and factual."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "model": "llama3-8b-8192",
                "temperature": 0.1,
                "max_tokens": 150,
                "top_p": 0.9
            }
            
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result['choices'][0]['message']['content'].strip()
                # Ensure answer is concise
                if len(answer) > 250:
                    sentences = answer.split('.')
                    if len(sentences) > 2:
                        answer = '.'.join(sentences[:2]) + '.'
                return answer
            else:
                print(f"Groq API error: {response.status_code}")
                return self.simple_fallback(query, contexts)
                
        except Exception as e:
            print(f"Groq error: {e}")
            return self.simple_fallback(query, contexts)
    
    def simple_fallback(self, query, contexts):
        """Simple medical fallback"""
        if not contexts:
            return "I cannot find this information in the provided medical textbooks."
        
        # Return most relevant medical context
        return contexts[0][:200] + "..." if len(contexts[0]) > 200 else contexts[0]
    
    def query(self, query, top_k=5):
        """Main query function"""
        contexts = self.search_contexts(query, top_k)
        answer = self.generate_with_groq(query, contexts)
        return answer, contexts

# Initialize RAG system
rag = SimpleRAG()

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    """Main API endpoint - follows exact specification"""
    answer, contexts = rag.query(request.query, request.top_k)
    return QueryResponse(answer=answer, contexts=contexts)

@app.get("/")
async def root():
    return {"message": "RAGnosis Medical API is running", "version": "1.0.0"}

@app.get("/health")
async def health():
    return {"status": "healthy", "specialties": 9}

@app.get("/files")
async def check_files():
    """Check if all medical files are present"""
    expected_files = [
        "Anatomy&Physiology.txt", "Cardiology.txt", "Dentistry.txt",
        "EmergencyMedicine.txt", "Gastrology.txt", "General.txt",
        "InfectiousDisease.txt", "InternalMedicine.txt", "Nephrology.txt"
    ]
    
    found_files = []
    missing_files = []
    
    for filename in expected_files:
        if os.path.exists(f"./books/{filename}"):
            found_files.append(filename)
        else:
            missing_files.append(filename)
    
    return {
        "found_files": found_files,
        "missing_files": missing_files,
        "total_expected": 9,
        "total_found": len(found_files)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
