from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from rag_system import RAGSystem

app = FastAPI(title="RAGnosis")
rag = RAGSystem()

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class QueryResponse(BaseModel):
    answer: str
    contexts: List[str]

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    answer, contexts = rag.query(request.query, request.top_k)
    return QueryResponse(answer=answer, contexts=contexts)

@app.get("/health")
async def health():
    return {"status": "healthy"}
