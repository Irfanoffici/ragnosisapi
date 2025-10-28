class RAGSystem:
    def __init__(self):
        self.vector_store = None
        self.setup()
    
    def setup(self):
        # Load PDFs, create embeddings, etc.
        pass
    
    def query(self, query: str, top_k: int):
        # Your RAG logic here
        contexts = ["Context 1", "Context 2"][:top_k]
        answer = "Sample answer based on contexts"
        return answer, contexts
