import pickle
import os
from qdrant_client import QdrantClient
from src.tools.rag.embeddings import Embedder

COLLECTION_NAME = "echelon_docs"
BM25_INDEX_PATH = "bm25_index.pkl"
DOCS_DB_PATH = "docs_db.pkl"

class HybridRetriever:
    def __init__(self, k_rrf=60):
        self.qdrant = QdrantClient("localhost", port=6333)
        self.embedder = Embedder()
        self.k_rrf = k_rrf
        
        self.bm25 = None
        self.docs_db = None
        
        if os.path.exists(BM25_INDEX_PATH) and os.path.exists(DOCS_DB_PATH):
            with open(BM25_INDEX_PATH, 'rb') as f:
                self.bm25 = pickle.load(f)
            with open(DOCS_DB_PATH, 'rb') as f:
                self.docs_db = pickle.load(f)
                
    def retrieve(self, query: str, top_k=5, fetch_k=20) -> list[str]:
        # 1. Dense Search (Qdrant)
        dense_results = []
        if self.qdrant.collection_exists(collection_name=COLLECTION_NAME):
            query_vector = self.embedder.embed_text(query)
            response = self.qdrant.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                limit=fetch_k
            )
            dense_results = [hit.payload["text"] for hit in response.points]
            
        # 2. Sparse Search (BM25)
        sparse_results = []
        if self.bm25 is not None and self.docs_db is not None:
            tokenized_query = query.split(" ")
            # get top fetch_k indices
            scores = self.bm25.get_scores(tokenized_query)
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:fetch_k]
            sparse_results = [self.docs_db[i] for i in top_indices if scores[i] > 0]
            
        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        
        for rank, doc in enumerate(dense_results):
            if doc not in rrf_scores:
                rrf_scores[doc] = 0
            rrf_scores[doc] += 1 / (self.k_rrf + rank + 1)
            
        for rank, doc in enumerate(sparse_results):
            if doc not in rrf_scores:
                rrf_scores[doc] = 0
            rrf_scores[doc] += 1 / (self.k_rrf + rank + 1)
            
        # Sort by RRF score
        fused_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [doc for doc, score in fused_results[:top_k]]
