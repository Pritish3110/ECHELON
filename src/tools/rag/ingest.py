import os
import uuid
import pickle
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from rank_bm25 import BM25Okapi

from src.tools.rag.chunker import get_chunker
from src.tools.rag.embeddings import Embedder

COLLECTION_NAME = "echelon_docs"
VECTOR_DIM = 384
BM25_INDEX_PATH = "bm25_index.pkl"
DOCS_DB_PATH = "docs_db.pkl"

def setup_qdrant(qdrant: QdrantClient):
    if not qdrant.collection_exists(collection_name=COLLECTION_NAME):
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )

def ingest_documents(texts: list[str]):
    qdrant = QdrantClient("localhost", port=6333)
    embedder = Embedder()
    
    setup_qdrant(qdrant)
    chunker = get_chunker()
    
    all_chunks = []
    for text in texts:
        chunks = chunker.split_text(text)
        all_chunks.extend(chunks)
        
    if not all_chunks:
        return
        
    # 1. Ingest into Qdrant (Dense)
    points = []
    for chunk in all_chunks:
        vector = embedder.embed_text(chunk)
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={"text": chunk}
            )
        )
    
    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        wait=True,
        points=points
    )
    
    # 2. Ingest into BM25 (Sparse)
    tokenized_corpus = [chunk.split(" ") for chunk in all_chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    
    with open(BM25_INDEX_PATH, 'wb') as f:
        pickle.dump(bm25, f)
        
    with open(DOCS_DB_PATH, 'wb') as f:
        pickle.dump(all_chunks, f)
        
    print(f"Ingested {len(all_chunks)} chunks successfully.")
