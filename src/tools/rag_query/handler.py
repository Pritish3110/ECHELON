import os
import uuid
import logging
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer
from src.llm.groq import GroqLLM

log = logging.getLogger(__name__)

class RAGHandler:
    """Handles semantic search over local documents using Qdrant and SentenceTransformers."""
    
    def __init__(self, collection_name="echelon_docs", db_path="qdrant_data"):
        self.collection_name = collection_name
        self.db_path = os.path.abspath(db_path)
        self.llm = GroqLLM()
        
        log.info("Loading SentenceTransformer for RAG (all-MiniLM-L6-v2)...")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Local disk deployment (no docker)
        self.client = QdrantClient(path=self.db_path)
        
        # Initialize collection if missing
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
            log.info(f"Created new Qdrant collection: {self.collection_name}")

    def chunk_text(self, text: str, max_words: int = 150) -> list[str]:
        """Simple word-based chunker for documents."""
        words = text.split()
        chunks = []
        for i in range(0, len(words), max_words):
            chunk = " ".join(words[i:i + max_words])
            chunks.append(chunk)
        return chunks

    def ingest_document(self, text: str, source_name: str) -> str:
        """Chunks, embeds, and stores a document in Qdrant."""
        chunks = self.chunk_text(text)
        if not chunks:
            return "Document is empty."
            
        vectors = self.embedder.encode(chunks)
        points = []
        
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            point_id = str(uuid.uuid4())
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector.tolist(),
                    payload={"source": source_name, "text": chunk, "chunk_id": i}
                )
            )
            
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        return f"Successfully ingested {source_name} ({len(chunks)} chunks)."

    def handle(self, query: str, context: str = "") -> str:
        """Queries the vector DB and synthesizes an answer."""
        query_vector = self.embedder.encode(query).tolist()
        
        try:
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=3
            )
            
            if not search_result:
                return "I couldn't find any relevant information in your knowledge base."
                
            # Build context string
            retrieved_texts = []
            for hit in search_result:
                source = hit.payload.get('source', 'Unknown')
                text = hit.payload.get('text', '')
                retrieved_texts.append(f"[Source: {source}]\n{text}")
                
            combined_context = "\n\n".join(retrieved_texts)
            
            # Synthesize answer using Groq
            system_prompt = (
                "You are ECHELON. Answer the user's question using ONLY the provided retrieved context. "
                "Be concise. If the context does not contain the answer, say 'I cannot find that in the documents.'"
            )
            
            prompt = f"Retrieved Context:\n{combined_context}\n\nUser Question: {query}\n\nAnswer:"
            
            return self.llm.generate(prompt, system_prompt)
            
        except Exception as e:
            log.error(f"RAG Query failed: {e}")
            return "There was an error searching the knowledge base."
