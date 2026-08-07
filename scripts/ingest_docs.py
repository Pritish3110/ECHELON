import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tools.rag.ingest import ingest_documents

if __name__ == "__main__":
    dummy_text = """
    ECHELON Architecture Overview:
    ECHELON is a voice-controlled AI assistant running locally on an RTX 3060.
    It uses a Hybrid RAG pipeline combining dense vector search via Qdrant and sparse search via BM25.
    The primary challenge during Phase 1 was VRAM fragmentation and PyTorch linking errors like libcublas.so.12 missing.
    ECHELON uses Faster-Whisper small for ASR, Kokoro-82M for TTS, and Groq's llama-3.3-70b for the LLM fallback.
    The total VRAM usage is strictly kept under 6GB, specifically hovering around 2616MiB during concurrent loads.
    
    Hindi translation overview:
    ECHELON एक AI सहायक है जो RTX 3060 पर चलता है। यह हिंदी और अंग्रेजी दोनों भाषाओं को समझ सकता है।
    यह Qdrant और BM25 का उपयोग करता है।
    """
    
    print("Ingesting dummy documents...")
    ingest_documents([dummy_text])
    print("Done.")
