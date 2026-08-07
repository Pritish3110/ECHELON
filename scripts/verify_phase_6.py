import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tools.rag_query.handler import RAGHandler
from src.memory.buffer import ConversationBuffer

def verify_memory_and_rag():
    print("Starting Phase 6 Memory & RAG Verification...\n")
    
    # 1. Memory Buffer
    print("=== TASK 1: Short-Term Memory Buffer ===")
    buffer = ConversationBuffer(max_turns=3)
    
    buffer.add_turn("Create a file called summary.txt", "Successfully saved text to summary.txt.")
    buffer.add_turn("Read it to me.", "Reading summary.txt...")
    
    context = buffer.get_context_string(turns=2)
    print("Injected Context for LLM:")
    print("-" * 20)
    print(context)
    print("-" * 20)
    
    assert "summary.txt" in context, "Context failed to retain history."
    print("Buffer verification passed.\n")
    
    # 2. RAG Indexing and Query
    print("=== TASK 2: Long-Term RAG Engine (Qdrant) ===")
    rag = RAGHandler()
    
    test_doc = """
    ECHELON Architecture Document.
    The system uses three primary LLM models: Whisper for speech recognition, Qwen 2.5 for local intent routing, and Groq (Llama 3.3) for cloud synthesis.
    The short-term memory node retains exactly 5 conversation turns in RAM.
    """
    
    print("Ingesting test document...")
    res = rag.ingest_document(test_doc, "architecture.md")
    print(res)
    
    print("\nQuerying Qdrant for specific fact (How many turns does memory keep?)...")
    answer = rag.handle("How many turns does the memory node retain?")
    print(f"RAG Synthesized Answer: {answer}")
    
    print("\nPhase 6 Verification Complete!")

if __name__ == "__main__":
    verify_memory_and_rag()
