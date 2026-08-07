import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tools.rag.retriever import HybridRetriever

def run_tests():
    retriever = HybridRetriever()
    
    print("=== Hybrid RAG Tests ===\n")
    
    # 1. Exact Keyword Match (BM25 should win)
    q1 = "libcublas.so.12"
    print(f"Query 1 (Exact Keyword): {q1}")
    r1 = retriever.retrieve(q1, top_k=2)
    for idx, doc in enumerate(r1):
        print(f"  {idx+1}. {doc}")
        
    print("\n" + "-"*40 + "\n")
    
    # 2. Semantic Paraphrase (Dense should win)
    q2 = "What GPU memory issues might occur?"
    print(f"Query 2 (Semantic Paraphrase): {q2}")
    r2 = retriever.retrieve(q2, top_k=2)
    for idx, doc in enumerate(r2):
        print(f"  {idx+1}. {doc}")
        
    print("\n" + "-"*40 + "\n")
    
    # 3. Hindi Semantic Query
    q3 = "क्या ECHELON हिंदी समझ सकता है?"
    print(f"Query 3 (Hindi Semantic): {q3}")
    r3 = retriever.retrieve(q3, top_k=2)
    for idx, doc in enumerate(r3):
        print(f"  {idx+1}. {doc}")
        
if __name__ == "__main__":
    run_tests()
