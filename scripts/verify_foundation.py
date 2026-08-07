import os
import sys
import time
import json
import urllib.request
import urllib.error
import subprocess
import numpy as np

# Inject CUDA 12 library paths for faster-whisper/ctranslate2
venv_lib_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "venv", "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages")
cublas_dir = os.path.join(venv_lib_dir, "nvidia", "cublas", "lib")
cudnn_dir = os.path.join(venv_lib_dir, "nvidia", "cudnn", "lib")
current_ld = os.environ.get("LD_LIBRARY_PATH", "")
if cublas_dir not in current_ld and os.path.exists(cublas_dir):
    new_ld = f"{cublas_dir}:{cudnn_dir}"
    if current_ld:
        new_ld += f":{current_ld}"
    os.environ["LD_LIBRARY_PATH"] = new_ld

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_vram_usage_mb():
    """Query nvidia-smi for current VRAM usage in MiB."""
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,memory.free", "--format=csv,nounits,noheader"],
            capture_output=True, text=True, check=True
        )
        parts = [int(p.strip()) for p in res.stdout.strip().split(",")]
        return {"used": parts[0], "total": parts[1], "free": parts[2]}
    except Exception as e:
        return {"error": str(e)}

def print_vram_status(label):
    stats = get_vram_usage_mb()
    if "error" in stats:
        print(f"[{label}] VRAM Query Failed: {stats['error']}")
    else:
        print(f"[{label}] VRAM Used: {stats['used']} MiB / {stats['total']} MiB | Headroom: {stats['free']} MiB")

def test_task_01_vram_and_models():
    print("\n" + "="*50)
    print("TASK-01: Concurrent VRAM Footprint & Model Coexistence Test")
    print("="*50)
    print_vram_status("Baseline Before Loading")
    
    print("\n1. Loading Faster-Whisper (small INT8)...")
    from src.voice.asr import ASR
    asr = ASR(model_size="small")
    print_vram_status("After ASR Load")
    
    print("\n2. Loading Kokoro-82M TTS...")
    from src.voice.tts import TTS
    tts = TTS(lang_code='a')
    print_vram_status("After ASR + TTS Load")
    
    print("\n3. Loading Silero VAD...")
    from src.voice.vad import VADFilter
    vad = VADFilter()
    print_vram_status("After ASR + TTS + VAD Load")
    
    print("\n4. Loading paraphrase-multilingual-MiniLM-L12-v2...")
    from src.tools.rag.embeddings import Embedder
    embedder = Embedder()
    _ = embedder.embed_text(["System foundation check"])
    print_vram_status("After Full Python Model Pipeline Load")
    
    return asr, tts, vad, embedder

def test_task_02_groq_api():
    print("\n" + "="*50)
    print("TASK-02: Groq Cloud Synthesis API Verification")
    print("="*50)
    
    groq_key = os.environ.get("GROQ_API_KEY")
    if not groq_key:
        from src.config.settings import settings
        groq_key = settings.groq_api_key
        
    if not groq_key or groq_key == "mock_key":
        print("❌ GROQ_API_KEY missing or not set in environment/settings.")
        return False
        
    try:
        from groq import Groq
        client = Groq(api_key=groq_key)
        start_t = time.time()
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "Respond with 'SYSTEM OK'."}],
            max_tokens=10,
            temperature=0
        )
        elapsed = (time.time() - start_t) * 1000
        reply = completion.choices[0].message.content.strip()
        print(f"✅ Groq API Status: HTTP 200 | Latency: {elapsed:.1f}ms | Reply: '{reply}'")
        return elapsed < 800
    except Exception as e:
        print(f"❌ Groq API Call Failed: {e}")
        return False

def test_task_03_ollama_qwen_router():
    print("\n" + "="*50)
    print("TASK-03: Ollama Qwen2.5 3B Local Router Verification")
    print("="*50)
    
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "qwen2.5:3b-instruct-q4_K_M",
        "prompt": "Respond with category 'general_chat' for query 'Hello'",
        "stream": False,
        "options": {"temperature": 0, "num_predict": 10}
    }
    
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        start_t = time.time()
        with urllib.request.urlopen(req, timeout=5) as resp:
            elapsed = (time.time() - start_t) * 1000
            res = json.loads(resp.read().decode("utf-8"))
            reply = res.get("response", "").strip()
            print(f"✅ Ollama Qwen2.5 3B Active | Latency: {elapsed:.1f}ms | Reply: '{reply}'")
            print_vram_status("VRAM with Ollama Qwen2.5 3B Loaded")
            return True
    except urllib.error.URLError as e:
        print(f"⚠️ Ollama service not reachable on http://localhost:11434 ({e.reason})")
        print("   Make sure Ollama is running (`ollama serve`) and `qwen2.5:3b-instruct-q4_K_M` is pulled.")
        return False
    except Exception as e:
        print(f"❌ Ollama Test Error: {e}")
        return False

def test_task_04_intent_classification():
    print("\n" + "="*50)
    print("TASK-04: Intent Classification Accuracy Benchmark (10 Utterances)")
    print("="*50)
    
    test_suite = [
        ("Turn the volume up to 80 percent", "system_control"),
        ("Check my current GPU memory usage", "system_control"),
        ("Read the contents of notes.txt", "file_ops"),
        ("Create a summary document in PDF format", "file_ops"),
        ("What does the architecture document say about chunking?", "rag_query"),
        ("Search knowledge base for Reciprocal Rank Fusion parameters", "rag_query"),
        ("What is the current weather in New Delhi?", "web_query"),
        ("Scrape text content from example.com", "web_query"),
        ("What was my preferred coding language from our previous discussion?", "memory_recall"),
        ("Explain how quantum computing works in simple terms", "general_chat"),
    ]
    
    from src.agent.router import IntentRouter
    router = IntentRouter()
    
    correct = 0
    
    for query, expected in test_suite:
        try:
            start_t = time.time()
            predicted = router.classify_intent(query)
            elapsed = (time.time() - start_t) * 1000
            
            is_correct = expected in predicted
            if is_correct:
                correct += 1
            status = "✅ PASS" if is_correct else f"❌ FAIL (Got: '{predicted}')"
            print(f"Query: '{query}' -> Target: {expected} | {status} ({elapsed:.1f}ms)")
        except Exception as e:
            print(f"Query: '{query}' -> Failed with error: {e}")
            
    print(f"\nRouter Benchmark Accuracy: {correct}/{len(test_suite)} ({correct/len(test_suite)*100:.1f}%)")
    return correct >= 9

def test_task_05_task_decomposition():
    print("\n" + "="*50)
    print("TASK-05: Task Decomposition Benchmark (3 Utterances)")
    print("="*50)
    
    from src.agent.decomposer import CommandDecomposer
    decomposer = CommandDecomposer()
    
    test_suite = [
        ("Read notes.txt and search the web for artificial intelligence", 2),
        ("Search knowledge base for RAG guidelines, then write a summary to summary.md", 2),
        ("Check system GPU usage and set volume to 50 percent", 2)
    ]
    
    success = 0
    for query, expected_steps in test_suite:
        try:
            steps = decomposer.decompose(query)
            if len(steps) == expected_steps:
                success += 1
                print(f"✅ PASS | '{query}' -> {steps}")
            else:
                print(f"❌ FAIL | '{query}' -> Got {len(steps)} steps: {steps}")
        except Exception as e:
            print(f"❌ ERROR | '{query}': {e}")
            
    return success == len(test_suite)

def main():
    print("Starting ECHELON Foundation Verification Suite...")
    print_vram_status("Initial System State")
    
    models = test_task_01_vram_and_models()
    groq_ok = test_task_02_groq_api()
    ollama_ok = test_task_03_ollama_qwen_router()
    
    if ollama_ok:
        test_task_04_intent_classification()
        test_task_05_task_decomposition()
    else:
        print("\nSkipping Task-04 and Task-05 benchmarks until Ollama is started.")
        
    print("\n" + "="*50)
    print("FINAL SUMMARY & VRAM AUDIT")
    print("="*50)
    print_vram_status("Final Coexistence State")
    print("Verification suite complete.")

if __name__ == "__main__":
    main()
