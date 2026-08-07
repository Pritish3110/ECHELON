import json
import logging
import urllib.request
import urllib.error
import time
import os

log = logging.getLogger(__name__)

class IntentRouter:
    """Zero-latency local router using Ollama and Qwen2.5 3B with Few-Shot prompting."""
    
    VALID_ROUTES = {
        "system_control", "file_ops", "rag_query", 
        "web_query", "memory_recall", "general_chat"
    }

    ROUTER_PROMPT = """You are a query classifier for a local voice assistant. Classify the user query into exactly ONE category.

Categories:
- system_control : OS actions, volume, brightness, media, system diagnostics.
- file_ops       : Reading, writing, organizing, creating local files or directories.
- rag_query      : Searching indexed knowledge base documents, querying architecture or parameters from documents.
- web_query      : Real-time web search, scraping external URLs, weather, news.
- memory_recall  : Recalling user preferences, past conversation context, or personal facts.
- general_chat   : Small talk, greetings, general knowledge, math, coding, or queries requiring no tools.

Examples:
User: "Turn the volume up to 80 percent"
Category: system_control

User: "Check my current GPU memory usage"
Category: system_control

User: "Read the contents of notes.txt"
Category: file_ops

User: "Create a summary document in PDF format"
Category: file_ops

User: "What does the architecture document say about chunking?"
Category: rag_query

User: "Search knowledge base for Reciprocal Rank Fusion parameters"
Category: rag_query

User: "What is the current weather in New Delhi?"
Category: web_query

User: "Scrape text content from example.com"
Category: web_query

User: "What was my preferred coding language from our previous discussion?"
Category: memory_recall

User: "Explain how quantum computing works in simple terms"
Category: general_chat

User: "{query}"
Category:"""

    def __init__(self, ollama_url: str = "http://localhost:11434/api/generate", model: str = "qwen2.5:3b-instruct-q4_K_M"):
        self.ollama_url = ollama_url
        self.model = model

    def classify_intent(self, user_input: str) -> str:
        """Classify user intent with validation and resilient fallback."""
        payload = {
            "model": self.model,
            "prompt": self.ROUTER_PROMPT.format(query=user_input),
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_predict": 10,
                "top_k": 1
            }
        }
        
        try:
            start_t = time.time()
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(self.ollama_url, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                res = json.loads(resp.read().decode("utf-8"))
                elapsed = (time.time() - start_t) * 1000
                raw = res.get("response", "").strip().lower()
                
                log.debug(f"Router local LLM replied in {elapsed:.1f}ms: '{raw}'")
                
                # Validation
                for route in self.VALID_ROUTES:
                    if route in raw:
                        return route
                        
                log.warning(f"Router returned ambiguous route '{raw}', defaulting to 'general_chat'")
                return "general_chat"
                
        except urllib.error.URLError as e:
            log.warning(f"Ollama unreachable ({e.reason}). Falling back to keyword router.")
            return self._keyword_fallback(user_input)
        except Exception as e:
            log.error(f"Router error: {e}. Defaulting to 'general_chat'.")
            return "general_chat"

    def _keyword_fallback(self, query: str) -> str:
        """Simple keyword-based routing when Ollama is offline (Resilience pattern)."""
        query = query.lower()
        if any(w in query for w in ["volume", "brightness", "gpu", "open", "launch", "system"]):
            return "system_control"
        if any(w in query for w in ["file", "pdf", "docx", "folder", "read", "create"]):
            return "file_ops"
        if any(w in query for w in ["document", "knowledge", "architecture", "doc", "rag"]):
            return "rag_query"
        if any(w in query for w in ["weather", "scrape", "web", "news"]):
            return "web_query"
        if any(w in query for w in ["remember", "yesterday", "i said", "preference"]):
            return "memory_recall"
        return "general_chat"
