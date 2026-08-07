import json
import logging
import urllib.request
import urllib.error

log = logging.getLogger(__name__)

class CommandDecomposer:
    """Decomposes multi-step tasks into actionable tuples using Qwen2.5 JSON mode."""
    
    DECOMPOSER_PROMPT = """You are a task decomposer. Break down the user's multi-step request into a sequential list of steps. 
Each step must have an 'intent' (system_control, file_ops, rag_query, web_query, memory_recall, general_chat) and a 'command' (the specific substring to execute).

Example input: "Read notes.txt and then search the web for artificial intelligence"
Example JSON output:
{{
  "steps": [
    {{"intent": "file_ops", "command": "Read notes.txt"}},
    {{"intent": "web_query", "command": "search the web for artificial intelligence"}}
  ]
}}

User input: "{query}"

Respond ONLY with a valid JSON object containing the "steps" array."""

    def __init__(self, ollama_url: str = "http://localhost:11434/api/generate", model: str = "qwen2.5:3b-instruct-q4_K_M"):
        self.ollama_url = ollama_url
        self.model = model

    def decompose(self, user_input: str) -> list[tuple[str, str]]:
        """Splits query into [(intent, command)]. Returns single tuple if single step."""
        
        lower_input = user_input.lower()
        if not any(word in lower_input for word in [" and ", " then ", " after "]):
            # If no obvious compound structure, return as single unknown intent (let the router handle it)
            return [("unknown", user_input)]
            
        payload = {
            "model": self.model,
            "prompt": self.DECOMPOSER_PROMPT.format(query=user_input),
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.0
            }
        }
        
        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(self.ollama_url, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                res = json.loads(resp.read().decode("utf-8"))
                raw = res.get("response", "").strip()
                
                parsed = json.loads(raw)
                if isinstance(parsed, dict) and "steps" in parsed:
                    steps_list = parsed["steps"]
                    if isinstance(steps_list, list) and len(steps_list) > 0:
                        result = []
                        for step in steps_list:
                            if isinstance(step, dict) and "intent" in step and "command" in step:
                                result.append((step["intent"], step["command"]))
                        if result:
                            return result
                            
                return [("unknown", user_input)]
        except Exception as e:
            log.error(f"Decomposition failed: {e}. Falling back to single step.")
            return [("unknown", user_input)]
