import logging
from groq import Groq
from src.config.settings import settings

log = logging.getLogger(__name__)

class GroqLLM:
    """Client for Groq's high-speed cloud LLM (llama-3.3-70b-versatile)."""
    
    def __init__(self, model: str = "llama-3.3-70b-versatile", temperature: float = 0.5):
        self.model = model
        self.temperature = temperature
        
        api_key = settings.groq_api_key
        if not api_key or api_key == "mock_key":
            log.warning("GROQ_API_KEY is missing or invalid. GroqLLM will fail on generation.")
            self.client = None
        else:
            self.client = Groq(api_key=api_key)

    def generate(self, prompt: str, system_prompt: str = "You are ECHELON, a helpful AI assistant.", history: list[dict] = None) -> str:
        """Generate text using Groq's API."""
        if not self.client:
            return "Error: Groq API key is not configured."
            
        messages = [{"role": "system", "content": system_prompt}]
        if history:
            for turn in history:
                messages.append({"role": "user", "content": turn["user"]})
                messages.append({"role": "assistant", "content": turn["echelon"]})
        messages.append({"role": "user", "content": prompt})
            
        try:
            log.debug(f"Calling Groq model {self.model}...")
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=150,  # Keep TTS responses short
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            log.error(f"Groq API call failed: {e}")
            return f"Error connecting to cloud synthesis: {e}"
