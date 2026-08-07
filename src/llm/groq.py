from groq import Groq
import logging
from src.config.settings import settings

log = logging.getLogger(__name__)

class GroqClient:
    def __init__(self):
        self.client = Groq(api_key=settings.groq_api_key)
        self.model_name = "llama-3.3-70b-versatile"

    def generate(self, prompt: str, system: str = None) -> str:
        log.debug("Calling Groq...")
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.7,
        )
        return completion.choices[0].message.content
