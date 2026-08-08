import logging
from src.tools.scraper.search import WebScraperTool
from src.llm.groq import GroqLLM

log = logging.getLogger(__name__)

class WebSearchHandler:
    def __init__(self):
        self.scraper = WebScraperTool()
        self.llm = GroqLLM()
        
    def handle(self, user_query: str, context: str = "") -> str:
        """
        Executes a web search and uses the LLM to synthesize an answer.
        """
        # First, extract a concise search query from the user's potentially conversational utterance.
        extract_prompt = f"User said: '{user_query}'. Extract the core search query to look up on a search engine. Return ONLY the search string, nothing else."
        search_query = self.llm.generate(extract_prompt, system_prompt="You are a search query extractor.")
        
        # Execute the search
        search_results = self.scraper.search(search_query)
        
        # Synthesize final response
        sys_prompt = "You are ECHELON. Based on the web search results provided, answer the user's original query. Keep your answer concise (2-3 sentences max) and conversational."
        final_prompt = f"Original Query: {user_query}\n\nWeb Search Results:\n{search_results}"
        
        response = self.llm.generate(final_prompt, system_prompt=sys_prompt)
        return response
