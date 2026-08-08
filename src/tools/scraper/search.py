import logging
from duckduckgo_search import DDGS
from firecrawl import FirecrawlApp
from src.config.settings import settings

log = logging.getLogger(__name__)

class WebScraperTool:
    """Uses Firecrawl for deep scraping/search, falls back to DuckDuckGo."""
    
    def __init__(self):
        self.firecrawl_api_key = getattr(settings, 'firecrawl_api_key', None)
        self.ddgs = DDGS()
        
        if self.firecrawl_api_key and self.firecrawl_api_key != "mock_key":
            self.firecrawl = FirecrawlApp(api_key=self.firecrawl_api_key)
        else:
            self.firecrawl = None
            log.warning("Firecrawl API key not set. WebScraperTool will rely entirely on DuckDuckGo.")

    def search(self, query: str, max_results: int = 3) -> str:
        """Search the web for a query and return a summarized RAG string."""
        log.info(f"Searching web for: {query}")
        
        # 1. Try Firecrawl Search
        if self.firecrawl:
            try:
                log.debug("Attempting Firecrawl search...")
                # Firecrawl search returns a dict with 'data'
                result = self.firecrawl.search(query=query, params={"limit": max_results})
                if result and 'data' in result:
                    results = result['data']
                    formatted = []
                    for i, r in enumerate(results[:max_results]):
                        # Some versions of Firecrawl SDK return objects, some dicts.
                        title = r.get('title', 'Unknown Title') if isinstance(r, dict) else getattr(r, 'title', 'Unknown Title')
                        url = r.get('url', 'Unknown URL') if isinstance(r, dict) else getattr(r, 'url', 'Unknown URL')
                        description = r.get('description', '') if isinstance(r, dict) else getattr(r, 'description', '')
                        formatted.append(f"[{i+1}] {title} ({url})\nSnippet: {description}")
                    
                    if formatted:
                        return "\n\n".join(formatted)
            except Exception as e:
                log.warning(f"Firecrawl search failed or hit limits: {e}. Falling back to DuckDuckGo.")
        
        # 2. Fallback to DuckDuckGo Search
        try:
            log.debug("Executing DuckDuckGo fallback search...")
            results = list(self.ddgs.text(query, max_results=max_results))
            formatted = []
            for i, r in enumerate(results):
                formatted.append(f"[{i+1}] {r.get('title')} ({r.get('href')})\nSnippet: {r.get('body')}")
            
            if formatted:
                return "\n\n".join(formatted)
            else:
                return "No results found on the web."
        except Exception as e:
            log.error(f"DuckDuckGo search also failed: {e}")
            return f"Error executing web search: {e}"

