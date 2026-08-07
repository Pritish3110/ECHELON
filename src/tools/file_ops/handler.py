import json
import logging
import urllib.request
from typing import Callable

from src.tools.file_ops.readers import FileReaders
from src.tools.file_ops.writers import FileWriters
from src.tools.file_ops.manager import FileManager
from src.tools.file_ops.organizer import FileOrganizer
from src.llm.groq import GroqLLM

log = logging.getLogger(__name__)

class FileOpsHandler:
    """Entry point for all file_ops intents. Parses voice commands and routes to the right utility."""
    
    PARAM_PROMPT = """You are a parameter extractor for a local file operations tool. 
Analyze the user's command and extract the necessary arguments.
IMPORTANT: If the user says 'underscore', 'dash', or 'dot' in a filename, convert them to '_', '-', and '.' respectively (e.g., 'project underscore architecture dot md' -> 'project_architecture.md').

Return a valid JSON object matching this schema:
{{
  "action": "read" | "write" | "move" | "copy" | "find" | "organize",
  "source_path": "<path or filename>",
  "dest_path": "<destination path if applicable, else empty>",
  "content": "<content to write, else empty>",
  "extension": "<file extension for organizing, else empty>"
}}

User Command: "{query}"

Respond ONLY with the JSON object."""

    def __init__(self, ask_callback: Callable[[str], bool]):
        self.ask_callback = ask_callback
        self.readers = FileReaders()
        self.writers = FileWriters(ask_callback)
        self.manager = FileManager(ask_callback)
        self.organizer = FileOrganizer(ask_callback)
        self.model = "qwen2.5:3b-instruct-q4_K_M"
        self.ollama_url = "http://localhost:11434/api/generate"

    def handle(self, command: str) -> str:
        """Parses the command and executes the proper file operation."""
        payload = {
            "model": self.model,
            "prompt": self.PARAM_PROMPT.format(query=command),
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.0}
        }
        
        try:
            req = urllib.request.Request(self.ollama_url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                res = json.loads(resp.read().decode())
                params = json.loads(res.get("response", "{}"))
                
                action = params.get("action", "")
                source = params.get("source_path", "")
                dest = params.get("dest_path", "")
                content = params.get("content", "")
                ext = params.get("extension", "")
                
                if action == "read":
                    return self.readers.read_file(source)
                elif action == "write":
                    return self.writers.write_file(source, content)
                elif action == "move":
                    return self.manager.move_file(source, dest)
                elif action == "copy":
                    return self.manager.copy_file(source, dest)
                elif action == "find":
                    return self.manager.find_file(source)
                elif action == "organize":
                    return self.organizer.move_by_extension(source, dest, ext)
                else:
                    return f"Unknown file operation parsed: {action}"
                    
        except Exception as e:
            log.error(f"Failed to extract file ops parameters: {e}")
            return "I couldn't understand the file parameters from that command."
