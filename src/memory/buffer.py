import logging
import json
import os

log = logging.getLogger(__name__)

class ConversationBuffer:
    """Disk-persisted sliding window for short-term conversational context."""
    
    def __init__(self, max_turns: int = 5, filepath: str = "data/memory/history.json"):
        self.max_turns = max_turns
        
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.filepath = os.path.join(project_root, filepath)
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        
        self.history = self._load()

    def _load(self) -> list[dict]:
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r") as f:
                    return json.load(f)
            except Exception as e:
                log.error(f"Failed to load memory from {self.filepath}: {e}")
        return []

    def _save(self):
        try:
            temp_path = self.filepath + ".tmp"
            with open(temp_path, "w") as f:
                json.dump(self.history, f, indent=2)
            os.replace(temp_path, self.filepath) # Atomic replace
        except Exception as e:
            log.error(f"Failed to save memory to {self.filepath}: {e}")

    def add_turn(self, user_msg: str, echelon_msg: str):
        self.history.append({"user": user_msg, "echelon": echelon_msg})
        if len(self.history) > self.max_turns:
            self.history.pop(0)
        self._save()

    def get_context_string(self, turns: int = 2) -> str:
        """Formats the last N turns into a string for LLM injection."""
        if not self.history:
            return "No previous conversation context."
            
        recent = self.history[-turns:]
        context = []
        for turn in recent:
            context.append(f"User: {turn['user']}")
            context.append(f"Echelon: {turn['echelon']}")
            
        return "\n".join(context)
        
    def get_full_history(self) -> list[dict]:
        return self.history
