import logging

log = logging.getLogger(__name__)

class ConversationBuffer:
    """In-memory sliding window for short-term conversational context."""
    
    def __init__(self, max_turns: int = 5):
        self.history = []
        self.max_turns = max_turns

    def add_turn(self, user_msg: str, echelon_msg: str):
        self.history.append({"user": user_msg, "echelon": echelon_msg})
        if len(self.history) > self.max_turns:
            self.history.pop(0)

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
