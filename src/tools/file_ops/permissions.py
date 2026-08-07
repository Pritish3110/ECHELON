from enum import Enum
import os
import logging
from typing import Callable

log = logging.getLogger(__name__)

class PermissionTier(Enum):
    ALWAYS_ALLOW = 1
    ALWAYS_ASK = 2
    ALWAYS_DENY = 3

class FilePermissions:
    """Security gate for all local file operations."""
    
    # Strictly define which operations fall into which tier
    OP_TIERS = {
        "read": PermissionTier.ALWAYS_ALLOW,
        "search": PermissionTier.ALWAYS_ALLOW,
        "create": PermissionTier.ALWAYS_ASK,
        "move": PermissionTier.ALWAYS_ASK,
        "rename": PermissionTier.ALWAYS_ASK,
        "copy": PermissionTier.ALWAYS_ASK,
        "delete": PermissionTier.ALWAYS_DENY,
        "overwrite": PermissionTier.ALWAYS_DENY
    }
    
    # We restrict all file operations to the user's home directory tree
    BASE_DIR = os.path.expanduser("~")
    
    @classmethod
    def validate(cls, operation: str, path: str, ask_callback: Callable[[str], bool] = None) -> bool:
        """
        Validates if an operation is permitted on the given path.
        If ALWAYS_ASK, it invokes the ask_callback.
        Returns True if permitted, False otherwise.
        """
        # 1. Path Safety Check: Prevent directory traversal escaping BASE_DIR
        abs_path = os.path.abspath(os.path.expanduser(path))
        if not abs_path.startswith(cls.BASE_DIR):
            log.warning(f"SECURITY: Attempted file operation outside home directory: {abs_path}")
            return False
            
        tier = cls.OP_TIERS.get(operation, PermissionTier.ALWAYS_DENY)
        
        if tier == PermissionTier.ALWAYS_DENY:
            log.warning(f"SECURITY: Denied '{operation}' on {abs_path} (ALWAYS_DENY constraint)")
            return False
            
        if tier == PermissionTier.ALWAYS_ALLOW:
            return True
            
        if tier == PermissionTier.ALWAYS_ASK:
            if not ask_callback:
                log.error(f"SECURITY: '{operation}' requires ask_callback, but none provided.")
                return False
            
            prompt_msg = f"Do you want me to {operation} the file at {abs_path}?"
            log.info(f"Asking user for permission: {prompt_msg}")
            
            # Pause execution and ask user
            approved = ask_callback(prompt_msg)
            if not approved:
                log.info(f"User denied '{operation}' on {abs_path}")
                return False
            
            log.info(f"User approved '{operation}' on {abs_path}")
            return True
            
        return False
