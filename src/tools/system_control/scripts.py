import subprocess
import logging
import os

log = logging.getLogger(__name__)

class ScriptController:
    """Safely launches predefined whitelisted bash scripts."""
    
    # STRICT WHITELIST: Only these scripts can be executed.
    # No arbitrary paths can be passed by the LLM.
    WHITELISTED_SCRIPTS = {
        "backup": os.path.expanduser("~/scripts/backup.sh"),
        "update_system": os.path.expanduser("~/scripts/update.sh")
    }
    
    def run_script(self, script_name: str) -> str:
        """Executes a whitelisted script."""
        script_name = script_name.lower().strip()
        
        target_path = self.WHITELISTED_SCRIPTS.get(script_name)
        
        if not target_path:
            log.warning(f"SECURITY: Attempted to run unauthorized script '{script_name}'")
            return f"Security Refusal: Execution of '{script_name}' is not permitted."
            
        if not os.path.exists(target_path):
            return f"Error: The registered script path for {script_name} does not exist."
            
        try:
            # We do NOT use shell=True.
            subprocess.Popen(["bash", target_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return f"Initiated the {script_name} script."
        except Exception as e:
            log.error(f"Failed to run script {script_name}: {e}")
            return f"Error running the {script_name} script."
