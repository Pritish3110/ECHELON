import json
import logging
import urllib.request

from src.tools.system_control.media import MediaController
from src.tools.system_control.diagnostics import DiagnosticsController
from src.tools.system_control.apps import AppController
from src.tools.system_control.scripts import ScriptController

log = logging.getLogger(__name__)

class SystemControlHandler:
    """Entry point for all system_control intents. Parses voice commands and executes system calls securely."""
    
    PARAM_PROMPT = """You are a parameter extractor for a local system operations tool. 
Analyze the user's command and extract the necessary arguments.
Return a valid JSON object matching this schema:
{{
  "action": "volume" | "brightness" | "diagnostics" | "launch_app" | "run_script",
  "target": "<diagnostics target (gpu/disk/ram) OR app name OR script name>",
  "value": <number for volume/brightness level, 0 if not applicable>
}}

Examples:
"Turn the volume up to 50" -> {{"action": "volume", "target": "", "value": 50}}
"Check the GPU memory" -> {{"action": "diagnostics", "target": "gpu", "value": 0}}
"Open firefox" -> {{"action": "launch_app", "target": "firefox", "value": 0}}
"Run the backup script" -> {{"action": "run_script", "target": "backup", "value": 0}}

User Command: "{query}"

Respond ONLY with the JSON object."""

    def __init__(self):
        self.media = MediaController()
        self.diagnostics = DiagnosticsController()
        self.apps = AppController()
        self.scripts = ScriptController()
        self.model = "qwen2.5:3b-instruct-q4_K_M"
        self.ollama_url = "http://localhost:11434/api/generate"

    def handle(self, command: str) -> str:
        """Parses the command and executes the proper system operation."""
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
                target = params.get("target", "").lower()
                value = params.get("value", 0)
                
                if action == "volume":
                    return self.media.set_volume(value)
                elif action == "brightness":
                    return self.media.set_brightness(value)
                elif action == "diagnostics":
                    if target == "gpu":
                        return self.diagnostics.get_gpu_usage()
                    elif target in ["disk", "storage"]:
                        return self.diagnostics.get_disk_space()
                    elif target in ["ram", "memory"]:
                        return self.diagnostics.get_ram_usage()
                    else:
                        return f"Unknown diagnostic target: {target}"
                elif action == "launch_app":
                    return self.apps.launch_app(target)
                elif action == "run_script":
                    return self.scripts.run_script(target)
                elif action == "":
                    return "I understood that as a system command, but I'm not sure which tool or app to use for it."
                else:
                    return f"Unknown system operation parsed: {action}"
                    
        except Exception as e:
            log.error(f"Failed to extract system ops parameters: {e}")
            return "I couldn't understand the system parameters from that command."
