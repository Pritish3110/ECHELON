import subprocess
import logging
from src.llm.groq import GroqLLM

log = logging.getLogger(__name__)

class DiagnosticsController:
    """Reads system diagnostics and formats them naturally for TTS."""
    
    def __init__(self):
        self.llm = GroqLLM()
        self.summarize_prompt = "You are an AI assistant. You will receive raw terminal output. Convert it into a brief, natural-sounding spoken response. (e.g., 'You are using 2 GB of your 6 GB video memory'). Keep it under 2 sentences."

    def get_gpu_usage(self) -> str:
        """Reads nvidia-smi VRAM usage."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader"], 
                capture_output=True, text=True, check=True
            )
            raw_out = result.stdout.strip()
            log.info(f"Raw GPU stats: {raw_out}")
            
            return self.llm.generate(f"Format this GPU status for voice: {raw_out}", self.summarize_prompt)
        except FileNotFoundError:
            return "I cannot find the nvidia-smi command on this system."
        except Exception as e:
            log.error(f"GPU diagnostic failed: {e}")
            return "Error reading GPU memory."

    def get_disk_space(self) -> str:
        """Reads root partition disk space."""
        try:
            result = subprocess.run(["df", "-h", "/"], capture_output=True, text=True, check=True)
            raw_out = result.stdout.strip()
            return self.llm.generate(f"Format this Disk status for voice: {raw_out}", self.summarize_prompt)
        except Exception as e:
            log.error(f"Disk diagnostic failed: {e}")
            return "Error reading disk space."

    def get_ram_usage(self) -> str:
        """Reads system RAM usage."""
        try:
            result = subprocess.run(["free", "-h"], capture_output=True, text=True, check=True)
            raw_out = result.stdout.strip()
            return self.llm.generate(f"Format this RAM status for voice: {raw_out}", self.summarize_prompt)
        except Exception as e:
            log.error(f"RAM diagnostic failed: {e}")
            return "Error reading system memory."
