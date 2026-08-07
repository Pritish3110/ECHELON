import subprocess
import logging

log = logging.getLogger(__name__)

class MediaController:
    """Controls volume, brightness, and basic media toggles."""
    
    def set_volume(self, level: int) -> str:
        """Sets the system volume (0-100) using amixer."""
        try:
            level = max(0, min(100, int(level)))
            # Using list for strict subprocess execution (no shell injection)
            subprocess.run(["amixer", "-D", "pulse", "sset", "Master", f"{level}%"], check=True, capture_output=True)
            return f"System volume set to {level} percent."
        except Exception as e:
            log.error(f"Failed to set volume: {e}")
            return "Error: Could not adjust the system volume. Make sure amixer is installed."

    def toggle_mute(self) -> str:
        try:
            subprocess.run(["amixer", "-D", "pulse", "sset", "Master", "toggle"], check=True, capture_output=True)
            return "Toggled system mute."
        except Exception as e:
            log.error(f"Failed to toggle mute: {e}")
            return "Error: Could not toggle mute."

    def set_brightness(self, level: int) -> str:
        """Sets display brightness using brightnessctl."""
        try:
            level = max(0, min(100, int(level)))
            subprocess.run(["brightnessctl", "set", f"{level}%"], check=True, capture_output=True)
            return f"Screen brightness set to {level} percent."
        except Exception as e:
            log.error(f"Failed to set brightness: {e}")
            return "Error: Could not adjust brightness. Make sure brightnessctl is installed."
