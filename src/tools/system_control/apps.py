import subprocess
import logging

log = logging.getLogger(__name__)

class AppController:
    """Safely launches and closes desktop applications."""
    
    ALLOWED_APPS = {
        "firefox": "firefox",
        "browser": "firefox",
        "chrome": "google-chrome",
        "terminal": "gnome-terminal",
        "calculator": "gnome-calculator",
        "gedit": "gedit",
        "files": "nautilus",
        "explorer": "nautilus",
        "settings": "gnome-control-center",
        "system monitor": "gnome-system-monitor",
        "spotify": "spotify"
    }
    
    def launch_app(self, app_name: str) -> str:
        """Launches an application by trying gtk-launch then falling back to generic execution."""
        app_name = app_name.lower().strip()
        
        target = self.ALLOWED_APPS.get(app_name)
        if not target:
            return f"Security Refusal: ECHELON is not permitted to launch the unverified application '{app_name}'."
            
        try:
            # Prevent launching duplicate windows if it's already running (saves VRAM)
            check = subprocess.run(["pgrep", target], capture_output=True, text=True)
            if check.stdout.strip():
                return f"{app_name.capitalize()} is already running."
                
            # We use Popen with a fully qualified list to detach the process
            # and avoid blocking the Python thread.
            subprocess.Popen([target], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return f"Opening {app_name}."
        except FileNotFoundError:
            return f"Error: The application {app_name} could not be found."
        except Exception as e:
            log.error(f"Failed to launch {app_name}: {e}")
            return f"Error launching {app_name}."

    def close_app(self, app_name: str) -> str:
        """Safely closes a running application using killall."""
        app_name = app_name.lower().strip()
        
        target = self.ALLOWED_APPS.get(app_name)
        if not target:
            return f"Security Refusal: ECHELON is not permitted to close the unverified application '{app_name}'."
            
        try:
            # We strictly pass the target as an isolated array element to prevent command injection
            result = subprocess.run(["killall", target], capture_output=True, text=True)
            if result.returncode == 0:
                return f"Closed {app_name}."
            else:
                # killall returns non-zero if no process was found
                return f"{app_name} is not currently running."
        except Exception as e:
            log.error(f"Failed to close {app_name}: {e}")
            return f"Error closing {app_name}."
