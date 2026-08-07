import subprocess
import logging

log = logging.getLogger(__name__)

class AppController:
    """Safely launches desktop applications."""
    
    def launch_app(self, app_name: str) -> str:
        """Launches an application by trying gtk-launch then falling back to generic execution."""
        app_name = app_name.lower().strip()
        
        # Security map: To prevent injecting arbitrary bash, we only allow launching basic apps
        # that don't take complex arguments.
        # This acts as a basic whitelist to prevent command execution via voice.
        ALLOWED_APPS = {
            "firefox": "firefox",
            "browser": "firefox",
            "chrome": "google-chrome",
            "terminal": "gnome-terminal",
            "calculator": "gnome-calculator",
            "gedit": "gedit",
            "files": "nautilus",
            "explorer": "nautilus"
        }
        
        target = ALLOWED_APPS.get(app_name)
        if not target:
            return f"Security Refusal: ECHELON is not permitted to launch the unverified application '{app_name}'."
            
        try:
            # We use Popen with a fully qualified list to detach the process
            # and avoid blocking the Python thread.
            subprocess.Popen([target], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return f"Opening {app_name}."
        except FileNotFoundError:
            return f"Error: The application {app_name} could not be found."
        except Exception as e:
            log.error(f"Failed to launch {app_name}: {e}")
            return f"Error launching {app_name}."
