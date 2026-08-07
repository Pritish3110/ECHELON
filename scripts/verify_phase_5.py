import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tools.system_control.media import MediaController
from src.tools.system_control.diagnostics import DiagnosticsController
from src.tools.system_control.apps import AppController
from src.tools.system_control.scripts import ScriptController

def verify_system_control():
    print("Starting Phase 5 System Control Verification...\n")
    
    # 1. Diagnostics (Safe, Read-Only)
    print("=== TASK 1: Diagnostics Controller ===")
    diag = DiagnosticsController()
    
    print("Fetching RAM Usage...")
    ram_msg = diag.get_ram_usage()
    print(f"Result: {ram_msg}")
    
    print("Fetching GPU Usage (nvidia-smi)...")
    gpu_msg = diag.get_gpu_usage()
    print(f"Result: {gpu_msg}")
    
    # 2. Apps (Strict whitelist check)
    print("\n=== TASK 2: App Controller ===")
    apps = AppController()
    
    # Test valid app
    print("Attempting to launch unverified app 'malware'...")
    res1 = apps.launch_app("malware")
    print(f"Result (should be refused): {res1}")
    
    print("Attempting to launch verified app 'calculator'...")
    res2 = apps.launch_app("calculator")
    print(f"Result: {res2}")
    
    # 3. Scripts (Strict whitelist check)
    print("\n=== TASK 3: Script Controller ===")
    scripts = ScriptController()
    
    print("Attempting to run unauthorized script 'destroy'...")
    res3 = scripts.run_script("destroy")
    print(f"Result (should be refused): {res3}")
    
    print("\nPhase 5 Verification Complete!")

if __name__ == "__main__":
    verify_system_control()
