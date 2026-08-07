import os
import shutil
import logging
import glob
from typing import Callable
from src.tools.file_ops.permissions import FilePermissions

log = logging.getLogger(__name__)

class FileManager:
    """Handles moving, renaming, copying, and finding files."""

    def __init__(self, ask_callback: Callable[[str], bool]):
        self.ask_callback = ask_callback

    def move_file(self, src: str, dest: str) -> str:
        if not os.path.exists(src):
            return f"Error: Source file {src} does not exist."
            
        if os.path.exists(dest):
            # This is technically an overwrite
            if not FilePermissions.validate("overwrite", dest, self.ask_callback):
                return f"Security Error: Permission denied to overwrite {dest}."
                
        if not FilePermissions.validate("move", src, self.ask_callback):
            return f"Security Error: Permission denied to move {src}."

        try:
            os.makedirs(os.path.dirname(os.path.abspath(dest)) or '.', exist_ok=True)
            shutil.move(src, dest)
            return f"Successfully moved {src} to {dest}."
        except Exception as e:
            log.error(f"Move failed: {e}")
            return f"Error moving file: {e}"

    def copy_file(self, src: str, dest: str) -> str:
        if not os.path.exists(src):
            return f"Error: Source file {src} does not exist."
            
        if os.path.exists(dest):
            if not FilePermissions.validate("overwrite", dest, self.ask_callback):
                return f"Security Error: Permission denied to overwrite {dest}."
                
        if not FilePermissions.validate("copy", src, self.ask_callback):
            return f"Security Error: Permission denied to copy {src}."

        try:
            os.makedirs(os.path.dirname(os.path.abspath(dest)) or '.', exist_ok=True)
            if os.path.isdir(src):
                shutil.copytree(src, dest)
            else:
                shutil.copy2(src, dest)
            return f"Successfully copied {src} to {dest}."
        except Exception as e:
            log.error(f"Copy failed: {e}")
            return f"Error copying file: {e}"

    def delete_file(self, path: str) -> str:
        """This will intentionally fail based on v1 rules (ALWAYS_DENY) unless overridden in OP_TIERS."""
        if not FilePermissions.validate("delete", path, self.ask_callback):
            return f"Security Refusal: ECHELON is not permitted to delete files in this version."
        
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
            return f"Successfully deleted {path}."
        except Exception as e:
            log.error(f"Delete failed: {e}")
            return f"Error deleting file: {e}"

    def find_file(self, filename: str, search_dir: str = "~") -> str:
        """Searches for a file by name starting from search_dir."""
        search_dir = os.path.expanduser(search_dir)
        
        if not FilePermissions.validate("search", search_dir):
            return f"Security Error: Cannot search in {search_dir}."

        results = []
        try:
            # Simple recursive glob
            pattern = os.path.join(search_dir, "**", f"*{filename}*")
            matches = glob.glob(pattern, recursive=True)
            
            # Filter matches to files only and cap at 5
            for match in matches:
                if os.path.isfile(match):
                    results.append(match)
                if len(results) >= 5:
                    break
                    
            if not results:
                return f"No files matching '{filename}' were found in {search_dir}."
                
            return "Found the following files:\n" + "\n".join(results)
        except Exception as e:
            log.error(f"Find failed: {e}")
            return f"Error searching for file: {e}"
