import os
import glob
import logging
from typing import Callable
from src.tools.file_ops.manager import FileManager

log = logging.getLogger(__name__)

class FileOrganizer:
    """Handles rule-based batch operations."""

    def __init__(self, ask_callback: Callable[[str], bool]):
        self.ask_callback = ask_callback
        self.manager = FileManager(ask_callback)

    def move_by_extension(self, src_dir: str, dest_dir: str, extension: str) -> str:
        """Moves all files of a specific extension from src_dir to dest_dir."""
        src_dir = os.path.expanduser(src_dir)
        dest_dir = os.path.expanduser(dest_dir)
        
        if not extension.startswith('.'):
            extension = '.' + extension
            
        pattern = os.path.join(src_dir, f"*{extension}")
        matches = glob.glob(pattern)
        
        files_to_move = [f for f in matches if os.path.isfile(f)]
        
        if not files_to_move:
            return f"No {extension} files found in {src_dir} to move."
            
        count = len(files_to_move)
        prompt = f"Found {count} {extension} files in {src_dir}. Do you want to move them to {dest_dir}?"
        
        if not self.ask_callback(prompt):
            return "Operation cancelled by user."
            
        success_count = 0
        for src_file in files_to_move:
            filename = os.path.basename(src_file)
            dest_file = os.path.join(dest_dir, filename)
            
            # Using manager internally so it routes through permissions properly
            result = self.manager.move_file(src_file, dest_file)
            if "Successfully" in result:
                success_count += 1
            else:
                log.warning(f"Failed to move {src_file}: {result}")
                
        return f"Successfully moved {success_count} out of {count} files to {dest_dir}."
