import time
import os
import sys
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tools.rag_query.handler import RAGHandler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

class KnowledgeEventHandler(FileSystemEventHandler):
    def __init__(self, rag_handler: RAGHandler, debounce_seconds: int = 2):
        self.rag_handler = rag_handler
        self.debounce_seconds = debounce_seconds
        self.last_processed = {}

    def _process_file(self, file_path: str):
        if not file_path.endswith(('.md', '.txt')):
            return

        current_time = time.time()
        last_time = self.last_processed.get(file_path, 0)
        
        # Debounce to prevent multiple reads during copy operations
        if current_time - last_time < self.debounce_seconds:
            return
            
        self.last_processed[file_path] = current_time
        filename = os.path.basename(file_path)
        
        log.info(f"Detected change in {filename}. Ingesting into Qdrant...")
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                
            result = self.rag_handler.ingest_document(content, source_name=filename)
            log.info(result)
        except Exception as e:
            log.error(f"Failed to ingest {filename}: {e}")

    def on_created(self, event):
        if not event.is_directory:
            self._process_file(event.src_path)

    def on_modified(self, event):
        if not event.is_directory:
            self._process_file(event.src_path)

def run_watcher():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    knowledge_dir = os.path.join(project_root, "data", "knowledge")
    os.makedirs(knowledge_dir, exist_ok=True)
    
    rag_handler = RAGHandler()
    event_handler = KnowledgeEventHandler(rag_handler)
    observer = Observer()
    observer.schedule(event_handler, knowledge_dir, recursive=False)
    
    log.info(f"Starting Knowledge Watcher on {knowledge_dir}...")
    log.info("Drop .md or .txt files here to auto-ingest into ECHELON's RAG brain.")
    
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        log.info("Stopping Knowledge Watcher.")
    observer.join()

if __name__ == "__main__":
    run_watcher()
