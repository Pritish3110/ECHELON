import os
import logging
from pypdf import PdfReader
import docx
from src.tools.file_ops.permissions import FilePermissions
from src.llm.groq import GroqLLM

log = logging.getLogger(__name__)

class FileReaders:
    """Handles reading and extracting text from various file formats."""

    def __init__(self, llm: GroqLLM = None):
        self.llm = llm or GroqLLM()

    def read_file(self, file_path: str) -> str:
        """Reads a file and returns its content. Summarizes if too long."""
        if not os.path.exists(file_path):
            return f"Error: The file {file_path} does not exist."
            
        if not FilePermissions.validate("read", file_path):
            return f"Security Error: Permission denied to read {file_path}."

        ext = os.path.splitext(file_path)[1].lower()
        content = ""
        
        try:
            if ext in ['.txt', '.md', '.csv', '.json', '.py']:
                content = self._read_text(file_path)
            elif ext == '.pdf':
                content = self._read_pdf(file_path)
            elif ext == '.docx':
                content = self._read_docx(file_path)
            else:
                return f"Error: Unsupported file format {ext} for reading."
        except Exception as e:
            log.error(f"Failed to read {file_path}: {e}")
            return f"Error reading file: {e}"
            
        # Long document handling
        words = content.split()
        if len(words) > 500:
            log.info(f"Document {file_path} exceeds 500 words. Summarizing...")
            return self._summarize_content(content)
            
        return content

    def _read_text(self, file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()

    def _read_pdf(self, file_path: str) -> str:
        reader = PdfReader(file_path)
        text = []
        for page in reader.pages:
            text.append(page.extract_text() or "")
        return "\n".join(text)

    def _read_docx(self, file_path: str) -> str:
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])

    def _summarize_content(self, content: str) -> str:
        """Summarizes long content using Groq LLM to avoid TTS spam."""
        prompt = f"Please provide a concise, 3-4 sentence summary of the following document:\n\n{content[:15000]}"
        return self.llm.generate(prompt, "You are a concise summarizer.")
