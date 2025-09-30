from dataclasses import dataclass
from pathlib import Path
from typing import IO, Optional

from langchain_unstructured import UnstructuredLoader

class DocumentLoader:
    """Loader that uses unstructured to load documents."""
    
    def __init__(self):
        self.loader = None
        self.documents = None

    def from_file_path(self, file_path: str | Path | list[str] | list[Path]):
        self.loader = UnstructuredLoader(file_path=file_path)
        return self.loader.load()

    def from_web_url(self, web_url: str):
        self.loader = UnstructuredLoader(web_url=web_url)
        return self.loader.load()
    
