from dataclasses import dataclass
from pathlib import Path
from typing import IO, Optional

from langchain_unstructured import UnstructuredLoader
from langchain_community.document_loaders import UnstructuredPDFLoader

class DocumentLoader:
    """Loader that uses unstructured to load documents."""
    
    def __init__(self):
        self.loader = None
        self.documents = None

    @classmethod
    def unstructured(cls, file_path: Optional[str | Path | list[str] | list[Path]] = None, web_url: Optional[str] = None):
        instance = cls()
        if file_path:
            instance.loader = instance.from_file_path(file_path=file_path)
        elif web_url:
            instance.loader = instance.from_web_url(web_url=web_url)
        else:
            raise ValueError("Either file_path or web_url must be provided.")
        instance.documents = instance.loader.load()
        return instance.documents
    
    @classmethod
    def from_pdf(cls, file_path: str | Path | list[str] | list[Path]):
        instance = cls()
        instance.loader = UnstructuredPDFLoader(file_path=file_path)
        instance.documents = instance.loader.load()
        return instance.documents

    @staticmethod
    def from_file_path(file_path: str | Path | list[str] | list[Path]):
        return UnstructuredLoader(file_path=file_path)

    @staticmethod
    def from_web_url(web_url: str):
        return UnstructuredLoader(web_url=web_url)
