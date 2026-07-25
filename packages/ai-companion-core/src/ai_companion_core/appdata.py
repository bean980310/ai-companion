from typing import Union
from pathlib import Path

StrPath = Union[str, Path]

APPDATA_PATH: Path = Path.home() / ".ai-companion"

if not APPDATA_PATH.exists():
    APPDATA_PATH.mkdir(parents=True)
