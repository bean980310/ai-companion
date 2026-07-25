from pathlib import Path
from typing import Union

StrPath = Union[str, Path]

APPDATA_PATH: Path = Path.home() / ".ai-companion"

if not APPDATA_PATH.exists():
    APPDATA_PATH.mkdir(parents=True)
