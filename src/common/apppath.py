import os
from typing import Union
from pathlib import Path

StrPath = Union[str, "os.PathLike[str]", Path, "os.PathLike[Path]"]

APPDATA_PATH: StrPath = Path.home() / ".ai-companion"

if not APPDATA_PATH.exists():
    APPDATA_PATH.mkdir(parents=True)
