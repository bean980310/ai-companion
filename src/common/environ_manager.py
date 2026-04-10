import os
from typing import Union
from pathlib import Path
from src.common.apppath import APPDATA_PATH

StrPath = Union[str, "os.PathLike[str]", Path, "os.PathLike[Path]"]


def load_env_variables(key: str, path: StrPath = APPDATA_PATH / ".env"):
    from dotenv import get_key

    return get_key(dotenv_path=path, key_to_get=key)


def save_env_variables(key: str, value: str, path: StrPath = APPDATA_PATH / ".env"):
    from dotenv import set_key

    if not path.exists():
        with open(path, "w") as f:
            f.write("")

    set_key(dotenv_path=path, key_to_set=key, value_to_set=value)

    return load_env_variables(key=key)


def delete_env_variables(key: str, path: StrPath = APPDATA_PATH / ".env"):
    from dotenv import unset_key

    if not path.exists():
        pass

    unset_key(dotenv_path=path, key_to_unset=key)
