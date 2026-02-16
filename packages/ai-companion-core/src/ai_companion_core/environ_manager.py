import os
from dotenv import load_dotenv
from typing import Union

StrPath = Union[str, "os.PathLike[str]"]

def load_env_variables(key: str, path: StrPath=os.getenv('AI_COMPANION_API_CONFIG')):
    from dotenv import get_key

    return get_key(dotenv_path=path, key_to_get=key)

def save_env_variables(key: str, value: str, path: StrPath=os.getenv('AI_COMPANION_API_CONFIG')):
    import os
    from dotenv import set_key

    if not os.path.exists(path):
        with open(path, 'w') as f:
            f.write('')

    set_key(dotenv_path=path, key_to_set=key, value_to_set=value)

    return load_env_variables(key=key)

def delete_env_variables(key: str, path: StrPath=os.getenv('AI_COMPANION_API_CONFIG')):
    import os
    from dotenv import unset_key

    if not os.path.exists(path):
        return

    unset_key(dotenv_path=path, key_to_unset=key)