import os

base_path = os.path.dirname(os.path.realpath(__file__))
                            
output_directory = os.path.join(base_path, "output")
temp_directory = os.path.join(base_path, "temp")
input_directory = os.path.join(base_path, "input")

def set_output_directory(output_dir: str) -> None:
    global output_directory
    output_directory = output_dir

def set_temp_directory(temp_dir: str) -> None:
    global temp_directory
    temp_directory = temp_dir

def set_input_directory(input_dir: str) -> None:
    global input_directory
    input_directory = input_dir
    
def get_output_directory() -> str:
    global output_directory
    return output_directory

def get_temp_directory() -> str:
    global temp_directory
    return temp_directory

def get_input_directory() -> str:
    global input_directory
    return input_directory

def get_user_directory() -> str:
    return user_directory

def set_user_directory(user_dir: str) -> None:
    global user_directory
    user_directory = user_dir
    
def annotated_filepath(name: str) -> tuple[str, str | None]:
    if name.endswith("[output]"):
        base_dir = get_output_directory()
        name = name[:-9]
    elif name.endswith("[input]"):
        base_dir = get_input_directory()
        name = name[:-8]
    elif name.endswith("[temp]"):
        base_dir = get_temp_directory()
        name = name[:-7]
    else:
        return name, None

    return name, base_dir

def get_annotated_filepath(name: str, default_dir: str | None=None) -> str:
    name, base_dir = annotated_filepath(name)

    if base_dir is None:
        if default_dir is not None:
            base_dir = default_dir
        else:
            base_dir = get_input_directory()  # fallback path

    return os.path.join(base_dir, name)
