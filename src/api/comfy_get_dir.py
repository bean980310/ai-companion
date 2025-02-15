import os

base_path = os.path.dirname(os.path.realpath(__file__))

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