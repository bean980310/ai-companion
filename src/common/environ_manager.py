import os
from dotenv import load_dotenv

def load_env_variables():
    load_dotenv()

def save_env_variables(key, value):
    if os.path.exists('.env'):
        with open('.env', 'a') as f:
            f.write(f"{key}={value}\n")
    else:
        with open('.env', 'w') as f:
            f.write('')

    load_env_variables()