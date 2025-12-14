from src.comfy_sdk import ComfyUI
import json
import sys
import os

# Add src to pythonpath so imports work during dev
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

def main():
    print("Initializing ComfyUI Client...")
    client = ComfyUI(host="127.0.0.1", port=8188)
    
    print("--- Connection Test ---")
    try:
        stats = client.system.stats()
        print(f"System Stats: {stats}")
    except Exception as e:
        print(f"Connection failed: {e}")
        print("Ensure ComfyUI is running.")
        return

    print("\n--- History Management ---")
    try:
        hist = client.prompt.history()
        print(f"Found history items: {len(hist)}")
        # client.prompt.clear()
    except Exception as e:
        print(f"Error fetching history: {e}")

    print("\n--- User Management ---")
    try:
        users = client.users.list()
        print(f"Users: {users}")
        # client.users.create("new_user")
    except Exception as e:
        print(f"Error managing users: {e}")

    print("\n--- Userdata ---")
    try:
        # data = client.userdata.get("some_file.json")
        pass
    except Exception as e:
        print(f"Error accessing userdata: {e}")

    print("\n--- Verification Complete ---")

if __name__ == "__main__":
    main()
