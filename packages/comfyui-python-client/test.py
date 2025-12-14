from src.comfy_sdk import ComfyUI
import json
import sys
import os

# Add src to pythonpath so imports work during dev
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

def main():
    print("Initializing ComfyUI Client...")
    client = ComfyUI(host="127.0.0.1", port=8000)
    
    print("--- Connection Test ---")
    try:
        stats = client.system.stats()
        print(f"System Stats: {stats}")
    except Exception as e:
        print(f"Connection failed: {e}")
        print("Ensure ComfyUI is running.")
        # Proceeding to show what code WOULD run
        
    print("\n--- Queue Status ---")
    try:
        status = client.queue.status()
        print(f"Queue Status: {status}")
    except Exception as e:
        print(f"Error fetching queue status: {e}")

    print("\n--- Mask Upload & Metadata ---")
    try:
        # Example mask upload
        # client.images.upload_mask(mask_data, "input.png", {"name": "input.png", "type": "input", "subfolder": ""})
        
        # Example metadata view
        # meta = client.images.metadata("output_0001.png")
        # print(f"Image Metadata: {meta}")
        pass
    except Exception as e:
        print(f"Error in image ops: {e}")

    print("\n--- Models & Templates ---")
    try:
        models = client.models.list("checkpoints")
        print(f"Found {len(models)} checkpoints.")
    except Exception as e:
        print(f"Error fetching models: {e}")

    print("\n--- Verification Complete ---")

if __name__ == "__main__":
    main()
