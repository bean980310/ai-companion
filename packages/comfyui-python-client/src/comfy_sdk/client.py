import json
import uuid
import websocket
import urllib.parse
import urllib.request
from typing import Dict, Optional, Any, Union
import time
from dataclasses import dataclass

from .api import ComfyResponse


class ComfyClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 8188):
        self.host = host
        self.port = port
        self.client_id = str(uuid.uuid4())
        self.ws: Optional[websocket.WebSocket] = None
        self.base_url = f"http://{host}:{port}"
        self.ws_url = f"ws://{host}:{port}/ws?clientId={self.client_id}"

    def connect(self):
        """Establish WebSocket connection."""
        try:
            self.ws = websocket.WebSocket()
            self.ws.connect(self.ws_url)
        except Exception as e:
            print(f"Failed to connect to WebSocket at {self.ws_url}: {e}")
            self.ws = None

    def close(self):
        if self.ws:
            self.ws.close()
            self.ws = None

    def queue_prompt(self, prompt: Dict[str, Any]) -> ComfyResponse:
        """
        Queue a workflow prompt to ComfyUI.
        """
        p = {"prompt": prompt, "client_id": self.client_id}
        data = json.dumps(p).encode("utf-8")
        req = urllib.request.Request(f"{self.base_url}/prompt", data=data)
        
        try:
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read())
                return ComfyResponse(
                    prompt_id=result.get("prompt_id"),
                    number=result.get("number"),
                    node_errors=result.get("node_errors")
                )
        except urllib.error.HTTPError as e:
            raise Exception(f"HTTP Error: {e.code} - {e.reason}")

    def get_queue(self) -> Dict[str, Any]:
        """
        Get the current queue status (GET /prompt).
        """
        with urllib.request.urlopen(f"{self.base_url}/prompt") as response:
            return json.loads(response.read())

    def get_history(self, prompt_id: str) -> Dict[str, Any]:
        """
        Retrieve history for a specific prompt_id.
        """
        with urllib.request.urlopen(f"{self.base_url}/history/{prompt_id}") as response:
            return json.loads(response.read())

    def get_all_history(self) -> Dict[str, Any]:
        """
        Retrieve all history.
        """
        with urllib.request.urlopen(f"{self.base_url}/history") as response:
            return json.loads(response.read())

    def delete_history(self, prompt_id: str):
        """
        Delete history for a specific prompt_id.
        """
        req = urllib.request.Request(f"{self.base_url}/history/{prompt_id}", method="DELETE")
        with urllib.request.urlopen(req) as response:
            return response.read()

    def clear_history(self):
        """
        Clear all history.
        """
        req = urllib.request.Request(f"{self.base_url}/history", method="DELETE")
        with urllib.request.urlopen(req) as response:
            return response.read()

    def get_images(self, filename: str, subfolder: str = "", folder_type: str = "output"):
        """
        Get image data from ComfyUI.
        """
        data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        url_values = urllib.parse.urlencode(data)
        with urllib.request.urlopen(f"{self.base_url}/view?{url_values}") as response:
            return response.read()

    def get_view_metadata(self, filename: str, subfolder: str = "", folder_type: str = "output") -> Dict[str, Any]:
        """
        Get image metadata.
        """
        data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        url_values = urllib.parse.urlencode(data)
        with urllib.request.urlopen(f"{self.base_url}/view_metadata?{url_values}") as response:
            return json.loads(response.read())

    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        with urllib.request.urlopen(f"{self.base_url}/system_stats") as response:
            return json.loads(response.read())

    def get_extensions(self) -> Dict[str, Any]:
        """Get list of extensions."""
        with urllib.request.urlopen(f"{self.base_url}/extensions") as response:
            return json.loads(response.read())

    def interrupt(self):
        """Interrupt current execution."""
        req = urllib.request.Request(f"{self.base_url}/interrupt", method="POST")
        with urllib.request.urlopen(req) as response:
            return response.read()

    def clear_queue(self):
        """Clear the execution queue."""
        req = urllib.request.Request(f"{self.base_url}/queue", method="DELETE")
        with urllib.request.urlopen(req) as response:
            return response.read()

    def free(self, unload_models: bool = False, free_memory: bool = False):
        """Free memory."""
        data = json.dumps({"unload_models": unload_models, "free_memory": free_memory}).encode("utf-8")
        req = urllib.request.Request(f"{self.base_url}/free", data=data, method="POST")
        req.add_header('Content-Type', 'application/json')
        with urllib.request.urlopen(req) as response:
            return response.read()
            
    def get_object_info(self, node_class: str = None) -> Dict[str, Any]:
        """Get object info (node definitions)."""
        url = f"{self.base_url}/object_info"
        if node_class:
            url += f"/{node_class}"
        with urllib.request.urlopen(url) as response:
            return json.loads(response.read())

    def get_embeddings(self) -> list[str]:
        """Get list of embeddings."""
        with urllib.request.urlopen(f"{self.base_url}/embeddings") as response:
            return json.loads(response.read())

    def get_features(self) -> Dict[str, Any]:
        """Get list of features."""
        with urllib.request.urlopen(f"{self.base_url}/features") as response:
            return json.loads(response.read())
            
    def get_models(self, folder: str = None) -> list[str]:
        """Get list of models in a specific folder."""
        url = f"{self.base_url}/models"
        if folder:
            url += f"/{folder}"
        with urllib.request.urlopen(url) as response:
            return json.loads(response.read())

    def get_workflow_templates(self) -> list[str]:
        """Get list of workflow templates."""
        with urllib.request.urlopen(f"{self.base_url}/workflow_templates") as response:
            return json.loads(response.read())

    def get_users(self) -> list[Dict[str, Any]]:
        """Get list of users."""
        with urllib.request.urlopen(f"{self.base_url}/users") as response:
            return json.loads(response.read())

    def create_user(self, username: str) -> Dict[str, Any]:
        """Create a new user."""
        data = json.dumps({"username": username}).encode("utf-8")
        req = urllib.request.Request(f"{self.base_url}/users", data=data, method="POST")
        req.add_header('Content-Type', 'application/json')
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read())

    def get_userdata(self, file: str) -> Any:
        """Get userdata file content."""
        with urllib.request.urlopen(f"{self.base_url}/userdata/{file}") as response:
            return response.read() # Returns bytes, caller can decode if json

    def move_userdata(self, file: str, dest: str):
        """Move userdata file."""
        data = json.dumps({"dest": dest}).encode("utf-8")
        req = urllib.request.Request(f"{self.base_url}/userdata/{file}/move", data=data, method="POST") # API path might be slightly different, checking docs... "/userdata/{file}/move/{dest}" in docs chunk 3?
        # Re-reading docs chunk 3: "/userdata/{file}/move/{dest}"
        # Let's verify. The docs say:
        # /userdata/{file}/move/{dest}
        
        # Wait, usually move is a POST/PUT. Let's assume URL encoding for dest if it's a path.
        # But actually, looking at general REST practices and ComfyUI source, let's try the path based first.
        # If dest is a full path, it might be complex.
        # But let's follow the docs literal string "/userdata/{file}/move/{dest}"
        
        req = urllib.request.Request(f"{self.base_url}/userdata/{file}/move/{dest}", method="POST")
        with urllib.request.urlopen(req) as response:
            return response.read()

    def upload_image(self, image_data: bytes, filename: str, overwrite: bool = False):
        """
        Upload an image to ComfyUI input folder.
        Uses requests_toolbelt for multipart upload if available, or simple post.
        For simplicity in this core client without heavy deps, using basic requests structure or built-in if possible.
        Ideally use 'requests' library if allowed, but strict stdlib is harder for multipart.
        Assuming 'requests' is available as per pyproject.toml usually.
        """
        import requests
        
        files = {
            'image': (filename, image_data),
            'overwrite': ('true' if overwrite else 'false').encode('utf-8')
        }
        
        try:
            response = requests.post(f"{self.base_url}/upload/image", files=files)
            response.raise_for_status()
            return response.json()
        except ImportError:
            raise ImportError("requests library is required for image upload")

    def upload_mask(self, image_data: bytes, filename: str, original_ref: Dict[str, str], overwrite: bool = False, mask_type: str = "mask"):
        """
        Upload a mask to ComfyUI input folder.
        original_ref: {'name': 'filename', 'type': 'output', 'subfolder': ''}
        """
        import requests
        
        files = {
            'image': (filename, image_data),
            'original_ref': (None, json.dumps(original_ref)),
            'overwrite': ('true' if overwrite else 'false').encode('utf-8'),
            'type': (mask_type)
        }
        
        try:
            response = requests.post(f"{self.base_url}/upload/mask", files=files)
            response.raise_for_status()
            return response.json()
        except ImportError:
            raise ImportError("requests library is required for mask upload")

    def wait_for_completion(self, prompt_id: str, timeout: int = 300) -> Dict[str, Any]:
        """
        Wait until the prompt execution is finished via WebSocket.
        """
        if not self.ws:
            self.connect()
            if not self.ws:
                raise ConnectionError("WebSocket is not connected")

        start_time = time.time()
        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError("Timed out waiting for execution completion")

            try:
                out = self.ws.recv()
                if isinstance(out, str):
                    message = json.loads(out)
                    if message["type"] == "executing":
                        data = message["data"]
                        if data["node"] is None and data["prompt_id"] == prompt_id:
                            # Execution finished
                            return self.get_history(prompt_id)
            except websocket.WebSocketConnectionClosedException:
                # Reconnect or fail
                self.connect()
                continue
            except Exception as e:
                print(f"Error reading websocket: {e}")
                break
        
        return {}
