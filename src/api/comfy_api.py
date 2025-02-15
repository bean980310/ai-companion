#This is an example that uses the websockets api and the SaveImageWebsocket node to get images directly without
#them being saved to disk

import websocket #NOTE: websocket-client (https://github.com/websocket-client/websocket-client)
import uuid
import json
import urllib.request
import urllib.parse
import pandas as pd
import os
from requests_toolbelt import MultipartEncoder
from PIL import Image
import io
import requests

class ComfyUIClient:
    def __init__(self, server_address="127.0.0.1:8000"):
        self.server_address=server_address
        self.client_id=str(uuid.uuid4())

    def queue_prompt(self, prompt):
        p = {"prompt": prompt, "client_id": self.client_id}
        data = json.dumps(p).encode('utf-8')
        req =  urllib.request.Request("http://{}/prompt".format(self.server_address), data=data,
        headers={'Content-Type': 'application/json'})
        return json.loads(urllib.request.urlopen(req).read())

    def get_image(self, filename, subfolder, folder_type):
        data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        url_values = urllib.parse.urlencode(data)
        with urllib.request.urlopen("http://{}/view?{}".format(self.server_address, url_values)) as response:
            return response.read()

    def get_history(self, prompt_id):
        with urllib.request.urlopen("http://{}/history/{}".format(self.server_address, prompt_id)) as response:
            return json.loads(response.read())

    def get_images(self, ws, prompt):
        prompt_id = self.queue_prompt(prompt)['prompt_id']
        output_images = {}
        output_dir = 'outputs'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        while True:
            out = ws.recv()
            if isinstance(out, str):
                message = json.loads(out)
                if message['type'] == 'executing':
                    data = message['data']
                    if data['node'] is None and data['prompt_id'] == prompt_id:
                        break

            else:
                continue
        
        history = self.get_history(prompt_id)[prompt_id]
        for node_id in history['outputs']:
            node_output = history['outputs'][node_id]
            images_output = []
            if 'images' in node_output:
                for image in node_output['images']:
                    image_data = self.get_image(image['filename'], image['subfolder'], image['type'])
                    images_output.append(image_data)
                                
                    save_path = os.path.join(output_dir, image['filename'])
                    try:
                        with open(save_path, "wb") as f:
                            f.write(image_data)
                            print(f"Saved image to {save_path}")
                    except Exception as e:
                        print(f"Error saving image {save_path}: {e}")
                                    
            output_images[node_id] = images_output
                    
        return output_images
    
    def upload_image(self, input_path, subfolder="", overwrite=False):
        with Image.open(input_path) as f:
            body={"image": f}
            data={}
            
            if overwrite:
                data["overwrite"] = "true"
                
            if subfolder:
                data['subfolder'] = subfolder
            
            response=requests.post(f"http://{self.server_address}/upload/image", files=body,data=data)
            
            if response.status_code == 200:
                data = response.json()
                path = data["name"]
                if "subfolder" in data:
                    if data["subfolder"] != "":
                        path = data["subfolder"] + "/" + path
                        
            else:
                print(f"Error uploading image {input_path}: {response.status_code} - {response.reason}")
                
            return path
    
    def text2image_generate(self, prompt: dict):
        ws_url = "ws://{}/ws?clientId={}".format(self.server_address, self.client_id)
        ws = websocket.WebSocket()
        ws.connect(ws_url)
        images = self.get_images(ws, prompt)
        ws.close()
        
        return images
    
    def image2image_generate(self, prompt: dict):
        ws_url = "ws://{}/ws?clientId={}".format(self.server_address, self.client_id)
        ws = websocket.WebSocket()
        ws.connect(ws_url)
        # self.upload_image(input_path, name)
        images = self.get_images(ws, prompt)
        ws.close()
        
        return images
        

client=ComfyUIClient()