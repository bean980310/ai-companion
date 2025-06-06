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
from PIL import Image, ImageOps
import io
import requests
import datetime
from io import BytesIO
import numpy as np
import torch

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
    
    def upload_image(self, input_img, subfolder="", overwrite=False):
        if input_img is None:
            return None
        
        with open(input_img, 'rb') as f:
            file_data = f.read()
        file_name = os.path.basename(input_img)
        
        files = {"image": (file_name, file_data, "image/png")}
        data = {}
        # data["image"] = (file_name, file_data, "image/png")
        if overwrite:
            data["overwrite"] = "true"
        
        if subfolder:
            data["subfolder"] = subfolder
            
        data["type"] = "input" 

        response = requests.post(f"http://{self.server_address}/upload/image", files=files, data=data)
        
        if response.status_code == 200:
            response_data = response.json()
            path = response_data["name"]
            image = path
            if "subfolder" in response_data and response_data["subfolder"]:
                path = response_data["subfolder"] + "/" + path
        else:
            print(f"Error uploading image: {response.status_code} - {response.reason}")
            path = None
            
        print("img2img upload:", path)
        return image
    
    def upload_mask(self, original_img, mask_img, subfolder="clipspace", overwrite=False):
        """
        inpaint 용 업로드 함수.
        보통 inpaint용 이미지는 배경과 마스크를 합성한 최종 이미지이므로,
        파일 이름이나 추가 전처리(예: 알파 채널 유지 등)를 다르게 할 수 있음.
        """
        
        bg_img = Image.open(mask_img['background'])
        buffer = BytesIO()
        bg_img.save(buffer, format="PNG")
            
        original_file_name = os.path.basename(original_img)
        
        mask_img=Image.open(mask_img['layers'][0])
        mask_temp=mask_img.getchannel('A')
        new_alpha=ImageOps.invert(mask_temp)
        new_mask=Image.new('L', mask_img.size)
        new_mask.putalpha(new_alpha)
        buffer = BytesIO()
        new_mask.save(buffer, format="PNG")
        file_data = buffer.getvalue()
        suffix=datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        
        new_file_name = f"clipspace-mask_{suffix}.png"
        
        files = {"image": (new_file_name, file_data, "image/png")}
        data = {}
        
        if overwrite:
            data["overwrite"] = "true"
        if subfolder:
            data["subfolder"] = subfolder
            
        original_ref = {
            "filename": original_file_name,
            "subfolder": "",
            "type": "input"
        }
        
        data["type"] = "input"  # 프론트엔드에서 보내는 것과 동일하게
        data["original_ref"] = json.dumps(original_ref)
            
        response = requests.post(f"http://{self.server_address}/upload/mask", files=files, data=data)
            
        if response.status_code == 200:
            response_data = response.json()
            path = response_data["name"]
            mask = path
            if "subfolder" in response_data and response_data["subfolder"]:
                path = response_data["subfolder"] + "/" + path
        else:
            print(f"Error uploading image: {response.status_code} - {response.reason}")
            path = None
            
        print("inpaint upload:", path)
        return mask
    
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