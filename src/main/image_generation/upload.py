from ...api import ComfyUIClient
import requests
import datetime

from PIL import Image, ImageOps

import os
import json

from .image import ImageProcessor

from io import BytesIO

class ComfyUIImageUpload(ComfyUIClient):
    def __init__(self, server_address="127.0.0.1:8000"):
        super().__init__(server_address=server_address)

    def upload_image(self, input_img, subfolder="", overwrite=False):
        if input_img is None:
            return None
        
        file_name, file_data = ImageProcessor.read_image(input_img)

        files = {"image": (file_name, file_data, "image/png")}
        data = {}
        data["image"] = (file_name, file_data, "image/png")
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
        
        temp, _ = ImageProcessor.read_image(mask_img['background'])
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