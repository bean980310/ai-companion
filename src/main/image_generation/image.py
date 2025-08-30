from io import BytesIO
import numpy as np
import torch
from PIL import Image, ImageOps, ImageFilter, ImageFile
import datetime
import os

class ImageProcessor:
    def __init__(self):
        pass

    @classmethod
    def read_image(cls, img):
        if isinstance(img, Image.Image):
            return cls.read_image_from_pil(img)
        if isinstance(img, (bytes, bytearray)):
            return cls.read_image_from_bytes(img)
        if isinstance(img, str):
            return cls.read_image_from_str(img)
        if isinstance(img, (np.ndarray, torch.Tensor)):
            return cls.read_image_from_array(img)
        
    @staticmethod
    def read_image_from_pil(img: Image.Image):
        fp = BytesIO()
        img.save(fp, format="PNG")
        filename = f"uploaded_image_{datetime.datetime.now().strftime('%y%m%d_%H%M%S')}.png"

        return filename.strip(), fp.getvalue()

    @staticmethod
    def read_image_from_bytes(img_bytes: bytes | bytearray):
        return Image.open(BytesIO(img_bytes))
    
    @staticmethod
    def read_image_from_str(img_path: str):
        im = Image.open(img_path)
        im.filename = os.path.basename(img_path)

        return im.filename.strip(), im.fp.read()

    @staticmethod
    def read_image_from_array(img_array: np.ndarray | torch.Tensor):
        if isinstance(img_array, torch.Tensor):
            img_array = img_array.detach().cuda().numpy() if torch.cuda.is_available() else img_array.detach().cpu().numpy()
        return Image.fromarray(np.uint8(img_array))
    
    @staticmethod
    def get_mask(img: Image.Image) -> Image.Image:
        if not isinstance(img, Image.Image):
            raise ValueError("Input must be a PIL Image")
        return img.getchannel("A")
    
    @staticmethod
    def get_data(im: ImageFile.ImageFile):
        return im.fp, im.filename