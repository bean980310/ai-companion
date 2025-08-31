from io import BytesIO
import numpy as np
import torch
import PIL
# import tensorflow as tf
# import tf_keras as keras
# import keras
from typing import Callable, Any
from PIL import Image, ImageOps, ImageFilter, ImageFile, ImageSequence, UnidentifiedImageError
import datetime
import os

class ImageProcessor:
    def __init__(self):
        pass

    @classmethod
    def read_image(cls, img):
        if isinstance(img, Image.Image):
            return cls.read_image_from_pil(img)
        # if isinstance(img, (bytes, bytearray)):
        #     return cls.read_image_from_bytes(img)
        if isinstance(img, str):
            return cls.read_image_from_str(img)
        # if isinstance(img, (np.ndarray, torch.Tensor)):
        #     return cls.read_image_from_array(img)

    # @classmethod
    # def read_image_for_inpaint(cls, img_list, original_img):

        
    @staticmethod
    def read_image_from_pil(img: Image.Image) -> tuple[str, bytes]:
        fp = BytesIO()
        img.save(fp, format="PNG")
        filename = f"uploaded_image_{datetime.datetime.now().strftime('%y%m%d_%H%M%S')}.png"

        return filename.strip(), fp.getvalue()

    @staticmethod
    def read_image_from_bytes(img_bytes: bytes | bytearray):
        return Image.open(BytesIO(img_bytes))
    
    def read_image_from_str(self, img_path: str) -> tuple[str, bytes]:
        im = self.open_image(img_path)
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
    
    @staticmethod
    def open_image(img_path: str) -> ImageFile.ImageFile:
        return Image.open(img_path)

    @staticmethod
    def load_image(img_path: str) -> Image.Image:
        operator = lambda fn, arg: fn(arg)
        img = Image.open(img_path)

        output_images = []
        output_masks = []
        w, h = None, None

        excluded_formats = ['MPO']

        for i in ImageSequence.Iterator(img):
            i = ImageOps.exif_transpose(i)

            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]

            if image.size[0] != w or image.size[1] != h:
                continue

            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            elif i.mode == 'P' and 'transparency' in i.info:
                mask = np.array(i.convert('RGBA').getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return (output_image, output_mask)