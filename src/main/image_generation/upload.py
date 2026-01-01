"""
ComfyUI Image Upload module using comfy_sdk library.
Provides image and mask upload functionality for ComfyUI.
"""

import os
import datetime
from io import BytesIO
from typing import Union, Dict, Any, Optional

from PIL import Image, ImageOps

from comfy_sdk import ComfyUI

from .image import ImageProcessor


class ComfyUIImageUpload:
    """
    Handles image and mask uploads to ComfyUI using comfy_sdk.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8188):
        """
        Initialize the ComfyUI image upload handler.

        Args:
            host: ComfyUI server host address
            port: ComfyUI server port
        """
        self.host = host
        self.port = port
        self._comfy = ComfyUI(host=host, port=port)

    @property
    def server_address(self) -> str:
        """Get the server address."""
        return f"{self.host}:{self.port}"

    def upload_image(
        self,
        input_img: Union[str, Image.Image, None],
        subfolder: str = "",
        overwrite: bool = False
    ) -> Optional[str]:
        """
        Upload an image to ComfyUI input folder.

        Args:
            input_img: Image path or PIL Image object
            subfolder: Target subfolder (currently unused by SDK)
            overwrite: Whether to overwrite existing file

        Returns:
            Uploaded image filename or None on failure
        """
        if input_img is None:
            return None

        file_name, file_data = ImageProcessor.read_image(input_img)

        try:
            result = self._comfy.images.upload(file_data, file_name, overwrite)
            image_name = result.get("name", file_name)
            print(f"img2img upload: {image_name}")
            return image_name
        except Exception as e:
            print(f"Error uploading image: {e}")
            return None

    def upload_mask(
        self,
        original_img: Union[str, Image.Image],
        mask_img: Dict[str, Any],
        subfolder: str = "clipspace",
        overwrite: bool = False
    ) -> Optional[str]:
        """
        Upload a mask image for inpainting.

        Args:
            original_img: Original image path or PIL Image
            mask_img: Mask data dictionary with 'background' and 'layers' keys
            subfolder: Target subfolder (currently unused by SDK)
            overwrite: Whether to overwrite existing file

        Returns:
            Uploaded mask filename or None on failure
        """
        if isinstance(original_img, str):
            original_file_name = os.path.basename(original_img)
        else:
            original_file_name = f"original_{datetime.datetime.now().strftime('%y%m%d_%H%M%S')}.png"

        with Image.open(mask_img['layers'][0]) as mask_pil:
            mask_temp = mask_pil.getchannel('A')
            new_alpha = ImageOps.invert(mask_temp)
            new_mask = Image.new('L', mask_pil.size)
            new_mask.putalpha(new_alpha)

            buffer = BytesIO()
            new_mask.save(buffer, format="PNG")
            file_data = buffer.getvalue()

        suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        new_file_name = f"clipspace-mask_{suffix}.png"

        original_ref = {
            "filename": original_file_name,
            "subfolder": "",
            "type": "input"
        }

        try:
            result = self._comfy.images.upload_mask(
                file_data,
                new_file_name,
                original_ref,
                overwrite,
                "input"
            )
            mask_name = result.get("name", new_file_name)
            print(f"inpaint upload: {mask_name}")
            return mask_name
        except Exception as e:
            print(f"Error uploading mask: {e}")
            return None