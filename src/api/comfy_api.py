# """
# ComfyUI Client Wrapper using comfy_sdk library.
# Provides a high-level interface for interacting with ComfyUI.
# """

# import os
# import datetime
# from io import BytesIO
# from typing import Union, Dict, Any, List, Optional

# from PIL import Image, ImageOps

# from comfy_sdk import ComfyUI


# class ComfyUIClientWrapper:
#     """
#     A wrapper class for ComfyUI API interactions using comfy_sdk.
#     Provides methods for image generation, uploading, and model management.
#     """

#     def __init__(self, host: str = "127.0.0.1", port: int = 8188):
#         """
#         Initialize the ComfyUI client wrapper.

#         Args:
#             host: ComfyUI server host address
#             port: ComfyUI server port
#         """
#         self.host = host
#         self.port = port
#         self._comfy = ComfyUI(host=host, port=port)
#         self._client = self._comfy.client

#     @property
#     def client_id(self) -> str:
#         """Get the client ID for WebSocket connections."""
#         return self._client.client_id

#     @property
#     def server_address(self) -> str:
#         """Get the server address."""
#         return f"{self.host}:{self.port}"

#     def queue_prompt(self, prompt: Dict[str, Any]) -> Dict[str, Any]:
#         """
#         Queue a workflow prompt for execution.

#         Args:
#             prompt: The workflow prompt dictionary

#         Returns:
#             Response containing prompt_id and other metadata
#         """
#         response = self._comfy.prompt.send(prompt)
#         return {
#             "prompt_id": response.prompt_id,
#             "number": response.number,
#             "node_errors": response.node_errors
#         }

#     def get_image(self, filename: str, subfolder: str = "", folder_type: str = "output") -> bytes:
#         """
#         Get image data from ComfyUI.

#         Args:
#             filename: Name of the image file
#             subfolder: Subfolder path
#             folder_type: Type of folder ('input', 'output', 'temp')

#         Returns:
#             Raw image bytes
#         """
#         return self._comfy.images.download(filename, subfolder, folder_type)

#     def get_history(self, prompt_id: str) -> Dict[str, Any]:
#         """
#         Get execution history for a specific prompt.

#         Args:
#             prompt_id: The prompt ID to retrieve history for

#         Returns:
#             History data for the prompt
#         """
#         return self._comfy.prompt.retrieve(prompt_id)

#     def generate_images(
#         self,
#         prompt: Dict[str, Any],
#         output_dir: str = "outputs",
#         save_to_disk: bool = True
#     ) -> Dict[str, List[bytes]]:
#         """
#         Generate images from a workflow prompt.

#         Args:
#             prompt: The workflow prompt dictionary
#             output_dir: Directory to save generated images
#             save_to_disk: Whether to save images to disk

#         Returns:
#             Dictionary mapping node IDs to lists of image data
#         """
#         response = self._comfy.prompt.send(prompt)
#         prompt_id = response.prompt_id

#         if save_to_disk and not os.path.exists(output_dir):
#             os.makedirs(output_dir)

#         history = self._comfy.prompt.wait(prompt_id)

#         output_images: Dict[str, List[bytes]] = {}

#         if prompt_id not in history:
#             return output_images

#         prompt_history = history[prompt_id]

#         for node_id, node_output in prompt_history.get("outputs", {}).items():
#             images_output = []

#             if "images" in node_output:
#                 for image_info in node_output["images"]:
#                     image_data = self.get_image(
#                         image_info["filename"],
#                         image_info.get("subfolder", ""),
#                         image_info.get("type", "output")
#                     )
#                     images_output.append(image_data)

#                     if save_to_disk:
#                         save_path = os.path.join(output_dir, image_info["filename"])
#                         try:
#                             with open(save_path, "wb") as f:
#                                 f.write(image_data)
#                             print(f"Saved image to {save_path}")
#                         except Exception as e:
#                             print(f"Error saving image {save_path}: {e}")

#             output_images[node_id] = images_output

#         return output_images

#     def upload_image(
#         self,
#         input_img: Union[str, Image.Image, None],
#         subfolder: str = "",
#         overwrite: bool = False
#     ) -> Optional[str]:
#         """
#         Upload an image to ComfyUI input folder.

#         Args:
#             input_img: Image path or PIL Image object
#             subfolder: Target subfolder
#             overwrite: Whether to overwrite existing file

#         Returns:
#             Uploaded image filename or None on failure
#         """
#         if input_img is None:
#             return None

#         if isinstance(input_img, Image.Image):
#             buffer = BytesIO()
#             input_img.save(buffer, format="PNG")
#             file_data = buffer.getvalue()
#             file_name = f"uploaded_image_{datetime.datetime.now().strftime('%y%m%d_%H%M%S')}.png"
#         else:
#             with Image.open(input_img) as im:
#                 buffer = BytesIO()
#                 im.save(buffer, format="PNG")
#                 file_data = buffer.getvalue()
#                 file_name = os.path.basename(input_img)

#         try:
#             result = self._comfy.images.upload(file_data, file_name, overwrite)
#             image_name = result.get("name", file_name)
#             print(f"img2img upload: {image_name}")
#             return image_name
#         except Exception as e:
#             print(f"Error uploading image: {e}")
#             return None

#     def upload_mask(
#         self,
#         original_img: Union[str, Image.Image],
#         mask_img: Dict[str, Any],
#         subfolder: str = "clipspace",
#         overwrite: bool = False
#     ) -> Optional[str]:
#         """
#         Upload a mask image for inpainting.

#         Args:
#             original_img: Original image path or PIL Image
#             mask_img: Mask data dictionary with 'background' and 'layers' keys
#             subfolder: Target subfolder
#             overwrite: Whether to overwrite existing file

#         Returns:
#             Uploaded mask filename or None on failure
#         """
#         if isinstance(original_img, str):
#             original_file_name = os.path.basename(original_img)
#         else:
#             original_file_name = f"original_{datetime.datetime.now().strftime('%y%m%d_%H%M%S')}.png"

#         with Image.open(mask_img['layers'][0]) as mask_pil:
#             mask_temp = mask_pil.getchannel('A')
#             new_alpha = ImageOps.invert(mask_temp)
#             new_mask = Image.new('L', mask_pil.size)
#             new_mask.putalpha(new_alpha)

#             buffer = BytesIO()
#             new_mask.save(buffer, format="PNG")
#             file_data = buffer.getvalue()

#         suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
#         new_file_name = f"clipspace-mask_{suffix}.png"

#         original_ref = {
#             "filename": original_file_name,
#             "subfolder": "",
#             "type": "input"
#         }

#         try:
#             result = self._comfy.images.upload_mask(
#                 file_data,
#                 new_file_name,
#                 original_ref,
#                 overwrite,
#                 "input"
#             )
#             mask_name = result.get("name", new_file_name)
#             print(f"inpaint upload: {mask_name}")
#             return mask_name
#         except Exception as e:
#             print(f"Error uploading mask: {e}")
#             return None

#     def text2image_generate(self, prompt: Dict[str, Any]) -> Dict[str, List[bytes]]:
#         """
#         Generate images from a text-to-image workflow.

#         Args:
#             prompt: The workflow prompt dictionary

#         Returns:
#             Dictionary mapping node IDs to lists of image data
#         """
#         return self.generate_images(prompt)

#     def image2image_generate(self, prompt: Dict[str, Any]) -> Dict[str, List[bytes]]:
#         """
#         Generate images from an image-to-image workflow.

#         Args:
#             prompt: The workflow prompt dictionary

#         Returns:
#             Dictionary mapping node IDs to lists of image data
#         """
#         return self.generate_images(prompt)

#     def get_models_list(self, folder: str) -> List[str]:
#         """
#         Get list of models in a specific folder.

#         Args:
#             folder: Model folder name (e.g., 'checkpoints', 'loras')

#         Returns:
#             List of model filenames
#         """
#         return self._comfy.models.list(folder)

#     def get_system_stats(self) -> Dict[str, Any]:
#         """Get ComfyUI system statistics."""
#         return self._comfy.system.stats()

#     def interrupt(self) -> None:
#         """Interrupt current execution."""
#         self._comfy.queue.interrupt()

#     def clear_queue(self) -> None:
#         """Clear the execution queue."""
#         self._comfy.queue.clear()

#     def get_queue_status(self) -> Dict[str, Any]:
#         """Get current queue status."""
#         return self._comfy.queue.status()


# Client = ComfyUIClientWrapper

# client = Client()