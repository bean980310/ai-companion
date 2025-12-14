from ..client import ComfyClient

class Images:
    def __init__(self, client: ComfyClient):
        self._client = client

    def upload(self, data: bytes, name: str, overwrite: bool = False):
        return self._client.upload_image(data, name, overwrite)

    def download(self, filename: str, subfolder: str = "", folder_type: str = "output"):
        return self._client.get_images(filename, subfolder, folder_type)

    def upload_mask(self, data: bytes, name: str, original_ref: dict, overwrite: bool = False, mask_type: str = "mask"):
        return self._client.upload_mask(data, name, original_ref, overwrite, mask_type)

    def metadata(self, filename: str, subfolder: str = "", folder_type: str = "output"):
        return self._client.get_view_metadata(filename, subfolder, folder_type)
