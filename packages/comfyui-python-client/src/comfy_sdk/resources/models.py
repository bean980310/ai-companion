from ..client import ComfyClient

class Models:
    def __init__(self, client: ComfyClient):
        self._client = client

    def list(self, folder: str = None) -> list[str]:
        """List models in a specific folder (e.g. 'checkpoints', 'loras')."""
        return self._client.get_models(folder)
