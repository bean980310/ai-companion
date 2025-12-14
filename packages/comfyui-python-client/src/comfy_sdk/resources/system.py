from ..client import ComfyClient

class System:
    def __init__(self, client: ComfyClient):
        self._client = client

    def stats(self):
        """Get system statistics."""
        return self._client.get_system_stats()

    def extensions(self):
        """Get list of extensions."""
        return self._client.get_extensions()

    def free(self, unload_models: bool = False, free_memory: bool = False):
        """Free memory (unload models/free vram)."""
        return self._client.free(unload_models, free_memory)

    def embeddings(self) -> list[str]:
        """Get list of embeddings."""
        return self._client.get_embeddings()

    def features(self) -> dict:
        """Get system features."""
        return self._client.get_features()
