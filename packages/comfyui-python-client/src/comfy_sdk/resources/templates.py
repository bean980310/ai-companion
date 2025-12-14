from ..client import ComfyClient

class Templates:
    def __init__(self, client: ComfyClient):
        self._client = client

    def list(self) -> list[str]:
        """List workflow templates."""
        return self._client.get_workflow_templates()
