from ..client import ComfyClient

class Prompt:
    def __init__(self, client: ComfyClient):
        self._client = client

    def send(self, workflow: dict):
        return self._client.queue_prompt(workflow)

    def retrieve(self, prompt_id: str):
        return self._client.get_history(prompt_id)
    
    def wait(self, prompt_id: str):
        return self._client.wait_for_completion(prompt_id)

    def history(self) -> dict:
        """Get all history."""
        return self._client.get_all_history()

    def delete(self, prompt_id: str):
        """Delete history for a prompt."""
        return self._client.delete_history(prompt_id)

    def clear(self):
        """Clear all history."""
        return self._client.clear_history()
