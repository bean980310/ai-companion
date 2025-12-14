from ..client import ComfyClient

class Queue:
    def __init__(self, client: ComfyClient):
        self._client = client

    def interrupt(self):
        """Interrupt current execution."""
        return self._client.interrupt()

    def clear(self):
        """Clear the execution queue."""
        return self._client.clear_queue()

    def status(self) -> dict:
        """Get queue status (running/pending info)."""
        return self._client.get_queue()
