from ..client import ComfyClient

class Users:
    def __init__(self, client: ComfyClient):
        self._client = client

    def list(self):
        """List all users."""
        return self._client.get_users()

    def create(self, username: str):
        """Create a new user."""
        return self._client.create_user(username)

class Userdata:
    def __init__(self, client: ComfyClient):
        self._client = client

    def get(self, file: str):
        """Get userdata file content."""
        return self._client.get_userdata(file)

    def move(self, file: str, dest: str):
        """Move userdata file."""
        return self._client.move_userdata(file, dest)
