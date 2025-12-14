from .client import ComfyClient
from .resources import Prompt, Images, System, Queue, Models, Templates, Users, Userdata

class ComfyUI:
    def __init__(self, host: str = "127.0.0.1", port: int = 8188):
        self.client = ComfyClient(host=host, port=port)
        self.prompt = Prompt(self.client)
        self.images = Images(self.client)
        self.system = System(self.client)
        self.queue = Queue(self.client)
        self.models = Models(self.client)
        self.templates = Templates(self.client)
        self.users = Users(self.client)
        self.userdata = Userdata(self.client)

__all__ = ["ComfyUI", "ComfyClient"]
