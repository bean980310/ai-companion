from dataclasses import dataclass
from typing import Optional, Dict

@dataclass
class ComfyResponse:
    prompt_id: str
    number: Optional[int] = None
    node_errors: Optional[Dict] = None
