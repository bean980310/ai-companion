import os
import traceback
import warnings
import platform

from pathlib import Path
from functools import partial

try:
    import openai
    import agents
    from agents import Agent, Runner
except ImportError:
    warnings.warn("openai-agents is not installed. Please install it to use this features.", UserWarning)

agent = Agent()