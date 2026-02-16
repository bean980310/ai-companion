import os
import traceback
import warnings
import platform

from pathlib import Path
from functools import partial

try:
    import crewai
except ImportError:
    warnings.warn("smolagents is not installed. Please install it to use smolagents features.", UserWarning)