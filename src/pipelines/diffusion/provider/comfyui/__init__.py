"""
ComfyUI Image Generation Provider.

Provides unified access to text-to-image, image-to-image, and inpainting pipelines
through a single ComfyUIProvider class.
"""

from enum import Enum
from typing import List, Tuple, Optional, Dict, Any, Union

import pandas as pd
from PIL import Image, ImageFile

from .comfy_api import ComfyUIClientWrapper
from .comfyui_tasks import Txt2ImgPipeline, Img2ImgPipeline, InpaintPipeline

from src import logger


class GenerationMode(Enum):
    """Generation mode enumeration."""
    TXT2IMG = "txt2img"
    IMG2IMG = "img2img"
    INPAINT = "inpaint"


class ComfyUIProvider:
    """
    Unified ComfyUI provider that manages all image generation pipelines.

    Provides a single interface to access txt2img, img2img, and inpaint
    functionality through ComfyUI.

    Example:
        provider = ComfyUIProvider(
            model="sd_xl_base_1.0.safetensors",
            model_type="checkpoint",
            loras=["detail_enhancer.safetensors"],
            vae="sdxl_vae.safetensors"
        )

        # Text-to-image generation
        images, history = provider.txt2img.generate(
            positive_prompt="a beautiful landscape",
            negative_prompt="blurry, low quality",
            ...
        )

        # Image-to-image generation
        images, history = provider.img2img.generate(
            positive_prompt="enhance details",
            image_input=source_image,
            ...
        )

        # Inpainting
        images, history = provider.inpaint.generate(
            positive_prompt="fill with grass",
            image_input=masked_image,
            ...
        )
    """

    def __init__(
        self,
        model: str,
        model_type: str = "checkpoint",
        refiner: str = None,
        loras: List[str] = None,
        vae: str = None,
        host: str = "127.0.0.1",
        port: int = 8188
    ):
        """
        Initialize the ComfyUI provider.

        Args:
            model: Main checkpoint model name
            model_type: Model type ('checkpoint' or 'diffusers')
            refiner: Refiner model name (for SDXL)
            loras: List of LoRA model names
            vae: VAE model name or 'Default'
            host: ComfyUI server host
            port: ComfyUI server port
        """
        self.model = model
        self.model_type = model_type
        self.refiner = refiner
        self.loras = loras or []
        self.vae = vae
        self.host = host
        self.port = port

        # Lazy-initialized pipelines
        self._txt2img: Optional[Txt2ImgPipeline] = None
        self._img2img: Optional[Img2ImgPipeline] = None
        self._inpaint: Optional[InpaintPipeline] = None

        # Shared client
        self._client: Optional[ComfyUIClientWrapper] = None

    @property
    def client(self) -> ComfyUIClientWrapper:
        """Get or create the ComfyUI client."""
        if self._client is None:
            self._client = ComfyUIClientWrapper(host=self.host, port=self.port)
        return self._client

    @property
    def txt2img(self) -> Txt2ImgPipeline:
        """Get or create the text-to-image pipeline."""
        if self._txt2img is None:
            self._txt2img = Txt2ImgPipeline(
                model=self.model,
                model_type=self.model_type,
                refiner=self.refiner,
                loras=self.loras,
                vae=self.vae
            )
            # Share client instance
            self._txt2img.client = self.client
        return self._txt2img

    @property
    def img2img(self) -> Img2ImgPipeline:
        """Get or create the image-to-image pipeline."""
        if self._img2img is None:
            self._img2img = Img2ImgPipeline(
                model=self.model,
                model_type=self.model_type,
                refiner=self.refiner,
                loras=self.loras,
                vae=self.vae
            )
            # Share client instance
            self._img2img.client = self.client
        return self._img2img

    @property
    def inpaint(self) -> InpaintPipeline:
        """Get or create the inpainting pipeline."""
        if self._inpaint is None:
            self._inpaint = InpaintPipeline(
                model=self.model,
                model_type=self.model_type,
                refiner=self.refiner,
                loras=self.loras,
                vae=self.vae
            )
            # Share client instance
            self._inpaint.client = self.client
        return self._inpaint

    def get_pipeline(self, mode: Union[GenerationMode, str]):
        """
        Get the appropriate pipeline for the given generation mode.

        Args:
            mode: Generation mode (GenerationMode enum or string)

        Returns:
            The corresponding pipeline instance

        Raises:
            ValueError: If mode is not recognized
        """
        if isinstance(mode, str):
            mode = GenerationMode(mode.lower())

        if mode == GenerationMode.TXT2IMG:
            return self.txt2img
        elif mode == GenerationMode.IMG2IMG:
            return self.img2img
        elif mode == GenerationMode.INPAINT:
            return self.inpaint
        else:
            raise ValueError(f"Unknown generation mode: {mode}")

    def update_model(
        self,
        model: str = None,
        refiner: str = None,
        loras: List[str] = None,
        vae: str = None
    ) -> None:
        """
        Update model configuration for all pipelines.

        Args:
            model: New main model name
            refiner: New refiner model name
            loras: New list of LoRA names
            vae: New VAE model name
        """
        if model is not None:
            self.model = model
        if refiner is not None:
            self.refiner = refiner
        if loras is not None:
            self.loras = loras
        if vae is not None:
            self.vae = vae

        # Update existing pipelines
        for pipeline in [self._txt2img, self._img2img, self._inpaint]:
            if pipeline is not None:
                if model is not None:
                    pipeline.model = model
                if refiner is not None:
                    pipeline.refiner = refiner
                if loras is not None:
                    pipeline.loras = loras
                if vae is not None:
                    pipeline.vae = vae

    def get_models_list(self, folder: str = "checkpoints") -> List[str]:
        """
        Get list of available models.

        Args:
            folder: Model folder name

        Returns:
            List of model filenames
        """
        return self.client.get_models_list(folder)

    def get_system_stats(self) -> Dict[str, Any]:
        """Get ComfyUI system statistics."""
        return self.client.get_system_stats()

    def interrupt(self) -> None:
        """Interrupt current generation."""
        self.client.interrupt()

    def clear_queue(self) -> None:
        """Clear the generation queue."""
        self.client.clear_queue()

    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status."""
        return self.client.get_queue_status()


# Convenience function for quick access
def create_provider(
    model: str,
    model_type: str = "checkpoint",
    refiner: str = None,
    loras: List[str] = None,
    vae: str = None,
    host: str = "127.0.0.1",
    port: int = 8188
) -> ComfyUIProvider:
    """
    Create a ComfyUI provider instance.

    Args:
        model: Main checkpoint model name
        model_type: Model type ('checkpoint' or 'diffusers')
        refiner: Refiner model name (for SDXL)
        loras: List of LoRA model names
        vae: VAE model name or 'Default'
        host: ComfyUI server host
        port: ComfyUI server port

    Returns:
        Configured ComfyUIProvider instance
    """
    return ComfyUIProvider(
        model=model,
        model_type=model_type,
        refiner=refiner,
        loras=loras,
        vae=vae,
        host=host,
        port=port
    )


__all__ = [
    # Main provider
    "ComfyUIProvider",
    "create_provider",
    "GenerationMode",
    # Individual pipelines (for direct access)
    "Txt2ImgPipeline",
    "Img2ImgPipeline",
    "InpaintPipeline",
    # Client wrapper
    "ComfyUIClientWrapper",
]
