"""
SOMA: Self-Organizing Memory Agent
==================================

Description:
------------
This module implements the Atomic Toolbox (often referred to as 'The Limbs') 
for the SOMA framework. It provides the programmatic interfaces for executing 
visual augmentations and counterfactual image editing (e.g., overlaying targets, 
removing distractors, or altering textures).



To maintain high-frequency execution in the main robotic control loop, this module 
is designed using a Client-Server (CS) architecture. Heavy foundational vision 
models (like Segment Anything 3) are not loaded into the local VRAM of the agent process. 
Instead, images are base64-encoded and transmitted to an independent HTTP service, 
ensuring the agent remains lightweight and non-blocking.
"""

import base64
import io
import logging
from typing import Any, Tuple

import numpy as np
import requests
from PIL import Image


def _encode_png_b64(image: np.ndarray) -> str:
    pil = Image.fromarray(image.astype(np.uint8)).convert("RGB")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _decode_png_b64(b64_str: str) -> np.ndarray:
    raw = base64.b64decode(b64_str)
    pil = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.array(pil)


class Sam3HttpClient:
    def __init__(self, base_url: str = "http://127.0.0.1:5001", timeout_s: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        resp = requests.post(url, json=payload, timeout=self.timeout_s)
        resp.raise_for_status()
        return resp.json()

    def visual_overlay(
        self,
        image: np.ndarray,
        *,
        target_object: str,
        color: Tuple[int, int, int] = (0, 255, 0),
        alpha: float = 0.4,
    ) -> np.ndarray:
        out = self._post(
            "/visual_overlay",
            {
                "image": _encode_png_b64(image),
                "target_object": target_object,
                "color": list(color),
                "alpha": alpha,
            },
        )
        if out.get("success") and out.get("image"):
            return _decode_png_b64(out["image"])
        return image

    def remove_distractor(self, image: np.ndarray, *, object_to_remove: str) -> np.ndarray:
        out = self._post(
            "/remove_distractor",
            {"image": _encode_png_b64(image), "object_to_remove": object_to_remove},
        )
        if out.get("success") and out.get("image"):
            return _decode_png_b64(out["image"])
        return image

    def replace_texture(self, image: np.ndarray, *, target_object: str, texture_image: np.ndarray) -> np.ndarray:
        out = self._post(
            "/replace_texture",
            {
                "image": _encode_png_b64(image),
                "target_object": target_object,
                "texture_image": _encode_png_b64(texture_image),
            },
        )
        if out.get("success") and out.get("image"):
            return _decode_png_b64(out["image"])
        return image

    def replace_background(
        self,
        image: np.ndarray,
        *,
        region_prompt: str,
        texture_image: np.ndarray,
        alpha: float = 1.0,
    ) -> np.ndarray:
        out = self._post(
            "/replace_background",
            {
                "image": _encode_png_b64(image),
                "region_prompt": region_prompt,
                "texture_image": _encode_png_b64(texture_image),
                "alpha": alpha,
            },
        )
        if out.get("success") and out.get("image"):
            return _decode_png_b64(out["image"])
        return image


class MCPTools:
    """
    SOMA Atomic Toolbox (The Limbs)

    Description:
    This version does not load the heavy SAM3 models into the main process. 
    Instead, it delegates visual operations to an independent SAM3 HTTP service 
    to ensure the primary control loop remains lightweight and fast.

    Tool API:
    - apply_visual_overlay(image, target, color)
    - remove_distractor(image, distractor)
    - replace_texture(image, target_object, texture_image)
    - replace_background(image, region_prompt, texture_image)
    """

    def __init__(self, sam3_base_url: str = "http://127.0.0.1:5001", timeout_s: float = 10.0):
        self.client = Sam3HttpClient(base_url=sam3_base_url, timeout_s=timeout_s)

    def apply_visual_overlay(
        self, image: np.ndarray, target: str, color: Tuple[int, int, int] = (0, 255, 0)
    ) -> np.ndarray:
        return self.client.visual_overlay(image, target_object=target, color=color)

    def remove_distractor(self, image: np.ndarray, distractor: str) -> np.ndarray:
        return self.client.remove_distractor(image, object_to_remove=distractor)

    def replace_texture(self, image: np.ndarray, target_object: str, texture_image: np.ndarray) -> np.ndarray:
        return self.client.replace_texture(image, target_object=target_object, texture_image=texture_image)

    def replace_background(
        self, image: np.ndarray, region_prompt: str, texture_image: np.ndarray, alpha: float = 1.0
    ) -> np.ndarray:
        return self.client.replace_background(
            image, region_prompt=region_prompt, texture_image=texture_image, alpha=alpha
        )

    def save_debug(self, image: np.ndarray, step: int, suffix: str = ""):
        path = f"debug_step_{step}_{suffix}.png"
        Image.fromarray(image.astype(np.uint8)).save(path)