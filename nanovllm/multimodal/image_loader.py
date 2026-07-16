from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen

from PIL import Image

from nanovllm.multimodal.schemas import ImageInput


@dataclass(slots=True)
class LoadedImage:
    image: Image.Image
    cache_key: str
    source_type: str


def load_image(image_input: ImageInput, *, timeout: float = 10.0) -> LoadedImage:
    if image_input.source_type == "path":
        data = Path(image_input.source).read_bytes()
    elif image_input.source_type == "base64":
        data = base64.b64decode(_strip_base64_prefix(image_input.source))
    elif image_input.source_type == "url":
        with urlopen(image_input.source, timeout=timeout) as response:
            data = response.read()
    else:
        raise ValueError(f"Unsupported image source type: {image_input.source_type}")

    image = Image.open(BytesIO(data)).convert("RGB")
    return LoadedImage(
        image=image,
        cache_key=hashlib.sha256(data).hexdigest(),
        source_type=image_input.source_type,
    )


def _strip_base64_prefix(value: str) -> str:
    if "," in value and value.startswith("data:"):
        return value.split(",", 1)[1]
    return value
