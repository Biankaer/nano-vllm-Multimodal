from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(slots=True)
class ImageInput:
    source: str
    source_type: Literal["url", "base64", "path"] = "url"
    detail: Literal["auto", "low", "high"] = "auto"


@dataclass(slots=True)
class MultimodalRequest:
    text: str
    images: list[ImageInput] = field(default_factory=list)
    request_id: str | None = None
