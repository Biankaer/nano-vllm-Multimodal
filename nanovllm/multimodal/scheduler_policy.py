from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class MultimodalBatchPolicy:
    """Placeholder policy for future multimodal scheduling.

    Later phases can use this to bucket requests by image count, image resolution,
    and text token budget before they reach the engine scheduler.
    """

    max_images_per_batch: int = 8
    max_text_tokens_per_batch: int = 16384
    max_image_tokens_per_batch: int = 8192
    enable_image_bucketization: bool = True
