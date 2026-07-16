from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

from nanovllm.multimodal.schemas import ImageInput, MultimodalRequest
from nanovllm.serving.schemas import ChatMessage


@dataclass(slots=True)
class AdaptedChatRequest:
    """Normalized representation shared by the API and engine layers."""

    multimodal_request: MultimodalRequest
    text_prompt: str

    @property
    def has_images(self) -> bool:
        return bool(self.multimodal_request.images)


def adapt_chat_messages(
    messages: list[ChatMessage],
    *,
    request_id: str | None = None,
    allow_local_files: bool = False,
) -> AdaptedChatRequest:
    qwen_messages: list[dict[str, object]] = []
    images: list[ImageInput] = []
    prompt_lines: list[str] = []

    for message in messages:
        if isinstance(message.content, str):
            qwen_messages.append({"role": message.role, "content": message.content})
            prompt_lines.append(f"{message.role}: {message.content}")
            continue

        qwen_content: list[dict[str, object]] = []
        text_parts: list[str] = []
        message_image_count = 0
        for part in message.content:
            if part.type == "text":
                qwen_content.append({"type": "text", "text": part.text})
                text_parts.append(part.text)
                continue

            image_input = image_url_to_input(
                part.image_url.url,
                detail=part.image_url.detail,
                allow_local_files=allow_local_files,
            )
            images.append(image_input)
            message_image_count += 1
            qwen_content.append(
                {
                    "type": "image",
                    "image": _qwen_image_source(image_input),
                    "detail": image_input.detail,
                }
            )

        qwen_messages.append({"role": message.role, "content": qwen_content})
        prompt_text = "\n".join(text_parts)
        if message_image_count:
            prompt_text = f"{prompt_text}\n[Attached images: {message_image_count}]".strip()
        prompt_lines.append(f"{message.role}: {prompt_text}")

    prompt_lines.append("assistant:")
    text_prompt = "\n".join(prompt_lines)
    return AdaptedChatRequest(
        multimodal_request=MultimodalRequest(
            text=text_prompt,
            images=images,
            request_id=request_id,
            messages=qwen_messages,
        ),
        text_prompt=text_prompt,
    )


def image_url_to_input(url: str, *, detail: str = "auto", allow_local_files: bool = False) -> ImageInput:
    """Convert an OpenAI image_url value into the internal source format.

    HTTP(S), base64 data URLs, file URLs, and server-local paths are supported.
    Local paths are a NanoInfer extension intended for trusted deployments.
    """

    if not url:
        raise ValueError("image_url.url cannot be empty")
    parsed = urlparse(url)
    if url.startswith("data:image/"):
        if ";base64," not in url:
            raise ValueError("image data URL must use base64 encoding")
        try:
            base64.b64decode(url.split(",", 1)[1], validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("image data URL contains invalid base64 data") from exc
        return ImageInput(source=url, source_type="base64", detail=detail)
    if parsed.scheme in {"http", "https"}:
        return ImageInput(source=url, source_type="url", detail=detail)
    if parsed.scheme == "file":
        if not allow_local_files:
            raise ValueError("local image files are disabled; set NANOINFER_MM_ALLOW_LOCAL_FILES=1 to enable them")
        return ImageInput(source=unquote(parsed.path), source_type="path", detail=detail)
    if parsed.scheme:
        raise ValueError(f"Unsupported image URL scheme: {parsed.scheme}")
    if not allow_local_files:
        raise ValueError("local image files are disabled; set NANOINFER_MM_ALLOW_LOCAL_FILES=1 to enable them")
    return ImageInput(source=str(Path(url).expanduser()), source_type="path", detail=detail)


def _qwen_image_source(image_input: ImageInput) -> str:
    if image_input.source_type == "path":
        return f"file://{Path(image_input.source).expanduser().resolve()}"
    return image_input.source
