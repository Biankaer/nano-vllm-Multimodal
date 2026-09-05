from __future__ import annotations

import time
import uuid
from typing import Annotated, Any, Literal

from pydantic import AliasChoices, BaseModel, Field


class CompletionRequest(BaseModel):
    model: str | None = None
    prompt: str | list[str]
    max_tokens: int = Field(
        default=64,
        ge=1,
        validation_alias=AliasChoices("max_tokens", "max_completion_tokens"),
    )
    temperature: float = Field(default=1.0, ge=0.0)
    stream: bool = False
    ignore_eos: bool = False


class CompletionChoice(BaseModel):
    index: int
    text: str
    finish_reason: str | None = "stop"


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex}")
    object: Literal["text_completion"] = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str | None = None
    choices: list[CompletionChoice]
    usage: UsageInfo = Field(default_factory=UsageInfo)


class TextContentPart(BaseModel):
    type: Literal["text"]
    text: str


class ImageUrl(BaseModel):
    url: str
    detail: Literal["auto", "low", "high"] = "auto"


class ImageContentPart(BaseModel):
    type: Literal["image_url"]
    image_url: ImageUrl


ChatContentPart = Annotated[TextContentPart | ImageContentPart, Field(discriminator="type")]


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[ChatContentPart]


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int = Field(
        default=64,
        ge=1,
        validation_alias=AliasChoices("max_tokens", "max_completion_tokens"),
    )
    temperature: float = Field(default=1.0, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    stream: bool = False
    stream_options: "StreamOptions | None" = None
    ignore_eos: bool = False


class StreamOptions(BaseModel):
    include_usage: bool = False


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str | None = "stop"


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str | None = None
    choices: list[ChatCompletionChoice]
    usage: UsageInfo = Field(default_factory=UsageInfo)


class ErrorResponse(BaseModel):
    error: dict[str, Any]
