from __future__ import annotations

from fastapi import APIRouter, HTTPException

from nanovllm.serving.engine_client import EngineClient
from nanovllm.serving.schemas import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
)


def create_openai_router(engine: EngineClient) -> APIRouter:
    router = APIRouter()

    @router.post("/v1/completions", response_model=CompletionResponse)
    def create_completion(request: CompletionRequest) -> CompletionResponse:
        if request.stream:
            raise HTTPException(status_code=501, detail="Streaming will be implemented in Phase 2.")

        prompts = request.prompt if isinstance(request.prompt, list) else [request.prompt]
        outputs = engine.generate(
            prompts,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            ignore_eos=request.ignore_eos,
        )
        choices = [
            CompletionChoice(index=i, text=output["text"], finish_reason="stop")
            for i, output in enumerate(outputs)
        ]
        return CompletionResponse(model=request.model, choices=choices)

    @router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    def create_chat_completion(request: ChatCompletionRequest) -> ChatCompletionResponse:
        if request.stream:
            raise HTTPException(status_code=501, detail="Streaming will be implemented in Phase 2.")

        if _has_image_content(request.messages):
            qwen_messages = _messages_to_qwen25_vl(request.messages)
            try:
                text = engine.generate_multimodal_chat(
                    qwen_messages,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                )
            except RuntimeError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
        else:
            prompt = _messages_to_prompt(request.messages)
            outputs = engine.generate(
                [prompt],
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                ignore_eos=request.ignore_eos,
            )
            text = outputs[0]["text"]

        message = ChatMessage(role="assistant", content=text)
        return ChatCompletionResponse(
            model=request.model,
            choices=[ChatCompletionChoice(index=0, message=message, finish_reason="stop")],
        )

    return router


def _messages_to_prompt(messages: list[ChatMessage]) -> str:
    lines: list[str] = []
    for message in messages:
        content = message.content
        if isinstance(content, str):
            text = content
        else:
            text_parts = [part.text for part in content if part.type == "text"]
            image_count = sum(1 for part in content if part.type == "image_url")
            text = "\n".join(text_parts)
            if image_count:
                text = f"{text}\n[Attached images: {image_count}]".strip()
        lines.append(f"{message.role}: {text}")
    lines.append("assistant:")
    return "\n".join(lines)


def _has_image_content(messages: list[ChatMessage]) -> bool:
    for message in messages:
        content = message.content
        if isinstance(content, list) and any(part.type == "image_url" for part in content):
            return True
    return False


def _messages_to_qwen25_vl(messages: list[ChatMessage]) -> list[dict[str, object]]:
    qwen_messages: list[dict[str, object]] = []
    for message in messages:
        content = message.content
        if isinstance(content, str):
            qwen_messages.append({"role": message.role, "content": content})
            continue

        qwen_content: list[dict[str, object]] = []
        for part in content:
            if part.type == "text":
                qwen_content.append({"type": "text", "text": part.text})
            elif part.type == "image_url":
                qwen_content.append({"type": "image", "image": part.image_url.url, "detail": part.image_url.detail})
        qwen_messages.append({"role": message.role, "content": qwen_content})
    return qwen_messages
