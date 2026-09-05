from __future__ import annotations

import asyncio
import json
import time
import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from nanovllm.multimodal.openai_adapter import adapt_chat_messages
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
    def create_chat_completion(request: ChatCompletionRequest):
        try:
            adapted = adapt_chat_messages(
                request.messages,
                allow_local_files=engine.config.multimodal_allow_local_files,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if request.stream:
            if engine.config.multimodal_backend != "native":
                raise HTTPException(
                    status_code=501,
                    detail="Streaming requires the native Qwen-VL backend; HF fallback is non-streaming only.",
                )
            try:
                handle = engine.stream_multimodal_chat(
                    adapted.multimodal_request,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    top_p=request.top_p,
                    ignore_eos=request.ignore_eos,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except RuntimeError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc

            completion_id = f"chatcmpl-{uuid.uuid4().hex}"
            created = int(time.time())
            include_usage = bool(
                request.stream_options is not None
                and request.stream_options.include_usage
            )

            def chunk(choices, *, usage=None):
                payload = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request.model,
                    "choices": choices,
                }
                if usage is not None:
                    payload["usage"] = usage
                return "data: " + json.dumps(
                    payload,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ) + "\n\n"

            def next_event(iterator):
                try:
                    return True, next(iterator)
                except StopIteration:
                    return False, None

            async def stream_events():
                final_event = None
                try:
                    yield chunk([{
                        "index": 0,
                        "delta": {"role": "assistant"},
                        "finish_reason": None,
                    }])
                    iterator = iter(handle)
                    while True:
                        has_event, event = await asyncio.to_thread(next_event, iterator)
                        if not has_event:
                            break
                        if event.text_delta:
                            yield chunk([{
                                "index": 0,
                                "delta": {"content": event.text_delta},
                                "finish_reason": None,
                            }])
                        if event.finished:
                            final_event = event
                            yield chunk([{
                                "index": 0,
                                "delta": {},
                                "finish_reason": event.finish_reason or "stop",
                            }])
                    if final_event is None:
                        raise RuntimeError("native stream ended without a terminal event")
                    if include_usage:
                        prompt_tokens = final_event.prompt_tokens or 0
                        completion_tokens = final_event.completion_tokens or 0
                        yield chunk([], usage={
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": prompt_tokens + completion_tokens,
                        })
                    yield "data: [DONE]\n\n"
                finally:
                    handle.close()

            return StreamingResponse(stream_events(), media_type="text/event-stream")

        if adapted.has_images:
            try:
                text = engine.generate_multimodal_chat(
                    adapted.multimodal_request,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    top_p=request.top_p,
                    ignore_eos=request.ignore_eos,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except RuntimeError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
        else:
            outputs = engine.generate(
                [adapted.text_prompt],
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
