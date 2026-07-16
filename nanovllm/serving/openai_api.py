from __future__ import annotations

from fastapi import APIRouter, HTTPException

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
    def create_chat_completion(request: ChatCompletionRequest) -> ChatCompletionResponse:
        if request.stream:
            raise HTTPException(status_code=501, detail="Streaming will be implemented in Phase 2.")

        try:
            adapted = adapt_chat_messages(
                request.messages,
                allow_local_files=engine.config.multimodal_allow_local_files,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if adapted.has_images:
            try:
                text = engine.generate_multimodal_chat(
                    adapted.multimodal_request,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    top_p=request.top_p,
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
