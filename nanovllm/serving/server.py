from __future__ import annotations

from fastapi import FastAPI

from nanovllm.serving.engine_client import EngineClient, EngineClientConfig
from nanovllm.serving.openai_api import create_openai_router


def create_app(
    model_path: str,
    *,
    enforce_eager: bool = True,
    tensor_parallel_size: int = 1,
) -> FastAPI:
    engine = EngineClient(
        EngineClientConfig(
            model_path=model_path,
            enforce_eager=enforce_eager,
            tensor_parallel_size=tensor_parallel_size,
        )
    )

    app = FastAPI(title="NanoInfer", version="0.1.0")
    app.include_router(create_openai_router(engine))

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app
