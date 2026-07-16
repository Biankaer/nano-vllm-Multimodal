from __future__ import annotations

from fastapi import FastAPI

from nanovllm.serving.engine_client import EngineClient, EngineClientConfig
from nanovllm.serving.openai_api import create_openai_router


def create_app(
    model_path: str,
    *,
    enforce_eager: bool = True,
    tensor_parallel_size: int = 1,
    multimodal_model_path: str | None = None,
    multimodal_device_map: str = "auto",
    multimodal_torch_dtype: str = "auto",
    multimodal_attn_implementation: str | None = None,
) -> FastAPI:
    engine = EngineClient(
        EngineClientConfig(
            model_path=model_path,
            enforce_eager=enforce_eager,
            tensor_parallel_size=tensor_parallel_size,
            multimodal_model_path=multimodal_model_path,
            multimodal_device_map=multimodal_device_map,
            multimodal_torch_dtype=multimodal_torch_dtype,
            multimodal_attn_implementation=multimodal_attn_implementation,
        )
    )

    app = FastAPI(title="NanoInfer", version="0.1.0")
    app.include_router(create_openai_router(engine))

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app
