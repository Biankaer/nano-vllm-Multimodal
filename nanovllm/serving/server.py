from __future__ import annotations

from fastapi import FastAPI

from nanovllm.serving.engine_client import EngineClient, EngineClientConfig
from nanovllm.serving.openai_api import create_openai_router


def create_app(
    model_path: str,
    *,
    enforce_eager: bool = False,
    gpu_memory_utilization: float = 0.5,
    tensor_parallel_size: int = 1,
    multimodal_model_path: str | None = None,
    multimodal_backend: str = "native",
    multimodal_device_map: str = "auto",
    multimodal_torch_dtype: str = "auto",
    multimodal_attn_implementation: str | None = None,
    multimodal_max_batch_size: int = 8,
    multimodal_max_images_per_batch: int = 8,
    multimodal_batch_wait_ms: float | None = None,
    multimodal_batch_quiet_ms: float = 3.0,
    multimodal_batch_max_wait_ms: float = 20.0,
    multimodal_cache_capacity: int = 256,
    multimodal_feature_cache_bytes: int = 2 * 1024**3,
    multimodal_allow_local_files: bool = False,
) -> FastAPI:
    engine = EngineClient(
        EngineClientConfig(
            model_path=model_path,
            enforce_eager=enforce_eager,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            multimodal_model_path=multimodal_model_path,
            multimodal_backend=multimodal_backend,
            multimodal_device_map=multimodal_device_map,
            multimodal_torch_dtype=multimodal_torch_dtype,
            multimodal_attn_implementation=multimodal_attn_implementation,
            multimodal_max_batch_size=multimodal_max_batch_size,
            multimodal_max_images_per_batch=multimodal_max_images_per_batch,
            multimodal_batch_wait_ms=multimodal_batch_wait_ms,
            multimodal_batch_quiet_ms=multimodal_batch_quiet_ms,
            multimodal_batch_max_wait_ms=multimodal_batch_max_wait_ms,
            multimodal_cache_capacity=multimodal_cache_capacity,
            multimodal_feature_cache_bytes=multimodal_feature_cache_bytes,
            multimodal_allow_local_files=multimodal_allow_local_files,
        )
    )

    app = FastAPI(title="NanoInfer", version="0.1.0")
    app.include_router(create_openai_router(engine))

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.on_event("shutdown")
    def shutdown() -> None:
        engine.shutdown()

    @app.get("/metrics/multimodal")
    def multimodal_metrics() -> dict:
        stats = engine.multimodal_stats()
        if stats is None:
            return {"status": "not_initialized"}
        return stats

    return app
