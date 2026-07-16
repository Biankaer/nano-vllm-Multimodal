# NanoInfer Modification Plan

## 1. Project Positioning

NanoInfer is a job-oriented fork of nano-vLLM. The goal is to turn a compact offline
LLM inference engine into a lightweight vLLM-style LLM/VLM serving system.

The final project should demonstrate:

- LLM serving architecture design.
- Request lifecycle management.
- Continuous batching and chunked prefill scheduling.
- KV cache and prefix cache optimization.
- OpenAI-compatible text and multimodal APIs.
- Multimodal serving optimizations for image-text models.
- Reproducible offline, serving, speculative, and multimodal benchmarks.

## 2. Target Resume Story

Suggested project title:

```text
NanoInfer: A Lightweight vLLM-style LLM/VLM Serving Engine
```

Suggested resume description:

```text
- Refactored nano-vLLM into an EngineCore / Executor / GPUWorker layered architecture,
  decoupling request lifecycle management, scheduling, and GPU execution paths.
- Implemented OpenAI-compatible completion and chat completion APIs with streaming SSE,
  request timeout, queueing, and serving benchmark support.
- Implemented token-budget based chunked prefill and KV cache statistics to reduce
  long-prompt blocking and quantify TTFT / TPOT / throughput tradeoffs.
- Added n-gram prompt lookup speculative decoding with draft / verify / commit /
  fallback stages and benchmarked decode throughput gains.
- Extended the engine toward multimodal VLM serving with image preprocessing,
  multimodal chat templates, image feature cache, image-text batching, and VLM
  benchmark coverage.
```

## 3. Architecture Roadmap

Current nano-vLLM path:

```text
LLMEngine -> Scheduler -> ModelRunner -> Qwen3ForCausalLM
```

Target architecture:

```text
API Layer
  -> EngineCore
      -> RequestManager
      -> Scheduler
      -> KVCacheManager / BlockManager
  -> Executor
      -> GPUWorker
          -> ModelRunner
              -> Text Model / Vision-Language Model
```

The first implementation should keep the original `LLM.generate()` API working.
New serving and multimodal modules should wrap or extend the existing engine rather
than rewriting the whole engine in one pass.

## 4. Implementation Phases

### Phase 1: Project Packaging and Serving Skeleton

Deliverables:

- Add this modification plan.
- Add `nanovllm/serving/` package.
- Add OpenAI-style request/response schemas.
- Add a FastAPI app skeleton.
- Add a simple blocking `/v1/completions` path.
- Keep original offline `example.py` usable.

Files:

- `docs/MODIFICATION_PLAN.md`
- `nanovllm/serving/schemas.py`
- `nanovllm/serving/openai_api.py`
- `nanovllm/serving/server.py`
- `nanovllm/serving/engine_client.py`

### Phase 2: EngineCore / Executor Split

Status: basic split implemented.

Deliverables:

- [x] Introduce `EngineCore` as request lifecycle owner.
- [x] Introduce `Executor` as model execution abstraction.
- [x] Move direct `ModelRunner.call("run", ...)` behind `Executor`.
- [x] Preserve existing scheduler behavior.
- [ ] Add async request admission for serving.
- [ ] Add per-request timing and state tracking.
- [ ] Add streaming token handoff.

Target files:

- `nanovllm/engine/engine_core.py`
- `nanovllm/engine/executor.py`
- `nanovllm/engine/gpu_worker.py`
- `nanovllm/engine/llm_engine.py`

### Phase 3: Continuous Batching and Chunked Prefill

Deliverables:

- Support dynamic request admission while serving loop is running.
- Add explicit token budget config.
- Track TTFT and TPOT per request.
- Improve long-prompt chunked prefill fairness.
- Add scheduler metrics.

Metrics:

- TTFT.
- TPOT.
- Prefill tokens/s.
- Decode tokens/s.
- Queue length.
- Running sequence count.
- KV cache free/used blocks.

### Phase 4: Speculative Decoding

Deliverables:

- Add speculative decoding interfaces:
  - draft
  - verify
  - commit
  - fallback
- Implement n-gram prompt lookup draft backend.
- Add benchmark for ordinary decode vs speculative decode.

Target files:

- `nanovllm/speculative/base.py`
- `nanovllm/speculative/ngram.py`
- `nanovllm/speculative/manager.py`
- `bench_speculative.py`

### Phase 5: Multimodal VLM Serving

Deliverables:

- Add multimodal request schema compatible with OpenAI chat content parts.
- Support image URL and base64 image input.
- Add image preprocessing abstraction.
- Add multimodal processor abstraction.
- Add image feature cache.
- Add image-text batching policy hooks.

Target files:

- `nanovllm/multimodal/schemas.py`
- `nanovllm/multimodal/image_loader.py`
- `nanovllm/multimodal/processor.py`
- `nanovllm/multimodal/cache.py`
- `nanovllm/multimodal/scheduler_policy.py`
- `bench_multimodal.py`

## 5. Multimodal Optimization Plan

Multimodal serving must not stop at "accept image input". It should include concrete
optimizations for image-text models.

### 5.1 Image Preprocessing Optimization

Goals:

- Avoid blocking the decode loop with slow image loading.
- Normalize image preprocessing across URL, local file, and base64 inputs.
- Track preprocessing latency separately from model latency.

Planned features:

- Async image fetching for URLs.
- Configurable resize/crop policy.
- Image hash for cache keys.
- Preprocessing timing metrics.

### 5.2 Vision Encoder Optimization

Goals:

- Decouple vision encoding from text decode.
- Batch images with similar shapes.
- Cache image features for repeated images.

Planned features:

- `ImageFeatureCache`.
- Batch vision encode interface.
- Image feature reuse for repeated multi-turn image QA.

### 5.3 Visual Token Compression

Goals:

- Reduce image token count.
- Reduce KV cache pressure.
- Improve TTFT for image-heavy requests.

Candidate strategies:

- Patch pooling.
- Top-k patch selection.
- Fixed latent token compression.
- Resolution-aware token budget.

### 5.4 Multimodal Scheduling

Goals:

- Avoid image-heavy requests blocking short text requests.
- Split serving into stages:
  - image preprocessing
  - vision encoding
  - text prefill
  - decode

Planned scheduling knobs:

- Text token budget.
- Image token budget.
- Max images per batch.
- Resolution buckets.
- Separate queues for text-only and image-text requests.

### 5.5 Multimodal KV Cache Optimization

Goals:

- Reuse repeated image prefixes.
- Track image token KV cache usage.
- Quantify image feature cache hit rate.

Metrics:

- Image feature cache hit rate.
- Multimodal prefix cache hit rate.
- Image token count per request.
- KV blocks used by image tokens.

## 6. Benchmark Plan

### Offline Benchmark

Compare:

- Original nano-vLLM.
- NanoInfer baseline.
- NanoInfer with chunked prefill.
- NanoInfer with speculative decoding.

Metrics:

- tokens/s.
- TTFT.
- TPOT.
- total latency.
- GPU memory usage.

### Serving Benchmark

Metrics:

- QPS.
- P50 / P90 / P99 latency.
- TTFT.
- TPOT.
- queue time.
- request timeout count.

### Multimodal Benchmark

Variables:

- Image count.
- Image resolution.
- Text prompt length.
- Batch size.
- Image cache hit/miss.

Metrics:

- image preprocessing latency.
- vision encode latency.
- TTFT.
- TPOT.
- GPU memory usage.
- image feature cache hit rate.

## 7. First Coding Milestone

The first coding milestone should be intentionally small:

1. Add serving package and schemas.
2. Add a FastAPI app factory.
3. Add an engine client wrapper around `LLM.generate()`.
4. Add basic text completions endpoint.
5. Add multimodal schema stubs and image cache skeleton.
6. Add compile checks.

This creates a clean project base without breaking the original engine.
