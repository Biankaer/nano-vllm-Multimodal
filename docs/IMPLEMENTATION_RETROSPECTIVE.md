# NanoInfer 修改与问题处理全记录

## 1. 文档目的

本文档记录从原始 `nano-vllm` 学习项目到当前 NanoInfer 多模态推理服务项目的全部工程修改，包括：

- 项目定位与贡献边界。
- 五个阶段性 Git 提交的修改内容。
- 新增和修改的模块及责任。
- 实现过程中遇到的问题、根因和解决方案。
- 已完成的验证、未完成的 GPU 验证与已知风险。
- 当前可以如何准确地描述这个项目。

这份文档既是开发复盘，也是后续服务器联调、性能优化和面试准备的依据。

## 2. 基线与贡献边界

### 2.1 继承的原始能力

项目基于原始 `nano-vllm` 实现，下列核心能力是原项目的基线：

- Qwen3 文本模型权重加载。
- Transformer 基础层，包括 attention、linear、RMSNorm、RoPE 和 sampler。
- KV cache block 管理。
- prefix block 复用。
- prefill 和 decode 调度。
- chunked prefill 的基础行为。
- tensor parallel 多进程执行。
- `LLM.generate()` 离线生成 API。

原始代码在之前的学习过程中已加入较详细的中文注释。NanoInfer 保留了这些注释，但没有把原项目的 CUDA、Triton 和模型算子宣称为本次新增贡献。

### 2.2 本项目新增的主要贡献

- 将原本集中的推理主循环拆分为 `EngineCore / Executor / GPUWorker` 层次。
- 增加 OpenAI 风格的 completion 和 chat completion 服务接口。
- 增加请求生命周期、调度器和 KV cache 指标。
- 接入 HuggingFace Qwen2.5-VL 作为第一个真实多模态后端。
- 实现 OpenAI 图文请求到 Qwen2.5-VL 消息格式的独立适配层。
- 实现 engine 所有的多模态请求队列、微批处理、图文预算调度和图像对象缓存。
- 实现并发多模态 benchmark 和 JSON 指标输出。

### 2.3 当前不应宣称已完成的能力

- 还不是自研 Qwen2.5-VL CUDA runtime。
- 还没有拆出并单独调度 vision encoder。
- image feature cache 已有 engine 层接口，但还没有复用真实的 vision embedding。
- 多模态 batching 当前是请求级微批处理，还不是 token 级 VLM continuous batching。
- 还没有 SSE 流式输出。
- 还没有真实 GPU benchmark 结果或性能提升百分比。

## 3. 当前架构

### 3.1 文本请求路径

```text
OpenAI API / Offline API
  -> EngineClient / LLM
  -> LLMEngine
  -> EngineCore
      -> Scheduler
      -> Executor
          -> GPUWorker
              -> ModelRunner
                  -> Qwen3ForCausalLM
```

### 3.2 多模态请求路径

```text
/v1/chat/completions
  -> OpenAI request schema
  -> OpenAI-to-Qwen adapter
  -> MultimodalRequest
  -> MultimodalEngineCore
      -> waiting queue
      -> configurable batch wait window
      -> MultimodalBatcher
          -> generation parameter compatibility
          -> image bucket compatibility
          -> text/image token budgets
      -> MultimodalManager
          -> decoded image LRU cache
          -> image feature cache hook
      -> Qwen25VLBackend.generate_chat_batch()
  -> HuggingFace Qwen2.5-VL
```

## 4. Git 阶段性修改记录

### 4.1 `69097c8 Initial NanoInfer serving and multimodal scaffold`

目标：将本地学习版 nano-vLLM 复制为独立求职项目，建立服务和多模态扩展骨架。

主要修改：

- 建立独立 Git 仓库和 GitHub 远程仓库。
- 新增 `docs/MODIFICATION_PLAN.md`，定义项目定位、分阶段目标和 benchmark 计划。
- 新增 `nanovllm/serving/` 包。
- 实现 OpenAI 风格的请求与响应 Pydantic schema。
- 实现 FastAPI app factory、`/health`、`/v1/completions` 和 `/v1/chat/completions`。
- 使用 `EngineClient` 封装原本的阻塞式 `LLM.generate()`。
- 新增 `nanovllm/multimodal/` 包。
- 建立 `ImageInput`、`MultimodalRequest`、图像加载器、LRU cache 和模型无关的 multimodal processor 接口。
- 在 `pyproject.toml` 中增加 serving、multimodal 和 all 可选依赖。
- 更新 README，说明项目定位、安装和服务使用方式。

此阶段的性质：服务端可扩展骨架，多模态还没有真实模型后端。

### 4.2 `2fd2378 Refactor engine into core executor worker layers`

目标：将 API 生命周期、调度和 GPU 执行解耦。

主要修改：

- 新增 `EngineCore`，接管 `schedule -> execute -> postprocess` 主循环。
- 新增 `Executor`，接管 tensor-parallel 子进程、Event 和执行调用。
- 新增 `GPUWorker`，封装单 rank `ModelRunner`。
- `LLMEngine` 不再直接启动多进程、持有 `ModelRunner` 或调用 `ModelRunner.call()`。
- `LLMEngine.step()` 改为委托 `EngineCore.step()`。
- 进程退出、worker 通知和 `join()` 下沉到 Executor/GPUWorker 责任边界。
- 保留原来的 `LLM.generate()` 对外 API 和原调度语义。
- 新增 `docs/ARCHITECTURE.md`。

此阶段的性质：结构性重构，目标是为后续 serving、metrics、speculative decoding 和 multimodal worker 留出边界，不改变原生成结果。

### 4.3 `4af75a9 Add engine metrics and request timing stats`

目标：让调度和请求生命周期可观测。

主要修改：

- 新增 `nanovllm/engine/metrics.py`。
- 定义 `SchedulerStats`、`EngineStepStats` 和 `RequestTimingStats`。
- `Sequence` 增加 `arrival_time`、`running_time`、`first_token_time` 和 `finish_time`。
- 增加 `mark_running()`、`mark_finished()` 和 `timing_stats()`。
- 计算 queue time、TTFT、TPOT 和 total latency。
- `Scheduler.stats()` 暴露 waiting/running 数量、KV block 使用量和 prefix cache entry 数量。
- `EngineCore` 记录每次 prefill/decode step 的批量、耗时和吞吐。
- `LLMEngine` 增加 scheduler、step 和 request stats 查询接口。

此阶段的性质：完成无第三方指标库依赖的内部 metrics 基础，但还没有 Prometheus 完整导出。

### 4.4 `9ca4c19 Add Qwen2.5-VL multimodal backend`

目标：从多模态 scaffold 进入真实可加载的开源 VLM 路径。

主要修改：

- 选择 Qwen2.5-VL 作为首个开源多模态模型。
- 新增 `Qwen25VLConfig` 和 `Qwen25VLBackend`。
- 使用 `AutoProcessor.apply_chat_template()` 构造 Qwen chat prompt。
- 使用 `qwen_vl_utils.process_vision_info()` 处理视觉输入。
- 优先使用 `Qwen2_5_VLForConditionalGeneration`，在 transformers 版本不提供该类时回退到 `AutoModelForImageTextToText`。
- 将 torch、transformers、AutoProcessor 和 qwen-vl-utils 放在后端首次调用时延迟导入。
- `EngineClient` 增加独立的 multimodal backend 配置和锁。
- `/v1/chat/completions` 根据是否包含图像自动路由到文本引擎或 Qwen2.5-VL。
- 增加 `NANOINFER_MM_MODEL`、dtype、device map 和 attention implementation 环境变量。
- 新增多模态调度策略占位类和 benchmark 占位脚本。

此阶段的性质：真实 HuggingFace VLM bridge，但请求仍是单条串行调用，图像缓存和 batching 还没有进入 engine。

### 4.5 `7269f17 Add multimodal request batching and benchmark`

目标：完善 Qwen2.5-VL 请求格式，并将缓存、batching 和调度开始下沉到 engine。

主要修改：

- 新增 `openai_adapter.py`，将请求转换逻辑从 API route 中拆出。
- 支持 HTTP(S) URL、base64 data URL、file URL 和受控的服务器本地路径。
- 保留多张图像与文本 part 的原始交错顺序。
- 为 chat schema 增加判别联合、非空 messages、`top_p` 和 `max_completion_tokens` 兼容。
- 新增 `MultimodalGenerationRequest`，保存请求、采样参数、到达时间和估算预算。
- 新增 `MultimodalBatcher`，实现 FIFO 队列、兼容性检查和预算限制。
- 新增 `MultimodalManager`，管理 decoded image cache 和 image feature cache hook。
- 新增 `MultimodalEngineCore`，使用后台线程、Condition 和 Event 聚合并发 API 请求。
- Qwen2.5-VL backend 增加 `generate_chat_batch()`，一次处理多个 prompt 和展平后的图像输入。
- 增加 batch size、图像数量、文本 token 估算和图像 token 估算四类限制。
- 只合并 `max_new_tokens`、`temperature`、`top_p` 和图像 bucket 兼容的请求。
- 增加 decoded image LRU cache 的 hit、miss 和 eviction 统计。
- 新增 `/metrics/multimodal` JSON 指标接口。
- 将 `bench_multimodal.py` 从占位脚本扩展为并发 benchmark 工具。
- benchmark 支持本地图片自动转 base64、多图、并发度、warmup、timeout、P50/P90/P99、吞吐和 JSON 文件输出。
- 增加多模态 batch wait、cache capacity 和本地文件权限的环境变量。
- 使用 lazy package import 降低导入轻量子模块时对 transformers 的不必要依赖。

此阶段的性质：已有真实的请求级微批处理与 decoded image cache，但还没有拆分 HuggingFace 模型内部的 vision encoder。

## 5. 按模块划分的全部修改

| 模块 | 文件 | 修改或新增内容 |
| --- | --- | --- |
| 项目文档 | `docs/MODIFICATION_PLAN.md` | 定义分阶段改造路线、多模态优化目标和 benchmark 计划 |
| 项目文档 | `docs/ARCHITECTURE.md` | 记录文本和多模态执行路径与模块职责 |
| 项目文档 | `README.md` | 项目定位、安装、服务启动、API 和 benchmark 示例 |
| 打包 | `pyproject.toml` | 项目名、Python 版本、serving/multimodal/all 可选依赖和 GitHub 地址 |
| 文本引擎 | `nanovllm/engine/llm_engine.py` | 从直接持有 Scheduler/ModelRunner 改为委托 EngineCore，并暴露 metrics |
| 引擎分层 | `nanovllm/engine/engine_core.py` | 请求生命周期、调度、执行与 step metrics |
| 引擎分层 | `nanovllm/engine/executor.py` | tensor-parallel 进程、Event、rank 0 worker 和 shutdown |
| 引擎分层 | `nanovllm/engine/gpu_worker.py` | 单 GPU/rank ModelRunner 包装 |
| 指标 | `nanovllm/engine/metrics.py` | scheduler、step 和 request 指标 dataclass |
| 序列状态 | `nanovllm/engine/sequence.py` | 到达、进入 running、首 token 和完成时间 |
| 文本调度 | `nanovllm/engine/scheduler.py` | 状态转换调用和 KV/prefix cache 指标 |
| 多模态引擎 | `nanovllm/engine/multimodal_engine.py` | 后台 worker、请求聚合、batch 执行、完成通知与指标 |
| 多模态 schema | `nanovllm/multimodal/schemas.py` | 内部图像输入和图文请求对象 |
| 图像加载 | `nanovllm/multimodal/image_loader.py` | URL、base64、本地路径加载、RGB 转换和内容 hash |
| 缓存 | `nanovllm/multimodal/cache.py` | LRU cache、hit/miss/eviction 统计、decoded asset cache |
| 预处理 | `nanovllm/multimodal/processor.py` | vision encoder protocol 和 feature cache 复用骨架 |
| Qwen 后端 | `nanovllm/multimodal/qwen25_vl.py` | 延迟加载、chat template、vision info、单请求与 batch generate |
| API 适配 | `nanovllm/multimodal/openai_adapter.py` | OpenAI content parts 到 Qwen message 和内部 ImageInput 的转换 |
| 内部请求 | `nanovllm/multimodal/request.py` | 采样参数、请求 ID、到达时间和 token 估算 |
| 微批处理 | `nanovllm/multimodal/batching.py` | FIFO、兼容性分组和预算检查 |
| 多模态管理 | `nanovllm/multimodal/manager.py` | 图像物化、cache key、asset cache 和 feature cache hook |
| 调度策略 | `nanovllm/multimodal/scheduler_policy.py` | batch/image/text/image-token 预算与 wait window |
| 服务 schema | `nanovllm/serving/schemas.py` | completion/chat schema、图文判别联合、top-p 和 token 字段兼容 |
| 服务路由 | `nanovllm/serving/openai_api.py` | 文本/图文路由、错误映射和适配器调用 |
| 服务客户端 | `nanovllm/serving/engine_client.py` | 延迟初始化文本与 VLM 引擎、调度策略和 shutdown |
| 服务工厂 | `nanovllm/serving/server.py` | FastAPI app、health、multimodal metrics 和 shutdown hook |
| 启动配置 | `nanovllm/serving/entrypoint.py` | 文本与多模态环境变量 |
| Benchmark | `bench_multimodal.py` | 并发压测、百分位延迟、吞吐、错误和服务端指标采集 |
| 包导入 | `nanovllm/__init__.py` | 使用 `__getattr__` 延迟导入 LLM 重依赖 |

## 6. 遇到的问题与解决方案

### 6.1 项目结构与兼容性

| 问题 | 根因 | 解决方案 | 状态 |
| --- | --- | --- | --- |
| 原项目不适合直接进行大幅度求职化改造 | 学习注释、原始实现和新功能会混在同一工作区 | 复制为 `nano-vllm-job-project`，建立独立 GitHub 仓库 | 已解决 |
| `LLMEngine` 同时管理 API、调度、GPU worker 和多进程 | 原项目追求极简可读性，没有 serving 扩展边界 | 拆分 `EngineCore / Executor / GPUWorker`，让 `LLMEngine` 只负责用户 API 和 tokenizer | 已解决 |
| 重构可能破坏原来 `LLM.generate()` | 调用路径和对象所有权发生了变化 | 保留 `add_request/step/is_finished/generate` 签名，内部改为委托 EngineCore | 代码兼容，待 GPU 回归验证 |
| Scheduler 初始化需要 `num_kvcache_blocks` | 该数值由 ModelRunner 根据显存计算并写回 Config | EngineCore 先初始化 Executor/GPUWorker，再初始化 Scheduler | 已解决 |
| tensor-parallel 子进程可能成为僵尸进程 | 只通知退出而不 `join()` 会留下未回收的进程资源 | Executor 保存 process 引用，shutdown 时通知 worker 并逐个 `join()` | 已解决 |
| 轻量子模块导入也会立即导入 transformers | `nanovllm.__init__` 原本顶层导入 `LLM` | 使用模块级 `__getattr__` 延迟导入，保留 `from nanovllm import LLM` API | 已解决 |

### 6.2 Metrics 与时间统计

| 问题 | 根因 | 解决方案 | 状态 |
| --- | --- | --- | --- |
| 无法计算请求排队、TTFT、TPOT 和总延迟 | Sequence 只保存 token 与 KV 状态 | 在请求创建、进入 running、首次 append token 和 finish 位置记录时间 | 已解决 |
| 系统时间调整可以导致延迟计算异常 | `time.time()` 可被 NTP 或手动时钟调整影响 | 请求生命周期使用 `monotonic()`，step 性能使用 `perf_counter()` | 已解决 |
| chunked prefill 可能多次经过 waiting | 如果每次调度都覆盖 running time，queue time 会失真 | `mark_running()` 只在 `running_time is None` 时记录 | 已解决 |
| 没有可供 benchmark 使用的 KV cache 状态 | BlockManager 数据没有向上层暴露 | Scheduler 组装 total/used/free blocks 和 prefix entries 快照 | 已解决 |
| 目前 API usage 计数仍为 0 | 文本和 VLM 服务还没有统一 tokenizer usage 统计 | 保留 OpenAI usage schema，等流式和异步请求管理阶段补齐 | 未解决 |

### 6.3 Qwen2.5-VL 后端

| 问题 | 根因 | 解决方案 | 状态 |
| --- | --- | --- | --- |
| 如何在不重写 VLM runtime 的情况下快速接入真实开源图文模型 | nano-vLLM 只有纯文本 Qwen3 runtime | 首阶段使用 HuggingFace Qwen2.5-VL bridge，后续再逐步下沉 vision execution | 已建立可用路径，待 GPU 验证 |
| transformers 版本对 Qwen2.5-VL 类名支持不同 | 不同版本可能没有专用类 | 优先取 `Qwen2_5_VLForConditionalGeneration`，否则回退到 `AutoModelForImageTextToText` | 已解决 |
| 纯文本用户不应被强制安装 qwen-vl-utils 和 torchvision | 多模态依赖较重且不是文本引擎必需 | 将多模态依赖放入 optional extras，并在首次调用 VLM 时延迟导入 | 已解决 |
| 单请求 `generate()` 不能利用并发 API 流量 | 原 Qwen backend 每次只构造一个 prompt | 增加 `generate_chat_batch()`，批量构造 prompt、processor input 和 decode output | 代码已完成，待真模型验证 |
| 多图 batch 中图像如何与 prompt 对齐 | Processor 通过每个 prompt 中的视觉占位符和展平图像列表对齐 | 保留消息中的图文顺序，按请求顺序展平 PIL image 列表 | 代码已完成，待 Qwen processor 集成验证 |
| HuggingFace `generate()` 封装了 vision encoder 和 language model | 无法在不拆模型 forward 的情况下直接注入缓存后的 vision embedding | 先启用 decoded image cache，同时在 Manager 中保留 feature cache API；后续拆 vision encoder | 部分解决 |

### 6.4 请求格式与输入安全

| 问题 | 根因 | 解决方案 | 状态 |
| --- | --- | --- | --- |
| OpenAI content parts 与 Qwen message 格式不同 | OpenAI 使用 `image_url`，Qwen 使用 `type=image` 和 `image` 字段 | 建立独立 adapter，不再在 FastAPI route 内手写转换 | 已解决 |
| 需要支持 URL、base64 和本地文件 | 不同来源的加载、缓存 key 和安全特征不同 | Adapter 识别 source type，ImageLoader 统一转换为 RGB PIL image | 已解决 |
| 无效 base64 会在 worker 内部才报错 | 最初只在加载时调用 `b64decode` | Adapter 在请求入口检查 `;base64,` 并用 `validate=True` 验证 | 已解决 |
| 开放服务器本地路径可导致文件访问风险 | 远程请求可以构造 `file://` 或绝对路径 | 默认禁止本地文件，只有设置 `NANOINFER_MM_ALLOW_LOCAL_FILES=1` 才允许 | 已解决 |
| 多图与文本交错顺序容易丢失 | 如果先提取所有文本再提取图像，会改变语义 | Adapter 按 content part 原始顺序生成 Qwen content | 已解决 |
| `temperature=0` 与原文本 sampler 不兼容 | 文本 sampler 直接用 temperature 做除法，原 SamplingParams 禁止 greedy | 共用 chat schema 仍限制 `temperature > 0`；Qwen backend 内部保留接近 0 时 greedy 的逻辑 | 当前一致，后续可独立扩展文本 greedy |

### 6.5 Batching 与调度

| 问题 | 根因 | 解决方案 | 状态 |
| --- | --- | --- | --- |
| FastAPI 并发请求最初被一把 multimodal lock 串行化 | HuggingFace model 不能被多线程同时调用 | 将直接锁改为单后台 worker，API 线程通过 Event 等待批量结果 | 已解决 |
| 请求到达时刻不同，难以自然形成 batch | 立即执行每条请求会浪费 GPU batch 能力 | 增加可配置的 `batch_wait_ms` 微批窗口 | 已解决 |
| 不同采样参数的请求不能共用一次 HF generate | HuggingFace generate kwargs 在当前实现中是 batch 级标量 | 只合并 `max_new_tokens/temperature/top_p` 完全兼容的请求 | 已解决 |
| 图像数量和详细度差异会造成 batch 形状不稳定 | 视觉 token 数量通常远大于普通文本 token | 增加 image count/detail bucket，并独立统计 image token 估算预算 | 已建立基础策略 |
| 单条请求可能自身超过 batch 预算 | 如果无条件放入队列，调度器可能违反配置 | admission 时检查图像数、文本估算和图像 token 估算，超限直接拒绝 | 已解决 |
| 外部 request ID 可能重复 | 调用者可以复用同一个 `MultimodalRequest` | 为每次 generation 额外生成内部 UUID，pending map 使用内部 ID | 已解决 |
| 多个首请求可能同时初始化两个 VLM core | `EngineClient.multimodal_core` 最初没有初始化锁 | 增加 double-check 和 `_multimodal_init_lock` | 已解决 |
| shutdown 时可能仍有队列中请求 | 立即退出会让 API 线程永久等待 | worker 收到 shutdown 后继续处理已入队请求，队列清空后退出并 join | 已解决 |

### 6.6 图像缓存

| 问题 | 根因 | 解决方案 | 状态 |
| --- | --- | --- | --- |
| 重复图像每次都需要下载/解码 | 原 HF bridge 将图像加载完全交给 `process_vision_info` | Manager 先转换为 PIL image 并放入 decoded asset LRU cache，后端直接接收 image object | 已解决 |
| 查 cache 需要在真正加载图像前获得 key | 如果先加载再 hash，URL 下载开销已经发生 | base64 使用内容 SHA256；path 使用规范路径、mtime 和 size；URL 使用 URL SHA256 | 已解决 |
| URL 内容可能在同一地址下更新 | 当前 URL key 只取决于 URL 文本 | 当前作为性能优先的简化策略；后续需加 TTL、ETag 或 content hash 刷新 | 已知限制 |
| cache 容量无限增长会占用大量 CPU 内存 | PIL image 对象可能很大 | 使用容量可配置的 LRU，记录 eviction | 已解决，后续可改为按字节计费 |
| feature cache 命名容易让人误以为已实际复用 vision embedding | 当前 HF generate 不接受这个抽象的 feature 输入 | 分开 `ImageAssetCache` 和 `ImageFeatureCache`，文档明确前者已启用、后者仅为 hook | 已澄清 |

### 6.7 Benchmark 与验证

| 问题 | 根因 | 解决方案 | 状态 |
| --- | --- | --- | --- |
| benchmark 最初只有一行占位输出 | 前一阶段只先确定了指标方向 | 实现完整 CLI、ThreadPoolExecutor 并发、latency percentile、throughput 和 JSON report | 已解决 |
| 不希望为 benchmark 客户端额外强制安装 httpx/requests | 服务器环境可能只有基础 Python | 使用标准库 `urllib.request` 和 `ThreadPoolExecutor` | 已解决 |
| 用户可能传基础 server URL 或完整 endpoint | CLI 参数语义容易不一致 | 自动归一化为 `/v1/chat/completions` | 已解决 |
| 本地图片路径不适合直接发给远程服务器 | 服务器无法访问 benchmark 机器的文件系统 | benchmark 自动读取本地图片并编码为 data URL | 已解决 |
| 只有客户端延迟无法确认是否真的 batching/cache | 客户端无法观察 engine 内部状态 | benchmark 完成后请求 `/metrics/multimodal`，记录平均 batch size 和 cache counters | 已解决 |
| 无流式响应时无法准确测 TTFT/TPOT | 客户端只在整个 JSON 响应完成后收到结果 | report 中将 TTFT/TPOT 显式设为 `null`，不伪造数据；等 SSE 后补齐 | 未解决，已正确标记 |

### 6.8 开发环境与仓库同步

| 问题 | 根因 | 解决方案 | 状态 |
| --- | --- | --- | --- |
| 本机没有 `transformers` | 当前本机不是服务器 GPU 运行环境 | 不在本机强行下载模型；使用 compileall 和 fake backend 进行静态/逻辑验证 | 服务器待验证 |
| 本机没有 `pydantic` | 没有安装 serving extras | schema/API 运行测试留到服务器，本地使用 compileall 检查语法 | 服务器待验证 |
| 无 GPU 无法加载 Qwen2.5-VL | 模型显存和 CUDA 环境不具备 | 用 fake backend 验证 queue、batching、Event 完成通知和 image cache | 服务器待验证 |
| 第一次 `git push` 无法解析 github.com | 当前 sandbox 默认网络受限 | 使用获授权的网络权限重试 `git push` | 已解决 |

## 7. 已完成的验证

### 7.1 静态检查

```bash
python -m compileall nanovllm bench_multimodal.py
git diff --check
python bench_multimodal.py --help
```

结果：通过。

### 7.2 无 GPU batching smoke test

测试方法：

- 使用 fake multimodal backend。
- 并发提交 4 条兼容请求。
- batch wait window 设为 20 ms。
- 验证后端只执行 1 个 batch，batch size 为 4。
- 验证所有 Event 均收到结果并正常退出。

结果：通过。

### 7.3 无 GPU image cache smoke test

测试方法：

- 构造两条使用同一 base64 PNG 的并发请求。
- 使用 fake backend 检查每条请求获得一张 PIL image。
- 验证 cache miss 为 1、cache hit 为 1。

结果：通过。

## 8. 尚未完成的服务器验证

下列项目需要在有 CUDA 和 Qwen2.5-VL 权重的服务器上完成：

- 安装 `pip install -e ".[all]"`。
- 启动纯文本 Qwen3 服务并做回归测试。
- 加载 Qwen2.5-VL 单图请求。
- 验证 HTTP URL、base64、多图和图文交错请求。
- 验证 Qwen processor 对 batch 内展平 PIL image 列表的对齐行为。
- 比较 batch size 1/2/4/8 的显存和吞吐。
- 验证不同图像数量、分辨率和 detail bucket。
- 运行 `bench_multimodal.py` 并保存 JSON 结果。
- 核对 `/metrics/multimodal` 中的 average batch size 和 cache hit rate。
- 测试 OOM、无效图片、URL 超时、模型加载失败和服务 shutdown。

## 9. 已知限制与后续优先级

### P0：服务器跑通与正确性

- 完成 Qwen2.5-VL 真实 batch 集成测试。
- 修复服务器上的 processor、device map、padding 或 dtype 兼容问题。
- 增加 API 集成测试和基本错误用例。

### P1：多模态核心优化

- 将图像下载和 decode 从唯一 VLM worker 中拆到异步预处理 worker pool。
- 拆分 Qwen2.5-VL vision encoder 和 language model 阶段。
- 真正缓存 image feature/vision embedding。
- 使用实际 processor 结果替换当前的字符数/4 文本 token 估算。
- 基于实际 patch grid 或 image token count 做分辨率 bucket。

### P2：服务完整性

- 增加 SSE streaming。
- 记录真实 TTFT 和 TPOT。
- 填充 OpenAI usage token 计数。
- 增加请求取消、timeout 后队列移除和 client disconnect 处理。
- 增加 Prometheus 指标、Dockerfile、CI 和压力测报告。

### P3：更深的推理优化

- 多模态 prefix/KV cache 复用。
- 视觉 token 压缩。
- mixed text/image token-level continuous batching。
- speculative decoding。
- URL cache TTL、ETag 和按字节容量计费。

## 10. 当前项目完成度结论

当前项目可以定义为：

> 已完成架构和核心功能骨架的 LLM/VLM serving MVP，已具备真实 Qwen2.5-VL 后端、请求级 batching、decoded image cache、调度预算和 benchmark 工具，但仍需 GPU 集成验证和 vision encoder 级优化。

完成度可以估计为整体规划的 55% 至 65%。

目前可以放入简历，但应将贡献准确描述为“架构重构、OpenAI 服务、Qwen2.5-VL 接入、请求级 batching、图像对象缓存和 benchmark”，不应在没有服务器实验数据时声称已实现 vision feature 复用或获得某个性能提升百分比。

## 11. 相关提交

```text
69097c8 Initial NanoInfer serving and multimodal scaffold
2fd2378 Refactor engine into core executor worker layers
4af75a9 Add engine metrics and request timing stats
9ca4c19 Add Qwen2.5-VL multimodal backend
7269f17 Add multimodal request batching and benchmark
```

仓库地址：

```text
https://github.com/Biankaer/nano-vllm-Multimodal
```
