# NanoInfer

NanoInfer 是基于 `nano-vllm` 扩展的轻量级 LLM/VLM 推理与服务框架。项目保留
原有 `nanovllm.LLM` 离线接口，并围绕原生多模态、Continuous Batching、Paged KV
Cache、CUDA 自定义算子、OpenAI-compatible Serving 和可复现性能评测进行完整化。

## 当前进度

已实现：

- Qwen2.5-VL 与 Qwen3-VL 原生静态图片推理，支持纯文本、单图、多图和重复图片。
- Qwen3-VL 视觉塔、交错 M-RoPE、DeepStack 注入与原生语言 Decoder。
- Qwen2.5-VL/Qwen3-VL Adapter 自动选择，保留 Qwen2.5-VL 的 Hugging Face 兼容路径。
- Dynamic Continuous Batching，支持 Decode 期间动态接入请求、取消请求和回收资源。
- Paged KV Cache、Chunked Prefill、Prefix Cache 和图片感知的前缀缓存身份标识。
- GPU 视觉特征 Bundle Cache，原子管理 final embedding 与 DeepStack embedding。
- PyTorch、Triton 和 C++/CUDA 三类 Paged KV Store/Gather 后端。
- `temperature=0` 真正 Greedy Decoding，并支持同一 Batch 内 Greedy/随机请求混合。
- `/v1/completions` 与 `/v1/chat/completions`，支持 Streaming SSE 及服务指标采集。
- Hugging Face 组件级、Logits 与 Token 级对齐，以及对 vLLM 的固定负载测试。

当前边界：

- Qwen3-VL 生产路径仅支持静态图片和单卡，`tensor_parallel_size` 必须为 `1`。
- 视频输入、Qwen3-VL 生产级 Hugging Face 兼容路径和原生 top-p 采样尚未实现。
- ROCm/HIP 尚无可用工具链，不宣称已完成移植验证。

## 架构概览

```text
OpenAI / Offline Request
        ↓
Prompt Processor + Decoded-image LRU
        ↓
Sequence + Scheduler + Dynamic Continuous Batching
        ↓
VisionFeatureProvider → GPU Vision Bundle Cache
        ↓
Qwen2.5-VL / Qwen3-VL Native Prefill
        ↓
Chunked Prefill + Prefix Cache + Paged KV Cache
        ↓
CUDA Graph Decode + Sampler
        ↓
Offline Result / OpenAI Streaming SSE
```

Qwen-VL Adapter 根据模型权重中的 `model_type` 自动选择模型、Processor、视觉编码器、
M-RoPE 策略和权重前缀。Qwen2.5-VL 继续使用分段 M-RoPE；Qwen3-VL 使用交错
T/H/W 频率，并在语言层 `0/1/2` 后注入三组 DeepStack 视觉特征。

## 安装

基础环境：

```bash
python -m pip install -e .
```

Serving 依赖：

```bash
python -m pip install -e '.[serving]'
```

完整多模态环境：

```bash
python -m pip install -e '.[all]'
```

Qwen3-VL 要求 `transformers>=4.57.6,<5`。建议使用独立环境，避免升级 Processor 影响
其他项目：

```bash
conda create -n nanoinfer-qwen3vl --clone llm
conda activate nanoinfer-qwen3vl
python -m pip install -U 'transformers==4.57.6' 'tokenizers>=0.22,<0.24'
python -m pip install -e '.[all]'
python -c "import transformers; print(transformers.__version__)"
```

## 模型准备

在已配置 Hugging Face 代理的终端下载模型权重：

```bash
huggingface-cli download \
  Qwen/Qwen3-VL-2B-Instruct \
  --local-dir /path/to/Qwen3-VL-2B-Instruct
```

当前原生 Qwen3-VL 路径以 `Qwen/Qwen3-VL-2B-Instruct` BF16 Dense Instruct 为主要验证
模型。加载时会根据 `config.json` 自动解析模型类型，无需额外指定模型类别。

## 离线推理

纯文本推理保留原有 `LLM.generate()` 接口：

```python
from nanovllm import LLM, SamplingParams

llm = LLM("/path/to/Qwen3-0.6B", enforce_eager=True, tensor_parallel_size=1)
outputs = llm.generate(
    ["请用一段话解释 KV Cache。"],
    SamplingParams(temperature=0.6, max_tokens=128),
)
print(outputs[0]["text"])
llm.exit()
```

Qwen3-VL 的纯文本、单图、多图和重复图片共用 `LLM.generate_multimodal()`：

```python
from PIL import Image
from nanovllm import LLM, SamplingParams

model = "/path/to/Qwen3-VL-2B-Instruct"
llm = LLM(
    model,
    tensor_parallel_size=1,
    vision_attn_implementation="flash_attention_2",  # 排障时可改为 sdpa
    language_attn_implementation="flash_attention_2",
)
image = Image.open("assets/logo.png").convert("RGB")
messages = [{
    "role": "user",
    "content": [
        {"type": "image"},
        {"type": "text", "text": "请描述这张图片。"},
    ],
}]
outputs = llm.generate_multimodal(
    [messages], [[image]], [["auto"]],
    SamplingParams(temperature=0, max_tokens=64),
)
print(outputs[0]["text"])
llm.exit()
```

## OpenAI-compatible Serving

启动 Qwen2.5-VL 原生服务：

```bash
NANOINFER_MODEL=/path/to/Qwen2.5-VL-3B-Instruct \
NANOINFER_ENFORCE_EAGER=0 \
NANOINFER_GPU_MEMORY_UTILIZATION=0.5 \
NANOINFER_MM_BACKEND=native \
NANOINFER_MM_FEATURE_CACHE_BYTES=2147483648 \
NANOINFER_MM_MAX_BATCH_SIZE=8 \
NANOINFER_MM_BATCH_QUIET_MS=3 \
NANOINFER_MM_BATCH_MAX_WAIT_MS=20 \
uvicorn nanovllm.serving.entrypoint:app --host 0.0.0.0 --port 8000
```

启动 Qwen3-VL 原生服务：

```bash
NANOINFER_MODEL=/path/to/Qwen3-VL-2B-Instruct \
NANOINFER_MM_MODEL=/path/to/Qwen3-VL-2B-Instruct \
NANOINFER_MM_BACKEND=native \
NANOINFER_MM_ATTN_IMPL=flash_attention_2 \
NANOINFER_MM_FEATURE_CACHE_BYTES=2147483648 \
uvicorn nanovllm.serving.entrypoint:app --host 0.0.0.0 --port 8000
```

`NANOINFER_MM_ATTN_IMPL=sdpa` 选择视觉数值对照路径。视觉 `auto` 在 Qwen3-VL 上解析为
FlashAttention 2，在 Qwen2.5-VL 上解析为 SDPA。`NANOINFER_MM_BATCH_WAIT_MS` 保留为
旧版固定等待窗口；默认调度使用 `quiet window + max wait`。

流式图文对话请求：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-vl-2b-instruct",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,REPLACE_ME"}},
        {"type": "text", "text": "请用中文描述这张图片。"}
      ]
    }],
    "max_tokens": 64,
    "temperature": 0,
    "stream": true
  }'
```

`image_url.url` 支持 HTTP(S) 和 base64 data URL。可信部署可设置
`NANOINFER_MM_ALLOW_LOCAL_FILES=1` 开启 `file://` 和服务器本地路径；默认禁用本地文件
访问。`/metrics/multimodal` 返回 Batch Size Histogram、排队/合批/执行时间和视觉缓存指标。

## 原生多模态运行时

原生路径自行管理从图片到 Decode 的完整数据流：

```text
OpenAI Request → Decoded-image LRU → Qwen Processor
  → input_ids + 3-axis M-RoPE + CPU image patches
  → MultimodalBatcher + Scheduler
  → lazy VisionFeatureProvider
      → byte-budgeted GPU VisionFeatureBundle LRU
  → Qwen2.5-VL / Qwen3-VL native prefill
      → Qwen3-VL DeepStack injection at layers 0/1/2
  → image-aware BlockManager + paged KV cache
  → native decode + sampler
```

GPU Feature Cache 将 final embedding 和三组 DeepStack embedding 作为一个原子 Bundle 进行
插入、pin、命中和淘汰。当 Prefix KV 已完整覆盖视觉 span 时，`ModelRunner` 不再
解析图片特征；当文本前缀变化但图片相同时，GPU Feature Cache 避免重复运行视觉塔。

Dynamic Continuous Batching 允许 Decode 期间接入新请求，通过请求级 Token/Event 路由进行
流式返回。客户端断开后会取消对应 Sequence，释放 KV Block、图片输入和视觉特征引用。

## Paged KV Store/Gather 后端

NanoInfer 通过 `paged_kv_backend` 提供 `pytorch / triton / cuda / auto` 四种选择：

- `pytorch`：使用 `index_copy_`/`index_select` 的易读对照实现。
- `triton`：使用专用 Store 和 Gather Kernel。
- `cuda`：惰性编译 PyTorch C++ Extension，并通过 `torch.library` 注册自定义算子。
- `auto`：优先选择 C++/CUDA，其次 Triton，最后回退到 PyTorch。

C++/CUDA 后端实际使用 CUDA Runtime API 进行显存分配、非阻塞 Transfer Stream、Event、
Pinned Host `cudaMemcpyAsync`、Current Stream Wait 和 `cudaLaunchKernel`。对齐的载荷使用
16-byte 传输。显式选择 `cuda` 但编译不可用时会返回完整错误上下文，不会静默回退。

```python
llm = LLM(
    "/path/to/Qwen3-VL-2B-Instruct",
    paged_kv_backend="cuda",
)
```

对不可信 Block Table 或 Slot Mapping 排障时，可显式开启会引入同步的 GPU 值校验：

```bash
NANOVLLM_PAGED_KV_VALIDATE_INDICES=1 python your_inference.py
```

执行三后端 Store/Gather 微基准测试：

```bash
TORCH_CUDA_ARCH_LIST=8.0 python bench_paged_kv.py \
  --backends pytorch,triton,cuda \
  --operations store,gather \
  --dtypes float16,bfloat16 \
  --tokens 1,8,257 \
  --kv-shapes 1x64,2x128,8x128 \
  --warmup 100 --iterations 1000 \
  --output-dir benchmark-results/paged-kv/a100-full
```

报告保留原始 CUDA Event 样本、有效带宽与 P50/P95/P99。解读端到端结果时需区分语言
Attention 路径：**FlashAttention：仅写入**，因为 Flash 直接读取 Paged Cache；
**SDPA：写入 + 聚合**，因为 SDPA 需先实例化连续 K/V。

Nsight Compute 采集：

```bash
python scripts/profile_paged_kv_ncu.py \
  --ncu /path/to/ncu \
  --case-selector \
  'backend=cuda;operation=store;dtype=bfloat16;tokens=257;blocks=64;block=16;heads=8;dim=128;mapping=random_with_skips;seed=17'
```

采集器请求 SpeedOfLight、MemoryWorkloadAnalysis、Occupancy、WarpStateStats 和 LaunchStats，
并仅保留 Paged KV Kernel。当前 A100 主机因驱动返回 `ERR_NVGPUCTRPERM` 无法读取性能
计数器，报告将缺失指标保留为 `null`，不进行推测。当前 ROCm 状态为
`not_tested_no_rocm_toolchain`。

## Qwen3-VL 正确性与性能验证

`verify_qwen3_vl_native.py` 会确定性生成 10 类用例：中英文纯文本、方形/横向/竖向图片、
两张不同图片、重复图片、中文图片问答、长 Prompt Chunked Prefill 和缓存复用。脚本不需要
额外测试数据集，HF 与 NanoInfer 分进程运行，避免两个完整模型同时占用 GPU。

```bash
MODEL=/path/to/Qwen3-VL-2B-Instruct
RESULTS=verification-results/qwen3-vl

python verify_qwen3_vl_native.py --stage hf --model "$MODEL" --results-dir "$RESULTS"
python verify_qwen3_vl_native.py --stage hf-flash --model "$MODEL" --results-dir "$RESULTS"
python verify_qwen3_vl_native.py --stage native-sdpa \
  --model "$MODEL" --results-dir "$RESULTS" --benchmark
python verify_qwen3_vl_native.py --stage native-flash \
  --model "$MODEL" --results-dir "$RESULTS" --benchmark
python verify_qwen3_vl_native.py --stage native-consistency \
  --model "$MODEL" --results-dir "$RESULTS"
python verify_qwen3_vl_native.py --stage compare --results-dir "$RESULTS"
```

SDPA 路径校验 Processor 元数据、Vision final/DeepStack、首个 Token 的 Logits 与完整 Greedy
Token。组件浮点判定标准为 `rtol=2e-2`、`atol=2e-2`、cosine `>=0.9999`，整数元数据和
argmax 必须完全一致。Flash 路径分别与 HF Flash 对齐，避免将 BF16 下 Flash/SDPA 本身的
数值差异误判为原生实现错误。

组件级对齐：

```bash
python verify_qwen3_vl_native.py --stage components-hf \
  --model "$MODEL" --results-dir "$RESULTS"
python verify_qwen3_vl_native.py --stage components-native \
  --model "$MODEL" --results-dir "$RESULTS"
python verify_qwen3_vl_native.py --stage compare-components \
  --component-hf "$RESULTS/components-hf.pt" \
  --component-native "$RESULTS/components-native.pt" \
  --results-dir "$RESULTS"
```

48 组性能矩阵覆盖 Vision SDPA/Flash、Batch `1/4/8`、单图/双图、冷/热缓存和
完整/Chunked Prefill：

```bash
python verify_qwen3_vl_native.py --stage benchmark-matrix \
  --model "$MODEL" --results-dir "$RESULTS"
```

Qwen2.5-VL 回归测试：

```bash
python verify_qwen25_vl_native.py \
  --model /path/to/Qwen2.5-VL-3B-Instruct \
  --image assets/logo.png \
  --mode all
```

CPU/单元测试：

```bash
python -m pytest -q
python -m compileall -q nanovllm tests
```

## Serving 性能对比

对比环境：A100 80GB、Qwen2.5-VL-3B-Instruct、BF16、CUDA Graph、64 个输出 Token、
`temperature=0`、`ignore_eos=true`。NanoInfer 视觉 Attention 使用 SDPA，vLLM 0.7.3 使用
XFormers。下表是固定测试负载下的服务性能比较，不表示所有场景下的普遍优势。

| 测试场景 | NanoInfer output tok/s | vLLM output tok/s | 吞吐变化 |
|---|---:|---:|---:|
| 单图 `C=1` | 108.1 | 105.2 | +2.8% |
| 单图 `C=4` | 387.9 | 345.4 | +12.3% |
| 多图 `C=4` | 378.4 | 301.9 | +25.3% |
| 单图 `C=8` | 684.7 | 556.2 | +23.1% |

`C=8` 三次 NanoInfer 结果为 `679.8 / 684.7 / 689.4 output tok/s`；TTFT P99 平均为
`116.2 ms`，vLLM 对照值为 `354.9 ms`；E2E P50 由 vLLM 的 `912.8 ms` 降至 `738.5 ms`。
Serving 默认显存预算调整后，峰值显存从旧 NanoInfer 约 `70.5 GiB` 降至 `37.8 GiB`，
降幅约 `46%`；vLLM 对应测试约为 `41.9 GiB`。

服务基准同时记录 TTFT、TPOT、ITL、E2E、吞吐、Batch Histogram、KV Cache 和视觉缓存指标。
对比矩阵尚未完成全部用例的完整 Token 级一致性校验，因此这些数据仅作为固定输入规模和
生成参数下的性能证据。

## 致谢

本项目基于 Xingkai Yu 的 `nano-vllm` 实现扩展，用于学习与验证大模型推理、
多模态 Serving 与 CUDA 性能工程。
