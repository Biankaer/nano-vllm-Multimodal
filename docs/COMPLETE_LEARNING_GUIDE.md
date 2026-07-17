# NanoInfer 完整学习指南：从 nano-vLLM 到 Native Qwen2.5-VL Runtime

## 0. 文档定位

这份文档将本地和服务器两个开发阶段连成一条完整学习路线：

```text
原始 nano-vLLM
  -> OpenAI 服务层
  -> EngineCore / Executor / GPUWorker
  -> metrics
  -> Hugging Face Qwen2.5-VL bridge
  -> 请求级 batching
  -> decoded-image cache
  -> Native Qwen2.5-VL decoder
  -> 三轴 MRoPE
  -> Vision Feature GPU LRU
  -> image-aware prefix cache
  -> paged KV cache
  -> chunked prefill
  -> CUDA Graph decode
  -> A100 数值对齐与压测
```

它不是两份旧文档的简单拼接。重复的基础概念只讲一次，重点是理解：

```text
为什么项目要从 HF bridge 走向 native runtime？
一张图如何进入 NanoInfer 的 Scheduler 和 paged KV cache？
三种缓存分别省掉什么？
为什么 Qwen2.5-VL 不能直接使用 Qwen3 decoder？
怎样证明 native runtime 是数值正确的？
```

### 建议阅读路线

```text
第 1 章：项目演进
第 2 章：文本引擎基础
第 5 章：为什么 HF bridge 不是 native runtime
第 6 章：Native 请求数据流
第 8 章：三种缓存
第 11 章：Scheduler 与多模态
第 15 章：正确性验证
```

## 1. 项目演进：每一步在解决什么

### 1.1 原始 nano-vLLM

```text
LLM.generate()
  -> Sequence
  -> Scheduler
  -> ModelRunner
  -> Qwen3ForCausalLM
```

已有 Qwen3 文本模型、prefill/decode、KV cache blocks、prefix cache、chunked prefill、tensor parallel 和 CUDA Graph decode。

局限是主要面向离线生成，而且 `LLMEngine` 同时管理请求、调度、GPU 执行和多进程。

### 1.2 服务化与分层

新增：

```text
FastAPI
OpenAI-compatible schema
EngineClient
EngineCore
Executor
GPUWorker
request / scheduler / KV metrics
```

架构变为：

```text
HTTP / Offline API
  -> EngineClient / LLMEngine
  -> EngineCore
      -> Scheduler
      -> Executor
          -> GPUWorker
              -> ModelRunner
```

这一步首先解决所有权：谁管请求、谁管 worker、谁管 CUDA，以及谁负责 shutdown 和 join。

### 1.3 Hugging Face Qwen2.5-VL Bridge

```text
OpenAI image-text request
  -> request adapter
  -> decoded-image cache
  -> request-level batcher
  -> HuggingFace Qwen2.5-VL generate()
```

这一阶段打通了真实图文 API、多图输入、请求级微批处理和 decoded-image LRU。

但 NanoInfer 没有执行 Qwen2.5-VL 的 decoder、KV cache 和 sampler。

### 1.4 Native Qwen2.5-VL Runtime

服务器开发将路径改为：

```text
OpenAI image-text request
  -> Qwen25VLPromptProcessor
  -> token IDs + MRoPE positions + visual metadata
  -> Sequence
  -> Scheduler
  -> image-aware BlockManager
  -> ModelRunner
      -> VisionFeatureProvider
      -> mixed text/vision embeddings
      -> Native Qwen2.5 decoder
      -> paged KV cache
      -> sampler / CUDA Graph
```

此时图文请求才真正进入 NanoInfer runtime。

### 本章先记住

```text
能调用 HF generate 不等于接入了自研 runtime。
只有 Scheduler、KV cache、decoder 和 sampler 真正看到 visual tokens，
才能说多模态请求进入了 Native runtime。
```

## 2. 文本推理引擎基础

### 2.1 Token 与 Embedding

```text
字符串
  -> tokenizer
  -> token IDs
  -> embedding table
  -> hidden vectors
  -> Transformer
  -> logits
  -> sampler
```

例如 `Describe this image.` 可能被转成 `[9707, 419, 2168, 13]`。Embedding table 再将每个整数 ID 转成 `hidden_size` 维向量。

### 2.2 Prefill

Prefill 处理 prompt 的全部或一个 chunk：

```text
prompt tokens
  -> Transformer forward
  -> hidden states
  -> 每层 K/V 写入 KV cache
  -> 采样第一个 completion token
```

### 2.3 Decode

```text
last token + historical KV cache
  -> next-token logits
  -> sampler
  -> append token
```

历史 token 的 K/V 已经在缓存中，不需要重新计算整段 prompt。

### 2.4 Paged KV Cache

Paged KV cache 将显存拆成固定 block：

```text
Sequence A -> physical blocks [3, 9, 12]
Sequence B -> physical blocks [1, 7]
```

`Sequence.block_table` 记录逻辑 token 块到物理 KV 块的映射。

### 2.5 Sequence 与 Scheduler

Sequence 状态：

```text
WAITING  -> 等待 prefill
RUNNING  -> 正在 decode
FINISHED -> 遇到 EOS 或 max_tokens
```

Scheduler 每轮决定：

```text
这轮是 prefill 还是 decode
选哪些 Sequence
每条序列计算多少 token
KV blocks 是否足够
哪些 prefix 可复用
是否需要抢占
```

Scheduler 不需要理解 PIL 图像。它只需要看到图像被物化成的 token 数量和 cache identity。

## 3. 服务化与引擎分层

### 3.1 每一层的责任

| 组件 | 责任 |
| --- | --- |
| LLMEngine | 用户 API、tokenizer、Sequence 创建、输出 decode |
| EngineCore | request lifecycle、schedule -> execute -> postprocess |
| Executor | worker 和 tensor-parallel 进程所有权 |
| GPUWorker | 稳定的 run/shutdown 边界 |
| ModelRunner | CUDA、模型、KV cache tensor、CUDA Graph |

### 3.2 EngineCore.step()

```python
seqs, is_prefill = scheduler.schedule()
token_ids = executor.run(seqs, is_prefill)
scheduler.postprocess(seqs, token_ids, is_prefill)
```

即：选择请求，执行模型，然后更新 Sequence、KV 和结束状态。

### 3.3 初始化顺序

ModelRunner 先根据实际 GPU 显存计算 `config.num_kvcache_blocks`，Scheduler 之后才能创建 BlockManager。

```text
Executor / ModelRunner
  -> 计算 KV 容量
  -> Scheduler / BlockManager
```

### 3.4 shutdown 与 join

```text
shutdown：通知 worker 退出
join：等待子进程真正结束并回收进程状态
```

谁创建 process/worker/cache，谁就负责释放它。

## 4. OpenAI 图文 API 与请求级 Batching

### 4.1 Adapter

OpenAI 使用 `image_url`，Qwen 消息使用 `type=image` 和 `image` 字段。Adapter 负责：

```text
格式转换
保留图文交错顺序
识别 URL/base64/path
验证 base64
默认禁止服务器本地文件访问
```

### 4.2 HTTP 并发不等于模型合批

FastAPI 能同时运行 8 个 handler，不代表 GPU 会看到 batch size 8。

`MultimodalEngineCore` 维护 waiting queue、batch wait window、兼容规则、唯一模型 worker 和每请求 Event。

### 4.3 合批兼容性

```text
max_new_tokens 相同
temperature 相同
top_p 相同
image bucket 兼容
请求数、图像数、文本 token 和 image token 预算不超限
```

### 4.4 Event 与 Condition

```text
Event：某一条请求的结果是否完成
Condition：waiting queue 的状态是否变化
```

```text
API thread
  -> enqueue request
  -> notify Condition
  -> Event.wait()

background worker
  -> wait for queue
  -> wait batch window
  -> pop compatible batch
  -> model execution
  -> Event.set()
```

## 5. 为什么 HF Bridge 不是 Native Runtime

HF 路径的 `model.generate()` 内部拥有 vision tower、visual embeddings、language decoder、KV cache 和 sampling loop。NanoInfer 只看到 messages 和最终文本。

这也是最初 `image_feature_cache` 指标一直为 0 的根本原因：visual feature 根本不归 NanoInfer 所有。

Native 迁移后的边界：

```text
Hugging Face 保留：Processor + Vision Tower Architecture

NanoInfer 拥有：
  Scheduler
  Native Decoder
  Paged KV Cache
  Prefix Cache
  Vision Feature Cache
  Sampler
  CUDA Graph
```

这样可以先对齐 vision features，再对齐 logits，然后验证 paged-KV decode 和完整服务。

## 6. 一条 Native Qwen2.5-VL 请求的完整数据流

### 6.1 API 适配与请求合批

```text
OpenAI content parts
  -> MultimodalRequest
  -> compatible request batch
  -> decoded PIL images
```

### 6.2 Prompt Processor

Qwen Processor 返回：

```text
input_ids
pixel_values
image_grid_thw
```

Native prompt processor 进一步构造：

```text
VisionInput
  feature_key
  pixel_values
  grid_thw

MultimodalPromptMetadata
  position_ids
  visual_spans
  visual_token_identities
  mrope_position_delta
```

`pixel_values` 是多张图 patch 的连续 tensor，必须根据每张图的 `T*H*W` 和 `image_grid_thw` 切分，不能按图像数量平均切。

### 6.3 为什么不把视觉 Tensor 放入 Sequence

Sequence 需要被 Scheduler 管理、pickle 和在 TP worker 之间传递。大型 patches/features 会：

```text
超过共享命令缓冲区
引入额外拷贝
混淆调度元数据与 GPU tensor 所有权
```

所以：

```text
Sequence 保存 identity、span 和 position metadata
GPU Worker 保存 CPU patches 与 GPU features
```

### 6.4 进入普通 Scheduler

Processor 物化完 visual tokens 后，图文请求与文本请求一样变成 Sequence：

```text
Sequence
  token_ids
  sampling_params
  multimodal_metadata
  block_table
  num_cached_tokens
  num_scheduled_tokens
```

Scheduler 不关心图像字节，只关心 token 和 KV cache。

### 6.5 只解析本轮需要的视觉行

假设 visual span 是 `[20, 344)`：

```text
调度 [0, 128)
  -> 需要 feature rows [0, 108)

调度 [128, 256)
  -> 需要 feature rows [108, 236)

prefix KV 已覆盖 [0, 400)
  -> 不需要查 feature cache，也不需要运行 Vision Encoder
```

### 6.6 VisionFeatureProvider

```text
feature hit
  -> 返回 GPU feature

feature miss
  -> 从 CPU registry 取 patches
  -> 将多个 miss 组成 vision batch
  -> Vision Encoder
  -> 按 grid 拆分输出
  -> 写入 GPU LRU
```

同一批次如果需要 `a, a, b`，provider 只编码 `a, b`，不会重复编码 `a`。

### 6.7 混合 Embedding

```python
embeddings = model.embed(input_ids)

for replacement in replacement_plan:
    embeddings[replacement.output_row] = features[
        replacement.feature_key
    ][replacement.feature_row]
```

`plan_visual_replacements()` 必须检查：

```text
feature 是否存在
hidden size 是否匹配
output row 是否越界
feature row 是否越界
placeholder 数量是否与 feature rows 一致
```

视觉行错位很可能不 crash，却会让模型理解错误图像，因此必须严格失败。

### 6.8 Native Decoder 与 Decode

```text
mixed embeddings
+ 3-axis positions
+ attention metadata
+ slot mapping
  -> Qwen25VLForCausalLM
  -> paged KV cache
```

之后 decode 只输入 last token 和继续的 MRoPE position，不再需要 image patches。

## 7. 为什么 Qwen2.5-VL 需要三轴 MRoPE

### 7.1 文本位置是一维的

```text
token:     A  B  C  D
position:  0  1  2  3
```

### 7.2 图像 token 是网格

假设 merge 后的图像 grid 是 `2 x 3`：

```text
(h=0,w=0) (h=0,w=1) (h=0,w=2)
(h=1,w=0) (h=1,w=1) (h=1,w=2)
```

简单展平成 0,1,2,3,4,5 会丢失二维邻接关系。Qwen2.5-VL 为每个视觉 token 构造：

```text
temporal position
height position
width position
```

静态图像的 temporal 轴通常不随 patch 增长，height/width 按 grid 变化。

### 7.3 一个位置例子

```text
             v0 v1 v2 v3 v4 v5
temporal:     5  5  5  5  5  5
height:       5  5  5  6  6  6
width:        5  6  7  5  6  7
```

后续文本 token 再回到三轴相等的继续位置。

### 7.4 mrope_position_delta

图像网格后的最大 MRoPE position 不一定等于 `prompt_length - 1`。两者的偏移用 delta 保存：

```text
decode_position = sequence_length - 1 + mrope_position_delta
```

这个状态必须经历 decode 和 preemption/re-prefill，不能只在首次 prefill 临时生成。

### 7.5 为什么 RoPE 错误难发现

RoPE 错了往往不会 shape error。模型仍会生成语法正常的文本，只是 logits 和图像理解会系统性偏移。

所以需要直接与 Transformers 的 `apply_multimodal_rotary_pos_emb` 比较数值。

## 8. 三种完全不同的缓存

| 缓存 | 存什么 | 位置 | 省掉什么 |
| --- | --- | --- | --- |
| Decoded Image Cache | PIL RGB 图像 | CPU / Manager | 下载、base64、JPEG/PNG 解码 |
| Vision Feature Cache | Vision Encoder 输出 | GPU / Worker | Vision Encoder |
| Prefix KV Cache | Decoder 每层 K/V blocks | GPU / BlockManager | 已有前缀的 Decoder Prefill |

### 8.1 命中 decoded image 之后

仍需要：

```text
image processor
Vision Encoder
Decoder Prefill
```

### 8.2 命中 vision feature 之后

仍需要：

```text
将 feature 合并进 token embeddings
Decoder Prefill
```

### 8.3 命中 prefix KV 之后

如果 prefix KV 已经覆盖完整图像 span，可以同时跳过：

```text
Vision Feature Cache lookup
Vision Encoder
该前缀的 Decoder Prefill
```

所以 prefix KV 复用是更彻底的复用。

## 9. GPU Vision Feature Cache

### 9.1 为什么按字节数限制

不同图像的 feature rows 差异很大。因此不能用“最多缓存 256 张图”控制显存，而应计算：

```python
nbytes = tensor.numel() * tensor.element_size()
```

所有 resident bytes 不得超过 `NANOINFER_MM_FEATURE_CACHE_BYTES`。

### 9.2 LRU

```text
缓存 [A, B]
访问 A -> [B, A]
插入 C 且空间不足
淘汰 B -> [A, C]
```

### 9.3 为什么需要 Pin

一个 forward 需要 A 和 B。拿到 A 后，缓存在插入 B 时不能驱逐 A，因为后面还要将 A 写入 embedding。

```python
with feature_store.pin(("A", "B")) as features:
    merge_into_embeddings(features)
```

Pin 期间 `pin_count > 0` 的 entry 不允许驱逐。

### 9.4 为什么 put_many 必须原子化

如果 Vision Encoder 一次返回 A/B/C，不能插入 A/B 后才发现 C 使总预算超限。否则旧 entry 已被驱逐，新工作集又不完整。

`put_many()` 先计算整个批次是否可以放入，再一次性修改缓存。

### 9.5 KV Cache 必须为 Feature Cache 预留显存

如果 KV cache 在启动时吃完所有剩余显存，第一张图的 feature 就无法插入。

```text
KV 可用显存
  = GPU 目标可用显存
  - 模型已用显存
  - warmup 峰值
  - Vision Feature Cache 预留
```

## 10. CPU Input Registry 与 Feature Key

### 10.1 为什么 CPU patches 需要引用计数

两条并发请求可能共享同一张图：

```text
Request A registers X -> refcount 1
Request B registers X -> refcount 2
A finishes             -> refcount 1
B finishes             -> refcount 0，删除 CPU patches
```

如果 A 完成时直接删除 X，B 在 GPU feature cache miss 时就无法重新编码。

GPU feature 不随请求结束删除，它的寿命由 LRU 字节预算决定。

### 10.2 Feature Key 必须包含哪些语义输入

```text
模型 fingerprint
processor fingerprint
图像像素内容
detail / 处理参数
grid
```

只用 URL 会在同 URL 内容更新时错命中；只用 Python object ID 又无法让两份内容相同的图像复用。

缓存 key 应描述“生成 feature 的函数的全部语义输入”，不只是输入的获取路径。

## 11. Image-aware Prefix Cache 与 Scheduler

### 11.1 最危险的错误：不同图像误复用 KV

两个图文 prompt 可能有完全相同的 token IDs：

```text
[vision_start, image_pad, image_pad, ..., vision_end, same question]
```

如果 BlockManager 只 hash token ID，它会认为两个 prefix 相同，从而让第二张图错误复用第一张图的 K/V。

这个问题非常危险：

```text
不一定 crash
速度甚至会变快
但模型看到了错误图像的历史 KV
```

### 11.2 Visual Token Identity

文本 token identity：

```text
TextTokenIdentity(token_id)
```

视觉 token identity：

```text
VisualTokenIdentity(
    token_id=image_pad_id,
    feature_key=...,
    feature_row=...,
)
```

`feature_key` 区分不同图像，`feature_row` 区分同一张图内不同 visual token。

因此：

```text
同图 + 同 feature row -> 可复用
不同图               -> 不可复用
```

### 11.3 Chunked Prefill 中的视觉特征

图文 prompt 有 2000 token，visual span 是 `[100, 900)`，Scheduler 分块：

```text
chunk 1 [0, 512)      -> 需要 feature rows [0, 412)
chunk 2 [512, 1024)   -> 需要 feature rows [412, 800)
chunk 3 [1024, 1536)  -> 不需要图像 feature
chunk 4 [1536, 2000)  -> 不需要图像 feature
```

ModelRunner 不应每个 chunk 都解析整张图。

### 11.4 Prefix Cache Hit 与 Feature Cache Hit 的区别

```text
feature hit
  -> 省掉 Vision Encoder
  -> 仍需 Decoder Prefill

prefix KV hit
  -> 省掉 Vision Encoder/feature lookup
  -> 同时省掉已命中前缀的 Decoder Prefill
```

### 11.5 Preemption 后的 Re-prefill

KV blocks 不足时，Scheduler 可能抢占 Sequence，之后重新 prefill。

Native Qwen2.5-VL 必须保留：

```text
image-aware prefix identity
MRoPE position metadata
mrope_position_delta
```

否则恢复后的位置或 KV 复用就会错乱。

## 12. 为什么不能直接复用 Qwen3 Decoder

| 差异 | Qwen3 | Qwen2.5-VL Language Decoder |
| --- | --- | --- |
| Position | 普通 RoPE | 三轴 MRoPE |
| Position axes | 1 | 3 |
| QKV bias | 按 Qwen3 contract | 需按 Qwen2.5 checkpoint 对齐 |
| Q/K Norm | Qwen3 结构 | Qwen2.5 该处不使用 |
| Input | 通常 token IDs | token IDs 或 mixed embeddings |

“都是 Qwen”不代表 checkpoint parameter contract 二进制兼容。

Native `Qwen25VLForCausalLM` 复用 NanoInfer 的：

```text
RMSNorm
Column/Row Parallel Linear
Packed QKV/MLP
Attention kernel
KV cache layout
Sampler
```

但按 Qwen2.5 实现 QKV bias、权重映射、tied embedding 和 MRoPE。

### 12.1 为什么需要 inputs_embeds

文本模型通常：

```python
hidden_states = embed_tokens(input_ids)
```

VLM 需要：

```python
text_embeddings = embed_tokens(input_ids)
mixed_embeddings = replace_visual_rows(text_embeddings, vision_features)
hidden_states = model(inputs_embeds=mixed_embeddings)
```

所以共享模型接口增加：

```text
model.embed(input_ids)
model(..., inputs_embeds=mixed_embeddings)
model.compute_logits(hidden_states)
```

Qwen3 也实现该通用接口，但默认文本路径仍使用 token IDs。

## 13. 权重加载与一个隐蔽的 BF16 数值 Bug

### 13.1 为什么需要“选择性严格加载”

完整 Qwen2.5-VL checkpoint 同时包含：

```text
visual.*
model.*
lm_head.*
```

只创建 decoder 时不应加载 `visual.*`；只创建 vision tower 时又不应加载 decoder 权重。

loader 需要支持：

```text
include_prefixes
strip_prefix
packed QKV/MLP mapping
bias shard loading
unknown selected weight 报错
missing required weight 报错
duplicate write 报错
```

原则是“严格地选择”，而不是“宽松地忽略”。

### 13.2 FP32 inv_freq 被 BF16 破坏

现象：权重一致，但 Native/HF vision features 仍存在系统偏差，首个偏差出现在 vision rotary embedding。

根因：

```python
module.to(dtype=torch.bfloat16)
```

不只转换 parameter，也会转换 buffer。FP32 `inv_freq` 被舍入为 BF16 后，再转回 FP32 也无法恢复已丢失的有效位。

这就像先将 `3.1415926` 舍入为 `3.14`，之后写成 `3.1400000` 也不能恢复原始数值。

解决：

```text
dtype 迁移前复制 FP32 buffers
迁移后恢复
根据 config 在 CPU FP32 中重建 canonical inv_freq
再移到 GPU
```

最终 A100 对齐中 Native/HF vision features 逐元素完全相等。

### 13.3 这个 Bug 的一般经验

```text
module.to(dtype=...) 不只影响参数。
频率、累加统计、归一化常量等 buffer 的精度应被明确管理。
```

## 14. CUDA Graph 如何支持多模态 Decode

CUDA Graph 用固定形状 buffer 捕获一段 CUDA 操作，decode 时直接 replay，减少 Python 和 kernel launch 开销。

文本模型的 position buffer 通常是：

```text
[max_batch_size]
```

Qwen2.5-VL 需要：

```text
[3, max_batch_size]
```

runtime 根据模型 contract 中的 `position_axes` 分配 buffer，而不是在通用 ModelRunner 中到处判断模型名。

Padding row 的 slot mapping 设为 `-1`，避免将 padding 写入 KV cache。

## 15. 怎么证明 Native Runtime 是正确的

“能生成一段通顺文本”不足以证明 runtime 正确。错误 MRoPE、错误 feature row 或错误权重也可能产生语法正常的文本。

### 15.1 第一层：Vision Features

使用同一权重、同一图像和同一 processor，比较 HF/Native vision output。

服务器结果：

```text
max absolute error = 0
mean absolute error = 0
```

### 15.2 第二层：Final-token Logits

服务器结果：

```text
cosine similarity = 0.9999787807
argmax = same
top-10 overlap = 10/10
```

### 15.3 第三层：Cache 行为

```text
首次同图请求 -> encoded_images +1
后续同图不同文本 -> feature cache hit，encoded_images 不增加
完全相同视觉前缀 -> prefix KV hit，不解析/编码 feature
```

### 15.4 第四层：端到端与回归

```text
CUDA Graph image generation passed
Qwen3-0.6B GPU regression passed
4 concurrent HTTP requests -> 4/4 HTTP 200, 1 model batch
100 native requests -> 100/100 success, image encoded once
CPU test suite -> 75 tests passed
```

### 15.5 数值 Gate 的思路

```text
从中间 tensor 开始
-> logits
-> top-k/argmax
-> cache counters
-> paged-KV/CUDA Graph
-> HTTP 端到端
-> 旧模型回归
```

越早定位第一个数值偏差，排错范围就越小。

## 16. 怎么理解 Benchmark 数据

当前服务器观测：

| 指标 | Native | HF |
| --- | ---: | ---: |
| 100 请求总时间 | 92.87 s | 432.62 s |
| 请求吞吐 | 1.077 req/s | 0.231 req/s |
| 平均延迟 | 7.09 s | 33.24 s |
| P50 | 7.08 s | 31.79 s |
| P90 | 7.31 s | 40.11 s |
| P99 | 8.17 s | 40.83 s |

在该次重复图像、concurrency=8 的共享 GPU 观测中，Native 显示出约 4.66x 的 request throughput。

但不能将 4.66x 宣称为独占 A100 上的通用 VLM 加速比，因为：

```text
Native 和 HF 在不同时段运行，共享 GPU 外部负载不同
图片重复，主要展示 feature cache 收益
没有 completion token usage，无法严格对齐 token workload
另一次受干扰的 Native 观测只有 0.410 req/s
```

正确表述是：

> 在当前共享 GPU、重复图片和并发 8 的指定请求场景中，Native runtime 展示出显著优于 HF bridge 的请求吞吐与延迟。

## 17. 当前限制

下列内容仍不应在简历中写成已支持：

```text
Native Qwen2.5-VL 只支持 TP=1
不支持视频
Native multimodal 当前要求 top_p=1.0
未实现 OpenAI SSE streaming
没有客户端 TTFT/TPOT
Vision Tower 架构仍复用 Transformers
图像预处理还没有 CPU worker pool
没有视觉 token pruning/resampler
未完成独占 GPU 的严格多轮基准
还不是通用 VLM plugin ABI
```

## 18. 常见问题与排查思路

### 18.1 `Connection refused`

这通常不是模型问题。先检查：

```text
uvicorn 是否仍在运行
启动端口与 benchmark 端口是否一致
模型懒加载是否还没完成
服务是否因 OOM 退出
```

### 18.2 Qwen2.5-VL 模型类不存在

检查 Transformers 版本。服务器验证使用 4.52.2，项目约束为 `transformers>=4.51.0`。

### 18.3 `qwen_vl_utils` 导入失败

`qwen-vl-utils` 不是 Transformers 核心依赖，需要安装 multimodal extras。

### 18.4 Vision FlashAttention 报 `wrap_triton` 不存在

这是 PyTorch/FlashAttention/Transformers 版本组合的内部 API 不兼容。Native vision provider 默认使用 SDPA，先完成数值对齐，再尝试更快 kernel。

### 18.5 Vision Features 与 HF 不一致

排查顺序：

```text
processor output
pixel_values/grid split
vision weights
RoPE inv_freq dtype
attention implementation
feature split order
```

服务器开发中真实问题是 FP32 `inv_freq` 在 `module.to(bfloat16)` 时被转换。

### 18.6 不同图像错误复用 KV

检查 visual token identity 是否包含：

```text
token_id
feature_key
feature_row
```

### 18.7 Feature Cache 一直为 0

先确认 backend mode。HF bridge 不属于 NanoInfer feature cache 的数据所有者，因此应明确报告 0。

### 18.8 Feature Cache OOM

检查：

```text
byte budget 是否为正
单 entry 是否大于总预算
pinned working set 是否大于总预算
KV cache 分配前是否扣除 feature cache reserve
```

### 18.9 Benchmark 最后写文件失败

运行前就应创建输出目录，不要等 100 个请求完成后才发现父目录不存在。

### 18.10 TTFT/TPOT 为 null

当前 API 不支持 streaming，客户端只在完整 JSON 生成后收到响应。正确做法是返回 null，而不是使用总延迟伪造 TTFT。

### 18.11 `libtinfo.so.6` 警告

Conda `LD_LIBRARY_PATH` 可能让系统 shell 加载不兼容的 libtinfo。服务器验证命令使用：

```bash
env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=0 <command>
```

## 19. 建议的代码阅读顺序

### 第一轮：文本引擎骨架

```text
example.py
nanovllm/llm.py
nanovllm/engine/llm_engine.py
nanovllm/engine/engine_core.py
nanovllm/engine/scheduler.py
nanovllm/engine/block_manager.py
nanovllm/engine/executor.py
nanovllm/engine/gpu_worker.py
nanovllm/engine/model_runner.py
```

目标：能说出 Sequence 从 waiting 到 running 再到 finished 的路径。

### 第二轮：服务与请求级合批

```text
nanovllm/serving/schemas.py
nanovllm/serving/openai_api.py
nanovllm/serving/engine_client.py
nanovllm/multimodal/openai_adapter.py
nanovllm/engine/multimodal_engine.py
nanovllm/multimodal/batching.py
```

目标：能解释 HTTP concurrency 如何变成 model batch。

### 第三轮：Prompt 和多模态元数据

服务器 Native 实现中的关键文件：

```text
nanovllm/multimodal/qwen25_vl_processor.py
nanovllm/multimodal/runtime.py
nanovllm/multimodal/qwen25_vl_runtime.py
nanovllm/engine/sequence.py
```

目标：能说出 input IDs、pixel values、grid、visual spans、cache identity 和 MRoPE positions 的关系。

### 第四轮：Vision Feature Cache

```text
nanovllm/multimodal/feature_store.py
nanovllm/multimodal/vision_provider.py
tests/test_vision_feature_store.py
```

目标：能解释 byte budget、LRU、pinning、atomic put_many、deduplication 和 CPU registry refcount。

### 第五轮：模型与 ModelRunner

```text
nanovllm/layers/multimodal_rotary_embedding.py
nanovllm/models/qwen25_vl.py
nanovllm/engine/model_runner.py
nanovllm/engine/block_manager.py
```

阅读 ModelRunner 时优先定位：

```text
plan_visual_replacements
prepare_prefill
prepare_decode
run_model
capture_cudagraph
```

### 第六轮：跟着测试理解合同

```text
tests/test_qwen25_vl_processor.py
tests/test_multimodal_prefix_cache.py
tests/test_vision_feature_store.py
tests/test_model_runner_multimodal.py
tests/test_native_serving.py
```

测试往往比大段实现更容易理解，因为每个测试只描述一个行为合同。

## 20. 可以动手完成的实验

1. 打印 `input_ids.shape`、`pixel_values.shape` 和 `image_grid_thw`。
2. 手工写出 `2 x 3` 图像 grid 的 temporal/height/width positions。
3. 修改图像一个像素，比较 feature key。
4. 将 GPU feature cache 预算设得很小，观察 entry 超限和 pinned working set 超限。
5. 连续请求同一张图，观察 `encoded_images`、feature hits 和 prefix entries。
6. 保持文本不变但更换图像，验证 prefix KV 不会错复用。
7. 使用 concurrency=1/2/4/8/16，比较 average batch size 和 P99。
8. 在独占 GPU 上比较 cold cache、feature hit、prefix hit 和 unique images 四个场景。

## 21. 自测题

1. 为什么 HF `generate()` 能返回图文结果，却不代表请求进入了 NanoInfer runtime？
2. 命中 Decoded Image Cache 后，为什么仍可能要执行 Vision Encoder？
3. 命中 Vision Feature Cache 后，为什么仍可能要执行 Decoder Prefill？
4. 两个 prompt 的 token IDs 完全一样，prefix identity 为什么仍可能不同？
5. Vision Feature 为什么不直接存入 Sequence？
6. Feature Cache 为什么需要 Pin？
7. `module.to(bfloat16)` 为什么可能破坏 FP32 `inv_freq`？
8. Qwen2.5-VL decode 为什么需要 `mrope_position_delta`？
9. 为什么只看生成文本无法证明 runtime 正确？
10. 为什么当前 4.66x 数据不能作为通用加速结论？

## 22. 完整学习结论

这个项目的核心不是“调通一个图文模型”，而是将多模态数据纳入自研 runtime 的正确性与资源所有权体系：

```text
Scheduler 要看到真实 visual tokens
Prefix cache 要区分不同图像
GPU worker 要拥有和按字节管理 visual features
Chunked prefill/prefix reuse 要决定本轮是否需要 feature
MRoPE、weight mapping 和 FP32 buffers 要与 reference 数值对齐
HTTP concurrency 要转化为真正 model batch
性能数据要与 GPU 环境、缓存场景和 token workload 一起解读
```

当你能清楚解释下面这条路径，就已经理解项目主体：

```text
OpenAI request
-> request-level batching
-> decoded image cache
-> Qwen processor
-> VisionInput + MRoPE metadata
-> Sequence/Scheduler
-> selective visual replacement plan
-> GPU feature cache / vision encoder
-> mixed embeddings
-> Native Qwen2.5 decoder
-> paged KV cache
-> decode/CUDA Graph
-> metrics and benchmark
```

## 23. 文档来源与代码版本边界

本文档整合了：

```text
docs/LEARNING_GUIDE.md
QWEN25_VL_NATIVE_LEARNING_GUIDE.md
QWEN25_VL_NATIVE_DEVELOPMENT_POSTMORTEM.md 中的验证边界
```

当前本地 `main` 仍主要是 HF bridge 与请求级 batching 版本，Native Runtime 文件来自服务器开发分支的文档记录。在服务器源码分支完成 Git 同步前，本文档中的 Native 代码路径是学习索引，不表示这些文件已经存在于当前本地 checkout。
