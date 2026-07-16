# NanoInfer 从 nano-vLLM 到多模态推理服务：逐步学习文档

## 0. 这份文档怎么看

这份文档不是文件清单，而是带你理解“为什么要这样改”。

我们会围绕两个具体请求展开：

```text
文本请求：请续写“人工智能推理引擎是”

图文请求：给定一张图，请回答“图中有什么？”
```

文本会按这个顺序讲解：

```text
1. 原始 nano-vLLM 如何生成文本
2. 为什么要增加服务层
3. 为什么要拆分 EngineCore / Executor / GPUWorker
4. 怎么计算 TTFT、TPOT 和 KV cache 使用率
5. OpenAI 图文请求如何转成 Qwen2.5-VL 请求
6. 图像缓存到底缓存了什么
7. 多个图文请求如何合并成 batch
8. Condition、Event 和后台 worker 是如何配合的
9. benchmark 在测什么
10. 当前项目还缺什么
```

建议第一遍只看每章的“先记住”和例子，第二遍再对着代码看。

### 快速阅读路线

如果你现在还不熟悉引擎，先读：

```text
第 1 章：先理解文本推理
第 3 章：理解 EngineCore 和 GPU 执行分层
第 5 章：理解 Qwen2.5-VL 输入和输出
第 6 章：理解 OpenAI 请求转换
第 8 章：理解 batching 判断
第 9 章：理解 Event、Condition 和后台 worker
第 13 章：把整条图文请求连起来
```

完整目录：

```text
1.  整体地图
2.  服务层
3.  EngineCore / Executor / GPUWorker
4.  metrics 与时间统计
5.  Qwen2.5-VL 后端
6.  OpenAI 请求适配
7.  图像缓存
8.  图文 batching
9.  MultimodalEngineCore 并发模型
10. Qwen 批量输入输出对齐
11. benchmark
12. 问题与解决
13. 跟踪完整图文请求
14. 尚未完成的内容
15. 建议学习顺序
16. 练习题
17. 面试讲法
18. 术语表
19. 阅读完成标准
```

## 1. 先建立一张整体地图

### 1.1 原始 nano-vLLM 在做什么

用户输入一段文本，模型需要经历以下过程：

```text
字符串 prompt
  -> tokenizer 转成 token IDs
  -> prefill：一次处理 prompt 中的多个 token
  -> 建立 KV cache
  -> decode：每轮为每个请求生成一个 token
  -> 遇到 EOS 或 max_tokens
  -> tokenizer 将 token IDs 转回字符串
```

例如 prompt 分词后是：

```python
[101, 2057, 2293, 3185]
```

prefill 阶段会处理这 4 个 token，并将每层 attention 中的 Key/Value 写入 KV cache。

假设模型之后生成：

```python
[2024, 2003, 2204]
```

decode 并不会每次重新计算整个 prompt，而是读取 KV cache，每轮只计算新增的一个 token。

### 1.2 prefill 和 decode 为什么要分开理解

prefill 和 decode 的计算特征很不一样。

| 阶段 | 一次处理多少 token | 常见瓶颈 | 关心指标 |
| --- | ---: | --- | --- |
| Prefill | 一个或多个 prompt 的大量 token | 计算量、长 prompt 阻塞 | tokens/s、TTFT |
| Decode | 每个请求每轮 1 个 token | 显存带宽、KV cache 容量 | seqs/s、TPOT |

这也是 `EngineCore.step()` 中使用正负号区分两种阶段的原因：

```python
num_tokens = (
    sum(seq.num_scheduled_tokens for seq in seqs)
    if is_prefill
    else -len(seqs)
)
```

这里的负号不是“负 token”，只是一个简单标记：

```text
num_tokens > 0：这一步是 prefill
num_tokens < 0：这一步是 decode，绝对值是请求数
```

### 1.3 原始项目的几个核心对象

#### Sequence

`Sequence` 表示一条正在生成的请求。

它不只是 token 列表，还保存：

```text
请求 ID
当前状态：WAITING / RUNNING / FINISHED
prompt token IDs
completion token IDs
KV cache block table
已缓存 token 数
本轮调度 token 数
temperature
max_tokens
ignore_eos
```

你可以把它理解为“一条请求的全部运行档案”。

#### Scheduler

`Scheduler` 决定这一步让哪些 Sequence 进入 GPU。

它管理两个队列：

```text
waiting：还没完成 prefill
running：已完成 prefill，正在 decode
```

它还要考虑：

```text
每次最多处理多少请求
每次 prefill 最多处理多少 token
KV cache 是否足够
是否可以复用 prefix cache
需不需要抢占其它序列
```

#### ModelRunner

`ModelRunner` 是真正与 CUDA、模型权重和 KV cache tensor 打交道的对象。

它负责：

```text
加载权重
分配 KV cache
准备 prefill/decode tensor
调用模型 forward
采样下一个 token
tensor parallel 通信
```

### 本章先记住

```text
Sequence = 请求状态
Scheduler = 决定谁上 GPU
ModelRunner = 真正跑模型
Prefill = 处理 prompt
Decode = 逐 token 生成
KV cache = 避免重复计算历史 token
```

## 2. 第一步修改：从离线脚本到 API 服务

### 2.1 原来只能怎么用

原来的主要用法是：

```python
llm = LLM(model_path)
outputs = llm.generate(prompts, sampling_params)
```

这种方式适合学习和离线 benchmark，但不适合真实服务，因为其它程序无法通过 HTTP 发送请求。

### 2.2 我们增加了哪些层

```text
HTTP JSON
  -> Pydantic schema
  -> FastAPI route
  -> EngineClient
  -> LLM.generate()
```

相关文件：

```text
nanovllm/serving/schemas.py
nanovllm/serving/openai_api.py
nanovllm/serving/engine_client.py
nanovllm/serving/server.py
nanovllm/serving/entrypoint.py
```

### 2.3 Schema 在解决什么问题

用户可以发送：

```json
{
  "model": "qwen3-0.6b",
  "messages": [
    {"role": "user", "content": "什么是 KV cache？"}
  ],
  "max_tokens": 128,
  "temperature": 0.7
}
```

Pydantic schema 会在进入引擎前检查：

```text
messages 是否为空
role 是否是合法值
max_tokens 是否大于等于 1
temperature 是否大于 0
top_p 是否在 (0, 1] 中
图文 content part 的 type 是否合法
```

这样做的价值是：错误请求尽量在 API 入口被拒绝，不让它们进入 GPU 执行路径。

### 2.4 EngineClient 为什么需要一把锁

文本引擎当前仍使用阻塞式 `LLM.generate()`。

如果两个 HTTP 线程同时操作同一个 Scheduler，可能会出现：

```text
请求 A 正在 schedule
请求 B 同时向 waiting 队列加入 Sequence
请求 A 和 B 同时调用 generate 主循环
输出归属和引擎状态变得不可控
```

因此当前文本路径使用 Lock 保证一次只有一个 `generate()` 调用。

但你也应该看到：这只是正确性保护，不是最终高性能方案。

真正的 serving continuous batching 应该是：

```text
所有 API 请求向同一个异步引擎提交
引擎运行过程中可以动态接纳新请求
调度器每轮重新组批
```

这一点还是文本引擎的后续工作。

### 本章先记住

```text
Schema 负责检查请求
Route 负责接收 HTTP
EngineClient 负责隔离 API 与推理引擎
Lock 解决当前的并发正确性，但会串行化请求
```

## 3. 第二步修改：拆分 EngineCore / Executor / GPUWorker

### 3.1 原来的 LLMEngine 职责太多

重构前，`LLMEngine` 同时负责：

```text
读取 Config
加载 tokenizer
创建 tensor-parallel 子进程
创建 ModelRunner
创建 Scheduler
接收请求
执行 schedule
执行 model forward
执行 postprocess
结束子进程
```

对一个 200 行左右的学习引擎来说，这很紧凑。

但当我们要加入服务、指标、多模态和更多执行后端时，它会越来越难修改。

### 3.2 拆分后每个对象只管一层

```text
LLMEngine
  用户接口、tokenizer、Sequence 创建、文本 decode

EngineCore
  请求生命周期、schedule -> execute -> postprocess

Executor
  启动和管理 GPU worker、tensor-parallel 进程

GPUWorker
  向上层提供稳定的 run/shutdown 接口

ModelRunner
  模型、CUDA、KV cache tensor、采样
```

### 3.3 EngineCore.step() 逐行理解

核心代码可以简化为：

```python
def step(self):
    started = perf_counter()
    seqs, is_prefill = self.scheduler.schedule()
    token_ids = self.executor.run(seqs, is_prefill)
    self.scheduler.postprocess(seqs, token_ids, is_prefill)
    outputs = [seq for seq in seqs if seq.is_finished]
    self._record_step_stats(...)
    return outputs
```

一步内发生了什么？

#### 第 1 步：Scheduler 选请求

```python
seqs, is_prefill = self.scheduler.schedule()
```

返回值例子：

```python
seqs = [seq_3, seq_7, seq_8]
is_prefill = False
```

意思是：这一轮对 3 条序列各生成一个 token。

#### 第 2 步：Executor 执行模型

```python
token_ids = self.executor.run(seqs, is_prefill)
```

返回值可能是：

```python
[532, 18, 9012]
```

与 3 条 Sequence 一一对应。

#### 第 3 步：postprocess 更新状态

```python
self.scheduler.postprocess(seqs, token_ids, is_prefill)
```

这里会：

```text
更新已进入 KV cache 的 token 数
向 Sequence 追加新 token
检查 EOS
检查 max_tokens
释放完成请求的 KV blocks
```

#### 第 4 步：收集完成请求

只有 `seq.is_finished` 为 True 的序列会被返回给上层。

未完成的序列仍保留在 Scheduler 中，下一轮继续 decode。

### 3.4 为什么 Executor 要先于 Scheduler 创建

`EngineCore.__init__()` 的顺序是：

```python
self.executor = Executor(config)
self.scheduler = Scheduler(config)
```

这不是随便写的。

`ModelRunner` 初始化时会根据 GPU 显存计算 KV cache 能分配多少个 block，然后写入：

```python
config.num_kvcache_blocks
```

`Scheduler` 创建 `BlockManager` 时又需要这个数值。

所以初始化顺序必须是：

```text
ModelRunner / Executor 先计算显存容量
  -> Config 获得 num_kvcache_blocks
  -> Scheduler 再创建 BlockManager
```

如果反过来，Scheduler 可能会拿到未初始化或错误的 KV block 数量。

### 3.5 process.join() 为什么必须保留

tensor parallel 会创建子进程：

```python
process = ctx.Process(...)
process.start()
self.processes.append(process)
```

shutdown 时：

```python
self.driver_worker.shutdown()
for process in self.processes:
    process.join()
```

`shutdown()` 是通知子进程结束。

`join()` 是主进程等待子进程真正退出，并回收进程状态。

如果只发送退出通知而不 `join()`，主进程可能在子进程尚未释放 CUDA/NCCL 资源时就退出，还可能留下僵尸进程。

### 本章先记住

```text
EngineCore 不跑 CUDA，它负责编排
Executor 拥有 worker 和子进程
GPUWorker 包装 ModelRunner
初始化顺序受 KV cache 容量计算影响
shutdown 负责通知，join 负责等待回收
```

## 4. 第三步修改：让引擎可观测

### 4.1 为什么“能生成”还不够

一个推理项目如果只输出文本，你不知道：

```text
请求排队多久
用户多久看到第一个 token
之后每个 token 多久
prefill 是否过慢
decode 是否过慢
KV cache 是否即将用完
batch 到底有多大
```

没有这些数据，就无法判断优化是否有效。

### 4.2 Sequence 的时间线

```text
arrival_time
    |
    | 排队，可能经历 chunked prefill
    v
running_time
    |
    | 完成最后一段 prefill 调度并开始生成
    v
first_token_time
    |
    | 继续 decode
    v
finish_time
```

当前实现的计算公式：

```text
queue_time = running_time - arrival_time
TTFT = first_token_time - arrival_time
total_latency = finish_time - arrival_time
TPOT = (finish_time - first_token_time) / (completion_tokens - 1)
```

例如：

```text
arrival_time = 10.000 s
running_time = 10.080 s
first_token_time = 10.120 s
finish_time = 10.570 s
completion_tokens = 10
```

那么：

```text
queue_time = 80 ms
TTFT = 120 ms
total_latency = 570 ms
TPOT = (570 - 120) / 9 = 50 ms/token
```

### 4.3 为什么用 monotonic()

`time.time()` 表示当前墙上时钟，可能被系统校时修改。

`monotonic()` 只保证单调递增，适合计算时间差。

计算延迟时，我们关心“过去了多久”，不关心“现在是几点”，所以使用 `monotonic()` 更正确。

### 4.4 SchedulerStats 在看什么

```python
SchedulerStats(
    waiting_seqs=3,
    running_seqs=7,
    total_kv_blocks=1000,
    used_kv_blocks=720,
    free_kv_blocks=280,
    prefix_cache_entries=42,
)
```

KV cache 使用率：

```text
720 / 1000 = 72%
```

当使用率持续接近 100% 时，Scheduler 更容易发生抢占，请求可能被放回 waiting 重新 prefill。

### 4.5 当前 metrics 的一个细节

`running_time` 是在 Scheduler 将请求的最后一段 prefill 安排进当前 step 时记录的，时点位于该 step 真正执行 model forward 之前。

因此当前 `queue_time` 更准确的含义是：

```text
从请求创建，到最后一段 prefill 获得调度的时间
```

它不包含最后这个 prefill step 的执行时间。TTFT 仍包含从到达到产生首 token 的完整时间。

### 本章先记住

```text
TTFT 主要影响“用户多久看到开始回答”
TPOT 主要影响“回答过程流不流畅”
KV cache 使用率影响能同时服务多少请求
metrics 的关键是时间点放在正确的状态转换位置
```

## 5. 第四步修改：接入 Qwen2.5-VL

### 5.1 图文模型比纯文本模型多了什么

纯文本路径：

```text
text -> tokenizer -> text token IDs -> language model
```

图文路径：

```text
image -> image loader/processor -> patches -> vision encoder -> visual embeddings
text  -> tokenizer ------------------------------------------> text embeddings
visual embeddings + text embeddings -> language model -> answer
```

所以多模态服务不是只给 API 加一个图片字段。

它还会带来：

```text
图片下载与解码延迟
不同分辨率导致的不同 patch/token 数
vision encoder 计算开销
图像 token 对 KV cache 的压力
重复图像是否可以复用
图像重请求与短文本请求的调度公平性
```

### 5.2 为什么先用 HuggingFace backend

如果一开始就自己实现 Qwen2.5-VL 的：

```text
vision transformer
multimodal rotary position encoding
image token merge
language model forward
CUDA kernels
tensor parallel
```

项目会很长时间都无法跑通第一个图文请求。

所以当前采用两步走：

```text
第一步：用 HuggingFace Qwen2.5-VL 打通真实模型路径
第二步：再将图像预处理、vision encoder、缓存和调度逐步下沉
```

这是一个工程取舍：先获得端到端正确性，再替换性能瓶颈。

### 5.3 Qwen25VLBackend.generate_chat() 做了什么

```text
messages
  -> apply_chat_template()
  -> process_vision_info()
  -> AutoProcessor(..., return_tensors="pt")
  -> inputs.to(model device)
  -> model.generate()
  -> 删除 input token 部分
  -> batch_decode()
  -> 返回文本
```

为什么要删除 input token 部分？

`generate()` 返回的序列通常是：

```text
[prompt token IDs] + [new token IDs]
```

用户只需要新生成的部分，所以需要：

```python
output_ids[input_ids.shape[-1]:]
```

### 5.4 为什么延迟导入多模态依赖

Qwen2.5-VL 需要：

```text
transformers
accelerate
qwen-vl-utils
Pillow
torchvision
```

但纯文本用户不一定需要它们。

因此 `_ensure_loaded()` 只在第一次真正调用 VLM 时导入依赖和加载模型。

这样有两个好处：

```text
启动纯文本服务时不会加载 VLM
缺少多模态依赖时，只在调用 VLM 时给出明确错误
```

### 5.5 为什么有两个模型类名

代码优先查找：

```python
Qwen2_5_VLForConditionalGeneration
```

如果当前 transformers 版本没有它，再回退到：

```python
AutoModelForImageTextToText
```

这是在处理不同 transformers 版本的 API 兼容性，而不是同时加载两个模型。

### 本章先记住

```text
Qwen2.5-VL 路径是真实模型路径，但底层仍由 HuggingFace 执行
延迟导入用于隔离文本与多模态依赖
先跑通模型，再拆 vision encoder，是当前的工程路线
```

## 6. 第五步修改：把 OpenAI 请求转成 Qwen 请求

### 6.1 两种请求格式不一样

OpenAI 风格：

```json
{
  "role": "user",
  "content": [
    {
      "type": "image_url",
      "image_url": {
        "url": "https://example.com/street.jpg",
        "detail": "low"
      }
    },
    {
      "type": "text",
      "text": "图中有什么？"
    }
  ]
}
```

Qwen2.5-VL 风格：

```python
{
    "role": "user",
    "content": [
        {
            "type": "image",
            "image": "https://example.com/street.jpg",
            "detail": "low",
        },
        {
            "type": "text",
            "text": "图中有什么？",
        },
    ],
}
```

两者的字段名和 `type` 都不一样，所以需要 adapter。

### 6.2 为什么不把转换逻辑留在 FastAPI route 里

最初转换逻辑直接写在 `openai_api.py`。

这会产生几个问题：

```text
route 既处理 HTTP，又理解 Qwen 格式
benchmark 或离线代码无法复用转换逻辑
增加新图像来源时需要改 API route
安全检查与 HTTP 错误处理混在一起
```

因此新增：

```text
nanovllm/multimodal/openai_adapter.py
```

它输出两个结果：

```python
AdaptedChatRequest(
    multimodal_request=...,
    text_prompt=...,
)
```

`multimodal_request` 给 VLM 路径使用。

`text_prompt` 给没有图片的普通文本 chat 路径使用。

### 6.3 图文顺序为什么不能打乱

下面两个请求语义不同：

```text
图 1 -> “这张图有什么？” -> 图 2 -> “两张图有什么变化？”
```

```text
图 1 -> 图 2 -> “这张图有什么？两张图有什么变化？”
```

如果适配器先收集所有图像，再收集所有文本，就会改变原请求的语义。

所以代码按 content part 的原始顺序逐个转换。

### 6.4 图像来源怎么判断

```text
data:image/...;base64,... -> source_type = base64
http://...                -> source_type = url
https://...               -> source_type = url
file:///...               -> source_type = path
没有 scheme 的字符串    -> source_type = path
```

base64 请求还会在 API 入口执行：

```python
base64.b64decode(encoded, validate=True)
```

这样无效数据不会进入后台 worker 后才报错。

### 6.5 为什么本地文件默认关闭

如果一个公开 HTTP 服务允许客户端传入：

```json
{
  "image_url": {
    "url": "file:///some/server/path/image.png"
  }
}
```

那么客户端就在尝试控制服务器读取本地文件。

即使当前只会用 Pillow 打开图片，这仍然是不应默认开放的权限。

所以默认：

```text
NANOINFER_MM_ALLOW_LOCAL_FILES=0
```

只有受信环境明确设置为 1 时才允许本地路径。

### 本章先记住

```text
Adapter 将 API 协议与模型协议解耦
图文 content 的顺序是语义的一部分
无效输入应该尽量在 API 入口失败
本地文件访问是安全能力，必须默认关闭
```

## 7. 第六步修改：多模态缓存到底缓存什么

### 7.1 一张图从请求到模型经历了什么

以 URL 图片为例：

```text
URL
  -> 发起 HTTP 请求
  -> 获得 JPEG/PNG 字节
  -> Pillow 解码
  -> 转成 RGB image
  -> resize / normalize / patchify
  -> vision encoder
  -> visual embeddings
  -> language model
```

如果用户连续对同一张图提问：

```text
第 1 次：这张图里有什么？
第 2 次：图中物体的颜色是什么？
第 3 次：请生成一段图片描述。
```

没有缓存时，下载、解码和 vision encoder 可能重复执行 3 次。

### 7.2 Asset cache 和 feature cache 不是同一个东西

#### ImageAssetCache

当前真正启用的缓存。

它保存：

```text
已下载并解码成 RGB 的 PIL image 对象
```

命中后可以节省：

```text
URL 下载
base64 解码
JPEG/PNG 解码
RGB 转换
```

但仍然需要执行：

```text
Qwen image processor
vision encoder
language model
```

#### ImageFeatureCache

当前只有接口，没有真正连接到 Qwen2.5-VL forward。

理想中它应保存：

```text
vision encoder 输出的 visual embeddings
```

命中后可以跳过：

```text
下载
图像解码
图像预处理
vision encoder
```

但这要求我们将 HuggingFace `generate()` 中封装的 vision encoder 拆出来，并让 language model 接受已缓存的 visual embeddings。

因此请准确记住：

```text
已实现：decoded image asset cache
未实现：true vision feature reuse
```

### 7.3 LRU 缓存如何工作

LRU 是 Least Recently Used，即“最久没被使用的数据先淘汰”。

假设缓存容量为 3：

```text
放入 A -> [A]
放入 B -> [A, B]
放入 C -> [A, B, C]
访问 A -> [B, C, A]
放入 D -> [C, A, D]
```

B 最久没被访问，所以被淘汰。

代码使用 `OrderedDict` 实现：

```python
value = self._items.get(key)
self._items.move_to_end(key)
```

命中时将 key 移到末尾，表示它是最近刚被使用的。

容量超限时：

```python
self._items.popitem(last=False)
```

`last=False` 表示弹出最左边、也就是最久没被使用的元素。

### 7.4 不同图像来源如何生成 cache key

#### Base64

```text
key = SHA256(图像字节)
```

两个请求只要图像内容完全相同，就会命中。

#### 本地路径

```text
key = 规范化路径 + 修改时间 + 文件大小
```

如果文件被覆盖，mtime 或 size 通常会变化，从而生成新 key。

#### URL

```text
key = SHA256(URL 字符串)
```

这样可以在下载之前查询缓存，但它有一个局限：

```text
如果同一 URL 对应的图像内容更新，缓存仍可能返回旧图像。
```

更成熟的方案是加入：

```text
TTL
HTTP ETag
Last-Modified
再验证
或下载后按真实内容 hash
```

### 7.5 一次缓存命中的完整例子

第一个请求：

```text
image URL = https://example.com/a.jpg
cache.get(url_hash) -> miss
load_image() -> 下载 + Pillow 解码
cache.put(url_hash, pil_image)
```

第二个请求：

```text
image URL = https://example.com/a.jpg
cache.get(url_hash) -> hit
直接获得 pil_image
```

指标变化：

```text
hits = 1
misses = 1
hit_rate = 1 / (1 + 1) = 50%
```

### 本章先记住

```text
Asset cache 节省下载和解码
Feature cache 理论上还能节省 vision encoder，但目前未真正接入
LRU 防止缓存无限增长
URL 作为 key 速度快，但有内容过期风险
```

## 8. 第七步修改：图文请求怎么组成 batch

### 8.1 为什么 batch 很重要

假设 GPU 一次处理 4 个请求的耗时只是单请求的 1.8 倍：

```text
单条串行：4 x 100 ms = 400 ms
一次 batch：180 ms
```

那么吞吐就可能提升。

但 batch 不是越大越好：

```text
batch 越大 -> GPU 利用率可能更高
batch 越大 -> 显存占用也更高
等待 batch 的窗口越长 -> 吞吐可能更高
等待 batch 的窗口越长 -> 单请求延迟也更高
```

这是服务系统中典型的 latency-throughput tradeoff。

### 8.2 为什么不是所有请求都能合并

假设有 4 条请求：

| 请求 | max_tokens | temperature | top_p | 图片 | detail |
| --- | ---: | ---: | ---: | ---: | --- |
| A | 128 | 0.2 | 1.0 | 1 | low |
| B | 128 | 0.2 | 1.0 | 1 | low |
| C | 128 | 0.8 | 1.0 | 1 | low |
| D | 128 | 0.2 | 1.0 | 1 | high |

当前策略下：

```text
A 和 B 可以合并
C 不能合并，因为 temperature 不同
D 不能合并，因为 image detail bucket 不同
```

为什么 temperature 不同不能合并？

当前 HuggingFace backend 调用的是：

```python
model.generate(
    max_new_tokens=...,
    temperature=...,
    top_p=...,
)
```

这些参数是一次 batch 共用的标量。

如果将 A 和 C 强行合并，就必须选择一个 temperature，从而悄悄改变其中一条请求的语义。

### 8.3 generation_key 是什么

```python
generation_key = (
    max_new_tokens,
    round(temperature, 6),
    round(top_p, 6),
)
```

只有 generation key 相同的请求才能进入同一个 batch。

浮点数保留 6 位小数，是为了避免微小的浮点表示差异导致本来相同的参数无法分组。

### 8.4 为什么需要四种预算

`MultimodalBatchPolicy` 限制：

```text
max_batch_size
max_images_per_batch
max_text_tokens_per_batch
max_image_tokens_per_batch
```

原因是不同输入的开销并不能只用“请求条数”表示。

例如：

```text
请求 A：1 张低分辨率图 + 20 token 文本
请求 B：8 张高分辨率图 + 2000 token 文本
```

两者都是“1 条请求”，但 GPU 开销完全不同。

### 8.5 当前 token 预算是怎么算的

文本 token 估算：

```python
estimated_text_tokens = max(1, (len(text) + 3) // 4)
```

即大约用 4 个字符估算 1 个 token。

这不是精确 tokenizer 结果，只是在模型 processor 之前用于 admission 的低成本估算。

图像 token 估算：

```text
low  -> 256
auto -> 1024
high -> 2048
```

这也是调度估算，不是 Qwen processor 已经返回的真实 visual token 数。

而且当前 `detail` 主要用于调度 bucket 和 token 估算，还没有在我们自己的预处理层将图片强制 resize 成不同尺寸。

### 8.6 pop_batch() 怎么选请求

简化逻辑：

```text
1. 取 waiting 队列第一条请求作为 batch 基准
2. 逐个检查后续请求
3. 如果采样参数兼容、图像 bucket 兼容、预算不超限，则加入 batch
4. 否则放入 deferred
5. 达到 max_batch_size 或队列结束后返回 batch
6. deferred 请求保留到下一轮
```

例如队列：

```text
[A-compatible, C-incompatible, B-compatible, D-incompatible]
```

可能得到：

```text
当前 batch = [A, B]
剩余队列 = [C, D]
```

### 8.7 为什么入队时就要检查单请求预算

假设单条请求就有 20 张图，而配置是：

```text
max_images_per_batch = 8
```

如果仍然允许它入队，无论怎么分 batch，它都不可能满足预算。

所以 `add()` 阶段直接拒绝，比让它永久留在队列里更清晰。

### 本章先记住

```text
Batching 的目标是提高吞吐，但会增加等待延迟
采样参数不同的请求不能随意合并
图像请求需要单独的 image token 预算
当前 token 数是 admission 估算，不是精确 processor 输出
```

## 9. 第八步修改：MultimodalEngineCore 的并发模型

### 9.1 先理解 API 线程和 GPU worker 的分工

假设同时有 3 个 HTTP 请求：

```text
API Thread A
API Thread B
API Thread C
```

我们不希望 3 个线程同时调用同一个 HuggingFace model，因为：

```text
GPU 内存和 model.generate() 状态难以保证线程安全
三次单请求 generate 无法自动合成一个 batch
```

所以引入一个唯一的后台 worker：

```text
API A --\
API B ----> waiting queue -> one background worker -> Qwen batch generate
API C --/
```

### 9.2 Event 在这里做什么

每条 API 请求创建一个 `_PendingRequest`：

```python
@dataclass
class _PendingRequest:
    request: MultimodalGenerationRequest
    done: Event
    result: str | None
    error: BaseException | None
```

API 线程入队后执行：

```python
pending.done.wait(timeout)
```

这句话的意思是：

```text
当前 API 线程暂停，等待后台 worker 完成这条请求。
```

后台 worker 写入结果后：

```python
pending.result = output
pending.done.set()
```

`set()` 会唤醒正在 `wait()` 的 API 线程。

所以 Event 解决的是：

```text
“某一条请求的结果好了吗？”
```

### 9.3 Condition 在这里做什么

Condition 同时结合了：

```text
一把保护共享状态的锁
一个可以等待和通知的条件队列
```

它保护的共享状态包括：

```text
waiting batcher
pending request map
shutdown flag
metrics counters
worker 初始化状态
```

当队列为空时，worker 执行：

```python
self._condition.wait()
```

它不会不停循环检查队列，因此不会空转占用 CPU。

新请求入队后：

```python
self._condition.notify()
```

通知 worker：队列现在有数据了。

所以 Condition 解决的是：

```text
“队列状态变了，worker 是不是该继续工作了？”
```

### 9.4 Event 和 Condition 的区别

| 对象 | 关心什么 | 本项目中的用法 |
| --- | --- | --- |
| Event | 一个具体事件是否完成 | 某条请求是否已有结果 |
| Condition | 共享状态是否满足继续执行的条件 | waiting queue 是否非空、batch wait 是否结束 |

### 9.5 batch wait window 如何工作

当第一条请求到达时，worker 不立即调用 GPU，而是计算：

```python
wait_deadline = now + batch_wait_ms
```

假设 `batch_wait_ms = 5`：

```text
t=0.0 ms：A 到达
t=1.0 ms：B 到达
t=3.5 ms：C 到达
t=5.0 ms：等待窗口结束，开始组 batch
```

如果 A/B/C 兼容，它们可能在一次 GPU generate 中执行。

如果窗口设为 0：

```text
单请求等待更少，但 batch size 可能更小。
```

如果窗口设得太长：

```text
batch size 可能更大，但每个请求会先多等一段时间。
```

### 9.6 从 generate() 到返回结果的完整时序

```text
API Thread
  |
  | 创建 MultimodalGenerationRequest
  | 创建 PendingRequest + Event
  | 获取 Condition lock
  | 放入 Batcher
  | 放入 pending map
  | notify worker
  | 释放 Condition lock
  | Event.wait()
  |
  |                         Background Worker
  |                              |
  |                              | 被 notify 唤醒
  |                              | 等待 batch window
  |                              | pop_batch()
  |                              | Manager.prepare_batch()
  |                              | backend.generate_chat_batch()
  |                              | 写入 result/error
  |                              | Event.set()
  |<-----------------------------|
  | 读取 result
  | 返回 HTTP response
```

### 9.7 为什么 pending map 使用内部 UUID

外部 `request_id` 可能由用户提供，也可能因为复用同一个请求对象而重复。

如果 pending map 直接使用它：

```python
pending["request-1"] = pending_a
pending["request-1"] = pending_b
```

第二次会覆盖第一次，导致 A 永远等不到结果。

所以每一次 generation 都额外生成：

```text
mm-internal-<uuid>
```

这是引擎内部的唯一标识，与用户看到的 request ID 分开。

### 9.8 为什么 VLM core 初始化需要锁

假设服务刚启动，同时到达两个图文请求：

```text
Thread A 检查 _multimodal_core is None
Thread B 也检查 _multimodal_core is None
Thread A 创建 core/model
Thread B 也创建 core/model
```

这可能导致重复加载模型，甚至直接 OOM。

因此使用 double-check：

```python
if self._multimodal_core is None:
    with self._multimodal_init_lock:
        if self._multimodal_core is None:
            self._multimodal_core = MultimodalEngineCore(...)
```

外层检查让已初始化后的访问不需要每次加锁。

内层检查用来处理多线程在锁外都看到 None 的情况。

### 9.9 错误如何回到 API 线程

后台 worker 捕获 batch 执行异常：

```python
except BaseException as exc:
    self._finish_batch(batch.requests, error=exc)
```

它会将同一个 batch 中的 pending request 都写入 error，然后 `Event.set()`。

API 线程被唤醒后检查：

```python
if pending.error is not None:
    raise RuntimeError(...) from pending.error
```

最后 FastAPI route 将它转成 HTTP 503。

这样 API 线程不会因为 worker 报错而永久卡在 `Event.wait()`。

### 9.10 当前 timeout 还有什么限制

`generate()` 支持：

```python
pending.done.wait(timeout)
```

如果超时，API 调用者可以收到 `TimeoutError`。

但当前超时并不会将已入队的请求从 batcher 中取消，worker 仍可能完成它。

更完整的设计需要：

```text
CANCELLED 状态
从 waiting queue 移除
正在执行时的取消语义
client disconnect 通知
完成结果丢弃
```

### 本章先记住

```text
API 线程只负责提交和等待
唯一后台 worker 负责真正调用 VLM
Event 用来等待单条请求完成
Condition 用来等待队列状态变化
batch wait window 在延迟和吞吐之间取舍
```

## 10. Qwen2.5-VL 批量生成怎么对齐输入输出

### 10.1 从单请求到批量请求

原来：

```python
generate_chat(messages) -> str
```

现在：

```python
generate_chat_batch(messages_batch, image_inputs_batch, ...) -> list[str]
```

`generate_chat()` 没有被删除，而是改成包装单元素 batch：

```python
return self.generate_chat_batch([messages], ...)[0]
```

这样可以保留单请求 API，同时只维护一份真正的生成逻辑。

### 10.2 prompts 怎么批量构造

```python
prompts = [
    processor.apply_chat_template(messages, ...)
    for messages in messages_batch
]
```

如果 batch 大小为 3，就得到：

```python
prompts = [prompt_a, prompt_b, prompt_c]
```

### 10.3 为什么要展平 image_inputs_batch

Manager 的输出保留了请求边界：

```python
image_inputs_batch = [
    [a_image_1, a_image_2],
    [b_image_1],
    [c_image_1, c_image_2],
]
```

Qwen processor 调用使用展平列表：

```python
images = [
    a_image_1,
    a_image_2,
    b_image_1,
    c_image_1,
    c_image_2,
]
```

对齐依赖两个顺序一致：

```text
prompts 顺序 = A, B, C
images 展平顺序 = A 的所有图, B 的所有图, C 的所有图
```

每个 prompt 里的视觉占位符数量告诉 processor 应该消费多少张图。

这部分已经按 Qwen 批处理方式编写，但仍需在服务器上使用真实 `AutoProcessor` 验证多图、不同图像数量 batch 的对齐行为。

### 10.4 为什么检查输出数量

```python
if len(outputs) != len(batch.requests):
    raise RuntimeError(...)
```

如果提交 4 条请求，后端只返回 3 条输出，我们不知道缺少的是哪一条。

此时不能将 3 条结果强行分配给前 3 个用户，而应该让整个 batch 明确失败。

### 本章先记住

```text
单请求 API 可以通过单元素 batch 复用批量实现
多图 batch 的核心是 prompt 顺序与 image flatten 顺序一致
输出数量必须与请求数量一致
```

## 11. Benchmark 脚本在学什么

### 11.1 不要只看“能否返回文本”

多模态服务需要至少关心：

```text
请求是否成功
总延迟
P50 / P90 / P99 延迟
每秒成功请求数
真实平均 batch size
图像缓存 hit/miss
错误数和错误类型
```

### 11.2 P50、P90、P99 是什么

假设 10 条请求的延迟排序后是：

```text
100, 105, 110, 115, 120, 130, 150, 180, 300, 800 ms
```

```text
P50 大约反映普通用户的经历
P90 反映较慢的 10% 请求开始处在什么水平
P99 反映尾延迟，容易暴露排队、缓存 miss 和慢图片下载问题
```

只看平均值可能掩盖 800 ms 这样的尾部请求。

### 11.3 为什么要 warmup

第一个请求可能包含：

```text
延迟加载模型
CUDA context 初始化
kernel 首次编译
allocator 预热
图像缓存初次 miss
```

如果把它直接算入稳态性能，会让结果难以与其它配置比较。

所以 benchmark 先执行 warmup，再开始正式计时。

### 11.4 为什么本地图片要自动转 base64

假设 benchmark 在你的笔记本上运行，服务在远程 GPU 服务器上运行。

如果发送：

```text
./assets/demo.jpg
```

远程服务器不存在你笔记本的这个路径。

因此 benchmark 在客户端读取文件，转成：

```text
data:image/jpeg;base64,...
```

再放进 HTTP JSON 发送。

### 11.5 为什么 benchmark 还要读取服务端 metrics

客户端只能看到：

```text
发送时间
响应时间
响应是否成功
```

它不知道：

```text
32 条请求到底执行了多少个 batch
平均 batch size 是 1.2 还是 6.5
图像缓存是否真正命中
```

所以 benchmark 完成后还请求：

```text
/metrics/multimodal
```

获得 engine 内部的 batch 和 cache 统计。

### 11.6 为什么 TTFT 和 TPOT 是 null

当前 API 不支持 streaming。

客户端只在模型生成完整段文本后收到一个 JSON，无法知道第一个 token 是什么时候产生的。

因此报告明确写：

```json
{
  "ttft_sec": null,
  "tpot_sec": null
}
```

这比用总延迟猜测或伪造 TTFT 更正确。

### 11.7 怎么运行 benchmark

```bash
python bench_multimodal.py \
  --server-url http://127.0.0.1:8000 \
  --model qwen2.5-vl \
  --image ./assets/demo.jpg \
  --prompt "请描述图中的物体和颜色。" \
  --num-requests 32 \
  --concurrency 8 \
  --max-tokens 128 \
  --output-json benchmark-results/qwen25-vl.json
```

重复使用 `--image` 可以构造多图请求：

```bash
--image image1.jpg --image image2.jpg
```

### 本章先记住

```text
平均值不能代替 P90/P99
warmup 用来隔离首次初始化开销
客户端延迟和服务端 batch/cache 指标需要一起看
没有 streaming 就不应该声称测得了真实 TTFT/TPOT
```

## 12. 开发中遇到的典型问题：从问题学设计

### 12.1 问题：重构后谁负责关闭子进程

错误倾向：

```text
LLMEngine 仍保留一份 process 列表
Executor 又保留一份 process 列表
两层都可以 shutdown
```

这会导致资源所有权不清晰，甚至重复退出或重复 join。

当前解决：

```text
Executor 拥有 process
Executor.shutdown() 负责回收
EngineCore.shutdown() 只做委托
LLMEngine.exit() 只做委托
```

学到的原则：

```text
谁创建和拥有资源，谁负责释放资源。
```

### 12.2 问题：用 slots=True 的 dataclass 没有 __dict__

`MultimodalManagerStats` 使用：

```python
@dataclass(slots=True)
```

最初如果在 API 里写：

```python
stats.manager.__dict__
```

会失败，因为 slots dataclass 不为每个实例保留普通 `__dict__`。

解决：

```python
from dataclasses import asdict

asdict(stats.manager)
```

学到的原则：

```text
slots 可以减少轻量状态对象的内存开销，但不能默认依赖 __dict__ 做序列化。
```

### 12.3 问题：外部 request ID 覆盖 pending request

已在第 9 章讲过，核心错误是将“用户可见 ID”当成“引擎内部唯一 ID”。

解决：每次 generation 生成独立 internal UUID。

学到的原则：

```text
用户提供的标识不应默认被信任为内部唯一主键。
```

### 12.4 问题：第一批并发请求重复初始化 VLM

根因是 lazy initialization 没有线程保护。

解决：double-check + initialization lock。

学到的原则：

```text
“延迟初始化”和“并发初始化”必须一起考虑。
```

### 12.5 问题：只用一把锁保护 Qwen 会导致完全串行

最初方式：

```python
with multimodal_lock:
    backend.generate_chat(...)
```

它保证了正确性，但每次只处理一条请求。

新方式：

```text
多个 API 线程并发入队
单 worker 将它们合并成 batch
单 worker 仍保证模型调用不并发冲突
```

学到的原则：

```text
串行化“模型执行者”不等于每次只能执行“单条请求”。
可以用队列将多个请求合并为一次串行的 batch 调用。
```

### 12.6 问题：temperature=0 与文本 sampler 冲突

Qwen HuggingFace backend 可以在 temperature 接近 0 时使用 greedy：

```python
do_sample = temperature > 1e-5
```

但原 nano-vLLM 的 sampler 会执行：

```python
logits / temperature
```

并且 `SamplingParams` 明确禁止 temperature 为 0。

如果 chat schema 允许 0，纯文本 chat 路径会在更深层才失败。

当前解决：共用 chat schema 仍限制 temperature 大于 0。

更好的后续方案：

```text
在原文本 Sampler 中正式实现 greedy 分支
或为文本与 VLM 使用更精确的采样能力描述
```

### 12.7 问题：本机没有 transformers 和 pydantic

这不是代码逻辑错误，而是开发环境与目标服务器环境不同。

我们没有在本机强行下载大模型，而是分层验证：

```text
compileall：检查语法
fake backend：检查队列、batching、Event 和错误路径
1x1 base64 PNG：检查 PIL image cache hit/miss
服务器：留给真实 FastAPI、transformers 和 GPU 集成验证
```

学到的原则：

```text
不具备完整环境时，仍然可以把纯逻辑与硬件执行分开验证；
但必须明确说明哪些没有真实跑过。
```

### 12.8 问题：普通 git push 无法访问 GitHub

根因是开发 sandbox 默认限制网络，命令报错：

```text
Could not resolve host: github.com
```

解决方式是在获得授权后使用可访问网络的执行环境重试。

这个问题与项目代码无关，但是开发与发布环境的一部分。

## 13. 请手动跟一次完整图文请求

请求：

```json
{
  "model": "qwen2.5-vl",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/png;base64,...",
            "detail": "low"
          }
        },
        {
          "type": "text",
          "text": "图中有什么？"
        }
      ]
    }
  ],
  "max_tokens": 128,
  "temperature": 0.2,
  "top_p": 1.0
}
```

### 第 1 步：Pydantic 校验

```text
messages 非空
content part type 合法
max_tokens >= 1
temperature > 0
0 < top_p <= 1
```

### 第 2 步：Adapter 转换

```text
image_url -> ImageInput(source_type="base64")
检查 base64 合法性
构造 Qwen content: type="image"
保留图在前、文本在后的顺序
```

### 第 3 步：EngineClient 获取 MultimodalEngineCore

```text
如果是第一次请求：
  获取 initialization lock
  创建 Qwen backend
  创建 policy
  创建 MultimodalEngineCore
```

### 第 4 步：创建内部请求

```text
保存 max_new_tokens=128
保存 temperature=0.2
保存 top_p=1.0
生成 internal UUID
估算 text tokens
估算 image tokens=256
```

### 第 5 步：入队与等待

```text
Batcher.add()
pending map 记录 Event
notify worker
API 线程 Event.wait()
```

### 第 6 步：Worker 组 batch

```text
等待 batch_wait_ms
选择采样参数和 image bucket 兼容的请求
检查四种预算
生成 MultimodalBatch
```

### 第 7 步：Manager 准备图像

```text
根据 base64 内容生成 SHA256 key
查 ImageAssetCache
命中：直接取 PIL image
未命中：解码 -> RGB PIL image -> 放入 cache
```

### 第 8 步：Qwen 批量执行

```text
apply_chat_template
flatten images
AutoProcessor
model.generate
trim prompt IDs
batch_decode
```

### 第 9 步：返回结果

```text
worker 将 outputs[i] 写入对应 pending request
Event.set()
API 线程被唤醒
FastAPI 构造 ChatCompletionResponse
```

当你能不看文档说出这 9 步，就已经理解当前多模态执行路径的主体了。

## 14. 当前项目还没做完什么

### 14.1 真实 vision feature cache

当前只缓存 PIL image。

下一步需要：

```text
找到 Qwen2.5-VL vision encoder 入口
将图像 processor 结果和 vision forward 拆出
缓存 visual embeddings
设计 dtype/device/cache capacity
让 language model 路径接收缓存 feature
```

### 14.2 异步图像加载

当前 Manager 在唯一 VLM worker 内执行 URL 下载和 Pillow 解码。

如果某个 URL 需要 3 秒才返回，后面的所有 batch 都可能被阻塞。

更好的设计：

```text
API admission
  -> async download pool
  -> image decode/preprocess pool
  -> ready queue
  -> VLM batch scheduler
```

### 14.3 token 级 continuous batching

当前多模态 batching 是：

```text
一批请求一起进入 HuggingFace generate
整批生成完才返回
```

更成熟的 VLM serving 应该允许：

```text
正在 decode 的请求继续运行
新请求动态进入 prefill
每个 step 重新组织 batch
完成的请求立即离开
```

这才是更接近 vLLM 的 continuous batching。

### 14.4 Streaming SSE

没有 streaming，用户必须等待整段回答生成完才能看到文本。

增加 streaming 后，还需解决：

```text
每个 token 如何从 worker 传给 API
如何维护每个 request 的输出队列
客户端断开时如何取消
如何准确记录 TTFT/TPOT
```

### 14.5 真实 GPU 数据

代码已完成静态编译和 fake backend smoke test，但还需要真实测量：

```text
batch size 1/2/4/8 吞吐
P50/P90/P99 延迟
GPU 显存峰值
不同图像分辨率
不同图像数量
cache hit/miss 对总延迟的影响
batch wait window 对延迟与吞吐的影响
```

## 15. 建议的学习顺序

### 第 1 轮：理解原始文本引擎

按顺序阅读：

```text
example.py
nanovllm/llm.py
nanovllm/engine/llm_engine.py
nanovllm/engine/sequence.py
nanovllm/engine/scheduler.py
nanovllm/engine/block_manager.py
nanovllm/engine/model_runner.py
```

学习目标：

```text
能画出 prefill/decode 主循环
能说出 Sequence 状态变化
能解释 KV block table
能解释为什么 decode 每轮只输入 last token
```

### 第 2 轮：理解引擎分层

按顺序阅读：

```text
nanovllm/engine/engine_core.py
nanovllm/engine/executor.py
nanovllm/engine/gpu_worker.py
docs/ARCHITECTURE.md
```

学习目标：

```text
能说出每层的所有权边界
能解释 Executor 为什么在 Scheduler 前初始化
能解释 process.join()
```

### 第 3 轮：理解服务和请求格式

按顺序阅读：

```text
nanovllm/serving/schemas.py
nanovllm/serving/openai_api.py
nanovllm/serving/engine_client.py
nanovllm/multimodal/openai_adapter.py
```

学习目标：

```text
能将 OpenAI image_url part 手动转成 Qwen image part
能说出为什么要单独设置 adapter
能解释本地文件默认关闭的原因
```

### 第 4 轮：理解多模态 batching

按顺序阅读：

```text
nanovllm/multimodal/request.py
nanovllm/multimodal/scheduler_policy.py
nanovllm/multimodal/batching.py
nanovllm/multimodal/manager.py
nanovllm/engine/multimodal_engine.py
nanovllm/multimodal/qwen25_vl.py
```

学习目标：

```text
能判断两条请求能否合并
能解释 Event 和 Condition 的区别
能手动跟踪一条请求的 pending 状态
能区分 asset cache 和 feature cache
```

### 第 5 轮：理解性能评价

按顺序阅读：

```text
nanovllm/engine/metrics.py
bench_multimodal.py
docs/MODIFICATION_PLAN.md
```

学习目标：

```text
能手动计算 TTFT 和 TPOT
能解释 P50/P90/P99
能设计 batch size 和 cache hit/miss 对比实验
```

## 16. 练习题

### 基础理解

1. 为什么 prefill 可以一次处理多个 token，而 decode 通常每轮每条 Sequence 只处理一个 token？
2. `Sequence.num_cached_tokens` 和 `num_scheduled_tokens` 有什么区别？
3. 为什么 `Scheduler` 需要了解 KV cache block 数量？
4. `EngineCore` 为什么不应该直接导入 Qwen3 模型类？

### Batching

5. 两条请求的 `temperature` 不同，当前为什么不能合并？
6. 如果关闭 image bucketization，可能带来什么好处和风险？
7. 为什么要区分 `max_images_per_batch` 和 `max_image_tokens_per_batch`？
8. `batch_wait_ms` 从 5 ms 改成 20 ms，你预期 batch size、P50 和 throughput 如何变化？

### Cache

9. 两个不同 URL 返回完全相同的图像，当前 URL cache 能否命中？
10. 同一 URL 的图像内容变了，当前 cache 可能出现什么问题？
11. 如果要真正缓存 vision feature，cache key 除了图像内容外还应包含哪些因素？提示：模型版本、processor 参数、dtype。

### 并发

12. Event 为什么是每个 pending request 一个，而不是全局共享一个？
13. 如果 worker 生成完结果但忘记 `Event.set()`，API 会表现成什么样？
14. 为什么不能只写 `if queue is empty: condition.wait()`，而通常要在 `while` 里重新检查条件？

### 工程延伸

15. 如果要增加 video input，你会修改哪些 schema、adapter、budget 和 manager 对象？
16. 如果要支持请求取消，`_PendingRequest` 和 batcher 需要增加什么状态？
17. 如果要测量真实 TTFT，API、worker 和 benchmark 分别要改什么？

## 17. 面试时怎么讲这个项目

可以按“问题 -> 设计 -> 当前结果 -> 局限”来讲。

### 例子

```text
原 nano-vLLM 是一个紧凑的离线文本推理实现，LLMEngine 同时负责调度、
GPU 执行和多进程生命周期。我将它拆分为 EngineCore、Executor 和 GPUWorker，
保留原有 LLM.generate API，并在 Sequence 和 Scheduler 中增加 TTFT、TPOT、
KV cache 和调度指标。

多模态方面，我先通过 HuggingFace 接入 Qwen2.5-VL，再将 OpenAI 图文请求适配、
图像对象 LRU 缓存、采样参数兼容分组、图文 token 预算和请求级微批处理
下沉到 MultimodalEngineCore。API 线程通过 Event 等待结果，后台 worker 通过
Condition 聚合请求并执行 Qwen batch generate。

当前已完成静态检查、fake backend batching 和 image cache smoke test。局限是尚未在
GPU 服务器完成真模型 batch 验证，且目前缓存的是 decoded image，还不是
vision encoder feature。
```

这样讲的好处是：

```text
不会把原项目的核心算子说成自己实现
能体现你理解了架构和并发设计
能准确区分已实现功能和待验证功能
面试官可以继续深挖 batch、cache、KV cache 或 metrics
```

## 18. 术语表

| 术语 | 简单解释 |
| --- | --- |
| Token | 模型处理文本的基本离散单位 |
| Prefill | 处理 prompt 并建立 KV cache 的阶段 |
| Decode | 利用 KV cache 逐 token 生成的阶段 |
| KV cache | 缓存 attention 历史 Key/Value，避免重算历史 token |
| Sequence | 一条推理请求的内部状态对象 |
| Scheduler | 决定每一步执行哪些 Sequence |
| Chunked prefill | 将长 prompt 分成多个 step 执行 |
| Continuous batching | 引擎运行中动态加入和移出请求 |
| Micro-batching | 等待很短时间，将同时到达的请求组成一批 |
| TTFT | Time To First Token，从请求到达到首 token 的时间 |
| TPOT | Time Per Output Token，首 token 之后每个输出 token 的平均时间 |
| Tail latency | 较慢的少数请求的延迟，常用 P90/P99 表示 |
| Vision encoder | 将图像 patch 转成 visual embeddings 的模型部分 |
| Asset cache | 缓存已加载/解码的图像对象 |
| Feature cache | 缓存 vision encoder 输出 |
| LRU | 容量超限时淘汰最久未使用元素的缓存策略 |
| Event | 用于等待某一个具体事件完成的线程同步对象 |
| Condition | 用于等待共享状态满足某个条件的线程同步对象 |
| Tensor parallel | 将同一个模型层的 tensor 计算拆到多张 GPU |

## 19. 阅读完成标准

当你能独立回答下列问题，可以认为已经理解当前项目的主体：

```text
1. 一条文本请求如何从 LLM.generate 走到 ModelRunner？
2. EngineCore、Executor 和 GPUWorker 分别拥有什么？
3. Scheduler 如何在 prefill 和 decode 之间选择？
4. TTFT 和 TPOT 在代码的哪个时间点记录？
5. OpenAI image_url 请求如何转成 Qwen message？
6. 为什么本地文件访问默认关闭？
7. Asset cache 和 feature cache 分别节省哪些计算？
8. 两条多模态请求能否进入同一 batch 由哪些条件决定？
9. Event 和 Condition 分别在解决什么同步问题？
10. 当前项目为什么还不能声称已实现 vision feature cache？
```

详细的文件变更、Git 提交和问题状态索引可以继续查看：

```text
docs/IMPLEMENTATION_RETROSPECTIVE.md
```
