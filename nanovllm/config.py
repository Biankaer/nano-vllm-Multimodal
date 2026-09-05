# os 用来检查模型路径是否存在。
import os
import json

# dataclass 自动生成 __init__、__repr__ 等方法，让配置类保持轻量。
from dataclasses import dataclass

# AutoConfig 从 HuggingFace 模型目录读取 config.json，得到 hidden_size、层数、头数等结构参数。
from transformers import AutoConfig
from nanovllm.multimodal.dependencies import require_transformers_version


def read_local_model_type(model_path: str) -> str | None:
    config_path = os.path.join(model_path, "config.json")
    try:
        with open(config_path, encoding="utf-8") as config_file:
            config = json.load(config_file)
    except FileNotFoundError as error:
        raise ValueError(f"model directory is missing config.json: {model_path}") from error
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"model directory has invalid config.json: {model_path}") from error
    if not isinstance(config, dict):
        raise ValueError(f"model directory has invalid config.json object: {model_path}")
    model_type = config.get("model_type")
    return model_type if isinstance(model_type, str) else None


def effective_text_config(hf_config):
    text_config = getattr(hf_config, "text_config", None)
    return text_config if text_config is not None else hf_config


def resolve_vision_attn_implementation(model_type: str | None, requested: str) -> str:
    supported = {"auto", "sdpa", "flash_attention_2"}
    if requested not in supported:
        raise ValueError(
            "vision_attn_implementation must be one of "
            "'auto', 'sdpa', or 'flash_attention_2'"
        )
    if requested != "auto":
        return requested
    if model_type == "qwen3_vl":
        return "flash_attention_2"
    return "sdpa"


def resolve_language_attn_implementation(model_type: str | None, requested: str) -> str:
    supported = {"auto", "sdpa", "flash_attention_2"}
    if requested not in supported:
        raise ValueError(
            "language_attn_implementation must be one of "
            "'auto', 'sdpa', or 'flash_attention_2'"
        )
    if requested != "auto":
        if requested == "sdpa" and model_type != "qwen3_vl":
            raise ValueError("language SDPA is currently implemented only for Qwen3-VL")
        return requested
    return "flash_attention_2"


# slots=True 会让实例不再使用 __dict__，字段固定，内存更省，也能避免写错字段名。
@dataclass(slots=True)
class Config:
    # 模型本地目录，里面通常包含 config.json、tokenizer 文件、safetensors 权重等。
    model: str

    # 一次 prefill 调度最多处理多少 token；它限制的是 token 总数，不是请求条数。
    max_num_batched_tokens: int = 16384

    # 一次调度最多同时处理多少条 Sequence；decode 阶段通常受这个限制更明显。
    max_num_seqs: int = 512

    # 引擎允许的最大上下文长度，会在 __post_init__ 中和模型最大位置长度取 min。
    max_model_len: int = 4096

    # 允许 KV cache 使用的 GPU 显存比例；越大可容纳请求越多，但留给临时张量的空间越小。
    gpu_memory_utilization: float = 0.9

    # 张量并行 GPU 数量。1 表示单卡；大于 1 时会启动多个 ModelRunner 进程。
    tensor_parallel_size: int = 1

    # True 表示强制 eager 执行；False 时 decode 可尝试使用 CUDA graph 提速。
    enforce_eager: bool = False

    # HuggingFace AutoConfig 对象，初始化后填充；用户一般不需要直接传。
    hf_config: AutoConfig | None = None

    # tokenizer 的 EOS token id，LLMEngine 初始化 tokenizer 后会写入。
    eos: int = -1

    # KV cache 的物理块大小。Sequence 会按这个粒度映射到 block table。
    kvcache_block_size: int = 256

    # KV cache 总块数，由 ModelRunner 根据显存动态计算后写入。
    num_kvcache_blocks: int = -1

    # 原生视觉特征缓存按 GPU 常驻字节数限制，而不是按图片条目数限制。
    vision_feature_cache_bytes: int = 2 * 1024**3

    # auto 会按模型类型选择原生视觉注意力后端。
    vision_attn_implementation: str = "auto"

    # Qwen3-VL 可显式选择 HF-aligned SDPA；默认保留 fused/paged Flash 生产路径。
    language_attn_implementation: str = "auto"

    # Paged KV data movement backend: PyTorch reference, Triton, or C++/CUDA.
    paged_kv_backend: str = "auto"

    def __post_init__(self):
        # 模型必须是本地目录；这个实现不直接支持传 HuggingFace repo id 在线加载。
        assert os.path.isdir(self.model)

        # block_size 必须是 256 的倍数，和 FlashAttention paged KV cache 的块粒度约束对齐。
        assert self.kvcache_block_size % 256 == 0

        # 简单限制 tensor parallel 规模，避免误传过大的并行数。
        assert 1 <= self.tensor_parallel_size <= 8

        if self.paged_kv_backend not in {"auto", "pytorch", "triton", "cuda"}:
            raise ValueError(
                "paged_kv_backend must be one of: auto, pytorch, triton, cuda"
            )

        local_model_type = read_local_model_type(self.model)
        if local_model_type == "qwen3_vl":
            require_transformers_version()

        # 读取模型结构配置，例如层数、头数、dtype、最大位置长度等。
        self.hf_config = AutoConfig.from_pretrained(self.model)
        model_type = getattr(self.hf_config, "model_type", None)
        text_config = effective_text_config(self.hf_config)
        self.vision_attn_implementation = resolve_vision_attn_implementation(
            model_type,
            self.vision_attn_implementation,
        )
        self.language_attn_implementation = resolve_language_attn_implementation(
            model_type,
            self.language_attn_implementation,
        )

        if self.language_attn_implementation == "sdpa" and not self.enforce_eager:
            raise ValueError("language SDPA currently requires enforce_eager=True")

        if self.vision_feature_cache_bytes <= 0:
            raise ValueError("vision feature cache byte budget must be positive")

        if model_type in {"qwen2_5_vl", "qwen3_vl"} and self.tensor_parallel_size != 1:
            raise ValueError(
                f"native {model_type} currently requires tensor_parallel_size=1"
            )

        # 用户传入的 max_model_len 不能超过模型 config 中声明的最大位置长度。
        self.max_model_len = min(self.max_model_len, text_config.max_position_embeddings)
