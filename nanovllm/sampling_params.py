# dataclass 用来快速定义“只保存参数”的轻量类。
from dataclasses import dataclass


# slots=True 固定字段集合，减少对象开销，也避免运行时随手加不存在的字段。
@dataclass(slots=True)
class SamplingParams:
    # 采样温度。0 表示 greedy；正值越低越保守，越高越随机。
    temperature: float = 1.0

    # 每条请求最多生成多少个新 token，不包含 prompt 本身。
    max_tokens: int = 64

    # False 表示遇到 eos token 就停止；True 表示忽略 eos，直到达到 max_tokens。
    ignore_eos: bool = False

    def __post_init__(self):
        if self.temperature < 0:
            raise ValueError("temperature cannot be negative")
