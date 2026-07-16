# LLMEngine 包含真正的推理引擎实现；这个文件只提供对外更简洁的类名 LLM。
from nanovllm.engine.llm_engine import LLMEngine


# LLM 继承 LLMEngine 但不新增逻辑。
# 这样用户可以写 from nanovllm import LLM，API 名字和 vLLM 更接近。
class LLM(LLMEngine):
    # pass 表示类体为空，所有方法都直接继承自 LLMEngine。
    pass
