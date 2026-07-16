# os 用来处理本地路径，这里主要用 expanduser 把 "~" 展开成用户主目录。
import os

# LLM 是 nano-vLLM 对外暴露的推理入口，SamplingParams 用来描述采样/生成参数。
from nanovllm import LLM, SamplingParams

# HuggingFace tokenizer 负责把文本 prompt 转成 token ids，也负责应用 chat template。
from transformers import AutoTokenizer


def main():
    # 模型目录。README 中推荐先把 Qwen3-0.6B 下载到这个路径。
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")

    # 加载 HuggingFace tokenizer；nano-vLLM 内部也会加载 tokenizer，
    # 这里额外加载一份是为了在示例中手动构造 chat 格式 prompt。
    tokenizer = AutoTokenizer.from_pretrained(path)

    # 创建 nano-vLLM 推理引擎。
    # enforce_eager=True 表示禁用 CUDA graph，初学/调试时更直观；
    # tensor_parallel_size=1 表示只用 1 张 GPU，不做张量并行切分。
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    # 设置采样参数：
    # temperature=0.6 会让分布比 1.0 更尖锐，输出更稳定；
    # max_tokens=256 表示每个 prompt 最多生成 256 个新 token。
    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)

    # 原始用户问题。此时还不是模型最终看到的 chat prompt。
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]

    # 把普通用户文本包进 Qwen chat template。
    # apply_chat_template 会生成类似 "<|im_start|>user ... assistant" 的模型输入格式。
    prompts = [
        tokenizer.apply_chat_template(
            # 单轮对话：只有一个 user 消息。
            [{"role": "user", "content": prompt}],
            # tokenize=False 表示先返回字符串，而不是直接返回 token ids。
            tokenize=False,
            # add_generation_prompt=True 会追加 assistant 起始标记，提示模型开始回答。
            add_generation_prompt=True,
        )
        # 对上面 prompts 列表中的每条原始 prompt 都做 chat template 转换。
        for prompt in prompts
    ]

    # 调用 nano-vLLM 批量生成。返回值是和 prompts 等长的列表。
    outputs = llm.generate(prompts, sampling_params)

    # 将输入 prompt 和生成结果一一打印，方便观察模型实际输出。
    for prompt, output in zip(prompts, outputs):
        # 打印空行，让不同样例的输出隔开。
        print("\n")
        # !r 使用 repr 格式，能看到换行符、特殊 token 等字符串细节。
        print(f"Prompt: {prompt!r}")
        # output["text"] 是 tokenizer.decode 后的生成文本。
        print(f"Completion: {output['text']!r}")


# Python 脚本入口保护：只有直接运行 example.py 时才执行 main()。
# 如果这个文件被 import，main() 不会自动跑。
if __name__ == "__main__":
    main()
