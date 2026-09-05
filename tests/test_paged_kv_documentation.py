import unittest
from pathlib import Path


class PagedKVDocumentationTests(unittest.TestCase):
    def test_readme_documents_chinese_runtime_and_paged_kv_contract(self):
        readme = Path("README.md").read_text(encoding="utf-8")

        required = (
            "## 当前进度",
            "## 原生多模态运行时",
            "## Paged KV Store/Gather 后端",
            "## Qwen3-VL 正确性与性能验证",
            "paged_kv_backend",
            "pytorch / triton / cuda / auto",
            "bench_paged_kv.py",
            "scripts/profile_paged_kv_ncu.py",
            "NANOVLLM_PAGED_KV_VALIDATE_INDICES=1",
            "FlashAttention：仅写入",
            "SDPA：写入 + 聚合",
            "not_tested_no_rocm_toolchain",
            "ERR_NVGPUCTRPERM",
        )
        for text in required:
            with self.subTest(text=text):
                self.assertIn(text, readme)


if __name__ == "__main__":
    unittest.main()
