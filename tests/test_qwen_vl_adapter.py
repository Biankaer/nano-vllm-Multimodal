import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import torch
from safetensors.torch import save_file

from nanovllm.models.qwen_vl_adapter import (
    Qwen25VLAdapter,
    Qwen3VLAdapter,
    resolve_qwen_vl_adapter,
    audit_qwen3_vl_checkpoint,
)


class QwenVLAdapterTests(unittest.TestCase):
    @staticmethod
    def write_shard(directory, filename, keys):
        save_file(
            {key: torch.zeros(1) for key in keys},
            str(Path(directory) / filename),
        )

    def test_qwen3_checkpoint_audit_accepts_complete_known_namespaces(self):
        with TemporaryDirectory() as directory:
            self.write_shard(directory, "model.safetensors", (
                "model.language_model.embed_tokens.weight",
                "model.visual.patch_embed.proj.weight",
                "lm_head.weight",
            ))
            report = audit_qwen3_vl_checkpoint(directory)
        self.assertEqual(report.language_weights, 1)
        self.assertEqual(report.visual_weights, 1)

    def test_qwen3_checkpoint_audit_rejects_unknown_root(self):
        with TemporaryDirectory() as directory:
            self.write_shard(directory, "model.safetensors", (
                "model.language_model.embed_tokens.weight",
                "model.visual.patch_embed.proj.weight",
                "unexpected.weight",
            ))
            with self.assertRaisesRegex(ValueError, "unknown Qwen3-VL checkpoint weight"):
                audit_qwen3_vl_checkpoint(directory)

    def test_qwen3_checkpoint_audit_rejects_cross_file_duplicate(self):
        with TemporaryDirectory() as directory:
            duplicate = "model.language_model.embed_tokens.weight"
            self.write_shard(directory, "a.safetensors", (
                duplicate, "model.visual.patch_embed.proj.weight",
            ))
            self.write_shard(directory, "b.safetensors", (duplicate,))
            with self.assertRaisesRegex(ValueError, "appears in multiple files"):
                audit_qwen3_vl_checkpoint(directory)

    def test_qwen3_checkpoint_audit_requires_language_and_visual_partitions(self):
        cases = (
            (("model.visual.patch_embed.proj.weight",), "language"),
            (("model.language_model.embed_tokens.weight",), "visual"),
        )
        for keys, missing in cases:
            with self.subTest(missing=missing), TemporaryDirectory() as directory:
                self.write_shard(directory, "model.safetensors", keys)
                with self.assertRaisesRegex(ValueError, f"missing.*{missing}"):
                    audit_qwen3_vl_checkpoint(directory)
    def test_resolves_supported_model_types_and_rejects_unknown(self):
        self.assertIsInstance(
            resolve_qwen_vl_adapter(SimpleNamespace(model_type="qwen2_5_vl")),
            Qwen25VLAdapter,
        )
        self.assertIsInstance(
            resolve_qwen_vl_adapter(SimpleNamespace(model_type="qwen3_vl")),
            Qwen3VLAdapter,
        )
        with self.assertRaisesRegex(ValueError, "unsupported.*unknown"):
            resolve_qwen_vl_adapter(SimpleNamespace(model_type="unknown"))

    def test_qwen3_language_loader_selects_only_language_and_lm_head(self):
        adapter = Qwen3VLAdapter(SimpleNamespace(text_config=object()))
        model = object()
        with patch("nanovllm.models.qwen_vl_adapter.load_model") as loader:
            adapter.load_language_weights(model, "/checkpoint")
        loader.assert_called_once_with(
            model,
            "/checkpoint",
            include_prefixes=("model.language_model.", "lm_head."),
        )

    def test_qwen25_language_loader_preserves_existing_prefix(self):
        adapter = Qwen25VLAdapter(SimpleNamespace(text_config=object()))
        with patch("nanovllm.models.qwen_vl_adapter.load_model") as loader:
            adapter.load_language_weights(object(), "/checkpoint")
        self.assertEqual(loader.call_args.kwargs["include_prefixes"], ("model.",))

    def test_qwen3_processor_import_is_lazy_until_creation(self):
        adapter = Qwen3VLAdapter(SimpleNamespace(text_config=object()))
        with patch(
            "nanovllm.multimodal.qwen3_vl_processor.Qwen3VLPromptProcessor.from_pretrained",
            return_value="processor",
        ) as factory:
            self.assertEqual(adapter.create_prompt_processor("/model"), "processor")
        factory.assert_called_once_with("/model")

    def test_adapters_declare_model_specific_vision_metadata(self):
        qwen25 = Qwen25VLAdapter(SimpleNamespace(text_config=object()))
        qwen3 = Qwen3VLAdapter(SimpleNamespace(text_config=object()))
        self.assertEqual(qwen25.default_vision_attention_backend, "sdpa")
        self.assertEqual(qwen25.deepstack_layer_count, 0)
        self.assertEqual(qwen3.default_vision_attention_backend, "flash_attention_2")
        self.assertEqual(qwen3.deepstack_layer_count, 3)


if __name__ == "__main__":
    unittest.main()
