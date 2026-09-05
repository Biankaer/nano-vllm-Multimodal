import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from safetensors.torch import save_file

from nanovllm.utils.loader import load_model


class SelectiveWeightLoaderTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_loads_only_included_prefix_and_strips_prefix(self):
        module = torch.nn.Linear(2, 2, bias=False)
        save_file(
            {
                "visual.weight": torch.full((2, 2), 3.0),
                "model.weight": torch.zeros(2, 2),
            },
            str(self.path / "model.safetensors"),
        )

        report = load_model(
            module,
            str(self.path),
            include_prefixes=("visual.",),
            strip_prefix="visual.",
        )

        torch.testing.assert_close(module.weight, torch.full((2, 2), 3.0))
        self.assertEqual(report.loaded_parameters, ("weight",))
        self.assertEqual(report.skipped_checkpoint_weights, ("model.weight",))

    def test_rejects_missing_selected_model_parameter(self):
        module = torch.nn.Linear(2, 2, bias=True)
        save_file(
            {"visual.weight": torch.ones(2, 2)},
            str(self.path / "model.safetensors"),
        )

        with self.assertRaisesRegex(ValueError, "bias"):
            load_model(
                module,
                str(self.path),
                include_prefixes=("visual.",),
                strip_prefix="visual.",
            )

    def test_rejects_unknown_selected_checkpoint_weight(self):
        module = torch.nn.Linear(2, 2, bias=False)
        save_file(
            {
                "visual.weight": torch.ones(2, 2),
                "visual.unknown": torch.ones(1),
            },
            str(self.path / "model.safetensors"),
        )

        with self.assertRaisesRegex(KeyError, "visual.unknown"):
            load_model(
                module,
                str(self.path),
                include_prefixes=("visual.",),
                strip_prefix="visual.",
            )

    def test_declared_alias_loads_a_shared_parameter_once(self):
        class SharedWeightModule(nn.Module):
            checkpoint_parameter_aliases = {
                "head.weight": "embedding.weight",
            }

            def __init__(self):
                super().__init__()
                self.embedding = nn.Linear(2, 2, bias=False)
                self.head = nn.Linear(2, 2, bias=False)
                self.head.weight = self.embedding.weight

        module = SharedWeightModule()
        weight = torch.full((2, 2), 7.0)
        save_file(
            {
                "embedding.weight": weight,
                "head.weight": weight.clone(),
            },
            str(self.path / "model.safetensors"),
        )

        report = load_model(module, str(self.path))

        torch.testing.assert_close(module.embedding.weight, weight)
        self.assertEqual(
            report.loaded_parameters,
            ("embedding.weight", "head.weight"),
        )

    def test_declared_alias_rejects_conflicting_checkpoint_values(self):
        class SharedWeightModule(nn.Module):
            checkpoint_parameter_aliases = {
                "head.weight": "embedding.weight",
            }

            def __init__(self):
                super().__init__()
                self.embedding = nn.Linear(2, 2, bias=False)
                self.head = nn.Linear(2, 2, bias=False)
                self.head.weight = self.embedding.weight

        module = SharedWeightModule()
        save_file(
            {
                "embedding.weight": torch.ones(2, 2),
                "head.weight": torch.zeros(2, 2),
            },
            str(self.path / "model.safetensors"),
        )

        with self.assertRaisesRegex(ValueError, "alias.*differ"):
            load_model(module, str(self.path))

    def test_declared_alias_requires_existing_shared_target_parameter(self):
        class InvalidAliasModule(nn.Module):
            checkpoint_parameter_aliases = {
                "head.weight": "missing.weight",
            }

            def __init__(self):
                super().__init__()
                self.head = nn.Linear(2, 2, bias=False)

        module = InvalidAliasModule()
        save_file(
            {"head.weight": torch.ones(2, 2)},
            str(self.path / "model.safetensors"),
        )

        with self.assertRaisesRegex(ValueError, "alias target"):
            load_model(module, str(self.path))

    def test_undeclared_shared_parameter_alias_remains_unknown(self):
        class SharedWeightModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Linear(2, 2, bias=False)
                self.head = nn.Linear(2, 2, bias=False)
                self.head.weight = self.embedding.weight

        module = SharedWeightModule()
        save_file(
            {
                "embedding.weight": torch.ones(2, 2),
                "head.weight": torch.ones(2, 2),
            },
            str(self.path / "model.safetensors"),
        )

        with self.assertRaisesRegex(KeyError, "head.weight"):
            load_model(module, str(self.path))

    def test_declared_alias_respects_include_and_strip_prefixes(self):
        class SharedWeightModule(nn.Module):
            checkpoint_parameter_aliases = {
                "head.weight": "embedding.weight",
            }

            def __init__(self):
                super().__init__()
                self.embedding = nn.Linear(2, 2, bias=False)
                self.head = nn.Linear(2, 2, bias=False)
                self.head.weight = self.embedding.weight

        module = SharedWeightModule()
        weight = torch.full((2, 2), 9.0)
        save_file(
            {
                "text.embedding.weight": weight,
                "text.head.weight": weight.clone(),
                "visual.weight": torch.zeros(2, 2),
            },
            str(self.path / "model.safetensors"),
        )

        report = load_model(
            module,
            str(self.path),
            include_prefixes=("text.",),
            strip_prefix="text.",
        )

        torch.testing.assert_close(module.embedding.weight, weight)
        self.assertEqual(report.skipped_checkpoint_weights, ("visual.weight",))

    def test_declared_alias_rejects_packed_source_and_target_names(self):
        class PackedAliasModule(nn.Module):
            packed_modules_mapping = {
                "q_proj": ("qkv_proj", "q"),
            }

            def __init__(self, aliases):
                super().__init__()
                self.checkpoint_parameter_aliases = aliases
                self.embedding = nn.Linear(2, 2, bias=False)
                self.qkv_proj = nn.Linear(2, 2, bias=False)

        invalid_aliases = (
            {"q_proj.weight": "embedding.weight"},
            {"head.weight": "qkv_proj.weight"},
        )
        for aliases in invalid_aliases:
            with self.subTest(aliases=aliases):
                with self.assertRaisesRegex(ValueError, "packed"):
                    load_model(PackedAliasModule(aliases), str(self.path))


if __name__ == "__main__":
    unittest.main()
