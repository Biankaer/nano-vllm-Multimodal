import tempfile
import unittest
from pathlib import Path

import torch
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


if __name__ == "__main__":
    unittest.main()
