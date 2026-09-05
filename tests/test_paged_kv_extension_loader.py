import importlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nanovllm.layers.paged_kv.loader as loader
import nanovllm.layers.paged_kv.cuda_backend as cuda_backend


class ExtensionFingerprintTests(unittest.TestCase):
    def test_fingerprint_changes_with_source_versions_flags_and_arch(self):
        baseline = loader.build_fingerprint(
            source_payloads=(b"cpp", b"cuda"),
            torch_version="2.5.1",
            torch_cuda_version="12.1",
            cuda_home="/cuda",
            flags=("-O3",),
            arch_list="8.0",
        )
        variants = [
            ((b"cpp changed", b"cuda"), "2.5.1", "12.1", "/cuda", ("-O3",), "8.0"),
            ((b"cpp", b"cuda"), "2.6.0", "12.1", "/cuda", ("-O3",), "8.0"),
            ((b"cpp", b"cuda"), "2.5.1", "12.4", "/cuda", ("-O3",), "8.0"),
            ((b"cpp", b"cuda"), "2.5.1", "12.1", "/cuda2", ("-O3",), "8.0"),
            ((b"cpp", b"cuda"), "2.5.1", "12.1", "/cuda", ("-O0",), "8.0"),
            ((b"cpp", b"cuda"), "2.5.1", "12.1", "/cuda", ("-O3",), "9.0"),
        ]

        for arguments in variants:
            with self.subTest(arguments=arguments[1:]):
                self.assertNotEqual(loader.build_fingerprint(*arguments), baseline)

    def test_import_does_not_eagerly_compile(self):
        with patch("torch.utils.cpp_extension.load") as compile_extension:
            importlib.reload(loader)

        compile_extension.assert_not_called()


class ExtensionLoadTests(unittest.TestCase):
    def setUp(self):
        loader.reset_extension_state_for_tests()

    def test_repeated_load_compiles_only_once(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cpp = Path(temp_dir, "paged_kv.cpp")
            cuda = Path(temp_dir, "paged_kv_cuda.cu")
            cpp.write_text("// cpp", encoding="utf-8")
            cuda.write_text("// cuda", encoding="utf-8")
            with (
                patch.object(loader, "extension_source_paths", return_value=(cpp, cuda)),
                patch("torch.utils.cpp_extension.load") as compile_extension,
            ):
                loader.load_paged_kv_extension()
                loader.load_paged_kv_extension()

        compile_extension.assert_called_once()
        self.assertFalse(compile_extension.call_args.kwargs["is_python_module"])

    def test_cuda_backend_fast_path_checks_loader_only_once(self):
        cuda_backend.reset_cuda_backend_extension_state_for_tests()

        with patch.object(cuda_backend, "load_paged_kv_extension") as load:
            cuda_backend.CUDAPagedKVBackend._load()
            cuda_backend.CUDAPagedKVBackend._load()

        load.assert_called_once_with()

    def test_uses_ninja_installed_next_to_selected_python(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            bin_dir = Path(temp_dir)
            ninja = bin_dir / "ninja"
            ninja.write_text("binary", encoding="utf-8")
            ninja.chmod(0o755)
            with (
                patch.object(loader.sys, "executable", str(bin_dir / "python")),
                patch.object(loader.shutil, "which", return_value=None),
                patch.dict(os.environ, {"PATH": "/usr/bin"}),
            ):
                loader.ensure_ninja_available()

                self.assertEqual(os.environ["PATH"].split(os.pathsep)[0], str(bin_dir))

    def test_compile_failure_contains_environment_and_source_context(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cpp = Path(temp_dir, "paged_kv.cpp")
            cuda = Path(temp_dir, "paged_kv_cuda.cu")
            cpp.write_text("// cpp", encoding="utf-8")
            cuda.write_text("// cuda", encoding="utf-8")
            with (
                patch.object(loader, "extension_source_paths", return_value=(cpp, cuda)),
                patch("torch.utils.cpp_extension.load", side_effect=RuntimeError("nvcc failed")),
            ):
                with self.assertRaises(RuntimeError) as raised:
                    loader.load_paged_kv_extension()

        message = str(raised.exception)
        self.assertIn("nvcc failed", message)
        self.assertIn("CUDA_HOME", message)
        self.assertIn("PyTorch CUDA", message)
        self.assertIn("paged_kv_cuda.cu", message)


if __name__ == "__main__":
    unittest.main()
