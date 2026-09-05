from __future__ import annotations

import hashlib
import os
import shutil
import sys
from pathlib import Path
from threading import Lock
from typing import Iterable

import torch
from torch.utils.cpp_extension import CUDA_HOME


_LOAD_LOCK = Lock()
_LOADED_BUILD_NAMES: set[str] = set()
_CXX_FLAGS = ("-O3",)
_CUDA_FLAGS = ("-O3", "-lineinfo")


def ensure_ninja_available() -> None:
    if shutil.which("ninja") is not None:
        return
    interpreter_bin = Path(sys.executable).resolve().parent
    candidate = interpreter_bin / "ninja"
    if candidate.is_file() and os.access(candidate, os.X_OK):
        os.environ["PATH"] = str(interpreter_bin) + os.pathsep + os.environ.get("PATH", "")
        return
    raise RuntimeError(
        f"Ninja is required for the Paged KV extension; expected {candidate} "
        "or an executable named ninja on PATH"
    )


def extension_source_paths() -> tuple[Path, Path]:
    root = Path(__file__).resolve().parents[3]
    source_root = root / "csrc" / "paged_kv"
    return source_root / "paged_kv.cpp", source_root / "paged_kv_cuda.cu"


def cuda_extension_buildable() -> bool:
    return CUDA_HOME is not None and all(
        path.is_file() for path in extension_source_paths()
    )


def build_fingerprint(
    source_payloads: Iterable[bytes],
    torch_version: str,
    torch_cuda_version: str | None,
    cuda_home: str | None,
    flags: Iterable[str],
    arch_list: str,
) -> str:
    digest = hashlib.sha256()
    fields = (
        *source_payloads,
        torch_version.encode(),
        str(torch_cuda_version).encode(),
        str(cuda_home).encode(),
        "\0".join(flags).encode(),
        arch_list.encode(),
    )
    for field in fields:
        digest.update(len(field).to_bytes(8, "little"))
        digest.update(field)
    return digest.hexdigest()


def extension_build_name(paths: tuple[Path, ...] | None = None) -> str:
    source_paths = paths or extension_source_paths()
    missing = [str(path) for path in source_paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"Paged KV extension sources are missing: {', '.join(missing)}")
    fingerprint = build_fingerprint(
        tuple(path.read_bytes() for path in source_paths),
        torch.__version__,
        torch.version.cuda,
        str(CUDA_HOME) if CUDA_HOME is not None else None,
        (*_CXX_FLAGS, *_CUDA_FLAGS),
        os.environ.get("TORCH_CUDA_ARCH_LIST", "native"),
    )
    return f"nanovllm_paged_kv_{fingerprint[:16]}"


def load_paged_kv_extension(*, verbose: bool = False) -> None:
    source_paths = tuple(extension_source_paths())
    build_name = extension_build_name(source_paths)
    with _LOAD_LOCK:
        if build_name in _LOADED_BUILD_NAMES:
            return
        try:
            ensure_ninja_available()
            from torch.utils.cpp_extension import load

            load(
                name=build_name,
                sources=[str(path) for path in source_paths],
                extra_cflags=list(_CXX_FLAGS),
                extra_cuda_cflags=list(_CUDA_FLAGS),
                with_cuda=True,
                is_python_module=False,
                verbose=verbose,
            )
        except BaseException as exc:
            source_list = ", ".join(str(path) for path in source_paths)
            raise RuntimeError(
                "Failed to build NanoInfer Paged KV CUDA extension. "
                f"CUDA_HOME={CUDA_HOME!s}; PyTorch CUDA={torch.version.cuda!s}; "
                f"sources={source_list}; original error: {exc}"
            ) from exc
        _LOADED_BUILD_NAMES.add(build_name)


def reset_extension_state_for_tests() -> None:
    with _LOAD_LOCK:
        _LOADED_BUILD_NAMES.clear()
