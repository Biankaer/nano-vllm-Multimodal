from __future__ import annotations

from importlib import metadata as importlib_metadata

from packaging.version import InvalidVersion, Version


_MINIMUM_TRANSFORMERS_VERSION = Version("4.57.6")


def require_transformers_version(current_version: str | None = None) -> None:
    installed = current_version
    if installed is None:
        try:
            installed = importlib_metadata.version("transformers")
        except importlib_metadata.PackageNotFoundError as exc:
            raise ImportError(
                "Qwen3-VL requires transformers>=4.57.6, but transformers is not installed"
            ) from exc
    try:
        installed_version = Version(installed)
    except InvalidVersion as exc:
        raise ImportError(
            f"Qwen3-VL requires transformers>=4.57.6; cannot parse installed version {installed!r}"
        ) from exc
    if installed_version < _MINIMUM_TRANSFORMERS_VERSION:
        raise ImportError(
            "Qwen3-VL requires transformers>=4.57.6; "
            f"found transformers=={installed}"
        )


__all__ = ["require_transformers_version"]
