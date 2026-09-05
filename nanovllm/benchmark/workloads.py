from __future__ import annotations

import base64
import hashlib
import io
import json
import mimetypes
import random
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

from nanovllm.benchmark.schema import WorkloadRequest


PROFILE_NAMES = (
    "text_short",
    "text_long",
    "image_single_unique",
    "image_single_reused",
    "image_multi",
    "mixed",
)


def _png_data_url(identity: str) -> str:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("Pillow is required to generate image workloads") from exc
    digest = hashlib.sha256(identity.encode("utf-8")).digest()
    background = tuple(48 + byte % 160 for byte in digest[:3])
    foreground = tuple(48 + byte % 160 for byte in digest[3:6])
    image = Image.new("RGB", (448, 448), background)
    draw = ImageDraw.Draw(image)
    for index in range(0, 24, 3):
        x = digest[index] * 400 // 255
        y = digest[index + 1] * 400 // 255
        size = 16 + digest[index + 2] % 32
        draw.rectangle((x, y, min(x + size, 447), min(y + size, 447)), fill=foreground)
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=False)
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")


def _text_request(request_id: str, *, long: bool = False) -> WorkloadRequest:
    if long:
        context = " ".join(
            f"Section {index}: deterministic inference systems need measurable latency and throughput."
            for index in range(180)
        )
        prompt = f"{context}\nSummarize the main engineering trade-offs in four points."
        max_tokens = 128
        tags = ("text", "long")
    else:
        prompt = (
            "Explain in concise terms how continuous batching, paged KV cache, and prefix "
            "caching interact in an LLM serving engine."
        )
        max_tokens = 64
        tags = ("text", "short")
    return WorkloadRequest(
        request_id=request_id,
        messages=({"role": "user", "content": prompt},),
        max_tokens=max_tokens,
        tags=tags,
    )


def _image_request(
    request_id: str,
    *,
    identities: tuple[str, ...],
    tag: str,
) -> WorkloadRequest:
    content: list[dict[str, Any]] = []
    for index, identity in enumerate(identities, start=1):
        content.append({"type": "text", "text": f"Image {index}:"})
        content.append({"type": "image_url", "image_url": {"url": _png_data_url(identity)}})
    content.append({"type": "text", "text": "Describe the geometric layout and dominant colors."})
    return WorkloadRequest(
        request_id=request_id,
        messages=({"role": "user", "content": content},),
        max_tokens=64,
        tags=(tag, "image"),
    )


def generate_profile(profile: str, count: int, seed: int = 0) -> tuple[WorkloadRequest, ...]:
    if profile not in PROFILE_NAMES:
        raise ValueError(f"unknown workload profile: {profile}")
    if count <= 0:
        raise ValueError("workload count must be positive")
    rng = random.Random(seed)
    mixed_kinds = ["text"] * ((count * 5) // 10)
    mixed_kinds += ["image_single"] * ((count * 3) // 10)
    mixed_kinds += ["image_multi"] * (count - len(mixed_kinds))
    rng.shuffle(mixed_kinds)

    requests: list[WorkloadRequest] = []
    for index in range(count):
        request_id = f"{profile}-{index:05d}"
        selected = mixed_kinds[index] if profile == "mixed" else profile
        if selected in {"text", "text_short"}:
            request = _text_request(request_id)
        elif selected == "text_long":
            request = _text_request(request_id, long=True)
        elif selected == "image_single_unique":
            request = _image_request(
                request_id,
                identities=(f"unique:{seed}:{request_id}",),
                tag="image_single",
            )
        elif selected in {"image_single_reused", "image_single"}:
            identity = f"reused:{seed}" if selected == "image_single_reused" else f"mixed:{seed}:{request_id}"
            request = _image_request(request_id, identities=(identity,), tag="image_single")
        elif selected in {"image_multi", "mixed_image_multi"}:
            request = _image_request(
                request_id,
                identities=(f"multi-a:{seed}:{request_id}", f"multi-b:{seed}:{request_id}"),
                tag="image_multi",
            )
        else:
            raise AssertionError(f"unhandled workload kind: {selected}")
        requests.append(request)
    return tuple(requests)


def canonical_workload_bytes(requests: Iterable[WorkloadRequest]) -> bytes:
    payload = {
        "schema_version": "1.0",
        "requests": [request.to_dict() for request in requests],
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def workload_sha256(requests: Iterable[WorkloadRequest]) -> str:
    return hashlib.sha256(canonical_workload_bytes(requests)).hexdigest()


def _materialize_image_url(url: str, *, root: Path) -> str:
    parsed = urlparse(url)
    if parsed.scheme in {"http", "https", "data"}:
        return url
    if parsed.scheme:
        raise ValueError(f"unsupported image URL scheme: {parsed.scheme}")
    image_path = (root / url).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"workload image does not exist: {image_path}")
    mime_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _validate_messages(messages: Any, *, root: Path) -> tuple[dict[str, Any], ...]:
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    materialized: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict) or not isinstance(message.get("role"), str):
            raise ValueError("each message requires a string role")
        content = message.get("content")
        if isinstance(content, str):
            materialized.append({"role": message["role"], "content": content})
            continue
        if not isinstance(content, list) or not content:
            raise ValueError("message content must be text or a non-empty parts list")
        parts: list[dict[str, Any]] = []
        for part in content:
            if not isinstance(part, dict) or not isinstance(part.get("type"), str):
                raise ValueError("each content part requires a type")
            part_type = part["type"]
            if "video" in part_type:
                raise ValueError("video workload inputs are not supported")
            if part_type == "text":
                if not isinstance(part.get("text"), str):
                    raise ValueError("text parts require text")
                parts.append({"type": "text", "text": part["text"]})
            elif part_type == "image_url":
                image_url = part.get("image_url")
                if not isinstance(image_url, dict) or not isinstance(image_url.get("url"), str):
                    raise ValueError("image_url parts require image_url.url")
                copied = dict(image_url)
                copied["url"] = _materialize_image_url(copied["url"], root=root)
                parts.append({"type": "image_url", "image_url": copied})
            else:
                raise ValueError(f"unsupported OpenAI content part: {part_type}")
        materialized.append({"role": message["role"], "content": parts})
    return tuple(materialized)


def load_jsonl_workload(path: str | Path) -> tuple[WorkloadRequest, ...]:
    workload_path = Path(path).resolve()
    requests: list[WorkloadRequest] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(workload_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on workload line {line_number}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"workload line {line_number} must be an object")
        request_id = record.get("id")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError(f"workload line {line_number} requires a non-empty id")
        if request_id in seen_ids:
            raise ValueError(f"duplicate workload request id: {request_id}")
        seen_ids.add(request_id)
        max_tokens = record.get("max_tokens")
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0:
            raise ValueError(f"workload {request_id} max_tokens must be positive")
        tags = record.get("tags", [])
        if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
            raise ValueError(f"workload {request_id} tags must be strings")
        requests.append(
            WorkloadRequest(
                request_id=request_id,
                messages=_validate_messages(record.get("messages"), root=workload_path.parent),
                max_tokens=max_tokens,
                tags=tuple(tags),
            )
        )
    if not requests:
        raise ValueError("workload must contain at least one request")
    return tuple(requests)
