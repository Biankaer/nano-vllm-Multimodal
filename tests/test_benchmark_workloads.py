import base64
import json
import tempfile
import unittest
from pathlib import Path

from nanovllm.benchmark.workloads import (
    canonical_workload_bytes,
    generate_profile,
    load_jsonl_workload,
    workload_sha256,
)


def image_bytes(request) -> list[bytes]:
    encoded = []
    for message in request.messages:
        content = message["content"]
        if not isinstance(content, list):
            continue
        for part in content:
            if part.get("type") != "image_url":
                continue
            url = part["image_url"]["url"]
            encoded.append(base64.b64decode(url.split(",", 1)[1]))
    return encoded


class BuiltinWorkloadTests(unittest.TestCase):
    def test_generation_is_deterministic_and_hash_is_stable(self):
        first = generate_profile("text_short", count=4, seed=17)
        second = generate_profile("text_short", count=4, seed=17)

        self.assertEqual(canonical_workload_bytes(first), canonical_workload_bytes(second))
        self.assertEqual(workload_sha256(first), workload_sha256(second))

    def test_unique_profile_uses_different_png_bytes_per_request(self):
        requests = generate_profile("image_single_unique", count=3, seed=3)

        images = [image_bytes(request)[0] for request in requests]
        self.assertEqual(len(set(images)), 3)

    def test_reused_profile_uses_identical_png_bytes(self):
        requests = generate_profile("image_single_reused", count=3, seed=3)

        images = [image_bytes(request)[0] for request in requests]
        self.assertEqual(len(set(images)), 1)

    def test_mixed_profile_has_stable_fifty_thirty_twenty_mix(self):
        requests = generate_profile("mixed", count=10, seed=7)
        kinds = [request.tags[0] for request in requests]

        self.assertEqual(kinds.count("text"), 5)
        self.assertEqual(kinds.count("image_single"), 3)
        self.assertEqual(kinds.count("image_multi"), 2)
        self.assertEqual(kinds, [request.tags[0] for request in generate_profile("mixed", 10, 7)])

    def test_unknown_profile_and_non_positive_count_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "profile"):
            generate_profile("unknown", count=1)
        with self.assertRaisesRegex(ValueError, "positive"):
            generate_profile("text_short", count=0)


class JSONLWorkloadTests(unittest.TestCase):
    def test_relative_image_is_materialized_as_data_url(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = root / "sample.png"
            expected_image = b"\x89PNG\r\n\x1a\nexample"
            image.write_bytes(expected_image)
            record = {
                "id": "custom-1",
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "sample.png"}},
                        {"type": "text", "text": "Describe it."},
                    ],
                }],
                "max_tokens": 32,
                "tags": ["custom"],
            }
            path = root / "workload.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")

            loaded = load_jsonl_workload(path)

        url = loaded[0].messages[0]["content"][0]["image_url"]["url"]
        self.assertTrue(url.startswith("data:image/png;base64,"))
        self.assertEqual(base64.b64decode(url.split(",", 1)[1]), expected_image)

    def test_duplicate_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "workload.jsonl"
            record = {
                "id": "duplicate",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 8,
            }
            path.write_text("\n".join([json.dumps(record), json.dumps(record)]), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_jsonl_workload(path)

    def test_missing_image_and_video_parts_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            missing = {
                "id": "missing",
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": "missing.png"}}
                ]}],
                "max_tokens": 8,
            }
            video = {
                "id": "video",
                "messages": [{"role": "user", "content": [
                    {"type": "video_url", "video_url": {"url": "clip.mp4"}}
                ]}],
                "max_tokens": 8,
            }
            missing_path = root / "missing.jsonl"
            video_path = root / "video.jsonl"
            missing_path.write_text(json.dumps(missing), encoding="utf-8")
            video_path.write_text(json.dumps(video), encoding="utf-8")

            with self.assertRaises(FileNotFoundError):
                load_jsonl_workload(missing_path)
            with self.assertRaisesRegex(ValueError, "video"):
                load_jsonl_workload(video_path)


if __name__ == "__main__":
    unittest.main()
