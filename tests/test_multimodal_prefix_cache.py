import subprocess
import sys
import unittest
from pathlib import Path

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence
from nanovllm.multimodal.runtime import MultimodalPromptMetadata, VisualTokenSpan


def make_metadata(prompt_length: int, feature_key: str) -> MultimodalPromptMetadata:
    positions = tuple(range(prompt_length))
    return MultimodalPromptMetadata(
        position_ids=(positions, positions, positions),
        rope_delta=0,
        visual_spans=(VisualTokenSpan(start=1, length=3, feature_key=feature_key),),
    )


def make_image_sequence(feature_key: str) -> Sequence:
    return Sequence(
        [100, 999, 999, 999, 200, 201, 202, 203],
        multimodal_metadata=make_metadata(8, feature_key),
    )


def cache_first_block(manager: BlockManager, sequence: Sequence) -> None:
    manager.allocate(sequence, num_cached_blocks=0)
    sequence.num_cached_tokens = 0
    sequence.num_scheduled_tokens = 4
    manager.hash_blocks(sequence)
    manager.deallocate(sequence)


class SequencePrefixCacheIdentityTests(unittest.TestCase):

    def setUp(self):
        self.original_block_size = Sequence.block_size
        Sequence.block_size = 4

    def tearDown(self):
        Sequence.block_size = self.original_block_size

    def test_text_tokens_keep_token_id_only_identity(self):
        sequence = Sequence([1, 2, 3, 4, 5])

        cache_key = sequence.block_cache_key(0)

        self.assertEqual([token.token_id for token in cache_key], [1, 2, 3, 4])
        self.assertTrue(all(token.multimodal_id is None for token in cache_key))

    def test_text_sequence_import_does_not_require_pillow(self):
        project_root = Path(__file__).resolve().parents[1]
        script = """
import sys

class BlockPillowImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'PIL' or fullname.startswith('PIL.'):
            raise ModuleNotFoundError('Pillow intentionally unavailable')
        return None

sys.meta_path.insert(0, BlockPillowImports())
from nanovllm.engine.sequence import Sequence
assert Sequence([1, 2, 3]).last_token == 3
"""

        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=project_root,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_different_images_change_visual_token_identity(self):
        image_a = make_image_sequence("model:processor:image-a:grid")
        image_b = make_image_sequence("model:processor:image-b:grid")

        self.assertNotEqual(image_a.block_cache_key(0), image_b.block_cache_key(0))
        self.assertEqual(
            image_a.block_cache_key(0)[1].multimodal_id,
            "model:processor:image-a:grid:0",
        )

    def test_prefill_pickle_state_retains_metadata(self):
        sequence = make_image_sequence("image-a")
        restored = Sequence.__new__(Sequence)

        restored.__setstate__(sequence.__getstate__())

        self.assertEqual(restored.multimodal_metadata, sequence.multimodal_metadata)
        self.assertEqual(restored.mrope_position_delta, 0)

    def test_decode_pickle_state_keeps_only_rope_delta(self):
        sequence = make_image_sequence("image-a")
        sequence.is_prefill = False
        restored = Sequence.__new__(Sequence)

        restored.__setstate__(sequence.__getstate__())

        self.assertIsNone(restored.multimodal_metadata)
        self.assertEqual(restored.mrope_position_delta, 0)


class MultimodalBlockManagerTests(unittest.TestCase):

    def setUp(self):
        self.original_block_size = Sequence.block_size
        Sequence.block_size = 4

    def tearDown(self):
        Sequence.block_size = self.original_block_size

    def test_same_image_prefix_reuses_cached_block(self):
        manager = BlockManager(num_blocks=8, block_size=4)
        cache_first_block(manager, make_image_sequence("image-a"))

        self.assertEqual(manager.can_allocate(make_image_sequence("image-a")), 1)

    def test_different_image_prefix_does_not_reuse_cached_block(self):
        manager = BlockManager(num_blocks=8, block_size=4)
        cache_first_block(manager, make_image_sequence("image-a"))

        self.assertEqual(manager.can_allocate(make_image_sequence("image-b")), 0)

    def test_text_only_prefix_reuse_is_unchanged(self):
        manager = BlockManager(num_blocks=8, block_size=4)
        first = Sequence([1, 2, 3, 4, 5, 6, 7, 8])
        cache_first_block(manager, first)

        self.assertEqual(manager.can_allocate(Sequence([1, 2, 3, 4, 9, 10, 11, 12])), 1)


if __name__ == "__main__":
    unittest.main()
