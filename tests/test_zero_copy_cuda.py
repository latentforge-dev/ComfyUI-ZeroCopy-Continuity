"""Unit test suite for ComfyUI-ZeroCopy-Continuity CUDA Manager & Ring Buffer.

High-level purpose:
    Verifies thread-safe singleton behavior, model anti-eviction pinning, LRU ring buffer
    capacity bounds, tail-frame window slicing, and lossless tensor handoff accuracy.
"""

import unittest
import torch

from zero_copy_cuda import (
    CUDA_MANAGER,
    LATENT_BUFFER,
    CineFluxZeroCopyLatentStore,
    CineFluxZeroCopyVRAMPin,
    ZeroCopyCudaManager,
    ZeroCopyLatentRingBuffer,
)


class DummyInnerModel:
    """Mock inner diffusion model for testing VRAM pinning flags."""


class DummyModelPatcher:
    """Mock ComfyUI ModelPatcher for testing VRAM pinning and unpinning."""

    def __init__(self) -> None:
        """Initializes mock patcher with inner model."""
        self.model = DummyInnerModel()


class TestZeroCopyContinuity(unittest.TestCase):
    """Tests for ZeroCopyCudaManager and ZeroCopyLatentRingBuffer."""

    def setUp(self) -> None:
        """Clears the global latent ring buffer before each test."""
        LATENT_BUFFER.clear()

    def test_singleton_instantiation(self) -> None:
        """Verifies that manager and buffer enforce singleton identity."""
        self.assertIs(ZeroCopyCudaManager(), CUDA_MANAGER)
        self.assertIs(ZeroCopyLatentRingBuffer(), LATENT_BUFFER)

    def test_vram_pin_and_unpin_node(self) -> None:
        """Verifies that CineFluxZeroCopyVRAMPin sets and clears pinning attributes."""
        node = CineFluxZeroCopyVRAMPin()
        patcher = DummyModelPatcher()

        pinned_tuple = node.apply_pin(patcher, enable_pinning=True, tune_allocator=True)
        self.assertIs(pinned_tuple[0], patcher)
        self.assertTrue(getattr(patcher, "_is_zero_copy_pinned", False))
        self.assertTrue(getattr(patcher.model, "_is_zero_copy_pinned", False))

        node.apply_pin(patcher, enable_pinning=False, tune_allocator=False)
        self.assertFalse(getattr(patcher, "_is_zero_copy_pinned", True))
        self.assertFalse(getattr(patcher.model, "_is_zero_copy_pinned", True))

    def test_latent_store_and_tail_slice_bit_identical(self) -> None:
        """Verifies bit-identical storage and tail frame extraction."""
        store_node = CineFluxZeroCopyLatentStore()
        # Shape: [B=1, C=24, T=6, H=8, W=8]
        raw_tensor = torch.arange(1 * 24 * 6 * 8 * 8, dtype=torch.float32).reshape(1, 24, 6, 8, 8)
        out_tuple = store_node.store({"samples": raw_tensor}, buffer_tag="shot_alpha")

        self.assertTrue(LATENT_BUFFER.has_latent("shot_alpha"))
        self.assertIs(out_tuple[0]["samples"], raw_tensor)

        tail = LATENT_BUFFER.get_tail_slice("shot_alpha", num_frames=1)
        self.assertIsNotNone(tail)
        assert tail is not None
        self.assertEqual(tuple(tail.shape), (1, 24, 1, 8, 8))
        # MSE between original terminal frame and extracted tail slice must be exactly 0.0
        mse = torch.mean((tail - raw_tensor[:, :, -1:, :, :]) ** 2).item()
        self.assertEqual(mse, 0.0)

    def test_lru_eviction_and_max_retained_frames(self) -> None:
        """Verifies LRU capacity bound and optional tail-frame slicing."""
        max_cap = LATENT_BUFFER._max_entries
        for idx in range(max_cap + 2):
            tensor = torch.ones((1, 24, 10, 4, 4), dtype=torch.float32) * float(idx)
            LATENT_BUFFER.store_latent(
                tag=f"tag_{idx}",
                latent_dict={"samples": tensor},
                max_retained_frames=2,
            )

        # Oldest two tags (tag_0, tag_1) must have been LRU-evicted
        self.assertFalse(LATENT_BUFFER.has_latent("tag_0"))
        self.assertFalse(LATENT_BUFFER.has_latent("tag_1"))
        self.assertTrue(LATENT_BUFFER.has_latent(f"tag_{max_cap + 1}"))

        stored = LATENT_BUFFER.get_latent(f"tag_{max_cap + 1}")
        assert stored is not None
        # Temporal dimension should be trimmed from 10 to 2
        self.assertEqual(tuple(stored["samples"].shape), (1, 24, 2, 4, 4))


if __name__ == "__main__":
    unittest.main()
