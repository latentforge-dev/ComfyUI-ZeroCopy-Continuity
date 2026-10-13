"""ComfyUI-ZeroCopy-Continuity: Permanent VRAM Pinning & Lossless Latent Continuity Chaining.

Author: Antigravity & LatentForge Team
License: MIT
Repository: https://github.com/latentforge/ComfyUI-ZeroCopy-Continuity
"""

from .zero_copy_cuda import (
    CineFluxZeroCopyVRAMPin,
    CineFluxZeroCopyLatentStore,
    CineFluxZeroCopyLatentHandoff,
)

NODE_CLASS_MAPPINGS = {
    "CineFluxZeroCopyVRAMPin": CineFluxZeroCopyVRAMPin,
    "CineFluxZeroCopyLatentStore": CineFluxZeroCopyLatentStore,
    "CineFluxZeroCopyLatentHandoff": CineFluxZeroCopyLatentHandoff,
    # Clean non-namespaced aliases for general community usage
    "ZeroCopyVRAMPin": CineFluxZeroCopyVRAMPin,
    "ZeroCopyLatentStore": CineFluxZeroCopyLatentStore,
    "ZeroCopyLatentHandoff": CineFluxZeroCopyLatentHandoff,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CineFluxZeroCopyVRAMPin": "⚡ CUDA Zero-Copy VRAM Pin (Anti-Eviction)",
    "CineFluxZeroCopyLatentStore": "💾 CUDA Zero-Copy Latent Store",
    "CineFluxZeroCopyLatentHandoff": "🔄 CUDA Zero-Copy Latent Handoff (Continuity)",
    "ZeroCopyVRAMPin": "⚡ CUDA Zero-Copy VRAM Pin (Anti-Eviction)",
    "ZeroCopyLatentStore": "💾 CUDA Zero-Copy Latent Store",
    "ZeroCopyLatentHandoff": "🔄 CUDA Zero-Copy Latent Handoff (Continuity)",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
