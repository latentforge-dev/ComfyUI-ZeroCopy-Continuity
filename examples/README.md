# 🎬 ComfyUI-ZeroCopy-Continuity Example Workflows

This directory contains reference workflows demonstrating:
1. **Model VRAM Pinning**: Pinning DiT weights (`MiniMax-H3`, `Wan 2.1`, `FLUX.1`) into VRAM to avoid CPU-RAM paging between generation steps.
2. **Latent Ring Buffer Store & Handoff**: Chaining consecutive shots with zero VAE reconstruction errors.

### Workflow Files:
* `zerocopy_continuity_basic.json`: Minimal 2-shot chaining graph showing how `ZeroCopyVRAMPin`, `ZeroCopyLatentStore`, and `ZeroCopyLatentHandoff` connect to standard KSamplers and conditioning.
