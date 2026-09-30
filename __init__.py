"""CFG Megapack for ComfyUI: classifier-free guidance taken apart into stages (when, weak branch, combine, where,
correct, govern, measure), one node per stage, and one node per published method on the same engine.

Optional: set CFG_MEGAPACK_VRAM_FRACTION (for example 0.6) before starting ComfyUI to cap this process's share of
GPU memory on a shared card (applied once, when the pack loads, before any model is loaded)."""
import logging
import os


async def comfy_entrypoint():
    """ComfyUI's entry point for V3 node packs. The node module is imported here, so the engine, the method library
    and the paper specifications stay importable (and testable) without ComfyUI."""
    from .cfg_megapack.nodes import CFGMegapackExtension
    return CFGMegapackExtension()


_fraction = (os.environ.get("CFG_MEGAPACK_VRAM_FRACTION") or os.environ.get("CFG_PROTOTYPES_VRAM_FRACTION") or "").strip()
if _fraction:
    import torch

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(float(_fraction))
        total = torch.cuda.get_device_properties(0).total_memory / 2 ** 20
        logging.info(f"[CFG Megapack] GPU memory capped at {float(_fraction):.3f} of {total:.0f} MiB "
                     f"= {float(_fraction) * total:.0f} MiB for this process")
