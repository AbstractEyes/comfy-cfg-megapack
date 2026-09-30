"""engine.py - the guidance plan and the per-step executor behind every CFG Megapack node.

Every node in this pack edits ONE stage of a guidance plan stored on the MODEL, then installs a single
sampler_cfg_function bound to the updated plan. ComfyUI has one cfg-function slot per model, so the stages
compose here, in a fixed order, instead of by stacking hooks that overwrite each other:

    1 when     schedule the guidance scale over the run and gate it to a window of the run
    2 weak     an extra pass of the model with its self-attention perturbed (PAG, SEG, temperature)
    3 mix      how the conditional and unconditional predictions combine (one rule)
    4 where    reshape the guidance term by frequency band, then by region
    5 correct  magnitude corrections, applied in the order the nodes were chained
    6 govern   the angle band: a projection that holds the angle between x0_hat and its home inside [min, max]
    7 measure  per-step statistics appended to a JSON-lines file

Notation used in every docstring: x is the latent the sampler passes (ComfyUI's coordinates), sigma its noise
level, x0_c / x0_u the conditional / unconditional denoised predictions, w the guidance scale, and the guidance
term G = x0_hat - x0_c (how far the guided estimate is pushed past the conditional one; plain CFG gives
G = (w - 1)(x0_c - x0_u)). Stages 2 and 4 act on G; stage 3 builds x0_hat; stage 5 corrects its size; stage 6
turns it back inside the angle band, the last word on its direction.

ComfyUI is imported lazily (only the extra perturbed pass needs it), so the maths runs and is tested on CPU
without a ComfyUI install.
"""
from __future__ import annotations

import copy
import json
import math
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from . import cfg_variants as cv

PLAN_KEY = "cfg_prototypes_plan"
SPACES = ("x0", "eps", "v")
SPACE_LABELS = {"auto (the method's own)": "auto", "denoised (x0)": "x0", "noise (eps)": "eps", "velocity (v)": "v"}
_EPS = 1e-8


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

def empty_plan() -> Dict[str, Any]:
    return {"when": None, "weak": None, "mix": None, "bands": None, "region": None, "correct": [], "govern": None,
            "measure": None}


def read_plan(model_patcher) -> Dict[str, Any]:
    """A private copy of the plan on this MODEL (or an empty plan)."""
    plan = model_patcher.model_options.get(PLAN_KEY)
    return copy.deepcopy(plan) if plan is not None else empty_plan()


def _ours(fn) -> bool:
    return bool(getattr(getattr(fn, "__self__", None), "_is_cfg_prototypes_runtime", False))


def install(model_patcher, plan: Dict[str, Any]) -> None:
    """Store the plan on the (already cloned) MODEL and install one cfg function bound to it, plus the post-CFG
    function that hands samplers its exact result (first in the list, so other nodes' post-CFG functions still act
    on this pack's result)."""
    model_patcher.model_options[PLAN_KEY] = plan
    runtime = GuidanceRuntime(plan)
    model_patcher.set_model_sampler_cfg_function(runtime.cfg_function, disable_cfg1_optimization=True)
    others = [f for f in model_patcher.model_options.get("sampler_post_cfg_function", []) if not _ours(f)]
    model_patcher.model_options["sampler_post_cfg_function"] = [runtime.post_cfg_function] + others
    previous = model_patcher.model_options.get("sampler_calc_cond_batch_function")
    if getattr(previous, "_cfg_prototypes", False):
        previous = getattr(previous, "_previous", None)
    if runtime.may_skip_uncond():
        model_patcher.set_model_sampler_calc_cond_batch_function(runtime.make_calc_cond_batch_function(previous))
    elif previous is not None:
        model_patcher.set_model_sampler_calc_cond_batch_function(previous)
    elif "sampler_calc_cond_batch_function" in model_patcher.model_options:
        del model_patcher.model_options["sampler_calc_cond_batch_function"]


def clear(model_patcher) -> None:
    """Remove this pack's plan and hooks from the (already cloned) MODEL."""
    model_patcher.model_options.pop(PLAN_KEY, None)
    fn = model_patcher.model_options.get("sampler_cfg_function")
    if _ours(fn):
        del model_patcher.model_options["sampler_cfg_function"]
    post = model_patcher.model_options.get("sampler_post_cfg_function")
    if post is not None:
        others = [f for f in post if not _ours(f)]
        if others:
            model_patcher.model_options["sampler_post_cfg_function"] = others
        else:
            del model_patcher.model_options["sampler_post_cfg_function"]
    ccb = model_patcher.model_options.get("sampler_calc_cond_batch_function")
    if getattr(ccb, "_cfg_prototypes", False):
        prev = getattr(ccb, "_previous", None)
        if prev is None:
            del model_patcher.model_options["sampler_calc_cond_batch_function"]
        else:
            model_patcher.model_options["sampler_calc_cond_batch_function"] = prev


def foreign_cfg_hooks(model_options) -> bool:
    """True when other nodes' pre- or post-CFG functions are on the model (ComfyUI's CFGZeroStar, CFGNorm, APG,
    TCFG, PAG, SAG, SLG and kin). They read the unconditional prediction, so this pack never skips that pass then."""
    mo = model_options or {}
    return bool(mo.get("sampler_pre_cfg_function") or
                [f for f in mo.get("sampler_post_cfg_function", []) if not _ours(f)])


# ---------------------------------------------------------------------------
# Spaces
# ---------------------------------------------------------------------------

def is_flow_sampling(model_sampling) -> bool:
    """True for ComfyUI's CONST (rectified-flow) model sampling and its subclasses."""
    return model_sampling is not None and any(c.__name__ == "CONST" for c in type(model_sampling).__mro__)


def _bview(v: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return v.reshape(v.shape[0], *([1] * (ref.ndim - 1)))


_FLOW_EDGE = 1e-4        # flow models: the noise space reads sigma as at most 1 - _FLOW_EDGE (see to_space)


def to_space(x0: torch.Tensor, x: torch.Tensor, sig: torch.Tensor, space: str, flow: bool) -> torch.Tensor:
    """x0 (denoised) -> the prediction in `space`. sig has shape (B, 1, ...).
    eps models (ComfyUI's VE form x = x0 + sigma n): eps = (x - x0) / sigma; v = a eps - s x0 with a = 1/sqrt(1+sigma^2),
    s = sigma a (the VP v-prediction). Flow models (x = (1 - sigma) x0 + sigma n): eps = n, v = n - x0 (the velocity).
    At sigma = 1 (a flow schedule's first step) the noise equals the latent whatever the prediction, so the noise
    space would lose it; there both directions read sigma as 1 - 1e-4, which keeps them exact inverses (linear rules
    unchanged, other rules see the limit). Conversions run in float64: the round trip x0 -> eps -> x0 then loses
    nothing measurable, so a linear rule gives the same image in any space."""
    if space == "x0":
        return x0
    x0, x, sig = x0.double(), x.double(), sig.double()
    if flow and space == "eps":
        sig = sig.clamp(max=1.0 - _FLOW_EDGE)
    alpha = (1.0 - sig) if flow else torch.ones_like(sig)
    n = (x - alpha * x0) / sig.clamp_min(_EPS)
    if space == "eps":
        return n
    if space == "v":
        if flow:
            return n - x0
        a = 1.0 / torch.sqrt(1.0 + sig * sig)
        return a * n - (sig * a) * x0
    raise ValueError(f"unknown space {space!r}")


def from_space(d: torch.Tensor, x: torch.Tensor, sig: torch.Tensor, space: str, flow: bool) -> torch.Tensor:
    """The inverse of to_space: a prediction in `space` -> x0 (float64 unless the space is x0)."""
    if space == "x0":
        return d
    d, x, sig = d.double(), x.double(), sig.double()
    if space == "eps":
        if flow:
            sig = sig.clamp(max=1.0 - _FLOW_EDGE)
        alpha = (1.0 - sig) if flow else torch.ones_like(sig)
        return (x - sig * d) / alpha
    if space == "v":
        if flow:
            return x - sig * d
        a = 1.0 / torch.sqrt(1.0 + sig * sig)
        return a * (a * x) - (sig * a) * d
    raise ValueError(f"unknown space {space!r}")


def native_space(registry_space: str, flow: bool) -> str:
    """Map a cfg_variants registry space tag to a concrete space for this model."""
    if registry_space in ("any", "x0"):
        return "x0"
    if registry_space == "v":
        return "x0" if flow else "v"          # ComfyUI's RescaleCFG convention: v for eps/v models, x0 for flow
    if registry_space == "noise":
        return "v" if flow else "eps"         # the paper's native output: eps for eps models, velocity for flow
    return "x0"


# ---------------------------------------------------------------------------
# Stage 1: when
# ---------------------------------------------------------------------------

SCHEDULE_SHAPES = ("constant", "linear_up", "linear_down", "cosine_up", "cosine_down", "v_shape", "lambda_shape",
                   "tv_cfg", "beta_pdf", "c2fg_exp", "cads_ramp")
OUTSIDE_MODES = {"no guidance (conditional)": "cond", "base scale (the other nodes still apply)": "base",
                 "fixed scale (the other nodes still apply)": "fixed", "plain CFG at base scale (the other nodes off)": "plain"}


def progress_from_sigmas(sigma: float, sample_sigmas) -> Tuple[float, int, float]:
    """(fractional step index, number of steps N, progress in [0, 1]) of sigma on the sampler's schedule.
    progress = step / (N - 1): 0 at the first (noisiest) step, 1 at the last."""
    s = [float(v) for v in (sample_sigmas.tolist() if torch.is_tensor(sample_sigmas) else sample_sigmas)]
    n = len(s) - 1
    if n < 1:
        return 0.0, 1, 0.0
    if sigma >= s[0]:
        idx = 0.0
    elif sigma <= s[-1]:
        idx = float(n)
    else:
        idx = float(n)
        for i in range(n):
            hi, lo = s[i], s[i + 1]
            if hi >= sigma >= lo:
                idx = i + (0.0 if hi == lo else (hi - sigma) / (hi - lo))
                break
    prog = idx / (n - 1) if n > 1 else 0.0
    return idx, n, min(max(prog, 0.0), 1.0)


def noise_level(sigma: float, flow: bool) -> float:
    """Normalized noise level in [0, 1] (1 = pure noise): the flow time, or sigma / (1 + sigma) for VE sigmas."""
    return float(sigma) if flow else float(sigma) / (1.0 + float(sigma))


def schedule_value(shape: str, w_bar: float, progress: float, t: float, step: float = 0.0,
                   times: Optional[List[float]] = None, a: float = -1.0, b: float = -1.0) -> float:
    """Scheduled guidance scale. a / b < 0 select each shape's default.
      constant                   w_bar
      linear_up / linear_down    1 + 2 (w_bar - 1) p   /   1 + 2 (w_bar - 1)(1 - p)       (Wang et al. 2024)
      cosine_up / cosine_down    1 + (w_bar - 1)(1 - cos(pi p))   /   1 + (w_bar - 1)(1 + cos(pi p))
      v_shape / lambda_shape     area-normalized V and Lambda shapes (Wang et al. 2024)
      tv_cfg                     middle-peaked ramp, area-normalized on the sampler's grid (a = peak position, 0.5)
      beta_pdf                   w_bar * Beta_pdf(p; a, b) (beta-CFG time profile, a = b = 2)
      c2fg_exp                   w_bar * exp(a (1 - t)) (C2FG, a = ln 2)
      cads_ramp                  w_bar * gamma_CADS(t; a, b) (tau1 = 0.6, tau2 = 0.9)
    p = sampling progress (0 first step, 1 last), t = normalized noise level (1 = noise). The linear, cosine, V and
    Lambda shapes keep the average scale over the run equal to w_bar."""
    if shape == "constant":
        return float(w_bar)
    if shape in ("linear_up", "linear_down", "cosine_up", "cosine_down", "v_shape", "lambda_shape"):
        name = {"linear_up": "linear", "linear_down": "invlinear", "cosine_up": "cosine",
                "cosine_down": "sine", "v_shape": "v_shape", "lambda_shape": "lambda_shape"}[shape]
        return float(cv.wang_schedule(w_bar, name, reading="A")(cv.GuidanceContext(progress=progress, t=t)))
    if shape == "tv_cfg":
        peak = 0.5 if a < 0 else a
        if times is not None and len(times) >= 2:
            table = cv.tv_cfg_table(w_bar, len(times) - 1, times, peak)
            return float(table[min(max(int(step), 0), len(table) - 1)])
        s = w_bar - 1
        return 1 + 2 * s * (progress / peak if progress <= peak else (1 - progress) / max(1 - peak, 1e-6))
    if shape == "beta_pdf":
        return float(w_bar) * cv.beta_pdf(progress, 2.0 if a < 0 else a, 2.0 if b < 0 else b)
    if shape == "c2fg_exp":
        lam = math.log(2) if a < 0 else a
        return float(w_bar) * math.exp(lam * (1 - t))
    if shape == "cads_ramp":
        return float(w_bar) * cv.cads_gamma(t, 0.6 if a < 0 else a, 0.9 if b < 0 else b)
    raise ValueError(f"unknown schedule shape {shape!r}")


# ---------------------------------------------------------------------------
# Stage 2: weak branch (perturbed self-attention)
# ---------------------------------------------------------------------------

WEAK_METHODS = {"pag (identity attention)": "pag", "seg (blurred queries)": "seg",
                "temperature (flattened attention)": "temperature", "skip (self-attention removed)": "skip"}
WEAK_MODES = {"add on top of CFG": "add", "replace the unconditional": "replace"}
# SDXL UNet self-attention blocks (ComfyUI keys); SD1.5 shares the middle block key.
BLOCK_PRESETS = {
    "middle (PAG / SEG default)": [("middle", 0)],
    "middle + first output": [("middle", 0), ("output", 0)],
    "deep output (output 0-2)": [("output", 0), ("output", 1), ("output", 2)],
    "deep input (input 7-8)": [("input", 7), ("input", 8)],
    "all SDXL attention blocks": [("input", 4), ("input", 5), ("input", 7), ("input", 8), ("middle", 0),
                                  ("output", 0), ("output", 1), ("output", 2), ("output", 3), ("output", 4),
                                  ("output", 5)],
}


def _blur_tokens(q: torch.Tensor, shape: Optional[List[int]], sigma: float) -> torch.Tensor:
    """Gaussian-blur the query tokens over the image grid (SEG, Hong 2024, arXiv 2408.00760).
    sigma >= 100 stands for an infinite blur: every token gets the mean query."""
    B, N, C = q.shape
    if sigma >= 100.0:
        return q.mean(dim=1, keepdim=True).expand(B, N, C)
    H = W = None
    if shape is not None and len(shape) >= 4 and int(shape[-2]) * int(shape[-1]) == N:
        H, W = int(shape[-2]), int(shape[-1])
    if H is None:
        side = int(round(math.sqrt(N)))
        if side * side != N:
            return q          # unknown grid: leave the queries alone rather than guess
        H = W = side
    g = q.transpose(1, 2).reshape(B, C, H, W)
    g = cv.gaussian_blur2d(g.float(), sigma=float(sigma)).to(q.dtype)
    return g.reshape(B, C, N).transpose(1, 2)


def make_attention_patch(method: str, blur_sigma: float = 10.0, temperature: float = 2.0) -> Callable:
    """An attn1 replacement (q, k, v, extra_options) -> attention output for the perturbed pass.
      pag          output = v (each token attends only to itself; Ahn et al. 2024, arXiv 2403.17377)
      seg          attention with Gaussian-blurred queries (Hong 2024, arXiv 2408.00760)
      temperature  attention logits divided by the temperature (flatter for T > 1; the ERG-style perturbation)"""
    def patch(q, k, v, extra_options, mask=None):
        if method == "pag":
            return v
        import comfy.ldm.modules.attention as attention  # ComfyUI runtime only
        heads = extra_options["n_heads"]
        if method == "seg":
            q = _blur_tokens(q, extra_options.get("activations_shape"), blur_sigma)
        elif method == "temperature":
            q = q / max(float(temperature), 1e-3)
        else:
            raise ValueError(f"unknown weak-branch method {method!r}")
        return attention.optimized_attention(q, k, v, heads, attn_precision=extra_options.get("attn_precision"))
    return patch


def make_skip_patch(blocks) -> Callable:
    """An attn1_output_patch for UNet blocks: the self-attention sublayer of the chosen blocks adds nothing (its
    output, after the out projection, is zero), the attention-sublayer form of STG's residual skip."""
    chosen = {tuple(b) for b in blocks}

    def patch(n, extra_options):
        block = extra_options.get("block")
        return torch.zeros_like(n) if block is not None and tuple(block) in chosen else n
    return patch


# Anima and the other Cosmos-Predict2 DiTs take only "attn1_patch" patches, on the inputs of the attention projections
# (no replacement of the attention itself). Of the three methods only SEG carries over there: it blurs the query input
# of the chosen blocks over the image. PAG needs the attention output replaced, and a temperature on the queries is
# undone by the blocks' query normalization. The SDXL block choices map onto ranges of the DiT's depth.
DIT_RANGES = {"middle (PAG / SEG default)": "middle", "middle + first output": "middle",
              "deep output (output 0-2)": "last third", "deep input (input 7-8)": "first third",
              "all SDXL attention blocks": "all"}


def dit_model(model):
    """The Cosmos-Predict2 family DiT inside a ComfyUI model (Anima, Cosmos), or None."""
    dm = getattr(model, "diffusion_model", None)
    try:
        from comfy.ldm.cosmos.predict2 import MiniTrainDIT  # ComfyUI runtime only
    except Exception:
        return None
    return dm if isinstance(dm, MiniTrainDIT) else None


def dit_block_indices(n_blocks: int, which: str) -> set:
    if which == "all":
        return set(range(n_blocks))
    third = max(n_blocks // 3, 1)
    if which == "first third":
        return set(range(third))
    if which == "last third":
        return set(range(n_blocks - third, n_blocks))
    mid = n_blocks // 2
    return {mid - 1, mid} if n_blocks > 1 else {0}        # the two middle blocks


def weak_support_error(model, method: str) -> Optional[str]:
    """Why this weak-branch method cannot run on this model, or None when it can."""
    if dit_model(model) is not None:
        if method not in ("seg", "skip"):
            return (f"CFG Weak Branch: {method} cannot run on Anima / Cosmos DiTs (their blocks only take patches on "
                    f"the attention inputs: PAG needs the attention output replaced, and a query temperature is undone "
                    f"by the query normalization). Use SEG or skip there.")
        return None
    dm = getattr(model, "diffusion_model", None)
    if dm is not None and not hasattr(dm, "input_blocks"):
        return (f"CFG Weak Branch: {type(dm).__name__} is neither an SDXL / SD1.5 UNet nor an Anima / Cosmos DiT; "
                f"its attention cannot be perturbed by this node.")
    return None


def dit_token_grid(dit, x: torch.Tensor) -> Optional[Tuple[int, int, int]]:
    """The (frames, height, width) token grid a Cosmos-Predict2 DiT makes of the latent x (B, C, T, H, W): each
    axis padded up to the patch size, then divided by it."""
    if x.ndim != 5:
        return None
    pt, ps = int(getattr(dit, "patch_temporal", 1)), int(getattr(dit, "patch_spatial", 2))
    T, H, W = (int(v) for v in x.shape[-3:])
    return -(-T // pt), -(-H // ps), -(-W // ps)


def _blur_grid(q: torch.Tensor, sigma: float, grid: Optional[Tuple[int, int, int]]) -> torch.Tensor:
    """Blur a DiT block's attention input q (B, N, D) over the image, N = T H W tokens of the (T, H, W) grid, each
    frame on its own; sigma >= 100 stands for an infinite blur (every token gets its frame's mean). A grid that does
    not match N leaves the queries alone rather than guess."""
    B, N, D = q.shape
    if grid is None or grid[0] * grid[1] * grid[2] != N:
        return q
    T, H, W = grid
    g = q.reshape(B, T, H, W, D)
    if sigma >= 100.0:
        return g.mean(dim=(2, 3), keepdim=True).expand_as(g).reshape(B, N, D)
    y = cv.gaussian_blur2d(g.permute(0, 1, 4, 2, 3).float(), sigma=float(sigma)).to(q.dtype)
    return y.permute(0, 1, 3, 4, 2).reshape(B, N, D)


def make_dit_patch(indices: set, blur_sigma: float, grid: Optional[Tuple[int, int, int]], method: str = "seg") -> Callable:
    """An attn1_patch for Cosmos-Predict2 blocks in `indices`. The patch sees the inputs of the q / k / v projections.
      seg   blurs the query input over the image; the projection is linear per token, so this equals blurring the
            queries (the blocks' query normalization and rotary positions then act on the blurred queries)
      skip  zeroes the value input: the projections carry no bias, so the self-attention adds exactly nothing"""
    def patch(q, k, v, pe=None, attn_mask=None, extra_options=None):
        if (extra_options or {}).get("block_index") not in indices:
            return {}
        if method == "skip":
            return {"v": torch.zeros_like(v)}
        return {"q": _blur_grid(q, blur_sigma, grid)}
    return patch


def perturbed_prediction(weak: Dict[str, Any], model, input_cond, x, sigma, model_options) -> torch.Tensor:
    """One extra pass of the model on the conditional input with the chosen self-attention blocks perturbed."""
    import comfy.model_patcher  # ComfyUI runtime only
    import comfy.samplers
    error = weak_support_error(model, weak["method"])
    if error:
        raise ValueError(error)
    mo = dict(model_options)
    mo["transformer_options"] = dict(mo.get("transformer_options", {}))
    dit = dit_model(model)
    if dit is not None:
        indices = dit_block_indices(len(dit.blocks), DIT_RANGES.get(weak.get("blocks_label", ""), "middle"))
        patches = dict(mo["transformer_options"].get("patches", {}))
        patch = make_dit_patch(indices, weak.get("blur_sigma", 10.0), dit_token_grid(dit, x), weak["method"])
        patches["attn1_patch"] = list(patches.get("attn1_patch", [])) + [patch]
        mo["transformer_options"]["patches"] = patches
    elif weak["method"] == "skip":
        patches = dict(mo["transformer_options"].get("patches", {}))
        patches["attn1_output_patch"] = list(patches.get("attn1_output_patch", [])) + [make_skip_patch(weak["blocks"])]
        mo["transformer_options"]["patches"] = patches
    else:
        patch = make_attention_patch(weak["method"], weak.get("blur_sigma", 10.0), weak.get("temperature", 2.0))
        for block_name, number in weak["blocks"]:
            mo = comfy.model_patcher.set_model_options_patch_replace(mo, patch, "attn1", block_name, number)
    (out,) = comfy.samplers.calc_cond_batch(model, [input_cond], x, sigma, mo)
    return out


# ---------------------------------------------------------------------------
# Stage 3: mix
# ---------------------------------------------------------------------------

# rule -> (cfg_variants registry name, knob names the node passes)
_PARAMS_CACHE: Dict[str, frozenset] = {}


def _combiner_params(reg_name: str) -> frozenset:
    """The keyword names a library combiner accepts (cached)."""
    if reg_name not in _PARAMS_CACHE:
        import inspect
        _PARAMS_CACHE[reg_name] = frozenset(inspect.signature(cv.REGISTRY[reg_name]["fn"]).parameters)
    return _PARAMS_CACHE[reg_name]


MIX_RULES: Dict[str, Tuple[str, Tuple[str, ...]]] = {
    "standard": ("cfg", ()),
    "cfg_zero_star": ("cfg_zero_star", ("zero_init_steps",)),
    "power_law": ("power_law_cfg", ("alpha",)),
    "magnitude_damped": ("mambo_g", ("alpha",)),
    "apg": ("apg", ("eta", "norm_threshold", "momentum")),
    "tangential_damping": ("tcfg", ()),
    "angle_limit": ("adg", ("max_angle",)),
    "mahiro": ("mahiro", ()),
    "pentachoron": ("pentachoron", ("k",)),
}
THREE_WAY_RULES = ("perp_neg", "separate_negative", "negative_as_null")


def three_way(rule: str, c: torch.Tensor, u: torch.Tensor, n: torch.Tensor, w: float, neg_scale: float) -> torch.Tensor:
    """Guidance with a null prediction u and a separate negative n (all in one space).
      perp_neg           u + w ((c - u) - s perp),  perp = (n - u) minus its part along (c - u), per sample
                         (Armandpour et al. 2023, arXiv 2304.04968; ComfyUI's PerpNegGuider sums over the whole batch)
      separate_negative  u + w (c - u) - s (n - u)   (composable negation, Liu et al. 2022, arXiv 2206.01714)
      negative_as_null   n + w (c - n)               (classic CFG with the negative prompt as the unconditional)"""
    if rule == "perp_neg":
        pos, neg = c - u, n - u
        coef = _bview((pos * neg).flatten(1).sum(1) / pos.flatten(1).pow(2).sum(1).clamp_min(_EPS), pos)
        perp = neg - coef * pos
        return u + w * (pos - neg_scale * perp)
    if rule == "separate_negative":
        return u + w * (c - u) - neg_scale * (n - u)
    if rule == "negative_as_null":
        return n + w * (c - n)
    raise ValueError(f"unknown three-way rule {rule!r}")


# Per-sample shapes ------------------------------------------------------------

def _per_sample(v: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return v.reshape(v.shape[0], *([1] * (ref.ndim - 1)))


# ---------------------------------------------------------------------------
# Stage 4: where
# ---------------------------------------------------------------------------

def split_bands(g: torch.Tensor, method: str, blur_sigma: float, fft_cutoff: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """(low, high) with low + high == g. 'gaussian': low = Gaussian blur (sigma in latent pixels);
    'fft': low = the frequencies below fft_cutoff (a fraction of the Nyquist radius)."""
    if g.ndim != 4:
        return g, torch.zeros_like(g)
    if method == "gaussian":
        low = cv.gaussian_blur2d(g, sigma=float(blur_sigma))
    elif method == "fft":
        H, W = g.shape[-2:]
        fy = torch.fft.fftfreq(H, device=g.device).abs().view(H, 1) * 2
        fx = torch.fft.fftfreq(W, device=g.device).abs().view(1, W) * 2
        radius = torch.sqrt(fy * fy + fx * fx)
        keep = (radius <= float(fft_cutoff)).to(g.dtype)
        low = torch.fft.ifft2(torch.fft.fft2(g) * keep).real
    else:
        raise ValueError(f"unknown band method {method!r}")
    return low, g - low


def region_map(mask: torch.Tensor, like: torch.Tensor, inside: float, outside: float, feather: float,
               invert: bool) -> torch.Tensor:
    """A (B, 1, H, W) multiplier map: `inside` where the mask is 1, `outside` where it is 0."""
    m = mask.float().to(like.device)
    if m.ndim == 2:
        m = m.unsqueeze(0)
    m = m.unsqueeze(1)
    m = F.interpolate(m, size=like.shape[-2:], mode="bilinear", align_corners=False)
    if feather > 0:
        m = cv.gaussian_blur2d(m, sigma=float(feather))
    m = m.clamp(0, 1)
    if invert:
        m = 1 - m
    B = like.shape[0]
    if m.shape[0] != B:
        m = m[:1].expand(B, -1, -1, -1) if m.shape[0] == 1 else m.repeat((B + m.shape[0] - 1) // m.shape[0], 1, 1, 1)[:B]
    return outside + (inside - outside) * m


# ---------------------------------------------------------------------------
# Stage 5: correct
# ---------------------------------------------------------------------------

CORRECTIONS = ("rescale_std", "norm_cap", "channel_norm_match", "energy_preserve", "percentile_rescale", "soft_clip")
CORRECTION_SPACE = {"rescale_std": "v", "norm_cap": "x0", "channel_norm_match": "noise", "energy_preserve": "noise",
                    "percentile_rescale": "x0", "soft_clip": "x0"}


def _std_ps(a):
    return _per_sample(a.flatten(1).std(dim=1), a)


def apply_correction(spec: Dict[str, Any], d: torch.Tensor, d_c: torch.Tensor, d_u: Optional[torch.Tensor],
                     w: float) -> torch.Tensor:
    """One magnitude correction of the guided prediction d (all tensors in the correction's space).
    strength blends between the uncorrected (0) and fully corrected (1) prediction.
      rescale_std          d * std(d_c) / std(d)                                  (Lin et al. 2023, arXiv 2305.08891)
      norm_cap             shrink the push past d_c until ||d|| <= cap_ratio ||d_c||  (PMC-CFG, arXiv 2609.24287)
      channel_norm_match   per pixel, the channel norm of d set to that of d_c     (CFGNorm / Qwen-Image practice)
      energy_preserve      d * sqrt(E(d_c) / E(d)), E = sum of squares            (EP-CFG, arXiv 2412.09966)
      percentile_rescale   per channel, the spread of d pulled down to that of a lower 'mimic' scale
                           (the idea of mcmonkey's dynamic thresholding; shrink only)
      soft_clip            the per-pixel size of the push past d_c tone-mapped (Reinhard; ComfyUI's tonemap op)"""
    method = spec["method"]
    s = float(spec.get("strength", 1.0))
    if method == "rescale_std":
        corr = d * (_std_ps(d_c) / _std_ps(d).clamp_min(_EPS))
    elif method == "norm_cap":
        G = d - d_c
        a = G.flatten(1).pow(2).sum(1)
        b = (d_c * G).flatten(1).sum(1)
        q = d_c.flatten(1).pow(2).sum(1)
        cap = float(spec.get("cap_ratio", 1.05))
        disc = (b * b + (cap * cap - 1) * a * q).clamp_min(0)
        beta = torch.where(a > 0, (-b + disc.sqrt()) / a.clamp_min(_EPS), torch.ones_like(a))
        over = (d.flatten(1).norm(dim=1) > cap * d_c.flatten(1).norm(dim=1))
        beta = torch.where(over, beta.clamp(0, 1), torch.ones_like(beta))
        corr = d_c + _per_sample(beta, d) * G
    elif method == "channel_norm_match":
        nc = torch.linalg.vector_norm(d_c, dim=1, keepdim=True)
        nd = torch.linalg.vector_norm(d, dim=1, keepdim=True)
        corr = d * torch.where(nd > 0, nc / nd.clamp_min(_EPS), torch.ones_like(nd))
    elif method == "energy_preserve":
        ec = d_c.flatten(1).pow(2).sum(1)
        ed = d.flatten(1).pow(2).sum(1).clamp_min(_EPS)
        corr = d * _per_sample((ec / ed).sqrt(), d)
    elif method == "percentile_rescale":
        if d_u is None:
            return d
        ref = d_u + float(spec.get("mimic_scale", 4.0)) * (d_c - d_u)
        p = float(spec.get("percentile", 0.995))
        B, C = d.shape[:2]
        dm = d.flatten(2).mean(-1, keepdim=True)
        rm = ref.flatten(2).mean(-1, keepdim=True)
        sd = torch.quantile((d.flatten(2) - dm).abs(), p, dim=-1, keepdim=True)
        sr = torch.quantile((ref.flatten(2) - rm).abs(), p, dim=-1, keepdim=True)
        k = (sr / sd.clamp_min(_EPS)).clamp(max=1.0)
        corr = ((d.flatten(2) - dm) * k + dm).reshape(d.shape)
    elif method == "soft_clip":
        G = d - d_c
        mag = torch.linalg.vector_norm(G, dim=1, keepdim=True) + 1e-10
        dims = tuple(range(1, mag.ndim))
        top = (mag.std(dim=dims, keepdim=True) * 5 + mag.mean(dim=dims, keepdim=True)) * float(spec.get("softness", 1.0))
        m = mag / top.clamp_min(_EPS)
        corr = d_c + G * ((m / (m + 1)) * top / mag)
    else:
        raise ValueError(f"unknown correction {method!r}")
    return s * corr + (1 - s) * d


# ---------------------------------------------------------------------------
# Stage 6: govern (the angle band)
# ---------------------------------------------------------------------------

GOVERN_UNITS = {"whole image (one vector per image)": "image",
                "each pixel (its channel vector)": "pixel",
                "each channel (its spatial map)": "channel"}
GOVERN_HOMES = {"conditional (how far guidance turns the prediction)": "cond",
                "unconditional (how far the prediction sits from the negative)": "uncond"}
_NO_PLANE = 1e-6          # sin(angle) below this counts as no direction to turn along (float32 rounding level)


def _to_units(a: torch.Tensor, unit: str) -> torch.Tensor:
    """Rows of vectors: one per image, one per pixel (over the channels) or one per channel (over the positions)."""
    if unit == "image":
        return a.reshape(a.shape[0], -1)
    if unit == "pixel":
        return a.movedim(1, -1).reshape(-1, a.shape[1])
    if unit == "channel":
        return a.reshape(a.shape[0] * a.shape[1], -1)
    raise ValueError(f"unknown governor unit {unit!r}")


def _from_units(v: torch.Tensor, unit: str, like: torch.Tensor) -> torch.Tensor:
    if unit == "pixel":
        return v.reshape(like.shape[0], *like.shape[2:], like.shape[1]).movedim(-1, 1)
    return v.reshape(like.shape)


def _angles(D: torch.Tensor, H: torch.Tensor):
    """Row-wise angle between D and H (radians, float64), with the pieces the turn needs."""
    nd, nh = D.norm(dim=1), H.norm(dim=1)
    hh = H / nh.clamp_min(1e-30).unsqueeze(1)
    along = (D * hh).sum(1)
    perp = D - along.unsqueeze(1) * hh
    sin_n = perp.norm(dim=1)                                  # |d| sin(angle)
    return torch.atan2(sin_n, along), nd, nh, hh, perp, sin_n


def govern_angles(d: torch.Tensor, home: torch.Tensor, min_deg: float, max_deg: float, unit: str):
    """The angle band, after the AlephLM anchor governor: a projection (never a reweighting), identity until a bound
    binds, only the offending units move, each exactly onto its bound.
    Per unit, psi = angle(d, home) in [0, 180] degrees (signs matter here, unlike the codebook governor's projective
    form). psi > max_deg turns d toward home (the leash); psi < min_deg turns d away from home along its own
    direction (the floor). The turn stays in the plane d spans with home and keeps the length of d:
        d' = |d| (cos(target) h + sin(target) t),   h = home / |home|,   t = (d - (d.h) h) / |d - (d.h) h|
    A unit with no plane (d along home, a zero vector, or d exactly opposite home) is left alone: with no guidance
    there is no direction to turn along. Float64 inside; units within the band come back bit-identical.
    Returns (governed d, fired mask shaped like d, census)."""
    D, Hm = _to_units(d, unit).double(), _to_units(home, unit).double()
    psi, nd, nh, hh, perp, sin_n = _angles(D, Hm)
    valid = (nd > 0) & (nh > 0)
    plane = valid & (sin_n > _NO_PLANE * nd)
    lo, hi = math.radians(float(min_deg)), math.radians(float(max_deg))
    over, under = plane & (psi > hi), plane & (psi < lo)
    fire = over | under
    target = torch.where(over, torch.full_like(psi, hi), torch.full_like(psi, lo))
    t = perp / sin_n.clamp_min(1e-30).unsqueeze(1)
    turned = nd.unsqueeze(1) * (torch.cos(target).unsqueeze(1) * hh + torch.sin(target).unsqueeze(1) * t)
    out = torch.where(fire.unsqueeze(1), turned, D)
    n_valid = max(int(valid.sum()), 1)
    deg = torch.rad2deg(psi[valid]) if bool(valid.any()) else torch.zeros(1, dtype=psi.dtype)
    census = {"gov_ceiling_share": float(over.sum()) / n_valid, "gov_floor_share": float(under.sum()) / n_valid,
              "gov_angle_mean_deg": float(deg.mean()), "gov_angle_min_deg": float(deg.min()),
              "gov_angle_max_deg": float(deg.max())}
    return _from_units(out, unit, d).to(d.dtype), _from_units(fire.unsqueeze(1).expand_as(D), unit, d), census


# ---------------------------------------------------------------------------
# Stage 7: measure
# ---------------------------------------------------------------------------

def _rms(a):
    return a.flatten(1).pow(2).mean(1).sqrt()


def _cos_ps(a, b):
    a, b = a.flatten(1), b.flatten(1)
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1)).clamp_min(_EPS)


def _angle_deg(a, b, unit):
    """Angles between a and b in degrees, one per unit (float64)."""
    return torch.rad2deg(_angles(_to_units(a, unit).double(), _to_units(b, unit).double())[0])


def step_stats(x0_c, x0_u, x0_hat, sig, w, x0_weak=None) -> Dict[str, float]:
    """Per-step numbers (means over the batch). Sizes are root-mean-square per element.
      delta_rms / delta_eps_rms   size of x0_c - x0_u, and of the same difference in noise units (divided by sigma)
      cos_c_u                     cosine between the two denoised predictions
      parallel_share              share of the difference's energy along x0_c (1 = pure rescaling of x0_c)
      push_ratio                  ||x0_hat - x0_c|| / ||x0_c||, how far guidance moved the estimate
      std_ratio                   std(x0_hat) / std(x0_c), the over-saturation signal guidance rescale corrects
      lowfreq_share               share of the difference's energy below a sigma-2 Gaussian blur (structure vs detail)
      x0_hat_p99 / x0_c_p99       99th percentile of |x0| (range blow-up)
      angle_hat_c_deg             angle between the guided and the conditional prediction, whole image (the governor's
                                  leash reads this angle); angle_hat_c_pixel_p50_deg / _p95_deg: the same per pixel
      angle_c_u_deg               angle between the conditional and the unconditional prediction, whole image
      weak_rms / cos_delta_weak   size of x0_c - x0_weak and its cosine with the text difference (when a weak branch runs)
    Governor columns (gov_*) come from the govern stage itself, in its own unit and space."""
    out: Dict[str, float] = {"w": float(w)}
    G = x0_hat - x0_c
    nc = x0_c.flatten(1).norm(dim=1).clamp_min(_EPS)
    out["push_ratio"] = float((G.flatten(1).norm(dim=1) / nc).mean())
    out["angle_hat_c_deg"] = float(_angle_deg(x0_hat, x0_c, "image").mean())
    if x0_c.ndim >= 3:
        px = _angle_deg(x0_hat, x0_c, "pixel").float()
        out["angle_hat_c_pixel_p50_deg"] = float(torch.quantile(px, 0.5))
        out["angle_hat_c_pixel_p95_deg"] = float(torch.quantile(px, 0.95))
    out["std_ratio"] = float((x0_hat.flatten(1).std(1) / x0_c.flatten(1).std(1).clamp_min(_EPS)).mean())
    out["x0_hat_p99"] = float(torch.quantile(x0_hat.abs().flatten(1).float(), 0.99, dim=1).mean())
    out["x0_c_p99"] = float(torch.quantile(x0_c.abs().flatten(1).float(), 0.99, dim=1).mean())
    if x0_u is not None:
        delta = x0_c - x0_u
        out["delta_rms"] = float(_rms(delta).mean())
        out["delta_eps_rms"] = float((_rms(delta) / sig.flatten().clamp_min(_EPS)).mean())
        out["cos_c_u"] = float(_cos_ps(x0_c, x0_u).mean())
        out["angle_c_u_deg"] = float(_angle_deg(x0_c, x0_u, "image").mean())
        par = (delta.flatten(1) * x0_c.flatten(1)).sum(1) ** 2 / (nc ** 2)
        out["parallel_share"] = float((par / delta.flatten(1).pow(2).sum(1).clamp_min(_EPS)).mean())
        if delta.ndim == 4:
            low = cv.gaussian_blur2d(delta, sigma=2.0)
            out["lowfreq_share"] = float((low.flatten(1).pow(2).sum(1) / delta.flatten(1).pow(2).sum(1).clamp_min(_EPS)).mean())
    if x0_weak is not None:
        wd = x0_c - x0_weak
        out["weak_rms"] = float(_rms(wd).mean())
        if x0_u is not None:
            out["cos_delta_weak"] = float(_cos_ps(x0_c - x0_u, wd).mean())
    return out


class ProbeWriter:
    """Appends one JSON line per sampler step to <folder>/<prefix>_<date>_<run>.jsonl; a new file per run."""

    def __init__(self, folder: str, prefix: str, print_every: int = 0):
        self.folder, self.prefix, self.print_every = folder, prefix, int(print_every)
        self.path: Optional[str] = None
        self.run = 0

    def new_run(self, header: Dict[str, Any]) -> None:
        os.makedirs(self.folder, exist_ok=True)
        self.run += 1
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.path = os.path.join(self.folder, f"{self.prefix}_{stamp}_{self.run:03d}.jsonl")
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"header": header}) + "\n")

    def write(self, row: Dict[str, Any]) -> None:
        if self.path is None:
            self.new_run({})
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        if self.print_every > 0 and int(row.get("step", 0)) % self.print_every == 0:
            short = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}
            print(f"[cfg probe] {short}")


def probe_folder() -> str:
    try:
        import folder_paths  # ComfyUI runtime
        return os.path.join(folder_paths.get_output_directory(), "cfg_probe")
    except Exception:
        return os.path.join(os.getcwd(), "cfg_probe")


# ---------------------------------------------------------------------------
# The runtime
# ---------------------------------------------------------------------------

class GuidanceRuntime:
    """Executes a plan at every sampler step. One instance per installed plan; per-run state (APG momentum, the
    probe file) resets whenever sigma rises, which marks a new sampling run."""

    _is_cfg_prototypes_runtime = True

    def __init__(self, plan: Dict[str, Any]):
        self.plan = plan
        self.prev_sigma: Optional[float] = None
        self.mix_state = None
        self._mix_state_for = None
        meas = plan.get("measure")
        self.probe = ProbeWriter(meas.get("folder") or probe_folder(), meas.get("prefix", "cfg_probe"),
                                 meas.get("print_every", 0)) if meas else None
        self._window_cache: Dict[Any, Tuple[float, float]] = {}
        self.last: Dict[str, Any] = {}
        self._exact = None      # (the step's input, its guided x0) from cfg_function, for post_cfg_function

    # -- run bookkeeping -------------------------------------------------------
    def _new_run(self) -> None:
        self.mix_state = None
        self._mix_state_for = None
        if self.probe is not None:
            self.probe.new_run({"plan": describe_plan(self.plan)})

    def _check_run(self, sigma0: float) -> None:
        if self.prev_sigma is None or sigma0 > self.prev_sigma * (1 + 1e-5) + 1e-8:
            self._new_run()
        self.prev_sigma = sigma0

    def may_skip_uncond(self) -> bool:
        """True when some steps never need the unconditional pass (saves one model evaluation there)."""
        when, weak = self.plan.get("when"), self.plan.get("weak")
        window_skip = bool(when and when.get("outside") == "cond" and when.get("skip_uncond", True)
                           and (when.get("start", 0.0) > 0.0 or when.get("end", 1.0) < 1.0))
        replace_skip = bool(weak and weak.get("mode") == "replace" and not self.plan.get("measure"))
        return window_skip or replace_skip

    # -- the window --------------------------------------------------------------
    def _window(self, model) -> Tuple[float, float]:
        """(sigma_lo, sigma_hi) of the active window, from the model's own percent -> sigma map (shift-aware)."""
        when = self.plan.get("when")
        if not when:
            return 0.0, float("inf")
        ms = getattr(model, "model_sampling", None)
        key = id(ms)
        edm = when.get("sigma_edm")
        if key not in self._window_cache and edm is not None:
            # a window in EDM sigma units (the guidance interval paper): used as is on eps models, whose ComfyUI
            # sigma is the EDM sigma; flow models see the noise level sigma / (1 + sigma)
            flow = is_flow_sampling(ms) if ms is not None else bool(self.plan.get("flow_hint", False))
            lo_e, hi_e = float(edm[0]), float(edm[1])
            if flow:
                lo_e, hi_e = lo_e / (1.0 + lo_e), (hi_e / (1.0 + hi_e) if math.isfinite(hi_e) else float("inf"))
            self._window_cache[key] = (lo_e, hi_e)
        if key not in self._window_cache:
            start, end = float(when.get("start", 0.0)), float(when.get("end", 1.0))
            if ms is None or not hasattr(ms, "percent_to_sigma"):
                hi = float("inf") if start <= 0 else float(when.get("sigma_hi", float("inf")))
                lo = 0.0 if end >= 1 else float(when.get("sigma_lo", 0.0))
            else:
                hi = float("inf") if start <= 0.0 else float(ms.percent_to_sigma(start))
                lo = 0.0 if end >= 1.0 else float(ms.percent_to_sigma(end))
            self._window_cache[key] = (lo, hi)
        return self._window_cache[key]

    def _inside(self, sig: torch.Tensor, model) -> torch.Tensor:
        lo, hi = self._window(model)
        return (sig >= lo) & (sig <= hi)

    def _outside(self, x0_c, x0_u, w_base):
        """Outside the window when the chain does not run there: the conditional alone, or plain CFG at the base
        scale (the 'plain CFG at base scale' mode)."""
        mode = (self.plan.get("when") or {}).get("outside", "cond")
        if mode != "plain" or x0_u is None:
            return x0_c
        return x0_u + w_base * (x0_c - x0_u)

    def _govern_on(self, gov: Dict[str, Any], sig0: float, prog: float, model) -> bool:
        """The governor's own window (shift-aware like the When window; by progress when no model is given)."""
        start, end = float(gov.get("start", 0.0)), float(gov.get("end", 1.0))
        if start <= 0.0 and end >= 1.0:
            return True
        ms = getattr(model, "model_sampling", None)
        if ms is None or not hasattr(ms, "percent_to_sigma"):
            return start <= prog <= end
        key = ("govern", id(ms))
        if key not in self._window_cache:
            hi = float("inf") if start <= 0.0 else float(ms.percent_to_sigma(start))
            lo = 0.0 if end >= 1.0 else float(ms.percent_to_sigma(end))
            self._window_cache[key] = (lo, hi)
        lo, hi = self._window_cache[key]
        return lo <= sig0 <= hi

    # -- one step ------------------------------------------------------------------
    def guided_x0(self, x, x0_c, x0_u, sigma, cond_scale, model=None, model_options=None, input_cond=None,
                  x0_n=None, three_way_rule=None, neg_scale=1.0, uncond_valid=True, three_way_knobs=None) -> torch.Tensor:
        plan = self.plan
        out_dtype = x0_c.dtype
        B = x.shape[0]
        # a single-frame 5-D latent (Anima and the other Wan-VAE image models): every stage works on (B, C, H, W),
        # the frame axis comes back at the end
        frame = x.ndim == 5 and x.shape[2] == 1
        if frame:
            x0_c = x0_c.squeeze(2)
            x0_u = x0_u.squeeze(2) if x0_u is not None else None
            x0_n = x0_n.squeeze(2) if x0_n is not None else None
        xs, c = (x.squeeze(2) if frame else x).float(), x0_c.float()
        u = x0_u.float() if (x0_u is not None and uncond_valid) else None
        sig = torch.as_tensor(sigma, dtype=torch.float32, device=xs.device).reshape(-1)
        if sig.numel() == 1 and B > 1:
            sig = sig.expand(B)
        sig0 = float(sig[0])
        self._check_run(sig0)
        ms = getattr(model, "model_sampling", None)
        flow = is_flow_sampling(ms) if ms is not None else bool(plan.get("flow_hint", False))
        to = (model_options or {}).get("transformer_options", {}) or {}
        ss = to.get("sample_sigmas")
        if ss is not None and len(ss) >= 2:
            step, n_steps, prog = progress_from_sigmas(sig0, ss)
            times = [noise_level(float(s), flow) for s in (ss.tolist() if torch.is_tensor(ss) else ss)]
        else:
            step, n_steps, prog, times = 0.0, 1, 1.0 - noise_level(sig0, flow), None
        t_now = noise_level(sig0, flow)
        mix = plan.get("mix") or {}
        w_base = float(cond_scale) if float(mix.get("scale", -1.0)) < 0 else float(mix["scale"])
        s = {"x": x, "xs": xs, "c": c, "u": u, "n": x0_n.float() if x0_n is not None else None, "sig": sig,
             "sig0": sig0, "sigv": _bview(sig.clamp_min(_EPS), xs), "flow": flow, "model": model,
             "model_options": model_options, "input_cond": input_cond, "sigma": sigma, "prog": prog, "step": step,
             "n_steps": n_steps, "t_now": t_now, "three_way_rule": three_way_rule, "neg_scale": neg_scale,
             "frame": frame, "three_way_knobs": three_way_knobs}

        # 1 when: the scheduled scale inside the window (the shape spans the window); outside it, the rest of the
        # chain at the outside scale, or no guidance, or plain CFG ------------------------------------------------
        when = plan.get("when")
        w = w_base
        inside = torch.ones(B, dtype=torch.bool, device=xs.device)
        mode = "base"
        if when:
            p_w, step_w, times_w = self._window_progress(sig0, ss, prog, step, times, model)
            w = schedule_value(when.get("shape", "constant"), w_base, p_w, t_now, step_w, times_w,
                               float(when.get("a", -1.0)), float(when.get("b", -1.0)))
            if float(when.get("floor", 0.0)) > 0:
                w = max(w, float(when["floor"]))
            inside = self._inside(sig, model).to(xs.device)
            mode = when.get("outside", "cond")
        w_out = float(when.get("outside_scale", 1.0)) if (when and mode == "fixed") else w_base
        weak = plan.get("weak") or {}
        chain_outside = mode in ("base", "fixed") and (u is not None or weak.get("mode") == "replace")
        record = {"step": int(math.floor(step + 1e-6)), "steps": n_steps, "sigma": sig0, "progress": prog}
        if not bool(inside.any()):
            if not chain_outside:
                x0_hat = self._outside(c, u, w_base)
                w_used = w_base if (mode == "plain" and u is not None) else 1.0
                self._measure(record | {"inside": False}, c, u, x0_hat, s["sigv"], w_used, None)
                self.last = {"w": w_used, "step": record["step"], "sigma": sig0}
                return x0_hat.to(out_dtype).unsqueeze(2) if frame else x0_hat.to(out_dtype)
            w = w_out
        x0_hat, x0_w, u_used = self._chain(s, w, record)
        if not bool(inside.all()):          # samples on both sides of an edge (per-sample sigmas; rare)
            other = self._chain(s, w_out, {})[0] if chain_outside else self._outside(c, u, w_base)
            x0_hat = torch.where(_bview(inside, xs), x0_hat, other)

        # 7 measure -------------------------------------------------------------------------------
        self._measure(record | {"inside": bool(inside.any())}, c, u_used if uncond_valid else None, x0_hat,
                      s["sigv"], w, x0_w)
        self.last = {"w": w, "step": record["step"], "sigma": sig0}
        return x0_hat.to(out_dtype).unsqueeze(2) if frame else x0_hat.to(out_dtype)

    def _window_progress(self, sig0: float, ss, prog: float, step: float, times, model):
        """(progress through the window, step index inside it, the window's noise-level grid): a schedule shape
        runs from the window's first step to its last instead of over the whole run. The window's edges stay in
        ComfyUI's percent convention (by noise level, shift-aware); the steps inside it are read off the sampler's
        own sigmas."""
        when = self.plan["when"]
        start, end = float(when.get("start", 0.0)), float(when.get("end", 1.0))
        if start <= 0.0 and end >= 1.0:
            return prog, step, times
        if ss is None or times is None:
            return min(max((prog - start) / max(end - start, 1e-6), 0.0), 1.0), step, None
        lo, hi = self._window(model)
        grid = [float(v) for v in (ss.tolist() if torch.is_tensor(ss) else ss)]
        idx = [i for i in range(len(grid) - 1) if lo <= grid[i] <= hi]
        if not idx:
            return prog, step, times
        i0, i1 = idx[0], idx[-1]
        p_w = 0.0 if i1 == i0 else min(max((step - i0) / (i1 - i0), 0.0), 1.0)
        return p_w, step - i0, times[i0:i1 + 2]

    def _chain(self, s: Dict[str, Any], w: float, record: Dict[str, Any]):
        """Stages 2-6 at the scale w: weak branch, mix, where, correct, govern.
        Returns (x0_hat, the weak prediction or None, the unconditional as used)."""
        plan = self.plan
        xs, c, u, n, sigv, flow, model = s["xs"], s["c"], s["u"], s["n"], s["sigv"], s["flow"], s["model"]
        model_options = s["model_options"]
        mix = plan.get("mix") or {}
        step_i = int(math.floor(s["step"] + 1e-6))

        # 2 weak ---------------------------------------------------------------------
        weak = plan.get("weak")
        x0_w = None
        weak_on = bool(weak) and (weak.get("mode") == "replace" or float(weak.get("scale", 0.0)) != 0.0)
        if weak_on and model is not None and s["input_cond"] is not None:
            x0_w = perturbed_prediction(weak, model, s["input_cond"], s["x"], s["sigma"], model_options or {}).float()
            if s["frame"]:
                x0_w = x0_w.squeeze(2)
            if weak.get("mode") == "replace":
                u = x0_w

        # 3 mix ------------------------------------------------------------------------
        if s["three_way_rule"] is not None and s["three_way_rule"].startswith("registry:"):
            reg_name = s["three_way_rule"].split(":", 1)[1]          # a paper guider: a library rule with d_n
            space = plan.get("three_way_space", "auto")
            if space == "auto":
                space = native_space(cv.REGISTRY[reg_name]["space"], flow)
            dc, du, dn = (to_space(t_, xs, sigv, space, flow) for t_ in (c, u if u is not None else c, n))
            knobs = s.get("three_way_knobs") or {}
            if cv.REGISTRY[reg_name].get("state") and self._mix_state_for != "three:" + reg_name:
                self.mix_state = cv.make_state(reg_name, **knobs)
                self._mix_state_for = "three:" + reg_name
            ctx = self._combiner_ctx(reg_name, s, space, c, u, step_i, d_n=dn)
            x0_hat = from_space(cv.apply_variant(reg_name, dc, du, w=w, ctx=ctx, **knobs), xs, sigv, space, flow)
        elif s["three_way_rule"] is not None:
            space = plan.get("three_way_space", "x0")
            if space == "auto":
                space = "x0"
            dc, du, dn = (to_space(t_, xs, sigv, space, flow) for t_ in (c, u if u is not None else c, n))
            x0_hat = from_space(three_way(s["three_way_rule"], dc, du, dn, w, s["neg_scale"]), xs, sigv, space, flow)
        else:
            if u is None:
                x0_hat = c
            else:
                if mix.get("kind") == "registry":       # a paper node: any combiner of the library
                    reg_name = mix["registry"]
                    state_key = "registry:" + reg_name
                else:
                    reg_name, _ = MIX_RULES[mix.get("rule", "standard")]
                    state_key = mix.get("rule", "standard")
                space = mix.get("space", "auto")
                if space == "auto":
                    space = native_space(cv.REGISTRY[reg_name]["space"], flow)
                dc, du = to_space(c, xs, sigv, space, flow), to_space(u, xs, sigv, space, flow)
                if cv.REGISTRY[reg_name].get("state") and self._mix_state_for != state_key:
                    self.mix_state = cv.make_state(reg_name, **mix.get("knobs", {}))
                    self._mix_state_for = state_key
                ctx = self._combiner_ctx(reg_name, s, space, c, u, step_i)
                d_hat = cv.apply_variant(reg_name, dc, du, w=w, ctx=ctx, **mix.get("knobs", {}))
                x0_hat = from_space(d_hat, xs, sigv, space, flow)
        if weak and x0_w is not None and weak.get("mode", "add") == "add":
            x0_hat = x0_hat + float(weak["scale"]) * (c - x0_w)

        # 4 where --------------------------------------------------------------------------
        # (a factor of 1 everywhere leaves the prediction untouched, bit for bit: c + (x0_hat - c) would round)
        bands, region = plan.get("bands"), plan.get("region")
        G = None
        if bands and (float(bands.get("low", 1.0)) != 1.0 or float(bands.get("high", 1.0)) != 1.0):
            low, high = split_bands(x0_hat - c, bands.get("method", "gaussian"), bands.get("blur_sigma", 2.0),
                                    bands.get("fft_cutoff", 0.25))
            G = float(bands.get("low", 1.0)) * low + float(bands.get("high", 1.0)) * high
        if region and region.get("mask") is not None:
            like = x0_hat - c if G is None else G
            m = region_map(region["mask"], like, float(region.get("inside", 1.0)), float(region.get("outside", 0.0)),
                           float(region.get("feather", 0.0)), bool(region.get("invert", False)))
            if not bool((m == 1.0).all()):
                G = like * m
        if G is not None:
            x0_hat = c + G

        # 5 correct ---------------------------------------------------------------------------
        for spec in plan.get("correct") or []:
            space = spec.get("space", "auto")
            if space == "auto":
                space = native_space(CORRECTION_SPACE[spec["method"]], flow)
            dh = to_space(x0_hat, xs, sigv, space, flow)
            dc = to_space(c, xs, sigv, space, flow)
            du = to_space(u, xs, sigv, space, flow) if u is not None else None
            x0_hat = from_space(apply_correction(spec, dh, dc, du, w), xs, sigv, space, flow)

        # 6 govern ------------------------------------------------------------------------------------
        gov = plan.get("govern")
        if gov:
            home = c if gov.get("home", "cond") == "cond" else u
            if home is not None and self._govern_on(gov, s["sig0"], s["prog"], model):
                space = gov.get("space", "auto")
                space = "x0" if space == "auto" else space
                governed, fired, census = govern_angles(to_space(x0_hat, xs, sigv, space, flow),
                                                        to_space(home, xs, sigv, space, flow),
                                                        gov.get("min_deg", 0.0), gov.get("max_deg", 180.0),
                                                        gov.get("unit", "image"))
                if bool(fired.any()):       # only offenders move; every other element keeps its exact value
                    x0_hat = torch.where(fired, from_space(governed, xs, sigv, space, flow).to(x0_hat.dtype), x0_hat)
                record.update(census)
        return x0_hat, x0_w, u

    def _combiner_ctx(self, reg_name: str, s: Dict[str, Any], space: str, c, u, step_i: int, **extra) -> Dict[str, Any]:
        """What a library combiner may read besides the two predictions (it takes only the keywords it names):
        the latent in the sampler's coordinates, sigma per sample, alpha (1 on eps models, 1 - sigma on flow
        models: x = alpha x0 + sigma n), the noise level t (1 = noise), progress, step and step count, the
        sampler's sigmas, the per-run state, both predictions as x0 and as noise, and the model kind."""
        xs, sigv, flow = s["xs"], s["sigv"], s["flow"]
        ctx = {"x_t": xs, "sigma": s["sig"], "t": s["t_now"], "progress": s["prog"], "step": step_i,
               "num_steps": s["n_steps"], "state": self.mix_state, "space": "x0" if space == "x0" else "noise",
               "flow": flow, "alpha": (1.0 - s["sig"]) if flow else torch.ones_like(s["sig"])}
        params = _combiner_params(reg_name)
        if "sigmas" in params:
            ss = ((s.get("model_options") or {}).get("transformer_options") or {}).get("sample_sigmas")
            ctx["sigmas"] = ss if ss is not None else None
        if "x0_c" in params or "x0_u" in params:
            ctx.update({"x0_c": c, "x0_u": u})
        if "n_c" in params or "n_u" in params:
            ctx.update({"n_c": to_space(c, xs, sigv, "eps", flow).to(c.dtype),
                        "n_u": to_space(u, xs, sigv, "eps", flow).to(c.dtype) if u is not None else None})
        ctx.update(extra)
        return ctx

    def _measure(self, record, c, u, x0_hat, sigv, w, x0_w) -> None:
        if self.probe is None:
            return
        row = dict(record)
        row.update(step_stats(c, u, x0_hat, sigv, w, x0_w))
        self.probe.write(row)

    # -- ComfyUI hooks ---------------------------------------------------------------------
    def cfg_function(self, args):
        x = args["input"]
        # the unconditional prediction is real unless ComfyUI skipped it (cfg 1 shortcut) or this pack dropped it
        valid = args.get("input_uncond") is not None
        if valid and self.may_skip_uncond() and not foreign_cfg_hooks(args.get("model_options")) and \
                self._uncond_unused(args["sigma"], args.get("model")):
            valid = False
        x0 = self.guided_x0(x, args["cond_denoised"], args["uncond_denoised"], args["sigma"], args["cond_scale"],
                            model=args.get("model"), model_options=args.get("model_options"),
                            input_cond=args.get("input_cond"), uncond_valid=valid)
        self._exact = (x, x0)
        return x - x0

    def post_cfg_function(self, args):
        """ComfyUI turns the CFG function's x - x0 back into x0 as x - (x - x0), which rounds at the latent's scale.
        Euler-type samplers read x - x0 and never see it; samplers that read the denoised image itself (the SDE and
        multistep ones) drift from plain sampling by that rounding (a few pixel levels on Anima with er_sde). This
        function runs first among the post-CFG functions and hands on the exact x0."""
        exact, self._exact = self._exact, None
        if exact is not None and exact[0] is args.get("input") and exact[1].shape == args["denoised"].shape:
            return exact[1]
        return args["denoised"]

    def make_calc_cond_batch_function(self, previous=None):
        """Drops the unconditional pass on steps that never read it (outside a 'no guidance' window, or when the
        weak branch replaces the unconditional)."""
        runtime = self

        def calc_cond_batch_function(args):
            conds = args["conds"]
            if len(conds) > 1 and conds[1] is not None and not foreign_cfg_hooks(args.get("model_options")) and \
                    runtime._uncond_unused(args["sigma"], args["model"]):
                args = dict(args)
                args["conds"] = [conds[0], None] + list(conds[2:])
            if previous is not None:
                return previous(args)
            import comfy.samplers  # ComfyUI runtime only
            return comfy.samplers.calc_cond_batch(args["model"], args["conds"], args["input"], args["sigma"],
                                                  args["model_options"])

        calc_cond_batch_function._cfg_prototypes = True
        calc_cond_batch_function._previous = previous
        return calc_cond_batch_function

    def _uncond_unused(self, sigma, model) -> bool:
        plan = self.plan
        weak = plan.get("weak")
        if weak and weak.get("mode") == "replace" and not plan.get("measure"):
            return True
        when = plan.get("when")
        if when and when.get("outside") == "cond" and when.get("skip_uncond", True):
            sig = torch.as_tensor(sigma, dtype=torch.float32).reshape(-1)
            return not bool(self._inside(sig, model).any())
        return False


# ---------------------------------------------------------------------------
# Readout
# ---------------------------------------------------------------------------

def describe_plan(plan: Optional[Dict[str, Any]]) -> str:
    """The plan in plain lines, in execution order."""
    if not plan:
        return "no CFG Megapack plan on this model (plain CFG)"
    lines = []
    when = plan.get("when")
    if when:
        window = "whole run" if when.get("start", 0) <= 0 and when.get("end", 1) >= 1 else \
            f"{when.get('start', 0):.2f}-{when.get('end', 1):.2f} of the run"
        if when.get("sigma_edm") is not None:
            lo_e, hi_e = when["sigma_edm"]
            window = f"sigma {lo_e:g} to {hi_e:g} (EDM units; flow models: noise level sigma / (1 + sigma))"
        outside = {v: k for k, v in OUTSIDE_MODES.items()}.get(when.get("outside", "cond"), when.get("outside"))
        lines.append(f"1 when: shape {when.get('shape', 'constant')}, window {window}, outside: {outside}"
                     + (f" ({when.get('outside_scale')})" if when.get("outside") == "fixed" else "")
                     + (f", floor {when['floor']}" if float(when.get("floor", 0)) > 0 else ""))
    else:
        lines.append("1 when: constant scale, whole run")
    weak = plan.get("weak")
    if weak:
        blocks = ", ".join(f"{b} {i}" for b, i in weak["blocks"])
        dit = DIT_RANGES.get(weak.get("blocks_label", ""), "middle")
        lines.append(f"2 weak: {weak['method']} on [{blocks}] (on an Anima / Cosmos DiT: its {dit} blocks), "
                     f"scale {weak['scale']}, mode {weak.get('mode', 'add')}"
                     + (f", blur sigma {weak.get('blur_sigma')}" if weak["method"] == "seg" else "")
                     + (f", temperature {weak.get('temperature')}" if weak["method"] == "temperature" else ""))
    else:
        lines.append("2 weak: none")
    mix = plan.get("mix")
    if mix:
        scale = "sampler cfg" if float(mix.get("scale", -1)) < 0 else mix["scale"]
        name = mix.get("title") or (mix["registry"] if mix.get("kind") == "registry" else mix.get("rule", "standard"))
        lines.append(f"3 mix: {name} ({mix.get('space', 'auto')} space), scale {scale}, "
                     f"knobs {mix.get('knobs', {})}")
    else:
        lines.append("3 mix: standard CFG at the sampler's cfg")
    bands, region = plan.get("bands"), plan.get("region")
    if bands:
        cut = f"blur sigma {bands.get('blur_sigma')}" if bands.get("method") == "gaussian" else f"cutoff {bands.get('fft_cutoff')}"
        lines.append(f"4 where: bands ({bands.get('method')}, {cut}) low x{bands.get('low')}, high x{bands.get('high')}")
    if region:
        lines.append(f"4 where: region mask inside x{region.get('inside')}, outside x{region.get('outside')}, "
                     f"feather {region.get('feather')}" + (", inverted" if region.get("invert") else ""))
    if not bands and not region:
        lines.append("4 where: everywhere, all frequencies")
    corr = plan.get("correct") or []
    if corr:
        for i, spec in enumerate(corr, 1):
            extra = {k: v for k, v in spec.items() if k not in ("method", "strength", "space")}
            lines.append(f"5 correct #{i}: {spec['method']} strength {spec.get('strength')} ({spec.get('space', 'auto')} space) {extra}")
    else:
        lines.append("5 correct: none")
    gov = plan.get("govern")
    if gov:
        units = {v: k for k, v in GOVERN_UNITS.items()}
        home = "conditional" if gov.get("home", "cond") == "cond" else "unconditional"
        window = "whole run" if gov.get("start", 0) <= 0 and gov.get("end", 1) >= 1 else \
            f"{gov.get('start', 0):.2f}-{gov.get('end', 1):.2f} of the run"
        lines.append(f"6 govern: angle to the {home} held in [{gov.get('min_deg', 0.0):g}, {gov.get('max_deg', 180.0):g}] "
                     f"degrees, unit {units.get(gov.get('unit', 'image'), gov.get('unit'))}, "
                     f"{gov.get('space', 'auto')} space, {window}")
    else:
        lines.append("6 govern: off")
    meas = plan.get("measure")
    lines.append(f"7 measure: probe '{meas.get('prefix')}' -> output/cfg_probe/" if meas else "7 measure: off")
    return "\n".join(lines)
