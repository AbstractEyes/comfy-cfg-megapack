"""
cfg_variants.py - classifier-free guidance (CFG) and its variants in one shared notation.

Pure PyTorch. Import it as a module or paste it into a notebook cell: the file only defines
functions, classes and two dicts (REGISTRY, VARIANT_NOTES); it has no side effects at import.

Shared notation
---------------
x_t      the noisy latent. Gaussian path x_t = alpha * x0 + sigma * n, n ~ N(0, I):
           VP (DDPM / DDIM)            alpha = sqrt(alphabar), sigma = sqrt(1 - alphabar)
           VE / EDM (and ComfyUI's
           internal space for eps and
           v models)                   alpha = 1,              sigma = sigma
           rectified flow (RF)         alpha = 1 - t,          sigma = t    (t = 1 is pure noise)
c, u, n  condition, null (unconditional) condition, negative condition.
D(x_t; c)  the model's native prediction: eps (noise), v (v = alpha * n - sigma * x0),
           x0 (denoised) or f (flow velocity, f = n - x0).
d_c = D(x_t; c), d_u = D(x_t; u), d_n = D(x_t; n); d_weak = any second prediction standing in for d_u
           (a perturbed pass, a weaker model, a negative prompt).
delta = d_c - d_u, the guidance direction.
Baseline CFG (Ho & Salimans convention): d_hat = d_u + w * delta, so w = 1 is the conditional
prediction and w = 0 the unconditional one. Papers that write d_c + s * delta use s = w - 1;
every docstring below says when a source uses the s form and converts it.

Spaces
------
Linear CFG gives the same guided x0 whichever of eps / v / f / x0 it is applied to (at fixed x_t and
sigma the conversions are affine with coefficients that sum to one). Every nonlinear variant loses that
invariance, so each docstring states the space its source used:
  [x0]     d_c / d_u must be x0 (denoised) estimates;
  [noise]  eps for eps models, the flow velocity f for flow models (the paper's native output);
  [v]      v-prediction;
  [any]    linear or per-sample scale invariant: any space gives the same x0.
Convert with to_x0 / from_x0 (or the explicit helpers) before calling when the space matters.

Conventions for every public function
-------------------------------------
* Tensors are batch-first, (B, ...). "Per sample" reductions run over all non-batch dims.
* Computation runs in float32; the result is returned in the dtype of the first tensor argument.
* w (and most scalar knobs) may be a float or a per-sample tensor of shape (B,).
* "Extra cost" counts network evaluations (NFE) per step beyond the two passes of plain CFG.

Registry
--------
REGISTRY[name] = {"fn", "family", "space", "defaults", "needs", "node", "state", "state_knobs",
                  "caller", "kind", "cite", "batch_coupled"}
apply_variant(name, d_c, d_u, w, ctx, **knobs) calls a registered combiner, filling any keyword it
accepts (x_t, sigma, alpha, t, progress, step, state, d_n, ...) from the ctx dict.
VARIANT_NOTES[name] = {"reason", "cite", "use"} lists catalogued methods that cannot be written as a
pure combiner of supplied predictions (they need attention hooks, training, a second model, autograd
through the network, or an external evaluator).
"""
import functools
import inspect
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

Tensor = torch.Tensor

# =============================================================================
# 0. Utilities (dtype handling, per-sample reductions, small signal helpers)
# =============================================================================

_LOW_PRECISION = (torch.float16, torch.bfloat16)
_EPS = 1e-12


def _up(a):
    """Upcast fp16/bf16 tensors (or a list/tuple of tensors) to float32."""
    if torch.is_tensor(a):
        return a.float() if a.dtype in _LOW_PRECISION else a
    if isinstance(a, (list, tuple)) and len(a) > 0 and all(torch.is_tensor(t) for t in a):
        return type(a)(_up(t) for t in a)
    return a


def _down(out, dtype):
    if torch.is_tensor(out):
        return out.to(dtype) if (out.is_floating_point() and out.dtype != dtype) else out
    if isinstance(out, tuple):
        return tuple(_down(o, dtype) for o in out)
    if isinstance(out, list):
        return [_down(o, dtype) for o in out]
    return out


def _first_float(args, kwargs):
    for a in list(args) + list(kwargs.values()):
        if torch.is_tensor(a) and a.is_floating_point():
            return a
        if isinstance(a, (list, tuple)):
            for t in a:
                if torch.is_tensor(t) and t.is_floating_point():
                    return t
    return None


def _keep_dtype(fn):
    """Compute in float32, return in the dtype of the first floating tensor argument."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        ref = _first_float(args, kwargs)
        if ref is None or ref.dtype not in _LOW_PRECISION:
            return fn(*args, **kwargs)
        out = fn(*[_up(a) for a in args], **{k: _up(v) for k, v in kwargs.items()})
        return _down(out, ref.dtype)

    return wrapper


def _bview(v: Tensor, ref: Tensor) -> Tensor:
    """Reshape a per-sample vector (B,) so it broadcasts against ref (B, ...)."""
    return v.reshape(v.shape[0], *([1] * (ref.ndim - 1)))


def _scalar(w, ref: Tensor):
    """Float, 0-d tensor or per-sample (B,) tensor -> something that broadcasts against ref."""
    if torch.is_tensor(w):
        w = w.to(device=ref.device, dtype=ref.dtype if ref.is_floating_point() else torch.float32)
        if w.ndim == 1 and ref.ndim > 1 and w.numel() == ref.shape[0]:
            return _bview(w, ref)
    return w


def _tensor(w, ref: Tensor) -> Tensor:
    w = _scalar(w, ref)
    if not torch.is_tensor(w):
        w = torch.tensor(float(w), device=ref.device, dtype=ref.dtype)
    return w


def _float(w) -> float:
    """A python float from a float or a tensor whose elements are all equal (checked on the first)."""
    if torch.is_tensor(w):
        return float(w.flatten()[0])
    return float(w)


def _dot(a: Tensor, b: Tensor) -> Tensor:
    return _bview((a * b).flatten(1).sum(1), a)


def _sqnorm(a: Tensor) -> Tensor:
    return _dot(a, a)


def _norm(a: Tensor) -> Tensor:
    return _bview(a.flatten(1).norm(dim=1), a)


def _std(a: Tensor) -> Tensor:
    return _bview(a.flatten(1).std(dim=1), a)


def _cos(a: Tensor, b: Tensor) -> Tensor:
    return _dot(a, b) / (_norm(a) * _norm(b)).clamp_min(_EPS)


def _project(v: Tensor, onto: Tensor) -> Tuple[Tensor, Tensor]:
    """(parallel, orthogonal) parts of v relative to onto, per sample over all non-batch dims."""
    coef = _dot(v, onto) / _sqnorm(onto).clamp_min(_EPS)
    par = coef * onto
    return par, v - par


def _quantile_rows(a2d: Tensor, q: float) -> Tensor:
    """Linear-interpolated q-quantile of every row of a 2-D tensor (torch.quantile's default rule,
    without its input-size limit)."""
    n = a2d.shape[1]
    if n == 1:
        return a2d[:, 0]
    s, _ = a2d.sort(dim=1)
    pos = min(max(float(q), 0.0), 1.0) * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return s[:, lo] + frac * (s[:, hi] - s[:, lo])


def _gaussian_kernel1d(sigma: float, size: int, device, dtype) -> Tensor:
    x = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def _pad_mode(pad: int, h: int, w: int) -> str:
    return "reflect" if (pad < h and pad < w) else "replicate"


def _sep_conv2d(x: Tensor, k: Tensor) -> Tensor:
    """Separable convolution with a 1-D kernel over the last two dims, reflect padding."""
    shape = x.shape
    h, w = shape[-2], shape[-1]
    y = x.reshape(-1, 1, h, w)
    pad = k.numel() // 2
    mode = _pad_mode(pad, h, w)
    y = F.pad(y, (pad, pad, 0, 0), mode=mode)
    y = F.conv2d(y, k.view(1, 1, 1, -1))
    y = F.pad(y, (0, 0, pad, pad), mode=mode)
    y = F.conv2d(y, k.view(1, 1, -1, 1))
    return y.reshape(shape)


@_keep_dtype
def gaussian_blur2d(x: Tensor, sigma: float = 1.0, kernel_size: Optional[int] = None) -> Tensor:
    """Separable Gaussian blur over the last two dims (reflect padding). x: (..., H, W)."""
    if sigma is None or sigma <= 0:
        return x
    if kernel_size is None:
        kernel_size = 2 * int(math.ceil(3 * sigma)) + 1
    return _sep_conv2d(x, _gaussian_kernel1d(float(sigma), int(kernel_size), x.device, x.dtype))


def _resize(x: Tensor, size, mode: str = "bilinear") -> Tensor:
    """F.interpolate over the last two dims for any number of leading dims."""
    shape = x.shape
    y = x.reshape(-1, 1, shape[-2], shape[-1])
    if mode in ("bilinear", "bicubic"):
        y = F.interpolate(y, size=tuple(size), mode=mode, align_corners=False)
    else:
        y = F.interpolate(y, size=tuple(size), mode=mode)
    return y.reshape(*shape[:-2], y.shape[-2], y.shape[-1])


def _dct_matrix(n: int, device=None, dtype=torch.float32) -> Tensor:
    """Orthonormal DCT-II matrix M (n x n): X = M @ x, x = M.T @ X."""
    k = torch.arange(n, device=device, dtype=torch.float64).unsqueeze(1)
    i = torch.arange(n, device=device, dtype=torch.float64).unsqueeze(0)
    m = torch.cos(math.pi * (2 * i + 1) * k / (2 * n)) * math.sqrt(2.0 / n)
    m[0] = m[0] / math.sqrt(2.0)
    return m.to(dtype)


def dct2(x: Tensor) -> Tensor:
    """Orthonormal 2-D DCT-II over the last two dims."""
    mh = _dct_matrix(x.shape[-2], x.device, x.dtype)
    mw = _dct_matrix(x.shape[-1], x.device, x.dtype)
    return mh @ x @ mw.T


def idct2(x: Tensor) -> Tensor:
    """Inverse of dct2 (orthonormal DCT-III) over the last two dims."""
    mh = _dct_matrix(x.shape[-2], x.device, x.dtype)
    mw = _dct_matrix(x.shape[-1], x.device, x.dtype)
    return mh.T @ x @ mw


# =============================================================================
# 1. Parameterizations: eps / v / flow velocity / score <-> x0
# =============================================================================
# All conversions take the path coefficients (alpha, sigma) of x_t = alpha * x0 + sigma * n.
# Use vp_alpha_sigma / ve_alpha_sigma / flow_alpha_sigma to get them.


def vp_alpha_sigma(alphabar):
    """VP (DDPM/DDIM) path: alpha = sqrt(alphabar), sigma = sqrt(1 - alphabar)."""
    if torch.is_tensor(alphabar):
        return alphabar.sqrt(), (1 - alphabar).sqrt()
    return math.sqrt(alphabar), math.sqrt(1.0 - alphabar)


def ve_alpha_sigma(sigma):
    """VE / EDM path (also ComfyUI's internal x for eps and v models): alpha = 1."""
    return (torch.ones_like(sigma) if torch.is_tensor(sigma) else 1.0), sigma


def flow_alpha_sigma(t):
    """Rectified flow: x_t = (1 - t) x0 + t n (t = 1 is noise)."""
    return 1 - t, t


def ve_sigma_from_alphabar(alphabar):
    """VE sigma with the same signal-to-noise ratio as a VP level: sqrt((1 - alphabar) / alphabar).
    The VE latent is x_ve = x_vp / sqrt(alphabar)."""
    if torch.is_tensor(alphabar):
        return ((1 - alphabar) / alphabar).sqrt()
    return math.sqrt((1.0 - alphabar) / alphabar)


def vp_from_ve(sigma):
    """VP coefficients for a VE level: alpha_vp = 1/sqrt(1+sigma^2), sigma_vp = sigma/sqrt(1+sigma^2);
    x_vp = alpha_vp * x_ve. This is how ComfyUI feeds eps / v models."""
    if torch.is_tensor(sigma):
        d = (1 + sigma ** 2).sqrt()
    else:
        d = math.sqrt(1.0 + sigma ** 2)
    return 1 / d, sigma / d


def snr_matched_flow_time(alpha, sigma):
    """Rectified-flow time with the same signal-to-noise ratio: t = sigma / (alpha + sigma)."""
    return sigma / (alpha + sigma)


@_keep_dtype
def eps_to_x0(eps, x_t, alpha, sigma):
    """x0 = (x_t - sigma eps) / alpha."""
    a, s = _scalar(alpha, eps), _scalar(sigma, eps)
    return (x_t - s * eps) / a


@_keep_dtype
def x0_to_eps(x0, x_t, alpha, sigma):
    """eps = (x_t - alpha x0) / sigma."""
    a, s = _scalar(alpha, x0), _scalar(sigma, x0)
    return (x_t - a * x0) / s


@_keep_dtype
def v_to_x0(v, x_t, alpha, sigma):
    """v = alpha * n - sigma * x0 (Salimans & Ho 2022). x0 = (alpha x_t - sigma v) / (alpha^2 + sigma^2)."""
    a, s = _scalar(alpha, v), _scalar(sigma, v)
    return (a * x_t - s * v) / (a * a + s * s)


@_keep_dtype
def x0_to_v(x0, x_t, alpha, sigma):
    """v = alpha n - sigma x0 with n = (x_t - alpha x0) / sigma."""
    a, s = _scalar(alpha, x0), _scalar(sigma, x0)
    n = (x_t - a * x0) / s
    return a * n - s * x0


@_keep_dtype
def flow_to_x0(f, x_t, alpha, sigma):
    """f = n - x0. For any affine path x0 = (x_t - sigma f) / (alpha + sigma); RF: x0 = x_t - t f."""
    a, s = _scalar(alpha, f), _scalar(sigma, f)
    return (x_t - s * f) / (a + s)


@_keep_dtype
def x0_to_flow(x0, x_t, alpha, sigma):
    """f = n - x0 with n = (x_t - alpha x0) / sigma (RF: f = (x_t - x0) / t)."""
    a, s = _scalar(alpha, x0), _scalar(sigma, x0)
    n = (x_t - a * x0) / s
    return n - x0


@_keep_dtype
def x0_to_score(x0, x_t, alpha, sigma):
    """Score of the noisy marginal: grad log p_t(x_t) = (alpha x0 - x_t) / sigma^2 = -eps / sigma."""
    a, s = _scalar(alpha, x0), _scalar(sigma, x0)
    return (a * x0 - x_t) / (s * s)


@_keep_dtype
def score_to_x0(score, x_t, alpha, sigma):
    """Tweedie: x0 = (x_t + sigma^2 score) / alpha."""
    a, s = _scalar(alpha, score), _scalar(sigma, score)
    return (x_t + s * s * score) / a


_TO_X0 = {"x0": None, "eps": eps_to_x0, "v": v_to_x0, "flow": flow_to_x0, "score": score_to_x0}
_FROM_X0 = {"x0": None, "eps": x0_to_eps, "v": x0_to_v, "flow": x0_to_flow, "score": x0_to_score}


def to_x0(pred, kind: str, x_t, alpha, sigma):
    """Convert a native prediction of kind 'x0' | 'eps' | 'v' | 'flow' | 'score' to an x0 estimate."""
    if kind not in _TO_X0:
        raise ValueError(f"unknown prediction kind {kind!r}; use one of {sorted(_TO_X0)}")
    fn = _TO_X0[kind]
    return pred if fn is None else fn(pred, x_t, alpha, sigma)


def from_x0(x0, kind: str, x_t, alpha, sigma):
    """Convert an x0 estimate to a prediction of kind 'x0' | 'eps' | 'v' | 'flow' | 'score'."""
    if kind not in _FROM_X0:
        raise ValueError(f"unknown prediction kind {kind!r}; use one of {sorted(_FROM_X0)}")
    fn = _FROM_X0[kind]
    return x0 if fn is None else fn(x0, x_t, alpha, sigma)


def w_from_s(s):
    """Ho-Salimans weight from the 'd_c + s * delta' form (original CFG paper, CAD, Muse, LCM): w = s + 1."""
    return s + 1


def s_from_w(w):
    """The 'd_c + s * delta' weight from the Ho-Salimans weight: s = w - 1."""
    return w - 1


# =============================================================================
# 2. Plain CFG and output-space combiners (family A)
# =============================================================================


@_keep_dtype
def cfg(d_c, d_u, w=7.5):
    """Classifier-free guidance [any]. Ho & Salimans (2021/2022), arXiv 2207.12598.

    d_hat = d_u + w * (d_c - d_u) = d_c + (w - 1) * delta.
    The paper writes eps_tilde = (1 + s) eps_c - s eps_u, i.e. s = w - 1. diffusers guidance_scale is w.
    Defaults: w = 7.5 (SD1.5 / SDXL, useful 4-9); SD3.5 3.5-7; Qwen-Image 4; Lumina-2 4.
    Extra cost: none (the null pass is the baseline second pass).
    """
    w = _scalar(w, d_c)
    return d_u + w * (d_c - d_u)


@_keep_dtype
def weak_branch_cfg(d_c, d_weak, w=2.0):
    """CFG against an arbitrary weak branch [any]: d_hat = d_weak + w * (d_c - d_weak).

    Covers autoguidance (Karras et al. 2024, arXiv 2406.02507; d_weak = a smaller / less-trained model,
    w about 2), negative prompts (d_weak = D(x_t; n)), ICG (Sadat et al. 2025, arXiv 2407.02687;
    d_weak = D(x_t; random condition)), SIMS, sparse guidance, ERG, internal guidance, dropout
    self-guidance, Domain Guidance / Unconditional Priors (d_weak = base model's null), NPO,
    personalization guidance and VPG. See REGISTRY[...]['caller'] for what each alias needs.
    Extra cost: whatever produces d_weak (usually 1 NFE, replacing the null pass).
    """
    w = _scalar(w, d_c)
    return d_weak + w * (d_c - d_weak)


@_keep_dtype
def rescale_cfg(d_c, d_u, w=7.5, phi=0.7):
    """Guidance rescale [v]. Lin, Liu, Li & Yang (2024), WACV, arXiv 2305.08891.

    d_cfg = d_u + w * delta;  d_resc = d_cfg * std(d_c) / std(d_cfg) (per sample, all non-batch dims);
    d_hat = phi * d_resc + (1 - phi) * d_cfg.
    The paper rescales its v-prediction output; diffusers rescales whatever the model outputs; ComfyUI's
    RescaleCFG converts eps/v to v, and uses x0 for flow models.
    Defaults: phi = 0.7 (paper range 0.5-0.75, with w = 7.5). phi = 0 is plain CFG.
    Extra cost: none.
    """
    w = _scalar(w, d_c)
    d_cfg = d_u + w * (d_c - d_u)
    ratio = _std(d_c) / _std(d_cfg).clamp_min(1e-8)
    return phi * d_cfg * ratio + (1 - phi) * d_cfg


@_keep_dtype
def dynamic_threshold(x0, p=0.995, s_max=1.0):
    """Imagen dynamic thresholding of an x0 estimate [x0, bounded pixel range]. Saharia et al. (2022),
    NeurIPS, arXiv 2205.11487.

    s = quantile_p(|x0|) per sample; s = clamp(s, 1, s_max); x0_thr = clip(x0, -s, s) / s.
    s_max = 1 gives static clipping to [-1, 1] (the diffusers default); raise s_max (1.5-5) for the
    dynamic behaviour. Meant for pixel-space models: latents are not bounded to [-1, 1].
    """
    s = _quantile_rows(x0.abs().flatten(1), p).clamp(min=1.0, max=float(s_max))
    s = _bview(s, x0)
    return torch.maximum(torch.minimum(x0, s), -s) / s


@_keep_dtype
def dynamic_threshold_cfg(d_c, d_u, w=7.5, p=0.995, s_max=1.0):
    """Plain CFG in x0 followed by Imagen dynamic thresholding [x0]. arXiv 2205.11487.
    Defaults p = 0.995, s_max = 1.0 (static clip). Extra cost: one quantile per sample."""
    return dynamic_threshold(cfg(d_c, d_u, w), p=p, s_max=s_max)


@_keep_dtype
def mimic_cfg(d_c, d_u, w=15.0, mimic_scale=7.0, threshold_percentile=1.0,
              separate_feature_channels=True, scaling_startpoint="MEAN", variability_measure="AD",
              interpolate_phi=1.0):
    """Mimic-scale dynamic thresholding ('CFG scale fix') [x0]. mcmonkey4eva (2023),
    sd-dynamic-thresholding (community, no paper). Confidence: high (read from the node source).

    mim = d_u + m * delta, cfg = d_u + w * delta (m = mimic_scale, w = the high real scale).
    Per (sample, channel) with spatial means mu: centred mim_c = mim - mu_mim, cfg_c = cfg - mu_cfg.
    AD: ref_m = max|mim_c|, ref_g = quantile_q(|cfg_c|); STD: ref_m = std(mim_c), ref_g = std(cfg_c).
    With separate_feature_channels off, the references are single scalars over the whole batch tensor.
    MEAN + AD: R = max(ref_m, ref_g); out = clamp(cfg_c, -R, R) / R * ref_m + mu_cfg.
    MEAN + STD: out = cfg_c / ref_g * ref_m + mu_cfg.   ZERO: out = cfg * ref_m / ref_g.
    Final: x0_hat = phi * out + (1 - phi) * cfg.  w == m returns plain CFG.
    Defaults: m = 7, q = 1.0, MEAN, AD, phi = 1; use w about 15-30. Extra cost: per-channel quantiles.
    """
    if not torch.is_tensor(w) and float(w) == float(mimic_scale):
        return cfg(d_c, d_u, w)
    w = _scalar(w, d_c)
    delta = d_c - d_u
    mim = d_u + mimic_scale * delta
    cfg_t = d_u + w * delta
    if cfg_t.ndim < 3:
        raise ValueError("mimic_cfg expects (B, C, ...) tensors")
    mim_f = mim.flatten(2)
    cfg_f = cfg_t.flatten(2)
    mu_m = mim_f.mean(dim=2, keepdim=True)
    mu_g = cfg_f.mean(dim=2, keepdim=True)
    mc = mim_f - mu_m
    gc = cfg_f - mu_g
    B, C, N = gc.shape
    if separate_feature_channels:
        if variability_measure == "STD":
            ref_m = mc.std(dim=2, keepdim=True)
            ref_g = gc.std(dim=2, keepdim=True)
        else:
            ref_m = mc.abs().amax(dim=2, keepdim=True)
            ref_g = _quantile_rows(gc.abs().reshape(B * C, N), threshold_percentile).view(B, C, 1)
    else:
        if variability_measure == "STD":
            ref_m = mc.std()
            ref_g = gc.std()
        else:
            ref_m = mc.abs().max()
            ref_g = _quantile_rows(gc.abs().reshape(1, -1), threshold_percentile)[0]
    ref_g = ref_g.clamp_min(1e-8) if torch.is_tensor(ref_g) else ref_g
    if scaling_startpoint == "ZERO":
        res = cfg_f * (ref_m / ref_g)
    elif variability_measure == "STD":
        res = gc / ref_g * ref_m + mu_g
    else:
        r_max = torch.maximum(ref_m, ref_g).clamp_min(1e-8)
        res = torch.maximum(torch.minimum(gc, r_max), -r_max) / r_max * ref_m + mu_g
    res = res.reshape(cfg_t.shape)
    return interpolate_phi * res + (1 - interpolate_phi) * cfg_t


class APGMomentum:
    """Reverse-momentum buffer for APG. One per generation. update() resets itself when sigma rises
    (a new sampling run), as ComfyUI's APG node does; call reset() explicitly otherwise."""

    def __init__(self):
        self.buffer = None
        self.prev_sigma = None

    def reset(self):
        self.buffer = None
        self.prev_sigma = None

    def update(self, g, beta, sigma=None):
        if sigma is not None:
            s = _float(sigma)
            if self.prev_sigma is not None and s > self.prev_sigma:
                self.reset()
            self.prev_sigma = s
        if beta == 0:
            return g
        if self.buffer is None or self.buffer.shape != g.shape:
            self.buffer = g
        else:
            self.buffer = g + beta * self.buffer
        return self.buffer


@_keep_dtype
def apg(d_c, d_u, w=15.0, eta=0.0, norm_threshold=15.0, momentum=-0.5, state=None, sigma=None,
        formulation="paper"):
    """Adaptive Projected Guidance [x0]. Sadat, Hilliges & Weber (2025), ICLR, arXiv 2410.02416.

    g = d_c - d_u (x0 estimates). (1) reverse momentum: g <- g + beta * g_prev (needs `state`,
    an APGMomentum; beta = momentum < 0). (2) norm cap: g <- g * min(1, r / ||g||). (3) projection on
    d_c: g_par = (<g, d_c> / ||d_c||^2) d_c, g_perp = g - g_par. (4) combination:
      'paper'     d_hat = d_c + (w - 1) * (g_perp + eta * g_par)
      'comfyui'   d_hat = d_c + w * (g_perp + eta * g_par)       (one unit more guidance)
      'diffusers' d_hat = d_u + w * (g_perp + eta * g_par)       (diffusers applies it to raw outputs)
    eta = 1, r = 0 (off) and momentum = 0 give plain CFG ('paper' and 'diffusers').
    Defaults: the paper's SDXL row (w 15, eta 0, r 15, beta -0.5). Other rows (w, eta, r, beta):
    SD2.1 10, 0, 7.5, -0.75; DiT-XL/2 4, 0, 5, -0.5; EDM2-S 4, 0, 2.5, -0.75. Choose r near the
    typical ||g||. ComfyUI defaults: eta 1, r 5, momentum 0. Extra cost: none (one latent buffer).
    """
    w = _scalar(w, d_c)
    g = d_c - d_u
    if state is not None and momentum != 0:
        g = state.update(g, momentum, sigma)
    if norm_threshold is not None and norm_threshold > 0:
        g = g * torch.clamp(norm_threshold / _norm(g).clamp_min(_EPS), max=1.0)
    g_par, g_perp = _project(g, d_c)
    mod = g_perp + eta * g_par
    if formulation == "paper":
        return d_c + (w - 1) * mod
    if formulation == "comfyui":
        return d_c + w * mod
    if formulation == "diffusers":
        return d_u + w * mod
    raise ValueError("formulation must be 'paper', 'comfyui' or 'diffusers'")


@_keep_dtype
def cfg_zero_star(d_c, d_u, w=4.0, step=None, zero_init_steps=1, x_t=None, space="noise"):
    """CFG-Zero* (optimized scale + zero-init) [noise: flow velocity]. Fan, Zheng, Yeh & Liu (2025),
    arXiv 2503.18886.

    s* = <d_c, d_u> / (||d_u||^2 + 1e-8) per sample; d_hat = s* d_u + w * (d_c - s* d_u).
    Zero-init: for solver steps step < zero_init_steps the output leaves the state unmoved:
    zeros in noise space (velocity 0 / eps 0 in VE), x_t in x0 space (pass space='x0' and x_t); in any
    other space zero-init is skipped. Pass step=None to disable zero-init. w = 1 returns d_c.
    Defaults: zero_init_steps = 1 (paper; 0-2 useful). Extra cost: none.
    """
    w = _scalar(w, d_c)
    s_star = _dot(d_c, d_u) / (_sqnorm(d_u) + 1e-8)
    out = s_star * d_u + w * (d_c - s_star * d_u)
    if step is not None and step < zero_init_steps:
        if space == "x0":
            if x_t is None:
                raise ValueError("zero-init in x0 space needs x_t")
            return x_t.to(out.dtype).clone()
        if space == "noise":
            return torch.zeros_like(out)
    return out


@_keep_dtype
def cfg_zero_star_comfy(d_c, d_u, w=4.0, x_t=None):
    """ComfyUI's CFGZeroStar node form [x0]. s* is computed from (x_t - d_c, x_t - d_u) (proportional
    to sigma*eps or t*f) but applied to the x0 estimates: x0_hat = s* x0_u + w (x0_c - s* x0_u).
    This differs from the velocity form by (s* - 1)(1 - w) x_t; it has no zero-init. arXiv 2503.18886."""
    if x_t is None:
        raise ValueError("cfg_zero_star_comfy needs x_t")
    w = _scalar(w, d_c)
    rc, ru = x_t - d_c, x_t - d_u
    s_star = _dot(rc, ru) / (_sqnorm(ru) + 1e-8)
    return s_star * d_u + w * (d_c - s_star * d_u)


@_keep_dtype
def tcfg(d_c, d_u, w=7.5):
    """Tangential Damping CFG [noise: eps / score]. Kwon, Kim, Jeong, Hsiao & Uh (2025), CVPR,
    arXiv 2503.18137.

    Per sample: A = [d_u; d_c] (2 x N). v1 = first right singular vector of A (computed exactly from
    the 2 x 2 Gram matrix). d_u' = <d_u, v1> v1 (drop d_u's tangential part). d_hat = d_u' + w (d_c - d_u').
    No extra knob. Paper scales: SD1.5 7.5, SDXL 5, SD3 7, PixArt-alpha 4.5. w = 1 returns d_c.
    Extra cost: a 2 x N projection per sample (negligible).
    """
    w = _scalar(w, d_c)
    B = d_c.shape[0]
    A = torch.stack([d_u.reshape(B, -1), d_c.reshape(B, -1)], dim=1)
    G = (A @ A.transpose(1, 2)).double()
    evals, evecs = torch.linalg.eigh(G)
    u1 = evecs[..., :, -1].to(A.dtype)
    s1 = evals[..., -1].clamp_min(0).sqrt().to(A.dtype)
    v1 = torch.einsum("bk,bkn->bn", u1, A) / s1.clamp_min(_EPS).unsqueeze(1)
    du_td = ((A[:, 0] * v1).sum(1, keepdim=True) * v1).reshape(d_u.shape)
    return du_td + w * (d_c - du_td)


def beta_pdf(p, a=2.0, b=2.0, peak_normalize=False):
    """Beta(a, b) density at p in [0, 1] (scipy.stats.beta.pdf); peak_normalize divides by its value at
    the mode (a - 1)/(a + b - 2)."""
    p = min(max(float(p), 0.0), 1.0)
    try:
        val = p ** (a - 1) * (1 - p) ** (b - 1)
    except ZeroDivisionError:
        return float("inf")
    val = val / math.exp(math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b))
    if peak_normalize and a > 1 and b > 1:
        mode = (a - 1) / (a + b - 2)
        val = val / beta_pdf(mode, a, b, False)
    return val


@_keep_dtype
def beta_cfg(d_c, d_u, w=7.5, progress=0.5, a=2.0, b=2.0, gamma=1.0, peak_normalize=False):
    """beta-CFG: Beta-shaped schedule with gradient-norm normalization [noise: eps].
    Malarz, Kasymov, Zieba, Tabor & Spurek (2025), ECAI, arXiv 2502.10574.

    p = i / (N - 1) (progress; 0 at the first, noisiest step). b(p) = Beta(a, b) pdf.
    d_hat = d_u + w * b(p) * delta / ||delta||^gamma (per-sample L2 norm over all elements).
    gamma = 0 with a = b = 1 is plain CFG. The norm division makes the effective scale depend on the
    tensor size and parameterization; re-tune w when moving between models.
    Defaults: w 7.5, (a, b) = (2, 2) (paper Table 1; (3, 2) better for w >= 5), gamma 1 (README).
    Extra cost: none.
    """
    w = _scalar(w, d_c)
    delta = d_c - d_u
    bp = beta_pdf(progress, a, b, peak_normalize)
    if gamma == 0:
        return d_u + w * bp * delta
    return d_u + w * bp * delta / _norm(delta).clamp_min(_EPS) ** gamma


@_keep_dtype
def adg(d_c, d_u, w=6.0, max_angle=math.pi / 3):
    """Angle Domain Guidance [x0]. Jin, Xiao, Liu & Gu (2025), ICML, arXiv 2506.11039.

    Whole latent as one vector per sample. theta = arccos(<x0_u, x0_c> / (||x0_u|| ||x0_c||));
    theta_w = min((w - 1) theta, max_angle); e = x0_c - (<x0_u, x0_c>/||x0_u||^2) x0_u;
    x0_hat = cos(theta_w) x0_c + sin(theta_w) / sin(theta) * e, with (w - 1) e when sin(theta) <= 1e-3
    (official code). Guarantee ||x0_hat|| <= sqrt(2) ||x0_c||. w = 1 returns x0_c.
    Defaults: w 6 (2-15 tested), max_angle pi/3. Extra cost: none.
    """
    w = _scalar(w, d_c)
    cos_t = _cos(d_u, d_c).clamp(-1.0, 1.0)
    theta = torch.arccos(cos_t)
    theta_w = torch.clamp((w - 1) * theta, max=float(max_angle))
    e = d_c - _dot(d_u, d_c) / _sqnorm(d_u).clamp_min(_EPS) * d_u
    sin_t = torch.sin(theta)
    big = sin_t > 1e-3
    coef = torch.where(big, torch.sin(theta_w) / torch.where(big, sin_t, torch.ones_like(sin_t)),
                       (w - 1) * torch.ones_like(sin_t))
    return torch.cos(theta_w) * d_c + coef * e


@_keep_dtype
def power_law_cfg(d_c, d_u, w=7.5, alpha=0.9, omega=None, score_sigma=None):
    """Power-law (nonlinear) CFG [noise; the paper uses scores]. Lehman Pavasovic, Verbeek, Biroli &
    Mezard (2025), arXiv 2502.07849.

    d_hat = d_c + omega * ||delta||^alpha * delta, i.e. per-sample w_eff = 1 + omega ||delta||^alpha.
    omega defaults to w - 1 so alpha = 0 is plain CFG at w. Pass score_sigma (sigma_t) to measure the
    norm in score units (||delta_S|| = ||delta_eps|| / sigma_t) as the paper does for eps models.
    Defaults: alpha 0.9 (paper). omega must be re-tuned per model and resolution: ||delta|| scales with
    sqrt(number of elements). Extra cost: none.
    """
    if omega is None:
        omega = _scalar(w, d_c) - 1
    delta = d_c - d_u
    n = _norm(delta)
    if score_sigma is not None:
        n = n / _scalar(score_sigma, d_c)
    return d_c + omega * n ** alpha * delta


def _energy(a: Tensor, robust: bool, lo: float, hi: float) -> Tensor:
    sq = a.flatten(1) ** 2
    if not robust:
        return sq.sum(1)
    p_lo = _quantile_rows(sq, lo / 100.0).unsqueeze(1)
    p_hi = _quantile_rows(sq, hi / 100.0).unsqueeze(1)
    m = (sq >= p_lo) & (sq <= p_hi)
    return (sq * m).sum(1)


@_keep_dtype
def ep_cfg(d_c, d_u, w=7.5, robust=True, lo=45.0, hi=55.0):
    """Energy-Preserving CFG [noise]. Zhang, Luan, Bi & Zhang (2024), arXiv 2412.09966 (medium).

    d_cfg = d_c + (w - 1) delta; d_hat = d_cfg * sqrt(E(d_c) / E(d_cfg)), E = sum of squares per sample.
    Robust variant: E counts only squared entries between the lo-th and hi-th percentiles of that
    tensor's squared entries. No blend with d_cfg. w = 1 returns d_c.
    Defaults: robust, lo 45, hi 55 (used for all paper results). Extra cost: two percentiles.
    """
    w = _scalar(w, d_c)
    d_cfg = d_c + (w - 1) * (d_c - d_u)
    e_c = _energy(d_c, robust, lo, hi)
    e_g = _energy(d_cfg, robust, lo, hi).clamp_min(_EPS)
    return d_cfg * _bview((e_c / e_g).sqrt(), d_c)


@_keep_dtype
def cfg_renorm(d_c, d_u, w=4.0, rho=1.0):
    """CFG-Renorm (Lumina-Image 2.0 / Z-Image) [noise: velocity; ComfyUI RenormCFG applies it to x0].
    Qin et al. (2025), arXiv 2503.21758 (adopting STIV, arXiv 2412.07730).

    d_cfg = d_u + w delta; if ||d_cfg|| >= rho ||d_c|| then d_hat = d_cfg * rho ||d_c|| / ||d_cfg||.
    rho <= 0 disables it. Truncation (CFG only early) is a schedule: use cutoff_schedule / with_interval.
    Defaults: rho 1.0 (1.0-1.5), w 4 (Lumina-2). Extra cost: none.
    """
    w = _scalar(w, d_c)
    d_cfg = d_u + w * (d_c - d_u)
    if rho is None or rho <= 0:
        return d_cfg
    cap = rho * _norm(d_c)
    n = _norm(d_cfg)
    return torch.where(n >= cap, d_cfg * cap / n.clamp_min(_EPS), d_cfg)


@_keep_dtype
def cfg_norm_per_token(d_c, d_u, w=4.0, dim=1, strength=1.0, mode="match"):
    """Per-token / per-pixel norm-matched CFG [noise]. Qwen-Image pipeline, HiDream-E1.1, ComfyUI CFGNorm
    (no paper).

    Norms are taken along `dim` only (dim = 1: channels of a (B, C, H, W) latent; dim = -1 for packed
    (B, L, D) tokens). d_cfg = d_u + w delta.
      'match'     d_hat = strength * d_cfg * ||d_c|| / ||d_cfg|| + (1 - strength) * d_cfg  (can amplify;
                  Qwen-Image at strength 1, ComfyUI CFGNorm pre_cfg=True)
      'attenuate' d_hat = d_cfg * clamp(||d_c|| / (||d_cfg|| + 1e-8), 0, 1) * strength  (ComfyUI CFGNorm
                  default, applied to x0; strength != 1 also scales the image)
    Defaults: strength 1, mode 'match'. Extra cost: none.
    """
    w = _scalar(w, d_c)
    d_cfg = d_u + w * (d_c - d_u)
    nc = torch.linalg.vector_norm(d_c, dim=dim, keepdim=True)
    ng = torch.linalg.vector_norm(d_cfg, dim=dim, keepdim=True)
    if mode == "match":
        ratio = torch.where(ng > 0, nc / ng.clamp_min(_EPS), torch.ones_like(ng))
        return strength * d_cfg * ratio + (1 - strength) * d_cfg
    if mode == "attenuate":
        return d_cfg * (nc / (ng + 1e-8)).clamp(0.0, 1.0) * strength
    raise ValueError("mode must be 'match' or 'attenuate'")


@_keep_dtype
def recfg(d_c, d_u, w=7.5, ratio=None):
    """ReCFG, rectified guidance coefficients [noise: eps]. Xia, Xue, Shen, Yi, Gong & Liu (2025), CVPR,
    arXiv 2410.18737.

    d_hat = w * d_c + gamma0 (.) d_u with gamma0 = (1 - w) * E[d_c] / E[d_u] (elementwise, per timestep).
    `ratio` is the precomputed table entry E[d_c] / E[d_u] for this timestep (broadcastable); build it
    offline with ReCFGRatioTable (paper: 500 samples per condition). ratio = 1 is plain CFG.
    Extra cost: none at sampling; the offline table pass (about 3 A100-hours in the paper).
    """
    w = _scalar(w, d_c)
    if ratio is None:
        ratio = 1.0
    return w * d_c + (1 - w) * ratio * d_u


class ReCFGRatioTable:
    """Accumulates E[d_c] and E[d_u] per timestep key to build the ReCFG ratio table."""

    def __init__(self, reduce_dims: Sequence[int] = (0,), min_abs: float = 1e-4):
        self.sum_c: Dict[Any, Tensor] = {}
        self.sum_u: Dict[Any, Tensor] = {}
        self.count: Dict[Any, int] = {}
        self.reduce_dims = tuple(reduce_dims)
        self.min_abs = min_abs

    def add(self, key, d_c: Tensor, d_u: Tensor):
        c = d_c.float().sum(dim=self.reduce_dims)
        u = d_u.float().sum(dim=self.reduce_dims)
        n = int(math.prod(d_c.shape[i] for i in self.reduce_dims))
        self.sum_c[key] = self.sum_c.get(key, 0) + c
        self.sum_u[key] = self.sum_u.get(key, 0) + u
        self.count[key] = self.count.get(key, 0) + n

    def ratio(self, key) -> Tensor:
        c, u = self.sum_c[key], self.sum_u[key]
        ok = u.abs() > self.min_abs * self.count[key]
        return torch.where(ok, c / torch.where(ok, u, torch.ones_like(u)), torch.ones_like(u))


@_keep_dtype
def mambo_g(d_c, d_u, w=10.0, alpha=8.0):
    """MAMBO-G, magnitude-aware guidance damping [noise: velocity]. Zhu et al. (2025), arXiv 2508.03442.

    r = ||delta|| / ||d_u|| per sample; w_eff = 1 + (w - 1) exp(-alpha r); d_hat = d_u + w_eff delta.
    (w plays the paper's w_max.) alpha = 0 is plain CFG. Defaults: w_max 10, alpha 8. Extra cost: none.
    """
    w = _scalar(w, d_c)
    delta = d_c - d_u
    r = _norm(delta) / _norm(d_u).clamp_min(_EPS)
    return d_u + (1 + (w - 1) * torch.exp(-alpha * r)) * delta


def _skim(A, B, k, s, x_t, flip_filter):
    D = B + k * (A - B)
    m = (torch.sign(A - B) == torch.sign(A)) & (torch.sign(D) == torch.sign(A))
    if flip_filter:
        m = m & (torch.sign(D) == torch.sign(D - x_t))
    return torch.where(m, A - (k - s) * (A - B) / k, A)


@_keep_dtype
def skimmed_cfg(d_c, d_u, w=7.5, x_t=None, skimming_scale=7.0, full_skim_negative=False,
                disable_flipping_filter=False):
    """Skimmed CFG, sign-masked scale fallback [x0]. Extraltodeus (2024-2025), Skimmed_CFG (community).

    skim(A, B, k, s): D = B + k (A - B); mask M = [sign(A - B) == sign(A)] & [sign(D) == sign(A)]
    & [sign(D) == sign(D - x_t)] (the last term is the flipping filter); on M, A <- A - (k - s)(A - B)/k.
    u' = skim(x0_u, x0_c, w, s_neg) with s_neg = 0 if full_skim_negative else s;
    c' = skim(x0_c, u', w - 1, s); x0_hat = u' + w (c' - u'). The masked elements are pulled toward what a
    guidance of scale s would give. x_t is ComfyUI's x (the latent in the sampler's coordinates).
    Defaults: s 7.0. Extra cost: none.
    """
    w = _float(w)
    if x_t is None and not disable_flipping_filter:
        raise ValueError("skimmed_cfg needs x_t unless disable_flipping_filter=True")
    flip = not disable_flipping_filter
    s_neg = 0.0 if full_skim_negative else float(skimming_scale)
    u2 = _skim(d_u, d_c, w, s_neg, x_t, flip) if w != 0 else d_u
    c2 = _skim(d_c, u2, w - 1, float(skimming_scale), x_t, flip) if (w - 1) != 0 else d_c
    return u2 + w * (c2 - u2)


@_keep_dtype
def automatic_cfg(d_c, d_u, w=8.0, reference_scale=8.0, top_k=0.25, mode="hard", w_last=None):
    """Automatic CFG, per-channel range-targeted scale [x0]. Extraltodeus (2024), ComfyUI-AutomaticCFG
    (community, no paper).

    x0_ref = x0_u + w_ref (x0_c - x0_u), w_ref = reference_scale (<= 0: use w). Per (sample, channel):
    range statistic R_k from the top-k and bottom-k values of each row (k = int(H * top_k)):
    'hard' R = (mean(top) + mean|bottom|)/2; 'soft' uses |mean(bottom)|; 'hard_squared' squares R;
    'range' centres the channel first. Channel scale w_k = w_ref * (w_last / 10) / R_k (w_last defaults
    to w), so each channel's guided range is about w/10. x0_hat[k] = x0_u[k] + w_k (x0_c[k] - x0_u[k]).
    mode None returns plain CFG. Extra cost: none.
    """
    if mode in (None, "none", "None"):
        return cfg(d_c, d_u, w)
    w_f = _float(w)
    w_ref = reference_scale if reference_scale and reference_scale > 0 else w_f
    x0_ref = d_u + w_ref * (d_c - d_u)
    if mode == "range":
        x0_ref = x0_ref - x0_ref.mean(dim=(-2, -1), keepdim=True)
    H, W = x0_ref.shape[-2], x0_ref.shape[-1]
    k = min(max(1, int(H * top_k)), W)
    top = x0_ref.topk(k, dim=-1).values
    bot = -((-x0_ref).topk(k, dim=-1).values)
    B, C = x0_ref.shape[:2]
    top_m = top.reshape(B, C, -1).mean(-1)
    if mode == "soft":
        low_m = bot.reshape(B, C, -1).mean(-1).abs()
    else:
        low_m = bot.abs().reshape(B, C, -1).mean(-1)
    R = (top_m + low_m) / 2
    if mode == "hard_squared":
        R = R ** 2
    target = (w_last if w_last is not None else w_f) / 10.0
    s = (w_ref * target / R.clamp_min(1e-6)).reshape(B, C, *([1] * (d_c.ndim - 2)))
    return d_u + s * (d_c - d_u)


@_keep_dtype
def mahiro(d_c, d_u, w=7.5):
    """Mahiro / 'Positive-Biased Guidance' [x0]. yoinked (2024), ComfyUI PR #5975 (no paper).

    x0_cfg = x0_u + w delta; L = w x0_c; U = w x0_u; M = (L + x0_cfg)/2; ssqrt(a) = sign(a) sqrt|a|.
    sim = mean over pixels AND batch of the channel cosine between ssqrt(U) and ssqrt(M) (one scalar:
    batch coupled). q = 2 (sim + 1); x0_hat = (q x0_cfg + (4 - q) L) / 4. sim = 1 gives plain CFG.
    Extra cost: none.
    """
    w = _scalar(w, d_c)
    x_cfg = d_u + w * (d_c - d_u)
    leap = w * d_c
    u_leap = w * d_u
    merge = (leap + x_cfg) / 2
    nu = torch.sqrt(u_leap.abs()) * u_leap.sign()
    nm = torch.sqrt(merge.abs()) * merge.sign()
    sim = F.cosine_similarity(nu, nm, dim=1).mean()
    q = 2 * (sim + 1)
    return (q * x_cfg + (4 - q) * leap) / 4


@_keep_dtype
def reinhard_cfg(d_c, d_u, w=7.5, multiplier=1.0):
    """Reinhard tonemapping of the guidance difference [x0]. comfyanonymous (2024),
    LatentOperationTonemapReinhard + LatentApplyOperationCFG (no paper).

    g = x0_c - x0_u; per pixel m = ||g||_channels + 1e-10; per sample top = mult (mean(m) + 5 std(m));
    m' = top * (m/top) / (m/top + 1); g' = g m'/m; x0_hat = x0_u + w g'. Large multiplier -> plain CFG.
    Defaults: multiplier 1. Extra cost: none.
    """
    w = _scalar(w, d_c)
    g = d_c - d_u
    mag = torch.linalg.vector_norm(g, dim=1, keepdim=True) + 1e-10
    dims = tuple(range(1, mag.ndim))
    top = (mag.std(dim=dims, keepdim=True) * 5 + mag.mean(dim=dims, keepdim=True)) * multiplier
    m = mag / top
    new = m / (m + 1) * top
    return d_u + w * (g / mag * new)


class SMCState:
    """Stores the corrected guidance error of the previous step for SMC-CFG."""

    def __init__(self):
        self.e_prev = None

    def reset(self):
        self.e_prev = None


SMC_AUTO_K = {"flow": 0.2, "eps": 0.01}


@_keep_dtype
def smc_cfg(d_c, d_u, w=7.5, state=None, lam=5.0, k=0.2, switching="sign", flow=False):
    """SMC-CFG, sliding-mode control CFG [noise: velocity]. Wang, Liu, Chi, Liu, Xue & Duan (2026),
    CFG-Ctrl, CVPR, arXiv 2603.03281 (medium).

    e = d_c - d_u; e_prev = stored corrected error (e at the first step);
    s = (e - e_prev) + lam * e_prev; u = -k sign(s) ('sign', the paper) or -k s/||s|| ('unit', the
    ComfyUI node's form); e_hat = e + u; store e_hat; d_hat = d_u + w e_hat. k = 0 is plain CFG.
    Defaults: official README lam 5, k 0.2 (paper grid: lam 6, k 0.1 SD3.5/Qwen, 0.7 Flux).
    k = None picks SMC_AUTO_K for the model: 0.2 on flow models (the paper's velocity setting) and 0.01 on
    noise-prediction models, where the denoised estimate moves by sigma * w * k per element (sigma up to 14.6
    on SDXL), so the paper's 0.2 wrecks the image there (measured at 1024 x 1024; 0.005 and 0.01 work).
    Extra cost: none (one latent buffer).
    """
    if k is None:
        k = SMC_AUTO_K["flow" if flow else "eps"]
    w = _scalar(w, d_c)
    e = d_c - d_u
    if state is None or state.e_prev is None or state.e_prev.shape != e.shape:
        e_prev = e
    else:
        e_prev = state.e_prev
    s = (e - e_prev) + lam * e_prev
    if switching == "sign":
        u = -k * torch.sign(s)
    else:
        u = -k * s / _norm(s).clamp_min(_EPS)
    e_hat = e + u
    if state is not None:
        state.e_prev = e_hat
    return d_u + w * e_hat


def pmc_beta(d_c, d_u, w=5.0, gamma_cap=1.05):
    """Per-sample PMC-CFG coefficient beta* = min(w - 1, beta_cap) (see pmc_cfg); d_c, d_u are x0."""
    d_c, d_u = _up(d_c), _up(d_u)
    wt = _tensor(w, d_c)
    D = d_c - d_u
    a = _sqnorm(D)
    b = _dot(d_c, D)
    q = _sqnorm(d_c)
    disc = (b * b + (gamma_cap ** 2 - 1) * a * q).clamp_min(0)
    safe_a = torch.where(a > 0, a, torch.ones_like(a))
    beta_cap = (-b + disc.sqrt()) / safe_a
    nominal = (wt - 1) * torch.ones_like(a)
    return torch.where(a > 0, torch.minimum(nominal, beta_cap), nominal)


@_keep_dtype
def pmc_cfg(d_c, d_u, w=5.0, gamma_cap=1.05):
    """PMC-CFG, posterior-mean-capped CFG [x0]. Peng & Ma (2026), arXiv 2609.24287.

    D = x0_c - x0_u; a = ||D||^2, b = <x0_c, D>, q = ||x0_c||^2 (per sample).
    beta_cap = (-b + sqrt(b^2 + (G^2 - 1) a q)) / a, the largest beta with ||x0_c + beta D|| <= G ||x0_c||;
    beta* = min(w - 1, beta_cap) (w - 1 if a = 0); x0_hat = x0_c + beta* D. Because beta* is per sample
    and the map is linear, the same beta* applied to velocities (f_c + beta* (f_c - f_u)) gives the same
    x0 (use pmc_beta). w = 1 returns x0_c; large G is plain CFG.
    Defaults: G 1.05 (1.05-1.15; 1.15 on SD3.5). Extra cost: none.
    """
    beta = pmc_beta(d_c, d_u, w, gamma_cap)
    return d_c + beta * (d_c - d_u)


@_keep_dtype
def adamag(d_c, d_u, w=5.0, t=None, n_c=None, x_t=None, beta=0.1, gamma=4.0, w_min=1.0):
    """AdaMaG, adaptive manifold guidance [noise: flow velocity]. Esmati, Hyung, Dadashzadeh, Choo &
    Mirmehdi (2026), arXiv 2605.20079 (medium).

    RF convention (t = 1 noise). n_c = x_t + (1 - t) f_c, the conditional noise estimate (pass n_c
    directly for other paths). g = f_c - f_u; g_par = (<g, n_c>/||n_c||^2) n_c; g_perp = g - g_par.
    omega(t) = max(w_min, w t^gamma) (paper: (1 - tau)^gamma with tau = 0 at noise).
    f_hat = f_u + omega(t) (g_perp + beta g_par). beta = 1 with gamma = 0 is plain CFG.
    Defaults: beta 0.1, gamma 4, w = the usual CFG scale; w_min 1.0 (not stated in the paper text read).
    Extra cost: none.
    """
    if t is None:
        raise ValueError("adamag needs t (normalized noise level, 1 = noise)")
    tt = _scalar(t, d_c)
    if n_c is None:
        if x_t is None:
            raise ValueError("adamag needs n_c or x_t")
        n_c = x_t + (1 - tt) * d_c
    w = _scalar(w, d_c)
    g = d_c - d_u
    g_par, g_perp = _project(g, n_c)
    omega = torch.clamp(w * tt ** gamma, min=w_min) if torch.is_tensor(tt) else max(w_min, _float(w) * tt ** gamma)
    return d_u + omega * (g_perp + beta * g_par)


class FeedbackGuidanceState:
    """Running log posterior for Feedback Guidance (FBG). Koulischer et al. (2025), arXiv 2506.06085.

    configure(sigmas) derives the offset (delta) and temperature (tau) from pi, t0, t1 exactly as the
    official code's reparameterization: delta = log((1 - pi) lambda_ref/(lambda_ref - 1)) / ((1 - t0) N),
    tau = |2 s2(t1) delta / 10|, s2(t1) = transition variance at normalized time t1 (1 = noise),
    linearly interpolated on the step grid. Sigmas are EDM / VE sigmas (descending, final 0)."""

    def __init__(self, pi=0.95, t0=0.5, t1=0.4, lambda_max=10.0, lambda_ref=3.0, w_c=1.0,
                 temp=None, offset=None, log_posterior_init=0.0, zero_offset_below_t=None):
        self.pi = float(pi)
        self.t0, self.t1 = float(t0), float(t1)
        self.lambda_max = float(lambda_max)
        self.lambda_ref = float(lambda_ref)
        self.w_c = float(w_c)
        self._temp, self._offset = temp, offset
        self.temp, self.offset = temp, offset
        self.L0 = float(log_posterior_init)
        self.zero_offset_below_t = zero_offset_below_t
        self.num_steps = None
        self.sigmas = None
        self.L = None
        self._pending = None

    def reset(self):
        self.L = None
        self._pending = None
        self.num_steps = None
        self.sigmas = None
        self.temp, self.offset = self._temp, self._offset

    @property
    def configured(self):
        return self.num_steps is not None

    def configure(self, sigmas):
        sig = [float(s) for s in sigmas]
        n = len(sig) - 1
        if n < 1:
            raise ValueError("need at least two sigmas")
        self.sigmas, self.num_steps = sig, n
        s2 = [(sig[i] ** 2 - sig[i + 1] ** 2) * sig[i + 1] ** 2 / max(sig[i] ** 2, _EPS) for i in range(n)]
        times = [1.0 - i / n for i in range(n)]
        s2_t1 = s2[-1]
        for i in range(n - 1):
            if times[i] >= self.t1 >= times[i + 1]:
                f = (times[i] - self.t1) / max(times[i] - times[i + 1], _EPS)
                s2_t1 = s2[i] + f * (s2[i + 1] - s2[i])
                break
        if self._offset is None:
            self.offset = math.log((1 - self.pi) * self.lambda_ref / (self.lambda_ref - 1)) / ((1 - self.t0) * n)
        if self._temp is None:
            self.temp = abs(2 * s2_t1 * self.offset / 10.0)

    @property
    def L_min(self):
        lam_max = self.lambda_max - (self.w_c - 1)
        return math.log((1 - self.pi) * lam_max / max(lam_max - 1, _EPS))

    def scale(self, ref: Tensor) -> Tensor:
        if self.L is None or self.L.shape[0] != ref.shape[0]:
            self.L = torch.full((ref.shape[0],), self.L0, device=ref.device, dtype=torch.float32)
        eL = torch.exp(self.L)
        lam = eL / (eL - (1 - self.pi))
        return _bview(lam.to(ref.dtype), ref) + (self.w_c - 1)

    def update(self, x, x_next, d_c, d_u, sigma, sigma_next):
        """Posterior update after the step x (sigma) -> x_next (sigma_next); x0 estimates d_c, d_u."""
        s, sn = _float(sigma), _float(sigma_next)
        if sn <= 0 or s <= 0 or self.temp is None:
            return
        r = sn ** 2 / s ** 2
        s2 = (s ** 2 - sn ** 2) * r
        mu_c = r * x + (1 - r) * d_c
        mu_u = r * x + (1 - r) * d_u
        diff = ((x_next - mu_c) ** 2).flatten(1).sum(1) - ((x_next - mu_u) ** 2).flatten(1).sum(1)
        off = self.offset
        if self.zero_offset_below_t is not None and self.sigmas is not None:
            t_norm = 1.0 - self._step_of(s) / max(self.num_steps, 1)
            if t_norm < self.zero_offset_below_t:
                off = 0.0
        if self.L is None:
            self.L = torch.full_like(diff, self.L0)
        self.L = torch.clamp(self.L - self.temp / (2 * s2) * diff.float() + off, self.L_min, 3.0)

    def _step_of(self, sigma):
        for i, s in enumerate(self.sigmas):
            if s <= sigma + 1e-8:
                return i
        return len(self.sigmas) - 1

    def remember(self, x, d_c, d_u, sigma):
        self._pending = (x.detach().float(), d_c.detach().float(), d_u.detach().float(), _float(sigma))

    def observe(self, x_now, sigma_now):
        """Complete a pending update using the latent the sampler produced (single-stage samplers)."""
        if self._pending is None:
            return
        x, dc, du, s = self._pending
        sn = _float(sigma_now)
        if sn < s and x.shape == x_now.shape:
            self.update(x, x_now.float(), dc, du, s, sn)
        self._pending = None


@_keep_dtype
def fbg(d_c, d_u, w=1.0, state=None, x_t=None, sigma=None, sigmas=None, flow=False, hybrid=False):
    """Feedback Guidance, state-dependent scale from a running posterior [x0, EDM sigma].
    Koulischer, Handke, Deleu, Demeester & Ambrogioni (2025), NeurIPS, arXiv 2506.06085.

    lambda = e^L / (e^L - (1 - pi)) from the state's log posterior L (L = 0 at the start);
    x0_hat = x0_u + lambda (x0_c - x0_u); hybrid=True adds (w - 1) (FBG_CFG).
    After each step the state updates L <- clamp(L - tau/(2 s2) (||x' - mu_c||^2 - ||x' - mu_u||^2)
    + delta, L_min, 3), mu = r x + (1 - r) D, r = sigma'^2/sigma^2, s2 = (sigma^2 - sigma'^2) r.
    Two ways to drive it: call state.update(...) after each solver step yourself, or pass x_t and sigma
    on every call (lazy mode: the next call's x_t is taken as the previous step's result; valid for
    single-stage samplers). Flow models: pass flow=True (x_t and sigmas are mapped to VE, x/(1-t),
    t/(1-t)). Without a state this is plain CFG at w.
    Defaults (state): pi 0.95, t0 0.5, t1 0.4, lambda_max 10 (sampler defaults; paper T2I images use
    pi 0.85, t0 0.75, t1 0.5). Extra cost: none (two squared-distance sums per step).
    """
    if state is None:
        return cfg(d_c, d_u, w)
    xv, sv, sig_list = x_t, sigma, sigmas
    if flow:
        if sigma is not None:
            t_now = min(_float(sigma), 1 - 1e-4)
            sv = t_now / (1 - t_now)
            if x_t is not None:
                xv = x_t / (1 - t_now)
        if sigmas is not None:
            sig_list = [min(float(s), 1 - 1e-4) / (1 - min(float(s), 1 - 1e-4)) for s in sigmas]
    if sig_list is not None and not state.configured:
        state.configure(sig_list)
    if xv is not None and sv is not None:
        state.observe(xv, sv)
    lam = state.scale(d_c)
    if hybrid:
        lam = lam + (_scalar(w, d_c) - 1)
    out = d_u + lam * (d_c - d_u)
    if xv is not None and sv is not None:
        state.remember(xv, d_c, d_u, sv)
    return out


@_keep_dtype
def vags(d_c, d_u, w=7.0, t=None, kappa=1.0):
    """VAGS, velocity adaptive guidance scale [noise: velocity]. Luo, Aidara, Lu, Moebel, Han & Wang
    (2026), arXiv 2605.15661 (medium).

    sigma_i = 1 - t (signal level; t = 1 is noise); s = cos(d_u, d_c) per sample;
    w_i = w exp(kappa (2 sigma_i - 1) s); d_hat = d_u + w_i delta. kappa = 0 is plain CFG.
    Defaults: w 7, kappa 1.0 (0.9 on Flickr30K). Extra cost: about 1-2% wall clock.
    """
    if t is None:
        raise ValueError("vags needs t (1 = noise)")
    w = _scalar(w, d_c)
    sig = 1 - _scalar(t, d_c)
    w_i = w * torch.exp(kappa * (2 * sig - 1) * _cos(d_u, d_c))
    return d_u + w_i * (d_c - d_u)


class OECState:
    """Previous step's raw (uncorrected) d_c and d_u for CFG-OEC."""

    def __init__(self):
        self.prev = None

    def reset(self):
        self.prev = None


@_keep_dtype
def cfg_oec(d_c, d_u, w=7.5, state=None, tau=0.5):
    """CFG-OEC, orthogonal error correction of the null branch [noise: eps]. Yang, Lee & Han (2025),
    arXiv 2511.14075 (medium).

    From the previous step's predictions: eps~_u = 2 d_u - d_u_prev, eps~_c = 2 d_c - d_c_prev;
    A = d_c - eps~_c, B = d_u - eps~_u; s = cos(A, B). If s < tau: B_perp = B - (<A,B>/||A||^2) A,
    u_bar = eps~_u + B_perp, u~ = (1 - s) u_bar + s d_u; else u~ = d_u. First step: u~ = d_u.
    d_hat = u~ + w (d_c - u~). The cache keeps the uncorrected predictions. tau <= -1 is plain CFG.
    Defaults: tau 0.5 is a placeholder (the paper text read gives no value); tune it.
    Extra cost: none (two cached latents).
    """
    w = _scalar(w, d_c)
    if state is None or state.prev is None or state.prev[0].shape != d_c.shape:
        u_t = d_u
    else:
        pc, pu = state.prev
        et_u = 2 * d_u - pu
        et_c = 2 * d_c - pc
        A = d_c - et_c
        Bv = d_u - et_u
        s = _cos(A, Bv)
        b_perp = Bv - _dot(A, Bv) / _sqnorm(A).clamp_min(_EPS) * A
        u_bar = et_u + b_perp
        u_t = torch.where(s < tau, (1 - s) * u_bar + s * d_u, d_u)
    if state is not None:
        state.prev = (d_c.detach().clone(), d_u.detach().clone())
    return u_t + w * (d_c - u_t)


# ---------------------------------------------------------------------------
# Solver-coupled variants that need model evaluations (callables)
# ---------------------------------------------------------------------------


def _channel_project(v: Tensor, g: Tensor) -> Tensor:
    B, C = v.shape[:2]
    vf, gf = v.reshape(B, C, -1), g.reshape(B, C, -1)
    coef = (vf * gf).sum(-1, keepdim=True) / (gf * gf).sum(-1, keepdim=True).clamp_min(_EPS)
    return (coef * gf).reshape(v.shape)


def characteristic_guidance(x_t, eps_c_fn, eps_u_fn, w=6.0, sigma=1.0, projection="latent",
                            max_iter=10, tol=1e-3, relax=1.0, fallback=True):
    """Characteristic guidance, fixed-point nonlinear correction [noise: eps, VP]. Zheng & Lan (2024),
    ICML, arXiv 2312.07586.

    omega = w - 1, sigma = sqrt(1 - alphabar) (VP noise std). Solve
      dx = P[(eps(x + w dx; u) - eps(x + omega dx; c)) * sigma]
    by damped fixed-point iteration from dx = 0, then eps_CH = w eps(x + omega dx; c) - omega eps(x + w dx; u).
    P is a channel-wise projection onto g: 'latent' g = (eps_u(x) - eps_c(x)) sigma per channel,
    'pixel' g = 1 (channel mean), 'identity' P = I. Falls back to plain CFG if it does not converge
    (paper). eps_c_fn(x) / eps_u_fn(x) evaluate the model at the same noise level.
    Defaults: max_iter 10, tol 1e-3 (RMS change of dx). Extra cost: 2 NFE per iteration (up to about 20).
    The paper also uses SOR, RMSprop or Anderson acceleration; this is the plain damped iteration.
    """
    om = w - 1.0
    ec0, eu0 = eps_c_fn(x_t), eps_u_fn(x_t)
    if projection == "latent":
        g = (eu0 - ec0) * sigma
        P = lambda v: _channel_project(v, g)  # noqa: E731
    elif projection == "pixel":
        P = lambda v: _channel_project(v, torch.ones_like(v))  # noqa: E731
    else:
        P = lambda v: v  # noqa: E731
    dx = torch.zeros_like(x_t)
    ec, eu = ec0, eu0
    converged = False
    for _ in range(max_iter):
        step = P((eu - ec) * sigma) - dx
        dx = dx + relax * step
        ec, eu = eps_c_fn(x_t + om * dx), eps_u_fn(x_t + w * dx)
        if float(step.flatten(1).pow(2).mean(1).sqrt().max()) < tol:
            converged = True
            break
    if not converged and fallback:
        return w * ec0 - om * eu0
    return w * ec - om * eu


# =============================================================================
# 3. Schedules and interval gating (family B)
# =============================================================================


@dataclass
class GuidanceContext:
    """Where a guided step sits in the trajectory.

    step: solver step index (0 = first, noisiest); num_steps: N; progress: step/(N - 1) in [0, 1]
    (computed if not given); sigma: the native noise level; t: normalized noise level in [0, 1] with
    1 = pure noise (flow: t = sigma; VE: t = sigma/(1 + sigma), the SNR-matched flow time; falls back
    to 1 - progress)."""

    step: float = 0
    num_steps: int = 1
    sigma: Optional[float] = None
    t: Optional[float] = None
    progress: Optional[float] = None

    def __post_init__(self):
        if self.progress is None:
            self.progress = 0.0 if self.num_steps <= 1 else min(max(self.step / (self.num_steps - 1), 0.0), 1.0)
        if self.t is None:
            self.t = 1.0 - self.progress


Schedule = Callable[[GuidanceContext], float]


def constant_schedule(w: float) -> Schedule:
    """w at every step."""
    return lambda ctx: float(w)


def table_schedule(values: Sequence[float]) -> Schedule:
    """Per-step table (index = int(step), clamped)."""
    vals = [float(v) for v in values]
    return lambda ctx: vals[min(max(int(ctx.step), 0), len(vals) - 1)]


def wang_schedule(w_bar: float, shape: str = "linear", reading: str = "A", clamp_min: Optional[float] = None,
                  kappa: float = 1.0) -> Schedule:
    """Monotone / shaped CFG weight schedules. Wang, Dufour, Andreou, Cani, Fernandez Abrevaya, Picard &
    Kalogeiton (2024), TMLR, arXiv 2404.13040.

    tau = progress (0 at the noisiest step). omega(tau; ob), area-normalized to ob over [0, 1]:
    linear 2 ob tau; cosine ob (1 - cos(pi tau)); invlinear 2 ob (1 - tau); sine ob (1 + cos(pi tau));
    v_shape 2 ob |1 - 2 tau|; lambda_shape 2 ob (1 - |1 - 2 tau|); pcs ob (1 - cos(pi tau^kappa))/2
    (not normalized, peak ob). The paper is ambiguous about whether the ramp multiplies w or w - 1:
    reading 'A' (Eq. 4 literal): w(tau) = 1 + omega(tau; w_bar - 1), ramping around the conditional;
    reading 'B' (magnitudes as reported): w(tau) = omega(tau; w_bar), starting at the unconditional.
    clamp_min c gives the clamp variant max(c, w) (paper: c = 2 SD1.5, 4 SDXL, 0.5 SD3).
    """
    def omega(tau, ob):
        if shape == "linear":
            return 2 * ob * tau
        if shape == "cosine":
            return ob * (1 - math.cos(math.pi * tau))
        if shape == "invlinear":
            return 2 * ob * (1 - tau)
        if shape == "sine":
            return ob * (1 + math.cos(math.pi * tau))
        if shape == "v_shape":
            return 2 * ob * abs(1 - 2 * tau)
        if shape == "lambda_shape":
            return 2 * ob * (1 - abs(1 - 2 * tau))
        if shape == "pcs":
            return ob * (1 - math.cos(math.pi * tau ** kappa)) / 2
        raise ValueError(f"unknown shape {shape!r}")

    def sched(ctx):
        tau = ctx.progress
        wv = 1 + omega(tau, w_bar - 1) if reading == "A" else omega(tau, w_bar)
        return max(clamp_min, wv) if clamp_min is not None else wv

    return sched


def tv_cfg_table(w_bar: float, num_steps: int, times: Optional[Sequence[float]] = None, peak: float = 0.5) -> List[float]:
    """TV-CFG stage-wise, middle-peaked linear schedule. Jin, Shi & Gu (2025), ICLR 2026, arXiv 2509.22007.

    s = w_bar - 1, M = ceil(N * peak). w_raw(n) = (2s/M) n + w_bar - s for n <= M, and
    (2s/(N - M)) (N - n) + w_bar - s after (1 at the ends, 2 w_bar - 1 at the peak). Normalized
    w_n = A w_raw(n) with sum_n w_n (t_n - t_{n+1}) = w_bar (t_0 = 1 ... t_N = 0 model times;
    uniform if times is None). Returns N per-step weights (Ho convention)."""
    N = int(num_steps)
    if N < 1:
        return []
    s = w_bar - 1
    M = max(1, min(N, int(math.ceil(N * peak))))
    raw = []
    for n in range(N):
        if n <= M:
            raw.append((2 * s / M) * n + w_bar - s)
        else:
            raw.append((2 * s / max(N - M, 1)) * (N - n) + w_bar - s)
    if times is None:
        times = [1 - i / N for i in range(N + 1)]
    times = [float(t) for t in times]
    area = sum(raw[n] * (times[n] - times[n + 1]) for n in range(N))
    total = times[0] - times[N]
    A = (w_bar * total) / area if area != 0 else 1.0
    return [A * r for r in raw]


def tv_cfg_schedule(w_bar: float, num_steps: int, times: Optional[Sequence[float]] = None, peak: float = 0.5) -> Schedule:
    """TV-CFG as a per-step schedule (see tv_cfg_table)."""
    return table_schedule(tv_cfg_table(w_bar, num_steps, times, peak))


def c2fg_schedule(omega0: float, lam: float = math.log(2)) -> Schedule:
    """C2FG exponential control schedule. Gao et al. (2026), CVPR, arXiv 2603.08155.
    omega(t) = omega0 exp(lam (1 - t)), t = normalized noise level (1 = noise): omega0 at the start,
    omega0 e^lam at the end (Ho convention). Defaults: lam = ln 2 (DiT), 1 (SiT SDE), 0.2 (SD1.5)."""
    return lambda ctx: float(omega0) * math.exp(lam * (1 - ctx.t))


def beta_schedule(w: float, a: float = 2.0, b: float = 2.0, peak_normalize: bool = False) -> Schedule:
    """beta-CFG's time profile alone (its gamma = 0 'ddim_beta' form): w(p) = w * Beta_pdf(p; a, b).
    arXiv 2502.10574. Beta(2, 2) = 6 p (1 - p) has time average 1."""
    return lambda ctx: float(w) * beta_pdf(ctx.progress, a, b, peak_normalize)


def dg_cfg_table(w_bar: float, alphabar: Sequence[float], betas: Sequence[float],
                 dt: Optional[Sequence[float]] = None) -> List[float]:
    """DG-CFG distribution-guided schedule. Jiang & Ma (2026), arXiv 2607.19725.

    f(t) = (1 - alphabar_t) sqrt(alphabar_t) / beta(t); omega(t) = 1 + C (w_bar - 1) f(t) with
    C = sum(dt) / sum(f dt), so the time integral of (omega - 1) matches constant CFG. alphabar and
    betas are per-step arrays of the VP scheduler at the sampled timesteps; dt the step widths
    (uniform if None). Returns per-step weights."""
    ab = [float(a) for a in alphabar]
    bt = [float(b) for b in betas]
    n = len(ab)
    dts = [1.0] * n if dt is None else [float(d) for d in dt]
    f = [(1 - a) * math.sqrt(a) / max(b, _EPS) for a, b in zip(ab, bt)]
    denom = sum(fi * di for fi, di in zip(f, dts))
    C = sum(dts) / denom if denom > 0 else 0.0
    return [1 + C * (w_bar - 1) * fi for fi in f]


def cads_gamma(t: float, tau1: float = 0.6, tau2: float = 0.9) -> float:
    """CADS annealing gate (t = 1 noise): 1 for t <= tau1, linear to 0 at tau2, 0 above.
    Sadat, Buhmann, Bradley, Hilliges & Weber (2023/2024), ICLR, arXiv 2310.17347."""
    if t <= tau1:
        return 1.0
    if t >= tau2:
        return 0.0
    return (tau2 - t) / (tau2 - tau1)


def cads_dynamic_schedule(w: float, tau1: float = 0.6, tau2: float = 0.9) -> Schedule:
    """CADS's 'Dynamic CFG' companion: w(t) = gamma(t) w (pure unconditional above tau2)."""
    return lambda ctx: float(w) * cads_gamma(ctx.t, tau1, tau2)


@_keep_dtype
def cads_anneal_condition(y, t, tau1=0.6, tau2=0.9, noise_scale=0.25, psi=1.0, rescale=True, generator=None):
    """CADS condition annealing [condition embedding]. arXiv 2310.17347.

    y_hat = sqrt(gamma(t)) y + s sqrt(1 - gamma(t)) n, n ~ N(0, I); rescale to the clean vector's
    scalar mean/std per sample, y_r = (y_hat - mean(y_hat))/std(y_hat) std(y) + mean(y);
    y_final = psi y_r + (1 - psi) y_hat. Apply to BOTH the prompt and the null embedding with independent
    draws, then run ordinary CFG. Defaults (SD, w = 9): tau1 0.6, tau2 0.9, s 0.25, psi 1.
    Extra cost: none (noise draws)."""
    g = cads_gamma(_float(t), tau1, tau2)
    dev = generator.device if generator is not None else y.device
    n = torch.randn(y.shape, generator=generator, device=dev, dtype=torch.float32).to(y.device, y.dtype)
    y_hat = math.sqrt(g) * y + noise_scale * math.sqrt(1 - g) * n
    if not rescale:
        return y_hat
    y_r = (y_hat - y_hat.flatten(1).mean(1).view(-1, *([1] * (y.ndim - 1)))) \
        / _std(y_hat).clamp_min(_EPS) * _std(y) + y.flatten(1).mean(1).view(-1, *([1] * (y.ndim - 1)))
    return psi * y_r + (1 - psi) * y_hat


def interval_schedule(w: float, sigma_lo: float = 0.28, sigma_hi: float = 5.42, outside: float = 1.0) -> Schedule:
    """Guidance in a limited interval (LIG) as a schedule: w for sigma_lo < sigma <= sigma_hi, `outside`
    (1 = conditional) elsewhere. Kynkaanniemi et al. (2024), NeurIPS, arXiv 2404.07724.
    Defaults: SD-XL (0.28, 5.42] in EDM sigma units with w = 16 on 32 Heun steps."""
    return lambda ctx: float(w) if (ctx.sigma is not None and sigma_lo < ctx.sigma <= sigma_hi) else float(outside)


def cutoff_schedule(w: float, start: float = 0.0, end: float = 1.0, outside: float = 1.0) -> Schedule:
    """Progress-window gating: w for start <= progress <= end, `outside` elsewhere.
    Covers CFG truncation (Lumina-Image 2.0 cfg_trunc_ratio r: end = r), A1111 'skip early CFG'
    (start = p), diffusers CFGCutoffCallback (end = r) and the commitment horizon (arXiv 2608.08082)."""
    return lambda ctx: float(w) if (start <= ctx.progress <= end) else float(outside)


def early_high_late_uncond_schedule(w: float, switch: float = 0.5) -> Schedule:
    """Early-high guidance with a late unconditional window. Ventura, Achilli, Ambrogioni & Lucibello
    (2026), arXiv 2602.00716. w before the switch, 0 (d_hat = d_u) after. Paper: switch 50-70%."""
    return lambda ctx: float(w) if ctx.progress < switch else 0.0


def ramp_up_schedule(w: float, r: float = 0.5) -> Schedule:
    """Masked-diffusion ramp-up (Rojas et al. 2025, arXiv 2507.08965): w(t) = min(w, w (1 - t)/(1 - r)),
    t = 1 fully masked / pure noise."""
    return lambda ctx: min(float(w), float(w) * (1 - ctx.t) / max(1 - r, _EPS))


def compress_guidance_steps(num_steps: int, n_guided: int, k: float = 1.0) -> List[int]:
    """Guided step indices of Compress Guidance. Dinh, Liu & Xu (2024), arXiv 2408.11194 (medium).
    G_i = T - floor((T/|G|^k) i^k), i = 0..|G|-1, mapped to step indices (0 = noisiest)."""
    T, G = int(num_steps), max(1, int(n_guided))
    idx = sorted({min(T - 1, int(math.floor((T / G ** k) * i ** k))) for i in range(G)})
    return idx


def compress_guidance_schedule(w: float, num_steps: int, n_guided: int, k: float = 1.0) -> Schedule:
    """Compress Guidance in CFG form (RECONSTRUCTED; the paper prints only the classifier-gradient form):
    on a guided step w_eff = 1 + (w - 1) L_i with L_i the number of steps until the next guided step;
    elsewhere w = 1 (d_c, one NFE)."""
    guided = compress_guidance_steps(num_steps, n_guided, k)
    table = [1.0] * int(num_steps)
    for j, g in enumerate(guided):
        nxt = guided[j + 1] if j + 1 < len(guided) else int(num_steps)
        table[g] = 1 + (w - 1) * (nxt - g)
    return table_schedule(table)


def var_cfg_scale(cfg_value: float, scale_index: int, num_scales: int) -> float:
    """VAR next-scale ratio schedule (Tian et al. 2024, arXiv 2404.02905): w_s = 1 + cfg * s/(K - 1)."""
    return 1 + cfg_value * scale_index / max(num_scales - 1, 1)


def mar_cfg_scale(w: float, num_masked: int, total: int) -> float:
    """MAR / Muse linear schedule over unmasked tokens (Li et al. 2024, arXiv 2406.11838):
    w_k = 1 + (w - 1) (N - m_k)/N, m_k = tokens still masked."""
    return 1 + (w - 1) * (total - num_masked) / max(total, 1)


def with_schedule(fn: Callable, schedule: Schedule) -> Callable:
    """Wrap a combiner so its w comes from schedule(ctx): scheduled(d_c, d_u, ctx, **kw)."""

    def scheduled(d_c, d_u, ctx: GuidanceContext, **kwargs):
        return fn(d_c, d_u, w=schedule(ctx), **kwargs)

    scheduled.__name__ = f"scheduled_{getattr(fn, '__name__', 'fn')}"
    scheduled.__doc__ = f"{getattr(fn, '__name__', 'fn')} with a scheduled w (see with_schedule)."
    return scheduled


def outside_value(d_c, d_u, w, outside="cond"):
    """Fallback prediction outside a guidance window: 'cond' (d_c), 'uncond' (d_u), 'cfg' (plain CFG at w)
    or a number (plain CFG at that weight)."""
    if outside == "cond":
        return d_c
    if outside == "uncond":
        return d_u
    if outside == "cfg":
        return cfg(d_c, d_u, w)
    return cfg(d_c, d_u, float(outside))


def interval_mask(sigma, sigma_lo: float, sigma_hi: float, batch: int, device=None) -> Tensor:
    """Per-sample bool mask sigma_lo < sigma <= sigma_hi (LIG's half-open interval)."""
    s = torch.as_tensor(sigma, dtype=torch.float32, device=device).reshape(-1)
    if s.numel() == 1 and batch > 1:
        s = s.expand(batch)
    return (s > sigma_lo) & (s <= sigma_hi)


def with_interval(fn: Callable, sigma_lo: float = 0.28, sigma_hi: float = 5.42, outside="cond") -> Callable:
    """Gate a combiner to a noise interval (LIG, arXiv 2404.07724): inside sigma_lo < sigma <= sigma_hi
    call fn, outside return outside_value (default d_c, so the null pass can be skipped there).
    gated(d_c, d_u, w, sigma, **kw); sigma may be a float or a per-sample tensor."""

    def gated(d_c, d_u, w=7.5, sigma=None, **kwargs):
        if sigma is None:
            raise ValueError("with_interval needs sigma")
        inside = interval_mask(sigma, sigma_lo, sigma_hi, d_c.shape[0], d_c.device)
        if bool(inside.all()):
            return fn(d_c, d_u, w=w, **kwargs)
        fb = outside_value(d_c, d_u, w, outside)
        if not bool(inside.any()):
            return fb
        return torch.where(_bview(inside, d_c), fn(d_c, d_u, w=w, **kwargs), fb)

    gated.__name__ = f"interval_{getattr(fn, '__name__', 'fn')}"
    gated.__doc__ = f"{getattr(fn, '__name__', 'fn')} gated to ({sigma_lo}, {sigma_hi}] (see with_interval)."
    return gated


class AdaptiveGuidanceState:
    """Per-sample 'switched' flags for Adaptive Guidance. needs_uncond is False once every sample
    switched, so the caller can stop running the null pass."""

    def __init__(self):
        self.switched = None

    def reset(self):
        self.switched = None

    @property
    def needs_uncond(self):
        return self.switched is None or not bool(self.switched.all())


@_keep_dtype
def adaptive_guidance(d_c, d_u, w=7.5, state=None, threshold=0.991):
    """Adaptive Guidance (AG) [noise: eps]. Castillo et al. (2023/2025), AAAI, arXiv 2312.12487.

    gamma = cos(d_c, d_u) per sample. Full CFG while gamma <= threshold; from the first step where
    gamma > threshold on, d_hat = d_c (the null pass can be skipped: state.needs_uncond).
    Defaults: threshold 0.991 (20-step LDM; 0.993 -> 32 of 40 NFE). ComfyUI's AdaptiveGuidance pack
    computes the cosine on x0, so its thresholds differ. Extra cost: negative (about 25% fewer NFE).
    """
    w = _scalar(w, d_c)
    sw = _cos(d_c, d_u) > threshold
    if state is not None:
        if state.switched is not None and state.switched.shape == sw.shape:
            sw = sw | state.switched
        state.switched = sw
    return torch.where(sw, d_c, d_u + w * (d_c - d_u))


class TransitionPointState:
    def __init__(self):
        self.history: List[Tensor] = []
        self.switched = None

    def reset(self):
        self.history = []
        self.switched = None


@_keep_dtype
def transition_point_guidance(d_c, d_u, w=7.5, state=None, opposite=0.0):
    """Transition-point / opposite guidance against memorization [noise: eps]. Jain, Kobayashi, Shibuya,
    Takida, Memon, Togelius & Mitsufuji (2024), CVPR 2025, arXiv 2411.16738.

    Track d = ||delta||^2 per step. Before the transition use w_pre = -opposite (0: d_hat = d_u;
    opposite guidance: d_u - lambda delta). The transition is the step after a local minimum of d
    (d two steps ago > d one step ago < d now); from then on d_hat = d_u + w delta.
    Without a state this is plain CFG. Extra cost: none.
    """
    w = _scalar(w, d_c)
    if state is None:
        return d_u + w * (d_c - d_u)
    dist = (d_c - d_u).flatten(1).pow(2).sum(1)
    h = state.history
    h.append(dist)
    sw = state.switched if (state.switched is not None and state.switched.shape == dist.shape) \
        else torch.zeros_like(dist, dtype=torch.bool)
    if len(h) >= 3 and h[-3].shape == dist.shape:
        sw = sw | ((h[-3] > h[-2]) & (h[-2] < h[-1]))
    state.switched = sw
    w_eff = torch.where(_bview(sw, d_c), _tensor(w, d_c) * torch.ones_like(_bview(dist, d_c)),
                        torch.full_like(_bview(dist, d_c), -float(opposite)))
    return d_u + w_eff * (d_c - d_u)


@_keep_dtype
def windowed_negative(d_c, d_u, d_n, w=7.5, progress=0.0, start=0.17, end=0.5):
    """Delayed / windowed negative prompt [any]. Ban, Wang, Zhou, Cheng, Gong & Hsieh (2024), ECCV,
    arXiv 2406.02965.

    d_weak = d_n for start <= progress < end, else d_u (empty prompt); d_hat = d_weak + w (d_c - d_weak).
    Defaults: steps 5-15 of 30 (0.17-0.5); critical step about 5/30 for nouns, 10/30 for adjectives;
    end = 1 gives delay-only. Extra cost: none (the negative replaces the null inside the window,
    but both d_u and d_n are needed across the run)."""
    w = _scalar(w, d_c)
    p = _float(progress)
    weak = d_n if (start <= p < end) else d_u
    return weak + w * (d_c - weak)


@_keep_dtype
def segmented_guidance(d_c, d_u, d_weak, w=7.5, t=1.0, tau=0.2):
    """Weak-to-Strong Segmented Guidance (SGG) [noise: velocity]. Yuan et al. (2026), arXiv 2603.20584.
    g = d_c - d_u for t > tau (high noise: CFG), g = d_c - d_weak for t <= tau (condition-agnostic
    guidance from an inferior branch); d_hat = d_c + (w - 1) g. Defaults: tau 0.2 (0.1-0.3)."""
    w = _scalar(w, d_c)
    tt = _tensor(t, d_c)
    g = torch.where(tt > tau, d_c - d_u, d_c - d_weak)
    return d_c + (w - 1) * g


class CFGCacheState:
    """FasterCache CFG-Cache: stores the frequency-split guidance residual at refresh steps."""

    def __init__(self, interval=5, alpha_low=0.2, alpha_high=0.2, t0=0.5, radius_frac=0.2):
        self.interval, self.alpha_low, self.alpha_high = int(interval), alpha_low, alpha_high
        self.t0, self.radius_frac = t0, radius_frac
        self.dl = self.dh = None

    def reset(self):
        self.dl = self.dh = None

    def is_refresh_step(self, step: int, start_step: int = 0) -> bool:
        return self.dl is None or step < start_step or (step - start_step) % self.interval == 0


def _fft_low_mask(h, w, radius_frac, device):
    yy = torch.arange(h, device=device, dtype=torch.float32) - h // 2
    xx = torch.arange(w, device=device, dtype=torch.float32) - w // 2
    r = torch.sqrt(yy.view(-1, 1) ** 2 + xx.view(1, -1) ** 2)
    return (r <= radius_frac * min(h, w)).float()


@_keep_dtype
def cfg_cache(d_c, d_u=None, w=7.5, state=None, t=1.0):
    """CFG-Cache from FasterCache [noise]. Lv et al. (2024), ICLR 2025, arXiv 2410.19355.

    Refresh step (pass d_u): store DL = low(F(d_u) - F(d_c)), DH = high(...) (2-D FFT, low = disk of radius
    min(H, W) * radius_frac around DC) and return plain CFG. Reuse step (d_u=None, only the conditional
    pass run): d_u_hat = IFFT(F(d_c) + w1 DL + w2 DH), w1 = 1 + alpha_low [t > t0],
    w2 = 1 + alpha_high [t <= t0]; d_hat = d_u_hat + w (d_c - d_u_hat).
    Defaults: interval 5, alpha 0.2 / 0.2 (paper Fig. 16), t0 0.5 (placeholder: the paper's switch
    timestep was not recovered), radius 0.2 (diffusers min(H,W)//5). Extra cost: negative
    ((n - 1)/n of the null passes replaced by FFTs)."""
    if state is None:
        if d_u is None:
            raise ValueError("cfg_cache without a state needs d_u")
        return cfg(d_c, d_u, w)
    wv = _scalar(w, d_c)
    H, W = d_c.shape[-2:]
    mask = _fft_low_mask(H, W, state.radius_frac, d_c.device)
    Fc = torch.fft.fftshift(torch.fft.fft2(d_c), dim=(-2, -1))
    if d_u is not None:
        Fu = torch.fft.fftshift(torch.fft.fft2(d_u), dim=(-2, -1))
        state.dl = (Fu - Fc) * mask
        state.dh = (Fu - Fc) * (1 - mask)
        return d_u + wv * (d_c - d_u)
    if state.dl is None:
        raise ValueError("cfg_cache reuse step before any refresh step")
    tt = _float(t)
    w1 = 1 + state.alpha_low * (1.0 if tt > state.t0 else 0.0)
    w2 = 1 + state.alpha_high * (1.0 if tt <= state.t0 else 0.0)
    Uh = Fc + w1 * state.dl + w2 * state.dh
    u_hat = torch.fft.ifft2(torch.fft.ifftshift(Uh, dim=(-2, -1))).real
    return u_hat + wv * (d_c - u_hat)


# =============================================================================
# 4. Weak-branch and perturbation-guidance combiners (family C)
# =============================================================================


@_keep_dtype
def perturbation_guidance(d_c, d_pert, s=3.0, d_u=None, w=1.0, d_ref=None, rescale=0.0):
    """Perturbation guidance, the shared combiner of PAG / SEG / STG / SLG / S2 / TPG / SSG / ASAG / MA-DG /
    self-guidance / SWG [any]. The caller runs the perturbed pass (see REGISTRY[...]['caller']).

    base = d_c, or plain CFG d_u + w (d_c - d_u) when d_u is given;
    d_hat = base + s (ref - d_pert), ref = d_ref if given else d_c (the added-term convention).
    SEG's paper form perturbs the UNCONDITIONAL branch: pass d_ref = d_u, d_pert = D_seg(x_t; u).
    Optional STG/SLG rescale r: d_hat <- d_hat (r std(d_c)/std(d_hat) + 1 - r).
    s = 0 gives the base. Extra cost: 1 NFE (the perturbed pass) per step where active.
    """
    base = d_c if d_u is None else d_u + _scalar(w, d_c) * (d_c - d_u)
    ref = d_c if d_ref is None else d_ref
    out = base + s * (ref - d_pert)
    if rescale and rescale > 0:
        out = out * (rescale * _std(d_c) / _std(out).clamp_min(1e-8) + (1 - rescale))
    return out


@_keep_dtype
def stacked_guidance(d_c, d_u=None, w=1.0, terms=(), rescale=0.0):
    """Several guidance terms on one conditional prediction [any] (LTX-2 multimodal guidance,
    HaCohen et al. 2026, arXiv 2601.03233; any CFG + PAG + SEG stack).

    d_hat = d_c + (w - 1)(d_c - d_u) + sum_i s_i (d_c - d_i) for terms = [(d_i, s_i), ...].
    LTX-2 implementations: d_stg (self-attention to value pass-through in chosen blocks, s = stg scale)
    and d_mod (cross-modal attention severed, s = m - 1). Optional rescale as in perturbation_guidance."""
    out = d_c if d_u is None else d_c + (_scalar(w, d_c) - 1) * (d_c - d_u)
    for d_i, s_i in terms:
        out = out + s_i * (d_c - d_i.to(out.dtype))
    if rescale and rescale > 0:
        out = out * (rescale * _std(d_c) / _std(out).clamp_min(1e-8) + (1 - rescale))
    return out


@_keep_dtype
def autoguidance_mixed(d_c, d_u, d_weak, w=2.0, a=0.5):
    """CFG mixed with autoguidance (Karras et al. 2024, arXiv 2406.02507, DeepFloyd IF experiment) [any].
    w_u = (1 - a)(w - 1) + 1, w_c = a (w - 1) + 1; d_hat = d_c + (w_u - 1)(d_c - d_u) + (w_c - 1)(d_c - d_weak).
    a = 0 is plain CFG, a = 1 pure autoguidance (d_weak = conditional weak model)."""
    w = _scalar(w, d_c)
    wu = (1 - a) * (w - 1) + 1
    wc = a * (w - 1) + 1
    return d_c + (wu - 1) * (d_c - d_u) + (wc - 1) * (d_c - d_weak)


@_keep_dtype
def concept_guidance(d_cfg, d_skips, weights, lam=2.0):
    """Concept Guidance (CoG) [any]. Rohrich, Hans, Krause & Ommer (2026), GCPR, arXiv 2608.14172.
    d_neg = sum_i omega_i D_skip_i / sum_i omega_i over the top-k profiled skip layers (omega_i = measured
    target-score gain); d_hat = d_neg + lam (d_cfg - d_neg). Defaults: lam 2-2.5, k = 2-3 layers.
    Extra cost: +1 NFE per skipped-layer prediction; offline profiling per model and concept."""
    ws = [float(v) for v in weights]
    tot = sum(ws)
    d_neg = sum(wi * d for wi, d in zip(ws, d_skips)) / (tot if tot != 0 else 1.0)
    return d_neg + lam * (d_cfg - d_neg)


@_keep_dtype
def sag_guidance(d_c, d_u, w=7.5, d_sag=None, s=0.5, d_ref=None):
    """Self-Attention Guidance combiner [x0 (ComfyUI) or eps (paper)]. Hong, Lee, Jang & Kim (2023),
    ICCV, arXiv 2210.00939.

    d_hat = d_u + w (d_c - d_u) + s (d_ref - d_sag). ComfyUI x0 form: d_ref = x0_deg and d_sag = the
    unconditional x0 at the degraded input x_t' (build both with sag_mask_from_attention + sag_degrade).
    Paper eps form: d_ref = eps_u(x_t) (the default when d_ref is None), d_sag = eps_u(x_t').
    Defaults: s 0.5 (ComfyUI; paper SD 0.75-1.0), blur_sigma 2 (ComfyUI). Extra cost: 1 NFE.
    """
    if d_sag is None:
        raise ValueError("sag_guidance needs d_sag (the prediction at the degraded input)")
    ref = d_u if d_ref is None else d_ref
    return d_u + _scalar(w, d_c) * (d_c - d_u) + s * (ref - d_sag)


def sag_mask_from_attention(attn: Tensor, latent_hw: Tuple[int, int], grid_hw: Optional[Tuple[int, int]] = None,
                            threshold: float = 1.0) -> Tensor:
    """SAG saliency mask. attn: (B, heads, N, N) self-attention probabilities (rows = queries) of the
    unconditional pass. a_j = sum over queries of the head-mean A[q, j] (mean 1.0); mask = a > threshold,
    reshaped to the attention grid and nearest-upsampled to latent_hw. Returns (B, 1, H, W) float."""
    a = attn.float().mean(1).sum(1)
    B, N = a.shape
    H, W = latent_hw
    if grid_hw is None:
        h = int(round(math.sqrt(N * H / W)))
        w = N // max(h, 1)
        if h * w != N:
            raise ValueError("cannot infer the attention grid; pass grid_hw")
    else:
        h, w = grid_hw
    m = (a > threshold).float().view(B, 1, h, w)
    return F.interpolate(m, size=(H, W), mode="nearest")


@_keep_dtype
def sag_degrade(x0_u, x_t, mask, blur_sigma=2.0, alpha=1.0, kernel_size=9):
    """SAG degraded input. x0_deg = M blur(x0_u) + (1 - M) x0_u; re-noised with the model's own noise:
    x_t' = x_t + alpha (x0_deg - x0_u) (= alpha x0_deg + sigma n_u; ComfyUI's alpha = 1).
    Returns (x0_deg, x_t')."""
    blurred = gaussian_blur2d(x0_u, blur_sigma, kernel_size)
    x0_deg = mask * blurred + (1 - mask) * x0_u
    return x0_deg, x_t + _scalar(alpha, x0_u) * (x0_deg - x0_u)


def swg_windows(h: int, w: int, crop_frac: float = 0.625, overlap: float = 0.4, multiple: int = 1):
    """Crop boxes (y, x, kh, kw) for Sliding Window Guidance: k = crop_frac of each side (rounded to
    `multiple`), stride k (1 - overlap), last crop flush with the border."""
    def axis(n):
        k = max(multiple, int(round(n * crop_frac / multiple)) * multiple)
        k = min(k, n)
        stride = max(1, int(round(k * (1 - overlap))))
        pos = list(range(0, max(n - k, 0) + 1, stride))
        if pos[-1] != n - k:
            pos.append(n - k)
        return k, pos
    kh, ys = axis(h)
    kw, xs = axis(w)
    return [(y, x, kh, kw) for y in ys for x in xs]


def swg_weak_prediction(x_t: Tensor, fn: Callable[[Tensor], Tensor], crop_frac=0.625, overlap=0.4, multiple=1):
    """Sliding Window Guidance weak branch. Adaloglou, Kaiser, Iagudin & Kollmann (2025), BMVC,
    arXiv 2411.10257. Runs fn (the model at its native crop size, same condition and noise level) on
    overlapping crops, pastes and averages. Returns (d_swg, overlap_mask) for
    d_hat = d_c + s M (d_c - d_swg) (use perturbation_guidance with d_pert = d_c - M (d_c - d_swg)).
    Defaults: 2x2 crops of 5/8 of the side, overlap 0.4. Extra cost: about 1.56 full passes."""
    H, W = x_t.shape[-2:]
    acc = None
    cnt = torch.zeros(1, 1, H, W, device=x_t.device, dtype=torch.float32)
    for y, x, kh, kw in swg_windows(H, W, crop_frac, overlap, multiple):
        out = fn(x_t[..., y:y + kh, x:x + kw]).float()
        if acc is None:
            acc = torch.zeros(*out.shape[:-2], H, W, device=x_t.device, dtype=torch.float32)
        acc[..., y:y + kh, x:x + kw] += out
        cnt[..., y:y + kh, x:x + kw] += 1
    return (acc / cnt.clamp_min(1)).to(x_t.dtype), (cnt > 1).float()


@_keep_dtype
def icg_random_condition(cond, generator=None):
    """ICG weak condition (Sadat, Kansy, Hilliges & Weber 2025, ICLR, arXiv 2407.02687): a Gaussian vector
    with the per-sample mean and std of the real condition embedding. Feed D(x_t; c_hat) as d_weak."""
    dev = generator.device if generator is not None else cond.device
    n = torch.randn(cond.shape, generator=generator, device=dev, dtype=torch.float32).to(cond.device, cond.dtype)
    mu = cond.flatten(1).mean(1).view(-1, *([1] * (cond.ndim - 1)))
    return mu + _std(cond) * n


@_keep_dtype
def tsg_perturb_embedding(t_emb, t=1.0, s=1.0, alpha=0.0, generator=None):
    """Time-step Guidance perturbation (arXiv 2407.02687): t_emb~ = t_emb + s t^alpha n, n ~ N(0, I) on the
    time-embedding vector; the weak branch runs with t_emb~ (optionally only in the first layers)."""
    dev = generator.device if generator is not None else t_emb.device
    n = torch.randn(t_emb.shape, generator=generator, device=dev, dtype=torch.float32).to(t_emb.device, t_emb.dtype)
    return t_emb + s * (float(t) ** alpha) * n


@_keep_dtype
def saliency_blend(g_a, g_b, delta_a, delta_b, k_a=1.0, k_b=1.0, blur_sigma=1.0):
    """MagicFusion saliency-aware noise blending [noise]. Zhao, Zheng, Wang, Lan & Yang (2023), ICCV,
    arXiv 2303.13126. Saliency L_X = blur(|delta_X|) (channel mean), L'_X = softmax over pixels of k_X L_X;
    M = [L'_a > L'_b]; d_hat = M g_a + (1 - M) g_b (g = each branch's guided prediction). The official
    code blurs by down/up-sampling; a Gaussian blur is used here."""
    def sal(d, k):
        s = gaussian_blur2d(d.abs().mean(1, keepdim=True), blur_sigma)
        return torch.softmax(k * s.flatten(1), dim=1).view_as(s)
    m = (sal(delta_a, k_a) > sal(delta_b, k_b)).to(g_a.dtype)
    return m * g_a + (1 - m) * g_b


@_keep_dtype
def topk_direction(delta_aux, weight=1.0, k_ratio=0.05):
    """neutral-prompt AND_TOPK (ljleb, sd-webui-neutral-prompt): weight * delta_aux restricted to its
    top k_ratio elements by magnitude (per sample). Add the result to a guided prediction."""
    flat = delta_aux.abs().flatten(1)
    thr = _quantile_rows(flat, 1 - k_ratio).unsqueeze(1)
    m = (flat >= thr).view_as(delta_aux).to(delta_aux.dtype)
    return weight * m * delta_aux


# =============================================================================
# 5. Negative and compositional conditions (family D)
# =============================================================================


@_keep_dtype
def composable_and(d_u, d_cs, ws):
    """Composable Diffusion conjunction (AND) [any]. Liu, Li, Du, Torralba & Tenenbaum (2022), ECCV,
    arXiv 2206.01714. d_hat = d_u + sum_i w_i (d_{c_i} - d_u). Keep sum_i w_i about 5-8.
    Also Factored CFG (Xia et al. 2025, arXiv 2506.14399) with per-group partial conditions.
    Extra cost: +1 NFE per extra concept."""
    out = d_u
    for d, wi in zip(d_cs, ws):
        out = out + _scalar(wi, d_u) * (d - d_u)
    return out


@_keep_dtype
def composable_not(d_c, d_u, d_n, w=7.5):
    """Composable Diffusion negation (NOT) [any]. arXiv 2206.01714. d_hat = d_u + w (d_c - d_n): the base is
    the NULL, so it differs from negative prompting by one unit of (d_n - d_u). Also TraSCE's modified
    negative prompting (arXiv 2412.07658). Extra cost: +1 NFE (c, n, u)."""
    return d_u + _scalar(w, d_c) * (d_c - d_n)


@_keep_dtype
def signed_guidance(d_c, d_u, d_n, w=7.5, w_neg=20.0):
    """Positive and negative directions from the null [any]: d_hat = d_u + w (d_c - d_u) - w_neg (d_n - d_u).
    A1111 AND with negative weights; VL-DNP (Chang, Kim & Choi 2025, arXiv 2510.26052; w = 1 + w_pos,
    w_neg 20 best there). w_neg = w is composable NOT. Extra cost: +1 NFE."""
    return d_u + _scalar(w, d_c) * (d_c - d_u) - w_neg * (d_n - d_u)


@_keep_dtype
def perp_neg(d_c, d_u, d_n, w=7.5, neg_scale=1.0, batch_scalar=False):
    """Perp-Neg, perpendicular negative guidance [any]. Armandpour, Sadeghian, Zheng, Sadeghian & Zhou
    (2023), arXiv 2304.04968.

    delta_c = d_c - d_u; for each negative j: delta_j = d_nj - d_u, a_j = <delta_j, delta_c>/||delta_c||^2,
    perp_j = delta_j - a_j delta_c; d_hat = d_u + w (delta_c - sum_j beta_j perp_j).
    d_n may be a tensor or a list; neg_scale a float or a list (beta_j = -w_neg_j of the official code).
    batch_scalar=True reproduces ComfyUI's single projection over the whole batch tensor.
    Defaults: beta 1.0 (ComfyUI), 1.5 for a single negative in the paper, w 7.5. Extra cost: +1 NFE.
    """
    w = _scalar(w, d_c)
    dns = list(d_n) if isinstance(d_n, (list, tuple)) else [d_n]
    scales = list(neg_scale) if isinstance(neg_scale, (list, tuple)) else [neg_scale] * len(dns)
    pos = d_c - d_u
    total = torch.zeros_like(pos)
    for dn, b in zip(dns, scales):
        neg = dn - d_u
        if batch_scalar:
            coef = (neg * pos).sum() / (pos.norm() ** 2).clamp_min(_EPS)
        else:
            coef = _dot(neg, pos) / _sqnorm(pos).clamp_min(_EPS)
        total = total + b * (neg - coef * pos)
    return d_u + w * (pos - total)


@_keep_dtype
def contrastive_cfg(d_c, d_u, d_n, w=7.5, w_neg=None, tau=None):
    """ContrastiveCFG (CCFG) [noise: eps or velocity]. Chang, Lee, Chung & Ye (2024/2026), ICML,
    arXiv 2411.17077.

    r+^2 = ||d_c - d_u||^2, r-^2 = ||d_n - d_u||^2; lambda+ = 2/(1 + exp(-tau r+^2)) in [1, 2);
    lambda- = 2 exp(-tau r-^2)/(1 + exp(-tau r-^2)) in (0, 1];
    d_hat = d_u + w lambda+ (d_c - d_u) - w_neg lambda- (d_n - d_u).
    tau None calibrates per sample so that tau r+^2 = 1 (the entry's advice: tau r^2 about 1 at
    mid-schedule; the paper's constant was not recovered). w_neg None uses w. Extra cost: +1 NFE."""
    w = _scalar(w, d_c)
    wn = w if w_neg is None else _scalar(w_neg, d_c)
    rp = _sqnorm(d_c - d_u)
    rn = _sqnorm(d_n - d_u)
    tt = 1.0 / rp.clamp_min(_EPS) if tau is None else tau
    lp = 2 / (1 + torch.exp(-tt * rp))
    ln = 2 * torch.exp(-tt * rn) / (1 + torch.exp(-tt * rn))
    return d_u + w * lp * (d_c - d_u) - wn * ln * (d_n - d_u)


@_keep_dtype
def contrastive_guidance(d_base, d_pos, d_neg, lam=1.0):
    """Contrastive Guidance with minimal-pair prompts [noise]. Wu & De la Torre (2024), arXiv 2402.13490.
    d_hat = d_base + lam (d_{y+} - d_{y-}); d_base is usually the CFG prediction. lam 1 (experts),
    6-10 with SDEdit / CycleDiffusion, -8..8 as a slider. Extra cost: +1 or +2 NFE."""
    return d_base + lam * (d_pos - d_neg)


@_keep_dtype
def nested_cfg(preds, ws):
    """Ordered multi-condition CFG [any]: d_hat = d_0 + sum_k w_k (d_{<=k} - d_{<=k-1}), preds ordered from
    least to most conditioned. InstructPix2Pix (Brooks, Holynski & Efros 2023, CVPR, arXiv 2211.09800) is
    preds = [d(u_I, u_T), d(c_I, u_T), d(c_I, c_T)], ws = [s_I, s_T]."""
    out = preds[0]
    for k in range(1, len(preds)):
        out = out + _scalar(ws[k - 1], preds[0]) * (preds[k] - preds[k - 1])
    return out


@_keep_dtype
def ip2p_cfg(d_uu, d_iu, d_it, s_i=1.5, s_t=7.5):
    """InstructPix2Pix two-scale CFG [any]. arXiv 2211.09800.
    d_hat = d(0,0) + s_I [d(c_I,0) - d(0,0)] + s_T [d(c_I,c_T) - d(c_I,0)]. Defaults s_T 7.5, s_I 1.5.
    Extra cost: +1 NFE (3 passes)."""
    return nested_cfg([d_uu, d_iu, d_it], [s_i, s_t])


@_keep_dtype
def dual_cfg_comfy(d_cond1, d_cond2, d_neg, cfg_conds=8.0, cfg_cond2_negative=8.0, style="regular"):
    """ComfyUI DualCFGGuider [any]. 'regular': d_neg + s2 (d_cond2 - d_neg) + s1 (d_cond1 - d_cond2);
    'nested': d_neg + s2 ([d_cond2 + s1 (d_cond1 - d_cond2)] - d_neg). s1 = cfg_conds, s2 = cfg_cond2_negative."""
    s1, s2 = cfg_conds, cfg_cond2_negative
    if style == "nested":
        return d_neg + s2 * ((d_cond2 + s1 * (d_cond1 - d_cond2)) - d_neg)
    return d_neg + s2 * (d_cond2 - d_neg) + s1 * (d_cond1 - d_cond2)


class SEGAState:
    def __init__(self):
        self.nu = None

    def reset(self):
        self.nu = None


def _as_list(v, n):
    return list(v) if isinstance(v, (list, tuple)) else [v] * n


@_keep_dtype
def sega(d_c, d_u, d_edits, w=7.5, state=None, step=0, edit_scale=5.0, reverse=False, threshold=0.9,
         warmup=10, cooldown=None, momentum_scale=0.1, momentum_beta=0.4, concept_weights=1.0):
    """SEGA semantic guidance [noise: eps]. Brack, Friedrich, Hintersdorf, Struppek, Schramowski & Kersting
    (2023), NeurIPS, arXiv 2301.12247.

    Per edit concept e_i: psi_i = sgn_i (d_ei - d_u) (sgn = -1 for removal); mask per (sample, channel):
    |psi_i| >= its threshold-quantile over spatial positions; gamma_i = g_i s_e,i mask psi_i, active for
    warmup_i <= step < cooldown_i. Momentum: gamma_bar = gamma + s_m nu (once any concept is active);
    nu <- beta_m nu + (1 - beta_m) gamma_all (every step, warm-up included). d_hat = d_u + w delta + gamma_bar.
    Defaults (diffusers): s_e 5, threshold 0.9, warmup 10, s_m 0.1, beta_m 0.4. Extra cost: +1 NFE per
    concept. LEDITS++ (arXiv 2311.16711) adds cross-attention masks and an inversion (not included)."""
    w = _scalar(w, d_c)
    edits = list(d_edits) if isinstance(d_edits, (list, tuple)) else [d_edits]
    n = len(edits)
    scales, revs = _as_list(edit_scale, n), _as_list(reverse, n)
    thrs, wus, cds = _as_list(threshold, n), _as_list(warmup, n), _as_list(cooldown, n)
    cws = _as_list(concept_weights, n)
    gamma_all = torch.zeros_like(d_c)
    gamma_act = torch.zeros_like(d_c)
    any_active = False
    for i, de in enumerate(edits):
        psi = (-1.0 if revs[i] else 1.0) * (de - d_u)
        B, C = psi.shape[:2]
        flat = psi.abs().reshape(B * C, -1)
        eta = _quantile_rows(flat, thrs[i]).view(B, C, *([1] * (psi.ndim - 2)))
        g = cws[i] * scales[i] * (psi.abs() >= eta).to(psi.dtype) * psi
        gamma_all = gamma_all + g
        active = step >= wus[i] and (cds[i] is None or step < cds[i])
        if active:
            gamma_act = gamma_act + g
            any_active = True
    out = d_u + w * (d_c - d_u)
    if state is not None:
        nu = state.nu if (state.nu is not None and state.nu.shape == d_c.shape) else torch.zeros_like(d_c)
        if any_active:
            gamma_act = gamma_act + momentum_scale * nu
        state.nu = momentum_beta * nu + (1 - momentum_beta) * gamma_all
    return out + gamma_act if any_active else out


class SLDState:
    def __init__(self):
        self.nu = None

    def reset(self):
        self.nu = None


@_keep_dtype
def sld(d_c, d_u, d_n, w=7.5, state=None, step=0, warmup=10, s_s=1000.0, lam=0.01, s_m=0.3, beta_m=0.4):
    """Safe Latent Diffusion safety guidance [noise: eps]. Schramowski, Brack, Deiseroth & Kersting (2023),
    CVPR, arXiv 2211.05105. d_n = the safety-concept prediction D(x_t; S).

    mu = clamp(s_S |d_c - d_S|, max=1) where (d_c - d_S) < lam, else 0; gamma = mu (d_S - d_u) + s_m nu;
    nu <- beta_m nu + (1 - beta_m) gamma (every step); d_hat = d_u + w (d_c - d_u - [step >= warmup] gamma).
    Defaults: the paper's MEDIUM configuration (warmup 10, s_S 1000, lam 0.01, s_m 0.3, beta_m 0.4).
    STRONG = (7, 2000, 0.025, 0.5, 0.7), MAX = (0, 5000, 1.0, 0.5, 0.7). Extra cost: +1 NFE."""
    w = _scalar(w, d_c)
    diff = d_c - d_n
    mu = torch.where(diff < lam, torch.clamp(s_s * diff.abs(), max=1.0), torch.zeros_like(diff))
    gamma = mu * (d_n - d_u)
    if state is not None:
        nu = state.nu if (state.nu is not None and state.nu.shape == d_c.shape) else torch.zeros_like(d_c)
        gamma = gamma + s_m * nu
        state.nu = beta_m * nu + (1 - beta_m) * gamma
    g = d_c - d_u
    if step >= warmup:
        g = g - gamma
    return d_u + w * g


class DNGState:
    """Posterior p(n | x_k) for Dynamic Negative Guidance, updated along a stochastic trajectory."""

    def __init__(self, lambda0=5.0, prior=0.01, tau=0.2, delta=2e-4, p_min=1e-6, p_max=0.8):
        self.lambda0, self.prior, self.tau, self.delta = lambda0, prior, tau, delta
        self.p_min, self.p_max = p_min, p_max
        self.log_p = None

    def reset(self):
        self.log_p = None

    def scale(self, ref):
        if self.log_p is None or self.log_p.shape[0] != ref.shape[0]:
            self.log_p = torch.full((ref.shape[0],), math.log(self.prior), device=ref.device)
        p = self.log_p.exp().clamp(self.p_min, self.p_max)
        return _bview(self.lambda0 * p / (1 - p), ref).to(ref.dtype)

    def update(self, x_k, mu_b, mu_n, s_k):
        """After the ancestral step x_{k+1} -> x_k with noise std s_k; mu_b, mu_n are the posterior means
        computed with the base and the negative predictions."""
        s2 = float(s_k) ** 2
        if s2 <= 0:
            return
        if self.log_p is None:
            self.log_p = torch.full((x_k.shape[0],), math.log(self.prior), device=x_k.device)
        db = (x_k - mu_b).float().flatten(1).pow(2).sum(1)
        dn = (x_k - mu_n).float().flatten(1).pow(2).sum(1)
        self.log_p = self.log_p + self.tau / (2 * s2) * (db - dn) + self.delta / (2 * s2)


@_keep_dtype
def dng(d_c, d_u, w=1.0, d_n=None, state=None, base="cond"):
    """Dynamic Negative Guidance [noise: eps, DDPM]. Koulischer, Deleu, Raya, Demeester & Ambrogioni (2025),
    ICLR, arXiv 2410.14398 (medium).
    b = d_c ('cond', text-to-image), d_u ('uncond') or plain CFG ('cfg'); d_hat = b - lambda_k (d_n - b),
    lambda_k = lambda0 p_k/(1 - p_k), p_k = the state's posterior (call state.update after each step).
    Defaults (state): lambda0 5, prior 0.01, tau 0.2, delta 2e-4, p_max 0.8. Extra cost: +1 NFE."""
    if d_n is None:
        raise ValueError("dng needs d_n")
    b = d_c if base == "cond" else (d_u if base == "uncond" else cfg(d_c, d_u, w))
    lam = state.scale(d_c) if state is not None else 0.0
    return b - lam * (d_n - b)


class CRGState:
    def __init__(self):
        self.running = None

    def reset(self):
        self.running = None


@_keep_dtype
def concept_removal_guidance(d_c, d_u, d_n, state=None, alphabar=0.5, alpha_step=0.99, beta_step=0.01,
                             num_steps=50, K=1.0, tau=0.0, lambda0=2.0):
    """Concept Removal Guidance (CRG) [noise: eps, DDPM]. Choi, Oh, Choi, Seo & Kim (2026), ICML,
    arXiv 2606.29801 (medium).
    Delta = d_n - d_u; bracket_k = ||d_c - d_u||^2 - ||d_c - d_n||^2 (> 0 when the prediction is nearer the
    negative). CP = (K/2) bracket_k + (1/T) sum_{k' earlier} kappa_k' bracket_k', kappa = beta T /
    (2 alpha (1 - alphabar)). omega = (CP - tau)/(K ||Delta||^2) if CP > tau else 0;
    d_hat = d_c - lambda0 omega Delta. alphabar, alpha_step, beta_step describe the current DDPM step.
    tau has no default table in the paper (tune per task). Extra cost: +1 NFE."""
    Dl = d_n - d_u
    bracket = (d_c - d_u).flatten(1).pow(2).sum(1) - (d_c - d_n).flatten(1).pow(2).sum(1)
    run = state.running if (state is not None and state.running is not None) else torch.zeros_like(bracket)
    cp = K / 2 * bracket + run / num_steps
    omega = torch.where(cp > tau, (cp - tau) / (K * Dl.flatten(1).pow(2).sum(1)).clamp_min(_EPS), torch.zeros_like(cp))
    if state is not None:
        kappa = beta_step * num_steps / (2 * alpha_step * max(1 - alphabar, _EPS))
        state.running = run + kappa * bracket
    return d_c - lambda0 * _bview(omega, d_c) * Dl


def trasce_loss(d_c, d_n, sigma_l=1.0):
    """TraSCE localized loss (arXiv 2412.07658): L = -exp(-||d_c - d_n||^2 / (2 sigma_L^2)) per sample.
    The caller differentiates it w.r.t. x_t (autograd through the network) and steps
    x_t <- x_t - lambda_L grad L. Defaults: nudity lambda_L 1.5, sigma_L 1."""
    return -torch.exp(-(d_c - d_n).float().flatten(1).pow(2).sum(1) / (2 * sigma_l ** 2))


def fkc_log_weight_increment(d_c, d_u, sigma, sigma_next, w=7.5):
    """Feynman-Kac corrector weight increment for CFG particles [x0, EDM sigma]. Skreta et al. (2025),
    ICML, arXiv 2503.02819: dlogW = w (w - 1)(sigma^2 - sigma'^2) ||D_c - D_u||^2 / (2 sigma^4) per sample.
    Resample the particles with systematic_resample inside the active interval."""
    s, sn = _float(sigma), _float(sigma_next)
    return w * (w - 1) * (s * s - sn * sn) * (d_c - d_u).float().flatten(1).pow(2).sum(1) / (2 * s ** 4)


def systematic_resample(log_w: Tensor, generator=None) -> Tensor:
    """Systematic resampling indices for K particles from unnormalized log-weights (K,)."""
    K = log_w.numel()
    p = torch.softmax(log_w.float().reshape(-1), 0)
    u0 = torch.rand((), generator=generator).item()
    pos = (torch.arange(K, dtype=torch.float32) + u0) / K
    return torch.searchsorted(torch.cumsum(p, 0).cpu(), pos).clamp(max=K - 1)


@_keep_dtype
def superdiff_mix(d_a, d_b, d_u, w=7.5, kappa=0.5):
    """SuperDiff mixed prediction [eps, k-diffusion sigma]. Skreta, Atanackovic, Bose, Tong & Neklyudov
    (2025), ICLR, arXiv 2412.17762: d_hat = d_u + w [(d_b - d_u) + kappa (d_a - d_b)]."""
    return d_u + w * ((d_b - d_u) + _scalar(kappa, d_a) * (d_a - d_b))


def superdiff_kappa_or(logdens_a, logdens_b, temperature=1.0, logp=0.0):
    """SuperDiff OR: kappa = softmax([T (l_a + logp), T l_b])[0] per sample."""
    return torch.softmax(torch.stack([temperature * (logdens_a + logp), temperature * logdens_b], 0), 0)[0]


def superdiff_kappa_and(d_a, d_b, d_u, w, ds, dx_ind, sigma, lift=0.0):
    """SuperDiff AND closed-form kappa per sample (arXiv 2412.17762 reference script):
    kappa = [sum(|ds| (d_b - d_a)(d_b + d_a)) - sum(dx_ind (d_a - d_b)) + sigma lift / N] /
            [2 ds w sum((d_a - d_b)^2)], dx_ind = 2 ds (d_u + w (d_b - d_u)) + the step's noise."""
    a, b = d_a.float().flatten(1), d_b.float().flatten(1)
    N = a.shape[1]
    num = (abs(ds) * (b - a) * (b + a)).sum(1) - (dx_ind.float().flatten(1) * (a - b)).sum(1) + sigma * lift / N
    den = 2 * ds * w * ((a - b) ** 2).sum(1)
    return num / torch.where(den.abs() > _EPS, den, torch.full_like(den, _EPS))


def superdiff_update_logdens(logdens, d, dx, ds, sigma, mode="and"):
    """SuperDiff Ito log-density tracking (summed over latent elements): AND l += sum(-|ds|/sigma d^2 - dx d/sigma);
    OR l -= sum(d (dx + ds d))/sigma."""
    dd, xx = d.float().flatten(1), dx.float().flatten(1)
    if mode == "and":
        return logdens + (-abs(ds) / sigma * dd ** 2 - xx * dd / sigma).sum(1)
    return logdens - (dd * (xx + ds * dd)).sum(1) / sigma


# =============================================================================
# 6. Solver coupling (family E): CFG++, look-ahead correctors, momentum, re-noising
# =============================================================================


@_keep_dtype
def ddim_step(x_t, x0, alpha=1.0, sigma=1.0, alpha_next=1.0, sigma_next=0.0):
    """Deterministic DDIM / Euler step on any Gaussian path: n = (x_t - a x0)/s; x' = a' x0 + s' n."""
    a, s = _scalar(alpha, x0), _scalar(sigma, x0)
    an, sn = _scalar(alpha_next, x0), _scalar(sigma_next, x0)
    return an * x0 + sn * (x_t - a * x0) / s


@_keep_dtype
def cfgpp_step(x_t, d_c, d_u, lam=0.6, alpha=1.0, sigma=1.0, alpha_next=1.0, sigma_next=0.0):
    """CFG++ deterministic step [x0]. Chung, Kim, Park, Nam & Ye (2024), ICLR 2025, arXiv 2406.08070.

    x0_lam = x0_u + lam (x0_c - x0_u) (interpolation, lam in [0, 1]); n_u = (x_t - a x0_u)/s (the
    UNCONDITIONAL noise); x' = a' x0_lam + s' n_u. Plain CFG would re-noise with the guided noise.
    VE: alpha = 1; RF: alpha = 1 - t. Equivalent to a DDIM step with w_eff = cfgpp_equivalent_w(...).
    Defaults: lam 0.6 (about w 7.5 on SD1.5; 0.2 ~ 2, 0.4 ~ 5, 0.8 ~ 9, 1.0 ~ 12.5). Extra cost: none.
    """
    a, s = _scalar(alpha, d_c), _scalar(sigma, d_c)
    an, sn = _scalar(alpha_next, d_c), _scalar(sigma_next, d_c)
    x0_lam = d_u + lam * (d_c - d_u)
    n_u = (x_t - a * d_u) / s
    return an * x0_lam + sn * n_u


def cfgpp_equivalent_w(lam, alpha, sigma, alpha_next, sigma_next):
    """The CFG weight whose DDIM step equals a CFG++ step: w_eff = lam a' s / (a' s - s' a).
    VE: lam sigma/(sigma - sigma'); RF: lam t (1 - t')/(t - t')."""
    return lam * alpha_next * sigma / (alpha_next * sigma - sigma_next * alpha)


def ancestral_split(rho, rho_next, eta=1.0):
    """k-diffusion ancestral split of a VE step rho -> rho_next: (rho_down, rho_up)."""
    if rho_next <= 0:
        return 0.0, 0.0
    up = min(rho_next, eta * math.sqrt(max(rho_next ** 2 * (rho ** 2 - rho_next ** 2) / rho ** 2, 0.0)))
    down = math.sqrt(max(rho_next ** 2 - up ** 2, 0.0))
    return down, up


@_keep_dtype
def cfgpp_ancestral_step(x_t, d_c, d_u, lam=0.6, alpha=1.0, sigma=1.0, alpha_next=1.0, sigma_next=0.0,
                         eta=1.0, s_noise=1.0, noise=None):
    """Ancestral CFG++ (ComfyUI euler_ancestral_cfg_pp form) [x0]. arXiv 2406.08070.
    With rho = s/a: (rho_down, rho_up) = ancestral split of rho -> rho'; x' = a' (x0_lam + rho_down n_u
    + s_noise rho_up z). eta = 0 gives cfgpp_step. Pass `noise` for determinism."""
    a, s, an, sn = _float(alpha), _float(sigma), _float(alpha_next), _float(sigma_next)
    x0_lam = d_u + lam * (d_c - d_u)
    n_u = (x_t - a * d_u) / s
    rho, rho_n = s / a, (sn / an if an > 0 else float("inf"))
    down, up = ancestral_split(rho, rho_n, eta)
    z = torch.randn_like(x_t) if (noise is None and up > 0) else (noise if noise is not None else 0.0)
    return an * (x0_lam + down * n_u + s_noise * up * z)


def rectified_cfgpp_step(x_t, t, t_next, f_c_fn, f_u_fn, alpha=4.5, predictor="half", noise_sigma=0.0,
                         query_time="mid", generator=None):
    """Rectified-CFG++ predictor-corrector step [noise: flow velocity]. Saini, Gupta & Bovik (2025),
    NeurIPS, arXiv 2510.07631.

    RF, t = 1 noise, dt = t - t_next. x_mid = x_t - frac dt f_c(x_t, t) (frac 1/2 paper, 1 released code);
    optional x_mid += noise_sigma z (code: 0.005); f_hat = f_c(x_t, t) + alpha(t) [f_c(x_mid, t_q) -
    f_u(x_mid, t_q)], t_q = t - frac dt ('mid') or t ('current', what the released code effectively
    queries); x' = x_t - dt f_hat. alpha may be a float or a callable of t (paper: lam_max (1 - t)^gamma).
    x_mid = x_t reduces it to CFG with w = alpha + 1. f_*_fn(x, t) -> velocity.
    Defaults: alpha 4.5 (code), frac 1/2. Extra cost: 3 NFE per step (1.5-2x CFG).
    """
    dt = t - t_next
    fc = f_c_fn(x_t, t)
    frac = 0.5 if predictor == "half" else 1.0
    x_mid = x_t - frac * dt * fc
    if noise_sigma > 0:
        dev = generator.device if generator is not None else x_t.device
        x_mid = x_mid + noise_sigma * torch.randn(x_t.shape, generator=generator, device=dev).to(x_t)
    tq = t - frac * dt if query_time == "mid" else t
    a = alpha(t) if callable(alpha) else alpha
    f_hat = fc + a * (f_c_fn(x_mid, tq) - f_u_fn(x_mid, tq))
    return x_t - dt * f_hat


def cfg_mp_step(x_t, t, t_next, f_c_fn, f_u_fn, w=3.5, iterations=2, a=None, anderson=True, anderson_beta=1.0):
    """CFG-MP / CFG-MP+ manifold projection [noise: flow velocity]. Cai, Liu, Su & Wang (2026), ICML,
    arXiv 2601.21892.

    RF (t = 1 noise). y = x_t - dt [f_u + w (f_c - f_u)](x_t, t); then K iterations of
    G(y) = z - a f_c(z, t'), z = y + a f_u(y, t') (a = dt/2), whose fixed point has f_c(z) = f_u(y).
    CFG-MP+ uses Anderson acceleration AA(1, beta): g = -<r_prev, r - r_prev>/||r - r_prev||^2,
    y <- (1 - g)(y_prev + beta r_prev) + g (y + beta r). Defaults: K 2, a dt/2, AA on. The released code
    projects only while sigma > 0.6. Extra cost: 2 NFE per iteration (2 + 2K per step).
    """
    dt = t - t_next
    a = dt / 2 if a is None else a
    fu, fc = f_u_fn(x_t, t), f_c_fn(x_t, t)
    y = x_t - dt * (fu + w * (fc - fu))

    def G(v):
        z = v + a * f_u_fn(v, t_next)
        return z - a * f_c_fn(z, t_next)

    if iterations <= 0:
        return y
    if not anderson:
        for _ in range(iterations):
            y = G(y)
        return y
    y_prev = y
    g_prev = G(y_prev)
    r_prev = g_prev - y_prev
    y = y_prev + anderson_beta * r_prev
    for _ in range(1, iterations):
        g_cur = G(y)
        r = g_cur - y
        dr = r - r_prev
        gam = -_dot(r_prev, dr) / _sqnorm(dr).clamp_min(_EPS)
        y_new = (1 - gam) * (y_prev + anderson_beta * r_prev) + gam * (y + anderson_beta * r)
        y_prev, r_prev, y = y, r, y_new
    return y


@_keep_dtype
def cfg_euler_step_ve(y, d_c, d_u, w=7.5, sigma=1.0, sigma_next=0.0):
    """Plain CFG DDIM/Euler step in VE variables (y = x0 + sigma eps):
    y' = D_c + r (y - D_c) + (w - 1)(r - 1)(D_u - D_c), r = sigma'/sigma."""
    r = _scalar(sigma_next, d_c) / _scalar(sigma, d_c)
    w = _scalar(w, d_c)
    return d_c + r * (y - d_c) + (w - 1) * (r - 1) * (d_u - d_c)


@_keep_dtype
def terminal_repair_step(y, d_c, d_u, w=7.5, sigma=1.0, sigma_next=0.0):
    """Terminal-fitted repair of the DDIM guidance coefficient [x0, VE]. Zhang (2026), arXiv 2607.07665.
    y' = D_c + r (y - D_c) + (r^w - r)(D_u - D_c), r = sigma'/sigma (the paper's g = w). Agrees with plain
    CFG to first order in log step; differs for large steps and large w. RF: step y = x_t/(1 - t) with
    sigma = t/(1 - t), then x_t' = (1 - t') y'. Needs the next sigma (a sampler-level change)."""
    r = _scalar(sigma_next, d_c) / _scalar(sigma, d_c)
    w = _scalar(w, d_c)
    return d_c + r * (y - d_c) + (r ** w - r) * (d_u - d_c)


@_keep_dtype
def pcg_corrector(x, score_c, score_u, g_prime=2.0, step_size=0.01, noise=None):
    """Predictor-Corrector Guidance, Langevin corrector (Bradley & Nakkiran 2024, arXiv 2408.09000) [score].
    x <- x + (eps/2) [(1 - g') s_u + g' s_c] + sqrt(eps) z. The predictor is a conditional-only DDIM step
    (use ddim_step with the conditional x0). CFG-DDPM at weight w corresponds to g' = 2w - 1.
    Step size: DDPM beta_t, or sigma_t^2 - sigma_t'^2 in EDM terms. Also the ULA corrector of
    Reduce-Reuse-Recycle (arXiv 2302.11552). Extra cost: 2 NFE per corrector step."""
    z = torch.randn_like(x) if noise is None else noise
    return x + step_size / 2 * ((1 - g_prime) * score_u + g_prime * score_c) + math.sqrt(step_size) * z


class HeavyBallState:
    def __init__(self):
        self.v = None

    def reset(self):
        self.v = None


def heavy_ball(increment, state, beta=0.8):
    """Heavy-ball momentum on a solver increment (HB, Wizadwongsa et al. 2023, ICLR 2024, arXiv 2307.11118):
    v <- (1 - beta) v + beta F; the solver then uses x' = x + delta v. beta = 1 is the base solver."""
    if state.v is None or state.v.shape != increment.shape:
        state.v = increment
    else:
        state.v = (1 - beta) * state.v + beta * increment
    return state.v


@_keep_dtype
def dsg_step(mu, g, sigma_t=1.0, guidance_rate=0.1, noise=None):
    """DSG spherical Gaussian constraint (Yang et al. 2024, ICML, arXiv 2402.03201) [stochastic step].
    r = sqrt(n) sigma_t; d* = -r g/||g||; d_sample = sigma_t z; d_m = d_sample + g_r (d* - d_sample);
    x' = mu + r d_m/||d_m||. CFG adaptation (not in the paper): g = -(mu_cfg - mu_cond)."""
    z = torch.randn_like(mu) if noise is None else noise
    n = mu[0].numel()
    s = _scalar(sigma_t, mu)
    r = math.sqrt(n) * s
    d_star = -r * g / _norm(g).clamp_min(_EPS)
    d_sample = s * z
    d_m = d_sample + guidance_rate * (d_star - d_sample)
    return mu + r * d_m / _norm(d_m).clamp_min(_EPS)


@_keep_dtype
def particle_repulsion(x, strength=1.0, bandwidth=None):
    """Particle guidance repulsion on latents (Corso et al. 2023, ICLR 2024, arXiv 2310.13102):
    grad_i log Phi = strength sum_j (2/h) k_ij (x_i - x_j), k = exp(-||x_i - x_j||^2 / h), h = median of the
    pairwise squared distances. Add to the score (eps_i -= sigma_t * grad_i). Batch coupled by design."""
    B = x.shape[0]
    flat = x.flatten(1)
    d2 = torch.cdist(flat, flat) ** 2
    if bandwidth is None:
        off = d2[~torch.eye(B, dtype=torch.bool, device=x.device)]
        h = off.median().clamp_min(_EPS) if off.numel() > 0 else torch.tensor(1.0, device=x.device)
    else:
        h = torch.tensor(float(bandwidth), device=x.device)
    k = torch.exp(-d2 / h)
    grad = (2 / h) * (k.sum(1, keepdim=True) * flat - k @ flat)
    return strength * grad.view_as(x)


@_keep_dtype
def renoise(x0, noise, alpha=1.0, sigma=1.0):
    """x = alpha x0 + sigma n (CFGiG's noise-and-redenoise, restart sampling, self-recurrence)."""
    return _scalar(alpha, x0) * x0 + _scalar(sigma, x0) * noise


@_keep_dtype
def restart_renoise_flow(x, t_min, t_max, noise):
    """Restart jump for rectified flow (derived for RF from Xu et al. 2023, arXiv 2306.14878):
    x_tmax = ((1 - t_max)/(1 - t_min)) x_tmin + sqrt(t_max^2 - ((1 - t_max) t_min/(1 - t_min))^2) z."""
    c = (1 - t_max) / (1 - t_min)
    return c * x + math.sqrt(max(t_max ** 2 - (c * t_min) ** 2, 0.0)) * noise


def zigzag_step(x, denoise_fn, invert_fn):
    """Z-Sampling / W2SD reflection (Bai et al. 2024, ICLR 2025, arXiv 2412.10891; Bai, Sugiyama & Xie 2025,
    arXiv 2502.00473): x' = denoise(x) (strong: high guidance or the strong model), x~ = invert(x')
    (weak: low guidance, e.g. w = 1, or the weak model), return denoise(x~).
    Extra cost: +1 inversion and +1 guided denoise per reflected step."""
    return denoise_fn(invert_fn(denoise_fn(x)))


@_keep_dtype
def classifier_guidance(d, grad_log_p, s=1.0, alpha=1.0, sigma=1.0, kind="eps"):
    """Classifier guidance (Dhariwal & Nichol 2021, NeurIPS, arXiv 2105.05233) given grad_x log p(c | x_t):
    eps: d - s sigma grad; x0: d + s (sigma^2/alpha) grad; flow (RF): d - s (t/(1 - t)) grad with alpha = 1 - t.
    Implicit-classifier link: CFG at w equals s = w - 1 applied to d_c."""
    a, sg = _scalar(alpha, d), _scalar(sigma, d)
    if kind == "eps":
        return d - s * sg * grad_log_p
    if kind == "x0":
        return d + s * (sg * sg / a) * grad_log_p
    if kind == "flow":
        return d - s * (sg / a) * grad_log_p
    raise ValueError("kind must be 'eps', 'x0' or 'flow'")


@_keep_dtype
def temporal_score_rescale(x0, x_t, alpha=1.0, sigma=1.0, k=0.95, tsr_sigma=1.0):
    """Temporal Score Rescaling [x0]. Xu, Wu, Park, Zhou & Tulsiani (2025), ICML 2026, arXiv 2510.01184.
    SNR eta = alpha^2/sigma^2; r = (eta s^2 + 1)/(eta s^2/k + 1); eps' = r eps, i.e.
    x0' = x_t/alpha + r (x0 - x_t/alpha) (ComfyUI TSR node: torch.lerp(x/alpha, denoised, r)).
    k = 1 is off. Where alpha = 0 (pure noise, SNR 0) or sigma = 0 the estimate is returned unchanged,
    as the ComfyUI node does. Defaults: k 0.95, s 1.0 (ComfyUI; paper SD3/FLUX k 0.93, s 3)."""
    a, sg = _tensor(alpha, x0), _tensor(sigma, x0)
    if k == 1:
        return x0
    ok = (a > 1e-6) & (sg > 0)
    a_safe = torch.where(a > 1e-6, a, torch.ones_like(a))
    snr = a_safe * a_safe / (sg * sg).clamp_min(_EPS)
    r = (snr * tsr_sigma ** 2 + 1) / (snr * tsr_sigma ** 2 / k + 1)
    base = x_t / a_safe
    return torch.where(ok, base + r * (x0 - base), x0)


@_keep_dtype
def cfg_tsr(d_c, d_u, w=7.5, x_t=None, alpha=1.0, sigma=1.0, k=0.95, tsr_sigma=1.0):
    """Plain CFG in x0 followed by Temporal Score Rescaling [x0]. arXiv 2510.01184."""
    if x_t is None:
        raise ValueError("cfg_tsr needs x_t")
    return temporal_score_rescale(cfg(d_c, d_u, w), x_t, alpha, sigma, k, tsr_sigma)


@_keep_dtype
def epsilon_scaling(x0, x_t, factor=1.005, alpha=None):
    """Epsilon Scaling, exposure-bias correction [x0]. Ning, Li, Su, Salah & Ertugrul (2024), ICLR,
    arXiv 2308.15321. eps' = eps / lambda.
    alpha None: the ComfyUI node's form x0' = x_t - (x_t - x0)/lambda (exact for VE; for flow models it
    scales the velocity residual t f). alpha given: x0' = (x_t - (x_t - alpha x0)/lambda)/alpha, left
    unchanged where alpha = 0."""
    if alpha is None:
        return x_t - (x_t - x0) / factor
    a = _tensor(alpha, x0)
    a_safe = torch.where(a > 1e-6, a, torch.ones_like(a))
    return torch.where(a > 1e-6, (x_t - (x_t - a_safe * x0) / factor) / a_safe, x0)


@_keep_dtype
def cfg_epsilon_scaling(d_c, d_u, w=7.5, x_t=None, factor=1.005):
    """Plain CFG in x0 followed by Epsilon Scaling in the ComfyUI form [x0]. Default factor 1.005."""
    if x_t is None:
        raise ValueError("cfg_epsilon_scaling needs x_t")
    return epsilon_scaling(cfg(d_c, d_u, w), x_t, factor)


class HiGSState:
    def __init__(self):
        self.h = None
        self.wsum = 0.0

    def reset(self):
        self.h = None
        self.wsum = 0.0


def _dct_highpass(x: Tensor, cutoff: float, sharpness: float) -> Tensor:
    H, W = x.shape[-2:]
    fy = torch.arange(H, device=x.device, dtype=x.dtype) / H
    fx = torch.arange(W, device=x.device, dtype=x.dtype) / W
    R = torch.sqrt(fy.view(-1, 1) ** 2 + fx.view(1, -1) ** 2)
    Hm = torch.sigmoid(sharpness * (R - cutoff))
    return idct2(dct2(x) * Hm)


@_keep_dtype
def higs(d_c, d_u, w=7.5, state=None, t=1.0, w_h=2.0, alpha=0.75, t_min=0.4, t_max=0.95, eta=1.0,
         cutoff=0.05, sharpness=50.0, normalize_history=False):
    """HiGS, history-guided sampling [x0]. Sadat, Salehi & Weber (2025), arXiv 2509.22300.

    g_k = d_u + w delta (guided x0). History h_k = sum_{i<k} alpha (1 - alpha)^(k-1-i) g_i (EMA of past
    guided predictions; normalize_history divides by the weight sum). Delta = g_k - h_k; projection on g_k:
    Delta = Delta_perp + eta Delta_par; soft DCT high-pass H(R) = sigmoid(sharpness (R - cutoff));
    x0_hat = g_k + w_H(t) filtered, w_H(t) = w_h sqrt((t - t_min)/(t_max - t_min)) for t_min < t <= t_max
    (t = 1 noise), else 0. First step: plain CFG.
    Defaults: w_h 2 (up to 3), alpha 0.75, t_min 0.4, t_max 0.95, cutoff 0.05. eta 1 and sharpness 50
    are placeholders (not recorded in the catalog). Extra cost: no NFE; one latent buffer + a DCT pair.
    """
    g = d_u + _scalar(w, d_c) * (d_c - d_u)
    if state is None:
        return g
    out = g
    if state.h is not None and state.h.shape == g.shape:
        h = state.h / state.wsum if (normalize_history and state.wsum > 0) else state.h
        delta = g - h
        par, perp = _project(delta, g)
        delta = perp + eta * par
        filt = _dct_highpass(delta, cutoff, sharpness)
        tt = _tensor(t, d_c) * torch.ones_like(_bview(torch.ones(g.shape[0], device=g.device), g))
        ramp = torch.sqrt(((tt - t_min) / max(t_max - t_min, _EPS)).clamp(0, 1))
        wt = torch.where((tt > t_min) & (tt <= t_max), w_h * ramp, torch.zeros_like(tt))
        out = g + wt * filt
    state.h = alpha * g if state.h is None or state.h.shape != g.shape else alpha * g + (1 - alpha) * state.h
    state.wsum = alpha + (1 - alpha) * state.wsum
    return out


@_keep_dtype
def low_pass_latent(x, factor=2.5):
    """ALG low-pass of a conditioning image latent (Choi et al. 2025, arXiv 2506.08456): bilinear
    downsample by `factor` then bilinear upsample back."""
    H, W = x.shape[-2:]
    small = _resize(x, (max(1, int(round(H / factor))), max(1, int(round(W / factor)))), "bilinear")
    return _resize(small, (H, W), "bilinear")


def lying_sigma(sigma, dishonesty_factor=-0.05):
    """Lying Sigma Sampler (ComfyUI-Detail-Daemon): the model is told sigma (1 + f) while the sampler
    integrates with sigma. Default f = -0.05 (README -0.1..-0.01)."""
    return sigma * (1 + dishonesty_factor)


def detail_daemon_sigma(sigma, amount=0.1, cfg_scale=7.5):
    """Detail Daemon's under-reported sigma: sigma_model = sigma max(1e-6, 1 - 0.1 amount cfg)."""
    return sigma * max(1e-6, 1 - 0.1 * amount * cfg_scale)


# =============================================================================
# 7. Spatial / frequency / feature-space variants (family F)
# =============================================================================


@_keep_dtype
def samg(d_c, d_u, w=7.5, w_min=None, w_max=None, tau=1e-8):
    """SAMG, spatial adaptive multi guidance [noise]. Li et al. (2026), arXiv 2604.26503.
    E(x) = mean over channels of delta^2; E_hat = (E - min E)/(max E - min E + tau) per sample;
    Omega(x) = w_max - E_hat (w_max - w_min); d_hat = d_u + Omega (.) delta (high-energy pixels get w_min).
    Defaults: [5, 12] at base 7.5 (SD1.5/SDXL); [3, 7.5] SD3.5-M; w_min/w_max None derive them from w
    as [w 2/3, w 1.6] (the SD1.5 ratio). w_min = w_max = w is plain CFG. Extra cost: none."""
    wf = _float(w)
    lo = wf * 2.0 / 3.0 if w_min is None else w_min
    hi = wf * 1.6 if w_max is None else w_max
    delta = d_c - d_u
    E = (delta ** 2).mean(1, keepdim=True)
    B = E.shape[0]
    Ef = E.reshape(B, -1)
    mn = _bview(Ef.min(1).values, E)
    mx = _bview(Ef.max(1).values, E)
    Eh = (E - mn) / (mx - mn + tau)
    omega = hi - Eh * (hi - lo)
    return d_u + omega * delta


def _blur5(x: Tensor) -> Tensor:
    k = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], device=x.device, dtype=x.dtype) / 16.0
    return _sep_conv2d(x, k)


def _pyr_down(x):
    return _blur5(x)[..., ::2, ::2]


def _pyr_up(x, size):
    return _blur5(_resize(x, size, "bilinear"))


def laplacian_pyramid(x: Tensor, levels: int = 2) -> List[Tensor]:
    """Laplacian pyramid over the last two dims; level 0 is the highest-frequency band, the last is the
    low-pass residual. reconstruct_laplacian inverts it exactly."""
    out, G = [], x
    for _ in range(levels - 1):
        down = _pyr_down(G)
        out.append(G - _pyr_up(down, G.shape[-2:]))
        G = down
    out.append(G)
    return out


def reconstruct_laplacian(pyr: Sequence[Tensor]) -> Tensor:
    """Exact inverse of laplacian_pyramid."""
    x = pyr[-1]
    for L in reversed(pyr[:-1]):
        x = L + _pyr_up(x, L.shape[-2:])
    return x


@_keep_dtype
def fdg(d_c, d_u, w=7.5, w_low=None, levels=2, level_scales=None, parallel_weight=1.0, formulation="paper"):
    """Frequency-Decoupled Guidance [x0]. Sadat, Vontobel, Salehi & Weber (2025), arXiv 2506.19713.

    Laplacian pyramid of both x0 estimates; per level k: delta_k = P_c[k] - P_u[k], guided
    g_k = P_u[k] + w_k delta_k. Optional projection (beta = parallel_weight): delta_k' = beta par + orth
    relative to P_c[k]; 'paper' g_k = P_c[k] + (w_k - 1) delta_k', 'diffusers' g_k = P_u[k] + w_k delta_k'
    (identical at beta = 1). Reconstruct. level_scales (high -> low) overrides; otherwise w_high = w on
    level 0 and w_low on the last level (linear in between). w_low None = w/2 (SDXL FID table (5, 10) and
    diffusers [10, 5]). w_low = w is plain CFG. Extra cost: a few blurs; no NFE.
    """
    wf = _float(w)
    lo = wf / 2.0 if w_low is None else float(w_low)
    if level_scales is None:
        level_scales = [wf] if levels == 1 else [wf + (lo - wf) * i / (levels - 1) for i in range(levels)]
    pc, pu = laplacian_pyramid(d_c, levels), laplacian_pyramid(d_u, levels)
    out = []
    for k in range(levels):
        dk = pc[k] - pu[k]
        if parallel_weight != 1.0:
            par, orth = _project(dk, pc[k])
            dk = parallel_weight * par + orth
        wk = level_scales[k]
        out.append(pc[k] + (wk - 1) * dk if formulation == "paper" else pu[k] + wk * dk)
    return reconstruct_laplacian(out)


@_keep_dtype
def fresca_filter(x, scale_low=1.0, scale_high=1.25, freq_cutoff=20):
    """FreSca Fourier band scaling (ComfyUI form): centred 2-D FFT, the block of +-f_c indices around DC
    (f_c = min(freq_cutoff, size//2)) scaled by scale_low, everything else by scale_high."""
    X = torch.fft.fftshift(torch.fft.fftn(x, dim=(-2, -1)), dim=(-2, -1))
    mask = torch.full(X.shape[-2:], float(scale_high), device=x.device)
    H, W = X.shape[-2:]
    ch, cw = H // 2, W // 2
    fh, fw = min(freq_cutoff, ch), min(freq_cutoff, cw)
    mask[ch - fh:ch + fh, cw - fw:cw + fw] = float(scale_low)
    X = X * mask
    return torch.fft.ifftn(torch.fft.ifftshift(X, dim=(-2, -1)), dim=(-2, -1)).real


@_keep_dtype
def fresca(d_c, d_u, w=7.5, scale_low=1.0, scale_high=1.25, freq_cutoff=20):
    """FreSca, Fourier band scaling of the guidance difference [any: linear]. Huang et al. (2025),
    arXiv 2504.02154; ComfyUI FreSca node. d_hat = d_u + w filter(delta).
    Defaults: ComfyUI (1.0, 1.25, 20); paper SDXL scale_high 1.5, SD3 1.2. scales (1, 1) is plain CFG."""
    return d_u + _scalar(w, d_c) * fresca_filter(d_c - d_u, scale_low, scale_high, freq_cutoff)


_WAVELETS = {
    "haar": [1 / math.sqrt(2), 1 / math.sqrt(2)],
    "sym4": [-0.07576571478927333, -0.02963552764599851, 0.49761866763201545, 0.8037387518059161,
             0.29785779560527736, -0.09921954357684722, -0.012603967262037833, 0.0322231006040427],
}


def _dwt_matrix(n: int, wavelet: str, device, dtype) -> Tensor:
    """Periodized orthogonal DWT analysis matrix (n even): rows [low-pass; high-pass]; inverse = transpose."""
    h = _WAVELETS[wavelet]
    L = len(h)
    g = [((-1) ** k) * h[L - 1 - k] for k in range(L)]
    M = torch.zeros(n, n, dtype=torch.float64)
    for i in range(n // 2):
        for k in range(L):
            M[i, (2 * i + k) % n] += h[k]
            M[n // 2 + i, (2 * i + k) % n] += g[k]
    return M.to(device=device, dtype=dtype)


def dwt2(x: Tensor, wavelet: str = "sym4") -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Single-level periodized 2-D DWT over the last two dims (even sizes). Returns (LL, LH, HL, HH)."""
    H, W = x.shape[-2:]
    if H % 2 or W % 2:
        raise ValueError("dwt2 needs even spatial sizes")
    Y = _dwt_matrix(H, wavelet, x.device, x.dtype) @ x @ _dwt_matrix(W, wavelet, x.device, x.dtype).T
    h, w = H // 2, W // 2
    return Y[..., :h, :w], Y[..., :h, w:], Y[..., h:, :w], Y[..., h:, w:]


def idwt2(bands, wavelet: str = "sym4") -> Tensor:
    """Inverse of dwt2 (the analysis matrix is orthogonal, so its transpose)."""
    LL, LH, HL, HH = bands
    Y = torch.cat([torch.cat([LL, LH], -1), torch.cat([HL, HH], -1)], -2)
    H, W = Y.shape[-2:]
    return _dwt_matrix(H, wavelet, Y.device, Y.dtype).T @ Y @ _dwt_matrix(W, wavelet, Y.device, Y.dtype)


@_keep_dtype
def hiwave(d_c, d_u, w=7.5, w_low=1.0, wavelet="sym4"):
    """HiWave wavelet detail guidance [x0]. Vontobel, Sadat, Salehi & Weber (2025), SIGGRAPH Asia,
    arXiv 2506.20452. Single-level DWT: low band L_hat = L(x0_u) + w_low (L(x0_c) - L(x0_u)) (paper
    w_low = 1: the conditional low band, no guidance); each high band B_hat = B(x0_u) + w (B(x0_c) - B(x0_u));
    x0_hat = IDWT. The paper's pipeline (upscale, patch-wise inversion, skip residual) is not included.
    Uses periodized boundaries (pytorch_wavelets defaults to zero padding). Defaults: w 7.5, sym4."""
    wv = _scalar(w, d_c)
    bc, bu = dwt2(d_c, wavelet), dwt2(d_u, wavelet)
    LL = bu[0] + w_low * (bc[0] - bu[0])
    highs = [u + wv * (c - u) for c, u in zip(bc[1:], bu[1:])]
    return idwt2([LL] + highs, wavelet)


class LFCFGState:
    def __init__(self):
        self.prev = None

    def reset(self):
        self.prev = None


def _lf_lowpass(x, k):
    H, W = x.shape[-2:]
    small = _resize(x, (max(1, H // k), max(1, W // k)), "area")
    return _resize(small, (H, W), "bilinear")


@_keep_dtype
def lf_cfg(d_c, d_u, w=7.5, state=None, rho=0.5, k=8):
    """LF-CFG, low-frequency improved CFG [noise: velocity]. Song & Lai (2025), arXiv 2506.21452.
    Low-pass f_l = downsample by k then interpolate back (area / bilinear here; the paper's kernel is not
    stated); d^l = f_l(d), d^h = d - d^l. Per pixel change rate r_j = ||d_j^l - d_j^l(previous step)||_ch,
    mask m_j = [r_j < mean(r_j) + std(r_j)] (slowly changing, redundant). Combination 3:
    d_hat = d_u^l + rho w (m_c d_c^l - m_u d_u^l) + w ((1 - m_c) d_c^l - (1 - m_u) d_u^l) + d_u^h + w (d_c^h - d_u^h).
    First step (no history): plain CFG. rho = 1 is plain CFG. Defaults: rho 0.5, k 8. Extra cost: none."""
    wv = _scalar(w, d_c)
    cl, ul = _lf_lowpass(d_c, k), _lf_lowpass(d_u, k)
    if state is None or state.prev is None or state.prev[0].shape != cl.shape:
        out = d_u + wv * (d_c - d_u)
    else:
        pcl, pul = state.prev

        def mask(cur, prev):
            r = torch.linalg.vector_norm(cur - prev, dim=1, keepdim=True)
            B = r.shape[0]
            rf = r.reshape(B, -1)
            thr = _bview(rf.mean(1) + rf.std(1), r)
            return (r < thr).to(cur.dtype)

        mc, mu = mask(cl, pcl), mask(ul, pul)
        ch, uh = d_c - cl, d_u - ul
        out = ul + rho * wv * (mc * cl - mu * ul) + wv * ((1 - mc) * cl - (1 - mu) * ul) + uh + wv * (ch - uh)
    if state is not None:
        state.prev = (cl.detach(), ul.detach())
    return out


class ZeResFDGState:
    def __init__(self):
        self.rho = None
        self.mode = None

    def reset(self):
        self.rho = None
        self.mode = None


@_keep_dtype
def zeresfdg(d_c, d_u, w=4.5, mode="auto", state=None, blur_sigma=1.0, lam_low=0.6, lam_high=1.3,
             rescale_mix=0.7, ema_beta=0.8, tau_lo=0.45, tau_hi=0.60, high_share_mode="rescale_fdg"):
    """ZeResFDG (CADE 2.5): FDG + rescale + zero-projection [noise: eps]. Rychkovskiy (2025),
    arXiv 2510.12954 (medium).
    FD(v) = lam_l G*v + lam_h (v - G*v) with a Gaussian low-pass G (sigma 1).
    'cfgzero_fd': a = <d_c, d_u>/<d_u, d_u>; r = d_c - a d_u; d_hat = a d_u + w FD(r).
    'rescale_fdg': d_cfg = d_u + w FD(delta); d_res = d_cfg std(d_c)/std(d_cfg); d_hat = m d_res + (1 - m) d_cfg.
    'auto': high-frequency share r_HF = ||delta_h||^2/||delta||^2, EMA rho (beta 0.8), hysteresis
    (tau_lo, tau_hi): rho > tau_hi selects high_share_mode, rho < tau_lo the other (the paper does not say
    which side selects which mode: high_share_mode is a guess). lam_l = lam_h = 1 with m = 0 in
    'rescale_fdg' is plain CFG. Defaults: paper (0.6, 1.3, 0.7, 0.8, 0.45, 0.60)."""
    wv = _scalar(w, d_c)
    delta = d_c - d_u

    def FD(v):
        lo = gaussian_blur2d(v, blur_sigma)
        return lam_low * lo + lam_high * (v - lo)

    use = mode
    if mode == "auto":
        lo_d = gaussian_blur2d(delta, blur_sigma)
        hi_d = delta - lo_d
        e_l, e_h = float((lo_d ** 2).sum()), float((hi_d ** 2).sum())
        r_hf = e_h / max(e_l + e_h, _EPS)
        other = "cfgzero_fd" if high_share_mode == "rescale_fdg" else "rescale_fdg"
        if state is not None:
            state.rho = r_hf if state.rho is None else ema_beta * state.rho + (1 - ema_beta) * r_hf
            rho = state.rho
            prev = state.mode or other
        else:
            rho, prev = r_hf, other
        use = high_share_mode if rho > tau_hi else (other if rho < tau_lo else prev)
        if state is not None:
            state.mode = use
    if use == "cfgzero_fd":
        a = _dot(d_c, d_u) / _sqnorm(d_u).clamp_min(_EPS)
        return a * d_u + wv * FD(d_c - a * d_u)
    d_cfg = d_u + wv * FD(delta)
    d_res = d_cfg * _std(d_c) / _std(d_cfg).clamp_min(1e-8)
    return rescale_mix * d_res + (1 - rescale_mix) * d_cfg


def s_cfg_region_masks(cross_attn: Tensor, self_attn: Optional[Tensor], grid_hw: Tuple[int, int],
                       out_hw: Tuple[int, int], power_terms: int = 4, smooth_sigma: float = 0.5) -> Tensor:
    """S-CFG region masks from attention (Shen et al. 2024, CVPR, arXiv 2404.05384).
    cross_attn (B, N, T) probabilities (N = h w patches, token 0 = START / background), self_attn (B, N, N).
    C_bar = (1/R) sum_{r=1..R} S^r A; smooth each token map (3x3 Gaussian), divide by its spatial mean,
    argmax over tokens, nearest-upsample. Returns one-hot (B, T, H, W)."""
    C = cross_attn.float()
    if self_attn is not None and power_terms > 0:
        acc, cur = torch.zeros_like(C), C
        S = self_attn.float()
        for _ in range(power_terms):
            cur = S @ cur
            acc = acc + cur
        C = acc / power_terms
    B, N, T = C.shape
    h, w = grid_hw
    maps = C.transpose(1, 2).reshape(B, T, h, w)
    maps = gaussian_blur2d(maps, smooth_sigma, 3)
    maps = maps / maps.mean(dim=(-2, -1), keepdim=True).clamp_min(_EPS)
    idx = maps.argmax(1)
    onehot = F.one_hot(idx, T).permute(0, 3, 1, 2).float()
    return F.interpolate(onehot, size=tuple(out_hw), mode="nearest")


@_keep_dtype
def s_cfg(d_c, d_u, w=7.5, masks=None, background_index=0, clamp_min=0.8, clamp_max=3.0, cap=15.0,
          smooth_sigma=0.5):
    """S-CFG, semantic-aware CFG [noise: eps]. Shen, Song, Xue, Wang & Liu (2024), CVPR, arXiv 2404.05384.
    masks: one-hot region masks (B, T, H, W) (s_cfg_region_masks). eta = channel L2 of delta per pixel;
    benchmark mask m_b = 1 - m_background; rho_i = mean_{m_b} eta / mean_{m_i} eta; q = sum_i rho_i m_i,
    clamped to [clamp_min, clamp_max] and <= cap/w (official code; None disables), smoothed;
    d_hat = d_u + w q (.) delta. Extra cost: attention storage (grows with (HW)^2)."""
    if masks is None:
        raise ValueError("s_cfg needs region masks")
    wf = _float(w)
    delta = d_c - d_u
    eta = torch.linalg.vector_norm(delta, dim=1)
    m = masks.to(delta.dtype)
    mb = 1 - m[:, background_index]
    mean_b = (eta * mb).flatten(1).sum(1) / mb.flatten(1).sum(1).clamp_min(1.0)
    cnt = m.flatten(2).sum(2)
    mean_i = (eta.unsqueeze(1) * m).flatten(2).sum(2) / cnt.clamp_min(1.0)
    rho = torch.where(cnt > 0, mean_b.unsqueeze(1) / mean_i.clamp_min(_EPS), torch.ones_like(mean_i))
    q = (rho.view(*rho.shape, 1, 1) * m).sum(1, keepdim=True)
    if clamp_min is not None:
        q = q.clamp(min=clamp_min, max=clamp_max)
    if cap is not None and wf > 0:
        q = q.clamp(max=cap / wf)
    if smooth_sigma:
        q = gaussian_blur2d(q, smooth_sigma, 3)
    return d_u + wf * q * delta


@_keep_dtype
def masked_cond_average(preds, masks, strengths=None):
    """ComfyUI area / mask conditioning average (comfy/samplers.py calc_cond_batch):
    d = sum_i a_i M_i d_i / sum_i a_i M_i (zero where nothing covers a pixel)."""
    n = len(preds)
    strengths = [1.0] * n if strengths is None else strengths
    num = torch.zeros_like(preds[0])
    den = torch.zeros_like(preds[0])
    for d, m, a in zip(preds, masks, strengths):
        num = num + a * m * d
        den = den + a * m * torch.ones_like(d)
    return torch.where(den > 0, num / den.clamp_min(1e-37), torch.zeros_like(num))


# ---- feature-space combiners: the caller hooks the attention and passes the features ----


@_keep_dtype
def nag(z_pos, z_neg, scale=5.0, tau=2.5, alpha=0.25, norm="l1"):
    """Normalized Attention Guidance [attention-output features, per token]. Chen, Bandyopadhyay, Zou &
    Song (2025), NeurIPS, arXiv 2505.21179.
    Code form: z~ = scale z+ - (scale - 1) z- (the paper's phi = scale - 1); R = ||z~||/||z+|| per token
    (last dim; L1 in the paper and ComfyUI, L2 in the FLUX processor); z^ = z~ min(R, tau)/R;
    out = alpha z^ + (1 - alpha) z+. scale <= 1 returns z+.
    Defaults: scale 5 (Flux / SD3.5; 3 SDXL; 9 Wan), tau 2.5 (ComfyUI 1.5), alpha 0.25 (Flux).
    Extra cost: the attention call on the negative context (or a full pass in ComfyUI's built-in)."""
    if scale <= 1:
        return z_pos
    zt = scale * z_pos - (scale - 1) * z_neg
    p = 1 if norm == "l1" else 2
    R = torch.linalg.vector_norm(zt, ord=p, dim=-1, keepdim=True) / \
        torch.linalg.vector_norm(z_pos, ord=p, dim=-1, keepdim=True).clamp_min(_EPS)
    zh = zt * torch.clamp(R, max=tau) / R.clamp_min(_EPS)
    return alpha * zh + (1 - alpha) * z_pos


@_keep_dtype
def nasa(z_pos, z_neg, alpha=0.1):
    """NASA negative-away steer attention [cross-attention output]. Nguyen et al. (2024/2025), ICCV,
    arXiv 2412.02687: Z = Z+ - alpha Z-. alpha 0.1 (SD1.5), 0.2 (SD2.1), 0.5 (PixArt)."""
    return z_pos - alpha * z_neg


@_keep_dtype
def orthogonal_attention_negative(z_pos, z_neg, alpha=4.0):
    """Orthogonal negative guidance in attention features [image-to-text attention output, per token and
    head along the last dim]. Ko et al. (2026), arXiv 2605.29390: Z^ = Z+ - alpha (Z- - proj_{Z+} Z-).
    Defaults: FLUX-dev alpha 4 from step 2; FLUX-Schnell 2; SD3.5-L 4."""
    coef = (z_neg * z_pos).sum(-1, keepdim=True) / (z_pos * z_pos).sum(-1, keepdim=True).clamp_min(_EPS)
    return z_pos - alpha * (z_neg - coef * z_pos)


def sparsemax(z: Tensor, dim: int = -1) -> Tensor:
    """Sparsemax (entmax with alpha = 2), closed form."""
    zs, _ = torch.sort(z, dim=dim, descending=True)
    view = [1] * z.ndim
    view[dim] = -1
    k = torch.arange(1, z.shape[dim] + 1, device=z.device, dtype=z.dtype).view(view)
    cs = zs.cumsum(dim)
    ksz = (1 + k * zs > cs).sum(dim=dim, keepdim=True)
    tau = (torch.gather(cs, dim, ksz - 1) - 1) / ksz.to(z.dtype)
    return torch.clamp(z - tau, min=0)


def entmax_bisect(z: Tensor, alpha: float = 1.5, dim: int = -1, n_iter: int = 50) -> Tensor:
    """alpha-entmax by bisection (alpha = 1 softmax, 2 sparsemax, 1.5 entmax15)."""
    if alpha == 1:
        return torch.softmax(z, dim)
    zf = z.float() * (alpha - 1)
    d = z.shape[dim]
    mx = zf.max(dim=dim, keepdim=True).values
    lo = mx - 1.0
    hi = mx - (1.0 / d) ** (alpha - 1)
    for _ in range(n_iter):
        tau = (lo + hi) / 2
        p = torch.clamp(zf - tau, min=0) ** (1.0 / (alpha - 1))
        s = p.sum(dim=dim, keepdim=True)
        lo = torch.where(s >= 1, tau, lo)
        hi = torch.where(s < 1, tau, hi)
    p = torch.clamp(zf - (lo + hi) / 2, min=0) ** (1.0 / (alpha - 1))
    return (p / p.sum(dim=dim, keepdim=True).clamp_min(_EPS)).to(z.dtype)


@_keep_dtype
def pladis_attention(q, k, v, lam=2.0, alpha=1.5, scale=None):
    """PLADIS sparse-attention extrapolation [cross-attention]. Kim & Sim (2025), ICCV, arXiv 2503.07677.
    A_soft = softmax(QK^T/sqrt(d)), A_sparse = alpha-entmax(QK^T/sqrt(d));
    out = A_soft V + lam (A_sparse V - A_soft V). lam 0 is plain attention. Defaults: lam 2, alpha 1.5."""
    sc = scale if scale is not None else 1.0 / math.sqrt(q.shape[-1])
    logits = (q @ k.transpose(-2, -1)) * sc
    soft = torch.softmax(logits, -1) @ v
    sparse = (sparsemax(logits) if alpha == 2 else entmax_bisect(logits, alpha)) @ v
    return soft + lam * (sparse - soft)


@_keep_dtype
def gag(t_sparse, t_dense, lam=10.0, zeta=0.0, eta=float("inf")):
    """Geometry-Aware Attention Guidance [cross-attention outputs]. Kim (2026), arXiv 2603.02531.
    r = T_a - T_d (sparse minus dense output); r_par = proj onto T_a; r~ = r_par + zeta r_orth;
    out = T_a + lam min(1, eta/||r~||) r~ (per token, last dim). eta (norm cap) is not stated in the text
    read: inf = no cap. Defaults: lam about 10, zeta 0."""
    r = t_sparse - t_dense
    coef = (r * t_sparse).sum(-1, keepdim=True) / (t_sparse * t_sparse).sum(-1, keepdim=True).clamp_min(_EPS)
    r_par = coef * t_sparse
    rt = r_par + zeta * (r - r_par)
    n = torch.linalg.vector_norm(rt, dim=-1, keepdim=True)
    cap = torch.clamp(eta / n.clamp_min(_EPS), max=1.0) if math.isfinite(eta) else torch.ones_like(n)
    return t_sparse + lam * cap * rt


@_keep_dtype
def vsf_attention(q, k_pos, v_pos, k_neg, v_neg, alpha=3.5, neg_bias=-0.1, scale=None):
    """Value Sign Flip cross-attention [attention]. Guo & Du (2025/2026), ICLR, arXiv 2508.10931.
    Z = softmax(Q [K+; K-]^T/sqrt(d) + B) [V+; -alpha V-], B = neg_bias on the negative columns.
    Output-space CFG must be off. Defaults: SD3.5-L-turbo alpha 3.5, bias -0.1; Wan2.1 1.7, -0.2."""
    sc = scale if scale is not None else 1.0 / math.sqrt(q.shape[-1])
    K = torch.cat([k_pos, k_neg], -2)
    V = torch.cat([v_pos, -alpha * v_neg], -2)
    logits = (q @ K.transpose(-2, -1)) * sc
    bias = torch.zeros(K.shape[-2], device=q.device, dtype=logits.dtype)
    bias[k_pos.shape[-2]:] = neg_bias
    return torch.softmax(logits + bias, -1) @ V


@_keep_dtype
def negative_token_merge(o_src, o_ref, alpha=0.9, tau=0.65):
    """NegToMe [token features]. Singh et al. (2024), arXiv 2412.01339. S = cos(o_src, o_ref) (L_src x L_ref);
    target = o_ref[argmax_j S]; mask = max_j S > tau; o_src <- o_src + mask alpha (o_src - target).
    Defaults: alpha 0.9, tau 0.65, first ~10% of sampling."""
    a = F.normalize(o_src, dim=-1)
    b = F.normalize(o_ref, dim=-1)
    S = a @ b.transpose(-2, -1)
    best, idx = S.max(-1)
    tgt = torch.gather(o_ref, -2, idx.unsqueeze(-1).expand(*idx.shape, o_ref.shape[-1]))
    m = (best > tau).unsqueeze(-1).to(o_src.dtype)
    return o_src + m * alpha * (o_src - tgt)


@_keep_dtype
def modulation_guidance(y, y_pos, y_neg, w_m=3.0):
    """Modulation guidance on the pooled text embedding [adaLN input]. Starodubcev et al. (2026), ICLR,
    arXiv 2602.09268: y_hat = y + w_m (y(p+) - y(p-)), used from a chosen block onward. Default w_m 3."""
    return y + w_m * (y_pos - y_neg)


@_keep_dtype
def token_weight_emphasis(e, weights, e_empty=None, mode="comfy"):
    """Per-token prompt emphasis [text embeddings (B, L, D)].
    'comfy' (ComfyUI core): e' = e_empty + w (e - e_empty) per token (embedding-space CFG with the empty
    prompt as the null); 'a1111': e' = w e, then rescaled so the mean matches the original."""
    wt = weights.to(e.dtype) if torch.is_tensor(weights) else torch.as_tensor(weights, dtype=e.dtype, device=e.device)
    while wt.ndim < e.ndim:
        wt = wt.unsqueeze(-1)
    if mode == "comfy":
        if e_empty is None:
            raise ValueError("mode 'comfy' needs e_empty")
        return e_empty + wt * (e - e_empty)
    out = e * wt
    return out * (e.mean() / out.mean().clamp_min(_EPS) if out.mean().abs() > _EPS else 1.0)


# =============================================================================
# 8. Token / logit space (family G): LM CFG, contrastive decoding, discrete diffusion
# =============================================================================


@_keep_dtype
def lm_cfg(l_c, l_u, w=1.5, normalize=True):
    """CFG for language models [log-probabilities]. Sanchez et al. (2023), ICML 2024, arXiv 2306.17806.
    l_hat = l_u + w (l_c - l_u) on log_softmax'd logits (HF UnbatchedClassifierFreeGuidanceLogitsProcessor);
    the unconditional stream sees the same continuation without the prompt. Apply top-k/top-p after.
    Defaults: w 1.5 (1-2 benchmarks; 3 for chat adherence with a negative prompt). Also token-level
    CFG for AR image models (LlamaGen, Janus, Emu3) and D-CFG / discrete CFG on transition log-probs."""
    if normalize:
        l_c, l_u = F.log_softmax(l_c, -1), F.log_softmax(l_u, -1)
    return l_u + _scalar(w, l_c) * (l_c - l_u)


def plausibility_mask(logits, alpha=0.1):
    """V_head = {v : p(v) >= alpha max p} (contrastive decoding, VCD, DoLa)."""
    lp = F.log_softmax(logits.float(), -1)
    return lp >= math.log(alpha) + lp.max(-1, keepdim=True).values


@_keep_dtype
def contrastive_decoding(l_e, l_a, beta=0.5, alpha=0.1, form="obrien", amateur_temperature=1.0, fill=float("-inf")):
    """Contrastive decoding [logits]. Li et al. (2022), ACL 2023, arXiv 2210.15097; O'Brien & Lewis (2023),
    arXiv 2309.09117.
    'obrien': s = (1 + beta) l_e - beta l_a = l_a + w (l_e - l_a), w = 1 + beta (also VCD with an
    image-noised weak branch, arXiv 2311.16922, beta = alpha 1);
    'li': s = log p_e - log p_a(/tau_a) (the w -> infinity limit);
    'scaled': s = log p_e - beta log p_a (LoopCD, arXiv 2609.24196, beta = lambda 0.2-0.3).
    Tokens outside V_head (p_e >= alpha max p_e) get `fill`. Defaults: beta 0.5, alpha 0.1."""
    mask = plausibility_mask(l_e, alpha)
    if form == "obrien":
        s = (1 + beta) * l_e - beta * l_a
    elif form == "li":
        s = F.log_softmax(l_e, -1) - F.log_softmax(l_a / amateur_temperature, -1)
    elif form == "scaled":
        s = F.log_softmax(l_e, -1) - beta * F.log_softmax(l_a, -1)
    else:
        raise ValueError("form must be 'obrien', 'li' or 'scaled'")
    return s.masked_fill(~mask, fill)


@_keep_dtype
def context_aware_decoding(l_c, l_u, alpha=0.5):
    """Context-aware decoding (CAD) [logits]. Shi et al. (2023), NAACL 2024, arXiv 2305.14739.
    (1 + alpha) l_c - alpha l_u (w = 1 + alpha; the weak branch drops the evidence). alpha 0.5
    (summarization), 1.0 (knowledge conflicts)."""
    return (1 + alpha) * l_c - alpha * l_u


@_keep_dtype
def coherence_boost(l_full, l_short, a_k=-0.5, form="generation"):
    """Coherence boosting [log-probs]. Malkin, Wang & Jojic (2022), ACL, arXiv 2110.08294.
    'generation': (1 - a_k) log p_full + a_k log p_short (CFG with w = 1 - a_k against a k-token context);
    'ranking': log p_full + a_k log p_short. a_k -0.6..-0.5 GPT-2, -0.3..-0.2 GPT-3; k about 10 tokens."""
    lf, ls = F.log_softmax(l_full, -1), F.log_softmax(l_short, -1)
    if form == "generation":
        return (1 - a_k) * lf + a_k * ls
    return lf + a_k * ls


def _jsd2(lp_a, lp_b):
    """Jensen-Shannon divergence in bits (bounded in [0, 1]) along the last dim of log-probs."""
    pa, pb = lp_a.exp(), lp_b.exp()
    lm = torch.log((pa + pb) / 2 + 1e-30)
    kl_a = (pa * (lp_a - lm)).sum(-1, keepdim=True)
    kl_b = (pb * (lp_b - lm)).sum(-1, keepdim=True)
    return (0.5 * (kl_a + kl_b) / math.log(2)).clamp(0, 1)


@_keep_dtype
def adacad(l_c, l_u, floor=None):
    """AdaCAD, divergence-adaptive CAD [log-probs]. Wang, Prasad, Stengel-Eskin & Bansal (2024), NAACL 2025,
    arXiv 2409.07394. alpha_t = JSD(p_u || p_c) in [0, 1] (bits); optional floor (0.3 for summarization);
    l_hat = l_u + (1 + alpha_t)(l_c - l_u)."""
    lc, lu = F.log_softmax(l_c, -1), F.log_softmax(l_u, -1)
    a = _jsd2(lu, lc)
    if floor is not None:
        a = a.clamp_min(floor)
    return lu + (1 + a) * (lc - lu)


def _nucleus_mask(logits, top_p):
    p = torch.softmax(logits.float(), -1)
    sp, idx = p.sort(-1, descending=True)
    keep_sorted = (sp.cumsum(-1) - sp) < top_p
    keep = torch.zeros_like(keep_sorted)
    return keep.scatter(-1, idx, keep_sorted)


@_keep_dtype
def dexperts(z, z_pos, z_neg, alpha=2.0, top_p=0.9, fill=float("-inf")):
    """DExperts [logits]. Liu et al. (2021), ACL, arXiv 2105.03023. Truncate the base to its top-p set,
    then z' + alpha (z+ - z-). Defaults: alpha 2 (detoxification), top_p 0.9."""
    keep = _nucleus_mask(z, top_p)
    return z.masked_fill(~keep, fill) + alpha * (z_pos - z_neg)


@_keep_dtype
def dola(final_logits, layer_logits, alpha=0.1, fill=float("-inf")):
    """DoLa, decoding by contrasting layers [logits]. Chuang et al. (2023), ICLR 2024, arXiv 2309.03883.
    For each token choose the premature layer M = argmax_j JSD(q_N || q_j) over the candidate bucket;
    score = log q_N - log q_M on V_head (alpha 0.1). layer_logits: list of early-exit logits."""
    lpN = F.log_softmax(final_logits, -1)
    lps = torch.stack([F.log_softmax(l, -1) for l in layer_logits], 0)
    jsd = torch.stack([_jsd2(lpN, lp).squeeze(-1) for lp in lps], 0)
    best = jsd.argmax(0)
    lpM = torch.gather(lps, 0, best.unsqueeze(0).unsqueeze(-1).expand(1, *lpN.shape)).squeeze(0)
    return (lpN - lpM).masked_fill(~plausibility_mask(final_logits, alpha), fill)


@_keep_dtype
def igg(l_c, l_u, w_igg=2.1, w_cfg=None):
    """Information-Grounding Guidance [logits (B, L, V) for one scale]. Nguyen et al. (2025), ICML 2026,
    arXiv 2509.23876. delta = l_c - l_u; g = softmax(delta delta^T / sqrt(V)) delta (self-attention over
    the scale's positions); l_hat = l_u + w_cfg delta (if given) + w_igg g. The attention matrix = I gives
    CFG. Defaults: w_igg 2.10 (VAR-d36 at 512), pure IGG (no CFG term). Extra cost: one L x L attention."""
    delta = l_c - l_u
    V = delta.shape[-1]
    A = torch.softmax(delta @ delta.transpose(-2, -1) / math.sqrt(V), -1)
    out = l_u + w_igg * (A @ delta)
    if w_cfg is not None:
        out = out + w_cfg * delta
    return out


@_keep_dtype
def ssg_coarse_prior(l_prev, out_hw):
    """SSG coarse prior (Shin, Hur & Kim 2026, ICLR, arXiv 2602.05534) [logits (B, V, h, w)]: DCT of the
    bilinear upsample of the previous scale, with its low-frequency block replaced by the previous scale's
    own DCT coefficients (scaled by sqrt(hw/h'w') to keep amplitudes; the scaling is our choice)."""
    h0, w0 = l_prev.shape[-2:]
    H, W = out_hw
    up = _resize(l_prev, (H, W), "bilinear")
    D = dct2(up)
    Dp = dct2(l_prev) * math.sqrt((H * W) / (h0 * w0))
    D[..., :h0, :w0] = Dp
    return idct2(D)


@_keep_dtype
def ssg_guidance(l, l_prior, beta=1.0):
    """SSG guided logits: l + beta_k (l - l_prior), beta_k = beta (1 - (k - 1)/K) (linear decay over scales)."""
    return l + beta * (l - l_prior)


def softcfg_value_weights(p_max: Tensor) -> Tensor:
    """SoftCFG step-normalized value weights (Xu, Tiulpin & Blaschko 2025, arXiv 2510.00996) (medium):
    w_i = 1 - p_max(i); w_hat_i = 1 - (1 - w_i)/sum_j (1 - w_j). Multiply the unconditional branch's cached
    value vectors of past tokens by w_hat_i."""
    one_minus = p_max.float()
    return 1 - one_minus / one_minus.sum(-1, keepdim=True).clamp_min(_EPS)


@_keep_dtype
def normalized_masked_cfg(l_c, l_u, w=2.0):
    """Normalized CFG for masked diffusion [log-probs]. Rojas et al. (2025), arXiv 2507.08965 (medium):
    softmax(w l_c + (1 - w) l_u) keeps the total unmasking rate; unnormalized rate-space CFG multiplies it by
    Z_w (renyi_normalizer). Same formula as D-CFG's per-token geometric guidance (Schiff et al. 2024,
    arXiv 2412.10193, gamma 3)."""
    lc, lu = F.log_softmax(l_c, -1), F.log_softmax(l_u, -1)
    return F.log_softmax(w * lc + (1 - w) * lu, -1)


def renyi_normalizer(l_c, l_u, w=2.0):
    """Z_w = sum_v p_c^w p_u^(1 - w) = exp((w - 1) D_w(p_c || p_u)) (>= 1 for w > 1)."""
    lc, lu = F.log_softmax(l_c.float(), -1), F.log_softmax(l_u.float(), -1)
    return torch.logsumexp(w * lc + (1 - w) * lu, -1).exp()


@_keep_dtype
def prob_space_cfg(p_c, p_u, w=1.5):
    """Probability-space CFG (arithmetic extrapolation): clip(p_u + w (p_c - p_u), 0), renormalized."""
    p = (p_u + w * (p_c - p_u)).clamp_min(0)
    return p / p.sum(-1, keepdim=True).clamp_min(_EPS)


@_keep_dtype
def ctmc_rate_guidance(R_c, R_u, gamma=2.0):
    """Predictor-free guidance on CTMC rate matrices (Nisonoff et al. 2024, arXiv 2406.01572) [rates (..., S, S)]:
    off-diagonal R^gamma = R_c^gamma R_u^(1 - gamma); diagonal = -sum of the off-diagonal row."""
    S = R_c.shape[-1]
    eye = torch.eye(S, device=R_c.device, dtype=torch.bool)
    rc, ru = R_c.clamp_min(1e-30), R_u.clamp_min(1e-30)
    off = torch.exp(gamma * rc.log() + (1 - gamma) * ru.log())
    both_zero = (R_c <= 0) & (R_u <= 0)
    off = torch.where(both_zero | eye, torch.zeros_like(off), off)
    return off - torch.diag_embed(off.sum(-1))


@_keep_dtype
def aram(l_c, l_u, lambda_max=1.0, beta=0.1, eps=1e-6):
    """ARAM SNR-adaptive per-token weight (Kim & Ye 2026, arXiv 2603.17677) [log-probs] (medium):
    lambda = lambda_max tanh(beta (KL(p_c||p_u) + KL(p_u||p_c)) / (H(p_c) + eps)); l_u + lambda (l_c - l_u)."""
    lc, lu = F.log_softmax(l_c, -1), F.log_softmax(l_u, -1)
    pc, pu = lc.exp(), lu.exp()
    sig = (pc * (lc - lu)).sum(-1, keepdim=True) + (pu * (lu - lc)).sum(-1, keepdim=True)
    ent = -(pc * lc).sum(-1, keepdim=True)
    lam = lambda_max * torch.tanh(beta * sig / (ent + eps))
    return lu + lam * (lc - lu)


@_keep_dtype
def reward_weighted_cfg(l_u, l_ys, rewards, gamma=1.0, nucleus=0.95, fill=float("-inf")):
    """Reward-weighted CFG (Peysakhovich & Berman 2026, arXiv 2604.15577) [log-probs]:
    A(y) = standardized reward; l_hat = l_u + gamma sum_y A(y)(l_y - l_u), restricted to the unconditional
    95% nucleus."""
    lu = F.log_softmax(l_u, -1)
    r = torch.as_tensor(rewards, dtype=torch.float32, device=l_u.device)
    A = (r - r.mean()) / r.std().clamp_min(_EPS) if r.numel() > 1 else torch.zeros_like(r)
    out = lu
    for a, ly in zip(A, l_ys):
        out = out + gamma * a * (F.log_softmax(ly, -1) - lu)
    return out.masked_fill(~_nucleus_mask(l_u, nucleus), fill)


def acfg_remask(confidence: Tensor, remaskable: Tensor, rho: float = 0.7) -> Tensor:
    """A-CFG unconditional input (Li et al. 2025, arXiv 2505.20199): per row, re-mask the ceil(rho n)
    lowest-confidence remaskable positions. Returns a bool mask (True = re-mask)."""
    conf = confidence.float().masked_fill(~remaskable, float("inf"))
    out = torch.zeros_like(remaskable)
    counts = remaskable.sum(-1)
    order = conf.argsort(-1)
    for i in range(conf.shape[0]):
        k = int(math.ceil(rho * int(counts[i])))
        if k > 0:
            out[i, order[i, :k]] = True
    return out


@_keep_dtype
def sphere_project(z, radius=None):
    """SphereAR constant-norm projection (Ke & Xue 2025, arXiv 2509.24335): z <- R z/||z|| per token
    (last dim), R = sqrt(d) by default."""
    R = math.sqrt(z.shape[-1]) if radius is None else radius
    return R * z / torch.linalg.vector_norm(z, dim=-1, keepdim=True).clamp_min(_EPS)


# =============================================================================
# 9. Training-side helpers (family H) - loss targets only; nothing here trains
# =============================================================================


def condition_dropout(cond: Tensor, null: Tensor, p: float = 0.1, generator=None):
    """Replace each sample's condition with the null with probability p (Ho & Salimans; p 0.1-0.2).
    Returns (cond', dropped_mask)."""
    B = cond.shape[0]
    drop = torch.rand(B, generator=generator) < p
    drop = drop.to(cond.device)
    return torch.where(_bview(drop, cond), null.expand_as(cond), cond), drop


@_keep_dtype
def model_guidance_target(f, d_c_ema, d_u_ema, w_mg=1.45, t=None, t_high=0.75, dropped=None):
    """Model-guidance (MG) regression target (Tang, Bao, Chen & Guo 2025, arXiv 2502.12154):
    f' = f + (w_MG - 1) sg(D_ema(c) - D_ema(u)) for non-dropped samples with t < t_high, else f.
    Fixed point: CFG with w_eff = 1/(2 - w_MG). Defaults: w_MG 1.45, t_high 0.75 (RF, t = 1 noise)."""
    g = (w_mg - 1) * (d_c_ema - d_u_ema).detach()
    on = torch.ones(f.shape[0], dtype=torch.bool, device=f.device)
    if t is not None:
        on = on & (torch.as_tensor(t, device=f.device).reshape(-1).expand(f.shape[0]) < t_high)
    if dropped is not None:
        on = on & ~dropped.to(f.device)
    return f + _bview(on.to(f.dtype), f) * g


@_keep_dtype
def gft_prediction(d_theta, d_u_sg, beta):
    """Guidance-Free Training mixed prediction (Chen et al. 2025, ICML, arXiv 2501.15420):
    beta D_theta(x; c, beta) + (1 - beta) sg[D_theta(x; u, 1)], regressed onto the usual target;
    sample with beta = 1/w (one pass)."""
    b = _scalar(beta, d_theta)
    return b * d_theta + (1 - b) * d_u_sg.detach()


def contrastive_fm_loss(pred, target, lam=0.05):
    """Contrastive flow matching loss (Stoica et al. 2025, ICCV, arXiv 2506.05350):
    ||pred - f||^2 - lam ||pred - f~||^2 with f~ the target of another batch element (rolled)."""
    neg = torch.roll(target, 1, 0)
    return ((pred - target) ** 2).mean() - lam * ((pred - neg) ** 2).mean()


def cca_loss(logratio_pos, logratio_neg, beta=0.02, lam=500.0):
    """Condition Contrastive Alignment (Chen, Su, Sun & Zhu 2024, ICLR 2025, arXiv 2410.09347):
    -log sigmoid(beta lr_pos) - lam log sigmoid(-beta lr_neg), lr = log p_theta - log p_ref (sequence level)."""
    return (-F.logsigmoid(beta * logratio_pos) - lam * F.logsigmoid(-beta * logratio_neg)).mean()


# =============================================================================
# 10. Evaluation helper (family I)
# =============================================================================


def effective_guidance_scale(d_m, d_c, d_u):
    """GA-Eval effective guidance scale (Xie et al. 2026, arXiv 2602.22570): a_t = <d_m - d_u, delta>/<delta, delta>
    per sample; w_e,t = |a_t|. Average over steps and compare the method against CFG run at w_e."""
    delta = (d_c - d_u).float()
    a = ((d_m - d_u).float() * delta).flatten(1).sum(1) / delta.flatten(1).pow(2).sum(1).clamp_min(_EPS)
    return a.abs()


# =============================================================================
# 11. Registry and notes
# =============================================================================


def _reg(fn, family, space, defaults=None, needs=(), node=False, state=None, state_knobs=(), caller=None,
         kind="combiner", cite="", batch_coupled=False):
    return {"fn": fn, "family": family, "space": space, "defaults": dict(defaults or {}), "needs": tuple(needs),
            "node": node, "state": state, "state_knobs": tuple(state_knobs), "caller": caller, "kind": kind,
            "cite": cite, "batch_coupled": batch_coupled}


_PERT = "the caller runs D(x_t; c) with "

REGISTRY: Dict[str, Dict[str, Any]] = {
    # ---- family A (output space) ----
    "cfg": _reg(cfg, "A", "any", node=True, cite="Ho & Salimans 2022, arXiv 2207.12598"),
    "rescale_cfg": _reg(rescale_cfg, "A", "v", {"phi": 0.7}, node=True, cite="Lin et al. 2024, arXiv 2305.08891"),
    "dynamic_threshold_cfg": _reg(dynamic_threshold_cfg, "A", "x0", {"p": 0.995, "s_max": 1.0}, node=True,
                                  cite="Saharia et al. 2022, arXiv 2205.11487"),
    "mimic_cfg": _reg(mimic_cfg, "A", "x0", {"mimic_scale": 7.0, "threshold_percentile": 1.0,
                                             "separate_feature_channels": True, "scaling_startpoint": "MEAN",
                                             "variability_measure": "AD", "interpolate_phi": 1.0},
                      node=True, cite="mcmonkey 2023, sd-dynamic-thresholding"),
    "apg": _reg(apg, "A", "x0", {"eta": 0.0, "norm_threshold": 15.0, "momentum": -0.5, "formulation": "paper"},
                node=True, state=lambda k: APGMomentum(), cite="Sadat et al. 2025, arXiv 2410.02416"),
    "cfg_zero_star": _reg(cfg_zero_star, "A", "noise", {"zero_init_steps": 1}, node=True,
                          cite="Fan et al. 2025, arXiv 2503.18886"),
    "cfg_zero_star_comfy": _reg(cfg_zero_star_comfy, "A", "x0", node=True, cite="ComfyUI CFGZeroStar; arXiv 2503.18886"),
    "tcfg": _reg(tcfg, "A", "noise", node=True, cite="Kwon et al. 2025, arXiv 2503.18137"),
    "beta_cfg": _reg(beta_cfg, "A", "noise", {"a": 2.0, "b": 2.0, "gamma": 1.0, "peak_normalize": False},
                     node=True, cite="Malarz et al. 2025, arXiv 2502.10574"),
    "adg": _reg(adg, "A", "x0", {"max_angle": math.pi / 3}, node=True, cite="Jin et al. 2025, arXiv 2506.11039"),
    "power_law_cfg": _reg(power_law_cfg, "A", "noise", {"alpha": 0.9, "omega": None}, node=True,
                          cite="Lehman Pavasovic et al. 2025, arXiv 2502.07849"),
    "ep_cfg": _reg(ep_cfg, "A", "noise", {"robust": True, "lo": 45.0, "hi": 55.0}, node=True,
                   cite="Zhang et al. 2024, arXiv 2412.09966"),
    "cfg_renorm": _reg(cfg_renorm, "A", "noise", {"rho": 1.0}, node=True, cite="Qin et al. 2025, arXiv 2503.21758"),
    "cfg_norm_per_token": _reg(cfg_norm_per_token, "A", "noise", {"dim": 1, "strength": 1.0, "mode": "match"},
                               node=True, cite="Qwen-Image / ComfyUI CFGNorm"),
    "recfg": _reg(recfg, "A", "noise", {"ratio": None}, cite="Xia et al. 2025, arXiv 2410.18737",
                  caller="an offline ratio table E[d_c]/E[d_u] per timestep (ReCFGRatioTable)"),
    "mambo_g": _reg(mambo_g, "A", "noise", {"alpha": 8.0}, node=True, cite="Zhu et al. 2025, arXiv 2508.03442"),
    "skimmed_cfg": _reg(skimmed_cfg, "A", "x0", {"skimming_scale": 7.0, "full_skim_negative": False,
                                                 "disable_flipping_filter": False}, node=True,
                        cite="Extraltodeus, Skimmed_CFG"),
    "automatic_cfg": _reg(automatic_cfg, "A", "x0", {"reference_scale": 8.0, "top_k": 0.25, "mode": "hard"},
                          node=True, cite="Extraltodeus, ComfyUI-AutomaticCFG"),
    "mahiro": _reg(mahiro, "A", "x0", node=True, cite="ComfyUI Mahiro (PR #5975)", batch_coupled=True),
    "reinhard_cfg": _reg(reinhard_cfg, "A", "x0", {"multiplier": 1.0}, node=True,
                         cite="ComfyUI LatentOperationTonemapReinhard"),
    "smc_cfg": _reg(smc_cfg, "A", "noise", {"lam": 5.0, "k": 0.2, "switching": "sign"}, node=True,
                    state=lambda k: SMCState(), cite="Wang et al. 2026, arXiv 2603.03281"),
    "pmc_cfg": _reg(pmc_cfg, "A", "x0", {"gamma_cap": 1.05}, node=True, cite="Peng & Ma 2026, arXiv 2609.24287"),
    "adamag": _reg(adamag, "A", "noise", {"beta": 0.1, "gamma": 4.0, "w_min": 1.0}, node=True,
                   cite="Esmati et al. 2026, arXiv 2605.20079"),
    "fbg": _reg(fbg, "A", "x0", {"hybrid": False, "pi": 0.95, "t0": 0.5, "t1": 0.4, "lambda_max": 10.0},
                node=True, state_knobs=("pi", "t0", "t1", "lambda_max"),
                state=lambda k: FeedbackGuidanceState(pi=k.get("pi", 0.95), t0=k.get("t0", 0.5), t1=k.get("t1", 0.4),
                                                      lambda_max=k.get("lambda_max", 10.0)),
                cite="Koulischer et al. 2025, arXiv 2506.06085"),
    "vags": _reg(vags, "A", "noise", {"kappa": 1.0}, node=True, cite="Luo et al. 2026, arXiv 2605.15661"),
    "cfg_oec": _reg(cfg_oec, "A", "noise", {"tau": 0.5}, node=True, state=lambda k: OECState(),
                    cite="Yang et al. 2025, arXiv 2511.14075"),
    # ---- family B (schedules with state) ----
    "adaptive_guidance": _reg(adaptive_guidance, "B", "noise", {"threshold": 0.991}, node=True,
                              state=lambda k: AdaptiveGuidanceState(), cite="Castillo et al. 2023, arXiv 2312.12487"),
    "transition_point": _reg(transition_point_guidance, "B", "noise", {"opposite": 0.0}, node=True,
                             state=lambda k: TransitionPointState(), cite="Jain et al. 2024, arXiv 2411.16738"),
    "windowed_negative": _reg(windowed_negative, "B", "any", {"start": 0.17, "end": 0.5}, needs=("d_n",), node=True,
                              cite="Ban et al. 2024, arXiv 2406.02965"),
    "segmented_guidance": _reg(segmented_guidance, "B", "noise", {"tau": 0.2}, needs=("d_weak",),
                               caller="d_weak = a perturbed (e.g. layer-skip) or inferior-model conditional prediction",
                               cite="Yuan et al. 2026, arXiv 2603.20584"),
    "cfg_cache": _reg(cfg_cache, "B", "noise", {}, state=lambda k: CFGCacheState(),
                      caller="skip the null pass on reuse steps (pass d_u=None); state.is_refresh_step(step)",
                      cite="Lv et al. 2024, arXiv 2410.19355"),
    "higs": _reg(higs, "E", "x0", {"w_h": 2.0, "alpha": 0.75, "t_min": 0.4, "t_max": 0.95, "eta": 1.0,
                                   "cutoff": 0.05, "sharpness": 50.0}, node=True, state=lambda k: HiGSState(),
                 cite="Sadat et al. 2025, arXiv 2509.22300"),
    "tsr": _reg(cfg_tsr, "E", "x0", {"k": 0.95, "tsr_sigma": 1.0}, node=True, cite="Xu et al. 2025, arXiv 2510.01184"),
    "epsilon_scaling": _reg(cfg_epsilon_scaling, "E", "x0", {"factor": 1.005}, node=True,
                            cite="Ning et al. 2024, arXiv 2308.15321"),
    # ---- family C (weak branch / perturbation) ----
    "weak_branch_cfg": _reg(weak_branch_cfg, "C", "any", {}, needs=("d_weak",), caller="any weak prediction",
                            cite="shared form"),
    "autoguidance": _reg(weak_branch_cfg, "C", "any", {"w": 2.0}, needs=("d_weak",),
                         caller="d_weak = D_guide(x_t; c): a smaller and/or shorter-trained model of the same task "
                                "(EDM2-XS at 1/16 of the training for EDM2-S; w 2.05-2.10)",
                         cite="Karras et al. 2024, arXiv 2406.02507"),
    "domain_guidance": _reg(weak_branch_cfg, "C", "any", {"w": 1.5}, needs=("d_weak",),
                            caller="d_c from the fine-tuned model, d_weak = base model's null prediction (same latent "
                                   "space and parameterization)", cite="Zhong et al. 2025 arXiv 2504.01521; "
                                                                       "Phunyaphibarn et al. 2025 arXiv 2503.20240"),
    "icg": _reg(weak_branch_cfg, "C", "any", {"w": 7.5}, needs=("d_weak",),
                caller="d_weak = D(x_t; c_hat), c_hat = icg_random_condition(cond) or a random prompt, resampled per step",
                cite="Sadat et al. 2025, arXiv 2407.02687"),
    "tsg": _reg(weak_branch_cfg, "C", "any", {"w": 2.0}, needs=("d_weak",),
                caller="d_weak = D(x_t; c) with the time embedding perturbed by tsg_perturb_embedding",
                cite="Sadat et al. 2025, arXiv 2407.02687"),
    "dropout_self_guidance": _reg(weak_branch_cfg, "C", "any", {"w": 3.0}, needs=("d_weak",),
                                  caller="d_weak = D(x_t; c) in train mode with dropout p = 0.1",
                                  cite="Gu & Hou 2025, arXiv 2510.17136"),
    "sims": _reg(weak_branch_cfg, "C", "any", {"w": 1.8}, needs=("d_weak",),
                 caller="d_weak = a copy fine-tuned on the model's own samples (w = 1 + omega, omega 0.6-1.5)",
                 cite="Alemohammad et al. 2024, arXiv 2408.16333"),
    "sparse_guidance": _reg(weak_branch_cfg, "C", "any", {"w": 2.0}, needs=("d_weak",),
                            caller="d_c at token sparsity 0.1-0.2, d_weak at 0.7-0.8 (a token-sparse trained model)",
                            cite="Krause et al. 2026, arXiv 2601.01608"),
    "erg": _reg(weak_branch_cfg, "C", "any", {"w": 5.0}, needs=("d_weak",),
                caller="d_weak = D^xi(x_t; c^tau): image-attention temperature 0.01 in middle blocks (+ text-encoder "
                       "temperature)", cite="Ifriqi et al. 2025, arXiv 2504.13987"),
    "internal_guidance": _reg(weak_branch_cfg, "C", "any", {"w": 1.9}, needs=("d_weak",),
                              caller="d_weak = an intermediate output head trained with an auxiliary loss",
                              cite="Zhou et al. 2025, arXiv 2512.24176"),
    "personalization_guidance": _reg(weak_branch_cfg, "C", "any", {"w": 7.5}, needs=("d_weak",),
                                     caller="d_weak = D(x_t; u) with weights om theta_ft + (1 - om) theta_pre "
                                            "(LoRA at strength om 0-0.3)", cite="Park et al. 2025, arXiv 2508.00319"),
    "autolora": _reg(weak_branch_cfg, "C", "any", {"w": 1.5}, needs=("d_weak",),
                     caller="d_c = CFG of the LoRA model (w2 5-7), d_weak = CFG of the base (w1 5-7); w = gamma 1.5",
                     cite="Kasymov et al. 2024, arXiv 2410.03941"),
    "npo": _reg(weak_branch_cfg, "C", "any", {"w": 5.0}, needs=("d_weak",),
                caller="d_weak from weights theta + a eta + b delta (a LoRA trained on reversed preference pairs)",
                cite="Wang et al. 2025, arXiv 2505.11245"),
    "vpg": _reg(weak_branch_cfg, "G", "any", {"w": 2.3}, needs=("d_weak",),
                caller="AR visual model: d_weak = logits with 10% of prefix token embeddings shuffled within the scale",
                cite="Liao et al. 2026, arXiv 2605.30317"),
    "autoguidance_mixed": _reg(autoguidance_mixed, "C", "any", {"a": 0.5}, needs=("d_weak",),
                               caller="d_weak = conditional prediction of the weak model",
                               cite="Karras et al. 2024, arXiv 2406.02507"),
    "perturbation_guidance": _reg(perturbation_guidance, "C", "any", {"s": 3.0}, needs=("d_pert",),
                                  caller="any perturbed conditional pass", cite="shared form"),
    "pag": _reg(perturbation_guidance, "C", "any", {"s": 3.0}, needs=("d_pert",),
                caller=_PERT + "self-attention replaced by V (identity map) in the UNet mid-block attn1 (SD/SDXL)",
                cite="Ahn et al. 2024, arXiv 2403.17377"),
    "softpag": _reg(perturbation_guidance, "C", "any", {"s": 3.0}, needs=("d_pert",),
                    caller=_PERT + "chosen heads' maps interpolated (1-u)A + uI, u = 0.5 (HeadHunter search)",
                    cite="Ahn et al. 2025, arXiv 2506.10978"),
    "seg": _reg(perturbation_guidance, "C", "any", {"s": 2.8}, needs=("d_pert",),
                caller=_PERT + "queries blurred with a 2-D Gaussian (sigma 10 or inf) in the mid-block self-attention; "
                               "paper form perturbs the null: pass d_ref=d_u", cite="Hong 2024, arXiv 2408.00760"),
    "stg": _reg(perturbation_guidance, "C", "any", {"s": 1.0}, needs=("d_pert",),
                caller=_PERT + "one mid-to-late block skipped (STG-R, s 1) or its attention map set to identity (STG-A, s 2)",
                cite="Hyung et al. 2025, arXiv 2411.18664"),
    "slg": _reg(perturbation_guidance, "C", "any", {"s": 3.0}, needs=("d_pert",),
                caller=_PERT + "joint blocks [7, 8, 9] skipped (SD3.5-M), only for 1%-15/20% of the steps",
                cite="Stability AI SD3.5 reference implementation"),
    "s2_guidance": _reg(perturbation_guidance, "C", "any", {"s": 1.0}, needs=("d_pert",),
                        caller=_PERT + "a random ~10% of transformer blocks dropped (first blocks protected), central 80%",
                        cite="Chen et al. 2025, arXiv 2508.12880"),
    "tpg": _reg(perturbation_guidance, "C", "any", {"s": 3.0}, needs=("d_pert",),
                caller=_PERT + "hidden tokens randomly permuted at the inputs of chosen blocks (SDXL d6-d23)",
                cite="Rajabi et al. 2025, arXiv 2506.10036"),
    "ssg": _reg(perturbation_guidance, "C", "any", {"s": 3.0}, needs=("d_pert",),
                caller=_PERT + "the 10% least-similar token pairs (and channels) swapped at chosen blocks (SDXL u0-u5)",
                cite="Zhang et al. 2026, arXiv 2604.08048"),
    "asag": _reg(perturbation_guidance, "C", "any", {"s": 1.5}, needs=("d_pert",),
                 caller=_PERT + "self-attention replaced by an adversarial Sinkhorn plan (cost = QK^T, 2 iterations)",
                 cite="Kim 2025, arXiv 2511.07499"),
    "ma_dg": _reg(perturbation_guidance, "C", "any", {"s": 2.0}, needs=("d_pert",),
                  caller=_PERT + "massive-activation channels scaled by rho_t from block k onward",
                  cite="Gan et al. 2026, arXiv 2607.02968"),
    "self_guidance": _reg(perturbation_guidance, "C", "any", {"s": 3.0}, needs=("d_pert",),
                          caller="d_pert = D(x_t; c, t + delta), the same input told a slightly noisier level "
                                 "(delta = 1% of the time range); SG-prev reuses the previous step's prediction",
                          cite="Li et al. 2024, arXiv 2412.05827"),
    "spg": _reg(perturbation_guidance, "C", "x0", {"s": 0.2}, needs=("d_pert",),
                caller="d_pert = x0 prediction at x_t' = renoise(temporal moving average of x0_hat, fresh noise)",
                cite="Jeon 2025, arXiv 2503.02577"),
    "swg": _reg(perturbation_guidance, "C", "any", {"s": 0.5}, needs=("d_pert",),
                caller="d_pert = d_c - M (d_c - d_swg), d_swg and M from swg_weak_prediction",
                cite="Adaloglou et al. 2025, arXiv 2411.10257"),
    "sag": _reg(sag_guidance, "C", "x0", {"s": 0.5}, needs=("d_sag",),
                caller="d_sag = null prediction at the degraded input (sag_mask_from_attention + sag_degrade); "
                       "ComfyUI x0 form passes d_ref = x0_deg", cite="Hong et al. 2023, arXiv 2210.00939"),
    "stacked_guidance": _reg(stacked_guidance, "C", "any", {}, caller="terms = [(d_i, s_i), ...] perturbed passes",
                             cite="HaCohen et al. 2026 (LTX-2), arXiv 2601.03233"),
    # ---- family D (negatives / composition) ----
    "composable_not": _reg(composable_not, "D", "any", needs=("d_n",), node=True,
                           cite="Liu et al. 2022, arXiv 2206.01714"),
    "signed_guidance": _reg(signed_guidance, "D", "any", {"w_neg": 20.0}, needs=("d_n",), node=True,
                            cite="A1111 AND; Chang et al. 2025, arXiv 2510.26052"),
    "perp_neg": _reg(perp_neg, "D", "any", {"neg_scale": 1.0, "batch_scalar": False}, needs=("d_n",), node=True,
                     cite="Armandpour et al. 2023, arXiv 2304.04968"),
    "ccfg": _reg(contrastive_cfg, "D", "noise", {"w_neg": None, "tau": None}, needs=("d_n",), node=True,
                 cite="Chang et al. 2024, arXiv 2411.17077"),
    "sld": _reg(sld, "D", "noise", {"warmup": 10, "s_s": 1000.0, "lam": 0.01, "s_m": 0.3, "beta_m": 0.4},
                needs=("d_n",), node=True, state=lambda k: SLDState(),
                caller="d_n = the safety-concept prediction D(x_t; S)", cite="Schramowski et al. 2023, arXiv 2211.05105"),
    "sega": _reg(sega, "D", "noise", {"edit_scale": 5.0, "threshold": 0.9, "warmup": 10, "momentum_scale": 0.1,
                                      "momentum_beta": 0.4}, needs=("d_edits",), state=lambda k: SEGAState(),
                 caller="d_edits = one prediction per edit concept", cite="Brack et al. 2023, arXiv 2301.12247"),
    "dng": _reg(dng, "D", "noise", {"base": "cond"}, needs=("d_n",),
                state=lambda k: DNGState(), caller="state.update(x_k, mu_b, mu_n, s_k) after each ancestral step",
                cite="Koulischer et al. 2025, arXiv 2410.14398"),
    "concept_removal": _reg(concept_removal_guidance, "D", "noise", {"K": 1.0, "tau": 0.0, "lambda0": 2.0},
                            needs=("d_n",), state=lambda k: CRGState(),
                            caller="DDPM alphabar / alpha / beta of the current step", cite="Choi et al. 2026, arXiv 2606.29801"),
    # ---- family F (spatial / frequency) ----
    "samg": _reg(samg, "F", "noise", {"w_min": None, "w_max": None}, node=True, cite="Li et al. 2026, arXiv 2604.26503"),
    "fdg": _reg(fdg, "F", "x0", {"w_low": None, "levels": 2, "parallel_weight": 1.0, "formulation": "paper"},
                node=True, cite="Sadat et al. 2025, arXiv 2506.19713"),
    "fresca": _reg(fresca, "F", "any", {"scale_low": 1.0, "scale_high": 1.25, "freq_cutoff": 20}, node=True,
                   cite="Huang et al. 2025, arXiv 2504.02154"),
    "hiwave": _reg(hiwave, "F", "x0", {"w_low": 1.0, "wavelet": "sym4"}, node=True,
                   cite="Vontobel et al. 2025, arXiv 2506.20452"),
    "lf_cfg": _reg(lf_cfg, "F", "noise", {"rho": 0.5, "k": 8}, node=True, state=lambda k: LFCFGState(),
                   cite="Song & Lai 2025, arXiv 2506.21452"),
    "zeresfdg": _reg(zeresfdg, "F", "noise", {"mode": "auto"}, node=True, state=lambda k: ZeResFDGState(),
                     cite="Rychkovskiy 2025, arXiv 2510.12954", batch_coupled=True),
    "s_cfg": _reg(s_cfg, "F", "noise", {"clamp_min": 0.8, "clamp_max": 3.0, "cap": 15.0}, needs=("masks",),
                  caller="masks from s_cfg_region_masks(cross_attn, self_attn, ...) of the conditional pass",
                  cite="Shen et al. 2024, arXiv 2404.05384"),
    # ---- feature space (the caller hooks attention) ----
    "nag": _reg(nag, "F", "feature", {"scale": 5.0, "tau": 2.5, "alpha": 0.25}, kind="feature",
                caller="attention outputs Z+ (positive context) and Z- (negative context) in each guided layer",
                cite="Chen et al. 2025, arXiv 2505.21179"),
    "nasa": _reg(nasa, "D", "feature", {"alpha": 0.1}, kind="feature", caller="cross-attention outputs",
                 cite="Nguyen et al. 2024, arXiv 2412.02687"),
    "orthogonal_attention_negative": _reg(orthogonal_attention_negative, "D", "feature", {"alpha": 4.0},
                                          kind="feature", caller="image-to-text attention outputs per head",
                                          cite="Ko et al. 2026, arXiv 2605.29390"),
    "gag": _reg(gag, "F", "feature", {"lam": 10.0, "zeta": 0.0}, kind="feature",
                caller="entmax and softmax attention outputs of the same logits", cite="Kim 2026, arXiv 2603.02531"),
    "negative_token_merge": _reg(negative_token_merge, "D", "feature", {"alpha": 0.9, "tau": 0.65}, kind="feature",
                                 caller="block-output tokens of the sample and of the reference",
                                 cite="Singh et al. 2024, arXiv 2412.01339"),
    # ---- family G (logits) ----
    "lm_cfg": _reg(lm_cfg, "G", "logits", {"normalize": True}, kind="logits", cite="Sanchez et al. 2023, arXiv 2306.17806"),
    "context_aware_decoding": _reg(context_aware_decoding, "G", "logits", {"alpha": 0.5}, kind="logits",
                                   cite="Shi et al. 2023, arXiv 2305.14739"),
    "coherence_boost": _reg(coherence_boost, "G", "logits", {"a_k": -0.5}, kind="logits",
                            cite="Malkin et al. 2022, arXiv 2110.08294"),
    "adacad": _reg(adacad, "G", "logits", {}, kind="logits", cite="Wang et al. 2024, arXiv 2409.07394"),
    "contrastive_decoding": _reg(contrastive_decoding, "G", "logits", {"beta": 0.5, "alpha": 0.1, "form": "obrien"},
                                 kind="logits", cite="Li et al. 2022 arXiv 2210.15097; O'Brien & Lewis 2023 arXiv 2309.09117"),
    "vcd": _reg(contrastive_decoding, "G", "logits", {"beta": 1.0, "alpha": 0.1, "form": "obrien"}, kind="logits",
                caller="weak logits from the same VLM with a diffusion-noised image (T 500-999)",
                cite="Leng et al. 2023, arXiv 2311.16922"),
    "loopcd": _reg(contrastive_decoding, "G", "logits", {"beta": 0.3, "alpha": 0.1, "form": "scaled"}, kind="logits",
                   caller="weak logits read out after an early loop of a looped LM", cite="Yu et al. 2026, arXiv 2609.24196"),
    "dexperts": _reg(dexperts, "G", "logits", {"alpha": 2.0, "top_p": 0.9}, kind="logits",
                     cite="Liu et al. 2021, arXiv 2105.03023"),
    "igg": _reg(igg, "G", "logits", {"w_igg": 2.1}, kind="logits", cite="Nguyen et al. 2025, arXiv 2509.23876"),
    "normalized_masked_cfg": _reg(normalized_masked_cfg, "G", "logits", {}, kind="logits",
                                  cite="Rojas et al. 2025 arXiv 2507.08965; Schiff et al. 2024 arXiv 2412.10193"),
    "aram": _reg(aram, "G", "logits", {"lambda_max": 1.0, "beta": 0.1}, kind="logits",
                 cite="Kim & Ye 2026, arXiv 2603.17677"),
    "prob_space_cfg": _reg(prob_space_cfg, "G", "probs", {}, kind="logits", cite="guidance-space synthesis"),
}

VARIANT_NOTES: Dict[str, Dict[str, str]] = {
    "characteristic_guidance": {"reason": "needs 2 extra model evaluations per fixed-point iteration at shifted "
                                          "inputs; implemented with model callables", "cite": "arXiv 2312.07586",
                                "use": "characteristic_guidance(x_t, eps_c_fn, eps_u_fn, ...)"},
    "annealing_guidance_scale": {"reason": "the scale comes from a trained 52k-parameter MLP per base model",
                                 "cite": "Yehezkel et al. 2025, arXiv 2506.24108", "use": "cfg(...) with the MLP's w"},
    "dynamic_cfg_online_feedback": {"reason": "needs latent-space evaluators trained on the model's noisy latents",
                                    "cite": "Papalampidi et al. 2025, arXiv 2509.16131",
                                    "use": "score candidates cfg(d_c, d_u, w) for w in a grid"},
    "learn_to_guide": {"reason": "the weight is a trained network omega_phi(s, t, c)", "cite": "arXiv 2510.00815"},
    "adversarial_cfg_schedules": {"reason": "trained generator/discriminator pair per base model",
                                  "cite": "arXiv 2608.14038"},
    "prompt_aware_cfg": {"reason": "trained per-model scale predictor", "cite": "arXiv 2509.22728"},
    "info_theoretic_schedule": {"reason": "offline trajectory optimization with divergence estimates per model",
                                "cite": "arXiv 2606.24025", "use": "table_schedule(...) with the fitted values"},
    "linear_ag": {"reason": "the regressed null prediction needs an OLS fit on stored trajectories",
                  "cite": "Castillo et al., arXiv 2312.12487", "use": "adaptive_guidance for the switch part"},
    "compress_guidance": {"reason": "implemented as a schedule; its CFG form is reconstructed (the paper prints only "
                                    "the classifier-gradient form)", "cite": "arXiv 2408.11194",
                          "use": "compress_guidance_schedule"},
    "tv_cfg / c2fg / dg_cfg / wang schedules / cads / lig / cutoffs": {
        "reason": "weight schedules, not combiners: they produce w(t) for cfg or any combiner",
        "cite": "arXiv 2509.22007, 2603.08155, 2607.19725, 2404.13040, 2310.17347, 2404.07724",
        "use": "tv_cfg_schedule, c2fg_schedule, dg_cfg_table, wang_schedule, cads_dynamic_schedule, "
               "interval_schedule / with_interval, cutoff_schedule, with_schedule"},
    "cads_condition_annealing": {"reason": "acts on condition embeddings before the model",
                                 "cite": "arXiv 2310.17347", "use": "cads_anneal_condition"},
    "perturbed_attention_family": {"reason": "PAG, SoftPAG, SEG, SAG, STG, SLG, S2, TPG, SSG, ASAG, MA-DG, ERG, TSG, "
                                             "ICG need attention / block / embedding hooks to produce the weak pass; "
                                             "only the combination is pure",
                                   "cite": "see REGISTRY entries", "use": "perturbation_guidance / weak_branch_cfg"},
    "pladis": {"reason": "replaces cross-attention inside the network (single pass)", "cite": "arXiv 2503.07677",
               "use": "pladis_attention inside an attention processor"},
    "vsf": {"reason": "changes the attention inputs (negative tokens with flipped values)", "cite": "arXiv 2508.10931",
            "use": "vsf_attention inside an attention processor"},
    "safree": {"reason": "text-embedding projection with trigger detection plus feature-space Fourier re-attention "
                         "inside the UNet", "cite": "arXiv 2410.12761"},
    "ebca": {"reason": "context updates inside every cross-attention layer", "cite": "arXiv 2306.09869"},
    "diffusion_self_guidance": {"reason": "energies on attention maps and activations, gradients through the network",
                                "cite": "arXiv 2306.00986"},
    "attend_and_excite": {"reason": "attention-energy latent update needs autograd through the network",
                          "cite": "arXiv 2301.13826"},
    "layout_cross_attention_guidance": {"reason": "backward guidance through cross-attention maps (autograd)",
                                        "cite": "arXiv 2304.03373"},
    "freeu": {"reason": "rescales UNet decoder features and skips inside the network (both branches)",
              "cite": "arXiv 2309.11497"},
    "representation_guidance": {"reason": "needs a trained projector, an external encoder and autograd",
                                "cite": "arXiv 2601.22468"},
    "sage_moe": {"reason": "training-side subspace alignment loss", "cite": "arXiv 2609.34525"},
    "multidiffusion": {"reason": "fuses crop-wise sampler steps (a sampler, not a combiner)", "cite": "arXiv 2302.08113",
                       "use": "masked_cond_average for ComfyUI-style area conditioning"},
    "ledits_pp": {"reason": "needs an edit-friendly inversion and cross-attention masks", "cite": "arXiv 2311.16711",
                  "use": "sega for the guidance term"},
    "superdiff": {"reason": "needs a stochastic sampler with log-density tracking", "cite": "arXiv 2412.17762",
                  "use": "superdiff_mix, superdiff_kappa_and / _or, superdiff_update_logdens"},
    "fkc_rrr_mcmc": {"reason": "particle reweighting / MCMC correctors run inside the sampler; MALA/HMC need an "
                               "energy-parameterized model", "cite": "arXiv 2503.02819, 2302.11552",
                     "use": "fkc_log_weight_increment, systematic_resample, pcg_corrector (ULA)"},
    "cfgpp_family": {"reason": "sampler-step functions, not per-step combiners (the step needs the next sigma)",
                     "cite": "arXiv 2406.08070, 2510.07631, 2601.21892, 2607.07665",
                     "use": "cfgpp_step, cfgpp_ancestral_step, rectified_cfgpp_step, cfg_mp_step, terminal_repair_step"},
    "pcg": {"reason": "predictor-corrector sampler", "cite": "arXiv 2408.09000", "use": "ddim_step + pcg_corrector"},
    "cfgig": {"reason": "a sampling schedule: sample, re-noise to sigma*, re-denoise at a stronger w",
              "cite": "arXiv 2505.21101", "use": "renoise"},
    "restart_sampling": {"reason": "a sampler schedule of re-noising segments", "cite": "arXiv 2306.14878",
                         "use": "restart_renoise_flow / renoise"},
    "z_sampling_w2sd": {"reason": "needs an invertible step and extra model calls", "cite": "arXiv 2412.10891, 2502.00473",
                        "use": "zigzag_step(x, denoise_fn, invert_fn)"},
    "dpm_solver_pp": {"reason": "a standard solver (built into ComfyUI as dpmpp_2m etc.)", "cite": "arXiv 2211.01095"},
    "heavy_ball": {"reason": "momentum on the solver increment", "cite": "arXiv 2307.11118", "use": "heavy_ball"},
    "mpgd": {"reason": "loss-gradient guidance on x0; its CFG analogue is CFG++", "cite": "arXiv 2311.16424",
             "use": "cfgpp_step"},
    "dsg": {"reason": "a stochastic step constraint", "cite": "arXiv 2402.03201", "use": "dsg_step"},
    "reg": {"reason": "needs a vector-Jacobian product through the network", "cite": "arXiv 2501.18865"},
    "particle_guidance": {"reason": "couples a batch of samples; the DINO variant needs autograd",
                          "cite": "arXiv 2310.13102", "use": "particle_repulsion"},
    "guidance_distillation": {"reason": "training (w-conditioned student)", "cite": "arXiv 2210.03142"},
    "lcm": {"reason": "training (consistency distillation with a w embedding)", "cite": "arXiv 2310.04378"},
    "flux_embedded_guidance": {"reason": "the scale is a model input (guidance embedding); de-distillation is a "
                                         "fine-tune", "cite": "FLUX.1-dev model card; nyanko7 flux-dev-de-distill"},
    "agd_adapter_distillation": {"reason": "training", "cite": "arXiv 2503.07274"},
    "dmd_decoupled_dmd": {"reason": "training (CFG inside distribution-matching losses)", "cite": "arXiv 2511.22677"},
    "sid_lsg": {"reason": "training (data-free score identity distillation)", "cite": "arXiv 2406.01561"},
    "model_guidance_training": {"reason": "training objective", "cite": "arXiv 2502.12154",
                                "use": "model_guidance_target"},
    "guidance_free_training": {"reason": "training objective", "cite": "arXiv 2501.15420", "use": "gft_prediction"},
    "cca": {"reason": "training objective (AR models)", "cite": "arXiv 2410.09347", "use": "cca_loss"},
    "ddo": {"reason": "training objective with a frozen reference", "cite": "arXiv 2503.01103"},
    "mclr": {"reason": "training objective", "cite": "arXiv 2603.22364"},
    "contrastive_flow_matching": {"reason": "training objective", "cite": "arXiv 2506.05350",
                                  "use": "contrastive_fm_loss"},
    "coherence_aware_training": {"reason": "training with a per-sample coherence label", "cite": "arXiv 2405.20324"},
    "classifier_guidance": {"reason": "needs a noise-aware classifier and its gradient", "cite": "arXiv 2105.05233",
                            "use": "classifier_guidance(d, grad_log_p, ...)"},
    "discriminator_guidance": {"reason": "needs a trained discriminator and its gradient", "cite": "arXiv 2211.17091",
                               "use": "classifier_guidance with grad log(d/(1-d))"},
    "tfg": {"reason": "loss guidance with gradients through the denoiser and a target predictor",
            "cite": "arXiv 2409.15761"},
    "null_text_inversion": {"reason": "per-step optimization of the null embedding (autograd)", "cite": "arXiv 2211.09794"},
    "negative_prompt_inversion": {"reason": "input construction: the source prompt replaces the null",
                                  "cite": "arXiv 2305.16807", "use": "weak_branch_cfg with d_weak = D(x_t; c_src)"},
    "feature_self_guidance_diversity": {"reason": "feature-space dispersion with gradients at one block across a batch",
                                        "cite": "arXiv 2606.27371"},
    "dice_teefusion": {"reason": "training (guidance distilled into the text embedding)", "cite": "arXiv 2502.03726, 2507.18192"},
    "reneg": {"reason": "a learned negative embedding (reward training)", "cite": "arXiv 2412.19637",
                    "use": "cfg with the learned embedding as the null"},
    "esd_sliders": {"reason": "training (negative guidance distilled into weights / LoRAs)", "cite": "arXiv 2303.07345"},
    "divin": {"reason": "Langevin on the initial noise with gradients through the model", "cite": "arXiv 2606.02453"},
    "detail_daemon": {"reason": "changes the sigma passed to the model, not the combination",
                      "cite": "ComfyUI-Detail-Daemon", "use": "detail_daemon_sigma, lying_sigma"},
    "res4lyf_guides": {"reason": "sampler-integrated target-latent guides with implicit sub-steps", "cite": "RES4LYF"},
    "alg_low_pass_condition": {"reason": "modifies the conditioning image of an I2V model", "cite": "arXiv 2506.08456",
                               "use": "low_pass_latent"},
    "npc_vl_dnp": {"reason": "negatives written by a VLM/LLM during or before sampling",
                   "cite": "arXiv 2512.07702, 2510.26052", "use": "windowed_negative / signed_guidance"},
    "softcfg": {"reason": "scales the unconditional branch's KV cache", "cite": "arXiv 2510.00996",
                "use": "softcfg_value_weights"},
    "a_cfg": {"reason": "builds the unconditional input by re-masking tokens", "cite": "arXiv 2505.20199",
              "use": "acfg_remask + lm_cfg"},
    "uncage": {"reason": "changes the unmasking order from attention maps", "cite": "arXiv 2508.05399"},
    "infinity_layer_cfg": {"reason": "mixes hidden states at an inner block", "cite": "arXiv 2412.04431",
                           "use": "cfg on the hidden states"},
    "mgm_self_guidance": {"reason": "needs a trained smoothing module", "cite": "arXiv 2410.13136"},
    "vq_diffusion_learned_null": {"reason": "fine-tuned learnable null", "cite": "arXiv 2205.16007", "use": "lm_cfg"},
    "unsupervised_mdlm_cfg": {"reason": "input construction (prompt tokens masked)", "cite": "arXiv 2502.09992",
                              "use": "lm_cfg"},
    "ctmc_guidance": {"reason": "acts on rate matrices", "cite": "arXiv 2406.01572", "use": "ctmc_rate_guidance"},
    "ssg_scaled_spatial": {"reason": "acts across AR scales", "cite": "arXiv 2602.05534",
                           "use": "ssg_coarse_prior + ssg_guidance"},
    "sphere_ar": {"reason": "projection after a flow-head sample", "cite": "arXiv 2509.24335", "use": "sphere_project"},
    "rcfg_reward_weighted": {"reason": "needs attribute-conditional logits", "cite": "arXiv 2604.15577",
                             "use": "reward_weighted_cfg"},
    "dola": {"reason": "needs early-exit logits", "cite": "arXiv 2309.03883", "use": "dola"},
    "ga_eval": {"reason": "evaluation protocol", "cite": "arXiv 2602.22570", "use": "effective_guidance_scale"},
    "evaluation_suites": {"reason": "benchmarks (VBench, DrawBench, PartiPrompts, HPDv2, Pick-a-Pic, paired bootstrap)",
                          "cite": "arXiv 2311.17982, 2205.11487, 2206.10789, 2306.09341, 2305.01569, 2608.16786"},
    "cfg_theory": {"reason": "theory (what CFG samples; stage-wise dynamics)", "cite": "arXiv 2409.13074"},
}


def apply_variant(name: str, d_c, d_u, w=None, ctx: Optional[Dict[str, Any]] = None, **knobs):
    """Call REGISTRY[name] with its defaults, the given knobs, and any keyword it accepts from ctx.

    ctx keys used when the combiner accepts them: x_t, sigma, alpha, t, progress, step, num_steps, sigmas,
    state, space, d_n, d_weak, d_pert, x0_c, x0_u, n_c, n_u, flow. Knobs override defaults; unknown knob
    names raise TypeError. State-only knobs (REGISTRY[name]['state_knobs']) are not passed to the function."""
    spec = REGISTRY[name]
    fn = spec["fn"]
    params = inspect.signature(fn).parameters
    names = list(params)
    allowed = set(names[2:]) | set(spec.get("state_knobs", ()))
    bad = [k for k in knobs if k not in allowed]
    if bad:
        raise TypeError(f"{name}: unknown knob(s) {bad}; allowed: {sorted(allowed)}")
    merged = dict(spec["defaults"])
    merged.update(knobs)
    kwargs = {k: v for k, v in merged.items() if k in params}
    for p in names[2:]:
        if p not in kwargs and ctx and p in ctx and p != "w":
            kwargs[p] = ctx[p]
    if w is not None and "w" in params:
        kwargs["w"] = w
    return fn(d_c, d_u, **kwargs)


def make_state(name: str, **knobs):
    """A fresh state object for a stateful registry entry (None if it has none)."""
    spec = REGISTRY[name]
    if spec.get("state") is None:
        return None
    merged = dict(spec["defaults"])
    merged.update(knobs)
    return spec["state"](merged)


def describe(name: str) -> str:
    """One paragraph about a registry entry or a VARIANT_NOTES entry."""
    if name in REGISTRY:
        spec = REGISTRY[name]
        doc = (inspect.getdoc(spec["fn"]) or "").split("\n\n")[0]
        parts = [f"{name} [{spec['space']}] family {spec['family']}: {doc}", f"defaults: {spec['defaults']}"]
        if spec.get("needs"):
            parts.append(f"needs: {spec['needs']}")
        if spec.get("caller"):
            parts.append(f"caller: {spec['caller']}")
        parts.append(f"cite: {spec['cite']}")
        return "\n".join(parts)
    if name in VARIANT_NOTES:
        n = VARIANT_NOTES[name]
        return f"{name}: not a pure combiner - {n['reason']} ({n.get('cite', '')}). use: {n.get('use', '-')}"
    raise KeyError(name)
