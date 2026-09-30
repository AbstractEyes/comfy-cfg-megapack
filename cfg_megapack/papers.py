"""One node per paper: each writes one stage of the model's guidance plan with the paper's own knobs and defaults.

The paper nodes and the stage nodes share one engine, so they chain freely: a paper node of the combine stage
replaces an earlier combine node, a schedule paper replaces an earlier When node, and so on. This module holds the
specifications only (plain data, importable without ComfyUI); nodes.py turns them into node classes."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

from . import engine
from . import cfg_variants as cv


@dataclass(frozen=True)
class Knob:
    name: str                                   # the input's name on the node
    kind: str                                   # float | int | bool | combo
    default: Any
    lo: float = 0.0
    hi: float = 1.0
    step: float = 0.01
    options: Tuple[str, ...] = ()
    tip: str = ""
    lib: str = ""                               # the library keyword, when it differs from the name
    convert: Optional[Callable[[Any], Any]] = None


@dataclass(frozen=True)
class Paper:
    key: str                                    # node id CFGP_<key>
    title: str                                  # display name, the paper's method name first
    cite: str                                   # authors, year, venue
    link: str                                   # the arXiv abstract (or the source repository for community methods)
    stage: str                                  # combine | when | weak | guider
    line: str                                   # the research line (menu folder and README grouping)
    summary: str                                # what it does, in plain words, with the rule written out
    registry: str = ""                          # the library combiner (combine and guider papers)
    knobs: Tuple[Knob, ...] = ()
    build: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None     # when / weak papers: the stage entry
    flow_note: str = ""
    catalog: str = ""                           # the catalog entry id(s)
    space: str = "auto (the method's own)"      # the node's default space (combine and guider papers)


def _none_if_negative(v):
    return None if float(v) < 0 else float(v)


def _radians(v):
    return math.radians(float(v))


def _arxiv(i: str) -> str:
    return f"https://arxiv.org/abs/{i}"


LINES = {
    "combine": "combining the two predictions",
    "when": "when to guide",
    "weak": "weak branch (a degraded pass of the model itself)",
    "negatives": "negative prompts",
    "frequency": "frequency and space",
}

# ---------------------------------------------------------------------------------------------------------------
# combine: library combiners, one per paper
# ---------------------------------------------------------------------------------------------------------------

F, I, B, C = "float", "int", "bool", "combo"

PAPERS: Tuple[Paper, ...] = (
    Paper("CFG", "CFG: Classifier-Free Guidance (Ho & Salimans 2022)", "Ho & Salimans, NeurIPS 2021 workshop / arXiv 2022",
          _arxiv("2207.12598"), "combine", "combine",
          "The baseline every other node reshapes: u + w (c - u), the conditional prediction pushed away from the "
          "unconditional one by the scale w.", registry="cfg", catalog="A1"),
    Paper("RescaleCFG", "Guidance Rescale (Lin et al. 2024)", "Lin, Liu, Li & Yang, WACV 2024",
          _arxiv("2305.08891"), "combine", "combine",
          "Plain CFG, then its per-image standard deviation is pulled back to the conditional prediction's: "
          "x = phi * cfg * std(c) / std(cfg) + (1 - phi) * cfg. Fixes over-exposure at high scales.",
          registry="rescale_cfg", catalog="A2",
          knobs=(Knob("phi", F, 0.7, 0.0, 1.0, 0.01, tip="How much of the rescaled result is used (paper 0.5-0.75; 0 = plain CFG)."),)),
    Paper("DynamicThreshold", "Dynamic Thresholding (Imagen; Saharia et al. 2022)", "Saharia et al., NeurIPS 2022",
          _arxiv("2205.11487"), "combine", "combine",
          "Plain CFG on the denoised image, then each image is clipped to its p-th percentile of absolute values "
          "(at least s_max) and divided by it, which keeps pixel values in range at high scales.",
          registry="dynamic_threshold_cfg", catalog="A3",
          knobs=(Knob("p", F, 0.995, 0.5, 1.0, 0.001, tip="The percentile that sets each image's clip value."),
                 Knob("s_max", F, 1.0, 1.0, 20.0, 0.1, tip="The clip value never goes below this (1 = the static clip)."))),
    Paper("MimicScale", "Mimic-Scale Thresholding (mcmonkey 2023)", "mcmonkey4eva, sd-dynamic-thresholding (community)",
          "https://github.com/mcmonkeyprojects/sd-dynamic-thresholding", "combine", "combine",
          "Runs CFG at a high real scale (the sampler's cfg, 15-30) and rescales it per channel so its spread "
          "matches CFG at the low mimic scale: the prompt adherence of a high scale with the colors of a low one.",
          registry="mimic_cfg", catalog="A4",
          knobs=(Knob("mimic_scale", F, 7.0, 1.0, 30.0, 0.1, tip="The scale whose value range is imitated."),
                 Knob("threshold_percentile", F, 1.0, 0.5, 1.0, 0.001, tip="Percentile of the high-scale values used as their range (1 = the maximum)."),
                 Knob("separate_feature_channels", B, True, tip="Measure each channel on its own (off: one value per batch)."),
                 Knob("scaling_startpoint", C, "MEAN", options=("MEAN", "ZERO"), tip="Rescale around each channel's mean, or around zero."),
                 Knob("variability_measure", C, "AD", options=("AD", "STD"), tip="Measure the range by absolute deviation (clamped) or standard deviation."),
                 Knob("interpolate_phi", F, 1.0, 0.0, 1.0, 0.01, tip="Blend with plain high-scale CFG (1 = fully rescaled)."))),
    Paper("APG", "APG: Adaptive Projected Guidance (Sadat et al. 2025)", "Sadat, Hilliges & Weber, ICLR 2025",
          _arxiv("2410.02416"), "combine", "combine",
          "Splits the guidance difference g = c - u into its part along c and the rest, keeps the rest, "
          "down-weights the part along c (eta), caps the norm of g and adds reverse momentum: high scales without "
          "over-saturation. Paper form: c + (w - 1) (g_perp + eta g_par).",
          registry="apg", catalog="A5",
          knobs=(Knob("eta", F, 0.0, 0.0, 1.0, 0.01, tip="Weight of the part along c (1 with no cap and no momentum = plain CFG)."),
                 Knob("norm_threshold", F, 15.0, 0.0, 100.0, 0.5, tip="Cap on the norm of g (0 = no cap; choose near its typical size)."),
                 Knob("momentum", F, -0.5, -1.0, 1.0, 0.05, tip="Reverse momentum on g (paper SDXL -0.5; 0 = off)."),
                 Knob("formulation", C, "paper", options=("paper", "comfyui", "diffusers"), tip="paper: c + (w-1) g'; comfyui: c + w g'; diffusers: u + w g'."))),
    Paper("CFGZeroStar", "CFG-Zero* (Fan et al. 2025)", "Fan, Zheng, Yeh & Liu, arXiv 2025",
          _arxiv("2503.18886"), "combine", "combine",
          "Scales the unconditional prediction to best fit the conditional one first, s* = <c, u> / ||u||^2, then "
          "s* u + w (c - s* u); the first steps can leave the latent unmoved (zero-init). Built for flow models.",
          registry="cfg_zero_star", catalog="A6, B7",
          knobs=(Knob("zero_init_steps", I, 1, 0, 10, 1, tip="The first N steps leave the latent unmoved (0 = off)."),)),
    Paper("TCFG", "TCFG: Tangential Damping CFG (Kwon et al. 2025)", "Kwon, Kim, Jeong, Hsiao & Uh, CVPR 2025",
          _arxiv("2503.18137"), "combine", "combine",
          "Removes the part of the unconditional prediction that lies off the main direction shared by both "
          "predictions (a rank-one projection from their 2 x N matrix), then plain CFG.", registry="tcfg", catalog="A7"),
    Paper("BetaCFG", "beta-CFG (Malarz et al. 2025)", "Malarz, Kasymov, Zieba, Tabor & Spurek, ECAI 2025",
          _arxiv("2502.10574"), "combine", "combine",
          "A Beta-shaped schedule over the run with the guidance difference normalized: "
          "u + w * Beta(p; a, b) * (c - u) / ||c - u||^gamma. Re-tune the scale per model.",
          registry="beta_cfg", catalog="A9",
          knobs=(Knob("a", F, 2.0, 0.5, 10.0, 0.1, tip="Beta shape a (paper 2; 3 for scales of 5 and more)."),
                 Knob("b", F, 2.0, 0.5, 10.0, 0.1, tip="Beta shape b."),
                 Knob("gamma", F, 1.0, 0.0, 2.0, 0.05, tip="Power of the norm the difference is divided by (0 = no normalization)."),
                 Knob("peak_normalize", B, False, tip="Divide the Beta curve by its peak, so the scale never exceeds w."))),
    Paper("ADG", "ADG: Angle Domain Guidance (Jin et al. 2025)", "Jin, Xiao, Liu & Gu, ICML 2025",
          _arxiv("2506.11039"), "combine", "combine",
          "Guides by angle instead of length: the denoised prediction turns away from the unconditional one by "
          "(w - 1) times their angle, capped at a maximum, which bounds the norm of the result.",
          registry="adg", catalog="A10",
          knobs=(Knob("max_angle_degrees", F, 60.0, 1.0, 180.0, 1.0, lib="max_angle", convert=_radians,
                      tip="Largest turn away from the conditional prediction (paper 60)."),)),
    Paper("PowerLawCFG", "Power-Law CFG (Lehman Pavasovic et al. 2025)", "Lehman Pavasovic, Verbeek, Biroli & Mezard, arXiv 2025",
          _arxiv("2502.07849"), "combine", "combine",
          "The scale grows with the size of the difference: c + omega ||c - u||^alpha (c - u). Re-tune omega per "
          "model and resolution.", registry="power_law_cfg", catalog="A11",
          knobs=(Knob("alpha", F, 0.9, 0.0, 3.0, 0.01, tip="Exponent on the difference's norm (0 = plain CFG)."),
                 Knob("omega", F, -1.0, -1.0, 100.0, 0.01, convert=_none_if_negative, tip="The coefficient (-1 = w - 1)."))),
    Paper("EPCFG", "EP-CFG: Energy-Preserving CFG (Zhang et al. 2024)", "Zhang, Luan, Bi & Zhang, arXiv 2024",
          _arxiv("2412.09966"), "combine", "combine",
          "Plain CFG rescaled so its energy (sum of squares) equals the conditional prediction's; the robust form "
          "counts only the middle percentiles of the squared values.", registry="ep_cfg", catalog="A12",
          knobs=(Knob("robust", B, True, tip="Measure energy between two percentiles only (paper)."),
                 Knob("lo", F, 45.0, 0.0, 100.0, 1.0, tip="Lower percentile of the robust energy."),
                 Knob("hi", F, 55.0, 0.0, 100.0, 1.0, tip="Upper percentile of the robust energy."))),
    Paper("CFGRenorm", "CFG-Renorm (Qin et al. 2025; Lumina-Image 2.0)", "Qin et al., arXiv 2025 (after STIV)",
          _arxiv("2503.21758"), "combine", "combine",
          "Plain CFG, then its norm is capped at rho times the conditional prediction's norm.",
          registry="cfg_renorm", catalog="A13",
          knobs=(Knob("rho", F, 1.0, 0.0, 5.0, 0.05, tip="The cap as a multiple of ||c|| (0 = off; 1-1.5 used)."),)),
    Paper("CFGNorm", "CFGNorm: per-pixel norm matching (Qwen-Image)", "Qwen-Image pipeline; ComfyUI CFGNorm (no paper)",
          "https://github.com/QwenLM/Qwen-Image", "combine", "combine",
          "Plain CFG with each pixel's channel vector rescaled to the conditional prediction's length (match) or "
          "only shortened when longer (attenuate).", registry="cfg_norm_per_token", catalog="A14",
          knobs=(Knob("strength", F, 1.0, 0.0, 2.0, 0.01, tip="Blend of the rescaled and the plain result."),
                 Knob("mode", C, "match", options=("match", "attenuate"), tip="match (Qwen-Image) or attenuate (ComfyUI CFGNorm)."))),
    Paper("MAMBOG", "MAMBO-G: magnitude-aware guidance damping (Zhu et al. 2025)", "Zhu et al., arXiv 2025",
          _arxiv("2508.03442"), "combine", "combine",
          "The scale shrinks where the difference is large relative to the unconditional prediction: "
          "w_eff = 1 + (w - 1) exp(-alpha ||c - u|| / ||u||).", registry="mambo_g", catalog="A16",
          knobs=(Knob("alpha", F, 8.0, 0.0, 50.0, 0.1, tip="Damping strength (0 = plain CFG)."),)),
    Paper("SkimmedCFG", "Skimmed CFG (Extraltodeus)", "Extraltodeus, Skimmed_CFG (community)",
          "https://github.com/Extraltodeus/Skimmed_CFG", "combine", "combine",
          "Where guidance would push a value past both predictions in the same direction, that value is pulled "
          "back to what a lower skimming scale would give; lets high scales run without burning.",
          registry="skimmed_cfg", catalog="A17",
          knobs=(Knob("skimming_scale", F, 7.0, 0.0, 30.0, 0.1, tip="The scale the flagged values fall back to."),
                 Knob("full_skim_negative", B, False, tip="Skim the negative side completely."),
                 Knob("disable_flipping_filter", B, False, tip="Drop the check against the current latent."))),
    Paper("AutomaticCFG", "Automatic CFG (Extraltodeus)", "Extraltodeus, ComfyUI-AutomaticCFG (community)",
          "https://github.com/Extraltodeus/ComfyUI-AutomaticCFG", "combine", "combine",
          "Picks a scale per channel so each channel's guided range lands on a target set by the reference scale.",
          registry="automatic_cfg", catalog="A18",
          knobs=(Knob("reference_scale", F, 8.0, 0.0, 30.0, 0.1, tip="The scale whose range is targeted (0 = the sampler's cfg)."),
                 Knob("top_k", F, 0.25, 0.01, 1.0, 0.01, tip="Share of values averaged for the range."),
                 Knob("mode", C, "hard", options=("hard", "soft", "hard_squared", "range"), tip="How the range is measured."))),
    Paper("Mahiro", "Mahiro: positive-biased guidance (ComfyUI)", "ComfyUI Mahiro node, PR 5975 (community)",
          "https://github.com/comfyanonymous/ComfyUI/pull/5975", "combine", "combine",
          "Blends CFG toward the scaled conditional prediction by how similar the two are (a batch-wide cosine on "
          "signed square roots).", registry="mahiro", catalog="A19"),
    Paper("ReinhardCFG", "Reinhard tonemap of the guidance (ComfyUI)", "comfyanonymous, LatentOperationTonemapReinhard (community)",
          "https://github.com/comfyanonymous/ComfyUI", "combine", "combine",
          "Each pixel's guidance difference is tone-mapped with the Reinhard curve m / (m + 1) against a per-image "
          "ceiling, so large differences saturate smoothly.", registry="reinhard_cfg", catalog="A20",
          knobs=(Knob("multiplier", F, 1.0, 0.1, 100.0, 0.1, tip="Raises the ceiling (large = plain CFG)."),)),
    Paper("SMCCFG", "SMC-CFG: sliding-mode control CFG (Wang et al. 2026)", "Wang, Liu, Chi, Liu, Xue & Duan, CVPR 2026",
          _arxiv("2603.03281"), "combine", "combine",
          "Treats the guidance difference as a controlled signal: a sliding surface on its change between steps "
          "adds a bounded correction, -k sign(s).", registry="smc_cfg", catalog="A21",
          knobs=(Knob("lam", F, 5.0, 0.0, 20.0, 0.1, tip="Sliding-surface slope."),
                 Knob("k", F, -1.0, -1.0, 2.0, 0.001, convert=_none_if_negative,
                      tip="Correction size per element (0 = plain CFG; -1 = auto: the paper's 0.2 on flow models, "
                          "0.01 on noise-prediction models such as SDXL, where the paper's value wrecks the image)."),
                 Knob("switching", C, "sign", options=("sign", "unit"), tip="sign (paper) or unit vector (the ComfyUI node's form)."))),
    Paper("PMCCFG", "PMC-CFG: posterior-mean capped CFG (Peng & Ma 2026)", "Peng & Ma, arXiv 2026",
          _arxiv("2609.24287"), "combine", "combine",
          "Picks per image the largest guidance step along c - u that keeps the denoised prediction within "
          "gamma_cap times the conditional one's norm.", registry="pmc_cfg", catalog="A22",
          knobs=(Knob("gamma_cap", F, 1.05, 1.0, 2.0, 0.01, tip="The norm cap (1.05-1.15; large = plain CFG)."),)),
    Paper("AdaMaG", "AdaMaG: adaptive manifold guidance (Esmati et al. 2026)", "Esmati, Hyung, Dadashzadeh, Choo & Mirmehdi, arXiv 2026",
          _arxiv("2605.20079"), "combine", "combine",
          "Splits the guidance difference along the conditional noise estimate, keeps a little of that part "
          "(beta) and all of the rest, with a scale that falls as the noise level falls: max(w_min, w t^gamma).",
          registry="adamag", catalog="A23",
          knobs=(Knob("beta", F, 0.1, 0.0, 1.0, 0.01, tip="Weight of the part along the noise estimate."),
                 Knob("gamma", F, 4.0, 0.0, 10.0, 0.1, tip="How fast the scale falls with the noise level."),
                 Knob("w_min", F, 1.0, 0.0, 10.0, 0.1, tip="Lowest scale."))),
    Paper("FBG", "FBG: Feedback Guidance (Koulischer et al. 2025)", "Koulischer, Handke, Deleu, Demeester & Ambrogioni, NeurIPS 2025",
          _arxiv("2506.06085"), "combine", "combine",
          "Sets its own scale every step from a running estimate of how well the sample already fits the prompt "
          "(a posterior updated from the last step): strong while the fit is poor, near 1 once it is good. The "
          "sampler's cfg is used only with hybrid on.", registry="fbg", catalog="A24",
          knobs=(Knob("hybrid", B, False, tip="Add the sampler's plain CFG (w - 1) on top (FBG + CFG)."),
                 Knob("pi", F, 0.85, 0.5, 0.999, 0.005,
                      tip="Prior that the sample is conditional (the paper's Stable Diffusion images 0.85; its code's default 0.95)."),
                 Knob("t0", F, 0.75, 0.0, 1.0, 0.01,
                      tip="Noise level where the scale should reach its target (paper SD images 0.75; code 0.5)."),
                 Knob("t1", F, 0.5, 0.0, 1.0, 0.01, tip="Second calibration point (paper SD images 0.5; code 0.4)."),
                 Knob("lambda_max", F, 10.0, 1.0, 50.0, 0.5, tip="Highest scale."))),
    Paper("VAGS", "VAGS: velocity-adaptive guidance scale (Luo et al. 2026)", "Luo, Aidara, Lu, Moebel, Han & Wang, arXiv 2026",
          _arxiv("2605.15661"), "combine", "combine",
          "Scales w by exp(kappa (2 s - 1) cos(u, c)), s the signal level: more guidance where the predictions agree "
          "late in the run, less early.", registry="vags", catalog="A25",
          knobs=(Knob("kappa", F, 1.0, 0.0, 5.0, 0.05, tip="Strength of the adaptation (0 = plain CFG)."),)),
    Paper("CFGOEC", "CFG-OEC: orthogonal error correction (Yang et al. 2025)", "Yang, Lee & Han, arXiv 2025",
          _arxiv("2511.14075"), "combine", "combine",
          "Corrects the unconditional prediction with the part of its step-to-step error that is orthogonal to the "
          "conditional one's, when the two errors disagree (cosine below tau).", registry="cfg_oec", catalog="A28",
          knobs=(Knob("tau", F, 0.5, -1.0, 1.0, 0.01, tip="Correct when the errors' cosine is below this (-1 = plain CFG; the paper gives no value)."),)),
    Paper("HiGS", "HiGS: history-guided sampling (Sadat et al. 2025)", "Sadat, Salehi & Weber, arXiv 2025",
          _arxiv("2509.22300"), "combine", "combine",
          "Adds the high-frequency part of the difference between this step's guided prediction and an average "
          "of the past ones, in the middle of the run: sharper detail without extra passes.",
          registry="higs", catalog="C28",
          knobs=(Knob("w_h", F, 2.0, 0.0, 5.0, 0.05, tip="Strength of the history term."),
                 Knob("alpha", F, 0.75, 0.01, 1.0, 0.01, tip="Averaging weight of the history."),
                 Knob("t_min", F, 0.4, 0.0, 1.0, 0.01, tip="Active above this noise level."),
                 Knob("t_max", F, 0.95, 0.0, 1.0, 0.01, tip="Active below this noise level."),
                 Knob("eta", F, 1.0, 0.0, 1.0, 0.01, tip="Weight of the history term's part along the prediction."),
                 Knob("cutoff", F, 0.05, 0.0, 1.0, 0.01, tip="High-pass cutoff (share of the spectrum)."))),
    Paper("TSR", "TSR: Temporal Score Rescaling (Xu et al. 2025)", "Xu, Wu, Park, Zhou & Tulsiani, ICML 2026",
          _arxiv("2510.01184"), "combine", "combine",
          "Plain CFG, then the noise estimate is scaled by r = (eta s^2 + 1) / (eta s^2 / k + 1), eta the "
          "signal-to-noise ratio: k < 1 sharpens toward the dominant modes.", registry="tsr", catalog="A29",
          knobs=(Knob("k", F, 0.95, 0.5, 1.0, 0.005, tip="1 = off (paper SD3 / FLUX 0.93)."),
                 Knob("tsr_sigma", F, 1.0, 0.1, 10.0, 0.1, tip="The rescaling's own sigma (paper SD3 / FLUX 3)."))),
    Paper("EpsilonScaling", "Epsilon Scaling (Ning et al. 2024)", "Ning, Li, Su, Salah & Ertugrul, ICLR 2024",
          _arxiv("2308.15321"), "combine", "combine",
          "Plain CFG, then the noise estimate is divided by a factor slightly above 1 (exposure-bias correction).",
          registry="epsilon_scaling", catalog="E25",
          knobs=(Knob("factor", F, 1.005, 0.9, 1.1, 0.001, tip="The divisor (1 = off)."),)),
    # when to guide (state-switching combiners)
    Paper("AdaptiveGuidance", "Adaptive Guidance (Castillo et al. 2023)", "Castillo et al., AAAI 2025",
          _arxiv("2312.12487"), "combine", "when",
          "Full CFG until the two predictions agree (cosine above the threshold), then the conditional prediction "
          "alone for the rest of the run. The node reads the cosine of the denoised predictions, as ComfyUI's "
          "community node for this method does: SDXL's two noise predictions differ by about 1% of their size, so "
          "their cosine starts above 0.9999 and the paper's reading would stop guidance at the first step.",
          registry="adaptive_guidance", catalog="B4", space="denoised (x0)",
          knobs=(Knob("threshold", F, 0.999, 0.9, 1.0, 0.0005,
                      tip="Cosine of the denoised predictions where guidance stops. Measured over 50 steps: 0.999 "
                          "stops it after 18 steps on SDXL (cfg 7) and 25 on Anima (cfg 4.5); the paper's 0.991 "
                          "(read on its own models, about half of 20 steps) would stop it after 8 and 13 here."),)),
    Paper("TransitionPoint", "Transition-Point Guidance (Jain et al. 2024)", "Jain et al., CVPR 2025",
          _arxiv("2411.16738"), "combine", "when",
          "No guidance (or opposite guidance) until the difference between the predictions passes a local minimum, "
          "full CFG after it; against memorized images.", registry="transition_point", catalog="B19",
          knobs=(Knob("opposite", F, 0.0, 0.0, 10.0, 0.1, tip="Opposite guidance before the transition (0 = the unconditional prediction)."),)),
)

# ---------------------------------------------------------------------------------------------------------------
# when: schedules and windows (the When stage)
# ---------------------------------------------------------------------------------------------------------------


def _when(shape="constant", start=0.0, end=1.0, outside="cond", outside_scale=1.0, floor=0.0, a=-1.0, b=-1.0,
          skip_uncond=True, sigma_edm=None):
    spec = {"shape": shape, "start": float(start), "end": float(end), "outside": outside,
            "outside_scale": float(outside_scale), "floor": float(floor), "a": float(a), "b": float(b),
            "skip_uncond": bool(skip_uncond)}
    if sigma_edm is not None:
        spec["sigma_edm"] = sigma_edm
    return spec


WANG_SHAPES = {"linear up": "linear_up", "cosine up": "cosine_up", "linear down": "linear_down",
               "cosine down": "cosine_down", "V shape": "v_shape", "Lambda shape": "lambda_shape"}

PAPERS += (
    Paper("GuidanceInterval", "Guidance Interval (Kynkaanniemi et al. 2024)",
          "Kynkaanniemi, Aittala, Karras, Laine, Aila & Lehtinen, NeurIPS 2024", _arxiv("2404.07724"), "when", "when",
          "Guidance only at middle noise levels, sigma_low < sigma <= sigma_high (EDM units); the conditional "
          "prediction alone elsewhere, where the unconditional pass is skipped. Paper SD-XL: (0.28, 5.42] with "
          "cfg up to 16. On flow models the bounds become noise levels sigma / (1 + sigma).",
          knobs=(Knob("sigma_low", F, 0.28, 0.0, 100.0, 0.01, tip="Lower bound (EDM sigma)."),
                 Knob("sigma_high", F, 5.42, 0.0, 1000.0, 0.01, tip="Upper bound (EDM sigma; 1000 = no upper bound).")),
          build=lambda k: _when(sigma_edm=(float(k["sigma_low"]),
                                           float("inf") if float(k["sigma_high"]) >= 1000 else float(k["sigma_high"]))),
          catalog="B1"),
    Paper("WangSchedules", "Increasing guidance schedules (Wang et al. 2024)",
          "Wang, Dufour, Andreou, Cani, Fernandez Abrevaya, Picard & Kalogeiton, arXiv 2024", _arxiv("2404.13040"),
          "when", "when",
          "The scale changes over the run while its average stays the sampler's cfg: rising schedules (linear, "
          "cosine) worked best, with a floor (paper SDXL 4).",
          knobs=(Knob("shape", C, "linear up", options=tuple(WANG_SHAPES), tip="The schedule's shape."),
                 Knob("floor", F, 4.0, 0.0, 30.0, 0.1, tip="The scale never falls below this (paper: SDXL 4, SD1.5 2).")),
          build=lambda k: _when(shape=WANG_SHAPES[k["shape"]], floor=k["floor"]), catalog="B2"),
    Paper("TVCFG", "TV-CFG: stage-wise guidance (Jin et al. 2025)", "Jin, Shi & Gu, ICLR 2026",
          _arxiv("2509.22007"), "when", "when",
          "A triangular schedule peaking mid-run (from 1 up to 2w - 1 and back), normalized on the sampler's own "
          "steps so the time-average stays w.",
          knobs=(Knob("peak", F, 0.5, 0.05, 0.95, 0.01, tip="Where the peak sits (share of the run; 0.4-0.6 equally good)."),),
          build=lambda k: _when(shape="tv_cfg", a=k["peak"]), catalog="B9"),
    Paper("C2FG", "C2FG: exponentially growing guidance (Gao et al. 2026)", "Gao et al., arXiv 2026",
          _arxiv("2603.08155"), "when", "when",
          "The scale grows exponentially over the run: w exp(rate (1 - t)), t the noise level.",
          knobs=(Knob("rate", F, 0.693, 0.0, 3.0, 0.01, tip="Growth rate (ln 2 doubles the scale by the end; 0.2 on SD1.5 / SD3.5)."),),
          build=lambda k: _when(shape="c2fg_exp", a=k["rate"]), catalog="B10"),
    Paper("EarlyHighLateUncond", "Early-high, late-unconditional (Ventura et al. 2026)",
          "Ventura, Achilli, Ambrogioni & Lucibello, arXiv 2026", _arxiv("2602.00716"), "when", "when",
          "Full guidance early, then a scale of 0 (the unconditional prediction alone) late, which the paper "
          "finds reduces distortions. The switch is ComfyUI's percent of the run (by noise level).",
          knobs=(Knob("switch", F, 0.5, 0.05, 0.95, 0.01, tip="Where guidance switches off (paper 50-70% of the steps)."),
                 Knob("late_scale", F, 0.0, -1.0, 1.0, 0.05, tip="The scale after the switch (0 = unconditional, 1 = conditional).")),
          build=lambda k: _when(end=k["switch"], outside="fixed", outside_scale=k["late_scale"]), catalog="B20"),
    Paper("CFGTruncation", "CFG Truncation (Yi et al. 2024; Lumina-Image 2.0)", "Yi, Li, Xin & Li, NeurIPS 2024",
          _arxiv("2405.15330"), "when", "when",
          "Guidance only in the first part of the run, the conditional prediction alone after it (the "
          "unconditional pass is skipped there).",
          knobs=(Knob("ratio", F, 0.25, 0.01, 1.0, 0.01, tip="Share of the run with guidance (Lumina 0.25; 0.2-0.6 useful)."),),
          build=lambda k: _when(end=k["ratio"]), catalog="B5"),
)

# ---------------------------------------------------------------------------------------------------------------
# weak branch: a degraded pass of the model itself
# ---------------------------------------------------------------------------------------------------------------

BLOCK_TIP = ("Which self-attention blocks are perturbed (SDXL names; the middle block exists on SD1.5 too). On Anima "
             "and other Cosmos-Predict2 transformers: middle = the two middle blocks, deep input = the first third, "
             "deep output = the last third.")


def _weak(method, k):
    return {"method": method, "scale": float(k["scale"]), "mode": "add",
            "blocks": list(engine.BLOCK_PRESETS[k["blocks"]]), "blocks_label": k["blocks"],
            "blur_sigma": float(k.get("blur_sigma", 10.0)), "temperature": 2.0}


PAPERS += (
    Paper("PAG", "PAG: Perturbed-Attention Guidance (Ahn et al. 2024)", "Ahn et al., ECCV 2024",
          _arxiv("2403.17377"), "weak", "weak",
          "One extra pass on the prompt with the chosen self-attention replaced by the identity (each token sees "
          "only itself), and guidance away from it on top of CFG: + s (c - pag). SDXL and SD1.5 (UNets).",
          knobs=(Knob("scale", F, 3.0, 0.0, 20.0, 0.1, tip="Strength s (1.5-5; 1.5 when CFG is also on)."),
                 Knob("blocks", C, "middle (PAG / SEG default)", options=tuple(engine.BLOCK_PRESETS), tip=BLOCK_TIP)),
          build=lambda k: _weak("pag", k), catalog="C5"),
    Paper("SEG", "SEG: Smoothed Energy Guidance (Hong 2024)", "Hong, NeurIPS 2024",
          _arxiv("2408.00760"), "weak", "weak",
          "One extra pass with the chosen self-attention queries blurred over the image (a flatter attention "
          "energy), and guidance away from it: + s (c - seg). SDXL, SD1.5 and Anima.",
          knobs=(Knob("scale", F, 3.0, 0.0, 20.0, 0.1, tip="Strength s (paper 3)."),
                 Knob("blur_sigma", F, 10.0, 0.1, 100.0, 0.1, tip="Blur in tokens (100 = infinite: every query is the mean)."),
                 Knob("blocks", C, "middle (PAG / SEG default)", options=tuple(engine.BLOCK_PRESETS), tip=BLOCK_TIP)),
          build=lambda k: _weak("seg", k), catalog="C8"),
    Paper("STG", "STG: Spatiotemporal Skip Guidance, attention skip (Hyung et al. 2025)",
          "Hyung, Kim, Hong, Kim & Choo, CVPR 2025", _arxiv("2411.18664"), "weak", "weak",
          "One extra pass with the chosen self-attention layers skipped (they add nothing), and guidance away from "
          "it: + s (c - skip). The self-attention form of STG's layer skip; exact on Anima, whose attention "
          "carries no bias. SDXL, SD1.5 and Anima.",
          knobs=(Knob("scale", F, 1.0, 0.0, 20.0, 0.1, tip="Strength s (paper 1 for the residual skip)."),
                 Knob("blocks", C, "middle (PAG / SEG default)", options=tuple(engine.BLOCK_PRESETS), tip=BLOCK_TIP)),
          build=lambda k: _weak("skip", k), catalog="C9"),
)

# ---------------------------------------------------------------------------------------------------------------
# negatives: guiders with a positive, a negative and a null (empty) prompt
# ---------------------------------------------------------------------------------------------------------------

SLD_PRESETS = {"medium": {"warmup": 10, "s_s": 1000.0, "lam": 0.01, "s_m": 0.3, "beta_m": 0.4},
               "strong": {"warmup": 7, "s_s": 2000.0, "lam": 0.025, "s_m": 0.5, "beta_m": 0.7},
               "max": {"warmup": 0, "s_s": 5000.0, "lam": 1.0, "s_m": 0.5, "beta_m": 0.7}}

PAPERS += (
    Paper("PerpNeg", "Perp-Neg Guider (Armandpour et al. 2023)", "Armandpour, Sadeghian, Zheng, Sadeghian & Zhou, arXiv 2023",
          _arxiv("2304.04968"), "guider", "negatives",
          "Uses only the part of the negative's direction that is perpendicular to the positive's, so a negative "
          "cannot cancel what the prompt asks for: u + w ((c - u) - s perp(n - u)).", registry="perp_neg", catalog="D4",
          knobs=(Knob("neg_scale", F, 1.0, 0.0, 5.0, 0.05, tip="Weight of the perpendicular negative (paper 1.5 for one negative)."),)),
    Paper("ComposableNOT", "Composable NOT Guider (Liu et al. 2022)", "Liu, Li, Du, Torralba & Tenenbaum, ECCV 2022",
          _arxiv("2206.01714"), "guider", "negatives",
          "Composable diffusion's negation: u + w (c - n), the null prompt as the base and the negative as the "
          "direction to leave.", registry="composable_not", catalog="D3"),
    Paper("SignedGuidance", "Signed guidance: positive and negative from the null (A1111 AND; VL-DNP, Chang et al. 2025)",
          "AUTOMATIC1111 AND with negative weights; Chang, Kim & Choi, arXiv 2025", _arxiv("2510.26052"),
          "guider", "negatives",
          "Both prompts measured from the empty prompt: u + w (c - u) - w_neg (n - u).", registry="signed_guidance",
          catalog="D20",
          knobs=(Knob("w_neg", F, 5.0, 0.0, 30.0, 0.1, tip="Weight of the negative direction (w_neg = w is composable NOT)."),)),
    Paper("ContrastiveCFG", "ContrastiveCFG Guider (Chang et al. 2024)", "Chang, Lee, Chung & Ye, ICML 2026",
          _arxiv("2411.17077"), "guider", "negatives",
          "Positive and negative directions weighted by how far each already is from the null: more push while "
          "the positive is weak, less negative once the negative is far.", registry="ccfg", catalog="D12",
          knobs=(Knob("w_neg", F, -1.0, -1.0, 30.0, 0.1, convert=_none_if_negative, tip="Negative weight (-1 = w)."),
                 Knob("tau", F, -1.0, -1.0, 100.0, 0.01, convert=_none_if_negative, tip="Temperature (-1 = calibrated per image)."))),
    Paper("SafeLatentDiffusion", "Safe Latent Diffusion Guider (Schramowski et al. 2023)",
          "Schramowski, Brack, Deiseroth & Kersting, CVPR 2023", _arxiv("2211.05105"), "guider", "negatives",
          "Steers away from a concept (the negative prompt) only where the image moves toward it, with momentum and "
          "a warm-up; the paper's medium, strong and max settings.", registry="sld", catalog="D7",
          knobs=(Knob("preset", C, "medium", options=tuple(SLD_PRESETS), tip="The paper's configurations."),)),
    Paper("WindowedNegative", "Windowed negative prompt (Ban et al. 2024)",
          "Ban, Wang, Zhou, Cheng, Gong & Hsieh, ECCV 2024", _arxiv("2406.02965"), "guider", "negatives",
          "The negative prompt replaces the empty prompt only inside a window of the run: nouns need it from about "
          "step 5 of 30, adjectives from about 10.", registry="windowed_negative", catalog="D2",
          knobs=(Knob("start", F, 0.17, 0.0, 1.0, 0.01, tip="Where the negative starts (share of the run)."),
                 Knob("end", F, 0.5, 0.0, 1.0, 0.01, tip="Where it ends (1 = to the end: a delayed negative)."))),
)

# ---------------------------------------------------------------------------------------------------------------
# frequency and space
# ---------------------------------------------------------------------------------------------------------------

PAPERS += (
    Paper("SAMG", "SAMG: spatially adaptive guidance (Li et al. 2026)", "Li et al., arXiv 2026",
          _arxiv("2604.26503"), "combine", "frequency",
          "A scale per pixel: pixels where the guidance difference has high energy get the low scale, calm pixels "
          "the high one.", registry="samg", catalog="F2",
          knobs=(Knob("w_min", F, -1.0, -1.0, 30.0, 0.1, convert=_none_if_negative, tip="Scale at high-energy pixels (-1 = w 2/3)."),
                 Knob("w_max", F, -1.0, -1.0, 30.0, 0.1, convert=_none_if_negative, tip="Scale at calm pixels (-1 = w 1.6)."))),
    Paper("FDG", "FDG: Frequency-Decoupled Guidance (Sadat et al. 2025)", "Sadat, Vontobel, Salehi & Weber, arXiv 2025",
          _arxiv("2506.19713"), "combine", "frequency",
          "A Laplacian pyramid of both predictions, guided per level: the full scale on fine detail, a lower "
          "one on coarse structure (color and layout), which avoids over-saturation.", registry="fdg", catalog="F3",
          knobs=(Knob("w_low", F, -1.0, -1.0, 30.0, 0.1, convert=_none_if_negative, tip="Scale on the coarsest level (-1 = w / 2)."),
                 Knob("levels", I, 2, 1, 5, 1, tip="Pyramid levels."),
                 Knob("parallel_weight", F, 1.0, 0.0, 1.0, 0.01, tip="Weight of each level's part along the conditional (1 = none removed)."),
                 Knob("formulation", C, "paper", options=("paper", "diffusers"), tip="Identical at parallel_weight 1."))),
    Paper("FreSca", "FreSca: Fourier band scaling (Huang et al. 2025)", "Huang et al., arXiv 2025",
          _arxiv("2504.02154"), "combine", "frequency",
          "The guidance difference is split in the Fourier domain and its low and high bands scaled separately.",
          registry="fresca", catalog="F4",
          knobs=(Knob("scale_low", F, 1.0, 0.0, 3.0, 0.01, tip="Low-band factor."),
                 Knob("scale_high", F, 1.25, 0.0, 3.0, 0.01, tip="High-band factor (paper SDXL 1.5)."),
                 Knob("freq_cutoff", I, 20, 1, 256, 1, tip="Band edge, in frequency bins."))),
    Paper("HiWave", "HiWave: wavelet detail guidance (Vontobel et al. 2025)", "Vontobel, Sadat, Salehi & Weber, SIGGRAPH Asia 2025",
          _arxiv("2506.20452"), "combine", "frequency",
          "One wavelet level: guidance on the detail bands only, the coarse band left conditional (w_low 1). "
          "(The paper's upscaling pipeline is not part of this node.)", registry="hiwave", catalog="F5",
          knobs=(Knob("w_low", F, 1.0, 0.0, 20.0, 0.1, tip="Scale on the coarse band (1 = conditional)."),
                 Knob("wavelet", C, "sym4", options=tuple(cv._WAVELETS), tip="The wavelet."))),
    Paper("LFCFG", "LF-CFG: low-frequency improved CFG (Song & Lai 2025)", "Song & Lai, arXiv 2025",
          _arxiv("2506.21452"), "combine", "frequency",
          "Finds the slowly changing low-frequency regions of the guidance (redundant between steps) and scales them "
          "down; the rest gets plain CFG.", registry="lf_cfg", catalog="F6",
          knobs=(Knob("rho", F, 0.5, 0.0, 2.0, 0.01, tip="Scale on the slow low-frequency part (1 = plain CFG)."),
                 Knob("k", I, 8, 2, 32, 1, tip="Low-pass downsampling factor."))),
    Paper("ZeResFDG", "ZeResFDG (CADE 2.5; Rychkovskiy 2025)", "Rychkovskiy, arXiv 2025",
          _arxiv("2510.12954"), "combine", "frequency",
          "Frequency-decoupled guidance with rescale and zero-projection, switching between them by how much of "
          "the guidance sits at high frequencies.", registry="zeresfdg", catalog="F7",
          knobs=(Knob("mode", C, "auto", options=("auto", "cfgzero_fd", "rescale_fdg"), tip="auto switches by the high-frequency share."),
                 Knob("lam_low", F, 0.6, 0.0, 3.0, 0.01, tip="Low-band weight."),
                 Knob("lam_high", F, 1.3, 0.0, 3.0, 0.01, tip="High-band weight."),
                 Knob("rescale_mix", F, 0.7, 0.0, 1.0, 0.01, tip="Rescale blend."),
                 Knob("blur_sigma", F, 1.0, 0.1, 10.0, 0.1, tip="Low-pass blur."))),
)

BY_KEY: Dict[str, Paper] = {p.key: p for p in PAPERS}
assert len(BY_KEY) == len(PAPERS), "duplicate paper keys"


def knob_values(paper: Paper, values: Dict[str, Any]) -> Dict[str, Any]:
    """The library keywords for a paper's knob values (renamed and converted where the node shows other units)."""
    out = {}
    for k in paper.knobs:
        v = values.get(k.name, k.default)
        if k.kind == "float":
            v = float(v)
        elif k.kind == "int":
            v = int(v)
        elif k.kind == "bool":
            v = bool(v)
        out[k.lib or k.name] = k.convert(v) if k.convert else v
    return out


def stage_entry(paper: Paper, values: Dict[str, Any], scale: float = -1.0, space: str = "auto") -> Tuple[str, Dict[str, Any]]:
    """(plan key, entry) for a combine, when or weak paper."""
    if paper.stage == "combine":
        return "mix", {"kind": "registry", "registry": paper.registry, "knobs": knob_values(paper, values),
                       "space": space, "scale": float(scale), "title": paper.title}
    if paper.stage in ("when", "weak"):
        return paper.stage, paper.build({k.name: values.get(k.name, k.default) for k in paper.knobs})
    raise ValueError(f"{paper.key} is a guider paper")


def guider_knobs(paper: Paper, values: Dict[str, Any]) -> Dict[str, Any]:
    if paper.key == "SafeLatentDiffusion":
        return dict(SLD_PRESETS[values.get("preset", "medium")])
    return knob_values(paper, values)
