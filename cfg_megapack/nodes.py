"""nodes.py - the CFG Megapack nodes.

Each MODEL node sets one stage of the guidance plan (engine.py) and passes the model on; chain them between the
model loader and the sampler in any order. The stages always run in this order at each sampler step:
1 when -> 2 weak branch -> 3 mix -> 4 where -> 5 correct -> 6 measure. A later node of the same stage replaces an
earlier one, except corrections, which stack in the order chained. The guider node swaps the mix stage for a
three-prediction rule (positive, negative, null) and keeps every other stage of the model's plan.
"""
from __future__ import annotations

import copy
import math

from typing_extensions import override

import comfy.sampler_helpers
import comfy.samplers
import node_helpers
from comfy_api.latest import ComfyExtension, io, ui

from . import engine
from . import papers

CAT = "CFG Megapack"
SPACE_AUTO = list(engine.SPACE_LABELS)                                   # auto + the three spaces
SPACE_FIXED = [k for k, v in engine.SPACE_LABELS.items() if v != "auto"]
SPACE_TIP = ("Where the rule is computed. Linear rules give the same image in any space; nonlinear ones do not. "
             "'auto' uses the space the method was published in (noise for most, denoised for APG and the angle "
             "rule, velocity for flow models).")
SCALE_TIP = "The guidance scale w for this rule. -1 uses the sampler's cfg value."


def _patched(model, update):
    m = model.clone()
    plan = engine.read_plan(m)
    update(plan)
    engine.install(m, plan)
    return m


def _space(label: str) -> str:
    return engine.SPACE_LABELS[label]


# ---------------------------------------------------------------------------
# 1 when
# ---------------------------------------------------------------------------

class CFGWhen(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CFGP_When",
            display_name="CFG When: Schedule and Window",
            category=f"{CAT}/1 when",
            description=("Shapes the guidance scale inside a window of the run and sets what happens outside it. The "
                         "scale entering the schedule is the sampler's cfg (or a mix node's scale). The shape runs "
                         "from the window's first step to its last. Every other CFG Megapack node keeps working "
                         "inside the window and, with the 'still apply' modes, outside it too: only the scale changes."),
            search_aliases=["cfg schedule", "guidance interval", "guidance window", "cfg truncation"],
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input("shape", options=list(engine.SCHEDULE_SHAPES), default="constant",
                               tooltip=("How the scale changes across the window. linear/cosine up and down, V and Lambda "
                                        "keep the window's average scale equal to cfg (Wang et al. 2024). tv_cfg peaks in "
                                        "the middle. beta_pdf follows the window's progress; c2fg_exp and cads_ramp "
                                        "follow the noise level itself.")),
                io.Float.Input("start_percent", default=0.0, min=0.0, max=1.0, step=0.01,
                               tooltip=("The window starts here (0 = the first step). ComfyUI's percent: measured on the "
                                        "model's own noise schedule, like its timestep-range nodes, so it can differ "
                                        "from the step count on karras-type schedulers.")),
                io.Float.Input("end_percent", default=1.0, min=0.0, max=1.0, step=0.01,
                               tooltip="The window ends here (1 = the last step)."),
                io.Combo.Input("outside", options=list(engine.OUTSIDE_MODES), default="no guidance (conditional)",
                               tooltip=("Outside the window: no guidance (the conditional prediction alone; the "
                                        "unconditional pass can be skipped there); the base scale or outside_scale with "
                                        "every other CFG Megapack node still applied; or plain CFG at the base scale "
                                        "with the other nodes off.")),
                io.Float.Input("outside_scale", default=1.0, min=0.0, max=100.0, step=0.1,
                               tooltip="The scale outside the window with the fixed-scale mode."),
                io.Float.Input("floor", default=0.0, min=0.0, max=100.0, step=0.1, advanced=True,
                               tooltip="Lowest scheduled scale allowed (0 = no floor)."),
                io.Float.Input("shape_a", default=-1.0, min=-10.0, max=10.0, step=0.01, advanced=True,
                               tooltip="Shape parameter a (-1 = the shape's default): tv_cfg peak position, beta_pdf alpha, c2fg rate, CADS tau1."),
                io.Float.Input("shape_b", default=-1.0, min=-10.0, max=10.0, step=0.01, advanced=True,
                               tooltip="Shape parameter b (-1 = default): beta_pdf beta, CADS tau2."),
                io.Boolean.Input("skip_uncond_outside", default=True, advanced=True,
                                 tooltip="With outside = no guidance, skip the unconditional pass there (faster, same image)."),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, shape, start_percent, end_percent, outside, outside_scale, floor=0.0, shape_a=-1.0,
                shape_b=-1.0, skip_uncond_outside=True) -> io.NodeOutput:
        if end_percent < start_percent:
            raise ValueError("CFG When: end_percent must be at or after start_percent")
        spec = {"shape": shape, "start": float(start_percent), "end": float(end_percent),
                "outside": engine.OUTSIDE_MODES[outside], "outside_scale": float(outside_scale), "floor": float(floor),
                "a": float(shape_a), "b": float(shape_b), "skip_uncond": bool(skip_uncond_outside)}
        return io.NodeOutput(_patched(model, lambda p: p.__setitem__("when", spec)))


# ---------------------------------------------------------------------------
# 2 weak branch
# ---------------------------------------------------------------------------

class CFGWeakPerturbed(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CFGP_WeakPerturbed",
            display_name="CFG Weak Branch: Perturbed Self-Attention",
            category=f"{CAT}/2 weak branch",
            description=("Runs one extra pass of the model on the positive prompt with some self-attention blocks "
                         "perturbed, and guides away from it. PAG: each token attends only to itself (Ahn et al. 2024). "
                         "SEG: the attention queries are blurred over the image (Hong 2024). Temperature: the "
                         "attention is flattened. Skip: the self-attention adds nothing (STG's layer skip, Hyung et "
                         "al. 2025). Costs one extra model pass per step. SDXL / SD1.5 UNets: all four. Anima / "
                         "Cosmos DiTs: SEG and skip (their blocks only take patches on the attention inputs), on the "
                         "DiT blocks the block choice maps to (middle two, first or last third, or all)."),
            search_aliases=["pag", "perturbed attention guidance", "seg", "smoothed energy guidance", "self guidance",
                            "stg", "skip layer"],
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input("method", options=list(engine.WEAK_METHODS), default="pag (identity attention)",
                               tooltip="pag: identity attention. seg: blurred queries. temperature: flatter attention. "
                                       "skip: the self-attention sublayer removed. Anima: seg and skip only."),
                io.Float.Input("scale", default=3.0, min=0.0, max=100.0, step=0.05,
                               tooltip="add mode: how hard to push away from the perturbed prediction (PAG paper: 3). Unused in replace mode."),
                io.Combo.Input("mode", options=list(engine.WEAK_MODES), default="add on top of CFG",
                               tooltip=("add: result = mix + scale * (conditional - perturbed). replace: the perturbed "
                                        "prediction takes the unconditional's place in the mix, at the mix scale "
                                        "(the text-unconditional pass is then skipped).")),
                io.Combo.Input("blocks", options=list(engine.BLOCK_PRESETS), default="middle (PAG / SEG default)",
                               tooltip=("Which self-attention blocks are perturbed (SDXL names; the middle block also exists "
                                        "on SD1.5). On Anima / Cosmos DiTs: middle = the two middle blocks, deep input = "
                                        "the first third, deep output = the last third, all = every block.")),
                io.Float.Input("blur_sigma", default=10.0, min=0.1, max=100.0, step=0.1,
                               tooltip="SEG only: Gaussian blur of the queries in latent pixels. 100 = infinite (every query becomes the mean)."),
                io.Float.Input("temperature", default=2.0, min=0.05, max=20.0, step=0.05,
                               tooltip="Temperature only: attention logits are divided by this (above 1 = flatter)."),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, method, scale, mode, blocks, blur_sigma, temperature) -> io.NodeOutput:
        error = engine.weak_support_error(getattr(model, "model", None), engine.WEAK_METHODS[method])
        if error:
            raise ValueError(error)
        spec = {"method": engine.WEAK_METHODS[method], "scale": float(scale), "mode": engine.WEAK_MODES[mode],
                "blocks": list(engine.BLOCK_PRESETS[blocks]), "blocks_label": blocks,
                "blur_sigma": float(blur_sigma), "temperature": float(temperature)}
        return io.NodeOutput(_patched(model, lambda p: p.__setitem__("weak", spec)))


# ---------------------------------------------------------------------------
# 3 mix
# ---------------------------------------------------------------------------

class CFGMixScale(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CFGP_MixScale",
            display_name="CFG Mix: Scale Rules",
            category=f"{CAT}/3 mix",
            description=("Rules that keep the guidance direction and change only how far to go along it. "
                         "standard: u + w (c - u). cfg_zero_star: rescales the unconditional to best fit the "
                         "conditional first, optional zero steps at the start (Fan et al. 2025). power_law: the scale "
                         "grows with the size of the difference (Lehman Pavasovic et al. 2025). magnitude_damped: the "
                         "scale shrinks when the difference is large relative to u (MAMBO-G, Zhu et al. 2025)."),
            search_aliases=["cfg zero star", "cfg-zero*", "power law cfg", "mambo", "standard cfg"],
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input("rule", options=["standard", "cfg_zero_star", "power_law", "magnitude_damped"], default="standard"),
                io.Float.Input("scale", default=-1.0, min=-1.0, max=100.0, step=0.1, tooltip=SCALE_TIP),
                io.Int.Input("zero_init_steps", default=0, min=0, max=50,
                             tooltip="cfg_zero_star: the first N steps leave the latent unmoved (paper: 1 on flow models; 0 = off)."),
                io.Float.Input("power_alpha", default=0.9, min=0.0, max=3.0, step=0.01,
                               tooltip="power_law: exponent on the difference's norm (0 = standard CFG). The effective scale is 1 + (w - 1) ||c - u||^alpha, so retune w per model and resolution."),
                io.Float.Input("damping_alpha", default=8.0, min=0.0, max=100.0, step=0.1,
                               tooltip="magnitude_damped: damping strength (0 = standard CFG)."),
                io.Combo.Input("space", options=SPACE_AUTO, default="auto (the method's own)", tooltip=SPACE_TIP),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, rule, scale, zero_init_steps, power_alpha, damping_alpha, space) -> io.NodeOutput:
        knobs = {"standard": {}, "cfg_zero_star": {"zero_init_steps": int(zero_init_steps)},
                 "power_law": {"alpha": float(power_alpha)}, "magnitude_damped": {"alpha": float(damping_alpha)}}[rule]
        spec = {"kind": "rule", "rule": rule, "scale": float(scale), "knobs": knobs, "space": _space(space)}
        return io.NodeOutput(_patched(model, lambda p: p.__setitem__("mix", spec)))


class CFGMixDirection(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CFGP_MixDirection",
            display_name="CFG Mix: Direction Rules",
            category=f"{CAT}/3 mix",
            description=("Rules that change the direction of the guidance, not only its length. apg: splits the "
                         "difference into the part along the conditional prediction and the rest, down-weights the "
                         "first, caps the norm and adds reverse momentum (Sadat et al. 2024; paper form "
                         "c + (w - 1)(...), one unit less than ComfyUI's built-in APG). tangential_damping: removes "
                         "the unconditional's component off the shared direction (TCFG, Kwon et al. 2025). "
                         "angle_limit: rotates instead of extrapolating past a maximum angle (ADG, Jin et al. 2025). "
                         "mahiro: blends toward a scaled conditional by a similarity score (ComfyUI Mahiro)."),
            search_aliases=["apg", "adaptive projected guidance", "tcfg", "tangential", "adg", "angle domain", "mahiro"],
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input("rule", options=["apg", "tangential_damping", "angle_limit", "mahiro"], default="apg"),
                io.Float.Input("scale", default=-1.0, min=-1.0, max=100.0, step=0.1, tooltip=SCALE_TIP),
                io.Float.Input("eta", default=0.0, min=-10.0, max=10.0, step=0.01,
                               tooltip="apg: weight of the part along the conditional prediction (1 with no cap and no momentum = standard CFG; paper: 0)."),
                io.Float.Input("norm_threshold", default=15.0, min=0.0, max=200.0, step=0.1,
                               tooltip="apg: cap on the difference's norm, in the denoised latent's units (0 = off; paper SDXL row: 15)."),
                io.Float.Input("momentum", default=-0.5, min=-1.0, max=1.0, step=0.01,
                               tooltip="apg: reverse momentum over steps (negative = reverse, the paper's -0.5; 0 = off)."),
                io.Float.Input("max_angle_degrees", default=60.0, min=1.0, max=180.0, step=1.0,
                               tooltip="angle_limit: the largest rotation away from the conditional prediction."),
                io.Combo.Input("space", options=SPACE_AUTO, default="auto (the method's own)", tooltip=SPACE_TIP),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, rule, scale, eta, norm_threshold, momentum, max_angle_degrees, space) -> io.NodeOutput:
        knobs = {"apg": {"eta": float(eta), "norm_threshold": float(norm_threshold), "momentum": float(momentum)},
                 "tangential_damping": {}, "angle_limit": {"max_angle": math.radians(float(max_angle_degrees))},
                 "mahiro": {}}[rule]
        spec = {"kind": "rule", "rule": rule, "scale": float(scale), "knobs": knobs, "space": _space(space)}
        return io.NodeOutput(_patched(model, lambda p: p.__setitem__("mix", spec)))


class CFGMixPentachoron(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CFGP_MixPentachoron",
            display_name="CFG Mix: Pentachoron",
            category=f"{CAT}/3 mix",
            description=("In-house: the aleph weighting on the 4-simplex. Each pixel's plain-CFG push past the "
                         "conditional, (w - 1)(c - u), is read on the 5 vertices of a regular pentachoron in every "
                         "group of 4 latent channels and rebuilt as sum_k sinh(z_k) v_k / sum_j cosh(z_j): signed "
                         "amplitudes that never select. Small pushes come back as plain CFG; a pixel's push in a group "
                         "stays within 4 tau (tau = k times the image's RMS vertex coordinate), and a pixel pushing "
                         "hard along one vertex damps its other four. Latents whose channel count divides by 4: SDXL "
                         "and SD1.5 (one pentachoron), Anima (16 channels, four)."),
            search_aliases=["pentachoron", "4-simplex", "simplex guidance", "aleph"],
            inputs=[
                io.Model.Input("model"),
                io.Float.Input("k", default=1.0, min=0.01, max=1000000.0, step=0.01,
                               tooltip="Temperature: larger is closer to plain CFG (1000000 is plain CFG), smaller "
                                       "a firmer limit on each pixel's push."),
                io.Float.Input("scale", default=-1.0, min=-1.0, max=100.0, step=0.1, tooltip=SCALE_TIP),
                io.Combo.Input("space", options=SPACE_FIXED, default="noise (eps)",
                               tooltip="Where the rule is computed; the rule is nonlinear, so the space changes the "
                                       "image. Noise is its own space (on flow models, the noise itself)."),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, k, scale, space) -> io.NodeOutput:
        spec = {"kind": "rule", "rule": "pentachoron", "scale": float(scale), "knobs": {"k": float(k)},
                "space": _space(space)}
        return io.NodeOutput(_patched(model, lambda p: p.__setitem__("mix", spec)))


FORMULA_HELP = (
    "Write the guided prediction as a Python expression (or several lines that assign `result`). "
    "Variables: c and u (conditional and unconditional predictions in the chosen space), w (the scheduled scale), "
    "x (the latent), sigma, t (noise level, 1 = noise), p (progress, 0 = first step), step, steps, weak (the "
    "perturbed prediction when a weak-branch node is chained, else None). Flow matching (Anima, SD3, Flux type): "
    "flow (True there), shift (the model's timestep shift, sigma = shift t / (1 + (shift - 1) t); exp(mu) on "
    "Flux-type sampling; 1 on eps models), t_raw (the unshifted time t), a_t and s_t (x = a_t x0 + s_t noise), "
    "space, to_x0(a) / to_eps(a) / to_v(a) to convert a prediction from the formula's space and from_x0(a) / "
    "from_eps(a) / from_v(a) to bring one back (on flow models eps is the noise and v the velocity noise - x0; "
    "plain CFG on the denoised image from any space: from_x0(to_x0(u) + w * (to_x0(c) - to_x0(u)))). Anima's "
    "latents arrive as (batch, 16, height, width): the single-frame "
    "axis is taken off and put back. Helpers (per sample): dot, norm, cos, proj(a, onto), orth(a, onto), std, mean, "
    "lowpass(a, sigma), highpass(a, sigma), lerp, clamp, where, sqrt, exp, tanh, sign, and torch, F "
    "(torch.nn.functional), torch.fft, torch.linalg and math with their maths functions. Formulas travel inside "
    "workflow files, so they run as a checked maths language: arithmetic, assignments, assert, if, the variables and "
    "helpers above and tensor methods; no imports, no names starting with '_', no file access. Full Python only when "
    "ComfyUI starts with CFG_MEGAPACK_FORMULA_PYTHON=1 (off by default; only for workflows you trust). "
    "Example: u + w * orth(c - u, c) + 1.0 * proj(c - u, c)")


class CFGMixFormula(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CFGP_MixFormula",
            display_name="CFG Mix: Your Own Formula",
            category=f"{CAT}/3 mix",
            description=FORMULA_HELP,
            search_aliases=["custom cfg", "cfg expression", "guidance formula"],
            inputs=[
                io.Model.Input("model"),
                io.String.Input("formula", multiline=True, default="u + w * (c - u)", tooltip=FORMULA_HELP),
                io.Combo.Input("space", options=SPACE_FIXED, default="noise (eps)", tooltip=SPACE_TIP),
                io.Float.Input("scale", default=-1.0, min=-1.0, max=100.0, step=0.1, tooltip=SCALE_TIP),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, formula, space, scale) -> io.NodeOutput:
        engine.compile_formula(formula)          # syntax errors surface on the node, before sampling
        spec = {"kind": "formula", "formula": formula, "space": _space(space), "scale": float(scale)}
        return io.NodeOutput(_patched(model, lambda p: p.__setitem__("mix", spec)))


# ---------------------------------------------------------------------------
# 4 where
# ---------------------------------------------------------------------------

class CFGWhereBands(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CFGP_WhereBands",
            display_name="CFG Where: Frequency Bands",
            category=f"{CAT}/4 where",
            description=("Splits the guidance term (how far guidance pushes past the conditional prediction) into low "
                         "frequencies (layout, color) and high frequencies (detail) and scales each. With plain CFG "
                         "this is frequency-decoupled guidance: band scale = 1 + multiplier (cfg - 1) (FDG, 2025)."),
            search_aliases=["fdg", "frequency decoupled guidance", "fresca", "detail guidance"],
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input("method", options=["gaussian", "fft"], default="gaussian",
                               tooltip="gaussian: low band = Gaussian blur. fft: low band = frequencies under the cutoff."),
                io.Float.Input("low_multiplier", default=1.0, min=-5.0, max=10.0, step=0.01,
                               tooltip="Multiplier on the low-frequency part of the push (1 = unchanged, 0 = no guidance there)."),
                io.Float.Input("high_multiplier", default=1.0, min=-5.0, max=10.0, step=0.01,
                               tooltip="Multiplier on the high-frequency part of the push."),
                io.Float.Input("blur_sigma", default=2.0, min=0.1, max=32.0, step=0.1,
                               tooltip="gaussian: blur sigma in latent pixels (1 latent pixel = 8 image pixels on SDXL)."),
                io.Float.Input("fft_cutoff", default=0.25, min=0.01, max=1.0, step=0.01,
                               tooltip="fft: the cutoff as a fraction of the highest frequency."),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, method, low_multiplier, high_multiplier, blur_sigma, fft_cutoff) -> io.NodeOutput:
        spec = {"method": method, "low": float(low_multiplier), "high": float(high_multiplier),
                "blur_sigma": float(blur_sigma), "fft_cutoff": float(fft_cutoff)}
        return io.NodeOutput(_patched(model, lambda p: p.__setitem__("bands", spec)))


class CFGWhereRegion(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CFGP_WhereRegion",
            display_name="CFG Where: Region Mask",
            category=f"{CAT}/4 where",
            description=("Scales the guidance term by region: inside_multiplier where the mask is white, "
                         "outside_multiplier where it is black (1 = unchanged, 0 = the conditional prediction alone)."),
            search_aliases=["masked cfg", "regional guidance", "spatial cfg"],
            inputs=[
                io.Model.Input("model"),
                io.Mask.Input("mask"),
                io.Float.Input("inside_multiplier", default=1.0, min=-5.0, max=10.0, step=0.01),
                io.Float.Input("outside_multiplier", default=0.0, min=-5.0, max=10.0, step=0.01),
                io.Float.Input("feather", default=1.0, min=0.0, max=32.0, step=0.1,
                               tooltip="Blur of the mask edge in latent pixels (0 = hard edge)."),
                io.Boolean.Input("invert", default=False),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, mask, inside_multiplier, outside_multiplier, feather, invert) -> io.NodeOutput:
        spec = {"mask": mask.detach().cpu().float(), "inside": float(inside_multiplier),
                "outside": float(outside_multiplier), "feather": float(feather), "invert": bool(invert)}
        return io.NodeOutput(_patched(model, lambda p: p.__setitem__("region", spec)))


# ---------------------------------------------------------------------------
# 5 correct
# ---------------------------------------------------------------------------

class CFGCorrect(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CFGP_Correct",
            display_name="CFG Correct: Magnitude",
            category=f"{CAT}/5 correct",
            description=("Pulls the guided prediction's size back toward the conditional prediction's, against the "
                         "over-saturation high scales cause. Stacks: chain several and they apply in order. "
                         "rescale_std: match the standard deviation (Lin et al. 2023). norm_cap: shrink the push until "
                         "the norm is at most cap_ratio times the conditional's (PMC-CFG). channel_norm_match: per pixel "
                         "(CFGNorm). energy_preserve: match the total energy (EP-CFG). percentile_rescale: shrink each "
                         "channel's spread to what a lower mimic scale gives. soft_clip: tone-map large pushes."),
            search_aliases=["rescale cfg", "cfg rescale", "dynamic thresholding", "cfg norm", "ep-cfg", "pmc"],
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input("method", options=list(engine.CORRECTIONS), default="rescale_std"),
                io.Float.Input("strength", default=0.7, min=0.0, max=1.0, step=0.01,
                               tooltip="Blend between the uncorrected (0) and fully corrected (1) prediction. The published "
                                       "forms: 0.7 for rescale_std, 1.0 for every other method."),
                io.Float.Input("cap_ratio", default=1.05, min=1.0, max=3.0, step=0.01, tooltip="norm_cap: the allowed norm ratio."),
                io.Float.Input("mimic_scale", default=4.0, min=1.0, max=30.0, step=0.1, tooltip="percentile_rescale: the reference scale."),
                io.Float.Input("percentile", default=0.995, min=0.5, max=1.0, step=0.001, tooltip="percentile_rescale: the spread percentile."),
                io.Float.Input("softness", default=1.0, min=0.05, max=10.0, step=0.05, tooltip="soft_clip: larger = gentler."),
                io.Combo.Input("space", options=SPACE_AUTO, default="auto (the method's own)", tooltip=SPACE_TIP),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, method, strength, cap_ratio, mimic_scale, percentile, softness, space) -> io.NodeOutput:
        spec = {"method": method, "strength": float(strength), "space": _space(space)}
        spec.update({"norm_cap": {"cap_ratio": float(cap_ratio)},
                     "percentile_rescale": {"mimic_scale": float(mimic_scale), "percentile": float(percentile)},
                     "soft_clip": {"softness": float(softness)}}.get(method, {}))
        return io.NodeOutput(_patched(model, lambda p: p.setdefault("correct", []).append(spec)))


# ---------------------------------------------------------------------------
# 6 govern
# ---------------------------------------------------------------------------

class CFGGovernAngle(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CFGP_GovernAngle",
            display_name="CFG Govern: Angle Band",
            category=f"{CAT}/6 govern",
            description=("A middleman after every other stage that holds the angle between the guided prediction and "
                         "its home (the conditional prediction by default) inside a band: min <= angle <= max. A "
                         "unit outside the band is turned, in the plane it spans with its home, exactly onto the "
                         "nearer edge, keeping its length; units inside the band stay bit for bit. max is a leash on "
                         "how far guidance may turn the prediction (the angle idea of ADG, Jin et al. 2025, applied "
                         "as an exact cap after any mix rule); min is a floor that turns weak guidance further along "
                         "its own direction. Min 0 and max 180 change nothing. Modelled on the AlephLM anchor "
                         "governor: a projection, never a reweighting; nothing moves until a bound binds. The "
                         "Per-Step Probe records the angles and how often each edge fired."),
            search_aliases=["governor", "angle band", "angle cap", "max angle", "leash", "adg"],
            inputs=[
                io.Model.Input("model"),
                io.Float.Input("max_angle_degrees", default=30.0, min=0.0, max=180.0, step=0.5,
                               tooltip="The leash: the largest angle allowed between the guided prediction and its home (180 = off)."),
                io.Float.Input("min_angle_degrees", default=0.0, min=0.0, max=180.0, step=0.5,
                               tooltip="The floor: the smallest angle allowed (0 = off). A unit with no guidance at all has no direction to turn along and is left alone."),
                io.Combo.Input("unit", options=list(engine.GOVERN_UNITS), default="whole image (one vector per image)",
                               tooltip=("What one governed vector is: the whole latent of an image (like the angle_limit "
                                        "rule), each pixel's channel vector (local colour and tone direction), or each "
                                        "channel's map.")),
                io.Combo.Input("home", options=list(engine.GOVERN_HOMES),
                               default="conditional (how far guidance turns the prediction)",
                               tooltip=("The reference the angle is measured from. conditional: the band limits how far "
                                        "guidance turns the prediction. unconditional: the band limits how far the "
                                        "prediction sits from the negative (or from the weak branch when it replaces the "
                                        "unconditional).")),
                io.Float.Input("start_percent", default=0.0, min=0.0, max=1.0, step=0.01,
                               tooltip="The governor acts from this point of the run (inside the When window, if any)."),
                io.Float.Input("end_percent", default=1.0, min=0.0, max=1.0, step=0.01,
                               tooltip="The governor stops at this point of the run."),
                io.Combo.Input("space", options=SPACE_AUTO, default="auto (the method's own)",
                               tooltip="Where the angle is measured. auto = the denoised image (x0), as ADG does."),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, max_angle_degrees, min_angle_degrees, unit, home, start_percent, end_percent,
                space) -> io.NodeOutput:
        if min_angle_degrees > max_angle_degrees:
            raise ValueError("CFG Govern: min_angle_degrees must be at most max_angle_degrees")
        if end_percent < start_percent:
            raise ValueError("CFG Govern: end_percent must be at or after start_percent")
        spec = {"max_deg": float(max_angle_degrees), "min_deg": float(min_angle_degrees),
                "unit": engine.GOVERN_UNITS[unit], "home": engine.GOVERN_HOMES[home],
                "start": float(start_percent), "end": float(end_percent), "space": _space(space)}
        return io.NodeOutput(_patched(model, lambda p: p.__setitem__("govern", spec)))


# ---------------------------------------------------------------------------
# 7 measure + utilities
# ---------------------------------------------------------------------------

class CFGProbe(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CFGP_Probe",
            display_name="CFG Measure: Per-Step Probe",
            category=f"{CAT}/7 measure",
            description=("Writes one line of numbers per sampler step to output/cfg_probe/<prefix>_<time>_<run>.jsonl: "
                         "the scale used, the size of the difference and its cosine with the conditional, how far "
                         "guidance pushed, the std ratio (saturation), the low-frequency share and the |x0| range."),
            search_aliases=["cfg log", "guidance stats", "cfg probe"],
            inputs=[
                io.Model.Input("model"),
                io.String.Input("filename_prefix", default="cfg_probe"),
                io.Int.Input("print_every", default=0, min=0, max=1000,
                             tooltip="Also print every Nth step to the console (0 = file only)."),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, filename_prefix, print_every) -> io.NodeOutput:
        prefix = "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in (filename_prefix or "cfg_probe"))
        spec = {"prefix": prefix, "print_every": int(print_every)}
        return io.NodeOutput(_patched(model, lambda p: p.__setitem__("measure", spec)))


class CFGReadout(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CFGP_Readout",
            display_name="CFG Plan Readout",
            category=f"{CAT}/7 measure",
            description="Shows the guidance plan on this model, stage by stage, in the order it runs.",
            inputs=[io.Model.Input("model")],
            outputs=[io.String.Output(display_name="plan")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, model) -> io.NodeOutput:
        text = engine.describe_plan(model.model_options.get(engine.PLAN_KEY))
        return io.NodeOutput(text, ui=ui.PreviewText(text))


class CFGClear(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CFGP_Clear",
            display_name="CFG Clear Plan",
            category=f"{CAT}/7 measure",
            description="Removes every CFG Megapack stage from the model (back to the sampler's plain CFG).",
            inputs=[io.Model.Input("model")],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model) -> io.NodeOutput:
        m = model.clone()
        engine.clear(m)
        return io.NodeOutput(m)


# ---------------------------------------------------------------------------
# Guider: positive, negative and null
# ---------------------------------------------------------------------------

class ThreeWayGuider(comfy.samplers.CFGGuider):
    """Runs positive, negative and null in one batch and combines them with a three-prediction rule, then applies
    the rest of the model's CFG Megapack plan (when, weak branch, where, correct, measure)."""

    def set_conds(self, positive, negative, null):
        null = node_helpers.conditioning_set_values(null, {"prompt_type": "negative"})
        self.inner_set_conds({"positive": positive, "negative": negative, "null": null})

    def set_rule(self, cfg, rule, neg_scale, space, knobs=None):
        self.cfg = float(cfg)
        self.rule = rule                    # a three-way rule, or "registry:<name>" for a paper guider
        self.neg_scale = float(neg_scale)
        self.knobs = dict(knobs or {})
        plan = copy.deepcopy(self.model_patcher.model_options.get(engine.PLAN_KEY)) or engine.empty_plan()
        plan["three_way_space"] = space
        self.plan = plan
        self.runtime = engine.GuidanceRuntime(plan)

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        positive = self.conds.get("positive", None)
        negative = self.conds.get("negative", None)
        null = self.conds.get("null", None)
        conds = [positive, negative, null]
        # positive + negative go through the model as one batch, exactly as ComfyUI's own CFGGuider sends them, so
        # negative_as_null reproduces the built-in guider bit for bit; the null prediction is a separate pass
        # (the same three evaluations; a batch of three would change the GPU's rounding).
        pos_neg = comfy.samplers.calc_cond_batch(self.inner_model, [positive, negative], x, timestep, model_options)
        if self.rule == "negative_as_null":
            out = [pos_neg[0], pos_neg[1], pos_neg[1]]
        else:
            (x0_null,) = comfy.samplers.calc_cond_batch(self.inner_model, [null], x, timestep, model_options)
            out = [pos_neg[0], pos_neg[1], x0_null]
        for fn in model_options.get("sampler_pre_cfg_function", []):
            args = {"conds": conds, "conds_out": out, "cond_scale": self.cfg, "timestep": timestep, "input": x,
                    "sigma": timestep, "model": self.inner_model, "model_options": model_options}
            out = fn(args)
        x0_pos, x0_neg, x0_null = out[0], out[1], out[2]
        result = self.runtime.guided_x0(x, x0_pos, x0_null, timestep, self.cfg, model=self.inner_model,
                                        model_options=model_options, input_cond=positive, x0_n=x0_neg,
                                        three_way_rule=self.rule, neg_scale=self.neg_scale,
                                        three_way_knobs=self.knobs)
        for fn in model_options.get("sampler_post_cfg_function", []):
            args = {"denoised": result, "cond": positive, "uncond": null, "cond_scale": self.cfg,
                    "model": self.inner_model, "uncond_denoised": x0_null, "cond_denoised": x0_pos, "sigma": timestep,
                    "model_options": model_options, "input": x, "negative_cond": negative, "negative_denoised": x0_neg}
            result = fn(args)
        return result


class CFGThreeWayGuider(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CFGP_ThreeWayGuider",
            display_name="CFG Guider: Positive, Negative and Null",
            category=f"{CAT}/guiders",
            description=("A guider for SamplerCustomAdvanced that keeps the negative prompt separate from the true "
                         "unconditional (null = an empty prompt). perp_neg: subtract only the part of the negative "
                         "direction perpendicular to the positive one (Armandpour et al. 2023). separate_negative: "
                         "null + cfg (positive - null) - negative_scale (negative - null) (composable negation, Liu "
                         "et al. 2022). negative_as_null: classic CFG, for reference (bit-identical to ComfyUI's CFGGuider). The model's other CFG Megapack "
                         "stages still apply. Costs three predictions per step (two for negative_as_null)."),
            search_aliases=["perp neg", "perpneg", "negative prompt guidance", "composable diffusion"],
            inputs=[
                io.Model.Input("model"),
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Conditioning.Input("null", tooltip="The empty prompt: the true unconditional."),
                io.Float.Input("cfg", default=7.0, min=0.0, max=100.0, step=0.1, round=0.01),
                io.Combo.Input("rule", options=list(engine.THREE_WAY_RULES), default="perp_neg"),
                io.Float.Input("negative_scale", default=1.0, min=0.0, max=100.0, step=0.01),
                io.Combo.Input("space", options=SPACE_FIXED, default="denoised (x0)",
                               tooltip="perp_neg's projection depends on the space (ComfyUI's built-in uses denoised)."),
            ],
            outputs=[io.Guider.Output()],
        )

    @classmethod
    def execute(cls, model, positive, negative, null, cfg, rule, negative_scale, space) -> io.NodeOutput:
        guider = ThreeWayGuider(model)
        guider.set_conds(positive, negative, null)
        guider.set_rule(cfg, rule, negative_scale, _space(space))
        return io.NodeOutput(guider)


# ---------------------------------------------------------------------------
# One node per paper (papers.py holds the specifications)
# ---------------------------------------------------------------------------

def _paper_input(k: papers.Knob):
    if k.kind == "float":
        return io.Float.Input(k.name, default=float(k.default), min=float(k.lo), max=float(k.hi), step=float(k.step),
                              tooltip=k.tip)
    if k.kind == "int":
        return io.Int.Input(k.name, default=int(k.default), min=int(k.lo), max=int(k.hi), tooltip=k.tip)
    if k.kind == "bool":
        return io.Boolean.Input(k.name, default=bool(k.default), tooltip=k.tip)
    return io.Combo.Input(k.name, options=list(k.options), default=k.default, tooltip=k.tip)


def _paper_description(p: papers.Paper) -> str:
    return f"{p.summary}\n\nFrom: {p.cite}. {p.link}" + (f"\n\n{p.flow_note}" if p.flow_note else "")


def _paper_node(p: papers.Paper) -> type:
    category = f"{CAT}/papers/{papers.LINES[p.line]}"
    guider = p.stage == "guider"
    if guider:
        inputs = [io.Model.Input("model"), io.Conditioning.Input("positive"), io.Conditioning.Input("negative"),
                  io.Conditioning.Input("null", tooltip="The empty prompt: the true unconditional."),
                  io.Float.Input("cfg", default=7.0, min=0.0, max=100.0, step=0.1, round=0.01)]
    else:
        inputs = [io.Model.Input("model")]
        if p.stage == "combine":
            inputs.append(io.Float.Input("scale", default=-1.0, min=-1.0, max=100.0, step=0.1, tooltip=SCALE_TIP))
    inputs += [_paper_input(k) for k in p.knobs]
    if p.stage in ("combine", "guider"):
        inputs.append(io.Combo.Input("space", options=SPACE_AUTO, default=p.space, tooltip=SPACE_TIP))

    def define_schema(cls):
        return io.Schema(node_id=f"CFGP_{p.key}", display_name=p.title, category=category,
                         description=_paper_description(p), search_aliases=[p.key, p.title.split(":")[0]],
                         inputs=list(inputs), outputs=[io.Guider.Output() if guider else io.Model.Output()])

    def execute(cls, model, **kw):
        space = _space(kw.pop("space", p.space))
        if guider:
            positive, negative, null, cfg = kw.pop("positive"), kw.pop("negative"), kw.pop("null"), kw.pop("cfg")
            g = ThreeWayGuider(model)
            g.set_conds(positive, negative, null)
            g.set_rule(cfg, "registry:" + p.registry, 1.0, space, knobs=papers.guider_knobs(p, kw))
            return io.NodeOutput(g)
        key, entry = papers.stage_entry(p, kw, kw.pop("scale", -1.0), space)
        if key == "weak":
            error = engine.weak_support_error(getattr(model, "model", None), entry["method"])
            if error:
                raise ValueError(error.replace("CFG Weak Branch", p.title.split(":")[0]))
        return io.NodeOutput(_patched(model, lambda plan: plan.__setitem__(key, entry)))

    return type(f"CFGPaper_{p.key}", (io.ComfyNode,),
                {"define_schema": classmethod(define_schema), "execute": classmethod(execute), "__doc__": p.summary})


PAPER_NODES = [_paper_node(p) for p in papers.PAPERS]

NODES = [CFGWhen, CFGWeakPerturbed, CFGMixScale, CFGMixDirection, CFGMixPentachoron, CFGMixFormula, CFGWhereBands,
         CFGWhereRegion, CFGCorrect, CFGGovernAngle, CFGProbe, CFGReadout, CFGClear, CFGThreeWayGuider] + PAPER_NODES


class CFGMegapackExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return list(NODES)


async def comfy_entrypoint() -> CFGMegapackExtension:
    return CFGMegapackExtension()
