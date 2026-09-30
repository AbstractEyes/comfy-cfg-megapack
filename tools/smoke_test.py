"""smoke_test.py - drive a running ComfyUI through its HTTP API and exercise every CFG Megapack node on SDXL
(--model sdxl, the default) or on Anima (--model anima: the flow-matching DiT with its shift, 16-channel latents).

Small and fast on purpose: 128x128 (Anima 256x256), 6 steps. Checks:
  * all CFG Megapack nodes are registered (and ComfyUI-Manager answered at startup, read from the log)
  * every node runs and produces a finite, non-blank image
  * neutral settings reproduce the plain sampler's image (standard rule, formula u + w (c - u), a full-white
    region mask with outside 0, bands at 1/1, a clear-plan node after APG, the guider's negative_as_null)
  * the probe writes its per-step file; the readout returns the plan text
  * Anima: the formula sees flow = True and the shift of the ModelSamplingAuraFlow node; the weak branch runs SEG on
    the DiT and refuses PAG with its message
  * formulas are checked: an import is refused on the node and a torch.save while sampling, both naming the opt-in;
    with --formula-python (a server started with CFG_MEGAPACK_FORMULA_PYTHON=1) an imported module runs instead
Usage (ComfyUI already running):  python tools/smoke_test.py --port 8189 [--model anima]
Writes logs/smoke_results.json (smoke_results_anima.json; *_only.json for an --only run) and prints a table."""
import argparse
import glob
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                     # the repository = the custom node folder
sys.path.insert(0, ROOT)
from cfg_megapack import papers as _papers  # noqa: E402  (the paper node ids; imports without ComfyUI)

NODE_IDS = ["CFGP_When", "CFGP_WeakPerturbed", "CFGP_MixScale", "CFGP_MixDirection", "CFGP_MixFormula",
            "CFGP_WhereBands", "CFGP_WhereRegion", "CFGP_Correct", "CFGP_GovernAngle", "CFGP_Probe", "CFGP_Readout",
            "CFGP_Clear", "CFGP_ThreeWayGuider"] + [f"CFGP_{p.key}" for p in _papers.PAPERS]


def _find_comfy():
    """ComfyUI's folder: two levels up when this pack sits in ComfyUI/custom_nodes, else a sibling ComfyUI folder."""
    up = os.path.dirname(ROOT)
    if os.path.basename(up) == "custom_nodes":
        return os.path.dirname(up)
    return os.path.join(up, "ComfyUI") if os.path.isdir(os.path.join(up, "ComfyUI")) else ""

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8189)
ap.add_argument("--model", choices=("sdxl", "anima"), default="sdxl")
ap.add_argument("--size", type=int, default=0, help="default 128 (SDXL) / 256 (Anima)")
ap.add_argument("--steps", type=int, default=6)
ap.add_argument("--cfg", type=float, default=0.0, help="default 7.0 (SDXL) / 4.5 (Anima)")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--ckpt", default="sd_xl_base_1.0.safetensors")
ap.add_argument("--unet", default="anima-base-v1.0.safetensors")
ap.add_argument("--clip", default="qwen_3_06b_base.safetensors")
ap.add_argument("--vae", default="qwen_image_vae.safetensors")
ap.add_argument("--shift", type=float, default=3.0, help="Anima's timestep shift (ModelSamplingAuraFlow)")
ap.add_argument("--only", default="", help="comma-separated case names")
ap.add_argument("--papers", action="store_true", help="run every paper node with its defaults instead of the stage cases")
ap.add_argument("--formula-python", action="store_true",
                help="the server was started with CFG_MEGAPACK_FORMULA_PYTHON=1: check that a full-Python formula runs")
ap.add_argument("--output-dir", default="", help="the server's --output-directory, if it was given one")
ap.add_argument("--comfy", default="", help="the ComfyUI folder (found by itself when the pack is in custom_nodes)")
A = ap.parse_args()
ANIMA = A.model == "anima"
A.size = A.size or (256 if ANIMA else 128)
A.cfg = A.cfg or (4.5 if ANIMA else 7.0)
URL = f"http://127.0.0.1:{A.port}"
CLIENT = str(uuid.uuid4())
COMFY = A.comfy or _find_comfy()
OUTPUT = A.output_dir or (os.path.join(COMFY, "output") if COMFY else "")
if not OUTPUT:
    sys.exit("smoke_test: pass --comfy <ComfyUI folder> or --output-dir <the server's output folder>")
# Anima (model card): quality tags first, the recommended negative, er_sde on the simple schedule, shift 3
POS = ("masterpiece, best quality, score_7, safe, a red fox sitting in snow, detailed illustration" if ANIMA
       else "a red fox sitting in snow, detailed photo")
NEG = ("worst quality, low quality, score_1, score_2, score_3, artist name, blurry, jpeg artifacts, chromatic aberration"
       if ANIMA else "blurry, low quality")
SAMPLER, SCHEDULER = ("er_sde", "simple") if ANIMA else ("euler", "normal")
MODEL, CLIP, VAE = (["msf", 0], ["clipl", 0], ["vael", 0]) if ANIMA else (["ckpt", 0], ["ckpt", 1], ["ckpt", 2])


def loaders():
    if ANIMA:
        return {"unet": {"class_type": "UNETLoader", "inputs": {"unet_name": A.unet, "weight_dtype": "default"}},
                "msf": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["unet", 0], "shift": A.shift}},
                "clipl": {"class_type": "CLIPLoader", "inputs": {"clip_name": A.clip, "type": "stable_diffusion"}},
                "vael": {"class_type": "VAELoader", "inputs": {"vae_name": A.vae}}}
    return {"ckpt": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": A.ckpt}}}


def get(path):
    with urllib.request.urlopen(URL + path, timeout=30) as r:
        return json.loads(r.read())


def post(path, data):
    req = urllib.request.Request(URL + path, data=json.dumps(data).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"http_error": e.code, "body": e.read().decode(errors="replace")[:2000]}


# ---------------------------------------------------------------------------- graph builders

def base_graph(tag):
    """Nodes shared by every case; the model enters the sampler through the patch chain (see chain)."""
    g = loaders()
    g.update({
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"text": POS, "clip": CLIP}},
        "neg": {"class_type": "CLIPTextEncode", "inputs": {"text": NEG, "clip": CLIP}},
        "lat": {"class_type": "EmptyLatentImage", "inputs": {"width": A.size, "height": A.size, "batch_size": 1}},
        "dec": {"class_type": "VAEDecode", "inputs": {"samples": ["ks", 0], "vae": VAE}},
        "save": {"class_type": "SaveImage", "inputs": {"images": ["dec", 0],
                                                       "filename_prefix": f"cfgp_smoke{'_anima' if ANIMA else ''}/{tag}"}},
    })
    return g


def ksampler(model_ref):
    return {"class_type": "KSampler", "inputs": {"model": model_ref, "seed": A.seed, "steps": A.steps, "cfg": A.cfg,
                                                   "sampler_name": SAMPLER, "scheduler": SCHEDULER,
                                                   "positive": ["pos", 0], "negative": ["neg", 0],
                                                   "latent_image": ["lat", 0], "denoise": 1.0}}


def chain(tag, patches):
    """patches: list of (class_type, inputs without 'model'); applied in order between the loader and KSampler."""
    g = base_graph(tag)
    prev = MODEL
    for i, (ct, inputs) in enumerate(patches):
        nid = f"p{i}"
        g[nid] = {"class_type": ct, "inputs": dict(inputs, model=prev)}
        prev = [nid, 0]
    g["ks"] = ksampler(prev)
    return g


def with_readout(g, model_node):
    g["readout"] = {"class_type": "CFGP_Readout", "inputs": {"model": [model_node, 0]}}
    return g


def guider_graph(tag, rule, neg_scale=1.0, patches=()):
    g = base_graph(tag)
    prev = MODEL
    for i, (ct, inputs) in enumerate(patches):
        g[f"p{i}"] = {"class_type": ct, "inputs": dict(inputs, model=prev)}
        prev = [f"p{i}", 0]
    g["null"] = {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": CLIP}}
    g["guider"] = {"class_type": "CFGP_ThreeWayGuider", "inputs": {"model": prev, "positive": ["pos", 0], "negative": ["neg", 0],
                                                                  "null": ["null", 0], "cfg": A.cfg, "rule": rule,
                                                                  "negative_scale": neg_scale, "space": "denoised (x0)"}}
    g["noise"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": A.seed}}
    g["sampler"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": SAMPLER}}
    g["sigmas"] = {"class_type": "BasicScheduler", "inputs": {"model": prev, "scheduler": SCHEDULER, "steps": A.steps, "denoise": 1.0}}
    g["ks"] = {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["noise", 0], "guider": ["guider", 0],
                                                                "sampler": ["sampler", 0], "sigmas": ["sigmas", 0],
                                                                "latent_image": ["lat", 0]}}
    return g


def reference_guider_graph(tag):
    """The same custom-sampler path with ComfyUI's own CFGGuider (the reference for the three-way guider)."""
    g = guider_graph(tag, "negative_as_null")
    g["guider"] = {"class_type": "CFGGuider", "inputs": {"model": MODEL, "positive": ["pos", 0],
                                                         "negative": ["neg", 0], "cfg": A.cfg}}
    return g


MIXS = "CFGP_MixScale"
MIXD = "CFGP_MixDirection"


def mix_scale(rule, **kw):
    d = {"rule": rule, "scale": -1.0, "zero_init_steps": 0, "power_alpha": 0.9, "damping_alpha": 8.0,
         "space": "auto (the method's own)"}
    d.update(kw)
    return (MIXS, d)


def mix_dir(rule, **kw):
    d = {"rule": rule, "scale": -1.0, "eta": 0.0, "norm_threshold": 15.0, "momentum": -0.5, "max_angle_degrees": 60.0,
         "space": "auto (the method's own)"}
    d.update(kw)
    return (MIXD, d)


def when(**kw):
    d = {"shape": "constant", "start_percent": 0.0, "end_percent": 1.0, "outside": "no guidance (conditional)",
         "outside_scale": 1.0, "floor": 0.0, "shape_a": -1.0, "shape_b": -1.0, "skip_uncond_outside": True}
    d.update(kw)
    return ("CFGP_When", d)


def weak(**kw):
    d = {"method": "pag (identity attention)", "scale": 3.0, "mode": "add on top of CFG",
         "blocks": "middle (PAG / SEG default)", "blur_sigma": 10.0, "temperature": 2.0}
    d.update(kw)
    return ("CFGP_WeakPerturbed", d)


def correct(method, **kw):
    d = {"method": method, "strength": 0.7, "cap_ratio": 1.05, "mimic_scale": 4.0, "percentile": 0.995,
         "softness": 1.0, "space": "auto (the method's own)"}
    d.update(kw)
    return ("CFGP_Correct", d)


def bands(**kw):
    d = {"method": "gaussian", "low_multiplier": 1.0, "high_multiplier": 1.0, "blur_sigma": 2.0, "fft_cutoff": 0.25}
    d.update(kw)
    return ("CFGP_WhereBands", d)


def region_graph(tag, inside, outside, mask_value=1.0):
    g = chain(tag, [])
    g["mask"] = {"class_type": "SolidMask", "inputs": {"value": mask_value, "width": A.size, "height": A.size}}
    g["p0"] = {"class_type": "CFGP_WhereRegion", "inputs": {"model": MODEL, "mask": ["mask", 0],
                                                           "inside_multiplier": inside, "outside_multiplier": outside,
                                                           "feather": 0.0, "invert": False}}
    g["ks"] = ksampler(["p0", 0])
    return g


def formula(text, space="noise (eps)"):
    return ("CFGP_MixFormula", {"formula": text, "space": space, "scale": -1.0})


def govern(**kw):
    d = {"max_angle_degrees": 30.0, "min_angle_degrees": 0.0, "unit": "whole image (one vector per image)",
         "home": "conditional (how far guidance turns the prediction)", "start_percent": 0.0, "end_percent": 1.0,
         "space": "auto (the method's own)"}
    d.update(kw)
    return ("CFGP_GovernAngle", d)


PENTA = open(os.path.join(ROOT, "formulas", "pentachoron.txt"), encoding="utf-8").read()
assert PENTA.count("k = 1.0 ") == 1
PENTA_CFG = PENTA.replace("k = 1.0 ", "k = 1e6 ")          # the formula's plain-CFG limit
CHAIN_OUT = "base scale (the other nodes still apply)"
PLAIN_OUT = "plain CFG at base scale (the other nodes off)"


# name -> (graph, reference case or None, tolerance kind)
CASES = {
    "base": (chain("base", []), None),
    # the same plain model through a no-op clone, sampled with the model already resident: the reference for the
    # neutral cases (ComfyUI's first run after a model load rounds differently from later runs; both are repeatable)
    "plain_resident": (chain("plain_resident", [("CFGP_Clear", {})]), None),
    "standard_rule": (chain("standard_rule", [mix_scale("standard")]), "plain_resident"),
    "formula_cfg_eps": (chain("formula_cfg_eps", [formula("u + w * (c - u)")]), "plain_resident"),
    "formula_cfg_x0": (chain("formula_cfg_x0", [formula("u + w * (c - u)", "denoised (x0)")]), "plain_resident"),
    "bands_neutral": (chain("bands_neutral", [bands()]), "plain_resident"),
    "region_full_mask": (region_graph("region_full_mask", 1.0, 0.0), "plain_resident"),
    "clear_after_apg": (chain("clear_after_apg", [mix_dir("apg"), ("CFGP_Clear", {})]), "plain_resident"),
    "guider_reference": (reference_guider_graph("guider_reference"), None),
    "guider_negative_as_null": (guider_graph("guider_negative_as_null", "negative_as_null"), "guider_reference"),
    "cfg_zero_star": (chain("cfg_zero_star", [mix_scale("cfg_zero_star", zero_init_steps=1)]), None),
    "power_law": (chain("power_law", [mix_scale("power_law", power_alpha=0.3)]), None),
    "magnitude_damped": (chain("magnitude_damped", [mix_scale("magnitude_damped")]), None),
    "apg": (chain("apg", [mix_dir("apg")]), None),
    "tangential_damping": (chain("tangential_damping", [mix_dir("tangential_damping")]), None),
    "angle_limit": (chain("angle_limit", [mix_dir("angle_limit")]), None),
    "mahiro": (chain("mahiro", [mix_dir("mahiro")]), None),
    "formula_orth_boost": (chain("formula_orth_boost", [formula("u + w * orth(c - u, c) + proj(c - u, c)", "denoised (x0)")]), None),
    "when_linear_up_window": (chain("when_linear_up_window", [when(shape="linear_up", start_percent=0.1, end_percent=0.8)]), None),
    "when_tv_cfg": (chain("when_tv_cfg", [when(shape="tv_cfg")]), None),
    "weak_pag_add": (chain("weak_pag_add", [weak()]), None),
    "weak_seg": (chain("weak_seg", [weak(method="seg (blurred queries)")]), None),
    "weak_temperature": (chain("weak_temperature", [weak(method="temperature (flattened attention)", blocks="deep output (output 0-2)")]), None),
    "weak_pag_replace": (chain("weak_pag_replace", [weak(mode="replace the unconditional")]), None),
    "bands_detail_only": (chain("bands_detail_only", [bands(low_multiplier=0.3, high_multiplier=1.3, method="fft")]), None),
    "region_half": (region_graph("region_half", 1.5, 0.5, mask_value=1.0), None),
    "correct_rescale": (chain("correct_rescale", [correct("rescale_std")]), None),
    "correct_norm_cap": (chain("correct_norm_cap", [correct("norm_cap", strength=1.0)]), None),
    "correct_channel_norm": (chain("correct_channel_norm", [correct("channel_norm_match")]), None),
    "correct_energy": (chain("correct_energy", [correct("energy_preserve")]), None),
    "correct_percentile": (chain("correct_percentile", [correct("percentile_rescale", strength=1.0)]), None),
    "correct_soft_clip": (chain("correct_soft_clip", [correct("soft_clip", strength=1.0)]), None),
    "guider_perp_neg": (guider_graph("guider_perp_neg", "perp_neg"), None),
    "guider_separate_negative": (guider_graph("guider_separate_negative", "separate_negative", 0.5), None),
    "govern_neutral": (chain("govern_neutral", [govern(max_angle_degrees=180.0)]), "plain_resident"),
    "govern_image_20": (chain("govern_image_20", [govern(max_angle_degrees=20.0)]), None),
    "govern_pixel_20": (chain("govern_pixel_20", [govern(max_angle_degrees=20.0, unit="each pixel (its channel vector)")]), None),
    "govern_floor_15": (chain("govern_floor_15", [govern(max_angle_degrees=180.0, min_angle_degrees=15.0)]), None),
    "govern_uncond_home": (chain("govern_uncond_home", [govern(
        max_angle_degrees=30.0, home="unconditional (how far the prediction sits from the negative)")]), None),
    "when_window_chain_outside": (chain("when_window_chain_outside", [
        when(shape="cosine_down", start_percent=0.0, end_percent=0.3, outside=CHAIN_OUT),
        mix_dir("angle_limit", max_angle_degrees=13.0), bands(high_multiplier=4.0)]), None),
    "when_window_plain_outside": (chain("when_window_plain_outside", [
        when(shape="cosine_down", start_percent=0.0, end_percent=0.3, outside=PLAIN_OUT),
        mix_dir("angle_limit", max_angle_degrees=13.0), bands(high_multiplier=4.0)]), None),
    "formula_pentachoron": (chain("formula_pentachoron", [formula(PENTA)]), None),
    "formula_pentachoron_cfg_limit": (chain("formula_pentachoron_cfg_limit", [formula(PENTA_CFG)]), "plain_resident"),
    "full_stack_probe": (with_readout(chain("full_stack_probe", [
        when(shape="cosine_down", start_percent=0.0, end_percent=0.9), weak(scale=1.5), mix_dir("apg"),
        bands(high_multiplier=1.2), correct("rescale_std", strength=0.5), govern(max_angle_degrees=30.0),
        ("CFGP_Probe", {"filename_prefix": "smoke", "print_every": 0})]), "p6"), None),
}
# formulas travel inside workflow files: a server started without the full-Python opt-in refuses an import on the
# node (before sampling) and a file write while sampling, both naming the opt-in. A server started with it
# (--formula-python) runs an imported module instead, and gives plain CFG back.
if A.formula_python:
    SAFETY = {"formula_python_import": (chain("formula_python_import", [formula(
        "import torch.nn.functional as F\nresult = u + w * (c - u) + 0 * F.relu(c)", "denoised (x0)")]),
        "plain_resident")}
    SAFETY_ERROR = {}
else:
    SAFETY = {
        "formula_import_refused": (chain("formula_import_refused", [formula("import os\nresult = c")]), None),
        "formula_save_refused": (chain("formula_save_refused", [formula(
            "torch.save(c, 'cfg_smoke_refused.pt')\nresult = c")]), None),
    }
    SAFETY_ERROR = {n: "CFG_MEGAPACK_FORMULA_PYTHON" for n in SAFETY}
CASES.update(SAFETY)
EXPECT_ERROR = dict(SAFETY_ERROR)          # name -> a phrase the refusal must contain
# Neutral cases that convert to the noise or velocity space: the pack computes them in float64, plain CFG in float32,
# so each step differs by float rounding. SDXL shows none of it; Anima (flow, sigma near 1) grows it through the run,
# most in bf16 on a GPU (256 x 256, 6 steps, er_sde: mean 0.05 of 255 on the CPU, 2.6 on the GPU). Such a case
# passes as ROUNDING when its mean difference stays under 5 levels; formula_cfg_x0 (same arithmetic as plain CFG)
# must stay SAME, which rules out a fault in the formula path itself.
ROUNDING_ONLY = {"formula_cfg_eps", "formula_cfg_velocity", "formula_flow_variables", "formula_pentachoron_cfg_limit"}

if ANIMA:
    # the neutral set on 16-channel single-frame latents and the flow conversions, the flow variables inside ComfyUI,
    # SEG on the DiT (PAG refused with its message), and the variants most used on flow models
    FLOW_CHECK = (f"assert flow and abs(shift - {A.shift}) < 1e-6 and 0 < t_raw <= 1, (flow, shift, t_raw)\n"
                  "result = u + w * (c - u)")
    PICK = ("base", "plain_resident", "standard_rule", "formula_cfg_eps", "formula_cfg_x0", "bands_neutral", "region_full_mask",
            "guider_reference", "guider_negative_as_null", "govern_neutral", "formula_pentachoron_cfg_limit",
            "cfg_zero_star", "apg", "angle_limit", "weak_seg", "govern_image_20", "govern_pixel_20",
            "formula_pentachoron", "when_window_chain_outside", "correct_rescale")
    CASES = {**{n: CASES[n] for n in PICK},
             "formula_flow_variables": (chain("formula_flow_variables", [formula(FLOW_CHECK)]), "plain_resident"),
             "formula_cfg_velocity": (chain("formula_cfg_velocity", [formula("u + w * (c - u)", "velocity (v)")]),
                                      "plain_resident"),
             "weak_pag_refused": (chain("weak_pag_refused", [weak()]), None),
             "full_stack_probe": (with_readout(chain("full_stack_probe", [
                 when(shape="cosine_down", start_percent=0.0, end_percent=0.9),
                 weak(method="seg (blurred queries)", scale=1.5), mix_dir("apg"), bands(high_multiplier=1.2),
                 correct("rescale_std", strength=0.5), govern(max_angle_degrees=30.0),
                 ("CFGP_Probe", {"filename_prefix": "smoke", "print_every": 0})]), "p6"), None),
             **SAFETY}
    EXPECT_ERROR = {"weak_pag_refused": "Use SEG", **SAFETY_ERROR}


def paper_case(p):
    """A graph with one paper node at its defaults (a guider paper drives SamplerCustomAdvanced)."""
    inputs = {k.name: k.default for k in p.knobs}
    tag = f"paper_{p.key}"
    if p.stage == "guider":
        g = guider_graph(tag, "negative_as_null")
        g["guider"] = {"class_type": f"CFGP_{p.key}",
                       "inputs": dict(inputs, model=MODEL, positive=["pos", 0], negative=["neg", 0], null=["null", 0],
                                      cfg=A.cfg, space=p.space)}
        return g
    if p.stage == "combine":
        inputs.update(scale=-1.0, space=p.space)
    return chain(tag, [(f"CFGP_{p.key}", inputs)])


if A.papers:
    CASES = {"base": CASES["base"], "plain_resident": CASES["plain_resident"],
             **{f"paper_{p.key}": (paper_case(p), None) for p in _papers.PAPERS}}
    EXPECT_ERROR = {"paper_PAG": "Use SEG or skip"} if ANIMA else {}


def run_case(name, graph):
    t0 = time.time()
    r = post("/prompt", {"prompt": graph, "client_id": CLIENT})
    if "prompt_id" not in r:
        return {"ok": False, "error": r}
    pid = r["prompt_id"]
    while True:
        time.sleep(0.25)
        h = get(f"/history/{pid}")
        if pid in h:
            break
        if time.time() - t0 > 900:
            return {"ok": False, "error": "timeout"}
    entry = h[pid]
    status = entry.get("status", {})
    if status.get("status_str") != "success":
        msgs = [m for m in status.get("messages", []) if m[0] in ("execution_error", "execution_interrupted")]
        return {"ok": False, "error": msgs[-1][1] if msgs else status, "seconds": round(time.time() - t0, 2)}
    images = entry["outputs"].get("save", {}).get("images", [])
    out = {"ok": True, "seconds": round(time.time() - t0, 2)}
    if images:
        im = images[0]
        path = os.path.join(OUTPUT, im.get("subfolder", ""), im["filename"])
        arr = np.asarray(Image.open(path).convert("RGB")).astype(np.float32)
        out.update(path=path, mean=float(arr.mean()), std=float(arr.std()), finite=bool(np.isfinite(arr).all()))
    if "readout" in entry["outputs"]:
        out["readout"] = entry["outputs"]["readout"].get("text", [""])[0]
    return out


def main():
    info = get("/object_info")
    missing = [n for n in NODE_IDS if n not in info]
    print(f"registered CFG Megapack nodes: {len(NODE_IDS) - len(missing)}/{len(NODE_IDS)}" + (f" MISSING {missing}" if missing else ""))
    names = [n for n in CASES if not A.only or n in A.only.split(",")]
    for need in ("plain_resident", "base"):
        if need not in names:
            names = [need] + names
    devices = [d.get("name") for d in get("/system_stats").get("devices", [])]
    print(f"server devices: {devices}")
    results = {"_server": {"devices": devices, "when": time.strftime("%Y-%m-%d %H:%M:%S"), "model": A.model,
                           "size": A.size, "steps": A.steps, "cfg": A.cfg, "seed": A.seed,
                           "sampler": f"{SAMPLER} / {SCHEDULER}", "shift": A.shift if ANIMA else None}}
    for name in names:
        graph, _ = CASES[name]
        res = run_case(name, graph)
        if name in EXPECT_ERROR:             # a refusal case passes when it is refused with its message
            text = str(res.get("error", ""))
            res = {"ok": (not res["ok"]) and EXPECT_ERROR[name] in text, "expected_refusal": EXPECT_ERROR[name],
                   "refusal": text[:400], "seconds": res.get("seconds")}
            print(f"{'ok ' if res['ok'] else 'ERR'} {name:26s} {res.get('seconds', '')}s refused: {text[:160]}")
            results[name] = res
            continue
        results[name] = res
        tag = "ok " if res["ok"] else "ERR"
        print(f"{tag} {name:26s} {res.get('seconds', '')}s mean {res.get('mean', float('nan')):.1f} std {res.get('std', float('nan')):.1f}"
              + ("" if res["ok"] else f"  {str(res.get('error'))[:300]}"))
    # comparisons with references
    print("\nneutral settings vs their reference (max / mean absolute pixel difference, 0-255):")
    for name in names:
        ref = CASES[name][1]
        if ref and results.get(name, {}).get("ok") and results.get(ref, {}).get("ok"):
            a = np.asarray(Image.open(results[name]["path"]).convert("RGB")).astype(np.int16)
            b = np.asarray(Image.open(results[ref]["path"]).convert("RGB")).astype(np.int16)
            d = np.abs(a - b)
            results[name]["vs_ref"] = {"ref": ref, "max": int(d.max()), "mean": float(d.mean())}
            verdict = "SAME" if d.max() <= 2 else "DIFFERENT"
            if verdict == "DIFFERENT" and name in ROUNDING_ONLY and d.mean() <= 5.0:
                verdict = "ROUNDING"        # float64 space conversion vs float32 CFG; see ROUNDING_ONLY
            results[name]["vs_ref"]["verdict"] = verdict
            print(f"  {name:26s} vs {ref:18s} max {int(d.max()):3d} mean {d.mean():.3f}  {verdict}")
    # non-neutral cases must differ from base
    if results.get("base", {}).get("ok"):
        b = np.asarray(Image.open(results["base"]["path"]).convert("RGB")).astype(np.int16)
        print("\nvariants vs base (mean absolute pixel difference; 0 would mean the node did nothing):")
        for name in names:
            if name != "base" and not CASES[name][1] and results[name].get("ok") and "path" in results[name]:
                a = np.asarray(Image.open(results[name]["path"]).convert("RGB")).astype(np.int16)
                results[name]["vs_base_mean"] = float(np.abs(a - b).mean())
                print(f"  {name:26s} {results[name]['vs_base_mean']:.2f}")
    probes = sorted(glob.glob(os.path.join(OUTPUT, "cfg_probe", "smoke_*.jsonl")), key=os.path.getmtime)
    if probes:
        rows = [json.loads(l) for l in open(probes[-1], encoding="utf-8")]
        results["_probe"] = {"file": probes[-1], "lines": len(rows), "keys": sorted(rows[1].keys()) if len(rows) > 1 else []}
        print(f"\nprobe file {os.path.basename(probes[-1])}: {len(rows)} lines, keys {results['_probe']['keys']}")
    if "full_stack_probe" in results and results["full_stack_probe"].get("readout"):
        print("\nreadout of the full stack:\n" + results["full_stack_probe"]["readout"])
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    # a partial run never replaces the full record; each model keeps its own
    out_name = f"smoke_results{'_anima' if ANIMA else ''}{'_papers' if A.papers else ''}{'_only' if A.only else ''}.json"
    with open(os.path.join(ROOT, "logs", out_name), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    bad = [n for n in names if not results[n].get("ok")]
    diff = [n for n in names if results[n].get("vs_ref", {}).get("verdict") == "DIFFERENT"]
    print(f"\n{len(names) - len(bad)}/{len(names)} cases ran; neutral mismatches: {diff or 'none'}; errors: {bad or 'none'}")
    sys.exit(1 if (bad or diff or missing) else 0)


if __name__ == "__main__":
    main()
