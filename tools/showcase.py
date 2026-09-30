"""showcase.py - render the README / HOWTO comparisons: for every paper node (and a few stage nodes), plain CFG and
the node on the same prompt and seed, side by side with the method's name and source.

Drives a running ComfyUI through its HTTP API. Each render is a full-size PNG in ComfyUI's output folder (its
graph is embedded, so dropping it on ComfyUI loads the exact node layout); each pair becomes one captioned JPEG in
docs/images/<model>/, and the API graph of the variant is written to docs/showcase/<model>/<Key>.json.

  python tools/showcase.py --port 8188 --model sdxl            (1024 x 1024, 50 steps)
  python tools/showcase.py --port 8188 --model anima
  python tools/showcase.py --port 8188 --model sdxl --only APG,SEG --size 512 --steps 20
Needs the same models as the example workflows (SDXL base 1.0; Anima base, its text encoder and VAE)."""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from cfg_megapack import papers  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8188)
ap.add_argument("--model", choices=("sdxl", "anima"), default="sdxl")
ap.add_argument("--size", type=int, default=1024)
ap.add_argument("--steps", type=int, default=50)
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--only", default="", help="comma-separated keys")
ap.add_argument("--output-dir", default="", help="the server's output folder (to read the renders back)")
ap.add_argument("--comfy", default="", help="the ComfyUI folder, if the output folder is not given")
ap.add_argument("--docs", default=os.path.join(ROOT, "docs"), help="where the comparison images and graphs go")
A = ap.parse_args()
URL = f"http://127.0.0.1:{A.port}"
CLIENT = str(uuid.uuid4())
ANIMA = A.model == "anima"


def _comfy():
    up = os.path.dirname(ROOT)
    if os.path.basename(up) == "custom_nodes":
        return os.path.dirname(up)
    return os.path.join(up, "ComfyUI")


OUTPUT = A.output_dir or os.path.join(A.comfy or _comfy(), "output")

if ANIMA:
    POS = ("masterpiece, best quality, score_7, safe, 1girl, solo, red scarf, winter coat, snow, pine forest, "
           "looking at viewer, smile, detailed background, soft light")
    NEG = "worst quality, low quality, score_1, score_2, score_3, artist name, blurry, jpeg artifacts, chromatic aberration"
    SAMPLER, SCHEDULER, BASE_CFG, HIGH_CFG = "er_sde", "simple", 4.5, 9.0
    MODEL, CLIP, VAE = ["msf", 0], ["clipl", 0], ["vael", 0]
else:
    POS = "photograph of an old fisherman in a yellow raincoat on a harbor at dawn, weathered face, detailed, sharp focus"
    NEG = "blurry, low quality, watermark, text"
    SAMPLER, SCHEDULER, BASE_CFG, HIGH_CFG = "dpmpp_2m", "karras", 7.0, 14.0
    MODEL, CLIP, VAE = ["ckpt", 0], ["ckpt", 1], ["ckpt", 2]
# the negative-prompt guiders: a scene where the concept turns up unasked (a few plants), which the negative removes.
# The prompt does not ask for plants: when it does, the negative fights the prompt, the case Perp-Neg exists to keep.
GUIDE_POS = ("masterpiece, best quality, score_7, safe, 1girl, solo, reading a book in a cozy library, bookshelves, "
             "window, afternoon light" if ANIMA else
             "a cozy reading nook by a large window, bookshelves, armchair, afternoon light, photograph")
GUIDE_NEG = "plants, potted plant, leaves"

# the scale each comparison runs at: methods that fix high-scale artifacts are shown at the high scale
HIGH = {"RescaleCFG", "DynamicThreshold", "MimicScale", "APG", "ADG", "EPCFG", "CFGRenorm", "CFGNorm", "MAMBOG",
        "SkimmedCFG", "AutomaticCFG", "ReinhardCFG", "PMCCFG", "GuidanceInterval", "FDG", "Govern20", "Pentachoron",
        "CorrectRescale"}
# knob values that differ from the node defaults for the showcase (each is stated under its image). The two
# norm-dependent methods are sized from a probe run of plain CFG at 1024 x 1024 (SDXL, dpmpp_2m karras, cfg 7,
# 50 steps): ||c - u|| in noise units had a median of 2.74 (1.75-3.47), so power-law omega 2.42 and beta-CFG's
# scale 19.2 give an effective scale near 7 there.
OVERRIDES = {"sdxl": {"BetaCFG": {"scale": 19.2}, "PowerLawCFG": {"omega": 2.42}},
             "anima": {"C2FG": {"rate": 0.2}}}
ANIMA_SET = ["CFGZeroStar", "APG", "ADG", "PMCCFG", "CFGRenorm", "AdaMaG", "VAGS", "FBG", "SMCCFG", "TVCFG", "C2FG",
             "AdaptiveGuidance", "GuidanceInterval", "CFGTruncation", "SEG", "STG", "Govern20", "Pentachoron"]


def get(path):
    with urllib.request.urlopen(URL + path, timeout=60) as r:
        return json.loads(r.read())


def post(path, data):
    req = urllib.request.Request(URL + path, data=json.dumps(data).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"http_error": e.code, "body": e.read().decode(errors="replace")[:2000]}


def loaders():
    if ANIMA:
        return {"unet": {"class_type": "UNETLoader", "inputs": {"unet_name": "anima-base-v1.0.safetensors", "weight_dtype": "default"}},
                "msf": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["unet", 0], "shift": 3.0}},
                "clipl": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_06b_base.safetensors", "type": "stable_diffusion"}},
                "vael": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}}}
    return {"ckpt": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"}}}


def graph(tag, patches, cfg, pos=POS, neg=NEG):
    """Loader -> patches (MODEL nodes) -> KSampler -> decode -> save."""
    g = loaders()
    g.update({"pos": {"class_type": "CLIPTextEncode", "inputs": {"text": pos, "clip": CLIP}},
              "neg": {"class_type": "CLIPTextEncode", "inputs": {"text": neg, "clip": CLIP}},
              "lat": {"class_type": "EmptyLatentImage", "inputs": {"width": A.size, "height": A.size, "batch_size": 1}}})
    prev = MODEL
    for i, (ct, inputs) in enumerate(patches):
        g[f"p{i}"] = {"class_type": ct, "inputs": dict(inputs, model=prev)}
        prev = [f"p{i}", 0]
    g["ks"] = {"class_type": "KSampler", "inputs": {"model": prev, "seed": A.seed, "steps": A.steps, "cfg": cfg,
                                                     "sampler_name": SAMPLER, "scheduler": SCHEDULER,
                                                     "positive": ["pos", 0], "negative": ["neg", 0],
                                                     "latent_image": ["lat", 0], "denoise": 1.0}}
    g["dec"] = {"class_type": "VAEDecode", "inputs": {"samples": ["ks", 0], "vae": VAE}}
    g["save"] = {"class_type": "SaveImage", "inputs": {"images": ["dec", 0], "filename_prefix": f"cfg_showcase/{A.model}/{tag}"}}
    return g


def guider_graph(tag, guider_node, cfg):
    """SamplerCustomAdvanced with a guider: ComfyUI's CFGGuider for the reference, or a paper guider."""
    g = graph(tag, [], cfg, GUIDE_POS, GUIDE_NEG)
    del g["ks"]
    g["null"] = {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": CLIP}}
    g["guider"] = guider_node
    g["noise"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": A.seed}}
    g["sampler"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": SAMPLER}}
    g["sigmas"] = {"class_type": "BasicScheduler", "inputs": {"model": MODEL, "scheduler": SCHEDULER, "steps": A.steps, "denoise": 1.0}}
    g["ks"] = {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["noise", 0], "guider": ["guider", 0],
                                                                "sampler": ["sampler", 0], "sigmas": ["sigmas", 0],
                                                                "latent_image": ["lat", 0]}}
    return g


def paper_inputs(p, extra):
    vals = {k.name: k.default for k in p.knobs}
    if p.stage == "combine":
        vals.update(scale=-1.0, space=p.space)
    vals.update(extra)
    return vals


def stage_cases():
    """A few stage nodes, as (key, title, source, patches, extra text)."""
    gov = {"max_angle_degrees": 20.0, "min_angle_degrees": 0.0, "unit": "whole image (one vector per image)",
           "home": "conditional (how far guidance turns the prediction)", "start_percent": 0.0, "end_percent": 1.0,
           "space": "auto (the method's own)"}
    corr = {"method": "rescale_std", "strength": 0.7, "cap_ratio": 1.05, "mimic_scale": 4.0, "percentile": 0.995,
            "softness": 1.0, "space": "auto (the method's own)"}
    return {
        "Govern20": ("CFG Govern: Angle Band, 20 degrees", "in-house: after the AlephLM anchor governor",
                     [("CFGP_GovernAngle", gov)], "max 20 deg, whole image"),
        "Pentachoron": ("CFG Mix: Pentachoron", "in-house: the aleph weighting on the 4-simplex",
                        [("CFGP_MixPentachoron", {"k": 1.0, "scale": -1.0, "space": "noise (eps)"})], "k = 1"),
        "CorrectRescale": ("CFG Correct: Magnitude, rescale_std", "Lin et al. 2024 (guidance rescale as a correction)",
                           [("CFGP_Correct", corr)], "strength 0.7"),
    }


def run(g, tag):
    t0 = time.time()
    r = post("/prompt", {"prompt": g, "client_id": CLIENT})
    if "prompt_id" not in r:
        raise RuntimeError(f"{tag}: {r}")
    pid = r["prompt_id"]
    while True:
        time.sleep(0.5)
        h = get(f"/history/{pid}")
        if pid in h:
            break
        if time.time() - t0 > 1800:
            raise RuntimeError(f"{tag}: timeout")
    st = h[pid].get("status", {})
    if st.get("status_str") != "success":
        msgs = [m for m in st.get("messages", []) if m[0] in ("execution_error", "execution_interrupted")]
        raise RuntimeError(f"{tag}: {msgs[-1][1].get('exception_message') if msgs else st}")
    im = h[pid]["outputs"]["save"]["images"][0]
    return os.path.join(OUTPUT, im.get("subfolder", ""), im["filename"]), time.time() - t0


def rel(path):
    """A render's path inside the server's output folder (the record carries no local paths)."""
    return os.path.relpath(path, OUTPUT).replace(os.sep, "/")


def font(size):
    for f in ("C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(f):
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def compose(panels, title, source, out):
    """Renders at 512 px side by side (panels = [(path, label), ...]), a caption band above: the method and its
    source."""
    w, gap, band, lab = 512, 8, 64, 26
    canvas = Image.new("RGB", (len(panels) * w + (len(panels) - 1) * gap, band + w + lab), (250, 250, 248))
    d = ImageDraw.Draw(canvas)
    d.text((10, 8), title, fill=(20, 20, 20), font=font(21))
    d.text((10, 36), source, fill=(90, 90, 90), font=font(15))
    for i, (path, label) in enumerate(panels):
        x = i * (w + gap)
        canvas.paste(Image.open(path).convert("RGB").resize((w, w), Image.LANCZOS), (x, band))
        d.text((x + 10, band + w + 4), label, fill=(70, 70, 70), font=font(15))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    canvas.save(out, "JPEG", quality=86, optimize=True)


def main():
    only = [k for k in A.only.split(",") if k]
    img_dir = os.path.join(A.docs, "images", A.model)
    graph_dir = os.path.join(A.docs, "showcase", A.model)
    os.makedirs(graph_dir, exist_ok=True)
    stages = stage_cases()
    keys = [p.key for p in papers.PAPERS if p.key != "CFG"] + list(stages)
    if ANIMA:
        keys = [k for k in keys if k in ANIMA_SET]
    if only:
        keys = [k for k in keys if k in only]
    info = {"model": A.model, "size": A.size, "steps": A.steps, "seed": A.seed, "sampler": f"{SAMPLER} / {SCHEDULER}",
            "prompt": POS, "negative": NEG, "guider_prompt": GUIDE_POS, "guider_negative": GUIDE_NEG, "renders": {}}
    record = os.path.join(A.docs, "showcase", f"{A.model}_renders.json")
    if only and os.path.exists(record):           # a partial run updates the record instead of replacing it
        with open(record, encoding="utf-8") as f:
            info["renders"] = json.load(f).get("renders", {})
    refs = {}
    for cfg in sorted({HIGH_CFG if k in HIGH else BASE_CFG for k in keys}):
        refs[cfg], secs = run(graph(f"plain_cfg{cfg:g}", [], cfg), f"plain {cfg}")
        info["renders"][f"plain_cfg{cfg:g}"] = {"file": rel(refs[cfg]), "seconds": round(secs, 1)}
        print(f"plain cfg {cfg:g}: {secs:.1f}s", flush=True)
    guider_refs = None
    for key in keys:
        cfg = HIGH_CFG if key in HIGH else BASE_CFG
        extra = OVERRIDES[A.model].get(key, {})
        if key in stages:
            title, source, patches, note = stages[key]
            g = graph(key, patches, cfg)
        else:
            p = papers.BY_KEY[key]
            title, source = p.title, f"{p.cite} | {p.link}"
            note = ", ".join(f"{k} {v}" for k, v in extra.items())
            if p.stage == "guider":
                if guider_refs is None:       # the prompt without a negative, then ComfyUI's own negative prompt
                    guider_refs = []
                    for tag, neg_ref in (("guider_no_negative", "null"), ("guider_negative", "neg")):
                        ref_node = {"class_type": "CFGGuider", "inputs": {"model": MODEL, "positive": ["pos", 0],
                                                                           "negative": [neg_ref, 0], "cfg": BASE_CFG}}
                        path, secs = run(guider_graph(tag, ref_node, BASE_CFG), tag)
                        info["renders"][tag] = {"file": rel(path), "seconds": round(secs, 1)}
                        guider_refs.append(path)
                node = {"class_type": f"CFGP_{key}", "inputs": dict(paper_inputs(p, extra), model=MODEL, positive=["pos", 0],
                                                                    negative=["neg", 0], null=["null", 0], cfg=cfg,
                                                                    space=p.space)}
                g = guider_graph(key, node, cfg)
            else:
                g = graph(key, [(f"CFGP_{key}", paper_inputs(p, extra))], cfg)
        try:
            path, secs = run(g, key)
        except RuntimeError as e:
            print(f"ERR {key}: {e}", flush=True)
            info["renders"][key] = {"error": str(e)[:500]}
            continue
        with open(os.path.join(graph_dir, f"{key}.json"), "w", encoding="utf-8") as f:
            json.dump(g, f, indent=1)
        guider = key in papers.BY_KEY and papers.BY_KEY[key].stage == "guider"
        # a paper node is named before its colon (APG: ...); a stage node after it (CFG Mix: Pentachoron)
        short = title.split(": ", 1)[1].split(",")[0] if title.startswith("CFG ") and ": " in title else title.split(":")[0]
        right_label = f"{short.split(' (')[0]}, cfg {cfg:g}" + (f" ({note})" if note else "")
        if guider:
            panels = [(guider_refs[0], f"cfg {BASE_CFG:g}, no negative"),
                      (guider_refs[1], f"cfg {BASE_CFG:g}, negative prompt: {GUIDE_NEG}"),
                      (path, right_label)]
        else:
            panels = [(refs[cfg], f"plain CFG, cfg {cfg:g}"), (path, right_label)]
        compose(panels, title, source, os.path.join(img_dir, f"{key}.jpg"))
        info["renders"][key] = {"file": rel(path), "seconds": round(secs, 1), "cfg": cfg, "overrides": extra}
        print(f"ok  {key:22s} cfg {cfg:<4g} {secs:5.1f}s", flush=True)
    with open(record, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=1)
    bad = [k for k, v in info["renders"].items() if "error" in v]
    print(f"\n{len(keys) - len(bad)}/{len(keys)} comparisons written to {img_dir}; errors: {bad or 'none'}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
