"""determinism_check.py - is ComfyUI's own sampling repeatable run to run, and do the neutral CFG Megapack settings
match it? Clears ComfyUI's execution cache before every run so each graph really samples again.
Then the A/B shape of the example workflows: two samplers in one prompt on one loaded model (plain CFG and the
neutral standard rule), right after a load and again after one warm-up run.
Usage (ComfyUI running):  python tools/determinism_check.py --port 8189 --steps 6"""
import argparse
import json
import os
import sys
import time
import urllib.request

import numpy as np
from PIL import Image

sys.argv_backup = list(sys.argv)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8189)
ap.add_argument("--steps", type=int, default=6)
ap.add_argument("--size", type=int, default=128)
ap.add_argument("--output-dir", default="", help="the server's --output-directory, if it was given one")
args = ap.parse_args()
sys.argv = [sys.argv[0], "--port", str(args.port), "--steps", str(args.steps), "--size", str(args.size)]
if args.output_dir:
    sys.argv += ["--output-dir", args.output_dir]
import smoke_test as st  # noqa: E402  (reuses the graph builders; parses the same flags)

URL = f"http://127.0.0.1:{args.port}"


def free():
    req = urllib.request.Request(URL + "/free", data=json.dumps({"free_memory": True}).encode(),
                                 headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=30).read()
    time.sleep(0.5)


def img(res):
    return np.asarray(Image.open(res["path"]).convert("RGB")).astype(np.int16)


runs = [("plain_1", st.chain("det_plain_1", [])), ("plain_2", st.chain("det_plain_2", [])),
        ("plain_3", st.chain("det_plain_3", [])),
        ("standard_rule", st.chain("det_standard", [st.mix_scale("standard")])),
        ("formula_x0", st.chain("det_formula_x0", [st.formula("u + w * (c - u)", "denoised (x0)")])),
        ("formula_eps", st.chain("det_formula_eps", [st.formula("u + w * (c - u)")])),
        ("guider_builtin_1", st.reference_guider_graph("det_guider_builtin_1")),
        ("guider_builtin_2", st.reference_guider_graph("det_guider_builtin_2")),
        ("guider_negative_as_null", st.guider_graph("det_guider_nan", "negative_as_null"))]
res = {}
for name, g in runs:
    free()
    res[name] = st.run_case(name, g)
    print(f"{name:26s} ok={res[name]['ok']} {res[name].get('seconds')}s")


def cmp(a, b):
    d = np.abs(img(res[a]) - img(res[b]))
    print(f"  {a:24s} vs {b:22s} max {int(d.max()):3d}  mean {d.mean():.3f}  identical pixels {100 * (d.max(axis=2) == 0).mean():.1f}%")


print(f"\nsteps {args.steps}, {args.size}x{args.size}:")
for a, b in [("plain_2", "plain_1"), ("plain_3", "plain_2"), ("standard_rule", "plain_2"), ("formula_x0", "plain_2"),
             ("formula_eps", "plain_2"), ("guider_builtin_2", "guider_builtin_1"), ("guider_builtin_1", "plain_2"),
             ("guider_negative_as_null", "guider_builtin_2")]:
    cmp(a, b)


def ab_graph(tag, seed):
    """One prompt, one loader, two samplers: side a = plain CFG, side b = the neutral standard rule."""
    g = st.base_graph(tag)
    del g["dec"], g["save"]
    g["p0"] = {"class_type": st.MIXS, "inputs": dict(st.mix_scale("standard")[1], model=st.MODEL)}
    for side, model in (("a", st.MODEL), ("b", ["p0", 0])):
        g[f"ks_{side}"] = st.ksampler(model)
        g[f"ks_{side}"]["inputs"]["seed"] = seed
        g[f"dec_{side}"] = {"class_type": "VAEDecode", "inputs": {"samples": [f"ks_{side}", 0], "vae": st.VAE}}
        g[f"save_{side}"] = {"class_type": "SaveImage", "inputs": {"images": [f"dec_{side}", 0],
                                                                  "filename_prefix": f"cfgp_det/{tag}_{side}"}}
    return g


def run_ab(tag, seed):
    r = st.post("/prompt", {"prompt": ab_graph(tag, seed), "client_id": st.CLIENT})
    pid, t0 = r["prompt_id"], time.time()
    while True:
        time.sleep(0.25)
        h = st.get(f"/history/{pid}")
        if pid in h:
            break
        if time.time() - t0 > 900:
            raise TimeoutError(tag)
    entry = h[pid]
    if entry.get("status", {}).get("status_str") != "success":
        raise RuntimeError(f"{tag}: {entry.get('status')}")
    pics = {}
    for side in ("a", "b"):
        im = entry["outputs"][f"save_{side}"]["images"][0]
        path = os.path.join(st.OUTPUT, im.get("subfolder", ""), im["filename"])
        pics[side] = np.asarray(Image.open(path).convert("RGB")).astype(np.int16)
    return pics


def cmp_pics(label, x, y):
    d = np.abs(x - y)
    print(f"  {label:52s} max {int(d.max()):3d}  mean {d.mean():.3f}")


free()
first = run_ab("ab_after_load", st.A.seed)       # one of the two samplers is the first sampling after the load
run_ab("ab_warmup", st.A.seed + 1)                # the warm-up: any other seed, so the next queue samples again
warm = run_ab("ab_warm", st.A.seed)
fresh = img(res["plain_2"])                       # a first-sampling-after-load plain image (from the runs above)
print("\nA/B in one prompt (plain CFG = a, neutral standard rule = b):")
cmp_pics("right after a load: a vs b", first["a"], first["b"])
cmp_pics("after one warm-up: a vs b", warm["a"], warm["b"])
cmp_pics("after one warm-up a vs right-after-load a", warm["a"], first["a"])
cmp_pics("after one warm-up a vs right-after-load b", warm["a"], first["b"])
cmp_pics("right-after-load a vs a fresh-load plain run", first["a"], fresh)
cmp_pics("right-after-load b vs a fresh-load plain run", first["b"], fresh)
