"""make_docs.py - fill the generated parts of README.md (the paper table) and HOWTO.md (one section per paper node)
from cfg_megapack/papers.py and the comparison images in docs/images. Run after adding a paper or rendering:
  python tools/make_docs.py"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from cfg_megapack import papers  # noqa: E402

# What an image shows, where it would otherwise look like a fault (measured on the showcase settings)
NOTES = {
    "sdxl": {
        "ADG": "The formula matches the official code, and the faded color is what it does on SDXL at the paper's "
               "60-degree cap: the two denoised estimates start 45 degrees apart and stay more than 4.6 degrees apart "
               "for about the first 10 steps, so at cfg 14 the cap binds there, and each capped step keeps only half "
               "(cos 60) of the conditional estimate. A 30-degree cap keeps more color. On Anima, a flow model like "
               "the paper's SD3.5, the default keeps its color (below).",
        "AdaptiveGuidance": "Guidance stops after 18 of the 50 steps, when the cosine of the two denoised predictions "
                            "passes 0.999; the rest of the run is conditional only.",
        "SMCCFG": "k auto = 0.01 here. The paper's 0.2 wrecks SDXL images (tested with dpmpp_2m and with euler): on "
                  "the noise prediction the correction moves the denoised estimate by sigma x cfg x k per element, and "
                  "SDXL's sigma starts at 14.6. On Anima the paper's value works (below).",
        "FBG": "FBG sets its own scale; the sampler's cfg is not used. The node starts at the paper's settings for "
               "its Stable Diffusion images (pi 0.85, t0 0.75, t1 0.5); the code's defaults (0.95, 0.5, 0.4) leave "
               "the first steps almost unguided and wash the image out on SDXL.",
        "PMCCFG": "At gamma_cap 1.05 the layout and color come out close to unguided on SDXL. The paper names this "
                  "limit: the cap holds guidance near zero where the conditional estimate is small, as in the first "
                  "steps. A larger gamma_cap guides more; on Anima the default guides normally (below).",
        "HiWave": "Used alone, only the detail bands are guided and the coarse band (layout and color) runs at scale "
                  "1, so layout and color come out unguided. In the paper the layout comes from a base image that is "
                  "upscaled and inverted first; this node is the guidance rule of that pipeline.",
        "TransitionPoint": "No guidance until the difference between the predictions passes its first low point: "
                           "the composition forms unguided (the method guards against reproducing memorized "
                           "images), so layout and color differ from plain CFG.",
        "GuidanceInterval": "Guidance is off at the highest noise levels, so the composition forms with less "
                            "guidance: a different layout from plain CFG at the same detail, the paper's intended "
                            "effect (more variety).",
        "WindowedNegative": "The negative acts only from 17% to 50% of the run, after the layout has formed: the "
                            "plants go and the layout of the run without a negative stays.",
        "SafeLatentDiffusion": "From the 10th step on it steers away from the concept only where the image moves "
                               "toward it: the plants go and the layout of the run without a negative stays.",
        "PerpNeg": "Only the part of the negative that the prompt does not share is removed. This prompt does not ask "
                   "for plants, so they go, much as with the plain negative prompt.",
    },
    "anima": {
        "SMCCFG": "k auto = 0.2, the paper's own value, on this flow model.",
        "AdaptiveGuidance": "Guidance stops after 25 of the 50 steps (the cosine of the denoised predictions passes "
                            "0.999).",
        "FBG": "FBG sets its own scale; the sampler's cfg is not used (the paper's Stable Diffusion settings).",
    },
}

LINE_ORDER = ["combine", "when", "weak", "negatives", "frequency"]
LINE_TITLES = {"combine": "Combining the two predictions", "when": "When to guide",
               "weak": "Weak branch (a degraded pass of the model itself)", "negatives": "Negative prompts (guiders)",
               "frequency": "Frequency and space"}


def first_sentence(text):
    m = re.match(r"(.+?\.)(\s|$)", text.strip())
    return m.group(1) if m else text.strip()


def anchor(title):
    a = title.strip().lower()
    a = re.sub(r"[^\w\- ]", "", a)
    return a.replace(" ", "-")


def fill(path, tag, body):
    s = open(path, encoding="utf-8").read()
    begin, end = f"<!-- {tag}:begin -->", f"<!-- {tag}:end -->"
    i, j = s.index(begin) + len(begin), s.index(end)
    s = s[:i] + "\n" + body.rstrip() + "\n" + s[j:]
    open(path, "w", encoding="utf-8", newline="\n").write(s)


def by_line():
    for line in LINE_ORDER:
        yield line, [p for p in papers.PAPERS if p.line == line]


def readme_table():
    out = []
    for line, ps in by_line():
        out += [f"**{LINE_TITLES[line]}** ({len(ps)})", "", "| Node | Source | What it does |", "|---|---|---|"]
        for p in ps:
            src = f"[{p.cite}]({p.link})"
            out.append(f"| [{p.title}](HOWTO.md#{anchor(p.title)}) | {src} | {first_sentence(p.summary)} |")
        out.append("")
    return "\n".join(out)


def space_label(p):
    return "auto, the method's own" if p.space.startswith("auto") else p.space


def knob_text(k):
    if k.kind == "combo":
        return f"`{k.name}` ({', '.join(k.options)}; default {k.default})"
    return f"`{k.name}` (default {k.default})"


def howto_sections():
    out = []
    for line, ps in by_line():
        out += [f"### {LINE_TITLES[line]}", ""]
        for p in ps:
            out += [f"#### {p.title}", "", p.summary, ""]
            items = [knob_text(k) + (f": {k.tip}" if k.tip else "") for k in p.knobs]
            if p.stage == "combine":
                items = ["`scale`: the guidance scale (-1 = the sampler's cfg)",
                         f"`space`: where the rule is computed (default: {space_label(p)})"] + items
                out.append("Inputs:")
            elif p.stage == "guider":
                items = ["`positive`, `negative`, `null` (an empty prompt), `cfg`",
                         f"`space`: where the rule is computed (default: {space_label(p)})"] + items
                out.append("A guider for SamplerCustomAdvanced. Inputs:")
            elif items:
                out.append("Inputs:")
            if items:
                out += [""] + [f"- {i}" for i in items]
            out += ["", f"Source: {p.cite}, <{p.link}>. Node id `CFGP_{p.key}`.", ""]
            for model in ("sdxl", "anima"):
                img = os.path.join("docs", "images", model, f"{p.key}.jpg")
                if os.path.exists(os.path.join(ROOT, img)):
                    graph = f"docs/showcase/{model}/{p.key}.json"
                    name = "SDXL" if model == "sdxl" else "Anima"
                    note = NOTES.get(model, {}).get(p.key)
                    out += [f"![{p.title} on {name}]({img.replace(os.sep, '/')})", "",
                            f"<sub>{name}; node graph: [{graph}]({graph})" + (f". {note}" if note else "") + "</sub>", ""]
    return "\n".join(out)


fill(os.path.join(ROOT, "README.md"), "papers-table", readme_table())
fill(os.path.join(ROOT, "HOWTO.md"), "paper-sections", howto_sections())
n_img = sum(len(fs) for _, _, fs in os.walk(os.path.join(ROOT, "docs", "images")))
print(f"README table: {len(papers.PAPERS)} paper nodes; HOWTO sections written; {n_img} comparison images on disk")
