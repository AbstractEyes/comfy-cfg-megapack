"""get_anima.py - download CircleStone Labs' Anima (public; non-commercial licence, read it on the model page) into
a ComfyUI's model folders: base v1.0, the Qwen3 0.6B text encoder, the Qwen-Image VAE and the turbo LoRA v0.2.
No token needed (the repos are public); every file is checked against the hub's sha256; files already in place
and matching are skipped.
Run:  python tools/get_anima.py --comfy <your ComfyUI folder>"""
import hashlib
import os
import shutil

from huggingface_hub import HfApi, hf_hub_download

import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--comfy", required=True, help="the ComfyUI folder (the one holding models/)")
A = ap.parse_args()
MODELS = os.path.join(os.path.abspath(A.comfy), "models")
assert os.path.isdir(MODELS), f"no models folder under {A.comfy}"
STAGE = os.path.join(os.path.abspath(A.comfy), "downloads_tmp")
FILES = [
    ("circlestone-labs/Anima", "split_files/diffusion_models/anima-base-v1.0.safetensors", "diffusion_models"),
    ("circlestone-labs/Anima", "split_files/text_encoders/qwen_3_06b_base.safetensors", "text_encoders"),
    ("circlestone-labs/Anima", "split_files/vae/qwen_image_vae.safetensors", "vae"),
    ("circlestone-labs/Anima-Official-LoRAs", "anima-turbo-lora-v0.2.safetensors", "loras"),
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


api = HfApi(token=False)
for repo, name, folder in FILES:
    meta = {s.rfilename: s for s in api.model_info(repo, files_metadata=True).siblings}[name]
    want = meta.lfs.sha256 if meta.lfs else None
    dest = os.path.join(MODELS, folder, os.path.basename(name))
    if os.path.exists(dest) and os.path.getsize(dest) == meta.size and sha256(dest) == want:
        print(f"present  {dest}")
        continue
    path = hf_hub_download(repo, name, local_dir=STAGE, token=False)
    got = sha256(path)
    assert got == want, f"{name}: sha256 {got} != hub {want}"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.move(path, dest)
    print(f"ok       {dest}  {meta.size / 1e9:.3f} GB  sha256 {got}")
shutil.rmtree(STAGE, ignore_errors=True)
print("done")
