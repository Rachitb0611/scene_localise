import os
import sys
import subprocess
import requests

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WEIGHTS_DIR = os.path.join(ROOT, "weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)

REPOS = [
    ("GroundingDINO", "https://github.com/IDEA-Research/GroundingDINO.git"),
    ("CLIP", "https://github.com/openai/CLIP.git"),
    ("segment-anything", "https://github.com/facebookresearch/segment-anything.git"),
]

# Known public checkpoints
URLS = {
    "groundingdino_swint_ogc.pth": "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
    "sam_vit_h_4b8939.pth": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
}

def run(cmd, cwd=None):
    print(f"$ {cmd}")
    proc = subprocess.Popen(cmd, shell=True, cwd=cwd)
    proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")

def safe_git_clone(target_dir, url):
    if os.path.exists(os.path.join(ROOT, target_dir, ".git")):
        print(f"[skip] {target_dir} already cloned.")
        return
    print(f"[clone] {target_dir} from {url}")
    run(f"git clone {url} {target_dir}", cwd=ROOT)

def download(url, out_path):
    if os.path.exists(out_path):
        print(f"[skip] {os.path.basename(out_path)} already exists.")
        return
    try:
        print(f"[download] {url} -> {out_path}")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(out_path, "wb") as f:
                downloaded = 0
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total > 0:
                            done = int(50 * downloaded / total)
                            sys.stdout.write("\r[%s%s] %d%%" % (
                                "=" * done, " " * (50 - done),
                                100 * downloaded / total
                            ))
                            sys.stdout.flush()
        print("\n[ok] Downloaded.")
    except Exception as e:
        print(f"[warn] Could not download automatically: {e}")
        print(f"       Please download manually and place it at: {out_path}")

def main():
    print("== Clone upstream repositories ==")
    for name, url in REPOS:
        safe_git_clone(name, url)

    print("\n== Download model checkpoints ==")
    for fname, url in URLS.items():
        download(url, os.path.join(WEIGHTS_DIR, fname))

    print("\nAll set. Next steps:")
    print("1) Activate your virtualenv")
    print("2) pip install -r requirements.txt")
    print("3) python main_1.py")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"[fatal] {e}")
        sys.exit(1)
