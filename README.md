# Scene Grounding in Dense Visual Environments (CPU-only)

This repository contains a **CPU-only** pipeline for grounding natural language queries in dense images.  
It uses **GroundingDINO** for text-conditioned detection, optional **CLIP** re‑ranking, tiling for large images, and **SAM** for fine-grained segmentation of the selected region.

> ✅ You selected this approach as your final pipeline. This repo is structured so others can reproduce it easily on their machines, including cloning the required upstream repos and downloading the model weights via a single Python script.

---

## Quick Start

### 0) Clone this repository
```bash
git clone <your_repo_url>.git
cd scene-grounding-cpu
```

### 1) Create & activate a virtual environment
```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2) Install Python dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3) Clone required repos + download weights (GroundingDINO, SAM, OpenAI CLIP)
```bash
python scripts/fetch_assets.py
```
This will create/refresh:
- `GroundingDINO/` (git clone)
- `CLIP/` (OpenAI CLIP git clone)
- `segment-anything/` (SAM clone, optional—but we still download the SAM weight)
- `weights/groundingdino_swint_ogc.pth`
- `weights/sam_vit_h_4b8939.pth`

> If any URL is blocked in your environment, the script will suggest a manual backup URL and where to place the files.

### 4) Run the demo
```bash
# Provide an input image path when prompted (or place a sample in ./data/img.png)
python main_1.py
```
The script saves outputs under `results/` including: the best bounding box, SAM mask, and cropped region.

---

## Basic Idea (High Level)

1. **GroundingDINO**: Given an image and a natural-language text query, it proposes boxes that likely match the text.
2. **(Optional) CLIP Re-ranking**: Cropped regions are scored against the query using CLIP; scores are fused with detector confidence to pick the best box.
3. **Tiling (Optional)**: For very large images, we run detection per tile and fuse boxes (e.g., with Soft‑NMS / WBF).
4. **SAM Segmentation**: Use the final box as a prompt to SAM to obtain a tight mask of the object/region, then save overlays and RGBA cutouts.

---

## Installation Guide (Detailed)

1. **Python 3.9+ recommended**  
   Create a venv and install the exact requirements from `requirements.txt` for reproducibility.
2. **Assets Fetcher**  
   Run `python scripts/fetch_assets.py` to:
   - `git clone` GroundingDINO, OpenAI CLIP, and Segment-Anything (SAM) if not already present.
   - Download GroundingDINO weights and SAM ViT‑H weights to `./weights/`.
3. **First Run**  
   - Place a test image at `./data/img.png` or provide a path when prompted.
   - Run `python main_1.py` and follow prompts for the query and tiling option.

---

## Notes

- The provided `main_1.py` is CPU-only and includes optional tiling and a CLIP re‑ranker path.  
- If you change the weights’ filenames or locations, update `main_1.py` accordingly.
- If your environment blocks the default download URLs, place the weights manually into `./weights/` and rerun.

---

## Repository Layout

```
scene-grounding-cpu/
├─ main_1.py                # CPU pipeline (GroundingDINO + optional CLIP + SAM)
├─ requirements.txt
├─ scripts/
│  └─ fetch_assets.py       # Clones repos & downloads model weights
├─ docs/
│  ├─ REPORT.md             # ≤7 pages tech report with image placement guidance
│  └─ images/               # Put figures here (pipeline, architecture, examples)
├─ data/                    # (You provide) test images
├─ results/                 # Outputs (auto-created)
├─ weights/                 # Downloaded model weights
└─ .gitignore
```

---

## Troubleshooting

- **NumPy / Torch ABI mismatch**: If you see errors about NumPy versions, pin NumPy to a compatible version mentioned by PyTorch (or try `pip install numpy==1.26.*`).  
- **Torchvision NMS import**: If `torchvision.ops.nms` fails on CPU-only wheels, the code falls back to Soft‑NMS.  
- **CLIP not found**: The pipeline runs without CLIP (re‑ranking will be skipped). Ensure `clip-anytorch` is installed or use the cloned OpenAI CLIP package.

---

## Citation / Repositories

- GroundingDINO by IDEA-Research
- OpenAI CLIP
- Segment Anything (SAM) by Meta AI