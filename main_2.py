# improved_groundingdino_cpu_with_sam.py
import os
import sys
import urllib.request
import torch
import cv2
import numpy as np
import re
from PIL import Image
import traceback
import time
import math
from datetime import datetime

# Try import supervision, CLIP, torchvision nms, SAM
try:
    import supervision as sv
except Exception:
    sv = None

try:
    import clip
except Exception:
    clip = None

try:
    from torchvision.ops import nms as torch_nms
except Exception:
    torch_nms = None

try:
    from segment_anything import sam_model_registry, SamPredictor
except Exception:
    SamPredictor = None
    sam_model_registry = None

DEVICE = torch.device("cpu")  # explicit CPU

# ---------------------------
# Config: negative-query handling
# ---------------------------
NEGATIVE_QUERY_HANDLING = True
CLIP_RELEVANCE_THRESHOLD = 0.24       # cosine sim threshold for full-image vs query
DINO_STRICT_PROBE_THRESH = 0.35       # fallback probe box threshold if CLIP unavailable
RESULTS_ROOT = "results"

# ---------------------------
# IO utils
# ---------------------------
def download_file(url, save_path):
    print(f"Downloading {url} to {save_path}...")
    try:
        with urllib.request.urlopen(url) as response, open(save_path, 'wb') as out_file:
            out_file.write(response.read())
        print("Download complete.")
    except Exception as e:
        print(f"Error downloading {url}: {e}")

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def crop_rgba_with_mask(bgr_img, mask_bool):
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    alpha = (mask_bool.astype(np.uint8) * 255)
    rgba = np.dstack([rgb, alpha])
    return Image.fromarray(rgba, mode="RGBA")

# ---------------------------
# Soft-NMS (numpy)
# ---------------------------
def soft_nms_np(boxes, scores, sigma=0.5, score_thresh=0.001, method='linear'):
    boxes = boxes.astype(np.float32).copy()
    scores = scores.copy().astype(np.float32)
    N = boxes.shape[0]
    idxs = np.arange(N)
    keep = []
    while idxs.size > 0:
        max_idx = np.argmax(scores[idxs])
        cur = idxs[max_idx]
        keep.append(cur)
        cur_box = boxes[cur]
        rest = boxes[idxs]
        xx1 = np.maximum(cur_box[0], rest[:, 0])
        yy1 = np.maximum(cur_box[1], rest[:, 1])
        xx2 = np.minimum(cur_box[2], rest[:, 2])
        yy2 = np.minimum(cur_box[3], rest[:, 3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        area_cur = (cur_box[2] - cur_box[0]) * (cur_box[3] - cur_box[1])
        area_rest = (rest[:, 2] - rest[:, 0]) * (rest[:, 3] - rest[:, 1])
        union = area_cur + area_rest - inter
        ious = np.zeros_like(inter)
        union_pos = union > 0
        ious[union_pos] = inter[union_pos] / union[union_pos]
        for i_pos, iou in enumerate(ious):
            idx_global = idxs[i_pos]
            if idx_global == cur:
                continue
            if method == 'linear':
                if iou > 0.3:
                    scores[idx_global] = scores[idx_global] * (1 - iou)
            else:
                scores[idx_global] = scores[idx_global] * math.exp(-(iou * iou) / sigma)
        scores[cur] = -1
        idxs = idxs[scores[idxs] > score_thresh]
    return keep

# ---------------------------
# Simple Weighted Boxes Fusion
# ---------------------------
def simple_wbf(boxes_list, scores_list, labels_list=None, iou_thresh=0.5, skip_box_thr=0.0):
    boxes_all = []
    scores_all = []
    for boxes, scores in zip(boxes_list, scores_list):
        for b, s in zip(boxes, scores):
            if s >= skip_box_thr:
                boxes_all.append(b)
                scores_all.append(s)
    if not boxes_all:
        return [], []
    boxes_arr = np.array(boxes_all)
    scores_arr = np.array(scores_all)
    used = np.zeros(len(boxes_arr), dtype=bool)
    fused_boxes = []
    fused_scores = []
    for i in np.argsort(-scores_arr):
        if used[i]:
            continue
        cluster_idx = [i]
        used[i] = True
        for j in range(len(boxes_arr)):
            if used[j]:
                continue
            x1 = max(boxes_arr[i, 0], boxes_arr[j, 0])
            y1 = max(boxes_arr[i, 1], boxes_arr[j, 1])
            x2 = min(boxes_arr[i, 2], boxes_arr[j, 2])
            y2 = min(boxes_arr[i, 3], boxes_arr[j, 3])
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            inter = w * h
            area_i = (boxes_arr[i, 2] - boxes_arr[i, 0]) * (boxes_arr[i, 3] - boxes_arr[i, 1])
            area_j = (boxes_arr[j, 2] - boxes_arr[j, 0]) * (boxes_arr[j, 3] - boxes_arr[j, 1])
            union = area_i + area_j - inter
            iou = inter / union if union > 0 else 0
            if iou >= iou_thresh:
                cluster_idx.append(j)
                used[j] = True
        cluster_boxes = boxes_arr[cluster_idx]
        cluster_scores = scores_arr[cluster_idx]
        weights = cluster_scores / (cluster_scores.sum() + 1e-12)
        fused_box = (cluster_boxes * weights[:, None]).sum(axis=0)
        fused_score = cluster_scores.max()
        fused_boxes.append(fused_box.tolist())
        fused_scores.append(float(fused_score))
    return fused_boxes, fused_scores

# ---------------------------
# CLIP helpers (CPU)
# ---------------------------
class CLIPRerankerCPU:
    def __init__(self, device=DEVICE, model_name="ViT-B/32"):
        self.device = device
        self.model = None
        self.preprocess = None
        if clip is not None:
            try:
                self.model, self.preprocess = clip.load(model_name, device=str(device))
                self.model.eval()
            except Exception as e:
                print("Failed to load CLIP:", e)
                self.model = None

    def is_available(self):
        return (self.model is not None) and (self.preprocess is not None)

    def crop_and_preprocess_batch(self, image, boxes):
        imgs = []
        h, w = image.shape[:2]
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            cropped = image[y1:y2, x1:x2]
            cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(cropped)
            tensor = self.preprocess(pil)
            imgs.append(tensor)
        if len(imgs) == 0:
            return None
        return torch.stack(imgs).to(self.device)

    def full_image_similarity(self, image_bgr, query_text):
        if not self.is_available():
            return None
        try:
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            img_tensor = self.preprocess(pil).unsqueeze(0).to(self.device)
            with torch.no_grad():
                img_feat = self.model.encode_image(img_tensor)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                text_tokens = clip.tokenize([query_text]).to(self.device)
                txt_feat = self.model.encode_text(text_tokens)
                txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
                sim = (txt_feat @ img_feat.T).squeeze().item()  # cosine similarity in [-1, 1]
            return float(sim)
        except Exception as e:
            print("Error in CLIP full-image similarity:", e)
            return None

    def calculate_similarities(self, image, boxes, query_text):
        if not self.is_available() or len(boxes) == 0:
            return np.array([])
        try:
            batch = self.crop_and_preprocess_batch(image, boxes)
            with torch.no_grad():
                image_features = self.model.encode_image(batch)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_tokens = clip.tokenize([query_text]).to(self.device)
                text_features = self.model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                sims = (text_features @ image_features.T).squeeze(0).cpu().numpy()
            return sims
        except Exception as e:
            print("Error in CLIP similarity:", e)
            return np.array([])

# ---------------------------
# Tiling helper
# ---------------------------
def sliding_window_tiles(image, tile_size=800, overlap=0.2):
    h, w = image.shape[:2]
    stride_x = int(tile_size * (1 - overlap))
    stride_y = int(tile_size * (1 - overlap))
    tiles = []
    for y in range(0, max(1, h - tile_size + 1), stride_y) if h > tile_size else [0]:
        for x in range(0, max(1, w - tile_size + 1), stride_x) if w > tile_size else [0]:
            x2 = min(w, x + tile_size)
            y2 = min(h, y + tile_size)
            tile = image[y:y2, x:x2]
            tiles.append((x, y, x2, y2, tile))
    if len(tiles) == 0:
        tiles.append((0, 0, w, h, image.copy()))
    return tiles

# ---------------------------
# Post-processor
# ---------------------------
class EnhancedGroundingDINOPostProcessorCPU:
    def __init__(self, grounding_dino_model, use_clip=True):
        self.base_conf = 0.25
        self.nms_threshold = 0.6
        self.min_box_area = 64
        self.max_box_area_ratio = 0.95
        self.grounding_model = grounding_dino_model
        self.use_clip = use_clip and (clip is not None)
        self.clip_reranker = CLIPRerankerCPU(device=DEVICE) if self.use_clip else None

    def filter_and_nms(self, boxes, scores, image_shape):
        if len(boxes) == 0:
            return np.array([]), np.array([])
        boxes_arr = np.array(boxes).astype(np.float32)
        scores_arr = np.array(scores).astype(np.float32)
        h, w = image_shape
        image_area = h * w
        keep_idx = []
        for i, b in enumerate(boxes_arr):
            area = (b[2] - b[0]) * (b[3] - b[1])
            if scores_arr[i] >= self.base_conf and area >= self.min_box_area and area <= self.max_box_area_ratio * image_area:
                keep_idx.append(i)
        if not keep_idx:
            return np.array([]), np.array([])
        boxes_f = boxes_arr[keep_idx]
        scores_f = scores_arr[keep_idx]
        if torch_nms is not None:
            tb = torch.from_numpy(boxes_f)
            ts = torch.from_numpy(scores_f)
            keep = torch_nms(tb, ts, self.nms_threshold).cpu().numpy()
            return boxes_f[keep], scores_f[keep]
        keep_indices = soft_nms_np(boxes_f.copy(), scores_f.copy(), sigma=0.5, method='linear')
        return boxes_f[keep_indices], scores_f[keep_indices]

    def process_detections(self, detections, query, image):
        if getattr(detections, "xyxy", None) is None or len(detections.xyxy) == 0:
            return None
        image_shape = image.shape[:2]
        boxes = [list(map(float, b)) for b in detections.xyxy]
        scores = [float(s) for s in detections.confidence]
        boxes_nms, scores_nms = self.filter_and_nms(boxes, scores, image_shape)
        if boxes_nms.size == 0:
            return None
        boxes_list = boxes_nms.tolist()
        scores_list = scores_nms.tolist()
        if self.use_clip and self.clip_reranker and self.clip_reranker.is_available():
            sims = self.clip_reranker.calculate_similarities(image, boxes_list, query)
            if len(sims) == len(scores_list):
                combined = 0.6 * np.array(scores_list) + 0.4 * sims
                best_idx = int(np.argmax(combined))
                return np.array(boxes_list[best_idx]), float(combined[best_idx])
        best_idx = int(np.argmax(scores_list))
        return np.array(boxes_list[best_idx]), float(scores_list[best_idx])

# ---------------------------
# Relevance gate
# ---------------------------
def query_is_relevant(image_bgr, query, dino_model=None):
    """
    Returns (is_relevant: bool, reason: str, extra: dict)
    Uses CLIP full-image similarity if available; otherwise tries a strict DINO probe.
    """
    # 1) Try CLIP global similarity
    clip_rr = CLIPRerankerCPU(device=DEVICE)
    if clip_rr.is_available():
        sim = clip_rr.full_image_similarity(image_bgr, query)
        if sim is None:
            # fall back to DINO probe
            pass
        else:
            if sim >= CLIP_RELEVANCE_THRESHOLD:
                return True, f"CLIP similarity {sim:.3f} ≥ threshold {CLIP_RELEVANCE_THRESHOLD:.2f}", {"clip_sim": sim}
            else:
                return False, f"CLIP similarity {sim:.3f} below threshold {CLIP_RELEVANCE_THRESHOLD:.2f}", {"clip_sim": sim}

    # 2) Fallback: quick DINO probe (if model is available)
    if dino_model is not None:
        try:
            det = dino_model.predict_with_classes(
                image=image_bgr,
                classes=[query],
                box_threshold=DINO_STRICT_PROBE_THRESH,
                text_threshold=0.25
            )
            has_boxes = (getattr(det, "xyxy", None) is not None) and (len(det.xyxy) > 0)
            if has_boxes:
                return True, f"DINO found candidate boxes at strict threshold {DINO_STRICT_PROBE_THRESH}", {"probe_boxes": int(len(det.xyxy))}
            else:
                return False, f"No DINO boxes at strict threshold {DINO_STRICT_PROBE_THRESH}", {"probe_boxes": 0}
        except Exception as e:
            return False, f"DINO probe failed: {e}", {}

    # 3) If nothing else, accept to avoid false negatives
    return True, "No CLIP/DINO available; skipping relevance gate", {}

# ---------------------------
# Main Inference
# ---------------------------
def run_inference_on_image(image_path, model, query, use_tiling=False):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(image_path)

    # Negative-query handling (guard)
    if NEGATIVE_QUERY_HANDLING:
        ok, reason, extra = query_is_relevant(image, query, dino_model=model)
        if not ok:
            return None, image, {"rejected": True, "reason": reason, **extra}

    proc = EnhancedGroundingDINOPostProcessorCPU(model)
    if not use_tiling:
        detections = model.predict_with_classes(image=image, classes=[query], box_threshold=0.25, text_threshold=0.2)
        return proc.process_detections(detections, query, image), image, {"rejected": False}
    else:
        tiles = sliding_window_tiles(image, tile_size=1000, overlap=0.25)
        boxes_per_tile, scores_per_tile = [], []
        for (tx1, ty1, tx2, ty2, tile_img) in tiles:
            det = model.predict_with_classes(image=tile_img, classes=[query], box_threshold=0.2, text_threshold=0.15)
            if getattr(det, "xyxy", None) is None or len(det.xyxy) == 0:
                continue
            for b, s in zip(det.xyxy, det.confidence):
                abs_box = [float(b[0] + tx1), float(b[1] + ty1), float(b[2] + tx1), float(b[3] + ty1)]
                boxes_per_tile.append(abs_box)
                scores_per_tile.append(float(s))
        fused_boxes, fused_scores = simple_wbf([boxes_per_tile], [scores_per_tile], iou_thresh=0.35)
        class DummyDet:
            def __init__(self, xyxy, confidence):
                self.xyxy = np.array(xyxy)
                self.confidence = np.array(confidence)
        det_all = DummyDet(np.array(fused_boxes), np.array(fused_scores))
        return proc.process_detections(det_all, query, image), image, {"rejected": False}

# ---------------------------
# Entry
# ---------------------------
if __name__ == "__main__":
    try:
        HOME = os.getcwd()
        from groundingdino.util.inference import Model as GroundingDINOModel
        GROUNDING_DINO_CONFIG_PATH = os.path.join(HOME, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")

        weights_dir = os.path.join(HOME, "weights")
        os.makedirs(weights_dir, exist_ok=True)
        grounding_dino_weights_path = os.path.join(weights_dir, "groundingdino_swint_ogc.pth")
        if not os.path.exists(grounding_dino_weights_path):
            download_file(
                "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
                grounding_dino_weights_path
            )

        grounding_dino_model = GroundingDINOModel(
            model_config_path=GROUNDING_DINO_CONFIG_PATH,
            model_checkpoint_path=grounding_dino_weights_path,
            device=str(DEVICE)
        )

        # Load SAM
        SAM_CHECKPOINT_PATH = r"C://Users//Rachit//OneDrive//Desktop//aims_project//weights//sam_vit_h_4b8939.pth"
        if SamPredictor is None or sam_model_registry is None:
            raise RuntimeError("SAM is not installed. Install with: pip install git+https://github.com/facebookresearch/segment-anything.git")
        sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT_PATH)
        sam.to(device=DEVICE)
        sam_predictor = SamPredictor(sam)

        # CLI
        image_input = input("Enter image path (default: data/img.png): ").strip()
        if not image_input:
            SOURCE_IMAGE_PATH = os.path.join(HOME, "data", "img.png")
        else:
            SOURCE_IMAGE_PATH = image_input if os.path.isabs(image_input) else os.path.join(HOME, image_input)
        if not os.path.exists(SOURCE_IMAGE_PATH):
            sys.exit("Image not found.")

        QUERY = input("Enter your query: ").strip()
        USE_TILING = input("Use tiling for dense images? (y/n, default=n): ").strip().lower() in ['y', 'yes']

        (result, img, meta) = run_inference_on_image(SOURCE_IMAGE_PATH, grounding_dino_model, QUERY, use_tiling=USE_TILING)

        # Prepare base result folder
        img_name = os.path.splitext(os.path.basename(SOURCE_IMAGE_PATH))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = ensure_dir(os.path.join(HOME, RESULTS_ROOT, f"{img_name}_{timestamp}"))

        if meta.get("rejected", False):
            # Relevance rejected → write reason & preview thumb
            with open(os.path.join(base_dir, "rejected.txt"), "w", encoding="utf-8") as f:
                f.write(f"Query rejected as not related to the image.\nReason: {meta.get('reason','')}\n")
                if "clip_sim" in meta:
                    f.write(f"CLIP similarity: {meta['clip_sim']:.4f} (threshold {CLIP_RELEVANCE_THRESHOLD})\n")
            # Save small preview
            preview = cv2.resize(img, (min(640, img.shape[1]), int(img.shape[0] * min(640, img.shape[1]) / img.shape[1])))
            cv2.imwrite(os.path.join(base_dir, "preview.jpg"), preview)
            print(f"Query deemed irrelevant. See {os.path.join(base_dir, 'rejected.txt')}")
            sys.exit(0)

        if result:
            best_box, final_score = result
            x1, y1, x2, y2 = map(int, best_box)

            # Create results subfolders
            dino_dir = ensure_dir(os.path.join(base_dir, "dino"))
            sam_dir = ensure_dir(os.path.join(base_dir, "sam"))
            final_dir = ensure_dir(os.path.join(base_dir, "final"))

            # Save cropped bbox
            cropped = img[y1:y2, x1:x2].copy()
            cv2.imwrite(os.path.join(final_dir, "final_crop.jpg"), cropped)

            # Annotate bbox
            if sv is not None:
                det = sv.Detections(
                    xyxy=np.array(best_box).reshape(1, 4),
                    confidence=np.array([final_score]),
                    class_id=np.array([0])
                )
                annotator = sv.BoxAnnotator(thickness=3)
                annotated = annotator.annotate(scene=img.copy(), detections=det, labels=[f"{QUERY} ({final_score:.2f})"])
                cv2.imwrite(os.path.join(dino_dir, "annotated_bbox.jpg"), annotated)
            else:
                annotated = img.copy()
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.imwrite(os.path.join(dino_dir, "annotated_bbox.jpg"), annotated)

            # SAM segmentation
            sam_predictor.set_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            input_box = np.array([x1, y1, x2, y2])
            masks, _, _ = sam_predictor.predict(box=input_box[None, :], multimask_output=False)
            mask_bool = masks[0].astype(bool)

            # Save SAM outputs
            cv2.imwrite(os.path.join(sam_dir, "sam_mask.png"), (mask_bool.astype(np.uint8) * 255))
            overlay = img.copy()
            overlay[mask_bool] = (0, 255, 0)
            cv2.imwrite(os.path.join(sam_dir, "sam_overlay.jpg"), overlay)
            crop_rgba_with_mask(img, mask_bool).save(os.path.join(sam_dir, "object_rgba.png"))

            # Save overlay with bbox
            final_overlay = annotated.copy()
            final_overlay[mask_bool] = (0, 255, 0)
            cv2.imwrite(os.path.join(final_dir, "final_overlay.jpg"), final_overlay)

            # Save small preview
            preview = cv2.resize(img, (min(640, img.shape[1]), int(img.shape[0] * min(640, img.shape[1]) / img.shape[1])))
            cv2.imwrite(os.path.join(base_dir, "preview.jpg"), preview)

            print(f"Results saved in: {base_dir}")
        else:
            # No detection after passing relevance → log gracefully
            with open(os.path.join(base_dir, "no_detection.txt"), "w", encoding="utf-8") as f:
                f.write("No suitable detection found after relevance check.\n")
            preview = cv2.resize(img, (min(640, img.shape[1]), int(img.shape[0] * min(640, img.shape[1]) / img.shape[1])))
            cv2.imwrite(os.path.join(base_dir, "preview.jpg"), preview)
            print("No suitable detection found.")

    except Exception:
        traceback.print_exc()
        sys.exit(1)
