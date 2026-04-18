"""
Pseudo-video bidirectional bbox propagation on test set.

Run:
    python evaluate_gray_fast.py \
        --slices-dir ./data/brats_slices_2d \
        --checkpoint ./model/sam2-main/checkpoints/sam2.1_hiera_large.pt \
        --lora-weights ./outputs/sam2_brats_gray_1024/final.pt \
        --test-split ./outputs/sam2_brats_gray_1024/test_split.json \
        --output-dir ./outputs/eval_gray_1024 \
        --device cuda
"""

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from tqdm import tqdm

if hasattr(torch.amp, "autocast"):
    from torch.amp import autocast
else:
    from torch.cuda.amp import autocast

try:
    from peft import LoraConfig, TaskType, get_peft_model
except Exception as exc:
    raise ImportError("peft is required. pip install peft") from exc

PROJECT_ROOT = Path(__file__).resolve().parent
SAM2_REPO = PROJECT_ROOT / "model" / "sam2-main"
if str(SAM2_REPO) not in sys.path:
    sys.path.insert(0, str(SAM2_REPO))

from sam2.build_sam import build_sam2


#  Forward

def sam2_forward_with_boxes(model, images, boxes_xyxy):
    B = images.shape[0]
    backbone_out = model.forward_image(images)
    _, vision_feats, _, feat_sizes = model._prepare_backbone_features(backbone_out)

    if model.directly_add_no_mem_embed:
        vision_feats[-1] = vision_feats[-1] + model.no_mem_embed

    feats = [
        feat.permute(1, 2, 0).view(B, -1, *feat_size)
        for feat, feat_size in zip(vision_feats[::-1], feat_sizes[::-1])
    ][::-1]

    image_embed = feats[-1]
    high_res_feats = feats[:-1]

    box_coords = boxes_xyxy.reshape(-1, 2, 2)
    box_labels = torch.tensor(
        [[2, 3]], dtype=torch.int, device=boxes_xyxy.device
    ).repeat(B, 1)

    sparse_embeddings, dense_embeddings = model.sam_prompt_encoder(
        points=(box_coords, box_labels), boxes=None, masks=None,
    )

    h, w = image_embed.shape[-2:]
    dense_embeddings = F.interpolate(
        dense_embeddings, size=(h, w), mode="bilinear", align_corners=False
    )
    image_pe = model.sam_prompt_encoder.pe_layer((h, w)).unsqueeze(0)

    low_res_masks, _, _, _ = model.sam_mask_decoder(
        image_embeddings=image_embed,
        image_pe=image_pe,
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
        repeat_image=False,
        high_res_features=high_res_feats if len(high_res_feats) > 0 else None,
    )

    return F.interpolate(
        low_res_masks, size=(images.shape[-2], images.shape[-1]),
        mode="bilinear", align_corners=False,
    )


#  Utilities

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
_TARGET = 1024


def to_flair_1024(image_chw):
    """FLAIR channel 0 → grayscale 3ch → ImageNet norm → resize 1024."""
    img_01 = np.clip(image_chw * _STD + _MEAN, 0, 1)
    flair = img_01[0:1, :, :]
    flair_3ch = np.repeat(flair, 3, axis=0)
    normed = ((flair_3ch - _MEAN) / _STD).astype(np.float32)
    t = torch.from_numpy(normed).unsqueeze(0)
    t = F.interpolate(t, size=(_TARGET, _TARGET), mode="bilinear", align_corners=False)
    return t.squeeze(0).numpy()


def predict_one_slice(model, image_chw_1024, bbox_1024, device):
    """Predict on 1024x1024 input, return 1024x1024 mask."""
    img_t = torch.from_numpy(image_chw_1024).unsqueeze(0).to(device)
    box_t = torch.from_numpy(bbox_1024).unsqueeze(0).to(device)
    with torch.inference_mode():
        with autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
            logits = sam2_forward_with_boxes(model, img_t, box_t)
    return (torch.sigmoid(logits[0, 0]) > 0.5).cpu().numpy().astype(np.uint8)


def mask_to_bbox(mask_hw):
    ys, xs = np.where(mask_hw > 0)
    if len(xs) == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def dice_score(pred, gt):
    p, g = pred.astype(bool), gt.astype(bool)
    inter = int(np.logical_and(p, g).sum())
    denom = int(p.sum()) + int(g.sum())
    return 1.0 if denom == 0 else float(2.0 * inter / denom)


def iou_score(pred, gt):
    p, g = pred.astype(bool), gt.astype(bool)
    inter = int(np.logical_and(p, g).sum())
    union = int(np.logical_or(p, g).sum())
    return 1.0 if union == 0 else float(inter / union)


def denorm_to_uint8(img_chw):
    x = np.clip(img_chw * _STD + _MEAN, 0, 1)
    return (x * 255).astype(np.uint8).transpose(1, 2, 0)


def build_lora_config(r=16):
    return LoraConfig(
        r=r, lora_alpha=2 * r, lora_dropout=0.05, bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
    )

#  Data loading

def build_case_index(slices_dir, fg_only=True):
    with open(slices_dir / "slice_index.json", "r", encoding="utf-8") as f:
        all_records = json.load(f)
    if fg_only:
        all_records = [r for r in all_records if r["has_fg"]]
    case_map = defaultdict(list)
    for r in all_records:
        case_map[r.get("case", "unknown")].append(r)
    for cid in case_map:
        case_map[cid].sort(key=lambda r: r["z"])
    return case_map


def load_case_slices_1024(slices_dir, case_id, case_map):
    """Load slices → FLAIR grayscale → resize 1024 → also resize mask to 1024."""
    records = case_map.get(case_id, [])
    slices = []
    for r in records:
        data = np.load(str(slices_dir / r["npz_path"]))
        orig_size = data["image"].shape[-1]  # 256
        scale = _TARGET / orig_size

        # Image: FLAIR → 1024
        image_1024 = to_flair_1024(data["image"])

        # Mask: resize to 1024
        msk_t = torch.from_numpy(data["mask"]).unsqueeze(0)
        msk_t = F.interpolate(msk_t, size=(_TARGET, _TARGET), mode="nearest")
        mask_1024 = msk_t.squeeze(0).numpy()

        # Bbox: scale to 1024
        bbox_1024 = (data["bbox"] * scale).astype(np.float32)

        slices.append({
            "z": r["z"],
            "image": image_1024,
            "mask": mask_1024,
            "bbox": bbox_1024,
        })
    return slices

#  Pseudo-video propagation

@torch.no_grad()
def eval_pseudo_video(model, slices_list, device):
    n = len(slices_list)
    if n == 0:
        return []

    preds = [None] * n
    areas = [s["mask"][0].sum() for s in slices_list]
    init_idx = int(np.argmax(areas))
    gt_mask = (slices_list[init_idx]["mask"][0] > 0.5).astype(np.uint8)
    preds[init_idx] = gt_mask

    for i in range(init_idx + 1, n):
        bbox = mask_to_bbox(preds[i - 1])
        if bbox is None:
            preds[i] = np.zeros_like(gt_mask)
            continue
        preds[i] = predict_one_slice(model, slices_list[i]["image"], bbox, device)

    for i in range(init_idx - 1, -1, -1):
        bbox = mask_to_bbox(preds[i + 1])
        if bbox is None:
            preds[i] = np.zeros_like(gt_mask)
            continue
        preds[i] = predict_one_slice(model, slices_list[i]["image"], bbox, device)

    return preds


def compute_case_metrics(preds, slices_list):
    all_pred = np.concatenate([p.flatten() for p in preds])
    all_gt = np.concatenate(
        [(s["mask"][0] > 0.5).astype(np.uint8).flatten() for s in slices_list]
    )
    return dice_score(all_pred, all_gt), iou_score(all_pred, all_gt)
    
#  Visualization (downsample to 256 for saving)

def make_overlay(img_rgb, mask, color=(255, 0, 0), alpha=0.45):
    overlay = img_rgb.copy().astype(np.float32)
    m = mask > 0
    overlay[m] = (1 - alpha) * overlay[m] + alpha * np.array(color, dtype=np.float32)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def create_comparison_image(img_chw_1024, gt_mask_1024, pred_bl_1024, pred_lr_1024):
    # Downsample to 256 for visualization
    img_t = torch.from_numpy(img_chw_1024).unsqueeze(0)
    img_256 = F.interpolate(img_t, size=(256, 256), mode="bilinear", align_corners=False)
    img_rgb = denorm_to_uint8(img_256.squeeze(0).numpy())

    def down(m):
        mt = torch.from_numpy(m.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        mt = F.interpolate(mt, size=(256, 256), mode="nearest")
        return (mt.squeeze().numpy() > 0.5).astype(np.uint8)

    gt_s = down(gt_mask_1024)
    bl_s = down(pred_bl_1024)
    lr_s = down(pred_lr_1024)

    h, w = img_rgb.shape[:2]
    panels = [
        img_rgb,
        make_overlay(img_rgb, gt_s, (0, 255, 0)),
        make_overlay(img_rgb, bl_s, (255, 0, 0)),
        make_overlay(img_rgb, lr_s, (0, 100, 255)),
    ]
    titles = ["Image", "GT", "Baseline", "LoRA"]
    gap, title_h = 4, 22
    total_w = len(panels) * w + (len(panels) - 1) * gap
    canvas = Image.new("RGB", (total_w, h + title_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for i, (panel, title) in enumerate(zip(panels, titles)):
        x = i * (w + gap)
        canvas.paste(Image.fromarray(panel), (x, title_h))
        draw.text((x + 4, 3), title, fill=(0, 0, 0))
    return canvas


#  Main

def main():
    parser = argparse.ArgumentParser(
        description="Fair eval: FLAIR gray 1024, pseudo-video"
    )
    parser.add_argument("--slices-dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--lora-weights", type=str, required=True)
    parser.add_argument("--test-split", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./outputs/eval_gray_1024")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model-cfg", type=str,
                        default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--num-vis-cases", type=int, default=5)
    parser.add_argument("--num-vis-slices", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "vis_samples"
    vis_dir.mkdir(parents=True, exist_ok=True)
    slices_dir = Path(args.slices_dir)

    print(f"Loading test split: {args.test_split}")
    with open(args.test_split, "r", encoding="utf-8") as f:
        split_info = json.load(f)
    test_case_list = sorted(split_info["test_cases"])
    print(f"Test cases: {len(test_case_list)}")

    print("Building case index...")
    case_map = build_case_index(slices_dir, fg_only=split_info.get("fg_only", True))

    rng = random.Random(args.seed + 1)
    vis_cases = set(rng.sample(
        test_case_list, min(args.num_vis_cases, len(test_case_list))
    ))

    print("\nLoading Baseline (SAM2 Large)...")
    model_base = build_sam2(
        config_file=args.model_cfg, ckpt_path=str(args.checkpoint),
        device=args.device, mode="eval",
    )
    model_base.eval()

    print("Loading LoRA (FLAIR 1024)...")
    model_lora = build_sam2(
        config_file=args.model_cfg, ckpt_path=str(args.checkpoint),
        device=args.device, mode="eval",
    )
    model_lora = model_lora.to(args.device)
    model_lora = get_peft_model(model_lora, build_lora_config(r=16))
    state = torch.load(str(args.lora_weights), map_location=args.device)
    model_lora.load_state_dict(state.get("model", state), strict=False)
    model_lora.eval()
    print("Both models loaded.\n")

    all_results = {"baseline": [], "lora": []}

    for case_id in tqdm(test_case_list, desc="Evaluating"):
        slices_list = load_case_slices_1024(slices_dir, case_id, case_map)
        if len(slices_list) == 0:
            continue

        preds_bl = eval_pseudo_video(model_base, slices_list, args.device)
        preds_lr = eval_pseudo_video(model_lora, slices_list, args.device)

        for name, preds in [("baseline", preds_bl), ("lora", preds_lr)]:
            d, i = compute_case_metrics(preds, slices_list)
            all_results[name].append({
                "case": case_id, "dice": round(d, 6), "iou": round(i, 6),
            })

        if case_id in vis_cases:
            vis_indices = rng.sample(
                range(len(slices_list)), min(args.num_vis_slices, len(slices_list))
            )
            for idx in vis_indices:
                s = slices_list[idx]
                comp = create_comparison_image(
                    s["image"], (s["mask"][0] > 0.5).astype(np.uint8),
                    preds_bl[idx], preds_lr[idx],
                )
                comp.save(vis_dir / f"{case_id}_z{s['z']:03d}_compare.png")

    del model_base, model_lora
    torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("  COMPARISON  (FLAIR grayscale, 1024x1024, pseudo-video)")
    print("=" * 70)
    print(f"{'Method':<35} {'Mean Dice':>12} {'Std Dice':>12} {'Mean IoU':>12}")
    print("-" * 70)

    summary = {}
    for m in ["baseline", "lora"]:
        dices = [r["dice"] for r in all_results[m]]
        ious = [r["iou"] for r in all_results[m]]
        md, sd = float(np.mean(dices)), float(np.std(dices))
        mi, si = float(np.mean(ious)), float(np.std(ious))
        label = "Baseline (SAM2 Large)" if m == "baseline" else "LoRA fine-tuned (FLAIR 1024)"
        print(f"{label:<35} {md:>12.6f} {sd:>12.6f} {mi:>12.6f}")
        summary[m] = {"label": label, "mean_dice": round(md, 6),
                       "std_dice": round(sd, 6), "mean_iou": round(mi, 6),
                       "std_iou": round(si, 6), "num_cases": len(dices)}

    print("-" * 70)
    dd = summary["lora"]["mean_dice"] - summary["baseline"]["mean_dice"]
    di = summary["lora"]["mean_iou"] - summary["baseline"]["mean_iou"]
    print(f"Improvement: Dice {'+' if dd>=0 else ''}{dd:.6f} | "
          f"IoU {'+' if di>=0 else ''}{di:.6f}")
    print("=" * 70)

    with open(output_dir / "comparison.json", "w", encoding="utf-8") as f:
        json.dump({
            "test_cases": test_case_list,
            "num_test_cases": len(test_case_list),
            "inference": "FLAIR grayscale, 1024x1024, sam2_forward_with_boxes, pseudo-video",
            "summary": summary,
            "improvement": {"dice": round(dd, 6), "iou": round(di, 6)},
        }, f, ensure_ascii=False, indent=2)

    bl_map = {r["case"]: r for r in all_results["baseline"]}
    lr_map = {r["case"]: r for r in all_results["lora"]}
    rows = [{
        "case": cid,
        "baseline_dice": bl_map[cid]["dice"], "baseline_iou": bl_map[cid]["iou"],
        "lora_dice": lr_map[cid]["dice"], "lora_iou": lr_map[cid]["iou"],
        "dice_delta": round(lr_map[cid]["dice"] - bl_map[cid]["dice"], 6),
    } for cid in test_case_list if cid in bl_map]

    with open(output_dir / "per_case_comparison.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved: comparison.json, per_case_comparison.csv, "
          f"vis_samples/ ({len(list(vis_dir.glob('*.png')))} imgs)")


if __name__ == "__main__":
    main()
