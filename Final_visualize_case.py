import argparse
import json
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
    raise ImportError("peft is required") from exc

PROJECT_ROOT = Path(__file__).resolve().parent
SAM2_REPO = PROJECT_ROOT / "model" / "sam2-main"
if str(SAM2_REPO) not in sys.path:
    sys.path.insert(0, str(SAM2_REPO))

from sam2.build_sam import build_sam2


#  Forward (same as training)

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
    img_01 = np.clip(image_chw * _STD + _MEAN, 0, 1)
    flair = img_01[0:1, :, :]
    flair_3ch = np.repeat(flair, 3, axis=0)
    normed = ((flair_3ch - _MEAN) / _STD).astype(np.float32)
    t = torch.from_numpy(normed).unsqueeze(0)
    t = F.interpolate(t, size=(_TARGET, _TARGET), mode="bilinear", align_corners=False)
    return t.squeeze(0).numpy()


def predict_one_slice(model, image_chw_1024, bbox_1024, device):
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


def build_lora_config(r=16):
    return LoraConfig(
        r=r, lora_alpha=2 * r, lora_dropout=0.05, bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
    )


#  Data loading


def load_case_slices(slices_dir, case_id):
    with open(slices_dir / "slice_index.json", "r", encoding="utf-8") as f:
        all_records = json.load(f)
    records = [r for r in all_records if r.get("case") == case_id]
    records.sort(key=lambda r: r["z"])

    slices = []
    for r in records:
        data = np.load(str(slices_dir / r["npz_path"]))
        slices.append({
            "z": r["z"],
            "has_fg": r["has_fg"],
            "image": data["image"],
            "mask": data["mask"],
            "bbox": data["bbox"],
        })
    return slices



#  Pseudo-video propagation

@torch.no_grad()
def run_pseudo_video(model, slices_1024, masks_1024, device):
    """Run propagation on 1024x1024 images, return list of pred masks."""
    n = len(slices_1024)
    preds = [None] * n

    # Find largest GT foreground slice
    areas = [m.sum() for m in masks_1024]
    init_idx = int(np.argmax(areas))
    preds[init_idx] = (masks_1024[init_idx] > 0.5).astype(np.uint8)

    # Forward
    for i in range(init_idx + 1, n):
        bbox = mask_to_bbox(preds[i - 1])
        if bbox is None:
            preds[i] = np.zeros_like(preds[init_idx])
            continue
        preds[i] = predict_one_slice(model, slices_1024[i], bbox, device)

    # Backward
    for i in range(init_idx - 1, -1, -1):
        bbox = mask_to_bbox(preds[i + 1])
        if bbox is None:
            preds[i] = np.zeros_like(preds[init_idx])
            continue
        preds[i] = predict_one_slice(model, slices_1024[i], bbox, device)

    return preds



#  Visualization

def make_overlay(img_rgb, mask, color=(255, 0, 0), alpha=0.45):
    overlay = img_rgb.copy().astype(np.float32)
    m = mask > 0
    overlay[m] = (1 - alpha) * overlay[m] + alpha * np.array(color, dtype=np.float32)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def create_comparison_image(img_1024, gt_mask, pred_bl, pred_lr,
                            dice_bl, dice_lr, z_idx):
    """Image | GT | Baseline (dice) | LoRA (dice)"""
    # Convert to display RGB
    img_01 = np.clip(img_1024 * _STD + _MEAN, 0, 1)
    img_rgb = (img_01 * 255).astype(np.uint8).transpose(1, 2, 0)

    # Resize to 256 for smaller output files
    size = 256
    img_rgb = np.array(Image.fromarray(img_rgb).resize((size, size), Image.BILINEAR))
    gt_small = np.array(Image.fromarray(gt_mask * 255).resize((size, size), Image.NEAREST)) > 127
    bl_small = np.array(Image.fromarray(pred_bl * 255).resize((size, size), Image.NEAREST)) > 127
    lr_small = np.array(Image.fromarray(pred_lr * 255).resize((size, size), Image.NEAREST)) > 127

    panels = [
        img_rgb,
        make_overlay(img_rgb, gt_small, (0, 255, 0)),
        make_overlay(img_rgb, bl_small, (255, 0, 0)),
        make_overlay(img_rgb, lr_small, (0, 100, 255)),
    ]
    titles = [
        f"z={z_idx}",
        "GT",
        f"Base D={dice_bl:.3f}",
        f"LoRA D={dice_lr:.3f}",
    ]

    h, w = size, size
    gap = 4
    title_h = 22
    total_w = len(panels) * w + (len(panels) - 1) * gap
    total_h = h + title_h

    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for i, (panel, title) in enumerate(zip(panels, titles)):
        x_off = i * (w + gap)
        canvas.paste(Image.fromarray(panel), (x_off, title_h))
        draw.text((x_off + 4, 3), title, fill=(0, 0, 0))
    return canvas



#  Main


def main():
    parser = argparse.ArgumentParser(
        description="Visualize all slices of a single case"
    )
    parser.add_argument("--case-id", type=str, required=True,
                        help="e.g. BRATS_005")
    parser.add_argument("--slices-dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--lora-weights", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./outputs/vis_case")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model-cfg", type=str,
                        default="configs/sam2.1/sam2.1_hiera_l.yaml")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) / args.case_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load slices ----
    print(f"Loading slices for {args.case_id}...")
    slices_list = load_case_slices(Path(args.slices_dir), args.case_id)
    if len(slices_list) == 0:
        print(f"No slices found for {args.case_id}")
        return
    print(f"Found {len(slices_list)} slices")

    # ---- Prepare 1024x1024 data ----
    print("Converting to FLAIR 1024x1024...")
    slices_1024 = []
    masks_1024 = []
    for s in slices_list:
        img_1024 = to_flair_1024(s["image"])
        slices_1024.append(img_1024)

        # Resize mask to 1024
        msk_t = torch.from_numpy(s["mask"]).unsqueeze(0)
        msk_t = F.interpolate(msk_t, size=(_TARGET, _TARGET), mode="nearest")
        masks_1024.append((msk_t.squeeze().numpy() > 0.5).astype(np.uint8))

    # ---- Load models ----
    print("Loading Baseline...")
    model_base = build_sam2(
        config_file=args.model_cfg, ckpt_path=str(args.checkpoint),
        device=args.device, mode="eval",
    )
    model_base.eval()

    print("Loading LoRA...")
    model_lora = build_sam2(
        config_file=args.model_cfg, ckpt_path=str(args.checkpoint),
        device=args.device, mode="eval",
    )
    model_lora = model_lora.to(args.device)
    model_lora = get_peft_model(model_lora, build_lora_config(r=16))
    state = torch.load(str(args.lora_weights), map_location=args.device)
    model_lora.load_state_dict(state.get("model", state), strict=False)
    model_lora.eval()

    # ---- Run propagation ----
    print("Running Baseline propagation...")
    preds_bl = run_pseudo_video(model_base, slices_1024, masks_1024, args.device)

    print("Running LoRA propagation...")
    preds_lr = run_pseudo_video(model_lora, slices_1024, masks_1024, args.device)

    # ---- Save all slices ----
    print(f"Saving {len(slices_list)} comparison images...")
    for i, s in enumerate(tqdm(slices_list, desc="Saving")):
        gt = masks_1024[i]
        dice_bl = dice_score(preds_bl[i], gt)
        dice_lr = dice_score(preds_lr[i], gt)

        comp = create_comparison_image(
            slices_1024[i], gt, preds_bl[i], preds_lr[i],
            dice_bl, dice_lr, s["z"],
        )
        comp.save(output_dir / f"z{s['z']:03d}.png")

    # ---- Print per-slice dice ----
    print(f"\n{'z':>5} {'Baseline':>10} {'LoRA':>10} {'Delta':>10}")
    print("-" * 40)
    total_bl, total_lr = 0, 0
    for i, s in enumerate(slices_list):
        gt = masks_1024[i]
        d_bl = dice_score(preds_bl[i], gt)
        d_lr = dice_score(preds_lr[i], gt)
        delta = d_lr - d_bl
        total_bl += d_bl
        total_lr += d_lr
        print(f"{s['z']:>5} {d_bl:>10.4f} {d_lr:>10.4f} {'+' if delta>=0 else ''}{delta:>9.4f}")

    n = len(slices_list)
    print("-" * 40)
    print(f"{'Avg':>5} {total_bl/n:>10.4f} {total_lr/n:>10.4f} "
          f"{'+' if (total_lr-total_bl)>=0 else ''}{(total_lr-total_bl)/n:>9.4f}")

    print(f"\nSaved to: {output_dir}")

    del model_base, model_lora
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
