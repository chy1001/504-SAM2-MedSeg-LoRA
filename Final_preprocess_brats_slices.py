import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F


def normalize_percentile(x: np.ndarray, pmin: float = 1.0, pmax: float = 99.0) -> np.ndarray:
    """Normlization"""
    x = x.astype(np.float32)
    lo = np.percentile(x, pmin)
    hi = np.percentile(x, pmax)
    if hi <= lo:
        lo = float(x.min())
        hi = float(x.max())
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def mask_to_bbox(mask_hw: np.ndarray, image_size: int) -> np.ndarray:
    """extract bbox from gt; if none: return the entire bbox"""
    ys, xs = np.where(mask_hw > 0)
    if len(xs) == 0:
        return np.array([0, 0, image_size - 1, image_size - 1], dtype=np.float32)
    return np.array(
        [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())],
        dtype=np.float32,
    )


def resize_image_and_mask(
    img_hwc: np.ndarray,
    mask_hw: np.ndarray,
    out_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """resize"""
    img_t = torch.from_numpy(img_hwc).permute(2, 0, 1).unsqueeze(0)
    msk_t = torch.from_numpy(mask_hw.astype(np.float32)).unsqueeze(0).unsqueeze(0)

    img_t = F.interpolate(img_t, size=(out_size, out_size), mode="bilinear", align_corners=False)
    msk_t = F.interpolate(msk_t, size=(out_size, out_size), mode="nearest")

    img_out = img_t.squeeze(0).permute(1, 2, 0).numpy().astype(np.float32)
    msk_out = (msk_t.squeeze(0).squeeze(0).numpy() > 0.5).astype(np.uint8)
    return img_out, msk_out


def to_sam_normalized(img_hwc_01: np.ndarray) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (img_hwc_01 - mean) / std
    return np.transpose(x, (2, 0, 1)).astype(np.float32)  # CHW
    

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="slice BraTS NIfTI 3D into 2D .npz",
    )
    parser.add_argument(
        "--dataset-root", type=Path, required=True,
        help="Task01_BrainTumour root dir (imagesTr/, labelsTr/, dataset.json)",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="output dir，save slices as .npz",
    )
    parser.add_argument(
        "--image-size", type=int, default=256,
        help="resize to 256x256",
    )
    parser.add_argument(
        "--include-empty", action="store_true",
        help="if include no target slices",
    )
    parser.add_argument(
        "--max-cases", type=int, default=0,
        help="cases batch",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="randomize when max-cases > 0",
    )
    parser.add_argument(
        "--num-workers", type=int, default=1,
        help="num workers",
    )
    return parser.parse_args()


def get_case_pairs(dataset_root: Path, max_cases: int, seed: int):
    import random

    dataset_json = dataset_root / "dataset.json"
    if not dataset_json.exists():
        raise FileNotFoundError(f"dataset.json not found in {dataset_root}")

    with open(dataset_json, "r", encoding="utf-8") as f:
        meta = json.load(f)

    train_list = meta.get("training", [])
    pairs: List[Tuple[Path, Path]] = []
    for item in train_list:
        rel_img = item["image"].replace("./", "")
        rel_lbl = item["label"].replace("./", "")
        img_path = dataset_root / rel_img
        lbl_path = dataset_root / rel_lbl
        if img_path.exists() and lbl_path.exists() and not img_path.name.startswith("._"):
            pairs.append((img_path, lbl_path))

    if max_cases > 0 and max_cases < len(pairs):
        random.Random(seed).shuffle(pairs)
        pairs = pairs[:max_cases]

    return sorted(pairs, key=lambda p: p[0].name)


def process_one_case(
    img_path: Path,
    lbl_path: Path,
    output_dir: Path,
    image_size: int,
    include_empty: bool,
) -> List[dict]:
    case_name = img_path.name.replace(".nii.gz", "")
    case_dir = output_dir / case_name
    case_dir.mkdir(parents=True, exist_ok=True)

    img = np.asarray(nib.load(str(img_path)).get_fdata(dtype=np.float32))  # H,W,D,4
    lbl = np.asarray(nib.load(str(lbl_path)).get_fdata(dtype=np.float32))  # H,W,D

    if img.ndim != 4 or img.shape[-1] < 4:
        print(f"[WARN] Skipping {img_path.name}: unexpected shape {img.shape}")
        return []
    if lbl.ndim != 3:
        print(f"[WARN] Skipping {lbl_path.name}: unexpected label shape {lbl.shape}")
        return []

    d = img.shape[2]  # depth (number of slices along Z)
    slice_records = []

    for z in range(d):
        mask_hw = (lbl[:, :, z] > 0).astype(np.uint8)
        has_fg = bool(np.any(mask_hw > 0))

        if not has_fg and not include_empty:
            continue

        # FLAIR(0), T1Gd(2), T2(3) -> RGB
        slc = img[:, :, z, :]
        rgb = np.stack(
            [
                normalize_percentile(slc[:, :, 0]),
                normalize_percentile(slc[:, :, 2]),
                normalize_percentile(slc[:, :, 3]),
            ],
            axis=-1,
        ).astype(np.float32)

        # resize
        rgb, mask_hw = resize_image_and_mask(rgb, mask_hw, image_size)

        # SAM ImageNet normalization
        image_chw = to_sam_normalized(rgb)  # float32, [3, H, W]

        # mask -> [1, H, W]
        mask_1hw = mask_hw.astype(np.float32)[np.newaxis, :, :]

        # bbox
        bbox = mask_to_bbox(mask_hw, image_size)  # [4]

        # compressed as .npz
        npz_name = f"z_{z:03d}.npz"
        npz_path = case_dir / npz_name
        np.savez_compressed(
            str(npz_path),
            image=image_chw,   # float32 [3, H, W]
            mask=mask_1hw,     # float32 [1, H, W]
            bbox=bbox,         # float32 [4]
        )

        slice_records.append({
            "case": case_name,
            "z": z,
            "has_fg": has_fg,
            "npz_path": str(npz_path.relative_to(output_dir)),
        })

    return slice_records


def main():
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset root : {args.dataset_root}")
    print(f"Output dir   : {output_dir}")
    print(f"Image size   : {args.image_size}")
    print(f"Include empty: {args.include_empty}")

    pairs = get_case_pairs(args.dataset_root, args.max_cases, args.seed)
    print(f"Total cases  : {len(pairs)}")

    all_records: List[dict] = []
    try:
        from tqdm import tqdm
        iterator = tqdm(pairs, desc="Preprocessing cases")
    except ImportError:
        iterator = pairs

    for img_path, lbl_path in iterator:
        records = process_one_case(
            img_path=img_path,
            lbl_path=lbl_path,
            output_dir=output_dir,
            image_size=args.image_size,
            include_empty=args.include_empty,
        )
        all_records.extend(records)

    # save indices
    index_path = output_dir / "slice_index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)

    # save metadata
    fg_count = sum(1 for r in all_records if r["has_fg"])
    meta = {
        "total_slices": len(all_records),
        "fg_slices": fg_count,
        "empty_slices": len(all_records) - fg_count,
        "num_cases": len(pairs),
        "image_size": args.image_size,
        "include_empty": args.include_empty,
        "max_cases": args.max_cases,
        "seed": args.seed,
        "normalization": "ImageNet (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])",
        "channels": "FLAIR(0) + T1Gd(2) + T2(3) -> RGB",
    }
    meta_path = output_dir / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\nDone!")
    print(f"  Total slices: {len(all_records)}")
    print(f"  FG slices: {fg_count}")
    print(f"  Empty slices: {len(all_records) - fg_count}")
    print(f"  Index saved: {index_path}")
    print(f"  Meta saved: {meta_path}")
    print(f"start training")


if __name__ == "__main__":
    main()
