import argparse
import json
import math
import os
import random
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# PyTorch
if hasattr(torch.amp, "autocast"):
    from torch.amp import GradScaler, autocast
else:
    from torch.cuda.amp import GradScaler, autocast

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# Dependency checks
try:
    import bitsandbytes as bnb
except Exception as exc:
    raise ImportError(
        "bitsandbytes is required for 4-bit QLoRA. Install with: pip install bitsandbytes"
    ) from exc

try:
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
except Exception as exc:
    raise ImportError(
        "peft is required for LoRA. Install with: pip install peft"
    ) from exc


# SAM2 import
PROJECT_ROOT = Path(__file__).resolve().parent
SAM2_REPO = PROJECT_ROOT / "model" / "sam2-main"
if str(SAM2_REPO) not in sys.path:
    sys.path.insert(0, str(SAM2_REPO))
from sam2.build_sam import build_sam2  # noqa: E402

# Utils
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Dataset
class PreslicedBratsDataset(Dataset):
    def __init__(
        self,
        slices_dir: Path,
        fg_only: bool = True,
        allowed_cases: Optional[set] = None,
    ):
        super().__init__()
        self.slices_dir = Path(slices_dir)
        index_path = self.slices_dir / "slice_index.json"
        if not index_path.exists():
            raise FileNotFoundError(
                f"slice_index.json not found in {self.slices_dir}. "
                f"Please run preprocess_brats_slices.py first."
            )
        with open(index_path, "r", encoding="utf-8") as f:
            all_records = json.load(f)

        if fg_only:
            records = [r for r in all_records if r["has_fg"]]
        else:
            records = all_records

        for r in records:
            r["_case_id"] = self._infer_case_id(r)

        if allowed_cases is not None:
            records = [r for r in records if r["_case_id"] in allowed_cases]

        self.records = records

        if len(self.records) == 0:
            raise RuntimeError("No valid slices found in index.")

        # validation
        meta_path = self.slices_dir / "meta.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            print(f"[Dataset] Loaded {len(self.records)} slices "
                  f"(total={meta['total_slices']}, fg={meta['fg_slices']}, "
                  f"image_size={meta['image_size']})")

    @staticmethod
    def _infer_case_id(record: Dict) -> str:
        for k in ("case_id", "case", "patient_id", "subject_id"):
            if k in record:
                return str(record[k])
        npz_rel = str(record.get("npz_path", ""))
        p = Path(npz_rel)
        if len(p.parts) >= 2:
            return p.parts[0]
        stem = p.stem
        for token in ("_slice_", "_z", "_idx_"):
            if token in stem:
                return stem.split(token)[0]
        return stem

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        record = self.records[idx]
        npz_path = self.slices_dir / record["npz_path"]

        data = np.load(str(npz_path))
        image = data["image"]  # [3, H, W] float32, ImageNet normalized
        mask = data["mask"]    # [1, H, W] float32
        bbox = data["bbox"]    # [4] float32

        # --- Convert to FLAIR-only grayscale ---
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        image_01 = np.clip(image * std + mean, 0, 1)
        flair = image_01[0:1, :, :]
        flair_3ch = np.repeat(flair, 3, axis=0)
        image_flair = ((flair_3ch - mean) / std).astype(np.float32)

        # --- Resize to 1024x1024 ---
        orig_size = image_flair.shape[-1]
        target_size = 1024

        img_t = torch.from_numpy(image_flair).unsqueeze(0)
        img_t = F.interpolate(img_t, size=(target_size, target_size),
                              mode="bilinear", align_corners=False)
        image_out = img_t.squeeze(0)  # [3,1024,1024]

        msk_t = torch.from_numpy(mask).unsqueeze(0)
        msk_t = F.interpolate(msk_t, size=(target_size, target_size),
                              mode="nearest")
        mask_out = (msk_t.squeeze(0) > 0.5).float()

        # Scale bbox from 256 to 1024
        scale = target_size / orig_size
        bbox_out = torch.from_numpy(bbox * scale)

        return {
            "image": image_out,
            "mask": mask_out,
            "bbox": bbox_out,
        }



# QLoRA helpers

def replace_linear_with_4bit(
    module: nn.Module,
    module_prefix: str = "",
    target_scopes: Sequence[str] = ("image_encoder", "sam_mask_decoder"),
    compute_dtype: torch.dtype = torch.float16,
    device: Optional[torch.device] = None,
) -> int:
    replaced = 0
    for name, child in list(module.named_children()):
        full_name = f"{module_prefix}.{name}" if module_prefix else name

        if isinstance(child, nn.Linear) and any(scope in full_name for scope in target_scopes):
            new_layer = bnb.nn.Linear4bit(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                compute_dtype=compute_dtype,
                quant_type="nf4",
                compress_statistics=True,
            )
            new_layer.weight = nn.Parameter(child.weight.detach().clone())
            if child.bias is not None:
                new_layer.bias = nn.Parameter(child.bias.detach().clone())
            if device is not None:
                new_layer.to(device)
            setattr(module, name, new_layer)
            replaced += 1
        else:
            replaced += replace_linear_with_4bit(
                child,
                module_prefix=full_name,
                target_scopes=target_scopes,
                compute_dtype=compute_dtype,
                device=device,
            )
    return replaced


def build_lora_config(r: int = 16) -> LoraConfig:
    return LoraConfig(
        r=r,
        lora_alpha=2 * r,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2",
        ],
    )


def print_trainable_params(model: nn.Module) -> None:
    trainable = 0
    total = 0
    for p in model.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    pct = 100.0 * trainable / total if total > 0 else 0.0
    print(f"Trainable params: {trainable:,} / {total:,} ({pct:.4f}%)")


# SAM2 forward
def sam2_forward_with_boxes(
    model: nn.Module, images: torch.Tensor, boxes_xyxy: torch.Tensor
) -> torch.Tensor:
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
    box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=boxes_xyxy.device).repeat(B, 1)
    concat_points = (box_coords, box_labels)

    sparse_embeddings, dense_embeddings = model.sam_prompt_encoder(
        points=concat_points, boxes=None, masks=None,
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

    logits = F.interpolate(
        low_res_masks,
        size=(images.shape[-2], images.shape[-1]),
        mode="bilinear",
        align_corners=False,
    )
    return logits


# Loss
class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        num = 2.0 * (probs * targets).sum(dim=(1, 2, 3))
        den = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + self.eps
        dice = num / den
        return 1.0 - dice.mean()

# Training

@dataclass
class TrainConfig:
    slices_dir: Path
    output_dir: Path
    model_cfg: str
    checkpoint: Path
    epochs: int = 5
    lr: float = 2e-4
    weight_decay: float = 1e-4
    batch_size: int = 4
    grad_accum: int = 4
    num_workers: int = 4
    seed: int = 42
    device: str = "cuda"
    fg_only: bool = True


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    dice_loss_fn: DiceLoss,
    cfg: TrainConfig,
    epoch: int,
) -> float:
    model.train()
    bce_loss_fn = nn.BCEWithLogitsLoss()

    running = 0.0
    steps = 0
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch}")):
        images = batch["image"].to(cfg.device, non_blocking=True)
        masks = batch["mask"].to(cfg.device, non_blocking=True)
        bboxes = batch["bbox"].to(cfg.device, non_blocking=True)

        if hasattr(torch.amp, "autocast"):
            with autocast("cuda", dtype=torch.float16, enabled=(cfg.device == "cuda")):
                logits = sam2_forward_with_boxes(model, images, bboxes)
                loss_bce = bce_loss_fn(logits, masks)
                loss_dice = dice_loss_fn(logits, masks)
                loss = 0.5 * loss_bce + 0.5 * loss_dice
                loss = loss / cfg.grad_accum
        else:
            with autocast(device_type="cuda", dtype=torch.float16, enabled=(cfg.device == "cuda")):
                logits = sam2_forward_with_boxes(model, images, bboxes)
                loss_bce = bce_loss_fn(logits, masks)
                loss_dice = dice_loss_fn(logits, masks)
                loss = 0.5 * loss_bce + 0.5 * loss_dice
                loss = loss / cfg.grad_accum

        scaler.scale(loss).backward()

        if (step + 1) % cfg.grad_accum == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        running += loss.item() * cfg.grad_accum
        steps += 1

        if (step + 1) % 200 == 0:
            print(
                f"Epoch {epoch} | Step {step+1}/{len(loader)} | "
                f"Loss {running/steps:e}"
            )
            print("mask sum:", masks.sum().item())
            print("bbox:", bboxes[0])
            print("logits stats:", logits.min().item(), logits.max().item(), logits.mean().item())
            print("loss_bce:", loss_bce.item(), "loss_dice:", loss_dice.item(), "loss:", loss.item())

    # flush last partial accumulation
    if len(loader) % cfg.grad_accum != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    return running / max(steps, 1)


def _denorm_image(img_chw: torch.Tensor) -> np.ndarray:
    """
    img_chw: normalized tensor [3,H,W]
    return uint8 RGB image [H,W,3]
    """
    mean = torch.tensor([0.485, 0.456, 0.406], device=img_chw.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=img_chw.device).view(3, 1, 1)
    x = img_chw * std + mean
    x = torch.clamp(x, 0.0, 1.0)
    x = (x * 255.0).byte().permute(1, 2, 0).detach().cpu().numpy()
    return x


@torch.no_grad()
def evaluate_on_test(
    model: nn.Module,
    test_loader: DataLoader,
    cfg: TrainConfig,
    save_dir: Path,
) -> Dict[str, float]:
    model.eval()
    vis_dir = save_dir / "test_predictions"
    vis_dir.mkdir(parents=True, exist_ok=True)

    tp = 0
    fp = 0
    fn = 0
    eps = 1e-8
    sample_idx = 0

    for batch in tqdm(test_loader, desc="Evaluating on test"):
        images = batch["image"].to(cfg.device, non_blocking=True)
        masks = batch["mask"].to(cfg.device, non_blocking=True)
        bboxes = batch["bbox"].to(cfg.device, non_blocking=True)

        if hasattr(torch.amp, "autocast"):
            with autocast("cuda", dtype=torch.float16, enabled=(cfg.device == "cuda")):
                logits = sam2_forward_with_boxes(model, images, bboxes)
        else:
            with autocast(device_type="cuda", dtype=torch.float16, enabled=(cfg.device == "cuda")):
                logits = sam2_forward_with_boxes(model, images, bboxes)

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        gts = (masks > 0.5).float()

        tp += int((preds * gts).sum().item())
        fp += int((preds * (1.0 - gts)).sum().item())
        fn += int(((1.0 - preds) * gts).sum().item())

        # save visualization: original / pred mask / overlay
        bs = images.shape[0]
        for i in range(bs):
            img_rgb = _denorm_image(images[i])
            pred_mask = preds[i, 0].detach().cpu().numpy().astype(np.uint8) * 255
            pred_bool = preds[i, 0].detach().cpu().numpy() > 0.5

            overlay = img_rgb.copy().astype(np.float32)
            alpha = 0.45
            red = np.zeros_like(overlay)
            red[..., 0] = 255
            overlay[pred_bool] = (1 - alpha) * overlay[pred_bool] + alpha * red[pred_bool]
            overlay = np.clip(overlay, 0, 255).astype(np.uint8)

            Image.fromarray(img_rgb).save(vis_dir / f"{sample_idx:06d}_image.png")
            Image.fromarray(pred_mask).save(vis_dir / f"{sample_idx:06d}_pred_mask.png")
            Image.fromarray(overlay).save(vis_dir / f"{sample_idx:06d}_overlay.png")
            sample_idx += 1

    iou = tp / (tp + fp + fn + eps)
    dice = (2 * tp) / (2 * tp + fp + fn + eps)

    return {
        "test_iou": float(iou),
        "test_dice": float(dice),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "num_saved_slices": int(sample_idx),
    }


def main():
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tuning of SAM2 using pre-sliced .npz data"
    )
    parser.add_argument("--slices-dir", type=str, required=True,
                        help="preprocess_brats_slices.py")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="SAM2 base checkpoint path")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-empty", action="store_true",
                        help="Include slices without foreground (default: only foreground slices are used)")
    parser.add_argument("--test-ratio", type=float, default=0.2,
                        help="Case-level test split ratio (default: 0.2)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    seed_everything(args.seed)

    cfg = TrainConfig(
        slices_dir=Path(args.slices_dir),
        output_dir=Path(args.output_dir),
        checkpoint=Path(args.checkpoint),
        model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml",
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_accum=args.grad_accum,
        seed=args.seed,
        fg_only=not args.include_empty,
    )

    # ---- Dataset ----
    print("Loading pre-sliced dataset...")
    full_dataset = PreslicedBratsDataset(
        slices_dir=cfg.slices_dir,
        fg_only=cfg.fg_only,
    )
    if len(full_dataset.records) == 0:
        raise RuntimeError("No valid records found in pre-sliced dataset.")

    if not (0.0 < args.test_ratio < 1.0):
        raise ValueError("--test-ratio must be in (0, 1).")

    all_cases = sorted({r["_case_id"] for r in full_dataset.records})
    if len(all_cases) < 2:
        raise RuntimeError("Need at least 2 cases for train/test split.")

    rng = random.Random(cfg.seed)
    shuffled_cases = all_cases.copy()
    rng.shuffle(shuffled_cases)

    n_test = max(1, int(round(len(shuffled_cases) * args.test_ratio)))
    n_test = min(n_test, len(shuffled_cases) - 1)
    test_cases = set(shuffled_cases[:n_test])
    train_cases = set(shuffled_cases[n_test:])

    train_dataset = PreslicedBratsDataset(
        slices_dir=cfg.slices_dir,
        fg_only=cfg.fg_only,
        allowed_cases=train_cases,
    )

    print(
        f"Total cases: {len(all_cases)} | Train cases: {len(train_cases)} | "
        f"Test cases: {len(test_cases)}"
    )
    print(f"Train slices: {len(train_dataset)}")

    test_dataset = PreslicedBratsDataset(
        slices_dir=cfg.slices_dir,
        fg_only=cfg.fg_only,
        allowed_cases=test_cases,
    )
    print(f"Test slices: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device == "cuda"),
        drop_last=False,
        persistent_workers=(cfg.num_workers > 0),
    )

    # ---- Model ----
    print("Loading SAM2 model...")
    model = build_sam2(
        config_file=cfg.model_cfg,
        ckpt_path=str(cfg.checkpoint),
        device="cuda",
        mode="eval",
    )
    model = model.to(cfg.device)

    lora_cfg = build_lora_config(r=16)
    model = get_peft_model(model, lora_cfg)
    model.train()
    print_trainable_params(model)

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    if hasattr(torch.amp, "autocast"):
        scaler = GradScaler("cuda", enabled=(cfg.device == "cuda"))
    else:
        scaler = GradScaler(enabled=(cfg.device == "cuda"))

    history = []
    dice_loss_fn = DiceLoss()

    # ---- Training loop ----
    print("Start training...")
    for epoch in range(1, cfg.epochs + 1):
        avg_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            dice_loss_fn=dice_loss_fn,
            cfg=cfg,
            epoch=epoch,
        )

        print(f"Epoch [{epoch}/{cfg.epochs}] loss={avg_loss:.6f}")
        history.append({"epoch": epoch, "loss": avg_loss})

        ckpt_path = os.path.join(args.output_dir, f"epoch_{epoch}.pt")
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "history": history,
                "config": vars(args),
            },
            ckpt_path,
        )

    final_path = os.path.join(args.output_dir, "final.pt")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "history": history,
            "config": vars(args),
        },
        final_path,
    )

    with open(os.path.join(args.output_dir, "train_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # save
    split_info = {
        "seed": args.seed,
        "test_ratio": args.test_ratio,
        "fg_only": cfg.fg_only,
        "train_cases": sorted(train_cases),
        "test_cases": sorted(test_cases),
        "num_train_cases": len(train_cases),
        "num_test_cases": len(test_cases),
        "num_train_slices": len(train_dataset),
        "num_test_slices": len(test_dataset),
    }
    split_path = os.path.join(args.output_dir, "test_split.json")
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split_info, f, ensure_ascii=False, indent=2)
    print(f"Test split saved to: {split_path}")
    print(f"  Train cases: {len(train_cases)} | Test cases: {len(test_cases)}")
    print(f"  Test case IDs: {sorted(test_cases)}")

    print(f"Training complete. Final model saved to: {final_path}")


if __name__ == "__main__":
    main()
