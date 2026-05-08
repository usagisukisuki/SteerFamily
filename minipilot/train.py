"""
train.py - TinySteerViT 学習スクリプト

損失:
  HeatmapSoftCE(teacher=SteerViT pretrained, student=TinySteerViT)

学習対象:
  TinySteerViT の gated_cross_attn と lin_seg_head のみ。
  ViT-Tiny backbone / DistilRoBERTa / connector はすべて凍結。

出力 (--out ディレクトリ):
  train.log         : タイムスタンプ付き学習ログ
  history.json      : エポックごとの loss 記録
  best_model.pth    : val_loss 最良チェックポイント
  vis/epoch_NNN.png : heatmap 可視化 (--vis-every で制御)

使用例:
  python train.py --dataset refcoco
  python train.py --dataset refcoco --epochs 50 --batch-size 64

--vis-samples JSON フォーマット:
  [{"img_path": "/path/img.jpg", "text": "person on the left"}, ...]
"""

from __future__ import annotations
import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torchvision import transforms
from tqdm import tqdm

from dataset import build_loaders
from model import (
    HeatmapSoftCELoss,
    TinySteerViT,
    TEXT_ENCODERS,
    STEERVIT_CKPT,
    STEERVIT_GRID,
    STEERVIT_IMG_SIZE,
    TINY_VIT_GRID,
    TINY_VIT_IMG_SIZE,
    bbox_to_patch_dist,
    steervit_heatmap_from_text,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_FULL_MEAN = (0.485, 0.456, 0.406)
_FULL_STD  = (0.229, 0.224, 0.225)


# ─── Logging ──────────────────────────────────────────────────────────────────
def setup_logger(out_dir: Path) -> logging.Logger:
    logger = logging.getLogger("tinyvit_steer")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(out_dir / "train.log", mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


# ─── Probe 収集 ───────────────────────────────────────────────────────────────
def collect_probes_from_loader(val_loader, n: int) -> list[dict]:
    probes = []
    for full_imgs, bbox_rels, texts in val_loader:
        for i in range(full_imgs.shape[0]):
            probes.append({
                "full_tensor": full_imgs[i].clone(),
                "bbox_rel":    bbox_rels[i].clone(),
                "text":        texts[i],
            })
            if len(probes) >= n:
                return probes
    return probes


def load_probes_from_json(json_path: str, full_transform) -> list[dict]:
    """
    JSON フォーマット:
      [{"img_path": "...", "text": "...", "bbox": [x1,y1,x2,y2]}]  ← bbox は相対座標、省略可
    """
    from PIL import Image as PILImage
    with open(json_path, encoding="utf-8") as f:
        entries = json.load(f)
    probes = []
    for e in entries:
        bbox = e.get("bbox")
        probes.append({
            "full_tensor": full_transform(PILImage.open(e["img_path"]).convert("RGB")),
            "bbox_rel":    torch.tensor(bbox, dtype=torch.float32) if bbox else torch.tensor([0., 0., 1., 1.]),
            "text":        e["text"],
        })
    return probes


# ─── 可視化ヘルパー ───────────────────────────────────────────────────────────
def _tensor_to_display(t: torch.Tensor) -> np.ndarray:
    t = t.cpu().float()
    t = (t - t.min()) / (t.max() - t.min() + 1e-6)
    return (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def _heatmap_overlay(
    img_np: np.ndarray,
    heatmap_flat: np.ndarray,
    grid: int    = 14,
    alpha: float = 0.55,
    cmap: str    = "inferno",
) -> np.ndarray:
    from matplotlib import cm as mplcm
    from PIL import Image as PILImage
    H, W     = img_np.shape[:2]
    heat     = heatmap_flat.reshape(grid, grid)
    heat     = (heat - heat.min()) / (heat.max() - heat.min() + 1e-6)
    heat_pil = PILImage.fromarray((heat * 255).astype(np.uint8), mode="L")
    heat_pil = heat_pil.resize((W, H), PILImage.BILINEAR)
    heat_np  = np.array(heat_pil) / 255.0
    colored  = mplcm.get_cmap(cmap)(heat_np)[:, :, :3]
    return (alpha * colored * 255 + (1 - alpha) * img_np).clip(0, 255).astype(np.uint8)


def _draw_bbox(img_np: np.ndarray, bbox_rel: torch.Tensor) -> np.ndarray:
    """bbox_rel [x1,y1,x2,y2] in [0,1] を画像に矩形描画して返す。"""
    import copy
    img = copy.deepcopy(img_np)
    H, W = img.shape[:2]
    x1 = int(bbox_rel[0].item() * W)
    y1 = int(bbox_rel[1].item() * H)
    x2 = int(bbox_rel[2].item() * W)
    y2 = int(bbox_rel[3].item() * H)
    img[y1:y2, x1:min(x1+2, W)] = [255, 255, 0]
    img[y1:y2, max(x2-2, 0):x2] = [255, 255, 0]
    img[y1:min(y1+2, H), x1:x2] = [255, 255, 0]
    img[max(y2-2, 0):y2, x1:x2] = [255, 255, 0]
    return img


@torch.no_grad()
def run_visualization(
    sv,
    sv_teacher,       # None の場合は SteerViT heatmap 列を省略
    probes:   list[dict],
    epoch:    int,
    out_dir:  Path,
    device:   str,
    logger:   logging.Logger,
    teacher_mode: str = "steervit",
) -> None:
    """
    レイアウト (列):
      steervit : [full image + bbox] | [teacher (SteerViT)] | [student]
      bbox     : [full image + bbox] | [teacher (bbox mask)] | [student]
      both     : [full image + bbox] | [teacher (SteerViT)] | [teacher (bbox)] | [student]
    """
    if not probes:
        return

    student_num_prefix = sv.vision_model.trunk.num_prefix_tokens
    n_cols       = 4 if teacher_mode == "both" else 3
    n            = len(probes)
    was_training = sv.training
    sv.eval()

    fig, axes = plt.subplots(n, n_cols, figsize=(5 * n_cols, 5 * n), squeeze=False)

    for row, probe in enumerate(probes):
        full_t   = probe["full_tensor"].unsqueeze(0).to(device)
        bbox_rel = probe["bbox_rel"]
        text     = probe["text"]

        full_np  = _tensor_to_display(probe["full_tensor"])
        full_box = _draw_bbox(full_np, bbox_rel)

        student_heat = steervit_heatmap_from_text(
            sv, full_t, [text], student_num_prefix
        )[0].cpu().numpy()

        axs = axes[row]
        axs[0].imshow(full_box)
        axs[0].set_title(f'"{text}"', fontsize=9)

        col = 1
        if teacher_mode in ("steervit", "both") and sv_teacher is not None:
            teacher_num_prefix = sv_teacher.vision_model.trunk.num_prefix_tokens
            teacher_t = F.interpolate(
                full_t, size=(STEERVIT_IMG_SIZE, STEERVIT_IMG_SIZE),
                mode="bicubic", align_corners=False,
            )
            sv_heat = steervit_heatmap_from_text(
                sv_teacher, teacher_t, [text], teacher_num_prefix
            )[0].cpu().numpy()
            axs[col].imshow(_heatmap_overlay(full_np, sv_heat, grid=STEERVIT_GRID))
            axs[col].set_title(
                f"teacher (SteerViT)\nmax={sv_heat.max():.3f}  mean={sv_heat.mean():.3f}",
                fontsize=9,
            )
            col += 1

        if teacher_mode in ("bbox", "both"):
            bbox_dist = bbox_to_patch_dist(
                bbox_rel.unsqueeze(0).to(device), TINY_VIT_GRID
            )[0].cpu().numpy()
            axs[col].imshow(_heatmap_overlay(full_np, bbox_dist, grid=TINY_VIT_GRID))
            axs[col].set_title("teacher (bbox mask)", fontsize=9)
            col += 1

        axs[col].imshow(_heatmap_overlay(full_np, student_heat, grid=TINY_VIT_GRID))
        axs[col].set_title(
            f"student (TinySteerViT)\nmax={student_heat.max():.3f}  mean={student_heat.mean():.3f}",
            fontsize=9,
        )

        for ax in axs:
            ax.axis("off")

    fig.suptitle(f"Epoch {epoch}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    vis_dir   = out_dir / "vis"
    vis_dir.mkdir(exist_ok=True)
    save_path = vis_dir / f"epoch_{epoch:03d}.png"
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    sv.train(was_training)
    logger.info(f"  可視化保存: {save_path}")


# ─── Config loader ────────────────────────────────────────────────────────────
def _load_yaml(path: str) -> dict[str, Any]:
    try:
        import yaml
    except ImportError:
        raise SystemExit("PyYAML が必要です: pip install pyyaml")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _config_defaults(config_path: str | None) -> dict[str, Any]:
    """config.yaml から train セクション + paths セクションをフラットに返す。"""
    if not config_path:
        return {}
    cfg = _load_yaml(config_path)
    defaults: dict[str, Any] = {}
    defaults.update(cfg.get("paths", {}))
    defaults.update(cfg.get("train", {}))
    return {k: v for k, v in defaults.items() if v is not None}


# ─── Argument parser ──────────────────────────────────────────────────────────
def parse_args():
    # 1st pass: --config のみ先読み
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    pre_args, _ = pre.parse_known_args()

    cfg_defaults = _config_defaults(pre_args.config)

    p = argparse.ArgumentParser(description="TinySteerViT 学習")
    p.add_argument("--config", type=str, default=None,
                   help="YAML config ファイル (CLI 引数で上書き可)")
    # ── Paths ──────────────────────────────────────────────────────────────
    p.add_argument("--coco-root",     type=str, default="/data2/kato/dataset/coco")
    p.add_argument("--vg-root",       type=str, default="/data2/kato/dataset/vg")
    p.add_argument("--gqa-root",      type=str, default="/data2/kato/dataset/GQA")
    p.add_argument("--steervit-ckpt", type=str, default=STEERVIT_CKPT)
    # ── Model / teacher ────────────────────────────────────────────────────
    p.add_argument("--teacher",  type=str,   default="steervit",
                   choices=["steervit", "bbox", "both"],
                   help="teacher の種類: steervit / bbox (GT bbox mask) / both (平均)")
    p.add_argument("--no-pretrained-backbone", action="store_true", default=False,
                   help="ViT-Tiny backbone も完全ランダム初期化 (デフォルト: ImageNet pretrained)")
    p.add_argument("--text-encoder", type=str, default="distilroberta",
                   choices=list(TEXT_ENCODERS),
                   help=f"テキストエンコーダ: {list(TEXT_ENCODERS)}")
    # ── Data ───────────────────────────────────────────────────────────────
    p.add_argument("--dataset",      type=str,   default="refcoco",
                   choices=["refcoco", "refcoco+", "refcocog", "vg", "gqa", "coco", "all"])
    p.add_argument("--max-train",    type=int,   default=None)
    p.add_argument("--max-val",      type=int,   default=None)
    # ── Training ───────────────────────────────────────────────────────────
    p.add_argument("--epochs",       type=int,   default=50)
    p.add_argument("--batch-size",   type=int,   default=64)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--num-workers",  type=int,   default=8)
    p.add_argument("--out",          type=str,   default="./ckpts")
    p.add_argument("--resume",       type=str,   default=None,
                   help="チェックポイントから学習パラメータを再開ロード")
    p.add_argument("--amp",          action="store_true", default=True)
    p.add_argument("--seed",         type=int,   default=42)
    # ── Visualisation ──────────────────────────────────────────────────────
    p.add_argument("--vis-every",    type=int,   default=5,
                   help="N エポックごとに heatmap 可視化 (0 で無効)")
    p.add_argument("--vis-n",        type=int,   default=4,
                   help="--vis-samples 未指定時に val set から使う probe 数")
    p.add_argument("--vis-samples",  type=str,   default=None,
                   help="可視化 probe JSON (省略時は val set から自動選択)")
    # ── Attention distillation ────────────────────────────────────────────
    p.add_argument("--attn-distil",  action="store_true", default=False,
                   help="中間ブロックの SA / CA attention map を蒸留する (steervit / both モードのみ)")
    p.add_argument("--attn-weight",  type=float, default=1.0,
                   help="attention distil loss の heatmap loss に対する重み")
    p.add_argument("--sa-weight",    type=float, default=1.0,
                   help="SA (self-attention) 蒸留損失の重み (attn-weight と乗算)")
    p.add_argument("--ca-weight",    type=float, default=1.0,
                   help="CA (cross-attention) 蒸留損失の重み (attn-weight と乗算)")

    p.set_defaults(**cfg_defaults)
    return p.parse_args()


# ─── Epoch ────────────────────────────────────────────────────────────────────
def run_epoch(
    sv,
    heatmap_crit: HeatmapSoftCELoss,
    loader,
    optimizer:    torch.optim.Optimizer | None,
    scaler:       GradScaler | None,
    device:       str,
    train:        bool,
) -> float:
    sv.train(train)
    total_loss = 0.0
    n_batches  = 0

    trainable_params = [p for p in sv.parameters() if p.requires_grad]
    need_bbox = heatmap_crit.teacher_mode in ("bbox", "both")

    phase = "train" if train else "val"
    pbar  = tqdm(loader, desc=phase, leave=False, dynamic_ncols=True)
    for full_imgs, bbox_rels, texts in pbar:
        full_imgs = full_imgs.to(device, non_blocking=True)
        bboxes    = bbox_rels.to(device, non_blocking=True) if need_bbox else None

        with autocast("cuda", enabled=(scaler is not None)):
            loss = heatmap_crit(full_imgs, list(texts), bboxes=bboxes)

        if train:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()

        total_loss += loss.item()
        n_batches  += 1
        pbar.set_postfix(loss=f"{total_loss / n_batches:.4f}")

    return total_loss / max(n_batches, 1)


# ─── メイン ───────────────────────────────────────────────────────────────────
def main():
    args    = parse_args()
    torch.manual_seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(out_dir)
    logger.info("=" * 60)
    logger.info(f"Device: {DEVICE}  AMP: {args.amp}  teacher: {args.teacher}  text_encoder: {args.text_encoder}")
    logger.info(f"Args: {json.dumps(vars(args), ensure_ascii=False)}")

    # ── TinySteerViT (student) ────────────────────────────────────────────
    logger.info(f"TinySteerViT 構築中... (text_encoder={args.text_encoder})")
    sv = TinySteerViT(
        pretrained_backbone=not args.no_pretrained_backbone,
        text_encoder=args.text_encoder,
    ).to(DEVICE)

    # 学習対象: connector / gated_cross_attn / lin_seg_head
    for name, param in sv.named_parameters():
        param.requires_grad_(
            "connector" in name or "gated_cross_attn" in name or "lin_seg_head" in name
        )

    n_trainable = sum(p.numel() for p in sv.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in sv.parameters())
    logger.info(f"  学習パラメータ: {n_trainable:,} / {n_total:,}  (connector + gated_cross_attn + lin_seg_head)")

    # ── SteerViT teacher (steervit / both モードのみロード) ───────────────
    if args.teacher in ("steervit", "both"):
        logger.info("SteerViT teacher ロード中 (frozen)...")
        from steervit.model import SteerViT
        sv_teacher = SteerViT.from_pretrained(args.steervit_ckpt, device=DEVICE)
        sv_teacher.eval()
        for p in sv_teacher.parameters():
            p.requires_grad_(False)
    else:
        logger.info("teacher: bbox mask のみ (SteerViT はロードしない)")
        sv_teacher = None

    # ── Data loaders ──────────────────────────────────────────────────────
    logger.info(f"データローダー作成中... (full_img_size={TINY_VIT_IMG_SIZE})")
    train_loader, val_loader = build_loaders(
        dataset       = args.dataset,
        full_img_size = TINY_VIT_IMG_SIZE,
        batch_size    = args.batch_size,
        num_workers   = args.num_workers,
        coco_root     = args.coco_root,
        vg_root       = args.vg_root,
        gqa_root      = args.gqa_root,
        max_train     = args.max_train,
        max_val       = args.max_val,
        seed          = args.seed,
    )

    # ── Loss ──────────────────────────────────────────────────────────────
    attn_weight = args.attn_weight if args.attn_distil else 0.0
    if args.attn_distil and args.teacher == "bbox":
        logger.warning("--attn-distil は steervit / both モードのみ有効。bbox モードでは無視します。")
        attn_weight = 0.0
    heatmap_crit = HeatmapSoftCELoss(
        sv=sv, sv_teacher=sv_teacher,
        grid_size=TINY_VIT_GRID, teacher_grid_size=STEERVIT_GRID,
        teacher_img_size=STEERVIT_IMG_SIZE,
        teacher_mode=args.teacher,
        attn_weight=attn_weight,
        sa_weight=args.sa_weight,
        ca_weight=args.ca_weight,
    ).to(DEVICE)
    if attn_weight > 0:
        logger.info(
            f"  Attention distil 有効: attn_weight={attn_weight}  "
            f"sa_weight={args.sa_weight}  ca_weight={args.ca_weight}"
        )

    if args.resume:
        ckpt  = torch.load(args.resume, map_location=DEVICE)
        sv_sd = sv.state_dict()
        sv_sd.update(ckpt["sv_state"])
        sv.load_state_dict(sv_sd)
        logger.info(f"  -> {args.resume} から再開 (val_loss={ckpt['val_loss']:.4f})")

    # ── Optimizer & Scheduler ─────────────────────────────────────────────
    trainable = [p for p in sv.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    scaler = GradScaler("cuda") if args.amp and DEVICE == "cuda" else None

    # ── Probe 収集 ────────────────────────────────────────────────────────
    probes = []
    if args.vis_every > 0:
        if args.vis_samples:
            full_tf = transforms.Compose([
                transforms.Resize(
                    (TINY_VIT_IMG_SIZE, TINY_VIT_IMG_SIZE),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(_FULL_MEAN, _FULL_STD),
            ])
            probes = load_probes_from_json(args.vis_samples, full_tf)
            logger.info(f"可視化 probe: {len(probes)} 件 ({args.vis_samples})")
        else:
            probes = collect_probes_from_loader(val_loader, args.vis_n)
            logger.info(f"可視化 probe: {len(probes)} 件 (val set から自動収集)")

        run_visualization(sv, sv_teacher, probes, 0, out_dir, DEVICE, logger, teacher_mode=args.teacher)

    # ── 学習ループ ────────────────────────────────────────────────────────
    history       = []
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = run_epoch(sv, heatmap_crit, train_loader, optimizer, scaler, DEVICE, train=True)
        scheduler.step()

        with torch.no_grad():
            val_loss = run_epoch(sv, heatmap_crit, val_loader, None, None, DEVICE, train=False)

        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]["lr"]

        log = {
            "epoch":       epoch,
            "train_loss":  round(train_loss, 4),
            "val_loss":    round(val_loss,   4),
            "lr":          round(lr_now, 8),
            "elapsed_sec": round(elapsed, 1),
        }
        if attn_weight > 0:
            log["last_heatmap_loss"] = round(heatmap_crit.last_heatmap_loss, 4)
            log["last_sa_loss"]      = round(heatmap_crit.last_sa_loss, 4)
            log["last_ca_loss"]      = round(heatmap_crit.last_ca_loss, 4)
        history.append(log)
        attn_suffix = (
            f"  hmap={heatmap_crit.last_heatmap_loss:.4f}"
            f"  sa={heatmap_crit.last_sa_loss:.4f}"
            f"  ca={heatmap_crit.last_ca_loss:.4f}"
            if attn_weight > 0 else ""
        )
        logger.info(
            f"[{epoch:3d}/{args.epochs}] "
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
            f"lr={lr_now:.2e}  {elapsed:.1f}s{attn_suffix}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path     = out_dir / "best_model.pth"
            sv_keys       = [name for name, p in sv.named_parameters() if p.requires_grad]
            torch.save(
                {
                    "epoch":    epoch,
                    "sv_state": {k: sv.state_dict()[k].cpu() for k in sv_keys},
                    "val_loss": best_val_loss,
                    "args":     vars(args),
                },
                ckpt_path,
            )
            logger.info(f"  -> best checkpoint 保存: {ckpt_path}  (val_loss={best_val_loss:.4f})")

        with open(out_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        if probes and args.vis_every > 0 and epoch % args.vis_every == 0:
            run_visualization(sv, sv_teacher, probes, epoch, out_dir, DEVICE, logger, teacher_mode=args.teacher)

    logger.info(f"学習完了。best val_loss={best_val_loss:.4f}")
    logger.info(f"チェックポイント: {out_dir / 'best_model.pth'}")


if __name__ == "__main__":
    main()
