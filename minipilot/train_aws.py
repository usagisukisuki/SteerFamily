"""
train_aws.py - SageMaker 対応 TinySteerViT 学習スクリプト

【使い方 1】 SageMaker Training Job として Notebook からサブミット:

  from sagemaker.pytorch import PyTorch

  estimator = PyTorch(
      entry_point="train_aws.py",
      source_dir=".",          # dataset.py / model.py と同じディレクトリ
      role=role,
      instance_type="ml.p3.2xlarge",   # GPU インスタンス
      framework_version="2.1",
      py_version="py310",
      hyperparameters={
          "dataset":    "refcoco",
          "epochs":     50,
          "batch-size": 64,
          "teacher":    "bbox",
      },
  )
  estimator.fit({
      "coco":         "s3://your-bucket/datasets/coco/",
      "vg":           "s3://your-bucket/datasets/vg/",
      "gqa":          "s3://your-bucket/datasets/GQA/",
      "steervit-ckpt": "s3://your-bucket/models/steervit/",  # .pth を含むフォルダ
  })

【使い方 2】 Notebook Instance のターミナルから直接実行:

  python train_aws.py \\
      --coco-root /home/ec2-user/SageMaker/data/coco \\
      --s3-output s3://your-bucket/runs/exp01 \\
      --teacher bbox --dataset refcoco --epochs 30

SageMaker 入出力マッピング:
  SM_CHANNEL_COCO          → --coco-root
  SM_CHANNEL_VG            → --vg-root
  SM_CHANNEL_GQA           → --gqa-root
  SM_CHANNEL_STEERVIT_CKPT → --steervit-ckpt (ディレクトリ内の .pth を自動検索)
  SM_MODEL_DIR             → --out (チェックポイント保存先)
  SM_HPS                   → ハイパーパラメータ (CLI 引数で上書き可)
"""

from __future__ import annotations
import argparse
import json
import logging
import os
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

# SageMaker 環境検出
_IN_SM = "SM_TRAINING_ENV" in os.environ


def _install_steervit() -> None:
    """steervit を --no-deps でインストール（依存競合回避）。既インストール済みならスキップ。"""
    try:
        import steervit  # noqa: F401
    except ImportError:
        import subprocess
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--no-deps", "--quiet",
            "git+https://github.com/JonaRuthardt/SteerViT.git"
            "@a09bc78405d8bc4f686522df2e7f35ff3e611297",
        ])

if _IN_SM:
    _install_steervit()


# ─── SageMaker ユーティリティ ─────────────────────────────────────────────────

def _sm_channel(name: str) -> str | None:
    """SM_CHANNEL_{NAME} 環境変数からチャンネルパスを返す。なければ None。"""
    key = f"SM_CHANNEL_{name.upper().replace('-', '_')}"
    return os.environ.get(key)


def _sm_hyperparameters() -> dict[str, Any]:
    """SM_HPS からハイパーパラメータを辞書として取り出す。型変換も行う。"""
    if not _IN_SM:
        return {}
    hps = json.loads(os.environ.get("SM_HPS", "{}"))
    result: dict[str, Any] = {}
    for k, v in hps.items():
        if isinstance(v, str):
            if v.lower() == "true":
                result[k] = True
            elif v.lower() == "false":
                result[k] = False
            else:
                try:
                    result[k] = int(v)
                except ValueError:
                    try:
                        result[k] = float(v)
                    except ValueError:
                        result[k] = v
        else:
            result[k] = v
    # CLI 引数のキー形式 (batch-size → batch_size) に統一
    return {k.replace("-", "_"): v for k, v in result.items()}


def _resolve_steervit_ckpt(ckpt_arg: str, logger: logging.Logger) -> str:
    """
    SteerViT チェックポイントのパスを決定する。
      1. ckpt_arg がファイルなら直接使用
      2. ckpt_arg がディレクトリなら内部の *.pth を検索 (SM チャンネル想定)
      3. どちらも存在しなければ HuggingFace Hub から自動ダウンロード
    """
    p = Path(ckpt_arg)
    if p.is_file():
        return str(p)
    if p.is_dir():
        pths = sorted(p.glob("**/*.pth"))
        if pths:
            logger.info(f"  steervit-ckpt ディレクトリから発見: {pths[0]}")
            return str(pths[0])

    logger.warning(f"  SteerViT ckpt が見つかりません ({ckpt_arg})。HuggingFace Hub からダウンロードを試みます...")
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id="JonaRuthardt/SteerViT",
            filename="steervit_dinov2_base.pth",
        )
        logger.info(f"  HuggingFace Hub からダウンロード完了: {path}")
        return path
    except Exception as e:
        raise FileNotFoundError(
            f"SteerViT チェックポイントを取得できませんでした。\n"
            f"  試したパス: {ckpt_arg}\n"
            f"  HuggingFace エラー: {e}\n"
            f"  対処: --steervit-ckpt で .pth のパスを直接指定するか、\n"
            f"        SM チャンネル 'steervit-ckpt' に .pth を含む S3 フォルダを渡してください。"
        )


def _s3_upload(local_path: Path, s3_uri: str, logger: logging.Logger) -> None:
    """ファイルを S3 URI へアップロードする。boto3 が必要。"""
    try:
        import boto3
        from urllib.parse import urlparse
        parsed = urlparse(s3_uri)
        bucket  = parsed.netloc
        prefix  = parsed.path.lstrip("/")
        key     = f"{prefix}/{local_path.name}".lstrip("/")
        boto3.client("s3").upload_file(str(local_path), bucket, key)
        logger.info(f"  S3 アップロード完了: s3://{bucket}/{key}")
    except Exception as e:
        logger.warning(f"  S3 アップロード失敗 ({local_path.name}): {e}")


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


# ─── 可視化 ───────────────────────────────────────────────────────────────────

def _tensor_to_display(t: torch.Tensor) -> np.ndarray:
    t = t.cpu().float()
    t = (t - t.min()) / (t.max() - t.min() + 1e-6)
    return (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def _heatmap_overlay(img_np, heatmap_flat, grid=14, alpha=0.55, cmap="inferno"):
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


def _draw_bbox(img_np, bbox_rel):
    import copy
    img = copy.deepcopy(img_np)
    H, W = img.shape[:2]
    x1 = int(bbox_rel[0].item() * W); y1 = int(bbox_rel[1].item() * H)
    x2 = int(bbox_rel[2].item() * W); y2 = int(bbox_rel[3].item() * H)
    img[y1:y2, x1:min(x1+2, W)] = [255, 255, 0]
    img[y1:y2, max(x2-2, 0):x2] = [255, 255, 0]
    img[y1:min(y1+2, H), x1:x2] = [255, 255, 0]
    img[max(y2-2, 0):y2, x1:x2] = [255, 255, 0]
    return img


@torch.no_grad()
def run_visualization(sv, sv_teacher, probes, epoch, out_dir, device, logger, teacher_mode="steervit"):
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

        student_heat = steervit_heatmap_from_text(sv, full_t, [text], student_num_prefix)[0].cpu().numpy()
        axs = axes[row]
        axs[0].imshow(full_box)
        axs[0].set_title(f'"{text}"', fontsize=9)

        col = 1
        if teacher_mode in ("steervit", "both") and sv_teacher is not None:
            teacher_num_prefix = sv_teacher.vision_model.trunk.num_prefix_tokens
            teacher_t = F.interpolate(full_t, size=(STEERVIT_IMG_SIZE, STEERVIT_IMG_SIZE),
                                      mode="bicubic", align_corners=False)
            sv_heat = steervit_heatmap_from_text(sv_teacher, teacher_t, [text], teacher_num_prefix)[0].cpu().numpy()
            axs[col].imshow(_heatmap_overlay(full_np, sv_heat, grid=STEERVIT_GRID))
            axs[col].set_title(f"teacher (SteerViT)\nmax={sv_heat.max():.3f}  mean={sv_heat.mean():.3f}", fontsize=9)
            col += 1

        if teacher_mode in ("bbox", "both"):
            bbox_dist = bbox_to_patch_dist(bbox_rel.unsqueeze(0).to(device), TINY_VIT_GRID)[0].cpu().numpy()
            axs[col].imshow(_heatmap_overlay(full_np, bbox_dist, grid=TINY_VIT_GRID))
            axs[col].set_title("teacher (bbox mask)", fontsize=9)
            col += 1

        axs[col].imshow(_heatmap_overlay(full_np, student_heat, grid=TINY_VIT_GRID))
        axs[col].set_title(f"student (TinySteerViT)\nmax={student_heat.max():.3f}  mean={student_heat.mean():.3f}", fontsize=9)
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

    # SM チャンネルパスをデフォルト値として使用
    sm_coco = _sm_channel("coco")
    sm_vg   = _sm_channel("vg")
    sm_gqa  = _sm_channel("gqa")
    sm_ckpt = _sm_channel("steervit-ckpt") or _sm_channel("steervit_ckpt")
    sm_out  = os.environ.get("SM_MODEL_DIR", "./ckpts")

    p = argparse.ArgumentParser(description="TinySteerViT 学習 (SageMaker 対応)")
    p.add_argument("--config", type=str, default=None)
    # ── Paths ──────────────────────────────────────────────────────────────
    p.add_argument("--coco-root",     type=str, default=sm_coco or "/data2/kato/dataset/coco")
    p.add_argument("--vg-root",       type=str, default=sm_vg   or "/data2/kato/dataset/vg")
    p.add_argument("--gqa-root",      type=str, default=sm_gqa  or "/data2/kato/dataset/GQA")
    p.add_argument("--steervit-ckpt", type=str, default=sm_ckpt or STEERVIT_CKPT)
    # ── Model / teacher ────────────────────────────────────────────────────
    p.add_argument("--teacher",  type=str,   default="steervit",
                   choices=["steervit", "bbox", "both"])
    p.add_argument("--no-pretrained-backbone", action="store_true", default=False)
    p.add_argument("--text-encoder", type=str, default="distilroberta",
                   choices=list(TEXT_ENCODERS))
    # ── Data ───────────────────────────────────────────────────────────────
    p.add_argument("--dataset",   type=str, default="refcoco",
                   choices=["refcoco", "refcoco+", "refcocog", "vg", "gqa", "coco", "all"])
    p.add_argument("--max-train", type=int, default=None)
    p.add_argument("--max-val",   type=int, default=None)
    # ── Training ───────────────────────────────────────────────────────────
    p.add_argument("--epochs",       type=int,   default=50)
    p.add_argument("--batch-size",   type=int,   default=64)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--num-workers",  type=int,   default=4)
    p.add_argument("--out",          type=str,   default=sm_out)
    p.add_argument("--resume",       type=str,   default=None)
    p.add_argument("--amp",          action="store_true", default=True)
    p.add_argument("--seed",         type=int,   default=42)
    # ── Visualisation ──────────────────────────────────────────────────────
    p.add_argument("--vis-every",   type=int, default=5)
    p.add_argument("--vis-n",       type=int, default=4)
    p.add_argument("--vis-samples", type=str, default=None)
    # ── Attention distillation ────────────────────────────────────────────
    p.add_argument("--attn-distil", action="store_true", default=False)
    p.add_argument("--attn-weight", type=float, default=1.0)
    p.add_argument("--sa-weight",   type=float, default=1.0)
    p.add_argument("--ca-weight",   type=float, default=1.0)
    # ── AWS / S3 ───────────────────────────────────────────────────────────
    p.add_argument("--s3-output", type=str, default=None,
                   help="チェックポイントを逐次アップロードする S3 URI (例: s3://bucket/runs/exp01)")

    # 優先順位: CLI引数 > SM_HPS > config.yaml > ハードコードデフォルト
    p.set_defaults(**cfg_defaults)
    p.set_defaults(**_sm_hyperparameters())
    return p.parse_args()


# ─── Epoch ────────────────────────────────────────────────────────────────────

def run_epoch(sv, heatmap_crit, loader, optimizer, scaler, device, train, print_every=50):
    sv.train(train)
    total_loss   = 0.0
    n_batches    = 0
    n_total      = len(loader)
    trainable_params = [p for p in sv.parameters() if p.requires_grad]
    need_bbox    = heatmap_crit.teacher_mode in ("bbox", "both")
    phase        = "train" if train else "val"
    t0           = time.time()

    for full_imgs, bbox_rels, texts in loader:
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
        if n_batches % print_every == 0 or n_batches == n_total:
            elapsed = time.time() - t0
            avg_loss = total_loss / n_batches
            print(
                f"  [{phase}] {n_batches:4d}/{n_total}  loss={avg_loss:.4f}  {elapsed:.0f}s",
                flush=True,
            )
    return total_loss / max(n_batches, 1)


# ─── メイン ───────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(out_dir)
    logger.info("=" * 60)
    logger.info(f"SageMaker 環境: {_IN_SM}")
    logger.info(f"Device: {DEVICE}  AMP: {args.amp}  teacher: {args.teacher}  text_encoder: {args.text_encoder}")
    logger.info(f"Args: {json.dumps(vars(args), ensure_ascii=False)}")

    # ── TinySteerViT (student) ────────────────────────────────────────────
    logger.info(f"TinySteerViT 構築中... (text_encoder={args.text_encoder})")
    sv = TinySteerViT(
        pretrained_backbone=not args.no_pretrained_backbone,
        text_encoder=args.text_encoder,
    ).to(DEVICE)

    for name, param in sv.named_parameters():
        param.requires_grad_(
            "connector" in name or "gated_cross_attn" in name or "lin_seg_head" in name
        )

    n_trainable = sum(p.numel() for p in sv.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in sv.parameters())
    logger.info(f"  学習パラメータ: {n_trainable:,} / {n_total:,}")

    # ── SteerViT teacher ─────────────────────────────────────────────────
    if args.teacher in ("steervit", "both"):
        ckpt_path = _resolve_steervit_ckpt(args.steervit_ckpt, logger)
        logger.info(f"SteerViT teacher ロード中: {ckpt_path}")
        from steervit.model import SteerViT
        sv_teacher = SteerViT.from_pretrained(ckpt_path, device=DEVICE)
        sv_teacher.eval()
        for p in sv_teacher.parameters():
            p.requires_grad_(False)
    else:
        logger.info("teacher: bbox mask のみ (SteerViT はロードしない)")
        sv_teacher = None

    # ── Data loaders ──────────────────────────────────────────────────────
    logger.info(f"データローダー作成中... (dataset={args.dataset})")
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
                transforms.Resize((TINY_VIT_IMG_SIZE, TINY_VIT_IMG_SIZE),
                                  interpolation=transforms.InterpolationMode.BICUBIC),
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
            if args.s3_output:
                _s3_upload(ckpt_path, args.s3_output, logger)

        with open(out_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        if probes and args.vis_every > 0 and epoch % args.vis_every == 0:
            run_visualization(sv, sv_teacher, probes, epoch, out_dir, DEVICE, logger, teacher_mode=args.teacher)

    logger.info(f"学習完了。best val_loss={best_val_loss:.4f}")
    logger.info(f"チェックポイント: {out_dir / 'best_model.pth'}")

    if args.s3_output:
        for f in [out_dir / "history.json", out_dir / "train.log"]:
            if f.exists():
                _s3_upload(f, args.s3_output, logger)


if __name__ == "__main__":
    main()
