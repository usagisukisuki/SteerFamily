"""
precompute_teacher_aws.py - SteerViT teacher outputs の事前計算 (AWS / Notebook Instance 対応版)

precompute_teacher.py の AWS 対応版。以下の点が異なる:
  - tqdm 不使用 (SageMaker ログとの相性問題を回避)
  - steervit パッケージを自動インストール (_install_steervit)
  - --s3-output で生成キャッシュを S3 へ自動アップロード
  - SM_CHANNEL_* 環境変数からデータパスを取得
  - _teacher_forward_collect が旧 model.py にない場合は --no-attn を強制

以下をキャッシュとして保存:
  soft.dat   - softmax heatmap      (N, T) float16         T = teacher_grid**2
  sa.dat     - SA 圧縮分布          (N, n_blocks, T) float16  [--no-attn でスキップ]
  ca.dat     - CA 圧縮分布          (N, n_blocks, T) float16  [--no-attn でスキップ]
  meta.json  - メタデータ

使い方:
  python precompute_teacher_aws.py --dataset refcoco --out ./teacher_cache/refcoco
  python precompute_teacher_aws.py --dataset refcoco --out ./teacher_cache/refcoco --no-attn
  python precompute_teacher_aws.py --dataset refcoco --out ./teacher_cache/refcoco \\
      --s3-output s3://your-bucket/teacher_cache/refcoco
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import STEERVIT_CKPT, STEERVIT_GRID, STEERVIT_IMG_SIZE
try:
    from model import _teacher_forward_collect
    _HAS_COLLECT = True
except ImportError:
    _HAS_COLLECT = False

from dataset import (
    COCODetDataset,
    GQASceneGraphDataset,
    RefCOCOLocalDataset,
    VGRegionDataset,
    COCO_ROOT,
    GQA_ROOT,
    REFCOCO_DIR,
    REFCOCO_PLUS_DIR,
    REFCOCOG_DIR,
    VG_ROOT,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_IN_SM = "SM_TRAINING_ENV" in os.environ


# ─── SageMaker ユーティリティ ─────────────────────────────────────────────────

def _sm_channel(name: str) -> str | None:
    key = f"SM_CHANNEL_{name.upper().replace('-', '_')}"
    return os.environ.get(key)


def _install_steervit() -> None:
    """steervit を --no-deps でインストール。既インストール済みならスキップ。"""
    try:
        import steervit  # noqa: F401
    except ImportError:
        import subprocess
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--no-deps", "--quiet",
            "git+https://github.com/JonaRuthardt/SteerViT.git"
            "@a09bc78405d8bc4f686522df2e7f35ff3e611297",
        ])


def _s3_upload_dir(local_dir: str, s3_uri: str) -> None:
    """ディレクトリ内の全ファイルを S3 URI へアップロードする。"""
    try:
        import boto3
        from urllib.parse import urlparse
        parsed = urlparse(s3_uri)
        bucket = parsed.netloc
        prefix = parsed.path.lstrip("/")
        client = boto3.client("s3")
        for fname in os.listdir(local_dir):
            local_path = os.path.join(local_dir, fname)
            if os.path.isfile(local_path):
                key = f"{prefix}/{fname}".lstrip("/")
                client.upload_file(local_path, bucket, key)
                print(f"  S3 アップロード完了: s3://{bucket}/{key}")
    except Exception as e:
        print(f"  Warning: S3 アップロード失敗: {e}")


if _IN_SM:
    _install_steervit()


# ─── Dataset ─────────────────────────────────────────────────────────────────

def _build_dataset(args) -> torch.utils.data.Dataset:
    common_kw = dict(full_img_size=STEERVIT_IMG_SIZE, seed=args.seed)
    split = args.split

    if args.dataset == "refcoco":
        return RefCOCOLocalDataset(
            refcoco_dir=REFCOCO_DIR, coco_root=args.coco_root,
            split=split, fix_text=True, **common_kw,
        )
    elif args.dataset == "refcoco+":
        return RefCOCOLocalDataset(
            refcoco_dir=REFCOCO_PLUS_DIR, coco_root=args.coco_root,
            split=split, fix_text=True, **common_kw,
        )
    elif args.dataset == "refcocog":
        return RefCOCOLocalDataset(
            refcoco_dir=REFCOCOG_DIR, coco_root=args.coco_root,
            split=split, fix_text=True, **common_kw,
        )
    elif args.dataset == "vg":
        return VGRegionDataset(vg_root=args.vg_root, split=split, **common_kw)
    elif args.dataset == "gqa":
        return GQASceneGraphDataset(gqa_root=args.gqa_root, split=split, **common_kw)
    elif args.dataset == "coco":
        ann_map = {
            "train": os.path.join(args.coco_root, "annotations/instances_train2017.json"),
            "val":   os.path.join(args.coco_root, "annotations/instances_val2017.json"),
        }
        img_map = {
            "train": os.path.join(args.coco_root, "train2017"),
            "val":   os.path.join(args.coco_root, "val2017"),
        }
        if split not in ann_map:
            raise ValueError(f"coco dataset supports split=train|val, got {split!r}")
        return COCODetDataset(ann_file=ann_map[split], image_dir=img_map[split], **common_kw)
    else:
        raise ValueError(
            f"Unsupported dataset: {args.dataset!r}. "
            "Choose from refcoco, refcoco+, refcocog, vg, gqa, coco."
        )


def _collate(batch):
    full_imgs, bbox_rels, texts = zip(*batch)
    return torch.stack(full_imgs), torch.stack(bbox_rels), list(texts)


# ─── SA / CA 圧縮 ─────────────────────────────────────────────────────────────

def _compress_sa(sa_map: torch.Tensor, num_prefix: int) -> np.ndarray:
    """(B, H, N+prefix, N+prefix) → (B, T) float16: CLS→patch, head-averaged, normalized."""
    cls_row = sa_map[:, :, 0, num_prefix:].mean(dim=1)
    cls_row = cls_row / cls_row.sum(-1, keepdim=True).clamp(min=1e-8)
    return cls_row.cpu().to(torch.float16).numpy()


def _compress_ca(ca_map: torch.Tensor, num_prefix: int) -> np.ndarray:
    """(B, H, N+prefix, L) → (B, T) float16: patch→text, head+text averaged, normalized."""
    patch_ca = ca_map[:, :, num_prefix:, :].mean(dim=1).mean(dim=-1)
    patch_ca = patch_ca / patch_ca.sum(-1, keepdim=True).clamp(min=1e-8)
    return patch_ca.cpu().to(torch.float16).numpy()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    sm_coco = _sm_channel("coco")
    sm_vg   = _sm_channel("vg")
    sm_gqa  = _sm_channel("gqa")

    p = argparse.ArgumentParser(description="SteerViT teacher outputs の事前計算 (AWS 対応版)")
    p.add_argument("--dataset",       type=str, required=True,
                   choices=["refcoco", "refcoco+", "refcocog", "vg", "gqa", "coco"])
    p.add_argument("--out",           type=str, required=True,
                   help="出力ディレクトリ (meta.json / *.dat を書き出す)")
    p.add_argument("--split",         type=str, default="train",
                   help="split: train | val  (デフォルト: train)")
    p.add_argument("--steervit-ckpt", type=str, default=STEERVIT_CKPT)
    p.add_argument("--no-attn",       action="store_true", default=False,
                   help="SA/CA attention maps を保存しない (soft heatmap のみ、高速)")
    p.add_argument("--batch-size",    type=int, default=32)
    p.add_argument("--num-workers",   type=int, default=4)
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--coco-root",     type=str, default=sm_coco or COCO_ROOT)
    p.add_argument("--vg-root",       type=str, default=sm_vg   or VG_ROOT)
    p.add_argument("--gqa-root",      type=str, default=sm_gqa  or GQA_ROOT)
    p.add_argument("--s3-output",     type=str, default=None,
                   help="完了後にキャッシュをアップロードする S3 URI (例: s3://bucket/teacher_cache/refcoco)")
    p.add_argument("--print-every",   type=int, default=50,
                   help="N バッチごとに進捗を表示 (デフォルト: 50)")
    args = p.parse_args()

    # _teacher_forward_collect が旧 model.py にない場合は --no-attn を強制
    if not _HAS_COLLECT and not args.no_attn:
        print("Warning: _teacher_forward_collect が model.py にありません。--no-attn を強制します。")
        args.no_attn = True

    os.makedirs(args.out, exist_ok=True)

    # ── Load SteerViT teacher ──────────────────────────────────────────────────
    print(f"SteerViT ロード中: {args.steervit_ckpt}")
    _install_steervit()
    from steervit.model import SteerViT
    sv_teacher = SteerViT.from_pretrained(args.steervit_ckpt, device=DEVICE)
    sv_teacher.eval()
    for param in sv_teacher.parameters():
        param.requires_grad_(False)

    num_prefix_t = sv_teacher.vision_model.trunk.num_prefix_tokens
    T            = STEERVIT_GRID ** 2
    print(f"  num_prefix_tokens={num_prefix_t}  teacher_grid={STEERVIT_GRID}  T={T}")

    # ── Build dataset ──────────────────────────────────────────────────────────
    print(f"\nデータセット構築中: {args.dataset} / {args.split}")
    dataset = _build_dataset(args)
    N = len(dataset)
    print(f"  サンプル数: {N:,}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=_collate,
        drop_last=False,
    )
    n_batches = len(loader)

    # ── Probe first batch to discover n_blocks / ca_valid ─────────────────────
    if not args.no_attn:
        print("\nブロック数・CA 構成を確認中 (first batch probe)...")
        first_imgs, _, first_texts = next(iter(loader))
        with torch.no_grad():
            _, sa_probe, ca_probe = _teacher_forward_collect(
                sv_teacher, first_imgs[:1].to(DEVICE), first_texts[:1],
                STEERVIT_IMG_SIZE, num_prefix_t,
            )
        n_blocks = len(sa_probe)
        ca_valid = [c is not None for c in ca_probe]
        print(f"  n_blocks={n_blocks}  CA ありブロック={sum(ca_valid)}/{n_blocks}")
    else:
        n_blocks = 0
        ca_valid = []
        print("  --no-attn: SA/CA をスキップ")

    # ── Allocate memmaps ───────────────────────────────────────────────────────
    print("\nmemmap 確保中...")
    soft_path = os.path.join(args.out, "soft.dat")
    soft_mm   = np.memmap(soft_path, dtype="float16", mode="w+", shape=(N, T))
    print(f"  soft.dat : {N} × {T}  ({soft_mm.nbytes / 1e6:.1f} MB)")

    sa_mm = ca_mm = None
    sa_path = ca_path = None
    if not args.no_attn:
        sa_path = os.path.join(args.out, "sa.dat")
        ca_path = os.path.join(args.out, "ca.dat")
        sa_mm   = np.memmap(sa_path, dtype="float16", mode="w+", shape=(N, n_blocks, T))
        ca_mm   = np.memmap(ca_path, dtype="float16", mode="w+", shape=(N, n_blocks, T))
        print(f"  sa.dat   : {N} × {n_blocks} × {T}  ({sa_mm.nbytes / 1e6:.1f} MB)")
        print(f"  ca.dat   : {N} × {n_blocks} × {T}  ({ca_mm.nbytes / 1e6:.1f} MB)")

    # ── Main loop ──────────────────────────────────────────────────────────────
    print(f"\n計算中... (全 {n_batches} バッチ、{args.print_every} バッチごとに表示)")
    offset    = 0
    batch_idx = 0
    with torch.no_grad():
        for full_imgs, _bboxes, texts in loader:
            B_batch   = full_imgs.shape[0]
            full_imgs = full_imgs.to(DEVICE)
            end       = offset + B_batch

            if args.no_attn:
                teacher_imgs = F.interpolate(
                    full_imgs,
                    size=(STEERVIT_IMG_SIZE, STEERVIT_IMG_SIZE),
                    mode="bicubic", align_corners=False,
                )
                tf     = sv_teacher.forward(teacher_imgs, list(texts))
                tp     = tf[:, num_prefix_t:, :]
                logits = sv_teacher.lin_seg_head(tp).squeeze(-1)
                soft   = torch.softmax(logits, dim=-1)
                soft_mm[offset:end] = soft.cpu().to(torch.float16).numpy()
            else:
                soft, sa_maps, ca_maps = _teacher_forward_collect(
                    sv_teacher, full_imgs, list(texts),
                    STEERVIT_IMG_SIZE, num_prefix_t,
                )
                soft_mm[offset:end] = soft.cpu().to(torch.float16).numpy()
                for b_idx in range(n_blocks):
                    sa = sa_maps[b_idx]
                    ca = ca_maps[b_idx]
                    if sa is not None:
                        sa_mm[offset:end, b_idx, :] = _compress_sa(sa, num_prefix_t)
                    if ca is not None and ca_valid[b_idx]:
                        ca_mm[offset:end, b_idx, :] = _compress_ca(ca, num_prefix_t)

            offset    = end
            batch_idx += 1
            if batch_idx % args.print_every == 0 or batch_idx == n_batches:
                print(
                    f"  [{batch_idx:4d}/{n_batches}]  samples={offset:,}/{N:,}",
                    flush=True,
                )

    # Flush to disk
    soft_mm.flush()
    if not args.no_attn:
        sa_mm.flush()
        ca_mm.flush()

    # ── Write meta.json ────────────────────────────────────────────────────────
    meta = {
        "n_samples":          N,
        "teacher_grid":       STEERVIT_GRID,
        "n_blocks":           n_blocks,
        "ca_valid":           ca_valid,
        "teacher_num_prefix": num_prefix_t,
        "has_attn":           not args.no_attn,
        "dataset":            args.dataset,
        "split":              args.split,
        "steervit_ckpt":      args.steervit_ckpt,
    }
    meta_path = os.path.join(args.out, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # ── Summary ────────────────────────────────────────────────────────────────
    total_mb = os.path.getsize(soft_path) / 1e6
    print(f"\n完了: {args.out}")
    print(f"  soft.dat : {total_mb:.1f} MB")
    if not args.no_attn:
        sa_mb     = os.path.getsize(sa_path) / 1e6
        ca_mb     = os.path.getsize(ca_path) / 1e6
        total_mb += sa_mb + ca_mb
        print(f"  sa.dat   : {sa_mb:.1f} MB")
        print(f"  ca.dat   : {ca_mb:.1f} MB")
    print(f"  合計     : {total_mb:.1f} MB")
    print(f"  meta.json:\n{json.dumps(meta, indent=4, ensure_ascii=False)}")

    # ── S3 アップロード ────────────────────────────────────────────────────────
    if args.s3_output:
        print(f"\nS3 アップロード中: {args.s3_output}")
        _s3_upload_dir(args.out, args.s3_output)


if __name__ == "__main__":
    main()
