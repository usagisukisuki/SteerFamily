"""
test.py - CORE Benchmark Evaluation: TinySteerViT vs SteerViT

CORE (COnditional REtrieval) ベンチマーク (SteerViT 論文 arXiv:2604.02327) の
代替評価として RefCOCO val/testA/testB を使用する。

評価指標:
  PA   (Pointing Accuracy)     : argmax heatmap patch の中心が GT bbox 内 → hit
  AHP  (Avg Heatmap Precision) : GT bbox 内の heatmap 質量の割合
  PA_w (Wrong-prompt PA)       : テキストをシャッフルしたときの PA (↓ が大きいほどテキスト依存)

モデル:
  TinySteerViT : 学習済みチェックポイント (14×14 grid, 224px)
  SteerViT     : オリジナルモデル          (24×24 grid, 336px) [--no-steervit で省略可]

使用例:
  python test.py --ckpt ./ckpts_tinyvit_all/best_model.pth
  python test.py --ckpt ./ckpts_tinyvit_all/best_model.pth --split val
  python test.py --ckpt ./ckpts_tinyvit_all/best_model.pth --no-steervit
  python test.py --ckpt ./ckpts_tinyvit_all/best_model.pth --max-samples 500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from dataset import (
    RefCOCOLocalDataset, VGRegionDataset, GQASceneGraphDataset,
    REFCOCO_DIR, COCO_ROOT, VG_ROOT, GQA_ROOT,
)
from model import (
    TinySteerViT,
    TEXT_ENCODERS,
    STEERVIT_CKPT,
    STEERVIT_GRID,
    STEERVIT_IMG_SIZE,
    TINY_VIT_GRID,
    TINY_VIT_IMG_SIZE,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_MEAN = (0.485, 0.456, 0.406)
_STD  = (0.229, 0.224, 0.225)


# ─── Metrics ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def pointing_accuracy(
    heatmap_flat: torch.Tensor,   # (B, grid*grid)
    bbox_rel:     torch.Tensor,   # (B, 4) [x1,y1,x2,y2] in [0,1]
    grid:         int,
) -> torch.Tensor:
    """argmax patch の中心が GT bbox 内なら hit → (B,) bool."""
    argmax_idx = heatmap_flat.argmax(dim=-1)           # (B,)
    row = (argmax_idx // grid).float()
    col = (argmax_idx  % grid).float()
    cx  = (col + 0.5) / grid
    cy  = (row + 0.5) / grid
    x1, y1, x2, y2 = bbox_rel.unbind(dim=1)
    return (cx >= x1) & (cx <= x2) & (cy >= y1) & (cy <= y2)


@torch.no_grad()
def avg_heatmap_precision(
    heatmap_flat: torch.Tensor,   # (B, grid*grid) probabilities (sum=1)
    bbox_rel:     torch.Tensor,   # (B, 4) in [0,1]
    grid:         int,
) -> torch.Tensor:
    """GT bbox 内の heatmap 質量の割合 → (B,) float [0,1]."""
    B      = heatmap_flat.shape[0]
    g      = grid
    device = heatmap_flat.device

    t   = torch.arange(g, device=device, dtype=torch.float32)
    px1 = (t / g).view(1, 1, g)
    px2 = ((t + 1) / g).view(1, 1, g)
    py1 = (t / g).view(1, g, 1)
    py2 = ((t + 1) / g).view(1, g, 1)

    bx1 = bbox_rel[:, 0].view(B, 1, 1)
    by1 = bbox_rel[:, 1].view(B, 1, 1)
    bx2 = bbox_rel[:, 2].view(B, 1, 1)
    by2 = bbox_rel[:, 3].view(B, 1, 1)

    overlap_x = (torch.min(bx2, px2) - torch.max(bx1, px1)).clamp(min=0)
    overlap_y = (torch.min(by2, py2) - torch.max(by1, py1)).clamp(min=0)
    in_bbox   = ((overlap_x * overlap_y) > 0).reshape(B, g * g).float()

    return (heatmap_flat * in_bbox).sum(dim=-1)


# ─── Heatmap extraction ───────────────────────────────────────────────────────

@torch.no_grad()
def tiny_heatmap(sv: TinySteerViT, imgs: torch.Tensor, texts: list[str]) -> torch.Tensor:
    """TinySteerViT → softmax heatmap (B, 196)."""
    feats  = sv.forward(imgs, texts)
    patch  = feats[:, sv.vision_model.trunk.num_prefix_tokens:, :]
    logits = sv.lin_seg_head(patch).squeeze(-1)
    return F.softmax(logits, dim=-1)


@torch.no_grad()
def steervit_heatmap(sv_teacher, imgs: torch.Tensor, texts: list[str]) -> torch.Tensor:
    """SteerViT → softmax heatmap (B, 576) using 336px resized input."""
    imgs336 = F.interpolate(imgs, size=(STEERVIT_IMG_SIZE, STEERVIT_IMG_SIZE),
                            mode="bicubic", align_corners=False)
    feats   = sv_teacher.forward(imgs336, texts)
    patch   = feats[:, sv_teacher.vision_model.trunk.num_prefix_tokens:, :]
    logits  = sv_teacher.lin_seg_head(patch).squeeze(-1)
    return F.softmax(logits, dim=-1)


# ─── Single-model evaluation ──────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model,
    heatmap_fn,
    grid: int,
    loader: DataLoader,
    wrong_prompt: bool = True,
    desc: str = "",
) -> dict:
    """
    Returns:
        pa      : Pointing Accuracy  [0,1]
        ahp     : Avg Heatmap Precision [0,1]
        pa_wrong: Wrong-prompt PA (None if wrong_prompt=False)
    """
    model.eval()

    pa_hits, ahp_vals, n_total = [], [], 0

    pa_w_hits = [] if wrong_prompt else None

    for imgs, bbox_rels, texts in tqdm(loader, desc=desc, leave=False, dynamic_ncols=True):
        imgs      = imgs.to(DEVICE, non_blocking=True)
        bbox_rels = bbox_rels.to(DEVICE, non_blocking=True)
        B         = imgs.shape[0]

        # Correct prompts
        heat = heatmap_fn(imgs, list(texts))                 # (B, N)
        pa_hits.append(pointing_accuracy(heat, bbox_rels, grid))
        ahp_vals.append(avg_heatmap_precision(heat, bbox_rels, grid))

        # Wrong prompts: shift text by 1 within the batch
        if wrong_prompt and B > 1:
            shifted = list(texts[1:]) + [texts[0]]
            heat_w  = heatmap_fn(imgs, shifted)
            pa_w_hits.append(pointing_accuracy(heat_w, bbox_rels, grid))

        n_total += B

    pa  = torch.cat(pa_hits).float().mean().item()
    ahp = torch.cat(ahp_vals).float().mean().item()
    pa_w = None
    if wrong_prompt and pa_w_hits:
        pa_w = torch.cat(pa_w_hits).float().mean().item()

    return {"pa": pa, "ahp": ahp, "pa_wrong": pa_w, "n": n_total}


# ─── Model Statistics ─────────────────────────────────────────────────────────

def _human(n: int) -> str:
    if n >= 1e9: return f"{n/1e9:.2f} G"
    if n >= 1e6: return f"{n/1e6:.2f} M"
    if n >= 1e3: return f"{n/1e3:.2f} K"
    return str(int(n))


def _collect_raw_flops(model, forward_fn) -> dict[str, int]:
    """フックで各 Linear/Conv2d/LayerNorm/MHA の FLOPs を収集する。"""
    mha_out_proj = {
        f"{n}.out_proj"
        for n, m in model.named_modules()
        if isinstance(m, nn.MultiheadAttention)
    }
    flop_dict: dict[str, int] = {}
    handles = []

    def make_hook(name):
        def hook(module, inp, out):
            flops = 0
            if isinstance(module, nn.Linear):
                if name in mha_out_proj:
                    return  # MHA hook 側でカウント済み
                n_elem = inp[0].numel() // inp[0].shape[-1]
                flops = 2 * n_elem * module.in_features * module.out_features
            elif isinstance(module, nn.Conv2d):
                B_, C_out, H_, W_ = out.shape; kh, kw = module.kernel_size
                flops = (2 * B_ * C_out * H_ * W_
                         * (module.in_channels // module.groups) * kh * kw)
            elif isinstance(module, nn.LayerNorm):
                flops = 5 * inp[0].numel()
            elif isinstance(module, nn.MultiheadAttention):
                q, k = inp[0], inp[1]
                if q.dim() == 3: B_, Nq, E = q.shape; Nk = k.shape[1]
                else:            Nq, B_, E = q.shape; Nk = k.shape[0]
                hd = E // module.num_heads
                flops += (2*B_*Nq*E*E + 2*B_*Nk*E*E*2
                          + 4*B_*module.num_heads*Nq*Nk*hd + 2*B_*Nq*E*E)
            if flops > 0:
                flop_dict[name] = flop_dict.get(name, 0) + flops
        return hook

    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Linear, nn.Conv2d, nn.LayerNorm, nn.MultiheadAttention)):
            handles.append(mod.register_forward_hook(make_hook(name)))
    was_training = model.training
    model.eval()
    with torch.no_grad():
        forward_fn()
    if was_training:
        model.train()
    for h in handles:
        h.remove()
    return flop_dict


def _to_components(flop_dict: dict[str, int]) -> dict[str, int]:
    """Raw flop_dict をコンポーネント別に集約する。"""
    comps: dict[str, int] = {
        "text model": 0, "ViT backbone": 0, "GCA": 0, "connector": 0, "lin_seg_head": 0,
    }
    for name, f in flop_dict.items():
        if "text_model"         in name: comps["text model"]   += f
        elif "gated_cross_attn" in name: comps["GCA"]          += f
        elif "vision_model"     in name: comps["ViT backbone"] += f
        elif "connector"        in name: comps["connector"]     += f
        elif "lin_seg_head"     in name: comps["lin_seg_head"]  += f
    return comps


def _vit_self_attn_est(model, B: int = 1) -> int:
    """ViT 自己注意スコアの推定 FLOPs (QK^T + A@V, フック計測外)。"""
    trunk     = model.vision_model.trunk
    N         = trunk.num_prefix_tokens + model.num_img_tokens
    try:   num_heads = trunk.blocks[0].attn.num_heads
    except AttributeError: num_heads = 12
    head_dim  = trunk.embed_dim // num_heads
    n_blocks  = len(trunk.blocks)
    return 4 * B * num_heads * N * N * head_dim * n_blocks


def _save_comparison_chart(
    tiny_comps: dict[str, int],
    tiny_attn:  int,
    sv_comps:   dict[str, int] | None,
    sv_attn:    int,
    save_path:  str = "flops_chart.png",
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    CATS = ["ViT backbone", "ViT self-attn\n(est.)", "GCA", "text model",
            "connector", "lin_seg_head"]

    def _vals(comps, attn):
        return [
            comps.get("ViT backbone", 0),
            attn,
            comps.get("GCA", 0),
            comps.get("text model", 0),
            comps.get("connector", 0),
            comps.get("lin_seg_head", 0),
        ]

    tiny_vals = _vals(tiny_comps, tiny_attn)
    x         = np.arange(len(CATS))
    has_sv    = sv_comps is not None
    width     = 0.35 if has_sv else 0.5

    fig, ax = plt.subplots(figsize=(11, 6))

    def _fmt(v):
        if v >= 1e9: return f"{v/1e9:.1f}G"
        if v >= 1e6: return f"{v/1e6:.0f}M"
        return f"{v/1e3:.0f}K"

    def _draw_bars(xs, vals, color, hatch_idx, label):
        bars = ax.bar(xs, vals, width, label=label, color=color,
                      alpha=0.9, edgecolor="white", linewidth=0.5, zorder=3)
        bars[hatch_idx].set_hatch("///")
        bars[hatch_idx].set_alpha(0.65)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() * 1.15,
                        _fmt(v), ha="center", va="bottom",
                        fontsize=8, fontweight="bold", color="#333333")
        return bars

    # SteerViT 左、TinySteerViT 右
    if has_sv:
        sv_vals = _vals(sv_comps, sv_attn)
        _draw_bars(x - width / 2, sv_vals, "#ED7D31", hatch_idx=1, label="SteerViT")

    _draw_bars(x + width/2 if has_sv else x,
               tiny_vals, "#4472C4", hatch_idx=1, label="TinySteerViT")

    # 倍率アノテーション (SteerViT / TinySteerViT)
    if has_sv:
        for xi, (sv_v, tv) in enumerate(zip(sv_vals, tiny_vals)):
            if tv > 0 and sv_v > 0:
                ratio = sv_v / tv
                ypos  = max(tv, sv_v) * 3.5
                ax.text(xi, ypos, f"×{ratio:.0f}",
                        ha="center", va="bottom", fontsize=8.5,
                        color="#555555", style="italic")

    ax.set_yscale("log")
    ax.set_ylim(bottom=1e4)
    ax.set_ylabel("FLOPs  (log scale)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(CATS, fontsize=10)
    ax.set_title(
        "FLOPs Comparison: TinySteerViT vs SteerViT\n"
        "(B=1, text_seq=20.  Linear/Conv/Norm measured.  ViT self-attn estimated.)",
        fontsize=11, pad=10,
    )
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: _fmt(int(v))))
    ax.grid(axis="y", alpha=0.35, which="both", linewidth=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=10, framealpha=0.85)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → FLOPs chart saved: {save_path}")


def _save_param_chart(
    tiny_params: dict[str, tuple[int, int]],   # {cat: (total, trainable)}
    sv_params:   dict[str, tuple[int, int]] | None,
    save_path:   str = "params_chart.png",
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    CATS   = ["text model", "ViT backbone", "GCA", "connector", "lin_seg_head"]
    has_sv = sv_params is not None
    width  = 0.35 if has_sv else 0.5
    x      = np.arange(len(CATS))

    def _fmt(v):
        if v >= 1e9: return f"{v/1e9:.1f}G"
        if v >= 1e6: return f"{v/1e6:.0f}M"
        return f"{v/1e3:.0f}K"

    def _draw(xs, params, base_color, frozen_color, label_prefix):
        totals     = [params.get(c, (0, 0))[0] for c in CATS]
        trainables = [params.get(c, (0, 0))[1] for c in CATS]
        frozens    = [t - tr for t, tr in zip(totals, trainables)]

        # frozen portion (bottom)
        bars_f = ax.bar(xs, frozens, width, color=frozen_color,
                        alpha=0.7, edgecolor="white", linewidth=0.4, zorder=3,
                        label=f"{label_prefix} frozen")
        # trainable portion (top, stacked)
        bars_t = ax.bar(xs, trainables, width, bottom=frozens,
                        color=base_color, alpha=0.95,
                        edgecolor="white", linewidth=0.4, zorder=3,
                        label=f"{label_prefix} trainable")
        # value labels
        for xs_i, total in zip(xs, totals):
            if total > 0:
                ax.text(xs_i, total * 1.15, _fmt(total),
                        ha="center", va="bottom",
                        fontsize=7.5, fontweight="bold", color="#333333")

    fig, ax = plt.subplots(figsize=(11, 6))

    if has_sv:
        _draw(x - width / 2, sv_params,   "#ED7D31", "#FCDAB4", "SteerViT")
    _draw(x + width / 2 if has_sv else x,
          tiny_params, "#4472C4", "#BDD7EE", "TinySteerViT")

    # 倍率アノテーション
    if has_sv:
        for xi, cat in enumerate(CATS):
            tv = tiny_params.get(cat, (0, 0))[0]
            sv_v = sv_params.get(cat, (0, 0))[0]
            if tv > 0 and sv_v > 0:
                ratio = sv_v / tv
                ypos  = max(tv, sv_v) * 3.5
                ax.text(xi, ypos, f"×{ratio:.0f}",
                        ha="center", va="bottom", fontsize=8.5,
                        color="#555555", style="italic")

    ax.set_yscale("log")
    ax.set_ylim(bottom=100)
    ax.set_ylabel("Parameters  (log scale)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(CATS, fontsize=10)
    ax.set_title(
        "Parameter Count Comparison: TinySteerViT vs SteerViT\n"
        "(dark = trainable,  light = frozen)",
        fontsize=11, pad=10,
    )
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: _fmt(int(v))))
    ax.grid(axis="y", alpha=0.35, which="both", linewidth=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9, framealpha=0.85, ncol=2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Params chart saved: {save_path}")


def _print_param_table(title: str, groups: list[tuple[str, nn.Module]]) -> None:
    print("\n" + "=" * 74)
    print(f"  {title} — パラメータ数")
    print("=" * 74)
    print(f"  {'Component':<46}  {'Total':>9}  {'Trainable':>9}  {'Frozen':>9}")
    print("-" * 74)
    total_p = total_t = 0
    for label, mod in groups:
        n = sum(p.numel() for p in mod.parameters())
        t = sum(p.numel() for p in mod.parameters() if p.requires_grad)
        total_p += n; total_t += t
        print(f"  {label:<46}  {n:>9,}  {t:>9,}  {n-t:>9,}")
    print("-" * 74)
    print(f"  {'Total':<46}  {total_p:>9,}  {total_t:>9,}  {total_p-total_t:>9,}")


def print_model_stats(sv, sv_teacher=None) -> None:
    """パラメータ数と FLOPs を表示し、比較チャートを保存する (B=1, L=20)。"""
    B_d, L_d = 1, 20
    enc_name = getattr(sv, "text_encoder_name", "distilroberta")
    d_text   = TEXT_ENCODERS[enc_name]["d_text"] if enc_name in TEXT_ENCODERS else 768

    # ── TinySteerViT パラメータ表 ────────────────────────────────────────
    _print_param_table(f"TinySteerViT ({enc_name})", [
        (f"text_model ({enc_name}, frozen)",         sv.text_model),
        (f"connector  (Linear {d_text}→192)",         sv.connector),
        ("vision_model.trunk (ViT-Tiny, frozen)",   sv.vision_model.trunk),
        ("gated_cross_attn  (12 × GatedCrossAttn)", sv.vision_model.gated_cross_attn),
        ("lin_seg_head (Linear 192→1)",              sv.lin_seg_head),
    ])

    # ── SteerViT パラメータ表 ────────────────────────────────────────────
    if sv_teacher is not None:
        trunk_sv  = sv_teacher.vision_model.trunk
        gca_mods  = nn.ModuleList(
            b.gated_cross_attn
            for b in trunk_sv.blocks
            if getattr(b, "gated_cross_attn", None) is not None
        )
        # param counts computed directly (no proxy needed)
        n_gca  = sum(p.numel() for m in gca_mods for p in m.parameters())
        n_t_gca = sum(p.numel() for m in gca_mods for p in m.parameters() if p.requires_grad)
        n_trunk = sum(p.numel() for p in trunk_sv.parameters()) - n_gca
        n_t_trunk = sum(
            p.numel() for n, p in trunk_sv.named_parameters()
            if "gated_cross_attn" not in n and p.requires_grad
        )

        # Print table manually to avoid proxy module
        print("\n" + "=" * 74)
        print("  SteerViT — パラメータ数")
        print("=" * 74)
        print(f"  {'Component':<46}  {'Total':>9}  {'Trainable':>9}  {'Frozen':>9}")
        print("-" * 74)
        rows = [
            ("text_model (RoBERTa-large, frozen)",     sv_teacher.text_model,   False),
            ("connector  (MLP 1024→1024→768)",         sv_teacher.connector,    False),
            ("vision_model.trunk (ViT-B/14, w/o GCA)", None,                   False),
            ("gated_cross_attn  (6 × GCA in blocks)",  None,                   False),
            ("lin_seg_head (Linear 768→1)",             sv_teacher.lin_seg_head, False),
        ]
        manual = [(None, n_trunk, n_t_trunk), (None, n_gca, n_t_gca)]
        total_p = total_t = 0
        mi = 0
        for label, mod, _ in rows:
            if mod is not None:
                n = sum(p.numel() for p in mod.parameters())
                t = sum(p.numel() for p in mod.parameters() if p.requires_grad)
            else:
                _, n, t = manual[mi]; mi += 1
            total_p += n; total_t += t
            print(f"  {label:<46}  {n:>9,}  {t:>9,}  {n-t:>9,}")
        print("-" * 74)
        print(f"  {'Total':<46}  {total_p:>9,}  {total_t:>9,}  {total_p-total_t:>9,}")

    # ── TinySteerViT FLOPs ────────────────────────────────────────────────
    device_tiny  = next(sv.parameters()).device
    dummy_imgs   = torch.zeros(B_d, 3, TINY_VIT_IMG_SIZE, TINY_VIT_IMG_SIZE,
                               device=device_tiny)
    dummy_txt    = torch.zeros(B_d, L_d, d_text, device=device_tiny)
    dummy_ids_t  = torch.ones(B_d, L_d, dtype=torch.long, device=device_tiny)
    dummy_mask_t = torch.ones(B_d, L_d, dtype=torch.long, device=device_tiny)

    def tiny_fwd():
        sv.text_model(input_ids=dummy_ids_t, attention_mask=dummy_mask_t)
        text_proj = sv.connector(dummy_txt)
        vis_out   = sv.vision_model(dummy_imgs, text_proj)
        patch     = vis_out[:, sv.vision_model.trunk.num_prefix_tokens:, :]
        sv.lin_seg_head(patch)

    tiny_raw   = _collect_raw_flops(sv, tiny_fwd)
    tiny_comps = _to_components(tiny_raw)
    tiny_attn  = _vit_self_attn_est(sv, B_d)

    print(f"\n  TinySteerViT FLOPs  (B={B_d}, img={TINY_VIT_IMG_SIZE}px, "
          f"text_seq={L_d})")
    print(f"  ※ Linear/Conv/Norm を計測。ViT self-attn は推定値。")
    print("-" * 55)
    print(f"  {'Component':<32}  {'FLOPs':>10}  {'%':>5}")
    print("-" * 55)
    total_tiny = sum(tiny_comps.values()) + tiny_attn
    for cat in ["ViT backbone", "GCA", "ViT self-attn (est.)", "text model", "connector", "lin_seg_head"]:
        v = tiny_attn if cat == "ViT self-attn (est.)" else tiny_comps.get(cat, 0)
        print(f"  {cat:<32}  {_human(v):>10}  {100*v/total_tiny:>4.1f}%")
    print("-" * 55)
    print(f"  {'Grand Total (est.)':<32}  {_human(total_tiny):>10}")

    # ── SteerViT FLOPs ────────────────────────────────────────────────────
    sv_comps = sv_attn = None
    if sv_teacher is not None:
        device_sv       = next(sv_teacher.parameters()).device
        D_ROBERTA_LARGE = 1024
        dummy_imgs_sv   = torch.zeros(B_d, 3, STEERVIT_IMG_SIZE, STEERVIT_IMG_SIZE,
                                      device=device_sv)
        dummy_roberta   = torch.zeros(B_d, L_d, D_ROBERTA_LARGE, device=device_sv)
        dummy_ids_sv    = torch.ones(B_d, L_d, dtype=torch.long, device=device_sv)
        dummy_mask_sv   = torch.ones(B_d, L_d, dtype=torch.long, device=device_sv)

        def sv_fwd():
            sv_teacher.text_model(input_ids=dummy_ids_sv, attention_mask=dummy_mask_sv)
            text_proj = sv_teacher.connector(dummy_roberta)
            vis_out   = sv_teacher.vision_model(dummy_imgs_sv, text_proj)
            patch     = vis_out[:, sv_teacher.vision_model.trunk.num_prefix_tokens:, :]
            sv_teacher.lin_seg_head(patch)

        sv_raw   = _collect_raw_flops(sv_teacher, sv_fwd)
        sv_comps = _to_components(sv_raw)
        sv_attn  = _vit_self_attn_est(sv_teacher, B_d)

        print(f"\n  SteerViT FLOPs  (B={B_d}, img={STEERVIT_IMG_SIZE}px, "
              f"text_seq={L_d})")
        print(f"  ※ Linear/Conv/Norm を計測。ViT self-attn は推定値。")
        print("-" * 55)
        print(f"  {'Component':<32}  {'FLOPs':>10}  {'%':>5}")
        print("-" * 55)
        total_sv = sum(sv_comps.values()) + sv_attn
        for cat in ["ViT backbone", "GCA", "ViT self-attn (est.)", "text model", "connector", "lin_seg_head"]:
            v = sv_attn if cat == "ViT self-attn (est.)" else sv_comps.get(cat, 0)
            print(f"  {cat:<32}  {_human(v):>10}  {100*v/total_sv:>4.1f}%")
        print("-" * 55)
        print(f"  {'Grand Total (est.)':<32}  {_human(total_sv):>10}")

    # ── Param dicts for chart (trainable based on architecture design) ────
    def _count(mod):
        return sum(p.numel() for p in mod.parameters())

    n_tiny_text  = _count(sv.text_model)
    n_tiny_trunk = _count(sv.vision_model.trunk)
    n_tiny_gca   = _count(sv.vision_model.gated_cross_attn)
    n_tiny_conn  = _count(sv.connector)
    n_tiny_head  = _count(sv.lin_seg_head)

    tiny_params: dict[str, tuple[int, int]] = {
        "text model":   (n_tiny_text,  0),
        "ViT backbone": (n_tiny_trunk, 0),
        "GCA":          (n_tiny_gca,   n_tiny_gca),
        "connector":    (n_tiny_conn,  n_tiny_conn),
        "lin_seg_head": (n_tiny_head,  n_tiny_head),
    }

    sv_params_chart: dict[str, tuple[int, int]] | None = None
    if sv_teacher is not None:
        n_sv_text = _count(sv_teacher.text_model)
        n_sv_conn = _count(sv_teacher.connector)
        n_sv_head = _count(sv_teacher.lin_seg_head)
        sv_params_chart = {
            "text model":   (n_sv_text, 0),
            "ViT backbone": (n_trunk,   0),
            "GCA":          (n_gca,     n_gca),
            "connector":    (n_sv_conn, n_sv_conn),
            "lin_seg_head": (n_sv_head, n_sv_head),
        }

    print()
    _save_comparison_chart(tiny_comps, tiny_attn, sv_comps, sv_attn or 0)
    _save_param_chart(tiny_params, sv_params_chart, save_path="params_chart.png")


# ─── Load TinySteerViT ────────────────────────────────────────────────────────

def load_tiny(ckpt_path: str) -> TinySteerViT:
    ckpt         = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    train_args   = ckpt.get("args", {})
    text_encoder = train_args.get("text_encoder", "distilroberta")
    sv = TinySteerViT(
        pretrained_backbone=True,
        text_encoder=text_encoder,
    ).to(DEVICE)
    sd = sv.state_dict()
    sd.update(ckpt["sv_state"])
    sv.load_state_dict(sd)
    sv.eval()
    for p in sv.parameters():
        p.requires_grad_(False)
    return sv


# ─── Config loader ────────────────────────────────────────────────────────────

def _load_yaml(path: str) -> dict[str, Any]:
    try:
        import yaml
    except ImportError:
        raise SystemExit("PyYAML が必要です: pip install pyyaml")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _config_defaults(config_path: str | None) -> dict[str, Any]:
    """config.yaml から test セクション + paths セクションをフラットに返す。"""
    if not config_path:
        return {}
    cfg = _load_yaml(config_path)
    defaults: dict[str, Any] = {}
    defaults.update(cfg.get("paths", {}))
    defaults.update(cfg.get("test", {}))
    return {k: v for k, v in defaults.items() if v is not None}


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    # 1st pass: --config のみ先読み
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    pre_args, _ = pre.parse_known_args()

    cfg_defaults = _config_defaults(pre_args.config)

    p = argparse.ArgumentParser(description="CORE benchmark evaluation")
    p.add_argument("--config",      type=str, default=None,
                   help="YAML config ファイル (CLI 引数で上書き可)")
    # ── Paths ──────────────────────────────────────────────────────────────
    p.add_argument("--coco-root",     type=str, default="/data2/kato/dataset/coco")
    p.add_argument("--vg-root",       type=str, default="/data2/kato/dataset/vg")
    p.add_argument("--gqa-root",      type=str, default="/data2/kato/dataset/GQA")
    p.add_argument("--steervit-ckpt", type=str, default=STEERVIT_CKPT)
    # ── Evaluation ─────────────────────────────────────────────────────────
    p.add_argument("--ckpt",        type=str, default=None,
                   help="TinySteerViT チェックポイント (.pth)")
    p.add_argument("--split",       type=str, default=None,
                   choices=["val", "testA", "testB"],
                   help="評価 split (省略時はデータセットのデフォルト全 split)")
    p.add_argument("--dataset",     type=str, default="refcoco",
                   choices=["refcoco", "refcoco+", "refcocog", "vg", "gqa"],
                   help="評価データセット (デフォルト: refcoco)")
    p.add_argument("--max-samples", type=int, default=None,
                   help="各 split の最大サンプル数 (デフォルト: 全件)")
    p.add_argument("--batch-size",  type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-steervit", action="store_true",
                   help="SteerViT teacher を評価しない")
    p.add_argument("--no-wrong-prompt", action="store_true",
                   help="Wrong-prompt 制御実験をスキップ")

    p.set_defaults(**cfg_defaults)
    args = p.parse_args()

    if args.ckpt is None:
        p.error("--ckpt が必要です (または config.yaml の test.ckpt を設定してください)")

    return args


def _splits_for(dataset: str) -> list[str]:
    if dataset in ("refcocog", "vg", "gqa"):
        return ["val"]
    return ["val", "testA", "testB"]


def _make_loader(dataset: str, split: str, img_size: int,
                 max_samples: int | None, batch_size: int, num_workers: int,
                 coco_root: str = COCO_ROOT, vg_root: str = VG_ROOT,
                 gqa_root: str = GQA_ROOT):
    import os

    if dataset in ("refcoco", "refcoco+", "refcocog"):
        refcoco_dir = os.path.join(coco_root, dataset)
        ds = RefCOCOLocalDataset(
            refcoco_dir   = refcoco_dir,
            coco_root     = coco_root,
            split         = split,
            full_img_size = img_size,
            max_samples   = max_samples,
            seed          = 0,
        )
    elif dataset == "vg":
        ds = VGRegionDataset(
            vg_root       = vg_root,
            split         = split,
            full_img_size = img_size,
            max_samples   = max_samples,
            seed          = 0,
        )
    elif dataset == "gqa":
        ds = GQASceneGraphDataset(
            gqa_root      = gqa_root,
            split         = split,
            full_img_size = img_size,
            max_samples   = max_samples,
            seed          = 0,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    def collate(batch):
        imgs, bboxes, texts = zip(*batch)
        return torch.stack(imgs), torch.stack(bboxes), list(texts)

    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True, collate_fn=collate)


def _fmt(v: float | None) -> str:
    return f"{v * 100:.2f}%" if v is not None else "  N/A  "


def main():
    args   = parse_args()
    splits = [args.split] if args.split else _splits_for(args.dataset)
    wrong  = not args.no_wrong_prompt

    print(f"\n{'=' * 65}")
    print(f"  CORE Benchmark Evaluation")
    print(f"  dataset : {args.dataset}")
    print(f"  splits  : {splits}")
    print(f"  ckpt    : {args.ckpt}")
    print(f"  device  : {DEVICE}")
    print(f"{'=' * 65}\n")

    # ── Load TinySteerViT ─────────────────────────────────────────────────
    print("TinySteerViT ロード中...")
    tiny = load_tiny(args.ckpt)
    ckpt_info = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    print(f"  epoch={ckpt_info.get('epoch')}  val_loss={ckpt_info.get('val_loss', '?'):.4f}")
    print(f"  train args: {ckpt_info.get('args', {})}")

    # ── Load SteerViT ─────────────────────────────────────────────────────
    sv_teacher = None
    if not args.no_steervit:
        print("\nSteerViT teacher ロード中...")
        from steervit.model import SteerViT
        sv_teacher = SteerViT.from_pretrained(args.steervit_ckpt, device=DEVICE)
        sv_teacher.eval()

    # ── Model stats + comparison chart ────────────────────────────────────
    print_model_stats(tiny, sv_teacher=sv_teacher)

    # ── Header ────────────────────────────────────────────────────────────
    col_w = 10
    header_models = ["TinySteerViT"]
    if sv_teacher is not None:
        header_models.append("SteerViT")

    print(f"\n{'Split':<8} {'Model':<16} {'PA':>{col_w}} {'AHP':>{col_w}}", end="")
    if wrong:
        print(f" {'PA_wrong':>{col_w}} {'ΔPA':>{col_w}}", end="")
    print()
    print("-" * (8 + 16 + col_w * (4 if wrong else 2) + (4 if wrong else 2)))

    # ── Evaluation loop ───────────────────────────────────────────────────
    all_results = {}

    for split in splits:
        results = {}

        # TinySteerViT (224px)
        loader_tiny = _make_loader(
            args.dataset, split, TINY_VIT_IMG_SIZE,
            args.max_samples, args.batch_size, args.num_workers,
            coco_root=args.coco_root, vg_root=args.vg_root, gqa_root=args.gqa_root,
        )
        res = evaluate(
            tiny,
            lambda imgs, texts: tiny_heatmap(tiny, imgs, texts),
            TINY_VIT_GRID,
            loader_tiny,
            wrong_prompt = wrong,
            desc         = f"[TinySteerViT/{split}]",
        )
        results["TinySteerViT"] = res
        print(f"{split:<8} {'TinySteerViT':<16} {_fmt(res['pa']):>{col_w}} {_fmt(res['ahp']):>{col_w}}", end="")
        if wrong:
            delta = (res['pa'] - res['pa_wrong']) if res['pa_wrong'] is not None else None
            print(f" {_fmt(res['pa_wrong']):>{col_w}} {_fmt(delta):>{col_w}}", end="")
        print(f"  (n={res['n']:,})")

        # SteerViT (336px)
        if sv_teacher is not None:
            loader_sv = _make_loader(
                args.dataset, split, STEERVIT_IMG_SIZE,
                args.max_samples, args.batch_size, args.num_workers,
                coco_root=args.coco_root, vg_root=args.vg_root, gqa_root=args.gqa_root,
            )
            res_sv = evaluate(
                sv_teacher,
                lambda imgs, texts: steervit_heatmap(sv_teacher, imgs, texts),
                STEERVIT_GRID,
                loader_sv,
                wrong_prompt = wrong,
                desc         = f"[SteerViT/{split}]",
            )
            results["SteerViT"] = res_sv
            print(f"{split:<8} {'SteerViT':<16} {_fmt(res_sv['pa']):>{col_w}} {_fmt(res_sv['ahp']):>{col_w}}", end="")
            if wrong:
                delta_sv = (res_sv['pa'] - res_sv['pa_wrong']) if res_sv['pa_wrong'] is not None else None
                print(f" {_fmt(res_sv['pa_wrong']):>{col_w}} {_fmt(delta_sv):>{col_w}}", end="")
            print()

        all_results[split] = results

    # ── Summary ───────────────────────────────────────────────────────────
    if len(splits) > 1:
        print(f"\n{'─' * 55}")
        print("  Summary (mean over splits)")
        print(f"{'─' * 55}")
        for model_name in header_models:
            vals = [all_results[s][model_name] for s in splits if model_name in all_results.get(s, {})]
            if not vals:
                continue
            mean_pa  = sum(v["pa"]  for v in vals) / len(vals)
            mean_ahp = sum(v["ahp"] for v in vals) / len(vals)
            print(f"  {model_name:<16} PA={mean_pa*100:.2f}%  AHP={mean_ahp*100:.2f}%", end="")
            if wrong:
                pa_ws = [v["pa_wrong"] for v in vals if v["pa_wrong"] is not None]
                if pa_ws:
                    mean_pa_w = sum(pa_ws) / len(pa_ws)
                    mean_delta = mean_pa - mean_pa_w
                    print(f"  PA_wrong={mean_pa_w*100:.2f}%  ΔPA={mean_delta*100:.2f}%", end="")
            print()

    print()


if __name__ == "__main__":
    main()
