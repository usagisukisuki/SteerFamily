"""
model.py - TinySteerViT + HeatmapSoftCELoss

学習構成:
  Teacher (sv_teacher, 完全凍結, SteerViT pretrained):
    texts → RoBERTa 全トークン → sv_teacher → patch 分布  [勾配なし]

  Student (sv, TinySteerViT):
    texts → DistilRoBERTa (frozen) → connector (学習) → GatedCrossAttn (学習)
         → vit_tiny_patch16_224 (frozen) → lin_seg_head (学習) → patch logits

  Loss: SoftCE(student_logits, teacher_soft)
"""

from __future__ import annotations

import types

import torch
import torch.nn as nn
import torch.nn.functional as F

STEERVIT_CKPT = (
    "/home/kato/.cache/huggingface/hub/models--JonaRuthardt--SteerViT/"
    "snapshots/cdc29ddb5ddb8cfb6c0194c461eba14309b5afc7/steervit_dinov2_base.pth"
)

STEERVIT_IMG_SIZE = 336
STEERVIT_GRID     = 24   # 24×24 = 576 patches (DINOv2-base / 14px patch / 336px)

TINY_VIT_MODEL    = "vit_tiny_patch16_224"
TINY_VIT_IMG_SIZE = 224
TINY_VIT_GRID     = 14   # 14×14 = 196 patches (16px patch / 224px)
TINY_VIT_DIM      = 192  # ViT-Tiny embed_dim

FASTVIT_T8_MODEL      = "fastvit_t8.apple_dist_in1k"
FASTVIT_T8_IMG_SIZE   = 256
FASTVIT_T8_GRID       = 8    # 8×8 = 64 patches (stride 32 / 256px)
FASTVIT_T8_DIM        = 768  # final embed_dim (after final_conv)
FASTVIT_T8_EMBED_DIMS = (48, 96, 192, 384)  # per-stage dims (before final_conv)

MOBILEVIT_XXS_MODEL      = "mobilevit_xxs"
MOBILEVIT_XXS_IMG_SIZE   = 256
MOBILEVIT_XXS_GRID       = 8    # 8×8 = 64 patches (stride 32 / 256px)
MOBILEVIT_XXS_DIM        = 320  # final embed_dim (after final_conv)
MOBILEVIT_XXS_EMBED_DIMS = (16, 24, 48, 64, 80)  # per-stage dims (5 stages)

D_TEXT_DISTILROBERTA = 768   # 後方互換用

# ─── Vision encoder registry ─────────────────────────────────────────────────
VISION_ENCODERS: dict[str, dict] = {
    "vit_tiny": {
        "img_size": TINY_VIT_IMG_SIZE,
        "grid":     TINY_VIT_GRID,
        "dim":      TINY_VIT_DIM,
    },
    "fastvit_t8": {
        "img_size": FASTVIT_T8_IMG_SIZE,
        "grid":     FASTVIT_T8_GRID,
        "dim":      FASTVIT_T8_DIM,
    },
    "mobilevit_xxs": {
        "img_size": MOBILEVIT_XXS_IMG_SIZE,
        "grid":     MOBILEVIT_XXS_GRID,
        "dim":      MOBILEVIT_XXS_DIM,
    },
}

# ─── Text encoder registry ────────────────────────────────────────────────────
# key: --text-encoder の値  →  hf_id: HuggingFace model ID,  d_text: hidden_size
TEXT_ENCODERS: dict[str, dict] = {
    "distilroberta": {
        "hf_id":  "distilbert/distilroberta-base",
        "d_text": 768,
    },
    "tinybert-l6": {
        "hf_id":  "huawei-noah/TinyBERT_General_6L_768D",
        "d_text": 768,
    },
    "tinybert-l4": {
        "hf_id":  "huawei-noah/TinyBERT_General_4L_312D",
        "d_text": 312,
    },
    "minilm-l6": {
        "hf_id":  "sentence-transformers/all-MiniLM-L6-v2",
        "d_text": 384,
    },
}


# ─── Student self-attn capture ───────────────────────────────────────────────
def _student_sa_capture(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
    """Temporary patched forward for timm Attention that saves the attn map."""
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)
    attn = (q * self.scale) @ k.transpose(-2, -1)
    attn = attn.softmax(dim=-1)
    self._captured_sa = attn  # (B, H, N, N)
    attn = self.attn_drop(attn)
    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    return self.proj_drop(x)


# ─── GatedCrossAttn ──────────────────────────────────────────────────────────
class GatedCrossAttn(nn.Module):
    """Patch features (queries) attend to text features (keys/values) with tanh gate."""

    def __init__(self, d_model: int, d_kv: int, num_heads: int = 4):
        super().__init__()
        self.norm_q  = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_kv)
        self.proj_kv = nn.Linear(d_kv, d_model, bias=False)
        self.attn    = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.gate    = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x: torch.Tensor,
        kv: torch.Tensor,
        return_attn: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # x: (B, N, d_model)  kv: (B, M, d_kv)
        kv_proj = self.proj_kv(self.norm_kv(kv))
        out, attn_w = self.attn(
            self.norm_q(x), kv_proj, kv_proj,
            need_weights=return_attn,
            average_attn_weights=False,
        )
        result = x + self.gate.tanh() * out
        if return_attn:
            return result, attn_w  # attn_w: (B, H, N_q, N_k)
        return result


# ─── TinyViTBackbone ──────────────────────────────────────────────────────────
class TinyViTBackbone(nn.Module):
    """
    vit_tiny_patch16_224 with GatedCrossAttn injected after every transformer block.
    Interface: forward(images, text_feats, attn_mask=None) → (B, 197, 192)
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        import timm
        self.trunk = timm.create_model(
            TINY_VIT_MODEL, pretrained=pretrained, num_classes=0
        )
        self.trunk.num_prefix_tokens = 1
        d = self.trunk.embed_dim   # 192
        n = len(self.trunk.blocks) # 12
        self.gated_cross_attn = nn.ModuleList([
            GatedCrossAttn(d, d) for _ in range(n)
        ])

    def forward(
        self,
        x: torch.Tensor,
        text_feats: torch.Tensor,
        attn_mask=None,
        return_attn: bool = False,
    ) -> torch.Tensor | tuple:
        t = self.trunk
        x = t.patch_embed(x)
        x = t._pos_embed(x)
        x = t.patch_drop(x)
        x = t.norm_pre(x)

        sa_maps: list | None = [] if return_attn else None
        ca_maps: list | None = [] if return_attn else None

        for block, ca in zip(t.blocks, self.gated_cross_attn):
            if return_attn:
                orig_fwd = block.attn.forward
                block.attn.forward = types.MethodType(_student_sa_capture, block.attn)
                x = block(x)
                sa_maps.append(block.attn._captured_sa)  # keep in graph: grad flows through x to GCA
                block.attn.forward = orig_fwd
                x, ca_w = ca(x, text_feats, return_attn=True)
                ca_maps.append(ca_w)  # keep in graph: grad trains GCA directly
            else:
                x = block(x)
                x = ca(x, text_feats)

        x = t.norm(x)  # (B, 197, 192)
        if return_attn:
            return x, sa_maps, ca_maps
        return x


# ─── _StemStageBackbone (共通基底: FastViT / MobileViT スタイル) ──────────────
class _StemStageBackbone(nn.Module):
    """
    stem + stages + final_conv 構造を持つモデル用の共通バックボーン。
    (FastViT-T8, MobileViT-xxs など)

    各ステージの後に GatedCrossAttn を挿入し、最後に final_conv を適用する。
    num_prefix_tokens = 0 (CLS トークンなし)
    出力: (B, grid**2, final_dim)
    """

    def __init__(
        self,
        model_name:  str,
        embed_dims:  tuple[int, ...],
        final_dim:   int,
        pretrained:  bool = True,
    ):
        super().__init__()
        import timm
        self.trunk = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.trunk.num_prefix_tokens = 0
        self.trunk.embed_dim = final_dim
        # One GCA per stage; d_kv=final_dim (connector が final_dim 次元に射影)
        self.gated_cross_attn = nn.ModuleList([
            GatedCrossAttn(d_model=d, d_kv=final_dim)
            for d in embed_dims
        ])

    def forward(
        self,
        x: torch.Tensor,
        text_feats: torch.Tensor,
        attn_mask=None,
        return_attn: bool = False,
    ) -> torch.Tensor | tuple:
        t = self.trunk
        x = t.stem(x)

        ca_maps: list | None = [] if return_attn else None

        for stage, ca in zip(t.stages, self.gated_cross_attn):
            x = stage(x)                              # (B, C_i, H_i, W_i)
            B, C, H, W = x.shape
            x_flat = x.flatten(2).transpose(1, 2)    # (B, H*W, C_i)
            if return_attn:
                x_flat, ca_w = ca(x_flat, text_feats, return_attn=True)
                ca_maps.append(ca_w)  # keep in graph: grad trains GCA directly
            else:
                x_flat = ca(x_flat, text_feats)
            x = x_flat.transpose(1, 2).reshape(B, C, H, W)

        x = t.final_conv(x)                          # (B, final_dim, H', W')
        x_out = x.flatten(2).transpose(1, 2)         # (B, grid**2, final_dim)

        if return_attn:
            return x_out, [], ca_maps  # SA maps なし
        return x_out


# ─── FastViTBackbone ──────────────────────────────────────────────────────────
class FastViTBackbone(_StemStageBackbone):
    """fastvit_t8: 4 stages, output (B, 64, 768)"""

    def __init__(self, pretrained: bool = True):
        super().__init__(
            model_name = FASTVIT_T8_MODEL,
            embed_dims = FASTVIT_T8_EMBED_DIMS,
            final_dim  = FASTVIT_T8_DIM,
            pretrained = pretrained,
        )


# ─── MobileViTXXSBackbone ─────────────────────────────────────────────────────
class MobileViTXXSBackbone(_StemStageBackbone):
    """mobilevit_xxs: 5 stages, output (B, 64, 320)"""

    def __init__(self, pretrained: bool = True):
        super().__init__(
            model_name = MOBILEVIT_XXS_MODEL,
            embed_dims = MOBILEVIT_XXS_EMBED_DIMS,
            final_dim  = MOBILEVIT_XXS_DIM,
            pretrained = pretrained,
        )


# ─── TinySteerViT ─────────────────────────────────────────────────────────────
class TinySteerViT(nn.Module):
    """
    SteerViT-compatible model with pluggable vision backbone.

    学習対象: connector, gated_cross_attn, lin_seg_head。
    凍結:     vision_model.trunk, text_model

    vision_encoder: VISION_ENCODERS のキー ("vit_tiny" | "fastvit_t8")
    text_encoder:   TEXT_ENCODERS のキー (例: "distilroberta", "tinybert-l4", …)

    SteerViT と同一インターフェース:
      .vision_model.trunk.num_prefix_tokens : 1 (vit_tiny) / 0 (fastvit_t8)
      .num_img_tokens                       : 196 (vit_tiny) / 64 (fastvit_t8)
      .lin_seg_head                         : Linear(d_vis, 1)
    """

    def __init__(
        self,
        pretrained_backbone: bool = True,
        text_encoder: str = "distilroberta",
        vision_encoder: str = "vit_tiny",
    ):
        super().__init__()
        if text_encoder not in TEXT_ENCODERS:
            raise ValueError(
                f"Unknown text_encoder {text_encoder!r}. "
                f"Choose from {list(TEXT_ENCODERS)}"
            )
        if vision_encoder not in VISION_ENCODERS:
            raise ValueError(
                f"Unknown vision_encoder {vision_encoder!r}. "
                f"Choose from {list(VISION_ENCODERS)}"
            )
        enc_cfg = TEXT_ENCODERS[text_encoder]
        d_text  = enc_cfg["d_text"]
        hf_id   = enc_cfg["hf_id"]
        vis_cfg = VISION_ENCODERS[vision_encoder]

        self.text_encoder_name   = text_encoder
        self.vision_encoder_name = vision_encoder
        self.img_size            = vis_cfg["img_size"]
        self.grid_size           = vis_cfg["grid"]

        _backbone_cls = {
            "vit_tiny":      TinyViTBackbone,
            "fastvit_t8":    FastViTBackbone,
            "mobilevit_xxs": MobileViTXXSBackbone,
        }
        self.vision_model = _backbone_cls[vision_encoder](pretrained=pretrained_backbone)

        d_vis                  = self.vision_model.trunk.embed_dim
        self.connector         = nn.Linear(d_text, d_vis)
        self.lin_seg_head      = nn.Linear(d_vis, 1)
        self.num_img_tokens    = vis_cfg["grid"] ** 2

        from transformers import AutoTokenizer, AutoModel
        self.tokenizer  = AutoTokenizer.from_pretrained(hf_id)
        self.text_model = AutoModel.from_pretrained(hf_id)
        for p in self.text_model.parameters():
            p.requires_grad_(False)

    def forward(
        self,
        images: torch.Tensor,
        texts: list[str],
        return_attn: bool = False,
    ) -> torch.Tensor | tuple:
        """Full forward → (B, N+prefix, d_vis), or tuple with attn maps when return_attn=True."""
        device = images.device
        tok = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=64, return_tensors="pt",
        )
        tok = {k: v.to(device) for k, v in tok.items()}
        with torch.no_grad():
            text_out = self.text_model(**tok)
        text_seq   = text_out.last_hidden_state          # (B, L, d_text)
        text_feats = self.connector(text_seq)             # (B, L, d_vis)
        return self.vision_model(images, text_feats, return_attn=return_attn)


# ─── Bbox → patch distribution ───────────────────────────────────────────────
def bbox_to_patch_dist(
    bboxes:    torch.Tensor,   # (B, 4) [x1, y1, x2, y2] in [0, 1]
    grid_size: int,
) -> torch.Tensor:
    """
    bbox の overlap 面積から patch の確率分布 (B, grid_size**2) を作る。
    各 patch について bbox との intersection 面積を計算し正規化。
    """
    B      = bboxes.shape[0]
    g      = grid_size
    device = bboxes.device

    t   = torch.arange(g, device=device, dtype=torch.float32)
    px1 = (t / g).view(1, 1, g)        # (1, 1, g) 各列の左端
    px2 = ((t + 1) / g).view(1, 1, g)
    py1 = (t / g).view(1, g, 1)        # (1, g, 1) 各行の上端
    py2 = ((t + 1) / g).view(1, g, 1)

    bx1 = bboxes[:, 0].view(B, 1, 1)
    by1 = bboxes[:, 1].view(B, 1, 1)
    bx2 = bboxes[:, 2].view(B, 1, 1)
    by2 = bboxes[:, 3].view(B, 1, 1)

    overlap_x = (torch.min(bx2, px2) - torch.max(bx1, px1)).clamp(min=0)  # (B, 1, g)
    overlap_y = (torch.min(by2, py2) - torch.max(by1, py1)).clamp(min=0)  # (B, g, 1)

    area = (overlap_x * overlap_y).reshape(B, g * g)                       # (B, g*g)
    return area / area.sum(dim=-1, keepdim=True).clamp(min=1e-8)


# ─── TeacherCache ─────────────────────────────────────────────────────────────
class TeacherCache:
    """
    事前計算済み SteerViT 教師出力を保持する numpy memmap ベースのキャッシュ。

    キャッシュディレクトリ構成:
      meta.json  - メタデータ (n_samples, teacher_grid, n_blocks, ca_valid, ...)
      soft.dat   - (N, T) float16 memmap  T = teacher_grid**2
      sa.dat     - (N, B, T) float16 memmap  (attn=True 時のみ)
      ca.dat     - (N, B, T) float16 memmap  (attn=True 時のみ)

    SA/CA はヘッド平均・正規化済み分布として格納する:
      SA[i]: CLS→patch 注意の head 平均・正規化 (576,) per block
      CA[i]: patch→text 注意の head 平均・text 平均・正規化 (576,) per block (blocks with CA)
    """

    def __init__(self, cache_dir: str):
        import json, os
        import numpy as np
        self._np = np
        meta_path = os.path.join(cache_dir, "meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Teacher cache not found: {meta_path}")
        with open(meta_path, encoding="utf-8") as f:
            self.meta = json.load(f)

        N   = self.meta["n_samples"]
        T   = self.meta["teacher_grid"] ** 2
        B   = self.meta["n_blocks"]
        self.teacher_grid = self.meta["teacher_grid"]
        self.ca_valid     = self.meta["ca_valid"]  # list[bool] length B

        self.soft = np.memmap(
            os.path.join(cache_dir, "soft.dat"), dtype="float16", mode="r", shape=(N, T)
        )
        sa_path = os.path.join(cache_dir, "sa.dat")
        if os.path.exists(sa_path):
            self.sa = np.memmap(sa_path, dtype="float16", mode="r", shape=(N, B, T))
            self.ca = np.memmap(
                os.path.join(cache_dir, "ca.dat"), dtype="float16", mode="r", shape=(N, B, T)
            )
            self.has_attn = True
        else:
            self.sa = self.ca = None
            self.has_attn = False

    def get_soft(self, indices: list[int], device) -> torch.Tensor:
        arr = self._np.array(self.soft[indices], dtype=self._np.float32)
        return torch.from_numpy(arr).to(device)

    def get_attn(self, indices: list[int], device) -> tuple[list, list]:
        """
        Returns (sa_maps, ca_maps) with compressed distributions:
          sa_maps[i]: (B_batch, T) float32 tensor  or None
          ca_maps[i]: (B_batch, T) float32 tensor  or None
        AttnDistilLoss._sa_loss/_ca_loss detects dim()==2 and treats as pre-computed.
        """
        if not self.has_attn:
            raise RuntimeError("このキャッシュには SA/CA データが含まれていません (--no-attn で作成)")
        sa_arr = self._np.array(self.sa[indices], dtype=self._np.float32)  # (B, n_blocks, T)
        ca_arr = self._np.array(self.ca[indices], dtype=self._np.float32)
        n_blocks = sa_arr.shape[1]
        sa_maps = [torch.from_numpy(sa_arr[:, i, :]).to(device) for i in range(n_blocks)]
        ca_maps = [
            torch.from_numpy(ca_arr[:, i, :]).to(device) if self.ca_valid[i] else None
            for i in range(n_blocks)
        ]
        return sa_maps, ca_maps


# ─── Attention distillation helpers ──────────────────────────────────────────
@torch.no_grad()
def _teacher_forward_collect(
    sv_teacher,
    imgs: torch.Tensor,
    texts: list[str],
    teacher_img_size: int,
    num_prefix_t: int,
) -> tuple[torch.Tensor, list, list]:
    """
    Single teacher forward that returns (soft_heatmap, sa_maps, ca_maps).
    Temporarily enables attn capture on all blocks.
    """
    teacher_imgs = F.interpolate(
        imgs, size=(teacher_img_size, teacher_img_size),
        mode="bicubic", align_corners=False,
    )
    blocks = sv_teacher.vision_model.trunk.blocks

    orig_fused = []
    for blk in blocks:
        orig_fused.append(blk.attn.fused_attn)
        blk.attn.fused_attn = False
        gca = getattr(blk, "gated_cross_attn", None)
        if gca is not None:
            gca.cross_attn.save_attn = True

    tf = sv_teacher.forward(teacher_imgs, list(texts))

    sa_maps, ca_maps = [], []
    for blk, orig_f in zip(blocks, orig_fused):
        sa_maps.append(getattr(blk.attn, "attn_map", None))
        blk.attn.fused_attn = orig_f
        gca = getattr(blk, "gated_cross_attn", None)
        if gca is not None:
            ca_maps.append(gca.cross_attn.attn_map)
            gca.cross_attn.save_attn = False
            gca.cross_attn.attn_map = None
        else:
            ca_maps.append(None)

    tp      = tf[:, num_prefix_t:, :]
    logits  = sv_teacher.lin_seg_head(tp).squeeze(-1)
    soft    = F.softmax(logits, dim=-1)
    return soft, sa_maps, ca_maps


class AttnDistilLoss(nn.Module):
    """
    Block-wise KL-divergence distillation on attention maps.

    Self-attn (SA):
      - CLS→patch attention の head 平均 → (B, N_patches)
      - Teacher 24×24 を student 14×14 に bilinear 補間
      - KL div(teacher || student)

    Cross-attn (CA):
      - patch token の attention を head 平均 + text token 平均 → (B, N_patches)
      - Teacher が cross-attn を持たないブロック (None) はスキップ
      - 同様に bilinear 補間して KL div
    """

    def __init__(
        self,
        teacher_grid: int,
        student_grid: int,
        num_prefix_t: int,
        num_prefix_s: int,
    ):
        super().__init__()
        self.tg           = teacher_grid
        self.sg           = student_grid
        self.num_prefix_t = num_prefix_t
        self.num_prefix_s = num_prefix_s

    def _align(self, t_map: torch.Tensor) -> torch.Tensor:
        """(B, tg*tg) → (B, sg*sg) bilinear, re-normalized."""
        B = t_map.shape[0]
        t = t_map.reshape(B, 1, self.tg, self.tg)
        t = F.interpolate(t, size=(self.sg, self.sg), mode="bilinear", align_corners=False)
        t = t.reshape(B, self.sg * self.sg).clamp(min=0)
        return t / t.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    @staticmethod
    def _kl(t_soft: torch.Tensor, s_soft: torch.Tensor) -> torch.Tensor:
        """KL(t || s), both are probability vectors."""
        log_s = (s_soft + 1e-8).log()
        return -(t_soft * log_s).sum(dim=-1).mean()

    def _sa_loss(
        self,
        sa_t: list,  # list[(B, H_t, N_t+1, N_t+1) | (B, tg*tg) | None]
        sa_s: list,  # list[(B, H_s, N_s+1, N_s+1)]
    ) -> torch.Tensor:
        total, count = 0.0, 0
        for t_map, s_map in zip(sa_t, sa_s):
            if t_map is None:
                continue
            if t_map.dim() == 2:
                # 事前計算済み: (B, tg*tg) 正規化済み分布
                t_soft = t_map
            else:
                # 生の注意マップ: (B, H, N+1, N+1)
                t_vec  = t_map[:, :, 0, self.num_prefix_t:].mean(dim=1)
                t_soft = t_vec / t_vec.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            s_vec  = s_map[:, :, 0, self.num_prefix_s:].mean(dim=1)
            s_soft = s_vec / s_vec.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            t_soft = self._align(t_soft)
            total += self._kl(t_soft, s_soft)
            count += 1
        return total / max(count, 1) if count > 0 else torch.tensor(0.0)

    def _ca_loss(
        self,
        ca_t: list,  # list[(B, H_t, N_t+1, L_t) | (B, tg*tg) | None]
        ca_s: list,  # list[(B, H_s, N_s+1, L_s)]
    ) -> torch.Tensor:
        total, count = 0.0, 0
        for t_map, s_map in zip(ca_t, ca_s):
            if t_map is None:
                continue
            if t_map.dim() == 2:
                # 事前計算済み: (B, tg*tg) 正規化済み分布
                t_soft = t_map
            else:
                # 生の注意マップ: (B, H, N+1, L)
                t_vec  = t_map[:, :, self.num_prefix_t:, :].mean(dim=1).mean(dim=-1)
                t_soft = t_vec / t_vec.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            s_vec  = s_map[:, :, self.num_prefix_s:, :].mean(dim=1).mean(dim=-1)
            s_soft = s_vec / s_vec.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            t_soft = self._align(t_soft)
            total += self._kl(t_soft, s_soft)
            count += 1
        return total / max(count, 1) if count > 0 else torch.tensor(0.0)

    def forward(
        self,
        sa_t: list, ca_t: list,
        sa_s: list, ca_s: list,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._sa_loss(sa_t, sa_s), self._ca_loss(ca_t, ca_s)


# ─── SteerViT heatmap utility ─────────────────────────────────────────────────
@torch.no_grad()
def steervit_heatmap_from_text(
    sv,
    full_imgs:  torch.Tensor,
    texts:      list[str],
    num_prefix: int,
) -> torch.Tensor:
    """RoBERTa 全トークン列を使って softmax heatmap (B, N_patches) を返す。"""
    img_feats   = sv.forward(full_imgs, texts)
    patch_feats = img_feats[:, num_prefix:, :]
    logits      = sv.lin_seg_head(patch_feats).squeeze(-1)
    return F.softmax(logits, dim=-1)


# ─── Heatmap Soft CE Loss ─────────────────────────────────────────────────────
class HeatmapSoftCELoss(nn.Module):
    """
    Patch-wise Soft Cross-Entropy loss。

    teacher_mode:
      "steervit" : SteerViT pretrained の heatmap のみ (デフォルト)
      "bbox"     : GT bbox overlap から作った patch 分布のみ (SteerViT 不要)
      "both"     : SteerViT + bbox の平均

    Teacher SteerViT (sv_teacher, 完全凍結):
      入力画像を teacher_img_size にリサイズしてから forward。
      "bbox" モードでは sv_teacher は使用しない (None 可)。

    Student (sv, TinySteerViT):
      texts → DistilRoBERTa → connector → GatedCrossAttn → lin_seg_head → logits [grad]

    Loss: SoftCE(student_logits, teacher_soft)
    """

    def __init__(
        self,
        sv,
        sv_teacher,                        # "bbox" / cache モード時は None 可
        grid_size:         int            = TINY_VIT_GRID,
        teacher_grid_size: int            = STEERVIT_GRID,
        teacher_img_size:  int            = STEERVIT_IMG_SIZE,
        teacher_mode:      str            = "steervit",  # "steervit" | "bbox" | "both"
        attn_weight:       float          = 0.0,
        sa_weight:         float          = 1.0,
        ca_weight:         float          = 1.0,
        teacher_cache:     "TeacherCache | None" = None,  # 事前計算済みキャッシュ
    ):
        assert teacher_mode in ("steervit", "bbox", "both")
        super().__init__()
        self.sv                = sv
        self.sv_teacher        = sv_teacher
        self.teacher_cache     = teacher_cache
        self.grid_size         = grid_size
        self.teacher_grid_size = teacher_grid_size
        self.teacher_img_size  = teacher_img_size
        self.teacher_mode      = teacher_mode
        self.attn_weight       = attn_weight
        self.sa_weight         = sa_weight
        self.ca_weight         = ca_weight
        self.num_prefix        = sv.vision_model.trunk.num_prefix_tokens
        if sv_teacher is not None:
            self.teacher_num_prefix = sv_teacher.vision_model.trunk.num_prefix_tokens
        elif teacher_cache is not None:
            self.teacher_num_prefix = teacher_cache.meta.get("teacher_num_prefix", 1)
        else:
            self.teacher_num_prefix = 1

        self.attn_distil: AttnDistilLoss | None = None
        use_cache_attn = teacher_cache is not None and teacher_cache.has_attn
        if attn_weight > 0.0 and teacher_mode in ("steervit", "both"):
            if sv_teacher is not None or use_cache_attn:
                self.attn_distil = AttnDistilLoss(
                    teacher_grid=teacher_grid_size,
                    student_grid=grid_size,
                    num_prefix_t=self.teacher_num_prefix,
                    num_prefix_s=self.num_prefix,
                )

        # 最後の forward で記録する内訳 (ログ用)
        self.last_heatmap_loss: float = 0.0
        self.last_sa_loss:      float = 0.0
        self.last_ca_loss:      float = 0.0

    @staticmethod
    def _soft_ce(student_logits: torch.Tensor, teacher_soft: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(student_logits, dim=-1)
        return -(teacher_soft * log_p).sum(dim=-1).mean()

    def _adapt_steervit(self, teacher_soft: torch.Tensor) -> torch.Tensor:
        """SteerViT heatmap (B, tg*tg) → (B, g*g) via bilinear interp."""
        tg, g = self.teacher_grid_size, self.grid_size
        if tg == g:
            return teacher_soft
        B    = teacher_soft.shape[0]
        heat = teacher_soft.reshape(B, 1, tg, tg)
        heat = F.interpolate(heat, size=(g, g), mode="bilinear", align_corners=False)
        heat = heat.reshape(B, g * g)
        return heat / heat.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    def _steervit_soft(self, full_imgs: torch.Tensor, texts: list[str]) -> torch.Tensor:
        teacher_imgs = F.interpolate(
            full_imgs,
            size=(self.teacher_img_size, self.teacher_img_size),
            mode="bicubic", align_corners=False,
        )
        tf       = self.sv_teacher.forward(teacher_imgs, list(texts))
        tp       = tf[:, self.teacher_num_prefix:, :]
        logits   = self.sv_teacher.lin_seg_head(tp).squeeze(-1)
        soft     = F.softmax(logits, dim=-1)
        return self._adapt_steervit(soft)

    def forward(
        self,
        full_imgs:  torch.Tensor,                    # (B, 3, H, W)
        texts:      list[str],
        bboxes:     torch.Tensor | None = None,      # (B, 4) [x1,y1,x2,y2] in [0,1]
        sample_ids: list[int]    | None = None,      # キャッシュインデックス
    ) -> torch.Tensor:
        use_attn  = self.attn_distil is not None
        use_cache = self.teacher_cache is not None and sample_ids is not None
        device    = full_imgs.device

        with torch.no_grad():
            if use_cache and self.teacher_mode in ("steervit", "both"):
                # ─ キャッシュから教師出力をロード ────────────────────────────
                raw_soft = self.teacher_cache.get_soft(sample_ids, device)
                teacher_soft = self._adapt_steervit(raw_soft)
                if use_attn:
                    sa_t, ca_t = self.teacher_cache.get_attn(sample_ids, device)
                if self.teacher_mode == "both":
                    assert bboxes is not None, "both モードには bboxes が必要"
                    bbox_soft    = bbox_to_patch_dist(bboxes.to(device), self.grid_size)
                    teacher_soft = 0.5 * (teacher_soft + bbox_soft)

            elif self.teacher_mode == "steervit":
                if use_attn:
                    raw_soft, sa_t, ca_t = _teacher_forward_collect(
                        self.sv_teacher, full_imgs, texts,
                        self.teacher_img_size, self.teacher_num_prefix,
                    )
                    teacher_soft = self._adapt_steervit(raw_soft)
                else:
                    teacher_soft = self._steervit_soft(full_imgs, texts)

            elif self.teacher_mode == "bbox":
                assert bboxes is not None, "bbox モードには bboxes が必要"
                teacher_soft = bbox_to_patch_dist(
                    bboxes.to(device), self.grid_size
                )

            else:  # "both"
                assert bboxes is not None, "both モードには bboxes が必要"
                if use_attn:
                    raw_soft, sa_t, ca_t = _teacher_forward_collect(
                        self.sv_teacher, full_imgs, texts,
                        self.teacher_img_size, self.teacher_num_prefix,
                    )
                    sv_soft = self._adapt_steervit(raw_soft)
                else:
                    sv_soft = self._steervit_soft(full_imgs, texts)
                bbox_soft    = bbox_to_patch_dist(bboxes.to(full_imgs.device), self.grid_size)
                teacher_soft = 0.5 * (sv_soft + bbox_soft)

        # Student forward (+ attn maps if needed)
        if use_attn:
            sf, sa_s, ca_s = self.sv.forward(full_imgs, list(texts), return_attn=True)
        else:
            sf = self.sv.forward(full_imgs, list(texts))

        sp       = sf[:, self.num_prefix:, :]
        s_logits = self.sv.lin_seg_head(sp).squeeze(-1)
        heatmap_loss = self._soft_ce(s_logits, teacher_soft)

        self.last_heatmap_loss = heatmap_loss.item()
        self.last_sa_loss      = 0.0
        self.last_ca_loss      = 0.0

        if use_attn:
            sa_loss, ca_loss = self.attn_distil(sa_t, ca_t, sa_s, ca_s)
            self.last_sa_loss = sa_loss.item()
            self.last_ca_loss = ca_loss.item()
            attn_total = self.sa_weight * sa_loss + self.ca_weight * ca_loss
            return heatmap_loss + self.attn_weight * attn_total

        return heatmap_loss


# ─── 動作確認 ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import copy
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    for ve in ("vit_tiny", "fastvit_t8", "mobilevit_xxs"):
        print(f"\n--- vision_encoder={ve} ---")
        sv = TinySteerViT(pretrained_backbone=False, vision_encoder=ve).to(device)
        for name, p in sv.named_parameters():
            p.requires_grad_(any(k in name for k in ("connector", "gated_cross_attn", "lin_seg_head")))

        crit = HeatmapSoftCELoss(
            sv=sv, sv_teacher=None,
            grid_size=sv.grid_size, teacher_mode="bbox",
        ).to(device)

        B    = 2
        imgs = torch.randn(B, 3, sv.img_size, sv.img_size, device=device)
        txts = ["cat", "dog"]
        bboxes = torch.tensor([[0.1, 0.1, 0.5, 0.5], [0.3, 0.3, 0.9, 0.9]], device=device)

        loss = crit(imgs, txts, bboxes=bboxes)
        loss.backward()
        n_grad = sum(1 for p in sv.parameters() if p.requires_grad and p.grad is not None)
        print(f"  img={sv.img_size}px  grid={sv.grid_size}  tokens={sv.num_img_tokens}")
        print(f"  loss={loss.item():.4f}  params with grad: {n_grad}")
    print("\nOK ✓")
