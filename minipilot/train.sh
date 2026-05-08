#!/bin/bash
# 学習スクリプト
# config.yaml でデフォルト値を管理。CLI 引数で上書き可能。

# ── DistilRoBERTa (デフォルト) ────────────────────────────────────────────────
python train.py \
  --config config.yaml \
  --dataset all \
  --epochs 30 \
  --batch-size 512 \
  --out ./ckpts_tinyvit_all \
  --teacher bbox

# ── TinyBERT-L4 (最軽量テキストエンコーダ) ───────────────────────────────────
#python train.py \
#  --config config.yaml \
#  --text-encoder tinybert-l4 \
#  --dataset all \
#  --epochs 30 \
#  --batch-size 512 \
#  --out ./ckpts_tinybert_l4 \
#  --teacher bbox

# ── TinyBERT-L6 (768-dim, L4 より高精度) ─────────────────────────────────────
#python train.py \
#  --config config.yaml \
#  --text-encoder tinybert-l6 \
#  --dataset all \
#  --epochs 30 \
#  --batch-size 512 \
#  --out ./ckpts_tinybert_l6 \
#  --teacher bbox

# ── all-MiniLM-L6-v2 (384-dim, sentence 特化) ────────────────────────────────
#python train.py \
#  --config config.yaml \
#  --text-encoder minilm-l6 \
#  --dataset all \
#  --epochs 30 \
#  --batch-size 512 \
#  --out ./ckpts_minilm_l6 \
#  --teacher bbox

# ── Attention distillation (steervit teacher) ────────────────────────────────
#python train.py \
#  --config config.yaml \
#  --text-encoder distilroberta \
#  --dataset refcoco \
#  --epochs 50 \
#  --batch-size 64 \
#  --out ./ckpts_attn_distil \
#  --teacher steervit \
#  --attn-distil \
#  --attn-weight 1.0
