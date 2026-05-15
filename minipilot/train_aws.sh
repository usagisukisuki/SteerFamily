#!/bin/bash
# SageMaker Notebook Instance 上での直接実行スクリプト
# ターミナルから: bash train_aws.sh
# バックグラウンド実行: nohup bash train_aws.sh > train_output.log 2>&1 &

COCO_ROOT="/home/ec2-user/SageMaker/data/coco"
VG_ROOT="/home/ec2-user/SageMaker/data/vg"
GQA_ROOT="/home/ec2-user/SageMaker/data/GQA"

cd "$(dirname "$0")"

# ── DistilRoBERTa + bbox teacher (デフォルト) ─────────────────────────────────
python train_aws.py \
  --coco-root "$COCO_ROOT" \
  --vg-root   "$VG_ROOT" \
  --gqa-root  "$GQA_ROOT" \
  --teacher   bbox \
  --dataset   all \
  --epochs    30 \
  --batch-size 512 \
  --out       ./ckpts_tinyvit_all

# ── MiniLM-L6 ────────────────────────────────────────────────────────────────
#python train_aws.py \
#  --coco-root "$COCO_ROOT" \
#  --vg-root   "$VG_ROOT" \
#  --gqa-root  "$GQA_ROOT" \
#  --text-encoder minilm-l6 \
#  --teacher   bbox \
#  --dataset   all \
#  --epochs    30 \
#  --batch-size 512 \
#  --out       ./ckpts_minilm_l6

# ── TinyBERT-L4 ──────────────────────────────────────────────────────────────
#python train_aws.py \
#  --coco-root "$COCO_ROOT" \
#  --vg-root   "$VG_ROOT" \
#  --gqa-root  "$GQA_ROOT" \
#  --text-encoder tinybert-l4 \
#  --teacher   bbox \
#  --dataset   all \
#  --epochs    30 \
#  --batch-size 512 \
#  --out       ./ckpts_tinybert_l4

# ── TinyBERT-L6 ──────────────────────────────────────────────────────────────
#python train_aws.py \
#  --coco-root "$COCO_ROOT" \
#  --vg-root   "$VG_ROOT" \
#  --gqa-root  "$GQA_ROOT" \
#  --text-encoder tinybert-l6 \
#  --teacher   bbox \
#  --dataset   all \
#  --epochs    30 \
#  --batch-size 512 \
#  --out       ./ckpts_tinybert_l6
