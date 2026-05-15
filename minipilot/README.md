# MiniPilot

SteerViT (DINOv2-base pretrained) を teacher として軽量モデルを知識蒸留で学習するスクリプト群。  
ビジョンエンコーダとテキストエンコーダの組み合わせで 4 つのモデルラインナップを定義する。

## モデルラインナップ

| 名前 | Vision Encoder | Text Encoder | グリッド | `d_vis` | パラメータ (学習対象) |
|------|---------------|-------------|---------|---------|----------------------|
| **MiniPilot** | DINOv3 ViT-S/16 (`dinov3_vits`) | MiniLM-L6 (`minilm-l6`) | 14×14 (196) | 384 | ~4M |
| **NanoPilot** | ViT-Tiny/16 (`vit_tiny`) | MiniLM-L6 (`minilm-l6`) | 14×14 (196) | 192 | ~2M |
| **PicoPilot** | FastViT-T8 (`fastvit_t8`) | MiniLM-L6 (`minilm-l6`) | 8×8 (64) | 768 | ~3M |
| **FemtoPilot** | MobileViT-XXS (`mobilevit_xxs`) | MiniLM-L6 (`minilm-l6`) | 8×8 (64) | 320 | ~2M |

```bash
# MiniPilot
python train.py --config config.yaml --vision-encoder dinov3_vits --text-encoder minilm-l6 --out ./ckpts_minipilot

# NanoPilot
python train.py --config config.yaml --vision-encoder vit_tiny --text-encoder minilm-l6 --out ./ckpts_nanopilot

# PicoPilot
python train.py --config config.yaml --vision-encoder fastvit_t8 --text-encoder minilm-l6 --out ./ckpts_picopilot

# FemtoPilot
python train.py --config config.yaml --vision-encoder mobilevit_xxs --text-encoder minilm-l6 --out ./ckpts_femtopilot
```

---

## アーキテクチャ

```
テキスト入力
  └─ テキストエンコーダ (frozen, --text-encoder で選択)
       └─ connector: Linear(d_text → d_vis)  ★学習
            └─ GatedCrossAttn × n_stages  ★学習
                 └─ ビジョンバックボーン (pretrained, frozen, --vision-encoder で選択)
                      └─ lin_seg_head: Linear(d_vis → 1)  ★学習
                           └─ patch logits (grid×grid patches)

損失: SoftCE(student_logits, teacher_soft)
```

| コンポーネント | パラメータ数 | 学習 |
|--------------|------------|------|
| ビジョンバックボーン | 下表参照 | ✗ frozen (ImageNet pretrained) |
| テキストエンコーダ | 下表参照 | ✗ frozen |
| connector (d_text→d_vis) | d_text × d_vis | ✓ |
| GatedCrossAttn × n | ~1–3M | ✓ |
| lin_seg_head | d_vis + 1 | ✓ |

---

## ビジョンエンコーダ (`--vision-encoder`)

`--vision-encoder` でバックボーンを切り替えられる。

| キー | モデル | 入力解像度 | グリッド | `d_vis` | GCA 挿入箇所 |
|------|--------|-----------|---------|---------|-------------|
| `vit_tiny` (デフォルト) | `vit_tiny_patch16_224` | 224×224 | 14×14 (196) | 192 | 各 Transformer ブロック後 (×12) |
| `dinov3_vits` | `vit_small_patch16_dinov3.lvd1689m` | 224×224 | 14×14 (196) | 384 | 各 Transformer ブロック後 (×12) |
| `fastvit_t8` | `fastvit_t8.apple_dist_in1k` | 256×256 | 8×8 (64) | 768 | 各ステージ後 (×4) |
| `mobilevit_xxs` | `mobilevit_xxs` | 256×256 | 8×8 (64) | 320 | 各ステージ後 (×5) |

- `vit_tiny`: patch token ベースの ViT。CLS トークンあり (num_prefix=1)
- `fastvit_t8` / `mobilevit_xxs`: CNN ライクな stem+stages+final_conv 構造。CLS トークンなし (num_prefix=0)

ビジョンエンコーダを変えると `connector` のサイズ (`d_text → d_vis`) と GCA 数が自動的に変わる。  
チェックポイントに保存され、`test.py` ロード時に自動復元される。

```bash
python train.py --config config.yaml --vision-encoder fastvit_t8 --out ./ckpts_fastvit
python train.py --config config.yaml --vision-encoder mobilevit_xxs --out ./ckpts_mobilevit
```

---

## テキストエンコーダ (`--text-encoder`)

| キー | モデル | `d_text` | パラメータ数 |
|------|--------|----------|------------|
| `distilroberta` (デフォルト) | `distilbert/distilroberta-base` | 768 | ~82M |
| `tinybert-l6` | `huawei-noah/TinyBERT_General_6L_768D` | 768 | ~67M |
| `tinybert-l4` | `huawei-noah/TinyBERT_General_4L_312D` | 312 | ~14M |
| `minilm-l6` | `sentence-transformers/all-MiniLM-L6-v2` | 384 | ~23M |

テキストエンコーダを変えると `connector` のサイズ (`d_text → 192`) も自動的に変わる。  
チェックポイントにはテキストエンコーダの情報が保存され、`test.py` ロード時に自動復元される。

---

## Teacher の種類 (`--teacher`)

`--teacher` オプションで teacher の soft target を切り替えられる。

### `steervit` (デフォルト)

SteerViT pretrained の heatmap を teacher とする。

```
full_img (224px)
  → bicubic resize → 336px
  → SteerViT (DINOv2-base, frozen)
  → patch logits (24×24 = 576)
  → softmax → bilinear resize → 14×14 → teacher_soft
```

- 利点: テキストと画像の対応を学習済みモデルが生成する高精度な soft target
- 欠点: SteerViT のロードに GPU メモリが必要 (~1.5GB)

### `bbox`

GT bbox の patch overlap 面積を正規化した分布を teacher とする。

```
bbox [x1, y1, x2, y2] (相対座標)
  → 各 patch (14×14) との intersection 面積を計算
  → 面積で正規化 → teacher_soft
```

- 利点: SteerViT 不要・高速・アノテーション忠実
- 欠点: bbox 内を一様に扱うためテキストの意味は反映されない

### `both`

SteerViT heatmap と bbox mask の平均を teacher とする。

```
teacher_soft = 0.5 × steervit_soft + 0.5 × bbox_soft
```

- 利点: テキスト意味と GT 位置の両方を supervision に使える

---

## 環境

```bash
conda activate steerdet
```

必要パッケージ: `torch`, `timm >= 1.0`, `transformers`, `torchvision`, `pycocotools`, `tqdm`, `matplotlib`, `pyyaml`

---

## データセット

| データセット | パス | 備考 |
|-------------|------|------|
| RefCOCO | `/data2/kato/dataset/coco/refcoco/` | `refs(unc).p` + `instances.json` |
| COCO 2017 | `/data2/kato/dataset/coco/` | `train2017/`, `val2017/`, `annotations/` |
| Visual Genome | `/data2/kato/dataset/vg/` | `region_descriptions.json` |
| GQA | `/data2/kato/dataset/GQA/` | `*_sceneGraphs.json` |

パスは `config.yaml` の `paths` セクションで変更できる。

---

## Config ファイル (`config.yaml`)

全パラメータとパスを `config.yaml` で管理できる。  
**優先順位: CLI 引数 > config.yaml > argparse デフォルト値**

```yaml
paths:
  coco_root:     "/data2/kato/dataset/coco"
  vg_root:       "/data2/kato/dataset/vg"
  gqa_root:      "/data2/kato/dataset/GQA"
  steervit_ckpt: "/path/to/steervit_dinov2_base.pth"

train:
  text_encoder: "distilroberta"   # distilroberta | tinybert-l6 | tinybert-l4 | minilm-l6
  teacher:      "steervit"        # steervit | bbox | both
  dataset:      "refcoco"
  epochs:       50
  batch_size:   64
  lr:           3.0e-4
  out:          "./ckpts"
  ...

test:
  ckpt:       null   # test.py の --ckpt デフォルト値
  dataset:    "refcoco"
  batch_size: 64
  ...
```

```bash
# config.yaml をベースに実行 (CLI 引数で上書き可)
python train.py --config config.yaml

# config の一部だけ上書き
python train.py --config config.yaml --epochs 30 --batch-size 128 --text-encoder tinybert-l4
```

---

## 学習

```bash
# config.yaml のデフォルト設定で実行
python train.py --config config.yaml

# テキストエンコーダを変えて学習
python train.py --config config.yaml --text-encoder tinybert-l4 --out ./ckpts_tinybert_l4
python train.py --config config.yaml --text-encoder tinybert-l6 --out ./ckpts_tinybert_l6
python train.py --config config.yaml --text-encoder minilm-l6   --out ./ckpts_minilm_l6

# bbox mask teacher (SteerViT 不要・高速)
python train.py --config config.yaml --teacher bbox

# ビジョンエンコーダを切り替えて学習
python train.py --config config.yaml --vision-encoder dinov3_vits   --out ./ckpts_dinov3_vits
python train.py --config config.yaml --vision-encoder fastvit_t8    --out ./ckpts_fastvit
python train.py --config config.yaml --vision-encoder mobilevit_xxs --out ./ckpts_mobilevit

# 教師キャッシュを使った学習 (SteerViT をロードしない)
python precompute_teacher.py --dataset refcoco --out ./teacher_cache/refcoco       # 事前計算 (初回のみ)
python train.py --config config.yaml --teacher-cache ./teacher_cache/refcoco       # heatmap 蒸留のみ
python train.py --config config.yaml --teacher-cache ./teacher_cache/refcoco --attn-distil  # attn distil も

# 全データセットで学習
python train.py --config config.yaml --dataset all --batch-size 512 --teacher bbox
```

デフォルト設定:

| 設定 | 値 |
|------|---|
| backbone | vit_tiny_patch16_224 (ImageNet pretrained) |
| text encoder | `distilroberta` (frozen) |
| teacher | `steervit` |
| full image size (student) | 224×224 |
| full image size (teacher) | 336×336 (内部で bicubic リサイズ) |
| epochs | 50 |
| batch size | 64 |
| lr | 3e-4 (CosineAnnealing → 3e-6) |
| optimizer | AdamW (weight_decay=1e-2) |

---

## SageMaker での学習 (`train_aws.py`)

`train_aws.py` は `train.py` の SageMaker 対応版。  
**Training Job としてサブミット**するか、**Notebook Instance のターミナルから直接実行**できる。

### 方法 1 — Training Job (Notebook からサブミット)

#### 1-1. S3 へデータをアップロード

```python
import sagemaker

session = sagemaker.Session()
bucket  = session.default_bucket()   # または任意のバケット名

# ローカル → S3（大容量の場合は !aws s3 sync を推奨）
coco_uri = session.upload_data("./data/coco", bucket=bucket, key_prefix="datasets/coco")
vg_uri   = session.upload_data("./data/vg",   bucket=bucket, key_prefix="datasets/vg")
gqa_uri  = session.upload_data("./data/GQA",  bucket=bucket, key_prefix="datasets/GQA")
```

ターミナルからの場合:

```bash
aws s3 sync ./data/coco s3://your-bucket/datasets/coco
aws s3 sync ./data/vg   s3://your-bucket/datasets/vg
aws s3 sync ./data/GQA  s3://your-bucket/datasets/GQA
```

#### 1-2. Estimator を作成してサブミット

```python
import sagemaker
from sagemaker.pytorch import PyTorch

session = sagemaker.Session()
role    = sagemaker.get_execution_role()
bucket  = session.default_bucket()

estimator = PyTorch(
    entry_point="train_aws.py",
    source_dir="/home/ec2-user/SageMaker/SteerFamily/minipilot",  # ← 実際のパスに合わせる
    requirements_file="requirements_sm.txt",
    role=role,
    instance_type="ml.p3.2xlarge",   # V100×1。安価なら ml.g4dn.xlarge (T4)
    instance_count=1,
    framework_version="2.1",
    py_version="py310",
    hyperparameters={
        "dataset":    "refcoco",
        "teacher":    "bbox",        # bbox なら SteerViT 不要・高速
        "epochs":     50,
        "batch-size": 64,
        "s3-output":  f"s3://{bucket}/runs/exp01",
    },
    output_path=f"s3://{bucket}/runs/exp01/output",
    base_job_name="tinysteervit",
)

estimator.fit(
    inputs={
        "coco": f"s3://{bucket}/datasets/coco",
        "vg":   f"s3://{bucket}/datasets/vg",
        "gqa":  f"s3://{bucket}/datasets/GQA",
    },
    wait=False,   # True にするとログをその場でストリーミング表示
)

print("Job name:", estimator.latest_training_job.name)
```

#### 1-3. モニタリング

```python
# ログをストリーミング表示（wait=False でサブミットした後から attach 可能）
estimator.latest_training_job.wait(logs="All")

# ジョブ名で別セルから attach する場合
from sagemaker.estimator import Estimator
job = Estimator.attach("tinysteervit-2026-xx-xx-xx-xx-xx")
```

CloudWatch Logs の `/aws/sagemaker/TrainingJobs` からも確認できる。

#### 1-4. 結果の取得

学習終了後、`best_model.pth` は `output_path` に `model.tar.gz` として自動保存される。

```python
print(estimator.model_data)
# → s3://your-bucket/runs/exp01/output/tinysteervit-.../output/model.tar.gz

# ローカルにダウンロード & 展開
import boto3, tarfile
from urllib.parse import urlparse

uri    = estimator.model_data
parsed = urlparse(uri)
boto3.client("s3").download_file(parsed.netloc, parsed.path.lstrip("/"), "model.tar.gz")

with tarfile.open("model.tar.gz") as t:
    t.extractall("./downloaded_ckpt")
```

`--s3-output` を指定した場合は学習中にも逐次アップロードされるため、途中経過を S3 から確認できる。

#### インスタンスタイプの目安

| インスタンス | GPU | VRAM | 推奨用途 |
|---|---|---|---|
| `ml.g4dn.xlarge` | T4 ×1 | 16 GB | 動作確認・`bbox` teacher |
| `ml.p3.2xlarge` | V100 ×1 | 16 GB | 通常の学習 |
| `ml.p3.8xlarge` | V100 ×4 | 64 GB | `steervit` teacher・大バッチ |

> `--teacher bbox` なら SteerViT のロードが不要なので `ml.g4dn.xlarge` で十分動く。

#### `steervit` パッケージについて

`torch` との依存競合があるため `requirements_sm.txt` には含めていない。  
`train_aws.py` が SageMaker 環境を検出すると自動的に `--no-deps` でインストールする。  
`--teacher bbox` を使う場合はインストール自体が不要（`steervit` はインポートされない）。

SageMaker の入出力は以下のように自動マッピングされる:

| SM チャンネル / 環境変数 | 対応する引数 |
|---|---|
| `SM_CHANNEL_COCO` | `--coco-root` |
| `SM_CHANNEL_VG` | `--vg-root` |
| `SM_CHANNEL_GQA` | `--gqa-root` |
| `SM_CHANNEL_STEERVIT_CKPT` | `--steervit-ckpt` |
| `SM_MODEL_DIR` | `--out`（チェックポイント保存先） |
| `SM_HPS` | 全ハイパーパラメータ |

ハイパーパラメータの優先順位: **CLI 引数 > SM_HPS > config.yaml > デフォルト値**

### 方法 2 — Notebook Instance のターミナルから直接実行

```bash
python train_aws.py \
    --coco-root /home/ec2-user/SageMaker/data/coco \
    --teacher bbox \
    --dataset refcoco \
    --epochs 30 \
    --s3-output s3://your-bucket/runs/exp01
```

### SteerViT チェックポイントの解決順序

`--teacher steervit` または `--teacher both` のとき、以下の順で ckpt を探す:

1. `--steervit-ckpt` がファイルなら直接使用
2. ディレクトリなら内部の `*.pth` を自動検索（SM チャンネルはディレクトリで渡されるため）
3. どちらも存在しなければ `huggingface_hub` で自動ダウンロード

### `--s3-output` による S3 自動アップロード

```bash
python train_aws.py \
    --coco-root /data/coco \
    --teacher bbox \
    --s3-output s3://your-bucket/runs/exp01
```

- `best_model.pth` 更新のたびに即アップロード
- 学習完了後に `history.json` / `train.log` もアップロード
- boto3 が必要 (`pip install boto3`)

### train.py との主な違い

| 項目 | train.py | train_aws.py |
|---|---|---|
| データパス | CLI 引数 / config.yaml | SM チャンネル → CLI 引数 → config.yaml の順で解決 |
| ハイパーパラメータ | CLI 引数 / config.yaml | SM_HPS → CLI 引数 → config.yaml の順で解決 |
| 出力先デフォルト | `./ckpts` | `SM_MODEL_DIR`（未設定なら `./ckpts`） |
| SteerViT ckpt 解決 | ファイルパス直接指定のみ | ファイル / ディレクトリ / HuggingFace Hub の順 |
| S3 アップロード | なし | `--s3-output` で best_model.pth を逐次アップロード |
| `--num-workers` デフォルト | 8 | 4 |

---

## Attention Distillation (`--attn-distil`)

`--teacher steervit` または `--teacher both` のとき、出力 heatmap に加えて各 Transformer ブロックの **Self-Attention (SA)** と **Cross-Attention (CA)** の分布も teacher から蒸留できる。

### 蒸留する情報

| 種類 | 取得元 | 集約方法 |
|------|--------|----------|
| **SA** | 各ブロックの ViT self-attention | CLS→patch 注意を head 平均 → (B, N_patches) |
| **CA** | 各ブロックの GatedCrossAttn | patch→text 注意を head 平均 + text token 平均 → (B, N_patches) |

Teacher (24×24) と Student (14×14) の空間サイズは bilinear 補間で整合し、KL-div で蒸留する。

### 損失の構成

```
total_loss = heatmap_loss
           + attn_weight × (sa_weight × SA_loss + ca_weight × CA_loss)
```

### 使用例

```bash
# デフォルト重み (attn_weight=1.0, sa=1.0, ca=1.0)
python train.py --config config.yaml --teacher steervit --attn-distil

# heatmap : attn = 1 : 0.5, SA のみ
python train.py --config config.yaml --teacher steervit --attn-distil \
    --attn-weight 0.5 --ca-weight 0.0

# both モードでも使用可
python train.py --config config.yaml --teacher both --attn-distil \
    --attn-weight 1.0 --sa-weight 1.0 --ca-weight 0.5
```

> **注意**: `--teacher bbox` のときは attn-distil は無視される（SteerViT 不使用のため）。

### ログ出力

`--attn-distil` 有効時は各エポックのログに内訳が追記される:

```
[  1/ 50] train=16.01  val=15.83  lr=3.00e-04  12.3s  hmap=5.46  sa=5.28  ca=5.28
```

`history.json` にも `last_heatmap_loss` / `last_sa_loss` / `last_ca_loss` が記録される。

### 速度・メモリへの影響

attn-distil を有効にすると、通常の `--teacher steervit` に比べて以下のオーバーヘッドが生じる:

| 項目 | 影響 |
|------|------|
| Teacher forward 回数 | 変化なし（heatmap と attn を 1 回の forward で同時取得） |
| Teacher forward の速度 | Flash Attention が全 12 ブロックで無効化されるため遅くなる |
| Student forward の速度 | 全 12 ブロックで attention map を手動計算するため遅くなる |
| VRAM | 全ブロックの attention map を保持するため増加 |

→ 学習を高速化したい場合は **教師キャッシュ** (`precompute_teacher.py`) を使うこと。

---

## 教師キャッシュ (`precompute_teacher.py`)

`--teacher steervit` や `--attn-distil` を使うと毎バッチ SteerViT を実行するため学習が遅くなる。  
`precompute_teacher.py` で **SteerViT の出力を事前計算・保存**しておくと、学習中は SteerViT をロードせずにキャッシュから読むだけになる。

### 保存されるファイル

| ファイル | 内容 | 形状 | dtype |
|---------|------|------|-------|
| `soft.dat` | softmax heatmap | (N, 576) | float16 |
| `sa.dat` | SA 注意の圧縮分布 | (N, n_blocks, 576) | float16 |
| `ca.dat` | CA 注意の圧縮分布 | (N, n_blocks, 576) | float16 |
| `meta.json` | メタデータ | — | JSON |

`--no-attn` を付けると `sa.dat` / `ca.dat` を省略（heatmap 蒸留のみのとき推奨）。

SA/CA の圧縮方法:
- **SA**: CLS→patch 注意をヘッド平均 → 正規化 → (576,) per block
- **CA**: patch→text 注意をヘッド平均 + テキスト平均 → 正規化 → (576,) per block

### ステップ 1: キャッシュ生成

```bash
# heatmap + SA/CA を保存（--attn-distil と組み合わせる場合）
python precompute_teacher.py \
    --dataset refcoco \
    --out ./teacher_cache/refcoco \
    --split train

# heatmap のみ（高速、attn distil なしで使う場合）
python precompute_teacher.py \
    --dataset refcoco \
    --out ./teacher_cache/refcoco_noattn \
    --split train \
    --no-attn

# val set も同様に
python precompute_teacher.py --dataset refcoco --out ./teacher_cache/refcoco_val --split val
```

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--dataset` | (必須) | `refcoco` / `refcoco+` / `refcocog` / `vg` / `gqa` / `coco` |
| `--out` | (必須) | 出力ディレクトリ |
| `--split` | `train` | `train` / `val` |
| `--no-attn` | off | SA/CA を保存しない (soft heatmap のみ) |
| `--steervit-ckpt` | config | SteerViT チェックポイントパス |
| `--batch-size` | `32` | バッチサイズ |
| `--num-workers` | `4` | DataLoader ワーカー数 |

### ステップ 2: キャッシュを使って学習

```bash
# heatmap 蒸留のみ（SteerViT をロードしない）
python train.py --config config.yaml \
    --dataset refcoco \
    --teacher-cache ./teacher_cache/refcoco_noattn

# attn distil もキャッシュから（SA/CA 込みで生成したキャッシュが必要）
python train.py --config config.yaml \
    --dataset refcoco \
    --teacher-cache ./teacher_cache/refcoco \
    --attn-distil
```

`--teacher-cache` を指定すると:
- SteerViT を**ロードしない**（起動高速化・VRAM 節約）
- `fix_text=True` でテキスト選択を決定論的にしてキャッシュと一致させる
- バッチにサンプルインデックスを付加してキャッシュを正しく参照する

> **注意**: キャッシュは **train split のみ**を対象にすること。  
> val set は元々 SteerViT を実行しないので高速（キャッシュ不要）。

---

## オプション一覧

### train.py

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--config` | — | YAML config ファイル (CLI 引数で上書き可) |
| `--text-encoder` | `distilroberta` | `distilroberta` / `tinybert-l6` / `tinybert-l4` / `minilm-l6` |
| `--vision-encoder` | `vit_tiny` | `vit_tiny` / `dinov3_vits` / `fastvit_t8` / `mobilevit_xxs` |
| `--teacher` | `steervit` | teacher の種類: `steervit` / `bbox` / `both` |
| `--no-pretrained-backbone` | — | ビジョンバックボーンも完全ランダム初期化 |
| `--dataset` | `refcoco` | `refcoco` / `refcoco+` / `refcocog` / `vg` / `gqa` / `coco` / `all` |
| `--epochs` | `50` | 学習エポック数 |
| `--batch-size` | `64` | バッチサイズ |
| `--lr` | `3e-4` | 初期学習率 |
| `--weight-decay` | `1e-2` | AdamW weight decay |
| `--amp` | on | Mixed precision (FP16) |
| `--resume` | — | チェックポイントから再開 |
| `--out` | `./ckpts` | 出力ディレクトリ |
| `--vis-every` | `5` | N エポックごとに heatmap 可視化 (0 で無効) |
| `--vis-n` | `4` | val set から自動収集する probe 数 |
| `--vis-samples` | — | カスタム probe JSON |
| `--attn-distil` | off | SA / CA attention map の蒸留を有効化 |
| `--attn-weight` | `1.0` | attention distil loss 全体の重み |
| `--sa-weight` | `1.0` | SA loss の重み |
| `--ca-weight` | `1.0` | CA loss の重み |
| `--teacher-cache` | — | 事前計算済み教師キャッシュのディレクトリ (`precompute_teacher.py` で生成) |
| `--coco-root` | (config) | COCO データセットのルートパス |
| `--vg-root` | (config) | Visual Genome のルートパス |
| `--gqa-root` | (config) | GQA のルートパス |
| `--steervit-ckpt` | (config) | SteerViT チェックポイントパス |

### train_aws.py

`train.py` の全オプションに加えて以下を追加:

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--s3-output` | — | チェックポイントをアップロードする S3 URI (例: `s3://bucket/runs/exp01`) |

SM 環境変数によるパス上書き（自動）:

| 環境変数 | 上書き対象 |
|---|---|
| `SM_CHANNEL_COCO` | `--coco-root` |
| `SM_CHANNEL_VG` | `--vg-root` |
| `SM_CHANNEL_GQA` | `--gqa-root` |
| `SM_CHANNEL_STEERVIT_CKPT` | `--steervit-ckpt` |
| `SM_MODEL_DIR` | `--out` |
| `SM_HPS` | 全ハイパーパラメータ |

### test.py / test_aws.py

オプションは共通。

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--config` | — | YAML config ファイル |
| `--ckpt` | (必須) | TinySteerViT チェックポイント (.pth) |
| `--dataset` | `refcoco` | `refcoco` / `refcoco+` / `refcocog` / `vg` / `gqa` |
| `--split` | 全 split | `val` / `testA` / `testB` (refcocog・vg・gqa は `val` のみ) |
| `--max-samples` | 全件 | 各 split の最大サンプル数 |
| `--batch-size` | `64` | バッチサイズ |
| `--num-workers` | `4` | DataLoader ワーカー数 |
| `--no-steervit` | — | SteerViT teacher との比較をスキップ |
| `--no-wrong-prompt` | — | Wrong-prompt 制御実験をスキップ |
| `--log` | 自動生成 | 結果を保存する JSON ファイルパス |
| `--coco-root` | (config) | COCO データセットのルートパス |
| `--vg-root` | (config) | Visual Genome のルートパス |
| `--gqa-root` | (config) | GQA のルートパス |
| `--steervit-ckpt` | (config) | SteerViT チェックポイントパス |

---

## 出力

```
ckpts/
  train.log          # タイムスタンプ付きログ
  history.json       # エポックごとの train/val loss
  best_model.pth     # val_loss 最良チェックポイント
  vis/
    epoch_000.png    # 学習前 (初期状態)
    epoch_005.png
    ...
```

`best_model.pth` の構造:

```python
{
    "epoch":    int,
    "sv_state": dict,   # connector + gated_cross_attn + lin_seg_head の state_dict
    "val_loss": float,
    "args":     dict,   # text_encoder を含む学習時の全引数
}
```

再開:

```bash
python train.py --config config.yaml --resume ./ckpts/best_model.pth
```

---

## 可視化

各エポックの PNG の列構成は `--teacher` によって変わる:

| `--teacher` | 列1 | 列2 | 列3 | 列4 |
|------------|-----|-----|-----|-----|
| `steervit` | full image + bbox | teacher (SteerViT) | student | — |
| `bbox`     | full image + bbox | teacher (bbox mask) | student | — |
| `both`     | full image + bbox | teacher (SteerViT) | teacher (bbox mask) | student |

- **full image + bbox**: 入力画像にテキストと GT bbox を黄枠で描画
- **teacher (SteerViT)**: SteerViT が 336px 入力で生成した heatmap (24×24 grid)
- **teacher (bbox mask)**: GT bbox の patch overlap を正規化した分布 (14×14 grid)
- **student**: TinySteerViT の出力 heatmap (14×14 grid)

カスタム probe を使う場合:

```json
[
  {"img_path": "/path/to/image.jpg", "text": "person on the left", "bbox": [0.1, 0.2, 0.6, 0.9]},
  {"img_path": "/path/to/image2.jpg", "text": "red car"}
]
```

`bbox` は `[x1, y1, x2, y2]` 相対座標。省略時は `[0, 0, 1, 1]`（画像全体）を使用。

```bash
python train.py --config config.yaml --vis-samples probes.json --vis-every 1
```

---

## 評価 (CORE Benchmark)

論文 (arXiv:2604.02327) の CORE (COnditional REtrieval) ベンチマークの代替評価として、
RefCOCO / RefCOCO+ / RefCOCOg / Visual Genome / GQA の Pointing Accuracy を計測する。

テキストエンコーダはチェックポイントから自動復元される。

### 評価指標

| 指標 | 説明 |
|------|------|
| **PA** (Pointing Accuracy) | argmax heatmap patch の中心が GT bbox 内なら hit |
| **AHP** (Avg Heatmap Precision) | GT bbox 内の heatmap 質量の割合 |
| **PA_wrong** | テキストをシャッフルしたときの PA |
| **ΔPA** | PA − PA_wrong（大きいほどテキスト依存） |

### 実行例

```bash
# config.yaml のパスを使って評価
python test.py --config config.yaml --ckpt ./ckpts_tinyvit_all/best_model.pth

# split を指定
python test.py --config config.yaml --ckpt ./ckpts_tinyvit_all/best_model.pth --split val

# データセットを切り替え
python test.py --config config.yaml --ckpt ./ckpts_tinyvit_all/best_model.pth --dataset refcoco+
python test.py --config config.yaml --ckpt ./ckpts_tinyvit_all/best_model.pth --dataset vg
python test.py --config config.yaml --ckpt ./ckpts_tinyvit_all/best_model.pth --dataset gqa

# サンプル数を制限して高速確認
python test.py --config config.yaml --ckpt ./ckpts_tinyvit_all/best_model.pth --max-samples 500

# SteerViT との比較をスキップ (TinySteerViT のみ)
python test.py --config config.yaml --ckpt ./ckpts_tinyvit_all/best_model.pth --no-steervit
```

---

## AWS での評価 (`test_aws.py`)

`test_aws.py` は `test.py` の SageMaker / Notebook Instance 対応版。  
**Notebook Instance のターミナルから直接実行**することを想定している。

### test.py との主な違い

| 項目 | test.py | test_aws.py |
|------|---------|-------------|
| tqdm | 使用 | 不使用（SageMaker ログとの相性問題を回避） |
| steervit パッケージ | 事前インストール必須 | `_install_steervit()` で自動インストール |
| model.py の互換性 | 新版のみ (`vision_encoder` 引数あり) | 旧版・新版どちらでも動作 |

### 実行例

```bash
# Notebook Instance のターミナルから実行
cd /home/ec2-user/SageMaker/mywork/SteerFamily/minipilot

python test_aws.py --ckpt ./ckpts_minilm_l6/best_model.pth \
    --coco-root /home/ec2-user/SageMaker/mywork/dataset/coco

# TinySteerViT のみ (SteerViT をスキップ)
python test_aws.py --ckpt ./ckpts_minilm_l6/best_model.pth --no-steervit

# 高速確認 (各 split 500 件)
python test_aws.py --ckpt ./ckpts_minilm_l6/best_model.pth --max-samples 500
```

---

## ファイル構成

```
minipilot/
  config.yaml             # 全パラメータ・パスのデフォルト設定
  model.py                # GatedCrossAttn / TinyViTBackbone / FastViTBackbone
                          # MobileViTXXSBackbone / TinySteerViT / TeacherCache
                          # AttnDistilLoss / HeatmapSoftCELoss
                          # TEXT_ENCODERS / VISION_ENCODERS
  dataset.py              # RefCOCOLocalDataset / COCODetDataset / VGRegionDataset
                          # GQASceneGraphDataset / _IndexedWrapper / build_loaders
  train.py                # 学習スクリプト (ローカル実行)
  train_aws.py            # 学習スクリプト (SageMaker Training Job / Notebook Instance 対応)
  test.py                 # CORE benchmark 評価スクリプト (ローカル実行)
  test_aws.py             # CORE benchmark 評価スクリプト (SageMaker / Notebook Instance 対応)
  precompute_teacher.py     # SteerViT 教師出力の事前計算・キャッシュ生成 (ローカル)
  precompute_teacher_aws.py # 同上 (SageMaker / Notebook Instance 対応版)
  requirements.txt        # ローカル実行用 requirements
  requirements_sm.txt     # SageMaker Training Job 用 requirements (torch 除外済み)
  train.sh                # ローカル学習の実行例
  train_aws.sh            # AWS 学習の実行例
```
