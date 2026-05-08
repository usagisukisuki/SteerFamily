# tiny_steer

RefCOCO / COCO を使って TinySteerViT を学習するスクリプト。  
元の SteerViT (DINOv2-base pretrained) を teacher として、vit_tiny_patch16_224 ベースの軽量モデルを知識蒸留で学習する。

---

## アーキテクチャ

```
テキスト入力
  └─ テキストエンコーダ (frozen, --text-encoder で選択)
       └─ connector: Linear(d_text → 192)  ★学習
            └─ GatedCrossAttn × 12  ★学習
                 └─ vit_tiny_patch16_224 (ImageNet pretrained, frozen)
                      └─ lin_seg_head: Linear(192 → 1)  ★学習
                           └─ patch logits (14×14 = 196 patches)

損失: SoftCE(student_logits, teacher_soft)
```

| コンポーネント | パラメータ数 | 学習 |
|--------------|------------|------|
| ViT-Tiny backbone | ~5.7M | ✗ frozen (ImageNet pretrained) |
| テキストエンコーダ | 下表参照 | ✗ frozen |
| connector (d_text→192) | d_text × 192 | ✓ |
| GatedCrossAttn × 12 | ~2.2M | ✓ |
| lin_seg_head | ~193 | ✓ |

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

---

## オプション一覧

### train.py

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--config` | — | YAML config ファイル (CLI 引数で上書き可) |
| `--text-encoder` | `distilroberta` | `distilroberta` / `tinybert-l6` / `tinybert-l4` / `minilm-l6` |
| `--teacher` | `steervit` | teacher の種類: `steervit` / `bbox` / `both` |
| `--no-pretrained-backbone` | — | ViT-Tiny backbone も完全ランダム初期化 |
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
| `--coco-root` | (config) | COCO データセットのルートパス |
| `--vg-root` | (config) | Visual Genome のルートパス |
| `--gqa-root` | (config) | GQA のルートパス |
| `--steervit-ckpt` | (config) | SteerViT チェックポイントパス |

### test.py

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

## ファイル構成

```
tiny_steer/
  config.yaml   # 全パラメータ・パスのデフォルト設定
  model.py      # GatedCrossAttn / TinyViTBackbone / TinySteerViT
                # AttnDistilLoss / HeatmapSoftCELoss / TEXT_ENCODERS
  dataset.py    # RefCOCOLocalDataset / COCODetDataset / VGRegionDataset
                # GQASceneGraphDataset / build_loaders
  train.py      # 学習スクリプト
  train.sh      # 実行例 (各テキストエンコーダのコメントアウト例あり)
  test.py       # CORE benchmark 評価スクリプト
```
