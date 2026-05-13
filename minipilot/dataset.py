"""
dataset.py - Grounding Dataset

RefCOCOLocalDataset: ローカルの refcoco/refs(unc).p + instances.json を使用。
                     train/val/testA/testB split 対応。referring expression。
COCODetDataset:      COCO 2017 train/val のアノテーション (bbox + category name)。
VGRegionDataset:     Visual Genome region descriptions (phrase + bbox)。
GQASceneGraphDataset: GQA scene graphs (object name/attributes + bbox)。

__getitem__ 戻り値:
    full_img : (3, full_img_size, full_img_size)  SteerViT 用 full image
    bbox_rel : (4,) [x1/W, y1/H, x2/W, y2/H] 相対座標
    text     : str  (referring expression or category name)
"""

from __future__ import annotations
import json
import os
import pickle
import random
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms
from pycocotools.coco import COCO

COCO_ROOT         = "/data2/kato/dataset/coco"
REFCOCO_DIR       = os.path.join(COCO_ROOT, "refcoco")
REFCOCO_PLUS_DIR  = os.path.join(COCO_ROOT, "refcoco+")
REFCOCOG_DIR      = os.path.join(COCO_ROOT, "refcocog")
VG_ROOT           = "/data2/kato/dataset/vg"
GQA_ROOT          = "/data2/kato/dataset/GQA"
STEERVIT_IMG_SIZE = 336

_FULL_MEAN = (0.485, 0.456, 0.406)
_FULL_STD  = (0.229, 0.224, 0.225)


def _full_img_transform(size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(_FULL_MEAN, _FULL_STD),
    ])


import re
_FNAME_RE = re.compile(r"^(COCO_(train|val)2014_\d+)_\d+\.jpg$")

def _resolve_img_path(file_name: str, coco_root: str) -> str:
    m = _FNAME_RE.match(file_name)
    if m:
        base   = m.group(1)
        subset = m.group(2)
        return os.path.join(coco_root, f"{subset}2014", f"{base}.jpg")
    for sub in ("train2014", "val2014", "train2017", "val2017"):
        p = os.path.join(coco_root, sub, file_name)
        if os.path.exists(p):
            return p
    return os.path.join(coco_root, file_name)


class RefCOCOLocalDataset(Dataset):
    """
    ローカルの refcoco*/refs(unc|umd).p + instances.json から referring expression を読む。

    split: "train" / "val" / "testA" / "testB"
    fix_text=False (デフォルト): エポックごとにランダムな referring expression を選択。
    fix_text=True: idx ベースの決定論的選択 (teacher cache 使用時に必要)。
    """

    def __init__(
        self,
        refcoco_dir:    str           = REFCOCO_DIR,
        coco_root:      str           = COCO_ROOT,
        split:          str           = "train",
        full_img_size:  int           = STEERVIT_IMG_SIZE,
        min_area:       int           = 32 * 32,
        min_side:       int           = 16,
        max_samples:    Optional[int] = None,
        seed:           int           = 42,
        refs_filename:  Optional[str] = None,
        fix_text:       bool          = False,
    ):
        self.full_transform = _full_img_transform(full_img_size)
        self.rng            = random.Random(seed)
        self.seed           = seed
        self.fix_text       = fix_text

        if refs_filename is None:
            refs_filename = "refs(umd).p" if "refcocog" in refcoco_dir else "refs(unc).p"

        with open(os.path.join(refcoco_dir, "instances.json"), encoding="utf-8") as f:
            instances = json.load(f)
        ann_to_bbox = {a["id"]: a["bbox"] for a in instances["annotations"]}

        with open(os.path.join(refcoco_dir, refs_filename), "rb") as f:
            refs = pickle.load(f)

        samples = []
        for ref in refs:
            if ref["split"] != split:
                continue
            ann_id = ref["ann_id"]
            if ann_id not in ann_to_bbox:
                continue
            x, y, w, h = ann_to_bbox[ann_id]
            if w * h < min_area or w < min_side or h < min_side:
                continue
            sents = [s["sent"] for s in ref["sentences"]]
            if not sents:
                continue
            samples.append({
                "img_path": _resolve_img_path(ref["file_name"], coco_root),
                "bbox":     [x, y, w, h],
                "sents":    sents,
            })

        if max_samples is not None:
            samples = random.Random(seed).sample(samples, min(max_samples, len(samples)))

        self.samples = samples
        tag = os.path.basename(refcoco_dir)
        print(f"[RefCOCOLocalDataset:{tag}:{split}] {len(self.samples):,} サンプル")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        s   = self.samples[idx]
        img = Image.open(s["img_path"]).convert("RGB")
        W, H = img.size
        x, y, w, h = s["bbox"]
        bbox_rel = torch.tensor([
            max(0.0, x) / W,
            max(0.0, y) / H,
            min(float(W), x + w) / W,
            min(float(H), y + h) / H,
        ], dtype=torch.float32)
        if self.fix_text:
            text = random.Random(self.seed + idx).choice(s["sents"])
        else:
            text = self.rng.choice(s["sents"])
        return self.full_transform(img), bbox_rel, text


class COCODetDataset(Dataset):
    """COCO Detection データセット。instances_train2017.json 等を使用。"""

    def __init__(
        self,
        ann_file:      str,
        image_dir:     str,
        full_img_size: int           = STEERVIT_IMG_SIZE,
        min_area:      int           = 32 * 32,
        min_side:      int           = 16,
        max_samples:   Optional[int] = None,
        seed:          int           = 42,
    ):
        self.image_dir      = image_dir
        self.full_transform = _full_img_transform(full_img_size)

        coco      = COCO(ann_file)
        cat_names = {c["id"]: c["name"] for c in coco.dataset["categories"]}

        samples = []
        for ann in coco.dataset["annotations"]:
            x, y, w, h = ann["bbox"]
            if w * h < min_area or w < min_side or h < min_side:
                continue
            img_info = coco.loadImgs(ann["image_id"])[0]
            samples.append({
                "img_path": os.path.join(image_dir, img_info["file_name"]),
                "bbox":     [x, y, w, h],
                "img_w":    img_info["width"],
                "img_h":    img_info["height"],
                "text":     cat_names[ann["category_id"]],
            })

        if max_samples is not None:
            samples = random.Random(seed).sample(samples, min(max_samples, len(samples)))

        self.samples = samples
        print(f"[COCODetDataset] {len(self.samples):,} サンプル ({ann_file})")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        s    = self.samples[idx]
        img  = Image.open(s["img_path"]).convert("RGB")
        W, H = s["img_w"], s["img_h"]
        x, y, w, h = s["bbox"]
        bbox_rel = torch.tensor([
            max(0.0, x) / W,
            max(0.0, y) / H,
            min(float(W), x + w) / W,
            min(float(H), y + h) / H,
        ], dtype=torch.float32)
        return self.full_transform(img), bbox_rel, s["text"]


class VGRegionDataset(Dataset):
    """
    Visual Genome region descriptions データセット。

    region_descriptions.json から phrase (text) と bbox を読む。
    公式 train/val split がないため、画像 ID の昇順ソート後に先頭 (1-val_ratio) を
    train、末尾 val_ratio を val として使用する。
    """

    def __init__(
        self,
        vg_root:       str           = VG_ROOT,
        split:         str           = "train",
        full_img_size: int           = STEERVIT_IMG_SIZE,
        min_area:      int           = 32 * 32,
        min_side:      int           = 16,
        max_samples:   Optional[int] = None,
        seed:          int           = 42,
        val_ratio:     float         = 0.05,
    ):
        self.full_transform = _full_img_transform(full_img_size)
        self.rng            = random.Random(seed)

        img_dirs = [
            os.path.join(vg_root, "VG_100K"),
            os.path.join(vg_root, "VG_100K_2"),
        ]

        with open(os.path.join(vg_root, "region_descriptions.json"), encoding="utf-8") as f:
            raw = json.load(f)

        # 画像 ID でソートして決定論的に分割
        raw.sort(key=lambda x: x["id"])
        n_val   = max(1, int(len(raw) * val_ratio))
        if split == "val":
            raw = raw[-n_val:]
        else:
            raw = raw[:-n_val]

        samples = []
        for img_entry in raw:
            img_id = img_entry["id"]
            img_path = self._find_img(img_id, img_dirs)
            if img_path is None:
                continue
            for region in img_entry["regions"]:
                x, y = region["x"], region["y"]
                w, h = region["width"], region["height"]
                if w * h < min_area or w < min_side or h < min_side:
                    continue
                phrase = region.get("phrase", "").strip()
                if not phrase:
                    continue
                samples.append({
                    "img_path": img_path,
                    "img_id":   img_id,
                    "bbox":     [x, y, w, h],
                    "text":     phrase,
                })

        if max_samples is not None:
            samples = random.Random(seed).sample(samples, min(max_samples, len(samples)))

        self.samples = samples
        print(f"[VGRegionDataset:{split}] {len(self.samples):,} サンプル")

    @staticmethod
    def _find_img(img_id: int, img_dirs: list[str]) -> Optional[str]:
        fname = f"{img_id}.jpg"
        for d in img_dirs:
            p = os.path.join(d, fname)
            if os.path.exists(p):
                return p
        return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        s   = self.samples[idx]
        img = Image.open(s["img_path"]).convert("RGB")
        W, H = img.size
        x, y, w, h = s["bbox"]
        bbox_rel = torch.tensor([
            max(0.0, x) / W,
            max(0.0, y) / H,
            min(float(W), x + w) / W,
            min(float(H), y + h) / H,
        ], dtype=torch.float32)
        return self.full_transform(img), bbox_rel, s["text"]


class GQASceneGraphDataset(Dataset):
    """
    GQA scene graphs データセット。

    train_sceneGraphs.json / val_sceneGraphs.json からオブジェクト名 + 属性を text に使用。
    """

    def __init__(
        self,
        gqa_root:      str           = GQA_ROOT,
        split:         str           = "train",
        full_img_size: int           = STEERVIT_IMG_SIZE,
        min_area:      int           = 32 * 32,
        min_side:      int           = 16,
        max_samples:   Optional[int] = None,
        seed:          int           = 42,
    ):
        self.image_dir      = os.path.join(gqa_root, "images")
        self.full_transform = _full_img_transform(full_img_size)

        sg_file = os.path.join(gqa_root, f"{split}_sceneGraphs.json")
        with open(sg_file, encoding="utf-8") as f:
            scene_graphs = json.load(f)

        samples = []
        for img_id, sg in scene_graphs.items():
            img_w = sg["width"]
            img_h = sg["height"]
            img_path = os.path.join(self.image_dir, f"{img_id}.jpg")
            for obj in sg["objects"].values():
                x, y = obj["x"], obj["y"]
                w, h = obj["w"], obj["h"]
                if w * h < min_area or w < min_side or h < min_side:
                    continue
                name  = obj["name"]
                attrs = obj.get("attributes", [])
                text  = f"{attrs[0]} {name}" if attrs else name
                samples.append({
                    "img_path": img_path,
                    "bbox":     [x, y, w, h],
                    "img_w":    img_w,
                    "img_h":    img_h,
                    "text":     text,
                })

        if max_samples is not None:
            samples = random.Random(seed).sample(samples, min(max_samples, len(samples)))

        self.samples = samples
        print(f"[GQASceneGraphDataset:{split}] {len(self.samples):,} サンプル")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        s    = self.samples[idx]
        img  = Image.open(s["img_path"]).convert("RGB")
        W, H = s["img_w"], s["img_h"]
        x, y, w, h = s["bbox"]
        bbox_rel = torch.tensor([
            max(0.0, x) / W,
            max(0.0, y) / H,
            min(float(W), x + w) / W,
            min(float(H), y + h) / H,
        ], dtype=torch.float32)
        return self.full_transform(img), bbox_rel, s["text"]


class _IndexedWrapper(Dataset):
    """Dataset wrapper that appends the dataset-level index to each item."""
    def __init__(self, base: Dataset):
        self.base = base
    def __len__(self) -> int:
        return len(self.base)
    def __getitem__(self, idx: int) -> tuple:
        return (*self.base[idx], idx)


COCO_TRAIN_ANN  = os.path.join(COCO_ROOT, "annotations/instances_train2017.json")
COCO_VAL_ANN    = os.path.join(COCO_ROOT, "annotations/instances_val2017.json")
COCO_TRAIN_IMGS = os.path.join(COCO_ROOT, "train2017")
COCO_VAL_IMGS   = os.path.join(COCO_ROOT, "val2017")

_DATASET_BUILDERS = {
    "refcoco": lambda split, kw: RefCOCOLocalDataset(
        refcoco_dir=REFCOCO_DIR, split=split, **kw),
    "refcoco+": lambda split, kw: RefCOCOLocalDataset(
        refcoco_dir=REFCOCO_PLUS_DIR, split=split, **kw),
    "refcocog": lambda split, kw: RefCOCOLocalDataset(
        refcoco_dir=REFCOCOG_DIR, split=split, **kw),
    "vg": lambda split, kw: VGRegionDataset(split=split, **kw),
    "gqa": lambda split, kw: GQASceneGraphDataset(split=split, **kw),
}

_ALL_DATASETS = list(_DATASET_BUILDERS.keys())


def build_loaders(
    dataset:       str           = "refcoco",
    full_img_size: int           = STEERVIT_IMG_SIZE,
    batch_size:    int           = 128,
    num_workers:   int           = 8,
    coco_root:     str           = COCO_ROOT,
    vg_root:       str           = VG_ROOT,
    gqa_root:      str           = GQA_ROOT,
    max_train:     Optional[int] = None,
    max_val:       Optional[int] = None,
    seed:          int           = 42,
    fix_text:      bool          = False,
    with_index:    bool          = False,
):
    from torch.utils.data import DataLoader

    refcoco_dir = os.path.join(coco_root, "refcoco")
    refcoco_plus_dir = os.path.join(coco_root, "refcoco+")
    refcocog_dir = os.path.join(coco_root, "refcocog")
    train_ann  = os.path.join(coco_root, "annotations/instances_train2017.json")
    train_imgs = os.path.join(coco_root, "train2017")
    val_ann    = os.path.join(coco_root, "annotations/instances_val2017.json")
    val_imgs   = os.path.join(coco_root, "val2017")

    common_kw  = dict(full_img_size=full_img_size, seed=seed)
    refcoco_kw = {**common_kw, "fix_text": fix_text}

    builders = {
        "refcoco":  lambda split, kw: RefCOCOLocalDataset(refcoco_dir=refcoco_dir,      coco_root=coco_root, split=split, **kw),
        "refcoco+": lambda split, kw: RefCOCOLocalDataset(refcoco_dir=refcoco_plus_dir, coco_root=coco_root, split=split, **kw),
        "refcocog": lambda split, kw: RefCOCOLocalDataset(refcoco_dir=refcocog_dir,     coco_root=coco_root, split=split, **kw),
        "vg":       lambda split, kw: VGRegionDataset(vg_root=vg_root,   split=split, **kw),
        "gqa":      lambda split, kw: GQASceneGraphDataset(gqa_root=gqa_root, split=split, **kw),
    }
    refcoco_builders = {"refcoco", "refcoco+", "refcocog"}

    def _build(name, split, max_s):
        kw = {**(refcoco_kw if name in refcoco_builders else common_kw), "max_samples": max_s}
        return builders[name](split, kw)

    if dataset == "coco":
        train_ds = COCODetDataset(ann_file=train_ann, image_dir=train_imgs, max_samples=max_train, **common_kw)
        val_ds   = COCODetDataset(ann_file=val_ann,   image_dir=val_imgs,   max_samples=max_val,   **common_kw)

    elif dataset == "all":
        all_names = list(builders.keys())
        per_train = (max_train // len(all_names)) if max_train else None
        per_val   = (max_val   // len(all_names)) if max_val   else None
        train_ds  = ConcatDataset([_build(n, "train", per_train) for n in all_names])
        val_ds    = ConcatDataset([_build(n, "val",   per_val)   for n in all_names])

    elif dataset in builders:
        train_ds = _build(dataset, "train", max_train)
        val_ds   = _build(dataset, "val",   max_val)

    else:
        raise ValueError(f"Unknown dataset: {dataset!r}. Choose from {list(builders.keys()) + ['coco', 'all']}")

    if with_index:
        train_ds = _IndexedWrapper(train_ds)

    def collate(batch):
        full_imgs, bbox_rels, texts = zip(*batch)
        return torch.stack(full_imgs), torch.stack(bbox_rels), list(texts)

    def collate_with_idx(batch):
        full_imgs, bbox_rels, texts, idxs = zip(*batch)
        return torch.stack(full_imgs), torch.stack(bbox_rels), list(texts), torch.tensor(idxs)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        collate_fn=collate_with_idx if with_index else collate,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        collate_fn=collate,
    )
    return train_loader, val_loader


if __name__ == "__main__":
    train_loader, val_loader = build_loaders(dataset="all", batch_size=4, max_train=50, max_val=10)
    full_imgs, bbox_rels, texts = next(iter(train_loader))
    print(f"full_imgs: {full_imgs.shape}")
    print(f"bbox_rels: {bbox_rels.shape}  sample={bbox_rels[0]}")
    print(f"texts:     {texts}")
