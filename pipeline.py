"""
================================================================================
 FiftyOne Visual Analysis Pipeline — Large-Scale CV Dataset (S3 + COCO)
================================================================================
 Author  : Senior ML Infra Engineer
 Target  : Instance segmentation, 97 classes, Mask2Former predictions
 Dataset : ~340K images, ~67GB, stored in AWS S3 (COCO format)
 Memory  : Designed for 8–16 GB RAM (lazy loading, subset-only)
================================================================================

Architecture Overview
---------------------
  S3 Bucket (340K images)
       │
       ▼
  [1] fetch_s3_subset()          ← random / stratified / metrics-based sampling
       │  Downloads only N images + slim annotation slice
       ▼
  ./subset/
    images/                      ← downloaded image files
    annotations.json             ← COCO subset (only sampled image IDs)
       │
       ▼
  [2] load_fiftyone_dataset()    ← FiftyOne COCO importer
       │  Polygons → raster masks handled by FiftyOne automatically
       ▼
  fo.Dataset("fashion_subset")
       │
       ▼
  [3] attach_predictions()       ← COCO-style or custom JSON predictions
       │  Adds sample["predictions"] with Detections / polylines
       ▼
  [4] attach_metrics()           ← per-sample scalar fields
       │  bfscore, boundary_precision, boundary_recall,
       │  fp_rate, object_size, bleeding_flag
       ▼
  [5] build_views()              ← filtered dataset views
       │
       ▼
  [6] fo.launch_app()            ← local FiftyOne UI (Mac M1 compatible)

Usage
-----
  # Random sample of 3 000 images
  python pipeline.py --sample-size 3000 --mode random

  # Stratified sample of 5 000 images
  python pipeline.py --sample-size 5000 --mode stratified

  # Metrics-driven sample (provide a JSON file of per-image scores)
  python pipeline.py --sample-size 2000 --mode metrics \
                     --metrics-json ./precomputed_metrics.json

  # Full options
  python pipeline.py --help
"""

from __future__ import annotations

import argparse
from decimal import Decimal
import json
import logging
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import boto3
import botocore
import fiftyone as fo
import fiftyone.utils.coco as fouc
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global configuration  (override via CLI or by editing these defaults)
# ---------------------------------------------------------------------------
DEFAULT_CONFIG: Dict[str, Any] = {
    # ── S3 ──────────────────────────────────────────────────────────────────
    "s3_bucket": "makeover.dataset",          # <── change this
    "s3_images_prefix": "training_data/images/train",                # prefix for image objects
    "s3_annotations_key": "training_data/annotations/instances_train.json",
    "aws_profile": None,                          # None → use env vars / IAM role
    "aws_region": "ap-south-1",
    # ── Sampling ────────────────────────────────────────────────────────────
    "sample_size": 3_000,
    "sampling_mode": "random",                    # "random" | "stratified" | "metrics"
    "use_full_dataset": False,                    # True -> skip sampling and use full local dataset
    "metrics_json": None,                         # path to precomputed metrics JSON
    "random_seed": 42,
    # ── Local paths ─────────────────────────────────────────────────────────
    "local_root": "./subset",
    "images_dir": "./subset/images",
    "annotations_path": "./subset/annotations.json",
    # ── FiftyOne ────────────────────────────────────────────────────────────
    "dataset_name": "fashion_subset",
    "predictions_json": None,                     # path to predictions JSON (optional)
    "fp_bleed_threshold": 0.25,                   # threshold for bleeding_flag
    "overwrite_dataset": True,
    # ── Perf ────────────────────────────────────────────────────────────────
    "download_workers": 8,                        # parallel S3 downloads (ThreadPool)
    "annotation_cache_mb": 512,                   # soft cap for annotation JSON
    "stream_annotations": True,                   # use low-memory streaming parser when possible
    "annotation_cache_dir": None,                 # None -> ~/.cache/fiftyone_pipeline
    "allow_full_load_fallback": False,            # unsafe on low-memory machines
}


# ============================================================================
# STEP 1 — S3 SUBSET SAMPLING
# ============================================================================

def _get_s3_client(cfg: Dict[str, Any]) -> Any:
    """Return a boto3 S3 client, using a named profile if requested."""
    session_kwargs = {}
    if cfg.get("aws_profile"):
        session_kwargs["profile_name"] = cfg["aws_profile"]
    session = boto3.Session(**session_kwargs, region_name=cfg.get("aws_region"))
    return session.client("s3")


def _load_coco_annotations(cfg: Dict[str, Any], s3: Any) -> Dict:
    """
    Stream the COCO annotations JSON from S3 without writing it to disk first.
    For very large annotation files (>500 MB) this still fits in RAM because
    we only keep the Python dict — raw bytes are released after json.loads().
    """
    bucket = cfg["s3_bucket"]
    key = cfg["s3_annotations_key"]
    log.info("Downloading annotations from s3://%s/%s …", bucket, key)

    obj = s3.get_object(Bucket=bucket, Key=key)
    raw = obj["Body"].read()
    log.info("Annotation file size: %.1f MB", len(raw) / 1e6)
    coco = json.loads(raw)
    del raw  # free the bytes immediately
    return coco


def _download_annotations_local(cfg: Dict[str, Any], s3: Any) -> Path:
    """Download COCO annotations once to local cache and return file path."""
    bucket = cfg["s3_bucket"]
    key = cfg["s3_annotations_key"]

    cache_root = cfg.get("annotation_cache_dir")
    if cache_root:
        cache_dir = Path(cache_root)
    else:
        cache_dir = Path.home() / ".cache" / "fiftyone_pipeline"

    cache_dir = cache_dir / bucket / Path(key).parent
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_path = cache_dir / Path(key).name

    # Validate cache by comparing local size with S3 object size.
    expected_size = s3.head_object(Bucket=bucket, Key=key)["ContentLength"]

    if local_path.exists() and local_path.stat().st_size == expected_size:
        log.info("Using cached annotations file at %s", local_path)
        return local_path

    if local_path.exists():
        log.warning(
            "Cached annotations file is incomplete/corrupt (%d != %d bytes); re-downloading.",
            local_path.stat().st_size,
            expected_size,
        )

    log.info("Downloading annotations file to %s …", local_path)
    s3.download_file(bucket, key, str(local_path))
    size_mb = local_path.stat().st_size / 1e6
    log.info("Cached annotations file size: %.1f MB", size_mb)
    return local_path


def _build_coco_subset_streaming(
    annotation_path: Path,
    selected_ids: List[int],
) -> Dict[str, Any]:
    """
    Build COCO subset with low memory by streaming the source JSON.
    Requires `ijson`.
    """
    import ijson

    id_set = set(selected_ids)

    with open(annotation_path, "rb") as f:
        info = next(ijson.items(f, "info", use_float=True), {})

    with open(annotation_path, "rb") as f:
        licenses = list(ijson.items(f, "licenses.item", use_float=True))

    with open(annotation_path, "rb") as f:
        categories = list(ijson.items(f, "categories.item", use_float=True))

    subset_images = []
    id_to_fname: Dict[int, str] = {}
    with open(annotation_path, "rb") as f:
        for img in ijson.items(f, "images.item", use_float=True):
            iid = int(img["id"])
            if iid in id_set:
                subset_images.append(img)
                id_to_fname[iid] = img["file_name"]

    subset_anns = []
    with open(annotation_path, "rb") as f:
        for ann in ijson.items(f, "annotations.item", use_float=True):
            if int(ann["image_id"]) in id_set:
                subset_anns.append(ann)

    log.info(
        "Subset(streamed): %d images, %d annotations",
        len(subset_images), len(subset_anns),
    )

    return {
        "info": info,
        "licenses": licenses,
        "categories": categories,
        "images": subset_images,
        "annotations": subset_anns,
        "_id_to_fname": id_to_fname,
    }


def _stream_image_ids(annotation_path: Path) -> List[int]:
    """Return all COCO image IDs by streaming JSON with `ijson`."""
    import ijson

    image_ids: List[int] = []
    with open(annotation_path, "rb") as f:
        for img in ijson.items(f, "images.item", use_float=True):
            image_ids.append(int(img["id"]))
    return image_ids


def _json_default_encoder(obj: Any) -> Any:
    """Fallback encoder for json.dump when streaming parser yields Decimal."""
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def _resolve_local_data_path(images_dir: str, ann_path: str) -> str:
    """
    Resolve the effective local data_path for COCO import.

    Some datasets store files under images/train or images/val while annotation
    file_name entries are relative to those subfolders.
    """
    root = Path(images_dir)
    if not root.exists() or not Path(ann_path).exists():
        return images_dir

    try:
        with open(ann_path) as f:
            ann = json.load(f)
    except Exception:
        return images_dir

    filenames = [img.get("file_name") for img in ann.get("images", []) if img.get("file_name")]
    if not filenames:
        return images_dir

    # Sample a handful of paths for a fast and robust check.
    sample = filenames[: min(25, len(filenames))]
    candidates = [
        root,
        root / "train",
        root / "val",
        root / "validation",
        root / "test",
    ]

    best = root
    best_hits = -1
    for c in candidates:
        hits = sum(1 for fn in sample if (c / fn).exists())
        if hits > best_hits:
            best_hits = hits
            best = c

    if best_hits > 0 and best != root:
        log.info("Adjusted local data_path to %s based on annotation file_name paths.", best)

    return str(best)


def _safe_sample_field(sample: fo.Sample, field_name: str, default: Any = None) -> Any:
    """Safely read an optional field from a FiftyOne sample."""
    try:
        return sample.get_field(field_name)
    except (AttributeError, KeyError):
        return default


def _sample_random(image_ids: List[int], n: int, seed: int) -> List[int]:
    rng = random.Random(seed)
    return rng.sample(image_ids, min(n, len(image_ids)))


def _sample_stratified(coco: Dict, n: int, seed: int) -> List[int]:
    """
    Stratified sampling: pick images such that each category is represented
    proportionally to its frequency in the full dataset.
    """
    log.info("Building category → image index for stratified sampling …")
    cat_to_images: Dict[int, set] = defaultdict(set)
    for ann in coco["annotations"]:
        cat_to_images[ann["category_id"]].add(ann["image_id"])

    # Weight each category by log(count) to avoid over-representing huge classes
    categories = list(cat_to_images.keys())
    weights = np.array([math.log1p(len(cat_to_images[c])) for c in categories])
    weights /= weights.sum()

    quota = {
        cat: max(1, int(round(w * n)))
        for cat, w in zip(categories, weights)
    }

    rng = random.Random(seed)
    selected: set = set()
    for cat, q in quota.items():
        pool = list(cat_to_images[cat] - selected)
        chosen = rng.sample(pool, min(q, len(pool)))
        selected.update(chosen)
        if len(selected) >= n:
            break

    # Top-up with random images if needed
    all_ids = [img["id"] for img in coco["images"]]
    remaining = list(set(all_ids) - selected)
    short = n - len(selected)
    if short > 0:
        selected.update(rng.sample(remaining, min(short, len(remaining))))

    return list(selected)[:n]


def _sample_by_metrics(
    image_ids: List[int],
    n: int,
    metrics_json: str,
    seed: int,
) -> List[int]:
    """
    Sample the N images with the LOWEST bfscore (hardest examples).
    Metrics JSON format: { "image_id": { "bfscore": 0.42, ... }, ... }
    Falls back to random if the file is missing.
    """
    if not metrics_json or not Path(metrics_json).exists():
        log.warning("metrics_json not found — falling back to random sampling.")
        return _sample_random(image_ids, n, seed)

    with open(metrics_json) as f:
        metrics = json.load(f)

    valid_ids = set(image_ids)

    # Sort ascending by bfscore (lower = harder = more interesting to inspect)
    scored = []
    for iid, v in metrics.items():
        try:
            image_id = int(iid)
        except (TypeError, ValueError):
            continue

        score = v.get("bfscore", 1.0) if isinstance(v, dict) else 1.0
        scored.append((image_id, score))

    scored.sort(key=lambda x: x[1])
    selected = [iid for iid, _ in scored if iid in valid_ids][:n]

    # Top-up from the remaining pool to honor requested sample size.
    if len(selected) < n:
        remaining = list(valid_ids - set(selected))
        rng = random.Random(seed)
        selected.extend(rng.sample(remaining, min(n - len(selected), len(remaining))))

    log.info("Metrics-based sampling: %d images selected.", len(selected))
    return selected


def _build_coco_subset(coco: Dict, selected_ids: List[int]) -> Dict:
    """
    Slice the full COCO dict down to only the selected image IDs.
    Returns a valid COCO dict (info, licenses, categories, images, annotations).
    """
    id_set = set(selected_ids)
    subset_images = [img for img in coco["images"] if img["id"] in id_set]

    # Build a fast lookup: image_id → file_name
    id_to_fname = {img["id"]: img["file_name"] for img in subset_images}

    subset_anns = [
        ann for ann in coco["annotations"]
        if ann["image_id"] in id_set
    ]

    log.info(
        "Subset: %d images, %d annotations (from %d / %d)",
        len(subset_images), len(subset_anns),
        len(selected_ids), len(coco["images"]),
    )

    return {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "categories": coco["categories"],
        "images": subset_images,
        "annotations": subset_anns,
        "_id_to_fname": id_to_fname,   # helper — stripped before writing to disk
    }


def _download_images(
    s3: Any,
    bucket: str,
    prefix: str,
    id_to_fname: Dict[int, str],
    images_dir: str,
    workers: int,
) -> None:
    """
    Download images from S3 in parallel using a ThreadPoolExecutor.
    Skips files that already exist locally (resume-safe).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    os.makedirs(images_dir, exist_ok=True)
    tasks = []
    for img_id, fname in id_to_fname.items():
        local_path = Path(images_dir) / Path(fname.lstrip("/"))
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if local_path.exists():
            continue  # already downloaded — resume-safe
        s3_key = prefix.rstrip("/") + "/" + fname.lstrip("/")
        tasks.append((s3_key, str(local_path)))

    if not tasks:
        log.info("All images already present locally — skipping download.")
        return

    log.info("Downloading %d images with %d workers …", len(tasks), workers)

    def _download(args):
        key, local = args
        try:
            s3.download_file(bucket, key, local)
            return True
        except botocore.exceptions.ClientError as e:
            log.warning("Failed to download %s: %s", key, e)
            return False

    ok = err = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_download, t): t for t in tasks}
        with tqdm(total=len(tasks), unit="img", desc="S3 download") as pbar:
            for fut in as_completed(futs):
                if fut.result():
                    ok += 1
                else:
                    err += 1
                pbar.update(1)

    log.info("Download complete: %d OK, %d errors.", ok, err)


def fetch_s3_subset(cfg: Dict[str, Any]) -> Dict:
    """
    Main entry point for Step 1.

    1. Connects to S3.
    2. Downloads and parses COCO annotations (JSON only — no images yet).
    3. Samples cfg['sample_size'] image IDs using the chosen strategy.
    4. Downloads only those images.
    5. Writes the subset COCO JSON to disk.

    Returns the subset COCO dict.
    """
    local_ann = cfg.get("local_annotations_json")
    local_img_dir = cfg.get("local_images_dir")
    full_dataset = bool(cfg.get("use_full_dataset", False))

    if full_dataset:
        if not (local_ann and local_img_dir):
            raise ValueError(
                "--full-dataset requires local dataset paths. Provide both "
                "--local-images-dir and --local-annotations-json"
            )

        ann_local_path = Path(local_ann)
        images_root = Path(local_img_dir)
        if not ann_local_path.exists():
            raise FileNotFoundError(f"Local annotations file not found: {ann_local_path}")
        if not images_root.exists():
            raise FileNotFoundError(f"Local images directory not found: {images_root}")

        cfg["annotations_path"] = str(ann_local_path)
        cfg["images_dir"] = str(images_root)

        log.info("Full-dataset mode enabled: sampling skipped.")
        log.info("  local_annotations_json=%s", cfg["annotations_path"])
        log.info("  local_images_dir=%s", cfg["images_dir"])
        return {}

    # Local mode: sample from a local COCO file and reuse local images directly.
    if local_ann and local_img_dir:
        log.info("Local mode detected: using local annotations and local images (no S3 downloads).")
        log.info("  local_annotations_json=%s", local_ann)
        log.info("  local_images_dir=%s", local_img_dir)

        ann_local_path = Path(local_ann)
        if not ann_local_path.exists():
            raise FileNotFoundError(f"Local annotations file not found: {ann_local_path}")

        images_root = Path(local_img_dir)
        if not images_root.exists():
            raise FileNotFoundError(f"Local images directory not found: {images_root}")

        use_streaming = bool(cfg.get("stream_annotations", True))
        coco = None
        all_ids: List[int] = []

        if use_streaming and cfg["sampling_mode"] in {"random", "metrics"}:
            try:
                all_ids = _stream_image_ids(ann_local_path)
                log.info("Discovered %d image IDs via streaming parser (local).", len(all_ids))
            except ModuleNotFoundError:
                if cfg.get("allow_full_load_fallback", False):
                    log.warning("ijson is not installed; falling back to full in-memory JSON load.")
                    with open(ann_local_path) as f:
                        coco = json.load(f)
                    all_ids = [img["id"] for img in coco["images"]]
                else:
                    raise RuntimeError(
                        "ijson is required for low-memory streaming mode. "
                        f"Install it in this interpreter: {sys.executable} -m pip install ijson"
                    )
        else:
            with open(ann_local_path) as f:
                coco = json.load(f)
            all_ids = [img["id"] for img in coco["images"]]

        n = cfg["sample_size"]
        mode = cfg["sampling_mode"]
        seed = cfg["random_seed"]

        log.info("Sampling mode: %s | target: %d images", mode, n)

        if mode == "random":
            selected_ids = _sample_random(all_ids, n, seed)
        elif mode == "stratified":
            if coco is None:
                with open(ann_local_path) as f:
                    coco = json.load(f)
            selected_ids = _sample_stratified(coco, n, seed)
        elif mode == "metrics":
            selected_ids = _sample_by_metrics(all_ids, n, cfg.get("metrics_json"), seed)
        else:
            raise ValueError(f"Unknown sampling_mode: {mode!r}")

        if coco is None:
            subset = _build_coco_subset_streaming(ann_local_path, selected_ids)
        else:
            subset = _build_coco_subset(coco, selected_ids)

        ann_path = cfg["annotations_path"]
        os.makedirs(Path(ann_path).parent, exist_ok=True)
        disk_subset = {k: v for k, v in subset.items() if not k.startswith("_")}
        with open(ann_path, "w") as f:
            json.dump(disk_subset, f, default=_json_default_encoder)
        log.info("Subset annotations written to %s", ann_path)

        # Ensure downstream loader points to the provided local images.
        cfg["images_dir"] = str(images_root)
        log.info("Local mode enabled: skipping S3 image download.")
        return subset

    s3 = _get_s3_client(cfg)
    use_streaming = bool(cfg.get("stream_annotations", True))
    coco = None
    all_ids: List[int] = []
    ann_local_path: Optional[Path] = None

    if use_streaming and cfg["sampling_mode"] in {"random", "metrics"}:
        try:
            ann_local_path = _download_annotations_local(cfg, s3)
            all_ids = _stream_image_ids(ann_local_path)
            log.info("Discovered %d image IDs via streaming parser.", len(all_ids))
        except ModuleNotFoundError:
            if cfg.get("allow_full_load_fallback", False):
                log.warning("ijson is not installed; falling back to full in-memory JSON load.")
                coco = _load_coco_annotations(cfg, s3)
                all_ids = [img["id"] for img in coco["images"]]
            else:
                raise RuntimeError(
                    "ijson is required for low-memory streaming mode. "
                    f"Install it in this interpreter: {sys.executable} -m pip install ijson"
                )
    else:
        coco = _load_coco_annotations(cfg, s3)
        all_ids = [img["id"] for img in coco["images"]]

    n = cfg["sample_size"]
    mode = cfg["sampling_mode"]
    seed = cfg["random_seed"]

    log.info("Sampling mode: %s | target: %d images", mode, n)

    if mode == "random":
        selected_ids = _sample_random(all_ids, n, seed)
    elif mode == "stratified":
        if coco is None:
            coco = _load_coco_annotations(cfg, s3)
        selected_ids = _sample_stratified(coco, n, seed)
    elif mode == "metrics":
        selected_ids = _sample_by_metrics(all_ids, n, cfg.get("metrics_json"), seed)
    else:
        raise ValueError(f"Unknown sampling_mode: {mode!r}")

    if ann_local_path is not None and coco is None:
        subset = _build_coco_subset_streaming(ann_local_path, selected_ids)
    else:
        subset = _build_coco_subset(coco, selected_ids)
        del coco  # free full annotation dict (~GB scale) before downloading

    # ── Write subset annotations to disk ────────────────────────────────────
    ann_path = cfg["annotations_path"]
    os.makedirs(Path(ann_path).parent, exist_ok=True)
    disk_subset = {k: v for k, v in subset.items() if not k.startswith("_")}
    with open(ann_path, "w") as f:
        json.dump(disk_subset, f, default=_json_default_encoder)
    log.info("Subset annotations written to %s", ann_path)

    # ── Download images ──────────────────────────────────────────────────────
    _download_images(
        s3=s3,
        bucket=cfg["s3_bucket"],
        prefix=cfg["s3_images_prefix"],
        id_to_fname=subset["_id_to_fname"],
        images_dir=cfg["images_dir"],
        workers=cfg["download_workers"],
    )

    return subset


# ============================================================================
# STEP 2 — LOAD INTO FIFTYONE
# ============================================================================

def load_fiftyone_dataset(cfg: Dict[str, Any]) -> fo.Dataset:
    """
    Load the downloaded subset into a FiftyOne dataset using the COCO importer.

    FiftyOne's COCODetectionDataset importer handles:
    - polygon → rle mask conversion lazily (only when rendered)
    - label mapping from category_id → category name
    - bounding boxes + segmentation in a single pass
    """
    name = cfg["dataset_name"]
    ann_path = cfg["annotations_path"]
    images_dir = cfg["images_dir"]

    if cfg.get("local_annotations_json") and cfg.get("local_images_dir"):
        images_dir = _resolve_local_data_path(images_dir, ann_path)
        cfg["images_dir"] = images_dir

    # Delete existing dataset with the same name if requested
    if cfg.get("overwrite_dataset") and fo.dataset_exists(name):
        log.info("Deleting existing dataset '%s' …", name)
        fo.delete_dataset(name)

    log.info("Loading FiftyOne dataset '%s' from %s …", name, images_dir)

    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=images_dir,
        labels_path=ann_path,
        name=name,
        label_types=["detections", "segmentations"],  # load both boxes + masks
        # include_id=True keeps COCO image_id on each sample for prediction joins
        include_id=True,
        # use_polylines=False → store masks as raster (memory-efficient for FO UI)
        use_polylines=False,
        # Only load what's in our annotations (ignore unmatched images in dir)
        label_field="ground_truth",
    )

    dataset.persistent = True
    log.info("Dataset loaded: %d samples.", len(dataset))
    return dataset


# ============================================================================
# STEP 3 — ATTACH MODEL PREDICTIONS
# ============================================================================

def _load_predictions_json(predictions_json: str) -> Dict[Union[int, str], List[Dict[str, Any]]]:
    """
    Load predictions from disk.
    Accepts two formats:
      A) Standard COCO results JSON  — list of dicts with keys:
         { image_id, category_id, bbox, score, segmentation }
      B) Custom dict keyed by image filename or image_id:
         { "000001.jpg": [ { label, bbox, score, mask_rle } ] }

    Returns a dict mapping image_id or filename key → list of raw prediction dicts.
    """
    with open(predictions_json) as f:
        raw = json.load(f)

    preds: Dict[Union[int, str], List[Dict[str, Any]]] = defaultdict(list)

    if isinstance(raw, list):
        # COCO results format
        for p in raw:
            preds[int(p["image_id"])].append(p)
    elif isinstance(raw, dict):
        # Custom keyed format — values should be lists
        for k, v in raw.items():
            try:
                preds[int(k)].extend(v)
            except (TypeError, ValueError):
                # key is a filename — store as-is; resolved later via filename→id map
                preds[k].extend(v)
    else:
        raise ValueError("Unrecognised predictions JSON format.")

    return preds


def _coco_pred_to_fo_detection(pred: Dict, img_w: int, img_h: int, id_to_label: Dict) -> fo.Detection:
    """
    Convert a single COCO-format prediction dict into a fo.Detection
    with an attached segmentation mask.
    """
    label = id_to_label.get(pred.get("category_id", 0), "unknown")
    confidence = pred.get("score", None)

    # Bounding box: COCO uses [x, y, w, h] in absolute pixels
    bx, by, bw, bh = pred.get("bbox", [0, 0, 1, 1])
    # FiftyOne expects [x1, y1, w, h] relative [0, 1]
    rel_box = [bx / img_w, by / img_h, bw / img_w, bh / img_h]

    # Segmentation mask — handle polygon or RLE
    mask = None
    seg = pred.get("segmentation")
    if seg:
        try:
            from pycocotools import mask as coco_mask_utils
            if isinstance(seg, dict):                          # RLE
                rle = coco_mask_utils.frPyObjects(seg, img_h, img_w)
                binary = coco_mask_utils.decode(rle).astype(bool)
            else:                                              # polygon list
                rles = coco_mask_utils.frPyObjects(seg, img_h, img_w)
                rle = coco_mask_utils.merge(rles)
                binary = coco_mask_utils.decode(rle).astype(bool)

            # Crop to bounding box for storage efficiency
            x1, y1 = int(bx), int(by)
            x2, y2 = int(bx + bw), int(by + bh)
            mask = binary[y1:y2, x1:x2]
        except Exception as e:
            log.debug("Mask decode failed for pred: %s", e)

    return fo.Detection(
        label=label,
        bounding_box=rel_box,
        confidence=confidence,
        mask=mask,
    )


def attach_predictions(
    dataset: fo.Dataset,
    cfg: Dict[str, Any],
) -> None:
    """
    Attach Mask2Former (or any COCO-style) predictions to each sample.

    Predictions are stored in sample["predictions"] as a fo.Detections object.
    If no predictions JSON is provided, the field is left empty (None).
    """
    predictions_json = cfg.get("predictions_json")
    if not predictions_json:
        log.info("No predictions_json provided — skipping prediction attachment.")
        return

    if not Path(predictions_json).exists():
        log.warning("predictions_json not found at %s — skipping.", predictions_json)
        return

    log.info("Loading predictions from %s …", predictions_json)
    preds_by_id = _load_predictions_json(predictions_json)

    # Build category_id → label name map from the dataset schema
    id_to_label: Dict[int, str] = {}
    ann_path = cfg["annotations_path"]
    coco_id_to_fname: Dict[int, str] = {}
    if Path(ann_path).exists():
        with open(ann_path) as f:
            coco_ann = json.load(f)
        id_to_label = {c["id"]: c["name"] for c in coco_ann.get("categories", [])}
        coco_id_to_fname = {img["id"]: img["file_name"] for img in coco_ann.get("images", [])}

    log.info("Attaching predictions to %d samples …", len(dataset))

    # Iterate in batches to keep memory low
    with dataset.save_context() as ctx:
        for sample in tqdm(dataset, desc="Attaching predictions", unit="sample"):
            coco_id = _safe_sample_field(sample, "coco_id")   # stored by FO COCO importer
            if coco_id is None:
                # Fallback: derive from filename
                coco_id = int(Path(sample.filepath).stem.lstrip("0") or "0")

            raw_preds = preds_by_id.get(int(coco_id), [])
            if not raw_preds:
                key_candidates = []

                ann_fname = coco_id_to_fname.get(int(coco_id))
                if ann_fname:
                    key_candidates.extend([
                        ann_fname,
                        Path(ann_fname).name,
                        Path(ann_fname).stem,
                    ])

                sample_name = Path(sample.filepath).name
                key_candidates.extend([sample_name, Path(sample_name).stem])

                for key in key_candidates:
                    raw_preds = preds_by_id.get(key, [])
                    if raw_preds:
                        break

            if not raw_preds:
                sample["predictions"] = None
                ctx.save(sample)
                continue

            img_w = sample.metadata.width if sample.metadata else 640
            img_h = sample.metadata.height if sample.metadata else 480

            detections = [
                _coco_pred_to_fo_detection(p, img_w, img_h, id_to_label)
                for p in raw_preds
            ]
            sample["predictions"] = fo.Detections(detections=detections)
            ctx.save(sample)

    log.info("Predictions attached.")


# ============================================================================
# STEP 4 — CUSTOM METRICS
# ============================================================================

# ---------- Placeholder computation functions --------------------------------

def _compute_bfscore_placeholder(sample) -> float:
    """
    Boundary F-score placeholder.
    Real implementation: compare predicted boundary pixels vs GT boundary pixels
    using morphological dilation (see Csurka et al., 2013).
    Returns a random value in a realistic range for demonstration.
    """
    return float(np.random.uniform(0.2, 0.95))


def _compute_boundary_precision_placeholder(sample) -> float:
    return float(np.random.uniform(0.1, 0.99))


def _compute_boundary_recall_placeholder(sample) -> float:
    return float(np.random.uniform(0.1, 0.99))


def _compute_fp_rate_placeholder(sample) -> float:
    """False-positive rate: FP / (FP + TN)."""
    return float(np.random.uniform(0.0, 0.4))


def _compute_object_size(sample) -> float:
    """
    object_size = mean(annotation_area) / image_area.
    Uses ground_truth detections if available; falls back to 0.
    """
    gt = _safe_sample_field(sample, "ground_truth")
    if gt is None or not gt.detections:
        return 0.0

    meta = sample.metadata
    if meta is None:
        return 0.0

    img_area = meta.width * meta.height
    if img_area == 0:
        return 0.0

    # COCO bounding box area (relative → absolute)
    areas = []
    for det in gt.detections:
        _, _, bw, bh = det.bounding_box
        areas.append(bw * bh * img_area)

    return float(np.mean(areas) / img_area) if areas else 0.0


def _compute_bleeding_flag(fp_rate: Optional[float], fp_rate_threshold: float = 0.25) -> bool:
    """
    bleeding_flag is True when the prediction 'bleeds' outside the ground-truth
    region — approximated here by a high fp_rate.
    Replace with a real pixel-level overlap check in production.
    """
    if fp_rate is None:
        return False
    return bool(fp_rate > fp_rate_threshold)


# ---------- Main metrics attachment function ---------------------------------

def attach_metrics(
    dataset: fo.Dataset,
    metrics_json: Optional[str] = None,
    fp_bleed_threshold: float = 0.25,
) -> None:
    """
    Attach per-sample metrics as scalar fields on the FiftyOne dataset.

    If `metrics_json` is provided (format: { image_id: { metric: value } }),
    values are read from it; otherwise placeholder functions are used.

    Fields added:
        bfscore              float   [0, 1]
        boundary_precision   float   [0, 1]
        boundary_recall      float   [0, 1]
        fp_rate              float   [0, 1]
        object_size          float   [0, 1]   (relative to image area)
        bleeding_flag        bool
    """
    precomputed: Dict[int, Dict] = {}
    if metrics_json and Path(metrics_json).exists():
        with open(metrics_json) as f:
            raw = json.load(f)
        precomputed = {int(k): v for k, v in raw.items()}
        log.info("Loaded precomputed metrics for %d images.", len(precomputed))
    else:
        log.info("No metrics JSON found — using placeholder metric functions.")

    log.info("Attaching metrics to %d samples …", len(dataset))

    with dataset.save_context() as ctx:
        for sample in tqdm(dataset, desc="Attaching metrics", unit="sample"):
            coco_id = _safe_sample_field(sample, "coco_id", 0) or 0

            if precomputed:
                m = precomputed.get(int(coco_id), {})
                bfscore  = float(m.get("bfscore",            _compute_bfscore_placeholder(sample)))
                bp       = float(m.get("boundary_precision", _compute_boundary_precision_placeholder(sample)))
                br       = float(m.get("boundary_recall",    _compute_boundary_recall_placeholder(sample)))
                fpr      = float(m.get("fp_rate",            _compute_fp_rate_placeholder(sample)))
                os_      = float(m.get("object_size",        _compute_object_size(sample)))
                bleed    = bool(m.get("bleeding_flag",       _compute_bleeding_flag(fpr, fp_bleed_threshold)))
            else:
                bfscore  = _compute_bfscore_placeholder(sample)
                bp       = _compute_boundary_precision_placeholder(sample)
                br       = _compute_boundary_recall_placeholder(sample)
                fpr      = _compute_fp_rate_placeholder(sample)
                os_      = _compute_object_size(sample)
                bleed    = _compute_bleeding_flag(fpr, fp_bleed_threshold)

            sample["bfscore"]            = bfscore
            sample["boundary_precision"] = bp
            sample["boundary_recall"]    = br
            sample["fp_rate"]            = fpr
            sample["object_size"]        = os_
            sample["bleeding_flag"]      = bleed
            ctx.save(sample)

    dataset.add_dynamic_sample_fields()  # register new fields in the schema
    log.info("Metrics attached.")


# ============================================================================
# STEP 5 — POWERFUL VIEWS
# ============================================================================

def build_views(dataset: fo.Dataset) -> Dict[str, fo.DatasetView]:
    """
    Return a dict of named dataset views for common diagnostic scenarios.
    All views are lazy — no data is loaded until iterated or opened in the UI.
    """
    views = {}

    # ── Low boundary precision ───────────────────────────────────────────────
    views["low_precision"] = dataset.match(
        fo.ViewField("boundary_precision") < 0.3
    )

    # ── Bleeding artifacts ───────────────────────────────────────────────────
    views["bleeding_cases"] = dataset.match(
        fo.ViewField("bleeding_flag") == True
    )

    # ── Small objects ────────────────────────────────────────────────────────
    views["small_objects"] = dataset.match(
        fo.ViewField("object_size") < 0.05
    )

    # ── Combined: low precision AND small objects ────────────────────────────
    views["low_precision_small"] = dataset.match(
        (fo.ViewField("boundary_precision") < 0.3)
        & (fo.ViewField("object_size") < 0.05)
    )

    # ── High fp_rate (likely false positives) ────────────────────────────────
    views["high_fp_rate"] = dataset.match(
        fo.ViewField("fp_rate") > 0.3
    )

    # ── Worst performers (bottom 10 % by bfscore) ────────────────────────────
    n_worst = max(1, len(dataset) // 10)
    views["worst_bfscore"] = (
        dataset.sort_by("bfscore", reverse=False)
               .limit(n_worst)
    )

    for view_name, view in views.items():
        log.info("View '%s': %d samples.", view_name, len(view))

    return views


# ============================================================================
# STEP 6 — LAUNCH FIFTYONE UI
# ============================================================================

def launch_fiftyone_ui(
    dataset: fo.Dataset,
    views: Dict[str, fo.DatasetView],
    host: str = "0.0.0.0",
    port: int = 5151,
) -> fo.Session:
    """
    Launch the FiftyOne App locally.
    Mac M1 compatible — FiftyOne uses a Flask/Electron server that runs natively
    on arm64 as of v0.21+.

    The session is returned so the caller can keep it alive.
    """
    log.info("Launching FiftyOne App on %s:%d …", host, port)

    session = fo.launch_app(
        dataset=dataset,
        address=host,
        port=port,
        remote=True,   # remote=True → prints URL even in headless/SSH envs
        auto=False,    # don't try to open a browser automatically
    )

    url = f"http://{host}:{port}"
    print("\n" + "=" * 60)
    print(f"  FiftyOne App running at: {url}")
    print("=" * 60)
    print("\nAvailable views (load via session.view = views['<name>']):\n")
    for name in views:
        print(f"  • {name}")
    print()
    print("To switch view in the App:\n  session.view = views['bleeding_cases']")
    print("To stop the server:\n  session.close()\n")

    return session


# ============================================================================
# ORCHESTRATION
# ============================================================================

def run_pipeline(cfg: Dict[str, Any]) -> fo.Session:
    """Execute all pipeline steps in sequence."""
    t0 = time.perf_counter()

    # ── 1. Fetch subset from S3 ──────────────────────────────────────────────
    log.info("━━ Step 1: Subset Sampling (S3 or Local) ━━━━━━━━━━━━━━━━━━━")
    fetch_s3_subset(cfg)

    # ── 2. Load into FiftyOne ────────────────────────────────────────────────
    log.info("━━ Step 2: Load into FiftyOne ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    dataset = load_fiftyone_dataset(cfg)
    dataset.compute_metadata()  # populates width/height — needed for object_size

    # ── 3. Attach predictions ────────────────────────────────────────────────
    log.info("━━ Step 3: Attach Model Predictions ━━━━━━━━━━━━━━━━━━━━━━━━")
    attach_predictions(dataset, cfg)

    # ── 4. Attach metrics ────────────────────────────────────────────────────
    log.info("━━ Step 4: Attach Custom Metrics ━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    attach_metrics(
        dataset,
        metrics_json=cfg.get("metrics_json"),
        fp_bleed_threshold=cfg.get("fp_bleed_threshold", 0.25),
    )

    # ── 5. Build views ───────────────────────────────────────────────────────
    log.info("━━ Step 5: Build Diagnostic Views ━━━━━━━━━━━━━━━━━━━━━━━━━━")
    views = build_views(dataset)

    # ── 6. Launch UI ─────────────────────────────────────────────────────────
    log.info("━━ Step 6: Launch FiftyOne UI ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    session = launch_fiftyone_ui(
        dataset,
        views,
        host=cfg.get("host", "0.0.0.0"),
        port=cfg.get("port", 5151),
    )

    elapsed = time.perf_counter() - t0
    log.info("Pipeline completed in %.1f s.", elapsed)
    return session


# ============================================================================
# CLI
# ============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FiftyOne Visual Analysis Pipeline for large-scale S3/COCO CV datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--bucket",           default=DEFAULT_CONFIG["s3_bucket"],
                   help="S3 bucket name")
    p.add_argument("--images-prefix",    default=DEFAULT_CONFIG["s3_images_prefix"],
                   help="S3 prefix for image objects")
    p.add_argument("--annotations-key",  default=DEFAULT_CONFIG["s3_annotations_key"],
                   help="S3 key for the COCO annotations JSON")
    p.add_argument("--sample-size",      type=int, default=DEFAULT_CONFIG["sample_size"],
                   help="Number of images to sample (2K–10K)")
    p.add_argument("--mode",             choices=["random", "stratified", "metrics"],
                   default=DEFAULT_CONFIG["sampling_mode"],
                   help="Sampling strategy")
    p.add_argument("--full-dataset",     action="store_true",
                   help="Load full local dataset without subset sampling")
    p.add_argument("--metrics-json",     default=None,
                   help="Path to precomputed metrics JSON")
    p.add_argument("--predictions-json", default=None,
                   help="Path to COCO-style predictions JSON")
    p.add_argument("--fp-bleed-threshold", type=float, default=DEFAULT_CONFIG["fp_bleed_threshold"],
                   help="Threshold for setting bleeding_flag from fp_rate")
    p.add_argument("--output-dir",       default=DEFAULT_CONFIG["local_root"],
                   help="Local directory for downloaded subset")
    p.add_argument("--dataset-name",     default=DEFAULT_CONFIG["dataset_name"],
                   help="FiftyOne dataset name")
    p.add_argument("--port",             type=int, default=5151,
                   help="FiftyOne App port")
    p.add_argument("--host",             default="0.0.0.0",
                   help="FiftyOne bind host (use 0.0.0.0 for remote VM access)")
    p.add_argument("--workers",          type=int, default=DEFAULT_CONFIG["download_workers"],
                   help="Parallel S3 download workers")
    p.add_argument("--seed",             type=int, default=DEFAULT_CONFIG["random_seed"],
                   help="Random seed")
    p.add_argument("--aws-profile",      default=None,
                   help="AWS named profile (optional)")
    p.add_argument("--local-images-dir", default=None,
                   help="Local images directory (enables local mode with --local-annotations-json)")
    p.add_argument("--local-annotations-json", default=None,
                   help="Local COCO annotations JSON path (enables local mode with --local-images-dir)")
    p.add_argument("--no-stream-annotations", action="store_true",
                   help="Disable low-memory streaming parser for annotations")
    p.add_argument("--annotation-cache-dir", default=None,
                   help="Directory to cache large annotations JSON (default: ~/.cache/fiftyone_pipeline)")
    p.add_argument("--allow-full-load-fallback", action="store_true",
                   help="Allow fallback to full in-memory annotation load when ijson is unavailable (may OOM)")
    p.add_argument("--no-overwrite",     action="store_true",
                   help="Do not overwrite existing FiftyOne dataset")
    return p.parse_args()


def _args_to_cfg(args: argparse.Namespace) -> Dict[str, Any]:
    root = args.output_dir
    cfg = {**DEFAULT_CONFIG}
    cfg.update({
        "s3_bucket":          args.bucket,
        "s3_images_prefix":   args.images_prefix,
        "s3_annotations_key": args.annotations_key,
        "sample_size":        args.sample_size,
        "sampling_mode":      args.mode,
        "use_full_dataset":   args.full_dataset,
        "metrics_json":       args.metrics_json,
        "predictions_json":   args.predictions_json,
        "fp_bleed_threshold": args.fp_bleed_threshold,
        "local_root":         root,
        "images_dir":         str(Path(root) / "images"),
        "annotations_path":   str(Path(root) / "annotations.json"),
        "dataset_name":       args.dataset_name,
        "host":               args.host,
        "port":               args.port,
        "download_workers":   args.workers,
        "random_seed":        args.seed,
        "aws_profile":        args.aws_profile,
        "local_images_dir":   args.local_images_dir,
        "local_annotations_json": args.local_annotations_json,
        "stream_annotations": not args.no_stream_annotations,
        "annotation_cache_dir": args.annotation_cache_dir,
        "allow_full_load_fallback": args.allow_full_load_fallback,
        "overwrite_dataset":  not args.no_overwrite,
    })
    return cfg


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    args = _parse_args()
    cfg = _args_to_cfg(args)

    log.info("Config: sample_size=%d | mode=%s | bucket=%s",
             cfg["sample_size"], cfg["sampling_mode"], cfg["s3_bucket"])

    session = run_pipeline(cfg)

    # Keep the process alive so the FiftyOne UI stays up
    try:
        log.info("Press Ctrl+C to stop the FiftyOne App.")
        session.wait()
    except KeyboardInterrupt:
        log.info("Shutting down.")
        session.close()
        sys.exit(0)
