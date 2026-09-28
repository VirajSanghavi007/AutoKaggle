import hashlib
import json
import logging
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from retry import retry
from schema import NEGATED_METRICS, METRIC_DISPLAY_NAMES

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config" / "config.json"
CURRENT_RUN_PATH = BASE_DIR / "config" / "current_run.json"
DATA_DIR = BASE_DIR / "data"
NOTEBOOK_DIR = BASE_DIR / "notebooks"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
LOG_DIR = BASE_DIR / "logs"
SLUG_REGISTRY_PATH = NOTEBOOK_DIR / "slug_registry.json"

EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EXIT_NOTHING_TO_DO = 2

LEAKAGE_CORR_THRESHOLD = 0.98


def setup_logging(name: str) -> logging.Logger:
    LOG_DIR.mkdir(exist_ok=True)
    log_file = LOG_DIR / f"{name}_{datetime.now():%Y%m%d}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(name)


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"config.json not found at {CONFIG_PATH}")
    return json.loads(CONFIG_PATH.read_text())


def dataset_display_name(target_slug: str) -> str:
    """Best-effort human title from a Kaggle dataset slug, e.g.
    'yasserh/housing-prices-dataset' -> 'Housing Prices Dataset'. This is a heuristic
    (title-cased, hyphens/underscores to spaces) rather than a Kaggle API lookup of the
    dataset's real display title, so it can occasionally diverge from the exact title
    Kaggle shows - acceptable for a kernel title, not used for anything else."""
    name_part = target_slug.split("/")[-1]
    words = re.split(r"[-_]+", name_part)
    return " ".join(w.capitalize() for w in words if w)


def format_kernel_title(target_slug: str, metric: str, raw_score: float) -> str:
    """Required published-kernel title format: '<Dataset Name> (<METRIC> <value>)',
    e.g. 'Housing Prices Dataset (RMSE 1050295.64)'. raw_score is pipeline_builder's
    internal lower-is-better value - flip the sign back for metrics in NEGATED_METRICS
    before displaying, or a good ROC AUC/F1/accuracy would show as a negative number."""
    display_value = -raw_score if metric in NEGATED_METRICS else raw_score
    metric_label = METRIC_DISPLAY_NAMES.get(metric, metric.upper())
    return f"{dataset_display_name(target_slug)} ({metric_label} {display_value:.2f})"


def sanitize_kernel_slug(target_slug: str) -> str:
    slug = target_slug.split("/")[-1]
    slug = re.sub(r"[^a-zA-Z0-9\-]", "-", slug).lower().strip("-")
    return f"{slug}-autokaggle"


def resolve_kernel_slug(target_slug: str) -> str:
    base_slug = sanitize_kernel_slug(target_slug)
    registry = {}
    if SLUG_REGISTRY_PATH.exists():
        registry = json.loads(SLUG_REGISTRY_PATH.read_text())

    existing_owner = registry.get(base_slug)
    if existing_owner is None or existing_owner == target_slug:
        registry[base_slug] = target_slug
        NOTEBOOK_DIR.mkdir(exist_ok=True)
        SLUG_REGISTRY_PATH.write_text(json.dumps(registry, indent=2))
        return base_slug

    disambiguated = f"{base_slug}-{hashlib.sha1(target_slug.encode()).hexdigest()[:6]}"
    registry[disambiguated] = target_slug
    NOTEBOOK_DIR.mkdir(exist_ok=True)
    SLUG_REGISTRY_PATH.write_text(json.dumps(registry, indent=2))
    return disambiguated


def download_dataset(job: dict, job_data_dir: Path, log) -> Path:
    job_data_dir.mkdir(parents=True, exist_ok=True)
    slug = job["target_slug"]
    cmd = ["kaggle", "datasets", "download", "-d", slug, "-p", str(job_data_dir)]

    def _do():
        log.info("Downloading via: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"kaggle download failed for {slug}: {result.stderr}")
        log.info(result.stdout)

    retry(_do, log=log, what=f"kaggle download {slug}")

    for zip_file in job_data_dir.glob("*.zip"):
        subprocess.run(["tar", "-xf", str(zip_file), "-C", str(job_data_dir)], check=False)

    csv_files = sorted(job_data_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not csv_files:
        raise RuntimeError(f"no CSV found after download/unzip for {slug}")
    train_candidates = [p for p in csv_files if p.stem.lower() == "train"]
    return train_candidates[0] if train_candidates else csv_files[0]


def apply_leakage_backstop(df: pd.DataFrame, pipeline_config, log) -> list[str]:
    dropped = []
    target = pipeline_config.target
    if target not in df.columns or not pd.api.types.is_numeric_dtype(df[target]):
        return dropped
    flagged_names = {d.get("name") for d in pipeline_config.leakage_risk_columns if d.get("action") == "drop"}
    for col in list(pipeline_config.features.keys()):
        if col in flagged_names or col not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        try:
            corr = abs(float(df[col].corr(df[target])))
        except Exception:
            continue
        if corr is not None and not np.isnan(corr) and corr > LEAKAGE_CORR_THRESHOLD:
            dropped.append(col)
            del pipeline_config.features[col]
            log.warning(
                "Leakage backstop: dropped '%s' (|corr with target|=%.4f > %.2f), not caught by LLM",
                col, corr, LEAKAGE_CORR_THRESHOLD,
            )
    return dropped
