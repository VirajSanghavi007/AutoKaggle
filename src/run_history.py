import json
from datetime import datetime
from pathlib import Path

HISTORY_PATH = Path(__file__).resolve().parent.parent / "data" / "run_history.jsonl"


def append_run_record(target_slug: str, best_name: str, cv_score: float, baseline_score: float, status: str):
    HISTORY_PATH.parent.mkdir(exist_ok=True)
    improvement_ratio = None
    if cv_score not in (None, 0) and baseline_score is not None:
        try:
            improvement_ratio = round(baseline_score / cv_score, 4) if cv_score != 0 else None
        except Exception:
            improvement_ratio = None

    record = {
        "timestamp": datetime.now().isoformat(),
        "target_slug": target_slug,
        "best_model": best_name,
        "cv_score": cv_score,
        "baseline_score": baseline_score,
        "improvement_ratio": improvement_ratio,
        "status": status,
    }
    with open(HISTORY_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
