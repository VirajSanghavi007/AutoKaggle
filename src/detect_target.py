from pathlib import Path

import pandas as pd

_COMMON_TARGET_NAMES = [
    "target", "label", "labels", "class", "y", "outcome", "response",
    "survived", "saleprice", "price", "score", "result", "churn", "fraud",
]


def detect_target(train_csv_path: Path, data_dir: Path) -> str:
    """data_dir is unused now that competition submission-file heuristics are gone;
    kept in the signature for call-site compatibility."""
    df = pd.read_csv(train_csv_path, nrows=5)
    columns = list(df.columns)

    for name in _COMMON_TARGET_NAMES:
        for col in columns:
            if col.strip().lower() == name:
                return col

    return columns[-1]
