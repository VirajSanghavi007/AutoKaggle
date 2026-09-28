import json
import numpy as np
import pandas as pd


def _missingness_type(df: pd.DataFrame, col: str, target: str) -> str:
    if df[col].isna().sum() == 0:
        return "unknown"
    indicator = df[col].isna().astype(int)
    if indicator.nunique() < 2:
        return "unknown"

    correlated_with_others = False
    for other in df.columns:
        if other in (col, target):
            continue
        if not pd.api.types.is_numeric_dtype(df[other]):
            continue
        try:
            corr = indicator.corr(df[other].fillna(df[other].median()))
        except Exception:
            continue
        if corr is not None and not np.isnan(corr) and abs(corr) > 0.3:
            correlated_with_others = True
            break

    correlated_with_target = False
    if target in df.columns and pd.api.types.is_numeric_dtype(df[target]):
        try:
            corr_t = indicator.corr(df[target])
            if corr_t is not None and not np.isnan(corr_t) and abs(corr_t) > 0.3:
                correlated_with_target = True
        except Exception:
            pass

    if correlated_with_target:
        return "MNAR"
    if correlated_with_others:
        return "MAR"
    return "MCAR"


def run_eda(csv_path: str, target: str) -> dict:
    df = pd.read_csv(csv_path)
    n_rows, n_cols = df.shape

    summary = {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "target": target,
        "target_dtype": str(df[target].dtype) if target in df.columns else None,
        "features": {},
    }

    if target in df.columns and _is_discrete_dtype(df[target]):
        vc = df[target].value_counts(normalize=True)
        if len(vc) <= 20:
            summary["target_value_counts"] = vc.to_dict()
            if len(vc) >= 2:
                summary["imbalance_ratio"] = round(float(vc.max() / vc.min()), 3)

    for col in df.columns:
        if col == target:
            continue
        s = df[col]
        n_missing = int(s.isna().sum())
        col_info = {
            "dtype": str(s.dtype),
            "n_missing": n_missing,
            "pct_missing": round(n_missing / n_rows * 100, 2),
            "n_unique": int(s.nunique(dropna=True)),
            "missingness_type": _missingness_type(df, col, target) if n_missing > 0 else "unknown",
        }

        if pd.api.types.is_numeric_dtype(s):
            desc = s.describe()
            col_info.update({
                "kind": "numeric",
                "mean": float(desc.get("mean", np.nan)),
                "std": float(desc.get("std", np.nan)),
                "min": float(desc.get("min", np.nan)),
                "max": float(desc.get("max", np.nan)),
                "skew": float(s.skew()) if s.dropna().shape[0] > 1 else 0.0,
                "p1": float(s.quantile(0.01)),
                "p99": float(s.quantile(0.99)),
            })
        elif pd.api.types.is_datetime64_any_dtype(s) or _looks_like_date(s):
            col_info["kind"] = "datetime"
        else:
            col_info["kind"] = "categorical"
            col_info["cardinality"] = int(s.nunique(dropna=True))
            if col_info["cardinality"] <= 20:
                col_info["top_values"] = s.value_counts(normalize=True).head(10).round(3).to_dict()

        if target in df.columns and pd.api.types.is_numeric_dtype(df[target]) and pd.api.types.is_numeric_dtype(s):
            try:
                corr = float(s.corr(df[target]))
                col_info["corr_with_target"] = round(corr, 4) if not np.isnan(corr) else None
            except Exception:
                col_info["corr_with_target"] = None

        summary["features"][col] = col_info

    numeric_cols = df.select_dtypes(include=[np.number]).columns.drop(target, errors="ignore")
    if len(numeric_cols) >= 3:
        corr_matrix = df[numeric_cols].corr().abs()
        high_corr_pairs = int(((corr_matrix > 0.85) & (corr_matrix < 1.0)).sum().sum() / 2)
        summary["high_corr_pair_count"] = high_corr_pairs

    return summary


def _is_discrete_dtype(s: pd.Series) -> bool:
    """True for dtypes that make sense as classification labels: integers, strings/
    object (both the legacy numpy object dtype and pandas' pyarrow-backed 'str' dtype
    that pandas 3 reports by default), and categoricals."""
    return (
        pd.api.types.is_integer_dtype(s)
        or pd.api.types.is_object_dtype(s)
        or pd.api.types.is_string_dtype(s)
        or isinstance(s.dtype, pd.CategoricalDtype)
    )


def _looks_like_date(s: pd.Series) -> bool:
    if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
        return False
    sample = s.dropna().head(20)
    if sample.empty:
        return False
    try:
        pd.to_datetime(sample, errors="raise")
        return True
    except Exception:
        return False


if __name__ == "__main__":
    import sys
    result = run_eda(sys.argv[1], sys.argv[2])
    print(json.dumps(result, indent=2, default=str))
