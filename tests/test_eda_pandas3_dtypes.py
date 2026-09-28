import pandas as pd

from eda import run_eda, _looks_like_date, _is_discrete_dtype


def test_string_dtype_column_is_discrete(tmp_path):
    # pandas 3 reports plain string columns as dtype "str", not "object" - the target
    # value-count/imbalance detection must still recognize this as a classification label
    s = pd.Series(["yes", "no", "yes"])
    assert str(s.dtype) == "str"
    assert _is_discrete_dtype(s)


def test_date_like_string_column_is_detected_as_datetime():
    s = pd.Series(["2020-01-01", "2020-01-02", "2020-01-03"])
    assert str(s.dtype) == "str"
    assert _looks_like_date(s)


def test_numeric_column_is_not_treated_as_date():
    s = pd.Series([1.0, 2.0, 3.0])
    assert not _looks_like_date(s)


def test_run_eda_reports_value_counts_for_string_classification_target(tmp_path):
    csv_path = tmp_path / "train.csv"
    pd.DataFrame({
        "feature": [1, 2, 3, 4, 5, 6],
        "label": ["yes", "no", "yes", "no", "yes", "yes"],
    }).to_csv(csv_path, index=False)

    summary = run_eda(str(csv_path), "label")

    assert "target_value_counts" in summary
    assert "imbalance_ratio" in summary
    assert summary["imbalance_ratio"] > 1.0


def test_run_eda_detects_datetime_feature_column(tmp_path):
    csv_path = tmp_path / "train.csv"
    pd.DataFrame({
        "signup_date": ["2021-01-01", "2021-02-15", "2021-03-30"],
        "target": [1, 0, 1],
    }).to_csv(csv_path, index=False)

    summary = run_eda(str(csv_path), "target")

    assert summary["features"]["signup_date"]["kind"] == "datetime"
