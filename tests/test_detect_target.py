import pandas as pd

from detect_target import detect_target


def test_detects_common_target_name(tmp_path):
    csv_path = tmp_path / "train.csv"
    pd.DataFrame({"passengerid": [1, 2], "age": [22, 38], "Survived": [0, 1]}).to_csv(csv_path, index=False)
    assert detect_target(csv_path, tmp_path) == "Survived"


def test_falls_back_to_last_column_when_no_common_name(tmp_path):
    csv_path = tmp_path / "train.csv"
    pd.DataFrame({"a": [1, 2], "b": [3, 4], "unusual_output_col": [5, 6]}).to_csv(csv_path, index=False)
    assert detect_target(csv_path, tmp_path) == "unusual_output_col"
