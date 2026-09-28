import json

import validate_pipeline_config as vpc


def _valid_config_dict():
    return {
        "problem_type": "regression",
        "target": "price",
        "features": {
            "carat": {"dtype": "numeric", "imputer": "SimpleImputer_mean"},
        },
        "candidate_models": [
            {"rank": 1, "name": "lightgbm", "family": "gbm", "optuna_trials": 20, "sample_frac_for_optuna": 1.0},
        ],
        "metrics": {"primary": "rmse"},
    }


def test_validate_accepts_valid_config(tmp_path, monkeypatch):
    monkeypatch.setattr(vpc, "DATA_DIR", tmp_path)
    job_dir = tmp_path / "diamonds-autokaggle"
    job_dir.mkdir()
    (job_dir / "pipeline_config.json").write_text(json.dumps(_valid_config_dict()))

    ok, message = vpc.validate("diamonds-autokaggle")
    assert ok
    assert "valid" in message


def test_validate_rejects_bad_model_for_problem_type(tmp_path, monkeypatch):
    monkeypatch.setattr(vpc, "DATA_DIR", tmp_path)
    job_dir = tmp_path / "diamonds-autokaggle"
    job_dir.mkdir()
    bad_config = _valid_config_dict()
    bad_config["candidate_models"][0]["name"] = "logistic_regression"
    (job_dir / "pipeline_config.json").write_text(json.dumps(bad_config))

    ok, message = vpc.validate("diamonds-autokaggle")
    assert not ok
    assert "schema" in message


def test_validate_rejects_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(vpc, "DATA_DIR", tmp_path)
    ok, message = vpc.validate("does-not-exist")
    assert not ok
    assert "no pipeline_config.json" in message


def test_validate_applies_guardrail_corrections(tmp_path, monkeypatch):
    monkeypatch.setattr(vpc, "DATA_DIR", tmp_path)
    job_dir = tmp_path / "diamonds-autokaggle"
    job_dir.mkdir()
    config = _valid_config_dict()
    config["features"]["carat"]["encoder"] = "OneHotEncoder"  # numeric + encoder -> guardrail should strip it
    (job_dir / "pipeline_config.json").write_text(json.dumps(config))

    ok, message = vpc.validate("diamonds-autokaggle")
    assert ok
    assert "guardrails applied" in message
    saved = json.loads((job_dir / "pipeline_config.json").read_text())
    assert saved["features"]["carat"]["encoder"] is None
