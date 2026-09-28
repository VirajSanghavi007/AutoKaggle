import json

import pandas as pd

import run_pipeline
import job_common
import run_state


def test_resumed_bench_result_still_gets_datetime_features_expanded_before_notebook_build(tmp_path, monkeypatch):
    """Regression test for a real bug: if bench_result.json is already cached (crash
    happened after benchmarking but before notebook publish), run_pipeline used to skip
    straight to notebook building without ever calling expand_datetime_features, so the
    notebook's ColumnTransformer referenced derived columns (e.g. 'signup_year') that
    were never computed from the raw CSV and crashed at execution time."""
    data_dir = tmp_path / "data"
    checkpoint_dir = tmp_path / "checkpoints"
    notebook_dir = tmp_path / "notebooks"
    current_run_path = tmp_path / "current_run.json"

    monkeypatch.setattr(run_pipeline, "DATA_DIR", data_dir)
    monkeypatch.setattr(run_pipeline, "CHECKPOINT_DIR", checkpoint_dir)
    monkeypatch.setattr(run_pipeline, "NOTEBOOK_DIR", notebook_dir)
    monkeypatch.setattr(run_pipeline, "CURRENT_RUN_PATH", current_run_path)
    monkeypatch.setattr(run_pipeline, "publish_notebook", lambda *a, **k: None)
    monkeypatch.setattr(run_state, "CHECKPOINT_ROOT", checkpoint_dir)

    kernel_slug = job_common.resolve_kernel_slug("owner/dt-resume-test")
    job_dir = data_dir / kernel_slug
    job_dir.mkdir(parents=True)
    job_checkpoint_dir = checkpoint_dir / kernel_slug
    job_checkpoint_dir.mkdir(parents=True)

    df = pd.DataFrame({
        "signup": ["2020-01-15", "2021-06-30", "2022-12-01"] * 20,
        "x": list(range(60)),
        "target": [0, 1] * 30,
    })
    csv_path = job_dir / "train.csv"
    df.to_csv(csv_path, index=False)
    (job_dir / "_train_csv_path.txt").write_text(str(csv_path))

    pipeline_config_dict = {
        "problem_type": "classification",
        "target": "target",
        "datetime_features": {"signup": {}},
        "features": {"x": {"dtype": "numeric", "imputer": "SimpleImputer_mean"}},
        "candidate_models": [
            {"rank": 1, "name": "logistic_regression", "family": "linear", "optuna_trials": 2, "sample_frac_for_optuna": 1.0},
        ],
        "metrics": {"primary": "roc_auc"},
    }
    (job_dir / "pipeline_config.json").write_text(json.dumps(pipeline_config_dict))

    # simulate a cached benchmark result from BEFORE the crash - the winning model really
    # was trained with the derived datetime columns present
    fake_bench_result = {
        "best": {"name": "logistic_regression", "best_params": {"C": 1.0}, "cv_score": -0.7,
                  "score_kind": "cv", "use_pca": False, "device": "cpu", "gpu_used": False},
        "all_results": [
            {"name": "logistic_regression", "best_params": {"C": 1.0}, "cv_score": -0.7,
             "score_kind": "cv", "use_pca": False, "device": "cpu", "gpu_used": False},
        ],
        "baseline_score": -0.5,
        "stack_comparison": None,
        "use_pca": False,
        "val_size": 10,
        "test_size": 10,
        "test_score": -0.6,
        "metric": "roc_auc",
    }
    (job_dir / "bench_result.json").write_text(json.dumps(fake_bench_result))

    run_state_dict = {"eda_done": True, "llm_done": False, "bench_done": True,
                      "notebook_done": False, "published_done": False}
    (job_checkpoint_dir / "run_state.json").write_text(json.dumps(run_state_dict))

    job = {"target_slug": "owner/dt-resume-test", "raw_url": "https://www.kaggle.com/datasets/owner/dt-resume-test"}
    config = {"kaggle_repo": {}}

    exit_code = run_pipeline.run_pipeline(job, config, log=_NullLog())

    assert exit_code == run_pipeline.EXIT_SUCCESS

    saved_config = json.loads((job_dir / "pipeline_config.json").read_text())
    assert "signup_year" in saved_config["features"]

    nb_path = notebook_dir / f"{kernel_slug}.ipynb"
    assert nb_path.exists()

    current_run = json.loads(current_run_path.read_text())
    assert current_run["title"] == "Dt Resume Test (ROC AUC 0.70)"


def test_resumed_bench_result_missing_metric_key_falls_back_instead_of_crashing(tmp_path, monkeypatch):
    """Regression test: bench_result.json cached before the 'metric' field existed
    (schema evolves, checkpoints are long-lived) must not KeyError on resume."""
    data_dir = tmp_path / "data"
    checkpoint_dir = tmp_path / "checkpoints"
    notebook_dir = tmp_path / "notebooks"
    current_run_path = tmp_path / "current_run.json"

    monkeypatch.setattr(run_pipeline, "DATA_DIR", data_dir)
    monkeypatch.setattr(run_pipeline, "CHECKPOINT_DIR", checkpoint_dir)
    monkeypatch.setattr(run_pipeline, "NOTEBOOK_DIR", notebook_dir)
    monkeypatch.setattr(run_pipeline, "CURRENT_RUN_PATH", current_run_path)
    monkeypatch.setattr(run_pipeline, "publish_notebook", lambda *a, **k: None)
    monkeypatch.setattr(run_state, "CHECKPOINT_ROOT", checkpoint_dir)

    kernel_slug = job_common.resolve_kernel_slug("owner/legacy-checkpoint-test")
    job_dir = data_dir / kernel_slug
    job_dir.mkdir(parents=True)
    job_checkpoint_dir = checkpoint_dir / kernel_slug
    job_checkpoint_dir.mkdir(parents=True)

    df = pd.DataFrame({"x": list(range(60)), "target": [0, 1] * 30})
    csv_path = job_dir / "train.csv"
    df.to_csv(csv_path, index=False)
    (job_dir / "_train_csv_path.txt").write_text(str(csv_path))

    pipeline_config_dict = {
        "problem_type": "classification",
        "target": "target",
        "features": {"x": {"dtype": "numeric", "imputer": "SimpleImputer_mean"}},
        "candidate_models": [
            {"rank": 1, "name": "logistic_regression", "family": "linear", "optuna_trials": 2, "sample_frac_for_optuna": 1.0},
        ],
        "metrics": {"primary": "roc_auc"},
    }
    (job_dir / "pipeline_config.json").write_text(json.dumps(pipeline_config_dict))

    # old-schema bench_result.json - no "metric" key at all, exactly like a real
    # pre-existing checkpoint from before that field was added
    legacy_bench_result = {
        "best": {"name": "logistic_regression", "best_params": {"C": 1.0}, "cv_score": -0.7,
                  "use_pca": False},
        "all_results": [
            {"name": "logistic_regression", "best_params": {"C": 1.0}, "cv_score": -0.7, "use_pca": False},
        ],
        "baseline_score": -0.5,
        "stack_comparison": None,
        "use_pca": False,
        "val_size": 10,
        "test_size": 10,
    }
    (job_dir / "bench_result.json").write_text(json.dumps(legacy_bench_result))

    run_state_dict = {"eda_done": True, "llm_done": False, "bench_done": True,
                      "notebook_done": False, "published_done": False}
    (job_checkpoint_dir / "run_state.json").write_text(json.dumps(run_state_dict))

    job = {"target_slug": "owner/legacy-checkpoint-test", "raw_url": "https://www.kaggle.com/datasets/owner/legacy-checkpoint-test"}
    config = {"kaggle_repo": {}}

    exit_code = run_pipeline.run_pipeline(job, config, log=_NullLog())

    assert exit_code == run_pipeline.EXIT_SUCCESS
    current_run = json.loads(current_run_path.read_text())
    assert current_run["title"] == "Legacy Checkpoint Test (ROC AUC 0.70)"


class _NullLog:
    def info(self, *a, **k):
        pass

    def warning(self, *a, **k):
        pass

    def error(self, *a, **k):
        pass
