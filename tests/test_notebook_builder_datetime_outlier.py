from schema import PipelineConfig
import notebook_builder as nbb


def _bench_result(name, best_params, use_pca=False):
    result = {"name": name, "best_params": best_params, "cv_score": 1.0,
              "use_pca": use_pca, "device": "cpu", "gpu_used": False}
    return {"best": result, "all_results": [result], "baseline_score": 2.0}


def test_build_notebook_renders_datetime_expansion_when_configured():
    config = PipelineConfig(**{
        "problem_type": "regression",
        "target": "y",
        "datetime_features": {"signup": {}},
        "features": {"x": {"dtype": "numeric", "imputer": "SimpleImputer_mean"}},
        "candidate_models": [
            {"rank": 1, "name": "ridge", "family": "linear", "optuna_trials": 5, "sample_frac_for_optuna": 1.0},
        ],
        "metrics": {"primary": "rmse"},
    })
    bench_result = _bench_result("ridge", {"alpha": 1.0})
    nb = nbb.build_notebook(config, bench_result, "data.csv", "Datetime Test")
    all_source = "\n".join(c["source"] for c in nb["cells"] if c["cell_type"] == "code")
    assert "df['signup'] = pd.to_datetime(df['signup']" in all_source
    assert "df['signup_year'] = df['signup'].dt.year" in all_source
    assert "df['signup_month'] = df['signup'].dt.month" in all_source
    assert "df['signup_day'] = df['signup'].dt.day" in all_source
    assert "df['signup_dayofweek'] = df['signup'].dt.dayofweek" in all_source
    # the derived columns must be registered as ColumnTransformer inputs, or fitting fails
    assert "'signup_year'" in all_source


def test_build_notebook_skips_datetime_expansion_when_not_configured():
    config = PipelineConfig(**{
        "problem_type": "regression",
        "target": "y",
        "features": {"x": {"dtype": "numeric", "imputer": "SimpleImputer_mean"}},
        "candidate_models": [
            {"rank": 1, "name": "ridge", "family": "linear", "optuna_trials": 5, "sample_frac_for_optuna": 1.0},
        ],
        "metrics": {"primary": "rmse"},
    })
    bench_result = _bench_result("ridge", {"alpha": 1.0})
    nb = nbb.build_notebook(config, bench_result, "data.csv", "No Datetime Test")
    all_source = "\n".join(c["source"] for c in nb["cells"] if c["cell_type"] == "code")
    assert "to_datetime" not in all_source


def test_build_notebook_renders_outlier_clip_step_when_configured():
    config = PipelineConfig(**{
        "problem_type": "regression",
        "target": "y",
        "features": {
            "x": {"dtype": "numeric", "imputer": "SimpleImputer_mean",
                  "outlier_handling": {"method": "clip", "lower_pct": 2, "upper_pct": 98}},
        },
        "candidate_models": [
            {"rank": 1, "name": "ridge", "family": "linear", "optuna_trials": 5, "sample_frac_for_optuna": 1.0},
        ],
        "metrics": {"primary": "rmse"},
    })
    bench_result = _bench_result("ridge", {"alpha": 1.0})
    nb = nbb.build_notebook(config, bench_result, "data.csv", "Outlier Test")
    all_source = "\n".join(c["source"] for c in nb["cells"] if c["cell_type"] == "code")
    assert "class _PercentileClipper" in all_source
    assert "_PercentileClipper(lower_pct=2, upper_pct=98)" in all_source
    assert "('outlier_clip'," in all_source


def test_build_notebook_skips_outlier_clip_source_when_not_configured():
    config = PipelineConfig(**{
        "problem_type": "regression",
        "target": "y",
        "features": {"x": {"dtype": "numeric", "imputer": "SimpleImputer_mean"}},
        "candidate_models": [
            {"rank": 1, "name": "ridge", "family": "linear", "optuna_trials": 5, "sample_frac_for_optuna": 1.0},
        ],
        "metrics": {"primary": "rmse"},
    })
    bench_result = _bench_result("ridge", {"alpha": 1.0})
    nb = nbb.build_notebook(config, bench_result, "data.csv", "No Outlier Test")
    all_source = "\n".join(c["source"] for c in nb["cells"] if c["cell_type"] == "code")
    assert "_PercentileClipper" not in all_source


def test_build_notebook_datetime_and_outlier_together_actually_execute(tmp_path):
    import pandas as pd
    from pipeline_builder import run_benchmark
    from notebook_builder import execute_notebook, save_notebook

    df = pd.DataFrame({
        "signup": ["2020-01-15", "2021-06-30", "2022-12-01"] * 20,
        "x": list(range(60)),
        "target": [0, 1] * 30,
    })
    config = PipelineConfig(**{
        "problem_type": "classification",
        "target": "target",
        "datetime_features": {"signup": {}},
        "features": {
            "x": {"dtype": "numeric", "imputer": "SimpleImputer_mean",
                  "outlier_handling": {"method": "clip", "lower_pct": 1, "upper_pct": 99}},
        },
        "candidate_models": [
            {"rank": 1, "name": "logistic_regression", "family": "linear", "optuna_trials": 2, "sample_frac_for_optuna": 1.0},
        ],
        "metrics": {"primary": "roc_auc"},
    })
    result = run_benchmark(df, config, str(tmp_path / "ckpt"))
    nb = nbb.build_notebook(config, result, "data.csv", "Combined Test")
    nb_dir = tmp_path / "nb"
    nb_dir.mkdir()
    df.to_csv(nb_dir / "data.csv", index=False)
    save_notebook(nb, str(nb_dir / "test.ipynb"))
    ok, err = execute_notebook(nb, str(nb_dir))
    assert ok, err
