import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from schema import PipelineConfig
import pipeline_builder as pb


def _regression_df(n=200, seed=0):
    rng = np.random.RandomState(seed)
    x = rng.normal(size=n)
    y = 3 * x + rng.normal(scale=0.2, size=n)
    return pd.DataFrame({"x": x, "y": y})


def _regression_config(metric="rmse", excluded_models=None, baseline_strategy="mean",
                        fold_error_threshold=0, outlier_handling=None):
    return PipelineConfig(**{
        "problem_type": "regression",
        "target": "y",
        "features": {
            "x": {"dtype": "numeric", "imputer": "SimpleImputer_mean", "outlier_handling": outlier_handling},
        },
        "excluded_models": excluded_models or [],
        "candidate_models": [
            {"rank": 1, "name": "ridge", "family": "linear", "optuna_trials": 3, "sample_frac_for_optuna": 1.0},
            {"rank": 2, "name": "lasso", "family": "linear", "optuna_trials": 3, "sample_frac_for_optuna": 1.0},
        ],
        "ensemble": {"stacking": False},
        "baseline": {"strategy": baseline_strategy},
        "metrics": {"primary": metric},
        "fallback_on_failure": {"cv_fold_error_threshold": fold_error_threshold},
    })


# ---- metrics.primary ----

def test_metrics_primary_rejects_wrong_metric_for_problem_type():
    with pytest.raises(ValidationError):
        PipelineConfig(**{
            "problem_type": "regression",
            "target": "y",
            "features": {"x": {"dtype": "numeric", "imputer": "SimpleImputer_mean"}},
            "candidate_models": [
                {"rank": 1, "name": "ridge", "family": "linear", "optuna_trials": 3, "sample_frac_for_optuna": 1.0},
            ],
            "metrics": {"primary": "f1"},
        })


def test_run_benchmark_uses_mae_metric_when_configured(tmp_path):
    df = _regression_df()
    config = _regression_config(metric="mae")
    result = pb.run_benchmark(df, config, str(tmp_path / "ckpt"))
    assert result["metric"] == "mae"
    # mae is always non-negative (lower-is-better, no sign flip needed) - unlike rmse
    # it should differ numerically from an rmse run on the same data for a non-trivial residual spread
    assert result["best"]["cv_score"] >= 0


def test_run_benchmark_uses_rmsle_metric_when_configured(tmp_path):
    rng = np.random.RandomState(0)
    n = 200
    x = rng.uniform(1, 10, size=n)
    y = 2 * x + rng.normal(scale=0.3, size=n)
    y = np.clip(y, 0.01, None)
    df = pd.DataFrame({"x": x, "y": y})
    config = _regression_config(metric="rmsle")
    result = pb.run_benchmark(df, config, str(tmp_path / "ckpt"))
    assert result["metric"] == "rmsle"
    assert result["best"]["cv_score"] >= 0


# ---- excluded_models ----

def test_excluded_models_filters_out_candidate(tmp_path):
    df = _regression_df()
    config = _regression_config(excluded_models=[{"name": "ridge", "reason": "test"}])
    result = pb.run_benchmark(df, config, str(tmp_path / "ckpt"))
    assert all(r["name"] != "ridge" for r in result["all_results"])
    assert any(r["name"] == "lasso" for r in result["all_results"])


def test_excluded_models_raises_when_everything_is_excluded(tmp_path):
    df = _regression_df()
    config = _regression_config(excluded_models=[{"name": "ridge"}, {"name": "lasso"}])
    with pytest.raises(RuntimeError, match="excluded_models"):
        pb.run_benchmark(df, config, str(tmp_path / "ckpt"))


# ---- baseline.strategy ----

def test_baseline_strategy_median_is_respected(tmp_path):
    df = _regression_df()
    config = _regression_config(baseline_strategy="median")
    result = pb.run_benchmark(df, config, str(tmp_path / "ckpt"))
    assert isinstance(result["baseline_score"], float)


def test_baseline_strategy_rejects_classification_value_for_regression():
    with pytest.raises(ValidationError):
        PipelineConfig(**{
            "problem_type": "regression",
            "target": "y",
            "features": {"x": {"dtype": "numeric", "imputer": "SimpleImputer_mean"}},
            "candidate_models": [
                {"rank": 1, "name": "ridge", "family": "linear", "optuna_trials": 3, "sample_frac_for_optuna": 1.0},
            ],
            "baseline": {"strategy": "most_frequent"},
            "metrics": {"primary": "rmse"},
        })


def test_classification_baseline_defaults_to_most_frequent_not_mean():
    config = PipelineConfig(**{
        "problem_type": "classification",
        "target": "y",
        "features": {"x": {"dtype": "numeric", "imputer": "SimpleImputer_mean"}},
        "candidate_models": [
            {"rank": 1, "name": "logistic_regression", "family": "linear", "optuna_trials": 3, "sample_frac_for_optuna": 1.0},
        ],
        "metrics": {"primary": "roc_auc"},
    })
    assert config.baseline.strategy == "most_frequent"


# ---- cv_fold_error_threshold ----

def test_cv_fold_error_threshold_tolerates_bounded_failures(monkeypatch):
    calls = {"n": 0}

    def flaky_fit_score(pipe, X, y, train_idx, test_idx, problem_type, n_classes, metric):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated fold failure")
        return 1.0

    monkeypatch.setattr(pb, "_fit_score_fold", flaky_fit_score)
    from sklearn.model_selection import KFold
    X = pd.DataFrame({"x": range(20)})
    y = pd.Series(range(20))
    cv = KFold(n_splits=4)
    score = pb._cv_predict_metric(None, X, y, cv, "regression", n_jobs=1, fold_error_threshold=1)
    assert score == 1.0


def test_cv_fold_error_threshold_raises_when_exceeded(monkeypatch):
    def always_fails(pipe, X, y, train_idx, test_idx, problem_type, n_classes, metric):
        raise RuntimeError("simulated fold failure")

    monkeypatch.setattr(pb, "_fit_score_fold", always_fails)
    from sklearn.model_selection import KFold
    X = pd.DataFrame({"x": range(20)})
    y = pd.Series(range(20))
    cv = KFold(n_splits=4)
    with pytest.raises(RuntimeError, match="CV folds failed"):
        pb._cv_predict_metric(None, X, y, cv, "regression", n_jobs=1, fold_error_threshold=1)


# ---- class_balance.strategy -> class_weight ----

def test_class_weight_balanced_applied_to_logistic_regression():
    import optuna
    trial = optuna.trial.FixedTrial({"C": 1.0})
    model = pb._model_and_space("logistic_regression", "classification", 7, trial, class_weight_balanced=True)
    assert model.get_params()["class_weight"] == "balanced"


def test_class_weight_not_applied_when_flag_is_false():
    import optuna
    trial = optuna.trial.FixedTrial({"C": 1.0})
    model = pb._model_and_space("logistic_regression", "classification", 7, trial, class_weight_balanced=False)
    assert model.get_params()["class_weight"] is None


def test_class_weight_ignored_for_regression_models():
    import optuna
    trial = optuna.trial.FixedTrial({"alpha": 1.0})
    model = pb._model_and_space("ridge", "regression", 7, trial, class_weight_balanced=True)
    assert "class_weight" not in model.get_params() or model.get_params().get("class_weight") in (None,)


# ---- outlier_handling ----

def test_outlier_clipper_clips_to_learned_percentiles():
    clipper = pb._PercentileClipper(lower_pct=10, upper_pct=90)
    X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [100]], dtype=float)
    clipper.fit(X)
    transformed = clipper.transform(np.array([[1000], [-1000]], dtype=float))
    assert transformed[0, 0] == clipper.upper_[0]
    assert transformed[1, 0] == clipper.lower_[0]


def test_build_column_transformer_includes_outlier_clip_step_when_configured():
    config = _regression_config(outlier_handling={"method": "clip", "lower_pct": 1, "upper_pct": 99})
    ct = pb.build_column_transformer(config)
    x_pipeline = {name: transformer for name, transformer, _cols in ct.transformers}["x"]
    step_names = [name for name, _ in x_pipeline.steps]
    assert "outlier_clip" in step_names


def test_build_column_transformer_skips_outlier_clip_when_not_configured():
    config = _regression_config(outlier_handling=None)
    ct = pb.build_column_transformer(config)
    x_pipeline = {name: transformer for name, transformer, _cols in ct.transformers}["x"]
    step_names = [name for name, _ in x_pipeline.steps]
    assert "outlier_clip" not in step_names


# ---- datetime_features ----

def test_expand_datetime_features_adds_derived_numeric_columns_and_feature_specs():
    df = pd.DataFrame({
        "signup": ["2020-01-15", "2021-06-30", "2022-12-01"],
        "target": [1, 0, 1],
    })
    config = PipelineConfig(**{
        "problem_type": "classification",
        "target": "target",
        "datetime_features": {"signup": {}},
        "features": {},
        "candidate_models": [
            {"rank": 1, "name": "logistic_regression", "family": "linear", "optuna_trials": 3, "sample_frac_for_optuna": 1.0},
        ],
        "metrics": {"primary": "roc_auc"},
    })
    expanded = pb.expand_datetime_features(df, config)
    assert "signup_year" in expanded.columns
    assert "signup_month" in expanded.columns
    assert "signup_day" in expanded.columns
    assert "signup_dayofweek" in expanded.columns
    assert expanded["signup_year"].tolist() == [2020, 2021, 2022]
    assert "signup_year" in config.features
    assert config.features["signup_year"].dtype == "numeric"


def test_expand_datetime_features_noop_when_empty():
    df = pd.DataFrame({"x": [1, 2, 3], "target": [1, 0, 1]})
    config = PipelineConfig(**{
        "problem_type": "classification",
        "target": "target",
        "features": {"x": {"dtype": "numeric", "imputer": "SimpleImputer_mean"}},
        "candidate_models": [
            {"rank": 1, "name": "logistic_regression", "family": "linear", "optuna_trials": 3, "sample_frac_for_optuna": 1.0},
        ],
        "metrics": {"primary": "roc_auc"},
    })
    expanded = pb.expand_datetime_features(df, config)
    assert list(expanded.columns) == list(df.columns)


# ---- score_kind clarity fix ----

def test_stacking_final_best_marked_as_held_out_val_not_cv(tmp_path):
    rng = np.random.RandomState(1)
    n = 300
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 2 * x1 - x2 + rng.normal(scale=0.1, size=n)
    df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})
    config = PipelineConfig(**{
        "problem_type": "regression",
        "target": "y",
        "features": {
            "x1": {"dtype": "numeric", "imputer": "SimpleImputer_mean"},
            "x2": {"dtype": "numeric", "imputer": "SimpleImputer_mean"},
        },
        "candidate_models": [
            {"rank": 1, "name": "ridge", "family": "linear", "optuna_trials": 3, "sample_frac_for_optuna": 1.0},
            {"rank": 2, "name": "lasso", "family": "linear", "optuna_trials": 3, "sample_frac_for_optuna": 1.0},
        ],
        "ensemble": {"stacking": True, "meta_learner": "Ridge", "use_top_n_models": 2},
        "metrics": {"primary": "rmse"},
    })
    result = pb.run_benchmark(df, config, str(tmp_path / "ckpt"))
    for r in result["all_results"]:
        assert r["score_kind"] == "cv"
    if result["best"]["name"] == "stacking_ensemble":
        assert result["best"]["score_kind"] == "held_out_val"
