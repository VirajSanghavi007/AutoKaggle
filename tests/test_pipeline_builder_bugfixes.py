import numpy as np
import pandas as pd

from schema import PipelineConfig, apply_guardrails
import pipeline_builder as pb


def _classification_df_with_string_labels(n=200, seed=0):
    rng = np.random.RandomState(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    label = np.where(x1 + x2 > 0, "yes", "no")
    return pd.DataFrame({"x1": x1, "x2": x2, "label": label})


def _classification_config(model_name="xgboost", n_splits=3, extra_candidates=None):
    candidates = [
        {"rank": 1, "name": model_name, "family": "gbm", "optuna_trials": 3, "sample_frac_for_optuna": 1.0},
    ]
    if extra_candidates:
        candidates += extra_candidates
    return PipelineConfig(**{
        "problem_type": "classification",
        "target": "label",
        "features": {
            "x1": {"dtype": "numeric", "imputer": "SimpleImputer_mean"},
            "x2": {"dtype": "numeric", "imputer": "SimpleImputer_mean"},
        },
        "candidate_models": candidates,
        "cv": {"strategy": "StratifiedKFold", "n_splits": n_splits, "n_repeats": 1},
        "ensemble": {"stacking": False},
        "metrics": {"primary": "roc_auc"},
    })


def test_xgboost_fits_on_string_class_labels_without_crashing(tmp_path):
    df = _classification_df_with_string_labels()
    config = _classification_config("xgboost")
    result = pb.run_benchmark(df, config, str(tmp_path / "ckpt"))
    assert result["best"]["name"] == "xgboost"
    assert "test_score" in result


def test_pca_branch_does_not_crash_when_enabled(tmp_path):
    df = _classification_df_with_string_labels()
    config = _classification_config("ridge_classifier")
    config.preprocessing_branches = {"raw": {"enabled": True}, "pca": {"enabled": True, "n_components": 0.95}}
    # must not raise - this used to crash on the first suggest_int() call with an empty FixedTrial
    result = pb.run_benchmark(df, config, str(tmp_path / "ckpt"))
    assert "use_pca" in result


def test_run_benchmark_reports_test_score(tmp_path):
    rng = np.random.RandomState(0)
    n = 300
    x = rng.normal(size=n)
    y = 2 * x + rng.normal(scale=0.1, size=n)
    df = pd.DataFrame({"x": x, "target": y})
    config = PipelineConfig(**{
        "problem_type": "regression",
        "target": "target",
        "features": {"x": {"dtype": "numeric", "imputer": "SimpleImputer_mean"}},
        "candidate_models": [
            {"rank": 1, "name": "ridge", "family": "linear", "optuna_trials": 3, "sample_frac_for_optuna": 1.0},
        ],
        "ensemble": {"stacking": False},
        "metrics": {"primary": "rmse"},
    })
    result = pb.run_benchmark(df, config, str(tmp_path / "ckpt"))
    assert "test_score" in result
    assert isinstance(result["test_score"], float)
    assert result["test_size"] > 0


def test_smotenc_guardrail_downgrades_to_smote():
    config = PipelineConfig(**{
        "problem_type": "classification",
        "target": "label",
        "features": {"x": {"dtype": "numeric", "imputer": "SimpleImputer_mean"}},
        "candidate_models": [
            {"rank": 1, "name": "logistic_regression", "family": "linear", "optuna_trials": 3, "sample_frac_for_optuna": 1.0},
        ],
        "class_balance": {"applicable": True, "smote_variant": "SMOTENC"},
        "metrics": {"primary": "roc_auc"},
    })
    corrections = apply_guardrails(config)
    assert config.class_balance.smote_variant == "SMOTE"
    assert any("SMOTENC" in c for c in corrections)


def test_neural_net_gets_full_optuna_hyperparameter_search(tmp_path):
    df = _classification_df_with_string_labels(n=150)
    config = _classification_config("neural_net")
    config.candidate_models[0].optuna_trials = 4
    result = pb.run_benchmark(df, config, str(tmp_path / "ckpt"))
    best = [r for r in result["all_results"] if r["name"] == "neural_net"][0]
    tuned_keys = set(best["best_params"].keys())
    assert tuned_keys == {"hidden_dim", "n_layers", "learning_rate", "epochs", "batch_size"}
    assert best["n_trials_run"] >= 1
