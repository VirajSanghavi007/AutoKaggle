from schema import PipelineConfig
import notebook_builder as nbb


def _config(model_name="lightgbm"):
    return PipelineConfig(**{
        "problem_type": "regression",
        "target": "y",
        "features": {"x": {"dtype": "numeric", "imputer": "SimpleImputer_mean"}},
        "candidate_models": [
            {"rank": 1, "name": model_name, "family": "gbm", "optuna_trials": 5, "sample_frac_for_optuna": 1.0},
        ],
        "metrics": {"primary": "rmse"},
    })


def _bench_result(name, best_params, gpu_used=False, use_pca=False):
    result = {"name": name, "best_params": best_params, "cv_score": 1.0,
              "use_pca": use_pca, "device": "gpu" if gpu_used else "cpu", "gpu_used": gpu_used}
    return {
        "best": result,
        "all_results": [result],
        "baseline_score": 2.0,
    }


def test_inject_device_params_noop_for_non_gpu_capable_model():
    params = nbb._inject_device_params("ridge", {"alpha": 1.0}, gpu_used=True)
    assert params == {"alpha": 1.0}


def test_inject_device_params_lightgbm_cpu_when_not_gpu_used():
    # gpu_used=False reflects that pipeline_builder's own probe already determined this
    # library/hardware combo couldn't use GPU (e.g. lightgbm's plain wheel has no GPU
    # tree learner even when CUDA itself is available) - notebook must match that, not re-guess
    params = nbb._inject_device_params("lightgbm", {"n_estimators": 100}, gpu_used=False)
    assert params["device"] == "cpu"


def test_inject_device_params_xgboost_gpu_when_gpu_used():
    params = nbb._inject_device_params("xgboost", {"n_estimators": 100}, gpu_used=True)
    assert params["device"] == "cuda"
    assert params["tree_method"] == "hist"


def test_nn_model_source_cell_contains_expected_classes():
    source = nbb._nn_model_source_cell()
    assert "class TorchMLP" in source
    assert "class TorchMLPRegressor" in source
    assert "class TorchMLPClassifier" in source
    assert "def resolve_device" in source
    # must not depend on this repo's package layout - notebooks run standalone
    assert "from nn_model import" not in source
    assert "import nn_model" not in source


def test_build_notebook_includes_nn_source_cell_only_when_neural_net_used():
    config = _config("neural_net")
    bench_result = _bench_result("neural_net", {
        "hidden_dim": 32, "n_layers": 2, "learning_rate": 0.001, "epochs": 50, "batch_size": 32,
    })
    nb = nbb.build_notebook(config, bench_result, "data.csv", "Test NN Notebook")
    all_source = "\n".join(c["source"] for c in nb["cells"] if c["cell_type"] == "code")
    assert "class TorchMLPRegressor" in all_source
    assert "TorchMLPRegressor(" in all_source

    non_nn_config = _config("ridge")
    non_nn_result = _bench_result("ridge", {"alpha": 1.0})
    nb2 = nbb.build_notebook(non_nn_config, non_nn_result, "data.csv", "Test Non-NN Notebook")
    all_source2 = "\n".join(c["source"] for c in nb2["cells"] if c["cell_type"] == "code")
    assert "class TorchMLP" not in all_source2


def test_build_notebook_renders_gpu_device_when_gpu_used_is_true():
    config = _config("lightgbm")
    bench_result = _bench_result("lightgbm", {"n_estimators": 100, "learning_rate": 0.1}, gpu_used=True)
    nb = nbb.build_notebook(config, bench_result, "data.csv", "Test GPU Notebook")
    all_source = "\n".join(c["source"] for c in nb["cells"] if c["cell_type"] == "code")
    assert "device='gpu'" in all_source


def test_build_notebook_renders_pca_step_when_used():
    config = _config("ridge")
    bench_result = _bench_result("ridge", {"alpha": 1.0}, use_pca=True)
    nb = nbb.build_notebook(config, bench_result, "data.csv", "Test PCA Notebook")
    all_source = "\n".join(c["source"] for c in nb["cells"] if c["cell_type"] == "code")
    assert "from sklearn.decomposition import PCA" in all_source
    assert "('pca', PCA(" in all_source


def test_build_notebook_renders_smote_step_and_label_encoding_for_classification():
    config = PipelineConfig(**{
        "problem_type": "classification",
        "target": "y",
        "features": {"x": {"dtype": "numeric", "imputer": "SimpleImputer_mean"}},
        "candidate_models": [
            {"rank": 1, "name": "logistic_regression", "family": "linear", "optuna_trials": 5, "sample_frac_for_optuna": 1.0},
        ],
        "class_balance": {"applicable": True, "smote_variant": "SMOTE"},
        "metrics": {"primary": "roc_auc"},
    })
    bench_result = _bench_result("logistic_regression", {"C": 1.0})
    nb = nbb.build_notebook(config, bench_result, "data.csv", "Test SMOTE Notebook")
    all_source = "\n".join(c["source"] for c in nb["cells"] if c["cell_type"] == "code")
    assert "from imblearn.pipeline import Pipeline as ImbPipeline" in all_source
    assert "from imblearn.over_sampling import SMOTE" in all_source
    assert "('resample', SMOTE(random_state=SEED))" in all_source
    assert "LabelEncoder().fit_transform(df[TARGET])" in all_source


def test_build_notebook_stacking_includes_final_estimator():
    config = _config("ridge")
    r1 = {"name": "ridge", "best_params": {"alpha": 1.0}, "cv_score": 1.0, "use_pca": False, "gpu_used": False}
    r2 = {"name": "lasso", "best_params": {"alpha": 0.5}, "cv_score": 1.1, "use_pca": False, "gpu_used": False}
    best = {"name": "stacking_ensemble", "members": ["ridge", "lasso"], "cv_score": 0.9, "use_pca": False}
    bench_result = {"best": best, "all_results": [r1, r2], "baseline_score": 2.0}
    nb = nbb.build_notebook(config, bench_result, "data.csv", "Test Stack Notebook")
    all_source = "\n".join(c["source"] for c in nb["cells"] if c["cell_type"] == "code")
    assert "final_estimator=Ridge(random_state=SEED)" in all_source


def test_build_notebook_renders_class_weight_balanced_when_configured():
    config = PipelineConfig(**{
        "problem_type": "classification",
        "target": "y",
        "features": {"x": {"dtype": "numeric", "imputer": "SimpleImputer_mean"}},
        "candidate_models": [
            {"rank": 1, "name": "logistic_regression", "family": "linear", "optuna_trials": 5, "sample_frac_for_optuna": 1.0},
        ],
        "class_balance": {"applicable": True, "strategy": "class_weight"},
        "metrics": {"primary": "roc_auc"},
    })
    bench_result = _bench_result("logistic_regression", {"C": 1.0})
    nb = nbb.build_notebook(config, bench_result, "data.csv", "Test ClassWeight Notebook")
    all_source = "\n".join(c["source"] for c in nb["cells"] if c["cell_type"] == "code")
    assert "class_weight='balanced'" in all_source


def test_build_notebook_omits_class_weight_when_smote_is_used_instead():
    config = PipelineConfig(**{
        "problem_type": "classification",
        "target": "y",
        "features": {"x": {"dtype": "numeric", "imputer": "SimpleImputer_mean"}},
        "candidate_models": [
            {"rank": 1, "name": "logistic_regression", "family": "linear", "optuna_trials": 5, "sample_frac_for_optuna": 1.0},
        ],
        "class_balance": {"applicable": True, "smote_variant": "SMOTE"},
        "metrics": {"primary": "roc_auc"},
    })
    bench_result = _bench_result("logistic_regression", {"C": 1.0})
    nb = nbb.build_notebook(config, bench_result, "data.csv", "Test SMOTE Not ClassWeight Notebook")
    all_source = "\n".join(c["source"] for c in nb["cells"] if c["cell_type"] == "code")
    assert "class_weight" not in all_source
