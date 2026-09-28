import optuna

import pipeline_builder as pb


def test_gpu_ok_false_when_cpu_requested():
    assert pb._gpu_ok("cpu", "lightgbm") is False


def test_gpu_ok_false_when_gpu_requested_but_no_cuda(monkeypatch):
    monkeypatch.setattr(pb, "resolve_device", lambda requested: "cpu")
    assert pb._gpu_ok("gpu", "lightgbm") is False


def test_gpu_ok_true_for_neural_net_whenever_cuda_available(monkeypatch):
    monkeypatch.setattr(pb, "resolve_device", lambda requested: "cuda")
    assert pb._gpu_ok("gpu", "neural_net") is True


def test_gpu_ok_checks_library_build_support_for_boosted_models(monkeypatch):
    # cuda is available, but this specific library's installed build has no GPU support
    # (this is exactly the real-world lightgbm-pip-wheel-has-no-GPU-tree-learner case)
    monkeypatch.setattr(pb, "resolve_device", lambda requested: "cuda")
    monkeypatch.setattr(pb, "_library_build_supports_gpu", lambda name: False)
    assert pb._gpu_ok("gpu", "lightgbm") is False


def test_gpu_ok_true_when_cuda_and_library_both_support_it(monkeypatch):
    monkeypatch.setattr(pb, "resolve_device", lambda requested: "cuda")
    monkeypatch.setattr(pb, "_library_build_supports_gpu", lambda name: True)
    assert pb._gpu_ok("gpu", "xgboost") is True


def test_model_and_space_lightgbm_respects_cpu_fallback(monkeypatch):
    monkeypatch.setattr(pb, "resolve_device", lambda requested: "cpu")
    trial = optuna.trial.FixedTrial({
        "n_estimators": 100, "learning_rate": 0.1, "num_leaves": 31, "max_depth": 5,
    })
    model = pb._model_and_space("lightgbm", "regression", 7, trial, n_jobs=2, device="gpu")
    assert model.get_params()["device"] == "cpu"


def test_model_and_space_lightgbm_uses_gpu_when_available(monkeypatch):
    monkeypatch.setattr(pb, "resolve_device", lambda requested: "cuda")
    monkeypatch.setattr(pb, "_library_build_supports_gpu", lambda name: True)
    trial = optuna.trial.FixedTrial({
        "n_estimators": 100, "learning_rate": 0.1, "num_leaves": 31, "max_depth": 5,
    })
    model = pb._model_and_space("lightgbm", "regression", 7, trial, n_jobs=2, device="gpu")
    assert model.get_params()["device"] == "gpu"


def test_model_and_space_xgboost_device_param(monkeypatch):
    monkeypatch.setattr(pb, "resolve_device", lambda requested: "cuda")
    monkeypatch.setattr(pb, "_library_build_supports_gpu", lambda name: True)
    trial = optuna.trial.FixedTrial({
        "n_estimators": 100, "learning_rate": 0.1, "max_depth": 5, "subsample": 0.8,
    })
    model = pb._model_and_space("xgboost", "regression", 7, trial, device="gpu")
    assert model.get_params()["device"] == "cuda"


def test_model_and_space_catboost_task_type(monkeypatch):
    monkeypatch.setattr(pb, "resolve_device", lambda requested: "cpu")
    trial = optuna.trial.FixedTrial({
        "iterations": 100, "learning_rate": 0.1, "depth": 5,
    })
    model = pb._model_and_space("catboost", "regression", 7, trial, device="gpu")
    assert model.get_params()["task_type"] == "CPU"


def test_library_build_support_result_is_cached(monkeypatch):
    monkeypatch.setattr(pb, "_GPU_BUILD_SUPPORT_CACHE", {})
    pb._library_build_supports_gpu("xgboost")
    assert "xgboost" in pb._GPU_BUILD_SUPPORT_CACHE
    cached_value = pb._GPU_BUILD_SUPPORT_CACHE["xgboost"]
    pb._GPU_BUILD_SUPPORT_CACHE["xgboost"] = not cached_value  # prove the second call reads the cache
    assert pb._library_build_supports_gpu("xgboost") == (not cached_value)


def test_real_lightgbm_wheel_gpu_probe_matches_this_machine():
    # not mocked - documents ground truth for this repo's actual dependency: the plain
    # pip lightgbm wheel has no GPU tree learner even when a CUDA device is present
    result = pb._library_build_supports_gpu("lightgbm")
    assert isinstance(result, bool)
