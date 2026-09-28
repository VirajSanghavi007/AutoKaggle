import pytest
from pydantic import ValidationError

from schema import PipelineConfig, apply_guardrails


def _base_config_dict(model_name="lightgbm", device="cpu"):
    return {
        "problem_type": "regression",
        "target": "y",
        "features": {
            "x": {"dtype": "numeric", "imputer": "SimpleImputer_mean"},
        },
        "candidate_models": [
            {"rank": 1, "name": model_name, "family": "gbm", "optuna_trials": 5,
             "sample_frac_for_optuna": 1.0, "device": device},
        ],
        "metrics": {"primary": "rmse"},
    }


def test_device_defaults_to_cpu_when_omitted():
    config = PipelineConfig(**{
        "problem_type": "regression",
        "target": "y",
        "features": {"x": {"dtype": "numeric", "imputer": "SimpleImputer_mean"}},
        "candidate_models": [
            {"rank": 1, "name": "ridge", "family": "linear", "optuna_trials": 5, "sample_frac_for_optuna": 1.0},
        ],
        "metrics": {"primary": "rmse"},
    })
    assert config.candidate_models[0].device == "cpu"


def test_device_accepts_gpu():
    config = PipelineConfig(**_base_config_dict("lightgbm", "gpu"))
    assert config.candidate_models[0].device == "gpu"


def test_device_rejects_invalid_value():
    with pytest.raises(ValidationError):
        PipelineConfig(**_base_config_dict("lightgbm", "tpu"))


def test_neural_net_is_a_valid_model_name_for_both_problem_types():
    reg_config = PipelineConfig(**_base_config_dict("neural_net", "cpu"))
    assert reg_config.candidate_models[0].name == "neural_net"

    clf_dict = _base_config_dict("neural_net", "cpu")
    clf_dict["problem_type"] = "classification"
    clf_dict["metrics"] = {"primary": "roc_auc"}
    clf_config = PipelineConfig(**clf_dict)
    assert clf_config.candidate_models[0].name == "neural_net"


def test_guardrail_downgrades_gpu_to_cpu_for_non_gpu_capable_model():
    config = PipelineConfig(**_base_config_dict("ridge", "cpu"))
    config.candidate_models[0].device = "gpu"  # simulate an LLM mistake past validation
    corrections = apply_guardrails(config)
    assert config.candidate_models[0].device == "cpu"
    assert any("ridge" in c and "gpu" in c for c in corrections)


def test_guardrail_leaves_gpu_alone_for_gpu_capable_model():
    config = PipelineConfig(**_base_config_dict("lightgbm", "gpu"))
    corrections = apply_guardrails(config)
    assert config.candidate_models[0].device == "gpu"
    assert not any("lightgbm" in c for c in corrections)
