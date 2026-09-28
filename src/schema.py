from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator

IMPUTERS = [
    "SimpleImputer_mean", "SimpleImputer_median", "SimpleImputer_most_frequent",
    "SimpleImputer_constant", "KNNImputer", "IterativeImputer",
]
ENCODERS = [
    "OneHotEncoder", "OrdinalEncoder", "TargetEncoder", "BinaryEncoder",
    "FrequencyEncoder", "CatBoostEncoder", "HashingEncoder", None,
]
SCALERS = [
    "StandardScaler", "RobustScaler", "MinMaxScaler", "MaxAbsScaler",
    "PowerTransformer", "QuantileTransformer", None,
]
SMOTE_VARIANTS = [
    "SMOTE", "BorderlineSMOTE", "ADASYN", "SMOTENC", "SMOTEN",
    "SMOTETomek", "SMOTEENN", "RandomUnderSampler", None,
]
CV_STRATEGIES = ["KFold", "StratifiedKFold", "RepeatedKFold", "RepeatedStratifiedKFold"]

REGRESSION_MODELS = ["lightgbm", "xgboost", "catboost", "random_forest", "extra_trees",
                      "elasticnet", "ridge", "lasso", "linear_regression", "knn", "svr", "neural_net"]
CLASSIFICATION_MODELS = ["lightgbm", "xgboost", "catboost", "random_forest", "extra_trees",
                          "logistic_regression", "ridge_classifier", "naive_bayes", "knn", "svc", "neural_net"]

# Models with a real GPU-accelerated code path. Everything else always runs on CPU
# regardless of what the LLM requests in CandidateModel.device.
GPU_CAPABLE_MODELS = ["lightgbm", "xgboost", "catboost", "neural_net"]
DEVICES = ["cpu", "gpu"]

MISSINGNESS_TYPES = ["MCAR", "MAR", "MNAR", "unknown"]

REGRESSION_METRICS = ["rmse", "mae", "rmsle"]
CLASSIFICATION_METRICS = ["roc_auc", "f1", "accuracy", "log_loss"]
# metrics stored internally as their negation (see pipeline_builder._score_from_predictions's
# lower-is-better convention) - anything displayed to a human must flip the sign back
NEGATED_METRICS = {"roc_auc", "f1", "accuracy"}
METRIC_DISPLAY_NAMES = {
    "rmse": "RMSE", "mae": "MAE", "rmsle": "RMSLE",
    "roc_auc": "ROC AUC", "f1": "F1", "accuracy": "Accuracy", "log_loss": "Log Loss",
}
CLASS_BALANCE_STRATEGIES = ["smote", "class_weight", None]
BASELINE_REGRESSION_STRATEGIES = ["mean", "median"]
BASELINE_CLASSIFICATION_STRATEGIES = ["most_frequent", "stratified", "uniform"]


class FeatureSpec(BaseModel):
    dtype: Literal["numeric", "categorical", "ordinal"]
    cardinality: Optional[int] = None
    skew: Optional[float] = None
    missingness_type: str = "unknown"
    imputer: str
    encoder: Optional[str] = None
    scaler: Optional[str] = None
    outlier_handling: Optional[dict] = None
    confidence: float = Field(ge=0, le=1, default=0.5)
    flags: list[str] = Field(default_factory=list)

    @field_validator("imputer")
    @classmethod
    def check_imputer(cls, v):
        if v not in IMPUTERS:
            raise ValueError(f"imputer '{v}' not in allow-list {IMPUTERS}")
        return v

    @field_validator("encoder")
    @classmethod
    def check_encoder(cls, v):
        if v not in ENCODERS:
            raise ValueError(f"encoder '{v}' not in allow-list {ENCODERS}")
        return v

    @field_validator("scaler")
    @classmethod
    def check_scaler(cls, v):
        if v not in SCALERS:
            raise ValueError(f"scaler '{v}' not in allow-list {SCALERS}")
        return v

    @field_validator("missingness_type")
    @classmethod
    def check_missingness(cls, v):
        if v not in MISSINGNESS_TYPES:
            raise ValueError(f"missingness_type '{v}' not in allow-list {MISSINGNESS_TYPES}")
        return v


class CandidateModel(BaseModel):
    rank: int
    name: str
    family: str
    optuna_trials: int = Field(ge=1, le=200)
    sample_frac_for_optuna: float = Field(gt=0, le=1)
    device: str = "cpu"

    @field_validator("device")
    @classmethod
    def check_device(cls, v):
        if v not in DEVICES:
            raise ValueError(f"device '{v}' not in allow-list {DEVICES}")
        return v


class ClassBalance(BaseModel):
    applicable: bool = False
    imbalance_ratio: Optional[float] = None
    strategy: Optional[str] = None
    smote_variant: Optional[str] = None

    @field_validator("smote_variant")
    @classmethod
    def check_smote(cls, v):
        if v not in SMOTE_VARIANTS:
            raise ValueError(f"smote_variant '{v}' not in allow-list {SMOTE_VARIANTS}")
        return v

    @field_validator("strategy")
    @classmethod
    def check_strategy(cls, v):
        if v not in CLASS_BALANCE_STRATEGIES:
            raise ValueError(f"class_balance strategy '{v}' not in allow-list {CLASS_BALANCE_STRATEGIES}")
        return v


class CVConfig(BaseModel):
    strategy: str = "KFold"
    n_splits: int = Field(ge=2, le=20, default=5)
    n_repeats: int = Field(ge=1, le=10, default=1)

    @field_validator("strategy")
    @classmethod
    def check_strategy(cls, v):
        if v not in CV_STRATEGIES:
            raise ValueError(f"cv strategy '{v}' not in allow-list {CV_STRATEGIES}")
        return v


class SplitConfig(BaseModel):
    train: float = Field(gt=0, lt=1, default=0.6)
    val: float = Field(gt=0, lt=1, default=0.2)
    test: float = Field(gt=0, lt=1, default=0.2)

    @model_validator(mode="after")
    def check_sum(self):
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"train/val/test must sum to 1.0, got {total}")
        return self


class MetricsConfig(BaseModel):
    primary: str
    reported: list = Field(default_factory=list)


class BaselineConfig(BaseModel):
    strategy: str = "mean"


class PipelineConfig(BaseModel):
    json_schema_version: str = "1.0"
    seed: int = 7
    problem_type: Literal["regression", "classification"]
    target: str
    leakage_risk_columns: list[dict] = Field(default_factory=list)
    datetime_features: dict = Field(default_factory=dict)
    features: dict[str, FeatureSpec]
    class_balance: ClassBalance = Field(default_factory=ClassBalance)
    preprocessing_branches: dict = Field(default_factory=lambda: {"raw": {"enabled": True}})
    excluded_models: list[dict] = Field(default_factory=list)
    candidate_models: list[CandidateModel]
    cv: CVConfig = Field(default_factory=CVConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    ensemble: dict = Field(default_factory=lambda: {"stacking": True, "meta_learner": "Ridge", "use_top_n_models": 5})
    baseline: BaselineConfig = Field(default_factory=BaselineConfig, validate_default=True)
    metrics: MetricsConfig
    fallback_on_failure: dict = Field(default_factory=lambda: {
        "per_model": "skip_and_log_promote_next_rank",
        "all_models_fail": "abort_run_and_flag",
        "cv_fold_error_threshold": 1,
    })

    @field_validator("candidate_models")
    @classmethod
    def check_model_allowlist(cls, v, info):
        problem_type = info.data.get("problem_type")
        allowlist = REGRESSION_MODELS if problem_type == "regression" else CLASSIFICATION_MODELS
        for m in v:
            if m.name not in allowlist:
                raise ValueError(f"model '{m.name}' not valid for problem_type={problem_type}")
        ranks = [m.rank for m in v]
        if len(ranks) != len(set(ranks)):
            raise ValueError("duplicate ranks in candidate_models")
        return v

    @field_validator("metrics")
    @classmethod
    def check_metrics_primary(cls, v, info):
        problem_type = info.data.get("problem_type")
        allowlist = REGRESSION_METRICS if problem_type == "regression" else CLASSIFICATION_METRICS
        if v.primary not in allowlist:
            raise ValueError(f"metrics.primary '{v.primary}' not valid for problem_type={problem_type}, expected one of {allowlist}")
        return v

    @field_validator("baseline")
    @classmethod
    def check_baseline_strategy(cls, v, info):
        problem_type = info.data.get("problem_type")
        if problem_type == "classification":
            allowlist = BASELINE_CLASSIFICATION_STRATEGIES
            if v.strategy == "mean":
                # "mean" is BaselineConfig's own generic default (sensible for regression) -
                # silently resolve it to the classification-appropriate default instead of
                # rejecting a config that simply never mentioned baseline.strategy at all
                v.strategy = "most_frequent"
        else:
            allowlist = BASELINE_REGRESSION_STRATEGIES
        if v.strategy not in allowlist:
            raise ValueError(f"baseline.strategy '{v.strategy}' not valid for problem_type={problem_type}, expected one of {allowlist}")
        return v


def apply_guardrails(config: PipelineConfig, cardinality_high_threshold: int = 50) -> list[str]:
    corrections = []
    for name, feat in config.features.items():
        if feat.encoder == "OneHotEncoder" and feat.cardinality and feat.cardinality > cardinality_high_threshold:
            corrections.append(f"{name}: OneHotEncoder -> TargetEncoder (cardinality={feat.cardinality} > {cardinality_high_threshold})")
            feat.encoder = "TargetEncoder"
        if feat.dtype == "numeric" and feat.encoder is not None:
            corrections.append(f"{name}: dropped encoder '{feat.encoder}' on numeric feature")
            feat.encoder = None
    for cand in config.candidate_models:
        if cand.device == "gpu" and cand.name not in GPU_CAPABLE_MODELS:
            corrections.append(f"{cand.name}: device 'gpu' -> 'cpu' (no GPU code path for this model)")
            cand.device = "cpu"
    if config.class_balance.smote_variant == "SMOTENC":
        # SMOTENC needs categorical_features indices identifying still-categorical columns,
        # but this pipeline's ColumnTransformer always encodes categoricals to numeric before
        # resampling runs - SMOTENC's core assumption doesn't hold, so fall back to plain SMOTE
        corrections.append("class_balance.smote_variant: SMOTENC -> SMOTE (features are already numeric-encoded by the time resampling runs)")
        config.class_balance.smote_variant = "SMOTE"
    return corrections
