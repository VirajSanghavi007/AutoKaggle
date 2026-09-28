import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
from joblib import Parallel, delayed

from sklearn.base import clone, BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.preprocessing import (
    OneHotEncoder, OrdinalEncoder, StandardScaler, RobustScaler, MinMaxScaler,
    MaxAbsScaler, PowerTransformer, QuantileTransformer, LabelEncoder,
)
from sklearn.decomposition import PCA
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression, RidgeClassifier,
)
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier, ExtraTreesRegressor, ExtraTreesClassifier,
    StackingRegressor, StackingClassifier,
)
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.model_selection import (
    KFold, StratifiedKFold, RepeatedKFold, RepeatedStratifiedKFold, train_test_split,
)
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, roc_auc_score, f1_score, accuracy_score, log_loss,
)

from category_encoders import TargetEncoder, BinaryEncoder, CountEncoder, CatBoostEncoder, HashingEncoder

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN, SMOTENC, SMOTEN
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.under_sampling import RandomUnderSampler

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, CatBoostClassifier

from schema import PipelineConfig, FeatureSpec
from progress import OptunaProgressCallback
from nn_model import TorchMLPRegressor, TorchMLPClassifier, resolve_device

optuna.logging.set_verbosity(optuna.logging.WARNING)

_IMPUTER_MAP = {
    "SimpleImputer_mean": lambda: SimpleImputer(strategy="mean"),
    "SimpleImputer_median": lambda: SimpleImputer(strategy="median"),
    "SimpleImputer_most_frequent": lambda: SimpleImputer(strategy="most_frequent"),
    "SimpleImputer_constant": lambda: SimpleImputer(strategy="constant", fill_value="missing"),
    "KNNImputer": lambda: KNNImputer(),
    "IterativeImputer": lambda: IterativeImputer(),
}
_SCALER_MAP = {
    "StandardScaler": StandardScaler,
    "RobustScaler": RobustScaler,
    "MinMaxScaler": MinMaxScaler,
    "MaxAbsScaler": MaxAbsScaler,
    "PowerTransformer": PowerTransformer,
    "QuantileTransformer": QuantileTransformer,
    None: None,
}
_SMOTE_MAP = {
    "SMOTE": SMOTE,
    "BorderlineSMOTE": BorderlineSMOTE,
    "ADASYN": ADASYN,
    "SMOTENC": SMOTENC,
    "SMOTEN": SMOTEN,
    "SMOTETomek": SMOTETomek,
    "SMOTEENN": SMOTEENN,
    "RandomUnderSampler": RandomUnderSampler,
    None: None,
}


class _PercentileClipper(BaseEstimator, TransformerMixin):
    """Winsorizes a numeric feature to percentile bounds learned on the training fold -
    implements FeatureSpec.outlier_handling, e.g. {"method": "clip", "lower_pct": 1,
    "upper_pct": 99}. Runs after imputation (so there are no NaNs left to percentile
    over) and before scaling."""

    def __init__(self, lower_pct=1, upper_pct=99):
        self.lower_pct = lower_pct
        self.upper_pct = upper_pct

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lower_ = np.percentile(X, self.lower_pct, axis=0)
        self.upper_ = np.percentile(X, self.upper_pct, axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.clip(X, self.lower_, self.upper_)


def _encoder_factory(name):
    if name == "OneHotEncoder":
        return OneHotEncoder(handle_unknown="ignore")
    if name == "OrdinalEncoder":
        return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    if name == "TargetEncoder":
        return TargetEncoder()
    if name == "BinaryEncoder":
        return BinaryEncoder()
    if name == "FrequencyEncoder":
        return CountEncoder(normalize=True)
    if name == "CatBoostEncoder":
        return CatBoostEncoder()
    if name == "HashingEncoder":
        return HashingEncoder()
    return None


def build_column_transformer(config: PipelineConfig) -> ColumnTransformer:
    transformers = []
    for name, feat in config.features.items():
        steps = [("imputer", _IMPUTER_MAP[feat.imputer]())]
        if feat.dtype == "numeric" and feat.outlier_handling and feat.outlier_handling.get("method") == "clip":
            steps.append(("outlier_clip", _PercentileClipper(
                lower_pct=feat.outlier_handling.get("lower_pct", 1),
                upper_pct=feat.outlier_handling.get("upper_pct", 99),
            )))
        encoder = _encoder_factory(feat.encoder)
        if encoder is not None:
            steps.append(("encoder", encoder))
        scaler_cls = _SCALER_MAP[feat.scaler]
        if scaler_cls is not None:
            steps.append(("scaler", scaler_cls()))
        transformers.append((name, Pipeline(steps), [name]))
    return ColumnTransformer(transformers, remainder="drop")


def expand_datetime_features(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Derives year/month/day/dayofweek numeric columns from each column listed in
    config.datetime_features and registers a FeatureSpec for each derived column so
    build_column_transformer picks them up. The raw datetime column itself is left out
    of config.features (as it already was), so ColumnTransformer's remainder='drop'
    excludes it automatically."""
    if not config.datetime_features:
        return df
    df = df.copy()
    for col in config.datetime_features:
        if col not in df.columns:
            continue
        parsed = pd.to_datetime(df[col], errors="coerce")
        for part in ("year", "month", "day", "dayofweek"):
            new_col = f"{col}_{part}"
            df[new_col] = getattr(parsed.dt, part)
            if new_col not in config.features:
                config.features[new_col] = FeatureSpec(dtype="numeric", imputer="SimpleImputer_median")
    return df


def build_cv(config: PipelineConfig, problem_type: str):
    strategy = config.cv.strategy
    n_splits = config.cv.n_splits
    n_repeats = config.cv.n_repeats
    seed = config.seed
    if strategy == "KFold":
        return KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    if strategy == "StratifiedKFold":
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    if strategy == "RepeatedKFold":
        return RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    if strategy == "RepeatedStratifiedKFold":
        return RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    raise ValueError(f"unknown cv strategy: {strategy}")


def _thread_budget(n_parallel_units: int) -> tuple[int, int]:
    """Split available cores between an outer parallel unit (e.g. CV folds) and each
    unit's own internal parallelism (e.g. a forest's tree-building), so the product
    stays near cpu_count instead of oversubscribing when both run in threads at once."""
    total_cores = os.cpu_count() or 4
    outer_jobs = max(1, min(n_parallel_units, total_cores))
    inner_jobs = max(1, total_cores // outer_jobs)
    return outer_jobs, inner_jobs


_GPU_BUILD_SUPPORT_CACHE: dict[str, bool] = {}


def _library_build_supports_gpu(name: str) -> bool:
    """A CUDA device existing is not the same as this installed library's build
    supporting it - the plain pip lightgbm wheel has no GPU tree learner at all and
    raises at fit time (`GPU Tree Learner was not enabled in this build`), even on a
    machine with a working GPU that xgboost/catboost/torch can all use fine. Probe
    once per library per process with a throwaway fit and cache the result."""
    if name in _GPU_BUILD_SUPPORT_CACHE:
        return _GPU_BUILD_SUPPORT_CACHE[name]
    X_probe = np.zeros((10, 2), dtype=np.float32)
    y_probe = np.zeros(10, dtype=np.float32)
    ok = False
    try:
        if name == "lightgbm":
            lgb.LGBMRegressor(n_estimators=1, device="gpu", verbosity=-1).fit(X_probe, y_probe)
            ok = True
        elif name == "xgboost":
            xgb.XGBRegressor(n_estimators=1, tree_method="hist", device="cuda").fit(X_probe, y_probe)
            ok = True
        elif name == "catboost":
            CatBoostRegressor(iterations=1, task_type="GPU", verbose=False).fit(X_probe, y_probe)
            ok = True
    except Exception:
        ok = False
    _GPU_BUILD_SUPPORT_CACHE[name] = ok
    return ok


def _gpu_ok(device: str, name: str) -> bool:
    """The LLM's device request only takes effect if a real GPU is actually usable -
    otherwise every model silently runs on CPU instead of erroring at fit time."""
    if device != "gpu" or resolve_device("gpu") != "cuda":
        return False
    if name == "neural_net":
        return True  # torch's own cuda check above is authoritative here
    return _library_build_supports_gpu(name)


def _model_and_space(name: str, problem_type: str, seed: int, trial: optuna.Trial, n_jobs: int = 1,
                      device: str = "cpu", class_weight_balanced: bool = False):
    is_reg = problem_type == "regression"
    gpu = _gpu_ok(device, name)
    class_weight = "balanced" if (class_weight_balanced and not is_reg) else None
    if name == "lightgbm":
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 800),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 255),
            max_depth=trial.suggest_int("max_depth", 3, 12),
            random_state=seed,
            n_jobs=n_jobs,
            device="gpu" if gpu else "cpu",
        )
        return lgb.LGBMRegressor(**params) if is_reg else lgb.LGBMClassifier(**params)
    if name == "xgboost":
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 800),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            random_state=seed,
            n_jobs=n_jobs,
            tree_method="hist",
            device="cuda" if gpu else "cpu",
        )
        return xgb.XGBRegressor(**params) if is_reg else xgb.XGBClassifier(**params)
    if name == "catboost":
        params = dict(
            iterations=trial.suggest_int("iterations", 100, 600),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            depth=trial.suggest_int("depth", 3, 10),
            random_seed=seed, verbose=False,
            thread_count=n_jobs,
            task_type="GPU" if gpu else "CPU",
        )
        return CatBoostRegressor(**params) if is_reg else CatBoostClassifier(**params)
    if name == "neural_net":
        params = dict(
            hidden_dim=trial.suggest_categorical("hidden_dim", [32, 64, 128]),
            n_layers=trial.suggest_int("n_layers", 1, 3),
            learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            epochs=trial.suggest_int("epochs", 30, 150),
            batch_size=trial.suggest_categorical("batch_size", [32, 64, 128]),
            device="gpu" if gpu else "cpu",
            random_state=seed,
        )
        return TorchMLPRegressor(**params) if is_reg else TorchMLPClassifier(**params)
    if name == "random_forest":
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 600),
            max_depth=trial.suggest_int("max_depth", 3, 20),
            random_state=seed,
            n_jobs=n_jobs,
        )
        if not is_reg:
            params["class_weight"] = class_weight
        return RandomForestRegressor(**params) if is_reg else RandomForestClassifier(**params)
    if name == "extra_trees":
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 600),
            max_depth=trial.suggest_int("max_depth", 3, 20),
            random_state=seed,
            n_jobs=n_jobs,
        )
        if not is_reg:
            params["class_weight"] = class_weight
        return ExtraTreesRegressor(**params) if is_reg else ExtraTreesClassifier(**params)
    if name == "elasticnet":
        return ElasticNet(alpha=trial.suggest_float("alpha", 1e-4, 10, log=True),
                           l1_ratio=trial.suggest_float("l1_ratio", 0.0, 1.0), random_state=seed)
    if name == "ridge":
        return Ridge(alpha=trial.suggest_float("alpha", 1e-3, 100, log=True), random_state=seed)
    if name == "ridge_classifier":
        return RidgeClassifier(alpha=trial.suggest_float("alpha", 1e-3, 100, log=True), random_state=seed,
                                class_weight=class_weight)
    if name == "lasso":
        return Lasso(alpha=trial.suggest_float("alpha", 1e-4, 10, log=True), random_state=seed)
    if name == "linear_regression":
        return LinearRegression()
    if name == "logistic_regression":
        return LogisticRegression(C=trial.suggest_float("C", 1e-3, 100, log=True), max_iter=2000, random_state=seed,
                                   class_weight=class_weight)
    if name == "naive_bayes":
        return GaussianNB()
    if name == "knn":
        n_neighbors = trial.suggest_int("n_neighbors", 3, 30)
        return (
            KNeighborsRegressor(n_neighbors=n_neighbors, n_jobs=n_jobs) if is_reg
            else KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=n_jobs)
        )
    if name == "svr":
        return SVR(C=trial.suggest_float("C", 1e-2, 100, log=True))
    if name == "svc":
        return SVC(C=trial.suggest_float("C", 1e-2, 100, log=True), probability=True, class_weight=class_weight)
    raise ValueError(f"unknown model name: {name}")


def _predict_for_metric(pipe, X, problem_type, n_classes, metric="rmse"):
    if problem_type == "regression":
        return pipe.predict(X)
    if metric in ("accuracy", "f1"):
        return pipe.predict(X)
    # roc_auc / log_loss need probability-like scores, not hard labels
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X)
        if metric == "log_loss":
            return proba
        return proba[:, 1] if n_classes == 2 else proba
    if hasattr(pipe, "decision_function"):
        return pipe.decision_function(X)
    return pipe.predict(X)


def _score_from_predictions(y_true, y_score, problem_type, n_classes, metric="rmse"):
    """Lower is always better here (see CLAUDE.md's Metric convention), so every
    metric that's naturally 'higher is better' (roc_auc, f1, accuracy) is negated."""
    if problem_type == "regression":
        if metric == "mae":
            return float(mean_absolute_error(y_true, y_score))
        if metric == "rmsle":
            y_true_c = np.clip(np.asarray(y_true, dtype=float), 0, None)
            y_score_c = np.clip(np.asarray(y_score, dtype=float), 0, None)
            return float(np.sqrt(mean_squared_error(np.log1p(y_true_c), np.log1p(y_score_c))))
        return float(np.sqrt(mean_squared_error(y_true, y_score)))  # rmse (default)

    if metric == "accuracy":
        return -float(accuracy_score(y_true, y_score))
    if metric == "f1":
        average = "binary" if n_classes == 2 else "macro"
        return -float(f1_score(y_true, y_score, average=average))
    if metric == "log_loss":
        return float(log_loss(y_true, y_score))
    # roc_auc (default)
    if n_classes == 2:
        return -float(roc_auc_score(y_true, y_score))
    return -float(roc_auc_score(y_true, y_score, multi_class="ovr", average="macro"))


def _build_estimator_pipeline(ct, model, config: PipelineConfig, use_pca: bool):
    steps = [("prep", ct)]
    if use_pca:
        n_components = config.preprocessing_branches.get("pca", {}).get("n_components", 0.95)
        steps.append(("pca", PCA(n_components=n_components, random_state=config.seed)))
    smote_variant = config.class_balance.smote_variant
    if config.problem_type == "classification" and config.class_balance.applicable and smote_variant:
        sampler_cls = _SMOTE_MAP[smote_variant]
        steps.append(("resample", sampler_cls(random_state=config.seed)))
        steps.append(("model", model))
        return ImbPipeline(steps)
    steps.append(("model", model))
    return Pipeline(steps)


def _fit_score_fold(pipe, X, y, train_idx, test_idx, problem_type, n_classes, metric):
    # clone first: concurrent folds must not share one pipeline's mutable fitted state
    fold_pipe = clone(pipe)
    fold_pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
    y_score = _predict_for_metric(fold_pipe, X.iloc[test_idx], problem_type, n_classes, metric)
    return _score_from_predictions(y.iloc[test_idx], y_score, problem_type, n_classes, metric)


def _safe_fit_score_fold(pipe, X, y, train_idx, test_idx, problem_type, n_classes, metric):
    try:
        return _fit_score_fold(pipe, X, y, train_idx, test_idx, problem_type, n_classes, metric)
    except Exception:
        return None


def _cv_predict_metric(pipe, X, y, cv, problem_type, n_jobs: int = 1, metric: str = "rmse",
                        fold_error_threshold: int = 0):
    n_classes = y.nunique() if problem_type == "classification" else None
    split_arg = y if problem_type == "classification" else None
    splits = list(cv.split(X, split_arg))

    if fold_error_threshold <= 0:
        # fast path: let a fold's exception propagate immediately, same as before
        if n_jobs == 1:
            scores = [
                _fit_score_fold(pipe, X, y, train_idx, test_idx, problem_type, n_classes, metric)
                for train_idx, test_idx in splits
            ]
        else:
            scores = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_fit_score_fold)(pipe, X, y, train_idx, test_idx, problem_type, n_classes, metric)
                for train_idx, test_idx in splits
            )
        return float(np.mean(scores))

    # tolerate up to fold_error_threshold individual fold failures (config.fallback_on_failure
    # .cv_fold_error_threshold) instead of aborting the whole model on one bad fold
    if n_jobs == 1:
        raw = [
            _safe_fit_score_fold(pipe, X, y, train_idx, test_idx, problem_type, n_classes, metric)
            for train_idx, test_idx in splits
        ]
    else:
        raw = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_safe_fit_score_fold)(pipe, X, y, train_idx, test_idx, problem_type, n_classes, metric)
            for train_idx, test_idx in splits
        )
    scores = [s for s in raw if s is not None]
    failures = len(raw) - len(scores)
    if failures > fold_error_threshold or not scores:
        raise RuntimeError(f"{failures}/{len(raw)} CV folds failed, exceeding cv_fold_error_threshold={fold_error_threshold}")
    return float(np.mean(scores))


def _split_train_val_test(X, y, config: PipelineConfig):
    stratify = y if config.problem_type == "classification" else None
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=config.split.test, random_state=config.seed, stratify=stratify,
    )
    val_frac_of_trainval = config.split.val / (config.split.train + config.split.val)
    stratify2 = y_trainval if config.problem_type == "classification" else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_frac_of_trainval, random_state=config.seed, stratify=stratify2,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def _choose_pca_branch(cand, X_train, y_train, ct, cv, config: PipelineConfig, seed: int,
                        cv_n_jobs: int = 1, model_n_jobs: int = 1, metric: str = "rmse",
                        fold_error_threshold: int = 0, class_weight_balanced: bool = False) -> bool:
    if not config.preprocessing_branches.get("pca", {}).get("enabled", False):
        return False
    # FixedTrial({}) can't answer suggest_*() calls with no values to draw from - use a
    # real trial so both branches get an actual (identical, since the trial is reused)
    # sampled hyperparameter set instead of crashing on the first suggest_int/suggest_float
    trial = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=seed)).ask()
    model_raw = _model_and_space(cand.name, config.problem_type, seed, trial, n_jobs=model_n_jobs,
                                  device=cand.device, class_weight_balanced=class_weight_balanced)
    model_pca = _model_and_space(cand.name, config.problem_type, seed, trial, n_jobs=model_n_jobs,
                                  device=cand.device, class_weight_balanced=class_weight_balanced)
    pipe_raw = _build_estimator_pipeline(ct, model_raw, config, use_pca=False)
    pipe_pca = _build_estimator_pipeline(ct, model_pca, config, use_pca=True)
    score_raw = _cv_predict_metric(pipe_raw, X_train, y_train, cv, config.problem_type, n_jobs=cv_n_jobs,
                                    metric=metric, fold_error_threshold=fold_error_threshold)
    score_pca = _cv_predict_metric(pipe_pca, X_train, y_train, cv, config.problem_type, n_jobs=cv_n_jobs,
                                    metric=metric, fold_error_threshold=fold_error_threshold)
    return score_pca < score_raw


def _tune_one_model(cand, X_train, y_train, ct, cv, config: PipelineConfig, seed: int, use_pca: bool, log=None,
                     cv_n_jobs: int = 1, model_n_jobs: int = 1, metric: str = "rmse",
                     fold_error_threshold: int = 0, class_weight_balanced: bool = False) -> dict:
    # a GPU is a single shared device: don't also thread-parallelize folds/models onto it,
    # rely on the GPU's own speedup instead of CPU-style fan-out
    if _gpu_ok(cand.device, cand.name):
        cv_n_jobs, model_n_jobs = 1, 1

    sample_frac = cand.sample_frac_for_optuna
    if sample_frac < 1.0:
        X_sample = X_train.sample(frac=sample_frac, random_state=seed)
        y_sample = y_train.loc[X_sample.index]
    else:
        X_sample, y_sample = X_train, y_train

    def objective(trial):
        model = _model_and_space(cand.name, config.problem_type, seed, trial, n_jobs=model_n_jobs,
                                  device=cand.device, class_weight_balanced=class_weight_balanced)
        pipe = _build_estimator_pipeline(ct, model, config, use_pca)
        return _cv_predict_metric(pipe, X_sample, y_sample, cv, config.problem_type, n_jobs=cv_n_jobs,
                                   metric=metric, fold_error_threshold=fold_error_threshold)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    t0 = time.time()
    callbacks = [OptunaProgressCallback(cand.optuna_trials, cand.name, log)] if log else []
    study.optimize(objective, n_trials=cand.optuna_trials, timeout=900, callbacks=callbacks)

    best_trial = optuna.trial.FixedTrial(study.best_params)
    final_model = _model_and_space(cand.name, config.problem_type, seed, best_trial, n_jobs=model_n_jobs,
                                    device=cand.device, class_weight_balanced=class_weight_balanced)
    final_pipe = _build_estimator_pipeline(ct, final_model, config, use_pca)
    final_score = _cv_predict_metric(final_pipe, X_train, y_train, cv, config.problem_type, n_jobs=cv_n_jobs,
                                      metric=metric, fold_error_threshold=fold_error_threshold)

    return {
        "name": cand.name,
        "best_params": study.best_params,
        "cv_score": final_score,
        "score_kind": "cv",
        "use_pca": use_pca,
        "device": cand.device,
        "gpu_used": _gpu_ok(cand.device, cand.name),
        "optuna_seconds": time.time() - t0,
        "n_trials_run": len(study.trials),
    }


def _build_stack(top_results, ct, config: PipelineConfig, use_pca: bool, class_weight_balanced: bool = False):
    # single fit, not inside the fold-parallel CV loop - safe to use all cores per model
    estimators = []
    for r in top_results:
        trial = optuna.trial.FixedTrial(r["best_params"])
        model = _model_and_space(r["name"], config.problem_type, config.seed, trial,
                                  n_jobs=-1, device=r.get("device", "cpu"),
                                  class_weight_balanced=class_weight_balanced)
        pipe = _build_estimator_pipeline(ct, model, config, r.get("use_pca", use_pca))
        estimators.append((r["name"], pipe))

    if config.problem_type == "regression":
        final_estimator = Ridge(random_state=config.seed) if config.ensemble.get("meta_learner") == "Ridge" else LinearRegression()
        return StackingRegressor(estimators=estimators, final_estimator=final_estimator)
    final_estimator = LogisticRegression(max_iter=2000, random_state=config.seed)
    return StackingClassifier(estimators=estimators, final_estimator=final_estimator)


def run_benchmark(df: pd.DataFrame, config: PipelineConfig, checkpoint_dir: str, log=None) -> dict:
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    seed = config.seed
    metric = config.metrics.primary
    fold_error_threshold = config.fallback_on_failure.get("cv_fold_error_threshold", 0)
    class_weight_balanced = (
        config.problem_type == "classification"
        and config.class_balance.applicable
        and not config.class_balance.smote_variant
        and config.class_balance.strategy == "class_weight"
    )

    df = expand_datetime_features(df, config)

    y = df[config.target]
    if config.problem_type == "classification":
        # XGBoost (and some other classifiers) require labels 0..n-1; string/arbitrary
        # labels crash at fit time otherwise, so encode unconditionally for consistency
        y = pd.Series(LabelEncoder().fit_transform(y), index=y.index, name=y.name)
    X = df.drop(columns=[config.target])
    ct = build_column_transformer(config)
    cv = build_cv(config, config.problem_type)

    X_train, X_val, X_test, y_train, y_val, y_test = _split_train_val_test(X, y, config)

    n_folds = config.cv.n_splits * config.cv.n_repeats
    cv_n_jobs, model_n_jobs = _thread_budget(n_folds)
    if log:
        log.info("  parallelism: %d fold-jobs x %d threads/model (cpu_count=%d)",
                  cv_n_jobs, model_n_jobs, os.cpu_count() or 4)

    is_reg = config.problem_type == "regression"
    baseline_model = (
        DummyRegressor(strategy=config.baseline.strategy) if is_reg
        else DummyClassifier(strategy=config.baseline.strategy, random_state=seed)
    )
    baseline_pipe = Pipeline([("prep", ct), ("model", baseline_model)])
    baseline_score = _cv_predict_metric(baseline_pipe, X_train, y_train, cv, config.problem_type,
                                         n_jobs=cv_n_jobs, metric=metric, fold_error_threshold=fold_error_threshold)

    excluded_names = {d.get("name") for d in config.excluded_models}
    candidates = sorted(
        (c for c in config.candidate_models if c.name not in excluded_names),
        key=lambda c: c.rank,
    )
    if not candidates:
        raise RuntimeError("all candidate_models were removed by excluded_models - nothing to benchmark")
    top5, fallback = candidates[:5], candidates[5:]

    use_pca_global = None
    if top5:
        use_pca_global = _choose_pca_branch(top5[0], X_train, y_train, ct, cv, config, seed,
                                             cv_n_jobs=cv_n_jobs, model_n_jobs=model_n_jobs, metric=metric,
                                             fold_error_threshold=fold_error_threshold,
                                             class_weight_balanced=class_weight_balanced)

    results = []
    for i, cand in enumerate(top5):
        ckpt_path = Path(checkpoint_dir) / f"{cand.name}.json"
        if ckpt_path.exists():
            if log:
                log.info("  [%d/%d] %s - loaded from checkpoint", i + 1, len(top5), cand.name)
            results.append(json.loads(ckpt_path.read_text()))
            continue
        if log:
            log.info("  [%d/%d] tuning %s (%d optuna trials)", i + 1, len(top5), cand.name, cand.optuna_trials)
        try:
            best = _tune_one_model(cand, X_train, y_train, ct, cv, config, seed, use_pca_global or False, log,
                                    cv_n_jobs=cv_n_jobs, model_n_jobs=model_n_jobs, metric=metric,
                                    fold_error_threshold=fold_error_threshold,
                                    class_weight_balanced=class_weight_balanced)
            ckpt_path.write_text(json.dumps(best, default=str))
            results.append(best)
        except Exception as e:
            log_path = Path(checkpoint_dir) / f"{cand.name}_FAILED.log"
            log_path.write_text(str(e))
            if log:
                log.warning("  %s failed: %s", cand.name, e)
            if fallback:
                promoted = fallback.pop(0)
                top5.append(promoted)

    if not results:
        raise RuntimeError("all candidate models failed - aborting per fallback_on_failure.all_models_fail")

    results.sort(key=lambda r: r["cv_score"])
    best_single = results[0]

    n_classes_val = y_val.nunique() if config.problem_type == "classification" else None
    n_classes_test = y_test.nunique() if config.problem_type == "classification" else None

    # built unconditionally (not just inside the stacking branch) since the test-set
    # scoring below needs a fitted pipe for whichever single model actually wins
    single_pipe = _build_estimator_pipeline(
        ct,
        _model_and_space(best_single["name"], config.problem_type, seed,
                          optuna.trial.FixedTrial(best_single["best_params"]), n_jobs=-1,
                          device=best_single.get("device", "cpu"), class_weight_balanced=class_weight_balanced),
        config, best_single.get("use_pca", False),
    )
    single_pipe.fit(X_train, y_train)

    final_best = dict(best_single)
    final_pipe = single_pipe
    stack_result = None
    if config.ensemble.get("stacking", False) and len(results) >= 2:
        top_n = min(config.ensemble.get("use_top_n_models", 5), len(results))
        stack = _build_stack(results[:top_n], ct, config, use_pca_global or False,
                              class_weight_balanced=class_weight_balanced)
        stack.fit(X_train, y_train)
        stack_score_input = _predict_for_metric(stack, X_val, config.problem_type, n_classes_val, metric)
        stack_val_score = _score_from_predictions(y_val, stack_score_input, config.problem_type, n_classes_val, metric)

        single_score_input = _predict_for_metric(single_pipe, X_val, config.problem_type, n_classes_val, metric)
        single_val_score = _score_from_predictions(y_val, single_score_input, config.problem_type, n_classes_val, metric)

        stack_result = {"val_score": stack_val_score, "single_val_score": single_val_score}
        if stack_val_score < single_val_score:
            # note "cv_score" here is actually a held-out validation score, not a cross-validated
            # one (see "score_kind") - stacking is never itself cross-validated, only compared
            # against the single best model on the validation split
            final_best = {"name": "stacking_ensemble", "members": [r["name"] for r in results[:top_n]],
                          "cv_score": stack_val_score, "score_kind": "held_out_val",
                          "use_pca": use_pca_global or False}
            final_pipe = stack

    test_score_input = _predict_for_metric(final_pipe, X_test, config.problem_type, n_classes_test, metric)
    test_score = _score_from_predictions(y_test, test_score_input, config.problem_type, n_classes_test, metric)

    return {
        "best": final_best,
        "all_results": results,
        "baseline_score": baseline_score,
        "stack_comparison": stack_result,
        "use_pca": use_pca_global or False,
        "val_size": len(X_val),
        "test_size": len(X_test),
        "test_score": test_score,
        "metric": metric,
    }
