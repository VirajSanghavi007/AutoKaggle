import inspect
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

from schema import PipelineConfig
import nn_model
import pipeline_builder as _pipeline_builder  # only for inspect.getsource reuse below, never called

_ENCODER_IMPORTS = {
    "OneHotEncoder": "from sklearn.preprocessing import OneHotEncoder\n",
    "OrdinalEncoder": "from sklearn.preprocessing import OrdinalEncoder\n",
    "TargetEncoder": "from category_encoders import TargetEncoder\n",
    "BinaryEncoder": "from category_encoders import BinaryEncoder\n",
    "FrequencyEncoder": "from category_encoders import CountEncoder\n",
    "CatBoostEncoder": "from category_encoders import CatBoostEncoder\n",
    "HashingEncoder": "from category_encoders import HashingEncoder\n",
}
_ENCODER_EXPR = {
    "OneHotEncoder": "OneHotEncoder(handle_unknown='ignore')",
    "OrdinalEncoder": "OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)",
    "TargetEncoder": "TargetEncoder()",
    "BinaryEncoder": "BinaryEncoder()",
    "FrequencyEncoder": "CountEncoder(normalize=True)",
    "CatBoostEncoder": "CatBoostEncoder()",
    "HashingEncoder": "HashingEncoder()",
}
_IMPUTER_IMPORTS = {
    "SimpleImputer_mean": "from sklearn.impute import SimpleImputer\n",
    "SimpleImputer_median": "from sklearn.impute import SimpleImputer\n",
    "SimpleImputer_most_frequent": "from sklearn.impute import SimpleImputer\n",
    "SimpleImputer_constant": "from sklearn.impute import SimpleImputer\n",
    "KNNImputer": "from sklearn.impute import KNNImputer\n",
    "IterativeImputer": "from sklearn.experimental import enable_iterative_imputer\nfrom sklearn.impute import IterativeImputer\n",
}
_IMPUTER_EXPR = {
    "SimpleImputer_mean": "SimpleImputer(strategy='mean')",
    "SimpleImputer_median": "SimpleImputer(strategy='median')",
    "SimpleImputer_most_frequent": "SimpleImputer(strategy='most_frequent')",
    "SimpleImputer_constant": "SimpleImputer(strategy='constant', fill_value='missing')",
    "KNNImputer": "KNNImputer()",
    "IterativeImputer": "IterativeImputer()",
}
_SCALER_IMPORT = "from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler, PowerTransformer, QuantileTransformer\n"

_SMOTE_IMPORTS = {
    "SMOTE": "from imblearn.over_sampling import SMOTE\n",
    "BorderlineSMOTE": "from imblearn.over_sampling import BorderlineSMOTE\n",
    "ADASYN": "from imblearn.over_sampling import ADASYN\n",
    "SMOTEN": "from imblearn.over_sampling import SMOTEN\n",
    "SMOTETomek": "from imblearn.combine import SMOTETomek\n",
    "SMOTEENN": "from imblearn.combine import SMOTEENN\n",
    "RandomUnderSampler": "from imblearn.under_sampling import RandomUnderSampler\n",
}
_SMOTE_EXPR = {
    "SMOTE": "SMOTE(random_state=SEED)",
    "BorderlineSMOTE": "BorderlineSMOTE(random_state=SEED)",
    "ADASYN": "ADASYN(random_state=SEED)",
    "SMOTEN": "SMOTEN(random_state=SEED)",
    "SMOTETomek": "SMOTETomek(random_state=SEED)",
    "SMOTEENN": "SMOTEENN(random_state=SEED)",
    "RandomUnderSampler": "RandomUnderSampler(random_state=SEED)",
}

_MODEL_IMPORTS = {
    "lightgbm": "import lightgbm as lgb\n",
    "xgboost": "import xgboost as xgb\n",
    "catboost": "from catboost import CatBoostRegressor, CatBoostClassifier\n",
    "random_forest": "from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier\n",
    "extra_trees": "from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier\n",
    "elasticnet": "from sklearn.linear_model import ElasticNet\n",
    "ridge": "from sklearn.linear_model import Ridge\n",
    "ridge_classifier": "from sklearn.linear_model import RidgeClassifier\n",
    "lasso": "from sklearn.linear_model import Lasso\n",
    "linear_regression": "from sklearn.linear_model import LinearRegression\n",
    "logistic_regression": "from sklearn.linear_model import LogisticRegression\n",
    "naive_bayes": "from sklearn.naive_bayes import GaussianNB\n",
    "knn": "from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier\n",
    "svr": "from sklearn.svm import SVR\n",
    "svc": "from sklearn.svm import SVC\n",
    # neural_net has no import line: its class is inlined via _nn_model_source_cell()
}
_MODEL_CLASS = {
    "lightgbm": ("lgb.LGBMRegressor", "lgb.LGBMClassifier"),
    "xgboost": ("xgb.XGBRegressor", "xgb.XGBClassifier"),
    "catboost": ("CatBoostRegressor", "CatBoostClassifier"),
    "random_forest": ("RandomForestRegressor", "RandomForestClassifier"),
    "extra_trees": ("ExtraTreesRegressor", "ExtraTreesClassifier"),
    "elasticnet": ("ElasticNet", "ElasticNet"),
    "ridge": ("Ridge", "Ridge"),
    "ridge_classifier": ("RidgeClassifier", "RidgeClassifier"),
    "lasso": ("Lasso", "Lasso"),
    "linear_regression": ("LinearRegression", "LinearRegression"),
    "logistic_regression": ("LogisticRegression", "LogisticRegression"),
    "naive_bayes": ("GaussianNB", "GaussianNB"),
    "knn": ("KNeighborsRegressor", "KNeighborsClassifier"),
    "svr": ("SVR", "SVR"),
    "svc": ("SVC", "SVC"),
    "neural_net": ("TorchMLPRegressor", "TorchMLPClassifier"),
}
# models with a real GPU code path - kept in sync with schema.GPU_CAPABLE_MODELS
_GPU_CAPABLE_MODELS = {"lightgbm", "xgboost", "catboost", "neural_net"}


def _nn_model_source_cell() -> str:
    """Notebooks published to the Kaggle repo are self-contained (no access to this
    repo's src/), so inline nn_model's classes as source instead of importing them."""
    return (
        "import numpy as np\n"
        "from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin\n\n"
        + inspect.getsource(nn_model.resolve_device) + "\n"
        + inspect.getsource(nn_model.TorchMLP) + "\n"
        + inspect.getsource(nn_model.TorchMLPRegressor) + "\n"
        + inspect.getsource(nn_model.TorchMLPClassifier)
    )


def _outlier_clipper_source_cell() -> str:
    """Same reuse-not-duplicate approach as _nn_model_source_cell: inline the real
    _PercentileClipper class so the published notebook can't drift from what
    build_column_transformer actually used during benchmarking."""
    return (
        "from sklearn.base import BaseEstimator, TransformerMixin\n\n"
        + inspect.getsource(_pipeline_builder._PercentileClipper)
    )


def _any_feature_uses_outlier_clip(config: PipelineConfig) -> bool:
    return any(
        feat.dtype == "numeric" and feat.outlier_handling and feat.outlier_handling.get("method") == "clip"
        for feat in config.features.values()
    )


def _render_datetime_expansion_code(config: PipelineConfig) -> str:
    """Mirrors pipeline_builder.expand_datetime_features exactly (year/month/day/
    dayofweek derived columns) - without this, the notebook's ColumnTransformer
    references derived columns (e.g. 'signup_year') that were never computed from
    the raw CSV, and fails with a 'not columns of the dataframe' error at fit time."""
    lines = []
    for col in config.datetime_features:
        lines.append(f"df['{col}'] = pd.to_datetime(df['{col}'], errors='coerce')")
        for part in ("year", "month", "day", "dayofweek"):
            lines.append(f"df['{col}_{part}'] = df['{col}'].dt.{part}")
    return "\n".join(lines) + ("\n" if lines else "")


def _inject_device_params(name: str, params: dict, gpu_used: bool) -> dict:
    """gpu_used must be the value pipeline_builder actually recorded during benchmarking
    (bench result's 'gpu_used' key) rather than recomputed here - some libraries' plain
    pip wheels (lightgbm in particular) lack GPU support even when CUDA itself is present,
    and only the benchmarking run's own probe knows which models actually got GPU."""
    if name not in _GPU_CAPABLE_MODELS:
        return params
    params = dict(params)
    gpu_ok = gpu_used
    if name == "lightgbm":
        params["device"] = "gpu" if gpu_ok else "cpu"
    elif name == "xgboost":
        params["tree_method"] = "hist"
        params["device"] = "cuda" if gpu_ok else "cpu"
    elif name == "catboost":
        params["task_type"] = "GPU" if gpu_ok else "CPU"
    elif name == "neural_net":
        params["device"] = "cuda" if gpu_ok else "cpu"
    return params


def _model_import_line(name: str) -> str:
    return _MODEL_IMPORTS[name]


_CLASS_WEIGHT_CAPABLE_MODELS = {"logistic_regression", "ridge_classifier", "random_forest", "extra_trees", "svc"}


def _class_weight_balanced_in_use(config: PipelineConfig) -> bool:
    """Pure function of config (unlike gpu_used, no runtime probing involved), so
    unlike the GPU case this is safe to recompute here rather than needing a value
    recorded in the bench result - must match pipeline_builder.run_benchmark's
    identical computation exactly."""
    return (
        config.problem_type == "classification"
        and config.class_balance.applicable
        and not config.class_balance.smote_variant
        and config.class_balance.strategy == "class_weight"
    )


def _inject_class_weight_param(name: str, params: dict, config: PipelineConfig) -> dict:
    if name not in _CLASS_WEIGHT_CAPABLE_MODELS or not _class_weight_balanced_in_use(config):
        return params
    params = dict(params)
    params["class_weight"] = "balanced"
    return params


def _model_class_name(name: str, problem_type: str) -> str:
    reg_name, clf_name = _MODEL_CLASS[name]
    return reg_name if problem_type == "regression" else clf_name


def _render_model_code(name: str, problem_type: str, params: dict, var_name: str) -> str:
    cls_name = _model_class_name(name, problem_type)
    params = dict(params)
    if name not in ("naive_bayes", "linear_regression"):
        params.setdefault("random_state", "SEED")
    params_str = ", ".join(
        f"{k}=SEED" if k == "random_state" else f"{k}={v!r}"
        for k, v in params.items()
    )
    return f"{var_name} = {cls_name}({params_str})\n"


def _smote_variant_in_use(config: PipelineConfig) -> str | None:
    if config.problem_type == "classification" and config.class_balance.applicable and config.class_balance.smote_variant:
        return config.class_balance.smote_variant
    return None


def _pipeline_class_name(config: PipelineConfig) -> str:
    return "ImbPipeline" if _smote_variant_in_use(config) else "Pipeline"


def _render_estimator_pipeline_code(config: PipelineConfig, model_var: str, use_pca: bool, pipe_var: str) -> str:
    """Mirrors pipeline_builder._build_estimator_pipeline's step order exactly (prep ->
    optional pca -> optional resample -> model), so the published notebook matches what
    was actually benchmarked instead of silently dropping the PCA/SMOTE steps."""
    steps = ["('prep', preprocessor)"]
    if use_pca:
        n_components = config.preprocessing_branches.get("pca", {}).get("n_components", 0.95)
        steps.append(f"('pca', PCA(n_components={n_components!r}, random_state=SEED))")
    smote_variant = _smote_variant_in_use(config)
    if smote_variant:
        steps.append(f"('resample', {_SMOTE_EXPR[smote_variant]})")
    steps.append(f"('model', {model_var})")
    cls_name = _pipeline_class_name(config)
    return f"{pipe_var} = {cls_name}([{', '.join(steps)}])\n"


def _render_column_transformer_code(config: PipelineConfig) -> str:
    lines = ["transformers = []"]
    for name, feat in config.features.items():
        steps = [f"('imputer', {_IMPUTER_EXPR[feat.imputer]})"]
        if feat.dtype == "numeric" and feat.outlier_handling and feat.outlier_handling.get("method") == "clip":
            lower_pct = feat.outlier_handling.get("lower_pct", 1)
            upper_pct = feat.outlier_handling.get("upper_pct", 99)
            steps.append(f"('outlier_clip', _PercentileClipper(lower_pct={lower_pct!r}, upper_pct={upper_pct!r}))")
        if feat.encoder:
            steps.append(f"('encoder', {_ENCODER_EXPR[feat.encoder]})")
        if feat.scaler:
            steps.append(f"('scaler', {feat.scaler}())")
        lines.append(f"transformers.append(('{name}', Pipeline([{', '.join(steps)}]), ['{name}']))")
    lines.append("preprocessor = ColumnTransformer(transformers, remainder='drop')")
    return "\n".join(lines)


def _collect_imports(config: PipelineConfig, model_names: list[str], use_pca: bool) -> str:
    imports = [
        "import pandas as pd\nimport numpy as np\n",
        "from sklearn.compose import ColumnTransformer\n",
        "from sklearn.pipeline import Pipeline\n",
        _SCALER_IMPORT,
    ]
    if config.problem_type == "classification":
        imports.append("from sklearn.preprocessing import LabelEncoder\n")
    if use_pca:
        imports.append("from sklearn.decomposition import PCA\n")
    smote_variant = _smote_variant_in_use(config)
    if smote_variant:
        imports.append("from imblearn.pipeline import Pipeline as ImbPipeline\n")
        imports.append(_SMOTE_IMPORTS[smote_variant])
    seen_imputers = set()
    seen_encoders = set()
    for feat in config.features.values():
        if feat.imputer not in seen_imputers:
            imports.append(_IMPUTER_IMPORTS[feat.imputer])
            seen_imputers.add(feat.imputer)
        if feat.encoder and feat.encoder not in seen_encoders:
            imports.append(_ENCODER_IMPORTS[feat.encoder])
            seen_encoders.add(feat.encoder)
    seen_models = set()
    for m in model_names:
        if m not in seen_models:
            if m in _MODEL_IMPORTS:
                imports.append(_model_import_line(m))
            seen_models.add(m)
    if len(model_names) > 1:
        if config.problem_type == "regression":
            imports.append("from sklearn.ensemble import StackingRegressor\n")
        else:
            imports.append("from sklearn.ensemble import StackingClassifier\n")
    return "".join(imports)


def _meta_learner_expr(config: PipelineConfig) -> tuple[str, str]:
    """Mirrors pipeline_builder._build_stack's actual meta-learner choice - the previous
    codegen omitted final_estimator entirely, silently swapping in sklearn's stacking
    default instead of the meta-learner that was actually benchmarked."""
    if config.problem_type == "regression":
        if config.ensemble.get("meta_learner") == "Ridge":
            return "Ridge(random_state=SEED)", "from sklearn.linear_model import Ridge\n"
        return "LinearRegression()", "from sklearn.linear_model import LinearRegression\n"
    return "LogisticRegression(max_iter=2000, random_state=SEED)", "from sklearn.linear_model import LogisticRegression\n"


def build_notebook(config: PipelineConfig, best_result: dict, data_csv_name: str, title: str) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    cells = [nbf.v4.new_markdown_cell(f"# {title}\n\nAuto-generated by AutoKaggle v1. Seed={config.seed}.")]

    best = best_result["best"]
    is_stack = best["name"] == "stacking_ensemble"
    member_results = []
    if is_stack:
        members = best.get("members", [])
        member_results = [r for r in best_result["all_results"] if r["name"] in members]
        model_names = [r["name"] for r in member_results]
        use_pca = any(r.get("use_pca", False) for r in member_results)
    else:
        model_names = [best["name"]]
        use_pca = best.get("use_pca", False)

    imports = _collect_imports(config, model_names, use_pca)
    if is_stack:
        meta_learner_expr, meta_learner_import = _meta_learner_expr(config)
        imports += meta_learner_import
    cells.append(nbf.v4.new_code_cell(imports))
    if "neural_net" in model_names:
        cells.append(nbf.v4.new_code_cell(_nn_model_source_cell()))
    if _any_feature_uses_outlier_clip(config):
        cells.append(nbf.v4.new_code_cell(_outlier_clipper_source_cell()))
    cells.append(nbf.v4.new_code_cell(f"df = pd.read_csv('{data_csv_name}')\ndf.head()"))
    if config.datetime_features:
        cells.append(nbf.v4.new_code_cell(_render_datetime_expansion_code(config)))
    cells.append(nbf.v4.new_code_cell(f"SEED = {config.seed}\nTARGET = {config.target!r}\n"))
    cells.append(nbf.v4.new_code_cell(_render_column_transformer_code(config)))

    target_line = (
        "y = pd.Series(LabelEncoder().fit_transform(df[TARGET]), name=TARGET)\n"
        if config.problem_type == "classification"
        else "y = df[TARGET]\n"
    )

    if is_stack:
        model_lines = []
        estimator_names = []
        for i, r in enumerate(member_results):
            model_var = f"model_{i}_{r['name']}"
            pipe_var = f"pipe_{i}_{r['name']}"
            params = _inject_device_params(r["name"], r["best_params"], r.get("gpu_used", False))
            params = _inject_class_weight_param(r["name"], params, config)
            model_lines.append(_render_model_code(r["name"], config.problem_type, params, model_var))
            model_lines.append(_render_estimator_pipeline_code(config, model_var, r.get("use_pca", False), pipe_var))
            estimator_names.append((r["name"], pipe_var))
        estimators_str = ", ".join(f"('{n}', {v})" for n, v in estimator_names)
        stack_cls = "StackingRegressor" if config.problem_type == "regression" else "StackingClassifier"
        cells.append(nbf.v4.new_code_cell(
            "".join(model_lines) +
            f"\npipe = {stack_cls}(estimators=[{estimators_str}], final_estimator={meta_learner_expr})\n"
            f"X = df.drop(columns=[TARGET])\n{target_line}"
            "pipe.fit(X, y)\nprint('Stacking ensemble fit complete. CV/val score:', "
            f"{best['cv_score']!r}, 'baseline:', {best_result['baseline_score']!r})"
        ))
    else:
        params = _inject_device_params(best["name"], best["best_params"], best.get("gpu_used", False))
        params = _inject_class_weight_param(best["name"], params, config)
        model_code = _render_model_code(best["name"], config.problem_type, params, "model")
        pipe_code = _render_estimator_pipeline_code(config, "model", use_pca, "pipe")
        cells.append(nbf.v4.new_code_cell(
            model_code + pipe_code +
            f"\nX = df.drop(columns=[TARGET])\n{target_line}"
            "pipe.fit(X, y)\nprint('Fit complete. CV score from benchmarking:', "
            f"{best['cv_score']!r}, 'baseline:', {best_result['baseline_score']!r})"
        ))

    cells.append(nbf.v4.new_code_cell(_render_shap_cell()))

    nb["cells"] = cells
    return nb


def _render_shap_cell() -> str:
    """Model-agnostic SHAP feature importance (works for any of pipe/model/device
    combos above - tree, linear, NN, or stacking - since it explains via pipe.predict
    rather than a model-family-specific SHAP explainer)."""
    return (
        "import shap\n"
        "import matplotlib.pyplot as plt\n\n"
        "shap_sample = X.sample(min(100, len(X)), random_state=SEED)\n"
        "shap_explainer = shap.Explainer(pipe.predict, shap_sample)\n"
        "shap_values = shap_explainer(shap_sample)\n"
        "shap.plots.bar(shap_values, show=False)\n"
        "plt.tight_layout()\n"
        "plt.savefig('shap_feature_importance.png')\n"
        "plt.show()\n"
    )


def execute_notebook(nb: nbf.NotebookNode, working_dir: str) -> tuple[bool, str]:
    client = NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": working_dir}})
    try:
        client.execute()
        return True, ""
    except CellExecutionError as e:
        return False, str(e)
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def save_notebook(nb: nbf.NotebookNode, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, path)
