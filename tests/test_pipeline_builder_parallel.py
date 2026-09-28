import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

import pipeline_builder as pb


def test_thread_budget_splits_cores_between_folds_and_model():
    outer, inner = pb._thread_budget(5)
    assert outer == 5
    assert outer * inner <= (pb.os.cpu_count() or 4) + inner  # bounded, not oversubscribed by orders of magnitude
    assert inner >= 1


def test_thread_budget_falls_back_to_one_when_units_exceed_cores(monkeypatch):
    monkeypatch.setattr(pb.os, "cpu_count", lambda: 4)
    outer, inner = pb._thread_budget(100)
    assert outer == 4
    assert inner == 1


def _make_regression_frame(n=120, seed=0):
    rng = np.random.RandomState(seed)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = pd.Series(2 * X["a"] - X["b"] + rng.normal(scale=0.1, size=n), name="y")
    return X, y


def test_cv_predict_metric_parallel_matches_sequential():
    X, y = _make_regression_frame()
    pipe = Pipeline([("model", Ridge(alpha=1.0, random_state=7))])
    cv = KFold(n_splits=5, shuffle=True, random_state=7)

    sequential_score = pb._cv_predict_metric(pipe, X, y, cv, "regression", n_jobs=1)
    parallel_score = pb._cv_predict_metric(pipe, X, y, cv, "regression", n_jobs=4)

    assert sequential_score == parallel_score


def test_cv_predict_metric_does_not_mutate_shared_pipe_instance():
    X, y = _make_regression_frame()
    pipe = Pipeline([("model", Ridge(alpha=1.0, random_state=7))])
    cv = KFold(n_splits=5, shuffle=True, random_state=7)

    pb._cv_predict_metric(pipe, X, y, cv, "regression", n_jobs=4)
    # the shared pipe object itself must remain unfitted - folds must clone, not fit in place
    assert not hasattr(pipe.named_steps["model"], "coef_")
