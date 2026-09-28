import numpy as np
import pytest

torch = pytest.importorskip("torch")

import nn_model


def test_resolve_device_cpu_when_requested_cpu():
    assert nn_model.resolve_device("cpu") == "cpu"


def test_resolve_device_falls_back_to_cpu_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert nn_model.resolve_device("gpu") == "cpu"


def test_resolve_device_uses_cuda_when_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert nn_model.resolve_device("gpu") == "cuda"


def _make_regression_data(n=200, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, 3)).astype(np.float32)
    y = (2 * X[:, 0] - X[:, 1] + 0.5 * X[:, 2]).astype(np.float32)
    return X, y


def test_torch_mlp_regressor_fits_and_predicts_right_shape():
    X, y = _make_regression_data()
    model = nn_model.TorchMLPRegressor(hidden_dim=16, n_layers=1, epochs=50, device="cpu", random_state=7)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (len(y),)


def test_torch_mlp_regressor_learns_better_than_predicting_the_mean():
    X, y = _make_regression_data(n=400)
    model = nn_model.TorchMLPRegressor(hidden_dim=32, n_layers=2, epochs=150,
                                        learning_rate=1e-2, device="cpu", random_state=7)
    model.fit(X, y)
    preds = model.predict(X)
    mse_model = np.mean((preds - y) ** 2)
    mse_mean_baseline = np.mean((y.mean() - y) ** 2)
    assert mse_model < mse_mean_baseline


def test_torch_mlp_classifier_predicts_known_classes_and_valid_proba():
    rng = np.random.RandomState(0)
    X = np.vstack([rng.normal(loc=-2, size=(60, 2)), rng.normal(loc=2, size=(60, 2))]).astype(np.float32)
    y = np.array([0] * 60 + [1] * 60)
    model = nn_model.TorchMLPClassifier(hidden_dim=16, n_layers=1, epochs=80, device="cpu", random_state=7)
    model.fit(X, y)

    preds = model.predict(X)
    assert set(preds).issubset({0, 1})

    proba = model.predict_proba(X)
    assert proba.shape == (len(y), 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-4)

    accuracy = (preds == y).mean()
    assert accuracy > 0.85


def test_torch_mlp_classifier_predict_proba_requires_is_classifier():
    X, y = _make_regression_data()
    model = nn_model.TorchMLPRegressor(hidden_dim=8, n_layers=1, epochs=10, device="cpu", random_state=7)
    model.fit(X, y)
    with pytest.raises(AttributeError):
        model.predict_proba(X)


def test_torch_mlp_clone_and_get_params_round_trip():
    from sklearn.base import clone
    model = nn_model.TorchMLPRegressor(hidden_dim=32, n_layers=2, learning_rate=5e-3,
                                        epochs=10, batch_size=16, device="cpu", random_state=3)
    cloned = clone(model)
    assert cloned.get_params() == model.get_params()
