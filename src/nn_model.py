import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin


def resolve_device(requested: str) -> str:
    """Fall back to CPU whenever GPU was requested but isn't actually usable -
    the LLM's device choice is a preference, not a guarantee the hardware exists."""
    if requested != "gpu":
        return "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


class TorchMLP(BaseEstimator):
    """Small feedforward network usable as a sklearn Pipeline step for either
    regression or classification. Kept dependency-light (plain torch, no skorch)
    and self-contained so notebook_builder can inline its source directly into
    published notebooks, which never have access to this repo's src/ modules."""

    def __init__(self, hidden_dim=64, n_layers=2, learning_rate=1e-3, epochs=100,
                 batch_size=64, device="cpu", is_classifier=False, random_state=7):
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.is_classifier = is_classifier
        self.random_state = random_state

    def _build_network(self, n_features, n_outputs):
        import torch.nn as nn
        layers = []
        in_dim = n_features
        for _ in range(self.n_layers):
            layers += [nn.Linear(in_dim, self.hidden_dim), nn.ReLU()]
            in_dim = self.hidden_dim
        layers.append(nn.Linear(in_dim, n_outputs))
        return nn.Sequential(*layers)

    def fit(self, X, y):
        import torch
        import torch.nn as nn

        torch.manual_seed(self.random_state)
        device = resolve_device("gpu") if self.device in ("gpu", "cuda") else "cpu"
        self.device_ = device

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)

        if self.is_classifier:
            self.classes_ = np.unique(y)
            class_to_idx = {c: i for i, c in enumerate(self.classes_)}
            y_idx = np.array([class_to_idx[v] for v in y], dtype=np.int64)
            n_outputs = len(self.classes_)
            y_tensor = torch.tensor(y_idx, dtype=torch.long, device=device)
            loss_fn = nn.CrossEntropyLoss()
        else:
            n_outputs = 1
            y_tensor = torch.tensor(y.astype(np.float32).reshape(-1, 1), device=device)
            loss_fn = nn.MSELoss()

        X_tensor = torch.tensor(X, device=device)
        self.model_ = self._build_network(X.shape[1], n_outputs).to(device)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)

        n_samples = X_tensor.shape[0]
        self.model_.train()
        for _ in range(self.epochs):
            perm = torch.randperm(n_samples, device=device)
            for start in range(0, n_samples, self.batch_size):
                idx = perm[start:start + self.batch_size]
                optimizer.zero_grad()
                out = self.model_(X_tensor[idx])
                if not self.is_classifier:
                    loss = loss_fn(out, y_tensor[idx])
                else:
                    loss = loss_fn(out, y_tensor[idx])
                loss.backward()
                optimizer.step()
        return self

    def _forward_numpy(self, X):
        import torch
        X = np.asarray(X, dtype=np.float32)
        self.model_.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, device=self.device_)
            out = self.model_(X_tensor)
            return out.cpu().numpy()

    def predict(self, X):
        raw = self._forward_numpy(X)
        if self.is_classifier:
            return self.classes_[np.argmax(raw, axis=1)]
        return raw.reshape(-1)

    def predict_proba(self, X):
        if not self.is_classifier:
            raise AttributeError("predict_proba is only available when is_classifier=True")
        raw = self._forward_numpy(X)
        exp = np.exp(raw - raw.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)


class TorchMLPRegressor(RegressorMixin, TorchMLP):
    def __init__(self, hidden_dim=64, n_layers=2, learning_rate=1e-3, epochs=100,
                 batch_size=64, device="cpu", random_state=7):
        super().__init__(hidden_dim, n_layers, learning_rate, epochs, batch_size,
                          device, is_classifier=False, random_state=random_state)


class TorchMLPClassifier(ClassifierMixin, TorchMLP):
    def __init__(self, hidden_dim=64, n_layers=2, learning_rate=1e-3, epochs=100,
                 batch_size=64, device="cpu", random_state=7):
        super().__init__(hidden_dim, n_layers, learning_rate, epochs, batch_size,
                          device, is_classifier=True, random_state=random_state)
