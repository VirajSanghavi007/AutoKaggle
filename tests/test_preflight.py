import json
from types import SimpleNamespace

import preflight


def test_check_kaggle_repo_clone_fails_when_not_cloned(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "kaggle_repo": {"local_path": str(tmp_path / "missing"), "remote_url": "https://github.com/x/y.git"}
    }))
    monkeypatch.setattr(preflight, "CONFIG_PATH", config_path)

    errors = preflight.check_kaggle_repo_clone()
    assert any("not a git repo" in e for e in errors)


def test_check_kaggle_repo_clone_fails_on_remote_mismatch(tmp_path, monkeypatch):
    repo_dir = tmp_path / "repo"
    (repo_dir / ".git").mkdir(parents=True)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "kaggle_repo": {"local_path": str(repo_dir), "remote_url": "https://github.com/expected/repo.git"}
    }))
    monkeypatch.setattr(preflight, "CONFIG_PATH", config_path)
    monkeypatch.setattr(
        preflight.subprocess, "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout="https://github.com/other/repo.git\n", stderr=""),
    )

    errors = preflight.check_kaggle_repo_clone()
    assert any("expected" in e for e in errors)


def test_check_kaggle_repo_clone_passes_when_remote_matches(tmp_path, monkeypatch):
    repo_dir = tmp_path / "repo"
    (repo_dir / ".git").mkdir(parents=True)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "kaggle_repo": {"local_path": str(repo_dir), "remote_url": "https://github.com/expected/repo.git"}
    }))
    monkeypatch.setattr(preflight, "CONFIG_PATH", config_path)
    monkeypatch.setattr(
        preflight.subprocess, "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout="https://github.com/expected/repo.git\n", stderr=""),
    )

    assert preflight.check_kaggle_repo_clone() == []
