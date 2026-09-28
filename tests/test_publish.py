from types import SimpleNamespace

import pytest

import publish


def _ok(stdout=""):
    return SimpleNamespace(returncode=0, stdout=stdout, stderr="")


def test_publish_notebook_pulls_before_pushing_and_copies_only_notebook(tmp_path, monkeypatch):
    repo_dir = tmp_path / "kaggle_repo"
    (repo_dir / ".git").mkdir(parents=True)
    nb_path = tmp_path / "some-dataset-autokaggle.ipynb"
    nb_path.write_text("{}")

    calls = []

    def fake_run(cmd, capture_output=True, text=True):
        calls.append(cmd)
        if cmd[:2] == ["git", "-C"] and "status" in cmd:
            return _ok(stdout=" M diamonds/some-dataset-autokaggle.ipynb\n")
        return _ok()

    monkeypatch.setattr(publish.subprocess, "run", fake_run)

    kaggle_repo_config = {
        "local_path": str(repo_dir),
        "remote_url": "https://github.com/VirajSanghavi007/Kaggle.git",
        "branch": "main",
    }
    publish.publish_notebook(kaggle_repo_config, "someowner/some-dataset", nb_path)

    dest = repo_dir / "some-dataset" / nb_path.name
    assert dest.exists()
    assert not (repo_dir / "some-dataset" / "train.csv").exists()

    git_subcommands = [c[3] for c in calls if len(c) > 3]
    assert "pull" in git_subcommands
    assert git_subcommands.index("checkout") < git_subcommands.index("pull") < git_subcommands.index("add")
    assert "push" in git_subcommands
    assert git_subcommands.index("commit") < git_subcommands.index("push")


def test_publish_notebook_skips_commit_when_nothing_changed(tmp_path, monkeypatch):
    repo_dir = tmp_path / "kaggle_repo"
    (repo_dir / ".git").mkdir(parents=True)
    nb_path = tmp_path / "some-dataset-autokaggle.ipynb"
    nb_path.write_text("{}")

    calls = []

    def fake_run(cmd, capture_output=True, text=True):
        calls.append(cmd)
        return _ok(stdout="")

    monkeypatch.setattr(publish.subprocess, "run", fake_run)

    kaggle_repo_config = {
        "local_path": str(repo_dir),
        "remote_url": "https://github.com/VirajSanghavi007/Kaggle.git",
        "branch": "main",
    }
    publish.publish_notebook(kaggle_repo_config, "someowner/some-dataset", nb_path)

    git_subcommands = [c[3] for c in calls if len(c) > 3]
    assert "commit" not in git_subcommands
    assert "push" not in git_subcommands


def test_publish_notebook_raises_if_repo_not_cloned(tmp_path):
    kaggle_repo_config = {
        "local_path": str(tmp_path / "not_a_repo"),
        "remote_url": "https://github.com/VirajSanghavi007/Kaggle.git",
        "branch": "main",
    }
    with pytest.raises(publish.PublishError):
        publish.publish_notebook(kaggle_repo_config, "someowner/some-dataset", tmp_path / "nb.ipynb")
