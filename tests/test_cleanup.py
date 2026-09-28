import json
import logging

import cleanup


def _make_job(data_dir, checkpoint_dir, kernel_slug, published_done):
    job_data_dir = data_dir / kernel_slug
    job_data_dir.mkdir(parents=True)
    (job_data_dir / "eda_summary.json").write_text("{}")

    job_checkpoint_dir = checkpoint_dir / kernel_slug
    job_checkpoint_dir.mkdir(parents=True)
    (job_checkpoint_dir / "run_state.json").write_text(json.dumps({
        "eda_done": True, "llm_done": True, "bench_done": True,
        "notebook_done": True, "published_done": published_done,
    }))


def test_cleanup_removes_only_published_jobs(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    checkpoint_dir = tmp_path / "checkpoints"
    monkeypatch.setattr(cleanup, "DATA_DIR", data_dir)
    monkeypatch.setattr(cleanup, "CHECKPOINT_DIR", checkpoint_dir)
    monkeypatch.setattr(cleanup, "CHECKPOINT_ROOT", checkpoint_dir, raising=False)
    import run_state
    monkeypatch.setattr(run_state, "CHECKPOINT_ROOT", checkpoint_dir)

    _make_job(data_dir, checkpoint_dir, "published-autokaggle", published_done=True)
    _make_job(data_dir, checkpoint_dir, "in-progress-autokaggle", published_done=False)

    log = logging.getLogger("test_cleanup")
    cleaned = cleanup.cleanup_published_jobs(log)

    assert cleaned == ["published-autokaggle"]
    assert not (data_dir / "published-autokaggle").exists()
    assert not (checkpoint_dir / "published-autokaggle").exists()
    assert (data_dir / "in-progress-autokaggle").exists()
    assert (checkpoint_dir / "in-progress-autokaggle").exists()


def test_cleanup_handles_empty_data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(cleanup, "DATA_DIR", tmp_path / "does-not-exist")
    log = logging.getLogger("test_cleanup")
    assert cleanup.cleanup_published_jobs(log) == []
