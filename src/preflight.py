import json
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config" / "config.json"


def check_kaggle_credentials() -> list[str]:
    errors = []
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        errors.append(f"kaggle.json not found at {kaggle_json}")
        return errors
    try:
        data = json.loads(kaggle_json.read_text())
    except Exception as e:
        errors.append(f"kaggle.json is not valid JSON: {e}")
        return errors
    if "username" not in data or "key" not in data:
        errors.append("kaggle.json missing 'username' or 'key' field")
    return errors


def check_kaggle_repo_clone() -> list[str]:
    errors = []
    if not CONFIG_PATH.exists():
        errors.append(f"config.json not found at {CONFIG_PATH}")
        return errors
    config = json.loads(CONFIG_PATH.read_text())
    repo_cfg = config.get("kaggle_repo", {})
    local_path = repo_cfg.get("local_path")
    remote_url = repo_cfg.get("remote_url")
    if not local_path or not remote_url:
        errors.append("config.json is missing kaggle_repo.local_path or kaggle_repo.remote_url")
        return errors

    repo_dir = Path(local_path)
    if not (repo_dir / ".git").exists():
        errors.append(f"kaggle_repo.local_path '{local_path}' is not a git repo (clone it first)")
        return errors

    result = subprocess.run(
        ["git", "-C", str(repo_dir), "remote", "get-url", "origin"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        errors.append(f"could not read git remote at '{local_path}': {result.stderr.strip()}")
        return errors

    actual_remote = result.stdout.strip()
    if actual_remote.rstrip("/").removesuffix(".git") != remote_url.rstrip("/").removesuffix(".git"):
        errors.append(
            f"kaggle_repo.local_path '{local_path}' points at remote '{actual_remote}', "
            f"expected '{remote_url}'"
        )
    return errors


def run_all() -> list[str]:
    errors = []
    errors += check_kaggle_credentials()
    errors += check_kaggle_repo_clone()
    return errors


if __name__ == "__main__":
    problems = run_all()
    if problems:
        for p in problems:
            print(f"[PREFLIGHT FAIL] {p}")
        sys.exit(1)
    print("[PREFLIGHT OK] credentials and kaggle repo clone look valid")
    sys.exit(0)
