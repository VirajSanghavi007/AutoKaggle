import shutil
import subprocess
from pathlib import Path

from retry import retry


class PublishError(Exception):
    pass


def _run_git(args: list[str], cwd: Path, log=None):
    def _do():
        result = subprocess.run(["git", "-C", str(cwd)] + args, capture_output=True, text=True)
        if result.returncode != 0:
            raise PublishError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
        return result.stdout

    return retry(_do, log=log, what=f"git {' '.join(args)}")


def publish_notebook(kaggle_repo_config: dict, dataset_slug: str, nb_path: Path, log=None) -> None:
    """Copy the executed notebook into a local clone of the Kaggle repo and push it.

    Follows that repo's existing convention: one folder per dataset slug
    (e.g. 'diamonds/diamonds_price_prediction.ipynb'), notebook only, no CSV.
    """
    repo_dir = Path(kaggle_repo_config["local_path"])
    branch = kaggle_repo_config.get("branch", "main")

    if not (repo_dir / ".git").exists():
        raise PublishError(f"'{repo_dir}' is not a git repo - clone {kaggle_repo_config['remote_url']} there first")

    _run_git(["checkout", branch], repo_dir, log)
    _run_git(["pull", "--ff-only", "origin", branch], repo_dir, log)

    dataset_dir = repo_dir / dataset_slug.split("/")[-1]
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dataset_dir / nb_path.name
    shutil.copy(nb_path, dest_path)

    rel_path = dest_path.relative_to(repo_dir)
    _run_git(["add", str(rel_path)], repo_dir, log)

    status = subprocess.run(
        ["git", "-C", str(repo_dir), "status", "--porcelain", str(rel_path)],
        capture_output=True, text=True,
    )
    if not status.stdout.strip():
        if log:
            log.info("No changes to publish for %s (notebook identical to last run)", dataset_slug)
        return

    _run_git(["commit", "-m", f"AutoKaggle: {dataset_slug}"], repo_dir, log)
    _run_git(["push", "origin", branch], repo_dir, log)
    if log:
        log.info("Published %s to %s", rel_path, kaggle_repo_config["remote_url"])
