import json
import sys
from pathlib import Path

from retry import retry


def write_kernel_metadata(notebook_dir: str, kernel_slug: str, kaggle_username: str, title: str,
                           notebook_filename: str, dataset_sources: list[str], is_public: bool):
    metadata = {
        "id": f"{kaggle_username}/{kernel_slug}",
        "title": title,
        "code_file": notebook_filename,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": not is_public,
        "enable_gpu": False,
        "enable_internet": True,
        "dataset_sources": dataset_sources,
        "competition_sources": [],
        "kernel_sources": [],
    }
    path = Path(notebook_dir) / "kernel-metadata.json"
    path.write_text(json.dumps(metadata, indent=2))
    return path


def push_kernel(notebook_dir: str) -> tuple[bool, str]:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as e:
        return False, f"kaggle package/credentials not available: {e}"

    try:
        api = KaggleApi()
        api.authenticate()
        result = retry(lambda: api.kernels_push(str(notebook_dir)), what="kaggle kernels_push")
        return True, str(result)
    except Exception as e:
        return False, str(e)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("kaggle_push")

    base_dir = Path(__file__).resolve().parent.parent
    current_run_path = base_dir / "config" / "current_run.json"

    if not current_run_path.exists():
        log.error("no current_run.json found at %s - nothing to push", current_run_path)
        sys.exit(1)

    current_run = json.loads(current_run_path.read_text())

    kaggle_json_path = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json_path.exists():
        log.error("kaggle.json not found at %s", kaggle_json_path)
        sys.exit(1)
    kaggle_username = json.loads(kaggle_json_path.read_text()).get("username", "")
    if not kaggle_username:
        log.error("kaggle.json has no 'username' field")
        sys.exit(1)

    notebook_dir = base_dir / "notebooks"
    kernel_slug = current_run["kernel_slug"]
    title = current_run["title"]
    is_public = True  # every published kernel must be public - not a config toggle

    nb_path = notebook_dir / f"{kernel_slug}.ipynb"
    if not nb_path.exists():
        log.error("expected notebook not found: %s", nb_path)
        sys.exit(1)

    target_slug = current_run.get("target_slug", "")
    dataset_sources = [target_slug] if target_slug else []
    write_kernel_metadata(
        str(notebook_dir), kernel_slug, kaggle_username, title,
        nb_path.name, dataset_sources=dataset_sources, is_public=is_public,
    )
    ok, output = push_kernel(str(notebook_dir))
    log.info(output)
    sys.exit(0 if ok else 1)
