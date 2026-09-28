import shutil
import sys

from job_common import CHECKPOINT_DIR, DATA_DIR, setup_logging
from run_state import load_state


def cleanup_published_jobs(log) -> list[str]:
    cleaned = []
    if not DATA_DIR.exists():
        return cleaned
    for job_dir in DATA_DIR.iterdir():
        if not job_dir.is_dir():
            continue
        kernel_slug = job_dir.name
        state = load_state(kernel_slug)
        if not state.get("published_done"):
            continue
        shutil.rmtree(job_dir, ignore_errors=True)
        checkpoint_dir = CHECKPOINT_DIR / kernel_slug
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
        cleaned.append(kernel_slug)
        log.info("Cleaned up published job: %s", kernel_slug)
    return cleaned


def main():
    log = setup_logging("cleanup")
    cleaned = cleanup_published_jobs(log)
    log.info("Cleaned up %d published job(s)", len(cleaned))
    sys.exit(0)


if __name__ == "__main__":
    main()
