import json
import sys
import traceback
from pathlib import Path

from detect_target import detect_target
from eda import run_eda
from job_common import (
    DATA_DIR, EXIT_FAILURE, EXIT_NOTHING_TO_DO, EXIT_SUCCESS,
    load_config, resolve_kernel_slug, setup_logging, download_dataset,
)
from preflight import run_all as preflight_run_all
from progress import StepTracker
from queue_file import peek_next, QueueError
from run_state import load_state, save_state


def prepare_job(job: dict, log) -> int:
    kernel_slug = resolve_kernel_slug(job["target_slug"])
    job_data_dir = DATA_DIR / kernel_slug
    state = load_state(kernel_slug)
    tracker = StepTracker(total_steps=2, log=log)

    csv_path_marker = job_data_dir / "_train_csv_path.txt"
    if state["eda_done"] and csv_path_marker.exists():
        log.info("Resuming: dataset already downloaded, EDA already done for %s", kernel_slug)
        tracker.step("dataset downloaded")
        tracker.step("EDA complete")
        return EXIT_SUCCESS

    if csv_path_marker.exists():
        csv_path = Path(csv_path_marker.read_text().strip())
        log.info("Resuming: dataset already downloaded at %s", csv_path)
    else:
        csv_path = download_dataset(job, job_data_dir, log)
        csv_path_marker.write_text(str(csv_path))
    tracker.step("dataset downloaded")

    target = detect_target(csv_path, job_data_dir)
    log.info("Detected target column: %s", target)

    eda_summary = run_eda(str(csv_path), target)
    eda_path = job_data_dir / "eda_summary.json"
    eda_path.write_text(json.dumps(eda_summary, indent=2, default=str))
    state["eda_done"] = True
    save_state(kernel_slug, state)
    tracker.step("EDA complete")

    log.info(
        "Ready for pipeline selection: run the Claude Co-Work Routine against %s, "
        "then run run_pipeline.py",
        eda_path,
    )
    return EXIT_SUCCESS


def main():
    log = setup_logging("prepare_job")

    preflight_errors = preflight_run_all()
    if preflight_errors:
        for e in preflight_errors:
            log.error("Preflight check failed: %s", e)
        sys.exit(EXIT_FAILURE)

    try:
        job = peek_next()
    except QueueError as e:
        log.error("Queue error: %s", e)
        sys.exit(EXIT_FAILURE)

    if job is None:
        log.info("kaggledataset.txt is empty - nothing to do today.")
        sys.exit(EXIT_NOTHING_TO_DO)

    try:
        load_config()
    except FileNotFoundError as e:
        log.error(str(e))
        sys.exit(EXIT_FAILURE)

    log.info("Preparing queued job: %s", job["target_slug"])

    try:
        exit_code = prepare_job(job, log)
    except Exception:
        log.error("Unhandled exception while preparing %s:\n%s", job["target_slug"], traceback.format_exc())
        sys.exit(EXIT_FAILURE)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
