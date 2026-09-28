import json
import shutil
import sys
import traceback
from pathlib import Path

import pandas as pd

from job_common import (
    CHECKPOINT_DIR, CURRENT_RUN_PATH, DATA_DIR, EXIT_FAILURE, EXIT_NOTHING_TO_DO,
    EXIT_SUCCESS, NOTEBOOK_DIR, apply_leakage_backstop, format_kernel_title, load_config,
    resolve_kernel_slug, setup_logging,
)
from notebook_builder import build_notebook, execute_notebook, save_notebook
from pipeline_builder import run_benchmark, expand_datetime_features
from preflight import run_all as preflight_run_all
from progress import StepTracker
from publish import publish_notebook, PublishError
from queue_file import peek_next, remove_first, QueueError
from run_history import append_run_record
from run_state import load_state, save_state
from schema import PipelineConfig


def run_pipeline(job: dict, config: dict, log) -> int:
    kernel_slug = resolve_kernel_slug(job["target_slug"])
    job_data_dir = DATA_DIR / kernel_slug
    job_checkpoint_dir = CHECKPOINT_DIR / kernel_slug
    state = load_state(kernel_slug)
    tracker = StepTracker(total_steps=3, log=log)

    config_path = job_data_dir / "pipeline_config.json"
    if not state["eda_done"]:
        log.error("EDA not done yet for %s - run prepare_job.py first", kernel_slug)
        return EXIT_FAILURE
    if not config_path.exists():
        log.error(
            "pipeline_config.json not found at %s - waiting on the Claude Co-Work Routine",
            config_path,
        )
        return EXIT_NOTHING_TO_DO

    csv_path_marker = job_data_dir / "_train_csv_path.txt"
    csv_path = Path(csv_path_marker.read_text().strip())
    df = pd.read_csv(csv_path)

    pipeline_config = PipelineConfig(**json.loads(config_path.read_text()))
    state["llm_done"] = True
    save_state(kernel_slug, state)

    # must run before the bench_done resume-check below: it's a deterministic function
    # of (df, datetime_features) so it's safe to redo, but skipping it on a resume would
    # leave pipeline_config.features missing the derived columns that a cached
    # bench_result.json was actually trained on, producing a notebook with a different
    # (smaller) feature set than what was benchmarked
    had_datetime_features = bool(pipeline_config.datetime_features)
    df = expand_datetime_features(df, pipeline_config)
    if had_datetime_features:
        config_path.write_text(pipeline_config.model_dump_json(indent=2))

    dropped = apply_leakage_backstop(df, pipeline_config, log)
    if dropped:
        config_path.write_text(pipeline_config.model_dump_json(indent=2))

    bench_path = job_data_dir / "bench_result.json"
    if state["bench_done"] and bench_path.exists():
        bench_result = json.loads(bench_path.read_text())
        log.info("Resuming: reusing existing benchmark result")
    else:
        bench_result = run_benchmark(df, pipeline_config, str(job_checkpoint_dir), log)
        bench_path.write_text(json.dumps(bench_result, indent=2, default=str))
        state["bench_done"] = True
        save_state(kernel_slug, state)
    log.info(
        "Best model: %s (cv_score=%.5f, baseline=%.5f)",
        bench_result["best"]["name"], bench_result["best"]["cv_score"], bench_result["baseline_score"],
    )
    tracker.step("benchmarking complete")

    # recomputed at every build below (not a fixed constant): if the fallback loop below
    # promotes a different model as "best", its title must reflect THAT model's score,
    # not the original winner's - required kernel title format is
    # "<Dataset Name> (<METRIC> <value>)", e.g. "Housing Prices Dataset (RMSE 1050295.64)"
    def current_title():
        # a bench_result.json cached before "metric" existed in this dict (schema
        # evolves; checkpoints are long-lived) would KeyError here on a resume - fall
        # back to the natural default metric for this problem_type instead of crashing
        default_metric = "rmse" if pipeline_config.problem_type == "regression" else "roc_auc"
        metric = bench_result.get("metric", default_metric)
        return format_kernel_title(job["target_slug"], metric, bench_result["best"]["cv_score"])

    NOTEBOOK_DIR.mkdir(exist_ok=True)
    shutil.copy(csv_path, NOTEBOOK_DIR / csv_path.name)

    title = current_title()
    nb = build_notebook(pipeline_config, bench_result, csv_path.name, title)
    nb_path = NOTEBOOK_DIR / f"{kernel_slug}.ipynb"
    save_notebook(nb, str(nb_path))

    ok, err = execute_notebook(nb, str(NOTEBOOK_DIR))
    if not ok:
        log.error("Notebook execution failed: %s", err)
        remaining = [r for r in bench_result["all_results"] if r["name"] != bench_result["best"]["name"]]
        remaining.sort(key=lambda r: r["cv_score"])
        for fallback_result in remaining:
            log.warning("Falling back to model: %s", fallback_result["name"])
            bench_result["best"] = fallback_result
            title = current_title()
            nb = build_notebook(pipeline_config, bench_result, csv_path.name, title)
            save_notebook(nb, str(nb_path))
            ok, err = execute_notebook(nb, str(NOTEBOOK_DIR))
            if ok:
                break
        if not ok:
            log.error("All models failed notebook execution for %s", job["target_slug"])
            append_run_record(job["target_slug"], bench_result["best"]["name"],
                               bench_result["best"]["cv_score"], bench_result["baseline_score"], "notebook_failed")
            return EXIT_FAILURE

    log.info("Notebook executed successfully: %s", nb_path)
    state["notebook_done"] = True
    save_state(kernel_slug, state)
    tracker.step("notebook built and executed")

    if state.get("published_done"):
        log.info("Already published - skipping publish step")
    else:
        try:
            publish_notebook(config["kaggle_repo"], job["target_slug"], nb_path, log)
            state["published_done"] = True
            save_state(kernel_slug, state)
        except PublishError as e:
            log.error("Publish failed, leaving queue entry in place for retry: %s", e)
            append_run_record(job["target_slug"], bench_result["best"]["name"],
                               bench_result["best"]["cv_score"], bench_result["baseline_score"],
                               "pipeline_ok_publish_failed")
            return EXIT_FAILURE
    tracker.step("published to Kaggle repo")

    CURRENT_RUN_PATH.write_text(json.dumps({
        "target_slug": job["target_slug"],
        "kernel_slug": kernel_slug,
        "title": title,
    }, indent=2))

    append_run_record(job["target_slug"], bench_result["best"]["name"],
                       bench_result["best"]["cv_score"], bench_result["baseline_score"], "success")

    return EXIT_SUCCESS


def main():
    log = setup_logging("run_pipeline")

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
        config = load_config()
    except FileNotFoundError as e:
        log.error(str(e))
        sys.exit(EXIT_FAILURE)

    log.info("Running pipeline for queued job: %s", job["target_slug"])

    try:
        exit_code = run_pipeline(job, config, log)
    except Exception:
        log.error("Unhandled exception while processing %s:\n%s", job["target_slug"], traceback.format_exc())
        sys.exit(EXIT_FAILURE)

    if exit_code == EXIT_SUCCESS:
        remove_first(job["raw_url"])
        log.info("Removed completed entry from kaggledataset.txt: %s", job["raw_url"])

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
