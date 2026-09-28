# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

AutoKaggle: a scheduled Windows job that pops one Kaggle **dataset** URL off a queue file (no
competitions — out of scope, see below), runs EDA, hands the EDA summary to a **Claude Co-Work
Routine** (running in Claude Desktop, daily at 8am) which decides a modeling pipeline, benchmarks
candidate models with Optuna, generates and executes a Jupyter notebook, then publishes that
notebook into a *separate* GitHub repo — [VirajSanghavi007/Kaggle](https://github.com/VirajSanghavi007/Kaggle)
— and pushes it as a Kaggle kernel. `AutoKaggle` itself is not a git repo; do not add git remotes
or commit-to-self logic here.

## Commands

Setup (Windows, requires `uv` on PATH):
```bat
setup_env.bat
```
Creates `.venv` and installs `requirements.txt` (includes `pytest`).

Run the two-phase job manually (`~/.kaggle/kaggle.json` must be present; no Anthropic key needed —
the LLM step runs outside this process now):
```bat
.venv\Scripts\activate.bat
cd src
python prepare_job.py   REM download + EDA, writes data\<kernel_slug>\eda_summary.json
REM ... Claude Co-Work Routine runs here, writes data\<kernel_slug>\pipeline_config.json ...
python run_pipeline.py  REM benchmark + notebook + publish to the Kaggle repo
```
Exit codes for both scripts: `0` success, `1` failure, `2` nothing to do yet (`prepare_job.py`:
queue empty; `run_pipeline.py`: `pipeline_config.json` not written by the Routine yet — this is
not a failure, just re-run later).

Run the full scheduled flow as the daily task does (prepare -> [Routine runs separately] ->
pipeline -> kaggle kernel push):
```bat
run_autokaggle.bat
```

Register the Windows daily scheduled task (8am):
```powershell
.\register_task.ps1
```

Push the last completed run's notebook to Kaggle as a kernel on its own (reads `config/current_run.json`):
```bash
cd src && python kaggle_push.py
```

Reclaim disk space for jobs that are fully published (deletes `data/<slug>/` and
`checkpoints/<slug>/` once `run_state.json.published_done` is true; not on the per-job critical
path, run whenever):
```bash
cd src && python cleanup.py
```

Validate a Routine-produced `pipeline_config.json` against the schema before letting
`run_pipeline.py` consume it (this is what the Routine's own prompt should call after writing the
file):
```bash
cd src && python validate_pipeline_config.py <kernel_slug>
```

Run EDA standalone on a CSV:
```bash
cd src && python eda.py <path_to_csv> <target_column>
```

Check credentials + the Kaggle-repo clone without running a job:
```bash
cd src && python preflight.py
```

Run the test suite:
```bash
.venv\Scripts\python.exe -m pytest tests/ -q
```

## Queue and job identity

- `kaggledataset.txt` at repo root is the job queue: one Kaggle **dataset** URL per line
  (`kaggle.com/datasets/<owner>/<slug>`), `#`-prefixed lines are comments. `queue_file.py` parses
  the first non-comment line into `target_slug`; `remove_first()` deletes that exact line only
  after a job fully succeeds (pipeline run **and** publish, see Resumability below).
- Every job is keyed by a **kernel_slug** derived from the target slug
  (`job_common.py::sanitize_kernel_slug` + `resolve_kernel_slug`), tracked in
  `notebooks/slug_registry.json` to disambiguate collisions (e.g. two datasets whose owner differs
  but basename matches get a hash suffix).
- Per-job working state lives under `data/<kernel_slug>/` (downloaded CSV, `eda_summary.json`,
  `pipeline_config.json` — written by the Routine, `bench_result.json`) and
  `checkpoints/<kernel_slug>/` (per-model Optuna results as `<model>.json`, failures as
  `<model>_FAILED.log`, plus `run_state.json`).

## Two-phase pipeline with an external LLM step in the middle

The old single-process `main.py::run_job` is now split at the LLM boundary, because pipeline
selection is done by a Claude Co-Work Routine (an external, asynchronous actor) rather than an
in-process API call:

1. **`prepare_job.py`** — download (`job_common.py::download_dataset`, retried via `retry.py`),
   detect target (`detect_target.py`), run EDA (`eda.py::run_eda`), write
   `data/<kernel_slug>/eda_summary.json`, set `run_state.json.eda_done`. Exits; does not wait for
   the Routine.
2. **Claude Co-Work Routine** (no code here executes it) — reads `eda_summary.json`, writes
   `data/<kernel_slug>/pipeline_config.json` shaped like `schema.PipelineConfig`, and should run
   `validate_pipeline_config.py <kernel_slug>` itself and iterate until it passes before finishing,
   since nothing downstream re-prompts an LLM on a validation failure.
3. **`run_pipeline.py`** — requires `pipeline_config.json` to already exist (exits `2`, not `1`, if
   it doesn't — this is the normal "waiting on the Routine" state, retried by the next scheduled
   run); then runs `job_common.py::apply_leakage_backstop`, `pipeline_builder.run_benchmark`,
   `notebook_builder.build_notebook`/`execute_notebook` (falling back through remaining
   benchmarked models on execution failure, same as before), and finally `publish.py`.

Each stage still checks `run_state.json` (`run_state.py`: `eda_done`, `llm_done`, `bench_done`,
`notebook_done`, `published_done`) plus its output file before redoing work, so a crash mid-job
resumes rather than restarts. When touching either script, preserve this pattern: write the state
flag *and* the artifact file together, and check both before skipping a stage.

Optuna-level resume is separate and finer-grained: `pipeline_builder.run_benchmark` checkpoints
each candidate model's tuning result to `checkpoints/<kernel_slug>/<model>.json` individually, so a
crash mid-benchmark only re-tunes the models that hadn't finished.

## Publishing (two independent destinations)

- **GitHub**: `publish.py::publish_notebook` operates on a *persistent local clone* of
  `VirajSanghavi007/Kaggle`, currently at `C:\Storage\Kaggle\repo` (**not** `C:\Storage\Kaggle`
  itself — that's an unrelated, larger dataset-download project living alongside the actual git
  clone; path/remote in `config/config.json:kaggle_repo`, checked by
  `preflight.py::check_kaggle_repo_clone`). It `git pull --ff-only`s that clone, copies the
  executed `.ipynb` into `<repo>/<dataset-slug>/` (matching that repo's existing convention: one
  folder per dataset, notebook only — **do not copy the CSV in**), then commits and pushes. This
  is entirely separate from `AutoKaggle`'s own (nonexistent) git history — never add
  `git`/`commit`/`push` logic that targets `AutoKaggle`'s own working directory.
- **Kaggle kernel**: `kaggle_push.py` (run after `run_pipeline.py` succeeds, by
  `run_autokaggle.bat`) reads `config/current_run.json` and pushes the same notebook as a Kaggle
  kernel via the Kaggle API, with `dataset_sources` set to the job's `target_slug` so the kernel
  can reference the original dataset instead of needing a redundant re-download. **Every pushed
  kernel is public** — `is_public = True` is hardcoded in `kaggle_push.py`'s `__main__`, not a
  config toggle; there is no way to publish a private kernel from this codebase, by requirement.
- **Kernel title format is fixed**: `"<Dataset Name> (<METRIC> <value>)"`, e.g.
  `"Housing Prices Dataset (RMSE 1050295.64)"` — built by `job_common.py::format_kernel_title`
  (dataset name is a heuristic title-case of the slug's last path segment, not a Kaggle API lookup
  of the real display title) and recomputed in `run_pipeline.py` at *every* notebook build,
  including inside the execution-failure fallback loop, so a fallback model's title reflects that
  model's own score rather than the original winner's stale one. The metric value shown is always
  flipped back to its natural sign (`schema.NEGATED_METRICS`) — the internal `cv_score` is negative
  for roc_auc/f1/accuracy, and displaying that raw would put a negative AUC in a kernel title.
  `bench_result.get("metric", ...)` falls back to a problem-type default rather than a hard key
  lookup, since a `bench_result.json` cached before the `metric` field existed (checkpoints are
  long-lived; schema evolves) would otherwise `KeyError` on resume.
- `run_pipeline.py` treats these as gating the queue: `queue_file.remove_first()` (and
  `status="success"` in `run_history.jsonl`) only happens once the benchmark/notebook step *and*
  `publish.py` both succeed. A publish failure after a good pipeline run records
  `status="pipeline_ok_publish_failed"`, sets `published_done: false`, and leaves the queue entry
  in place so the *next* run retries only the publish step (via the `published_done` check in
  `run_pipeline.py`), not the whole benchmark.

## LLM-config allow-lists (must stay in sync by hand)

`schema.py`'s allow-list constants (`IMPUTERS`, `ENCODERS`, `SCALERS`, `SMOTE_VARIANTS`,
`CV_STRATEGIES`, `REGRESSION_MODELS`, `CLASSIFICATION_MODELS`) define every value the Routine is
allowed to put in `pipeline_config.json` — `schema.PipelineConfig`'s field validators reject
anything else. Changing what's allowed means updating three places together: the allow-list
constant, the Routine's own prompt/instructions (wherever those are authored — not in this repo's
code), and `pipeline_builder.py`'s + `notebook_builder.py`'s factory/import maps (if it affects
codegen). There's no single source of truth enforcing this beyond convention.

`schema.py::apply_guardrails` (also invoked by `validate_pipeline_config.py`) is a second,
non-LLM-trusting safety net after schema validation (e.g. downgrades OneHotEncoder to
TargetEncoder above a cardinality threshold, strips encoders accidentally applied to numeric
features). `job_common.py::apply_leakage_backstop` is a further backstop in `run_pipeline.py`
dropping any feature whose correlation with the target exceeds `LEAKAGE_CORR_THRESHOLD` (0.98)
that the Routine didn't flag as leakage.

`notebook_builder.py::build_notebook` re-renders the *same* config (imputer/encoder/scaler
choices, winning model + its tuned `best_params`, or all stack members if the ensemble won) as
standalone notebook source code — this is a separate codegen path from `pipeline_builder.py`'s
in-process sklearn objects, not a serialization of them, so the two must be kept in sync manually
for any new imputer/encoder/scaler/model option.

**This has been the single most common source of real bugs in this codebase**: PCA, SMOTE, the
stacking meta-learner, GPU device params, `datetime_features` expansion, `outlier_handling`
clipping, and `class_balance.strategy == "class_weight"` were each wired into
`pipeline_builder.run_benchmark` correctly at some point but initially *not* into
`notebook_builder.build_notebook` — every one of them either crashed the published notebook
outright (missing a derived column `ColumnTransformer` referenced) or, worse, silently trained on
a different feature set/model config than what was actually benchmarked, with no error at all.
**Whenever you add something to `pipeline_builder.py` that changes what gets fitted, grep
`notebook_builder.py` for the equivalent and add a real notebook-execution test (build the
notebook, `execute_notebook` it, assert `ok`), not just a "does the source text contain X" test** —
several of these bugs only showed up when the notebook was actually run, not from string-matching
its generated source. Also check the *resume* path in `run_pipeline.py`: `expand_datetime_features`
had to be called unconditionally before the `bench_done` checkpoint check, not only inside
`run_benchmark`, or a crash-and-resume after benchmarking but before publish would silently drop
the derived columns from the config that got used to build the notebook.

## Metric convention

Internally, lower is always better. `config.metrics.primary` selects which metric actually drives
model selection (`REGRESSION_METRICS = rmse/mae/rmsle`, `CLASSIFICATION_METRICS =
roc_auc/f1/accuracy/log_loss` in `schema.py`, validated against `problem_type`). rmse/mae/rmsle/
log_loss are used as-is; roc_auc/f1/accuracy are negated (see
`pipeline_builder._score_from_predictions`) so every metric shares the same `min()`/sort-ascending
logic throughout benchmarking and ensembling. Don't compare a raw score from this codebase against
a raw AUC/F1/accuracy value without flipping the sign back. Each per-model result also carries a
`score_kind` (`"cv"` for real cross-validated scores, `"held_out_val"` for the stacking ensemble's
comparison score) — the two are not directly comparable despite sharing the `cv_score` dict key
for backward compatibility with `run_pipeline.py`/`run_history.py`'s logging.

## GPU support

`CandidateModel.device` (`"cpu"`/`"gpu"`, allow-listed in `schema.DEVICES`) lets pipeline selection
request GPU for `lightgbm`/`xgboost`/`catboost`/`neural_net` (`schema.GPU_CAPABLE_MODELS`) — a
`gpu` request for any other model is guardrail-downgraded to `cpu`. A GPU request only actually
takes effect if `pipeline_builder._gpu_ok(device, name)` confirms both a real CUDA device
(`nn_model.resolve_device`) **and** that this specific installed library build supports GPU
(`_library_build_supports_gpu`, probed once per process and cached) — this distinction matters in
practice: the plain pip `lightgbm` wheel has no GPU tree learner at all
(`GPU Tree Learner was not enabled in this build`) even on a machine with a working CUDA device
that `xgboost`/`catboost`/`torch` can all use fine, so don't assume "CUDA is available" implies
every GPU-capable model can actually use it. Each benchmarked model's result records `gpu_used`
(what actually happened) separately from `device` (what was requested) — `notebook_builder.py`
renders the published notebook's device params from `gpu_used`, never by recomputing, so the
notebook can't drift from what benchmarking actually did.

When a GPU is actually used for a candidate, `_tune_one_model` forces `cv_n_jobs`/`model_n_jobs`
back to 1 for that candidate — a GPU is a single shared device, so thread-parallelizing CV folds
onto it doesn't help and risks memory contention; the GPU's own speedup is the parallelism.

`neural_net` (`nn_model.py::TorchMLPRegressor`/`TorchMLPClassifier`) is a small feedforward network
tuned by the *same* Optuna loop as every other candidate model (hidden_dim/n_layers/learning_rate/
epochs/batch_size are all `trial.suggest_*` calls in `pipeline_builder._model_and_space`) — there's
no separate NN-specific tuning path. `notebook_builder.py::_nn_model_source_cell` inlines
`nn_model.py`'s classes as source (via `inspect.getsource`) into the published notebook rather than
importing them, since notebooks published to the Kaggle repo have no access to this repo's `src/`.

## Config vs. queue vs. per-run state

- `config/config.json` — static settings: notebook title prefix/visibility, seed, and
  `kaggle_repo.{local_path,remote_url,branch}` for the publish clone. Read once per run.
- `config/current_run.json` — written at the end of a successful `run_pipeline.py`, describing
  *only* the most recently completed job (`target_slug`, `kernel_slug`, `title`); `kaggle_push.py`
  reads it standalone to know what to push. It is overwritten every run, not a history.
- `data/run_history.jsonl` — append-only log of every completed/failed job attempt
  (`run_history.py`), one JSON record per line, independent of `current_run.json`. Status values
  include `success`, `notebook_failed`, and `pipeline_ok_publish_failed`.

## Credentials

Requires `~/.kaggle/kaggle.json` (`{"username", "key"}`) and a working local clone of the
`VirajSanghavi007/Kaggle` repo pointed at by `config/config.json:kaggle_repo`, both checked up
front by `preflight.py` before any work starts — check there first when debugging an auth or
publish failure rather than deep in `kaggle_push.py`/`publish.py`. There is no longer an Anthropic
API key requirement anywhere in this codebase.

## Config fields consumed by the pipeline

Besides the allow-listed imputer/encoder/scaler/model/device choices, these `PipelineConfig`
fields are actually read (not just accepted and ignored):
`excluded_models` (filters candidates before ranking, in `run_benchmark`), `baseline.strategy`
(`schema.BASELINE_REGRESSION_STRATEGIES`/`BASELINE_CLASSIFICATION_STRATEGIES`, validated against
`problem_type` — note `BaselineConfig`'s field uses `validate_default=True` specifically so an
*omitted* `baseline` key still gets validated/corrected instead of silently defaulting to
regression's "mean" for a classification config), `class_balance.strategy == "class_weight"`
(applies `class_weight="balanced"` to `logistic_regression`/`ridge_classifier`/`random_forest`/
`extra_trees`/`svc` — mutually exclusive with `smote_variant`, doesn't affect
`lightgbm`/`xgboost`/`catboost`, which need model-specific weighting APIs not yet wired up),
`fallback_on_failure.cv_fold_error_threshold` (tolerates that many individual CV fold failures
before aborting a model, via `_cv_predict_metric`'s safe/unsafe fold-fitting paths), `metrics`
(see Metric convention above), and `datetime_features` (`pipeline_builder.expand_datetime_features`
derives `_year`/`_month`/`_day`/`_dayofweek` numeric columns and registers matching `FeatureSpec`
entries automatically — the raw datetime column itself is never added to `config.features`, so
it's excluded from `X` by the `ColumnTransformer`'s `remainder="drop"`). `FeatureSpec.outlier_handling`
(`{"method": "clip", "lower_pct": N, "upper_pct": N}`) inserts a `_PercentileClipper` step
(winsorizes to train-fold-learned percentiles) between imputation and scaling for that feature.

## Known gaps (not yet implemented)

- Only single-flat-CSV datasets are supported (`job_common.py::download_dataset` picks the
  newest/`train`-named CSV after unzipping) — no multi-file/multi-table dataset support.
- `logs/` still grows unbounded (`cleanup.py` only reclaims `data/`/`checkpoints/` for published
  jobs) — rotate/prune manually if it becomes a problem.
- No early stopping or Optuna pruning for the gradient-boosted models — every trial trains to
  its full `n_estimators`/`iterations`. Would need each model's own eval-set/callback API wired
  through `_tune_one_model`'s objective function.
- No `GroupKFold`/`TimeSeriesSplit` support in `build_cv` — grouped or time-ordered datasets can
  leak across folds. Would need a schema field naming which column is the group/time key.
- Narrow Optuna search spaces (no L1/L2 regularization knobs on the boosted models, no
  column-subsampling parameters beyond xgboost's `subsample`).
- The published notebook fits on all rows and only prints the benchmarking-time score — no
  EDA plots, no held-out evaluation output (confusion matrix / residual plot).
  `run_benchmark`'s held-out `test_score` isn't surfaced in the notebook either.
- `class_balance.strategy == "class_weight"` doesn't cover `lightgbm`/`xgboost`/`catboost`
  (each has a different weighting API — `class_weight` param, `scale_pos_weight`, `class_weights`
  respectively — not yet implemented).
- LIME was requested alongside SHAP but not implemented — only SHAP landed (see below).

## IN PROGRESS as of 2026-09-27 — broken, fix tomorrow before anything else

SHAP feature importance was just added to `notebook_builder.py::_render_shap_cell` (a model-
agnostic cell using `shap.Explainer(pipe.predict, shap_sample)` + `shap.plots.bar`, appended after
the fit cell in every notebook) and to `requirements.txt` (`shap`, pulls in `numba`/`llvmlite`).
It worked in one manual smoke test but **broke 2 existing tests** that were passing before this
change and were not re-investigated before stopping for the day:
- `tests/test_notebook_builder_datetime_outlier.py::test_build_notebook_datetime_and_outlier_together_actually_execute`
- `tests/test_run_pipeline_datetime_resume.py::test_resumed_bench_result_still_gets_datetime_features_expanded_before_notebook_build`

Both are real-notebook-execution tests (build a notebook, actually run it via `execute_notebook`,
assert `ok`) — the failure is almost certainly the new SHAP cell interacting badly with one of the
datetime-expansion or outlier-clipping code paths (e.g. `shap.Explainer` being handed a `shap_sample`
that still contains the raw (non-numeric) datetime column, since `X = df.drop(columns=[TARGET])`
keeps every column including ones the `ColumnTransformer` itself ignores). **Do not touch this
tomorrow without first running `.venv\Scripts\python.exe -m pytest tests/ -q` to see the actual
tracebacks** — they were not read before stopping, so the root cause here is a guess, not a
diagnosis. Full run before this change was 101/101 passing.
