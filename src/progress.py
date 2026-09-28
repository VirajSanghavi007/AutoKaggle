import time


class StepTracker:
    def __init__(self, total_steps: int, log):
        self.total_steps = total_steps
        self.log = log
        self.current = 0
        self.start_time = time.time()

    def step(self, name: str):
        self.current += 1
        elapsed = time.time() - self.start_time
        avg_per_step = elapsed / self.current
        remaining = self.total_steps - self.current
        eta = avg_per_step * remaining
        pct = round(self.current / self.total_steps * 100, 1)
        self.log.info(
            "[%d/%d] (%.1f%%) %s - elapsed %.1fs, eta %.1fs",
            self.current, self.total_steps, pct, name, elapsed, eta,
        )


class OptunaProgressCallback:
    def __init__(self, n_trials: int, model_name: str, log, report_every: int = 5):
        self.n_trials = n_trials
        self.model_name = model_name
        self.log = log
        self.report_every = max(1, report_every)
        self.start_time = time.time()

    def __call__(self, study, trial):
        done = trial.number + 1
        if done % self.report_every != 0 and done != self.n_trials:
            return
        elapsed = time.time() - self.start_time
        avg = elapsed / done
        eta = avg * (self.n_trials - done)
        pct = round(done / self.n_trials * 100, 1)
        best = study.best_value if study.best_trial else float("nan")
        self.log.info(
            "  optuna[%s] %d/%d (%.1f%%) best=%.5f elapsed=%.1fs eta=%.1fs",
            self.model_name, done, self.n_trials, pct, best, elapsed, eta,
        )
