import json
from pathlib import Path

CHECKPOINT_ROOT = Path(__file__).resolve().parent.parent / "checkpoints"


def state_path(kernel_slug: str) -> Path:
    return CHECKPOINT_ROOT / kernel_slug / "run_state.json"


def load_state(kernel_slug: str) -> dict:
    path = state_path(kernel_slug)
    if path.exists():
        return json.loads(path.read_text())
    return {
        "eda_done": False, "llm_done": False, "bench_done": False,
        "notebook_done": False, "published_done": False,
    }


def save_state(kernel_slug: str, state: dict):
    path = state_path(kernel_slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, default=str))
