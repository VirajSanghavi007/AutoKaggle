import json
import sys
from pathlib import Path

from schema import PipelineConfig, apply_guardrails

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def validate(kernel_slug: str) -> tuple[bool, str]:
    config_path = DATA_DIR / kernel_slug / "pipeline_config.json"
    if not config_path.exists():
        return False, f"no pipeline_config.json found at {config_path}"
    try:
        parsed = json.loads(config_path.read_text())
    except json.JSONDecodeError as e:
        return False, f"pipeline_config.json is not valid JSON: {e}"
    try:
        config = PipelineConfig(**parsed)
    except Exception as e:
        return False, f"pipeline_config.json does not match PipelineConfig schema: {e}"

    corrections = apply_guardrails(config)
    if corrections:
        config_path.write_text(config.model_dump_json(indent=2))
        return True, f"valid, guardrails applied: {corrections}"
    return True, "valid"


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python validate_pipeline_config.py <kernel_slug>")
        sys.exit(2)
    ok, message = validate(sys.argv[1])
    print(("[VALID] " if ok else "[INVALID] ") + message)
    sys.exit(0 if ok else 1)
