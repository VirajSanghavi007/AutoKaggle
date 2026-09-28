import re
from pathlib import Path

QUEUE_PATH = Path(__file__).resolve().parent.parent / "kaggledataset.txt"


class QueueError(Exception):
    pass


def _parse_kaggle_url(url: str) -> dict:
    url = url.strip()
    if not url:
        raise QueueError("empty url in queue")

    ds_match = re.search(r"kaggle\.com/datasets/([a-zA-Z0-9\-_]+/[a-zA-Z0-9\-_]+)", url)
    if ds_match:
        return {"target_slug": ds_match.group(1)}

    raise QueueError(f"could not parse a dataset slug from url: {url}")


def peek_next() -> dict | None:
    if not QUEUE_PATH.exists():
        return None
    lines = [l.strip() for l in QUEUE_PATH.read_text().splitlines()]
    lines = [l for l in lines if l and not l.startswith("#")]
    if not lines:
        return None
    parsed = _parse_kaggle_url(lines[0])
    parsed["raw_url"] = lines[0]
    return parsed


def remove_first(raw_url: str) -> None:
    if not QUEUE_PATH.exists():
        return
    lines = QUEUE_PATH.read_text().splitlines()
    removed = False
    kept = []
    for l in lines:
        if not removed and l.strip() == raw_url.strip():
            removed = True
            continue
        kept.append(l)
    QUEUE_PATH.write_text("\n".join(kept) + ("\n" if kept else ""))
