import time


def retry(fn, attempts: int = 3, base_delay: float = 2.0, log=None, what: str = "operation"):
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if log:
                log.warning("%s failed (attempt %d/%d): %s", what, attempt, attempts, e)
            if attempt < attempts:
                time.sleep(base_delay * (2 ** (attempt - 1)))
    raise last_exc
