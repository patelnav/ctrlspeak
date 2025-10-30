"""
Utilities to make tqdm safe in threaded contexts by using a thread-based lock.

This avoids tqdm attempting to create multiprocessing locks from a worker thread
on macOS, which can trigger "bad value(s) in fds_to_keep" errors.
"""

from __future__ import annotations

import threading

TQDM_LOCK_SET = False


def ensure_tqdm_thread_lock() -> None:
    """Set a thread-based lock for tqdm across std/auto variants.

    Safe to call multiple times. If tqdm is not available, this is a no-op.
    """
    global TQDM_LOCK_SET
    try:
        import tqdm  # type: ignore
        lock = threading.RLock()

        # Standard tqdm class
        if hasattr(tqdm, 'tqdm') and hasattr(tqdm.tqdm, 'set_lock'):
            tqdm.tqdm.set_lock(lock)

        # Explicit std module class
        if hasattr(tqdm, 'std') and hasattr(tqdm.std, 'tqdm') and hasattr(tqdm.std.tqdm, 'set_lock'):
            tqdm.std.tqdm.set_lock(lock)

        # auto module class
        if hasattr(tqdm, 'auto') and hasattr(tqdm.auto, 'tqdm') and hasattr(tqdm.auto.tqdm, 'set_lock'):
            tqdm.auto.tqdm.set_lock(lock)

        TQDM_LOCK_SET = True
    except Exception:
        # tqdm may not be installed or import may fail in stripped environments.
        # Silently ignore – this is a best-effort hardening step.
        pass

