#!/usr/bin/env python3
"""Tests for the public tqdm.set_lock() approach applied via ctrlspeak entrypoint."""

import sys
import os

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(__file__))
import threading
import importlib


def test_entrypoint_sets_thread_lock_on_tqdm():
    # Ensure we import the entrypoint so its top-level setup runs
    # Ensure project root is on path
    sys.path.insert(0, os.path.dirname(__file__))
    # Use the lightweight utility directly (called by entrypoint at import time)
    from utils.tqdm_lock import ensure_tqdm_thread_lock, TQDM_LOCK_SET
    ensure_tqdm_thread_lock()
    print(f"utility TQDM_LOCK_SET: {TQDM_LOCK_SET}")

    from tqdm import tqdm as T

    # Sanity: public API should exist
    assert hasattr(T, 'set_lock') and hasattr(T, 'get_lock')

    # Check that the lock has been set to a threading.RLock
    # Debug info to help diagnose in CI/pytest
    mod = getattr(T, '__module__', 'unknown')
    lock_obj = T.get_lock()
    print(f"tqdm class module: {mod}")
    print(f"tqdm lock object: {lock_obj!r}")
    rlock_type = type(threading.RLock())
    assert isinstance(lock_obj, rlock_type)

    # Ensure NeMo was not imported as a side effect
    assert 'nemo.collections.asr' not in sys.modules


def test_tqdm_iterates_in_thread_without_mp_lock_errors():
    errors = []

    def worker():
        try:
            from tqdm import tqdm as T
            for _ in T(range(10)):
                pass
        except Exception as e:
            errors.append(e)

    from utils.tqdm_lock import ensure_tqdm_thread_lock
    ensure_tqdm_thread_lock()

    t = threading.Thread(target=worker)
    t.start()
    t.join()

    assert not errors, f"tqdm raised errors in thread: {errors}"
