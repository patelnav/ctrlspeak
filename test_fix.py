#!/usr/bin/env python3
"""Quick test to verify tqdm patch works in worker thread."""
import sys
import os

# Apply the same patch as ctrlspeak.py
import builtins
_original_import = builtins.__import__

def _patched_import(name, *args, **kwargs):
    """Intercept tqdm imports and replace with no-op version."""
    module = _original_import(name, *args, **kwargs)

    if name == 'tqdm' or name.startswith('tqdm.'):
        class NoOpTqdm:
            def __init__(self, iterable=None, *args, **kwargs):
                self.iterable = iterable if iterable is not None else []
            def __iter__(self):
                return iter(self.iterable)
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def update(self, n=1):
                pass
            def close(self):
                pass
            def set_description(self, desc=None, refresh=True):
                pass
            @staticmethod
            def write(s, **kwargs):
                pass

        if hasattr(module, 'tqdm'):
            module.tqdm = NoOpTqdm
        if hasattr(module, 'std') and hasattr(module.std, 'tqdm'):
            module.std.tqdm = NoOpTqdm

    return module

builtins.__import__ = _patched_import

# Now test transcription
import threading
import queue
from models.factory import ModelFactory

def test_transcription():
    print("Loading model...")
    model = ModelFactory.get_model("nvidia/parakeet-tdt-0.6b-v3", verbose=False)
    model.load_model()
    print("✓ Model loaded successfully")

    # Test transcription in a thread (simulating the worker)
    result_queue = queue.Queue()

    def worker():
        try:
            # Try to transcribe a short audio file
            results = model.transcribe_batch(["test.wav"])
            result_queue.put(("success", results))
        except Exception as e:
            result_queue.put(("error", str(e)))

    print("Starting transcription worker thread...")
    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(timeout=10)

    if not result_queue.empty():
        status, data = result_queue.get()
        if status == "success":
            print("✓ Transcription completed without errors!")
            print(f"Result: {data}")
        else:
            print(f"✗ Error: {data}")
            sys.exit(1)
    else:
        print("✗ Worker thread timed out")
        sys.exit(1)

if __name__ == "__main__":
    test_transcription()
