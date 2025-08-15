### Why this is worth it
- **Apple Silicon perf/efficiency**: `parakeet-mlx` runs on MLX, which is designed for M‑series chips and should outperform PyTorch/MPS for this model class on macOS [[parakeet-mlx](https://github.com/senstella/parakeet-mlx)].  
- Community interest already captured in the project issue [[Issue #1](https://github.com/patelnav/ctrlspeak/issues/1)].

### Implementation plan

- **Dependencies (optional install)**
  - Add a separate optional requirements file: `requirements-mlx.txt` with:
    - `mlx` (Apple MLX)
    - `parakeet-mlx`
    - Reuse existing `numpy`/`soundfile` already in `requirements.txt`
  - Document install: `uv pip install -r requirements-mlx.txt` [[memory:3574993]].
  - Gate feature on macOS arm64 and successful import.

- **New model class**
  - Create `models/parakeet_mlx.py` with `ParakeetMLXModel(BaseSTTModel)`.
  - `load_model()`:
    - `from parakeet_mlx import from_pretrained`
    - Default HF ID: `"mlx-community/parakeet-tdt-0.6b-v2"` (as shown in the repo docs).
    - No torch device handling; MLX manages its own device.
  - `transcribe_batch(audio_paths: List[str]) -> List[str]`:
    - For each path, use `parakeet_mlx.audio.load_audio(path, model.preprocessor_config.sample_rate)`
    - Call `model.generate(...)` or the high-level transcribe util and extract clean text.
    - Return list of strings, mirroring current `BaseSTTModel` contract.

- **Factory integration**
  - Extend `ModelFactory._DEFAULT_ALIASES`:
    - Add `"parakeet-mlx": "mlx-community/parakeet-tdt-0.6b-v2"`.
  - In `ModelFactory.get_model(...)`:
    - Route to `ParakeetMLXModel` when:
      - alias is `"parakeet-mlx"`, or
      - model string starts with `"mlx-community/"` or `"mlx/"`.
    - If import fails, raise `ImportError` with a clear message to install `requirements-mlx.txt`.

- **CLI and UX**
  - `--list-models` should include `parakeet-mlx` with a note: “Apple Silicon/MLX (optional dependency)”.
  - Keep default as current NeMo `"parakeet"`; users can opt in to `"parakeet-mlx"`.
  - On non‑macOS or non‑arm64, still allow selection but warn and fall back if MLX stack is missing.

- **Transcription worker compatibility (Phase 1)**
  - No worker changes required: it writes temp WAVs and calls `transcribe_batch`, which the MLX model will support.
  - This minimizes risk and keeps latency characteristics unchanged initially.

- **Streaming (Phase 2, optional)**
  - Add optional streaming path leveraging `model.transcribe_stream(...)` from `parakeet-mlx` for lower latency.
  - Approach:
    - Add optional streaming interface to `BaseSTTModel` (e.g., `supports_streaming`, `start_stream`, `add_audio`, `get_stream_text`, `end_stream`).
    - In `transcription.py`, if the model supports streaming, bypass temp-file path and feed chunks directly.
    - Keep a feature flag to toggle this while stabilizing.

- **Tests**
  - Add a macOS/arm64‑only test (skip otherwise) that:
    - Loads `ParakeetMLXModel` (or `--model parakeet-mlx`) and transcribes `test.wav`.
    - Asserts non-empty/transcription shape and no exceptions.
  - Keep existing tests unchanged.

- **Docs**
  - README:
    - Add a “Apple Silicon (MLX) optional acceleration” section.
    - Show install via `uv pip install -r requirements-mlx.txt` [[memory:3574993]].
    - Show usage: `ctrlspeak --model parakeet-mlx`.
    - Briefly note the benefits and platform limits.

- **Homebrew (optional follow-up)**
  - If you want MLX out-of-the-box for Brew users, add an optional subformula or caveat in `ctrlspeak.rb` about installing MLX extras. Otherwise, keep it opt‑in via manual install.

- **Compatibility notes**
  - Verify MLX supports Python 3.13.3 on your machine. If not, pin to a supported version in `requirements-mlx.txt` or note the constraint in README.
  - Keep NeMo/PyTorch path as the cross‑platform default.

### Minimal code touchpoints
- Add `models/parakeet_mlx.py`
- Edit `models/factory.py` (aliases + routing)
- Optionally add `requirements-mlx.txt`
- Update `ctrlspeak.py` list-models output and README

If you want, I can implement Phase 1 (non‑streaming MLX model) now, then we can iterate on the streaming worker after validation.

- Added a concrete, low-risk plan to add `parakeet-mlx` alongside existing NeMo models.
- Kept cross-platform default unchanged; MLX is optional and gated.
- Outlined follow-ups for streaming, tests, docs, and optional Homebrew note.