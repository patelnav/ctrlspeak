# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**ctrlSPEAK** is a macOS-native speech-to-text utility that converts voice to text via triple-tap Ctrl hotkey. Features a Textual TUI, multi-model support, and real-time audio segmentation with automatic paste.

**Requirements:** macOS 12.3+, Python 3.13.3 (pyenv)

## Development Commands

```bash
# Setup
source .venv/bin/activate              # Always activate first
uv pip install -r requirements.txt     # Core deps
uv pip install -r requirements-nvidia.txt   # Optional: NVIDIA models
uv pip install -r requirements-whisper.txt  # Optional: Whisper

# Run
python ctrlspeak.py                    # Default (parakeet MLX)
python ctrlspeak.py --model canary     # NVIDIA multilingual
python ctrlspeak.py --debug            # Verbose logging
python ctrlspeak.py --list-models      # Show available models

# Test
python -m pytest tests/ -v
python -m pytest tests/test_file.py::test_name -v

# Code quality
uv format .                            # Black style
uv lint .                              # Ruff style
```

## Architecture Overview

### Data Flow
```
Triple-tap Ctrl → on_activate() → start_recording()
                                       ↓
                          AudioManager captures audio
                                       ↓
                    RMS segmentation detects silence gaps
                                       ↓
                   Audio chunks queued to transcription_queue
                                       ↓
                  transcription_worker processes in background
                                       ↓
                     Results appended to transcribed_chunks
                                       ↓
Triple-tap Ctrl → stop_recording() → join queue → paste to clipboard
```

### Key Components

**Global State (`state.py`)**: Central hub for shared variables - model instance, audio manager, transcription queue, device info. `KNOWN_MODELS` dict defines all supported models.

**Hotkey System (`hotkeys.py` + `utils/keyboard_shortcuts.py`)**: Uses pynput for global Ctrl key detection. Triple-tap within 0.5s triggers `on_activate()`. Manages recording start/stop, queue draining, and clipboard paste.

**Audio Pipeline (`utils/audio.py`)**: `AudioManager` class handles recording at 16kHz. RMS-based segmentation cuts on 1.0s silence. Min chunk 0.5s. Segments queued for transcription.

**Transcription Worker (`transcription.py`)**: Daemon thread pulls from queue, writes temp WAV, calls `model.transcribe_batch()`, appends text to results list. Terminates on `None` sentinel.

**Model System (`models/`)**: Factory pattern in `factory.py` resolves aliases → model classes. Base class in `base_model.py` defines interface: `load_model()`, `transcribe_batch()`. Implementations: `parakeet_mlx.py` (Apple Silicon), `parakeet.py`/`canary.py` (NVIDIA NeMo), `whisper.py` (OpenAI).

**TUI (`ui/`)**: Textual framework app. `AppState` dataclass for UI state (recording, device, model, language). Screens: RecordingScreen (main), DeviceSelection, ModelSelection, ModelLoading, Help, LogViewer.

### Threading Model
- **Main thread**: Textual async UI
- **Keyboard thread**: pynput listener (background)
- **Audio thread**: sounddevice stream callback
- **Transcription thread**: queue worker daemon

## Model Aliases

| Alias | Model | Framework |
|-------|-------|-----------|
| `parakeet` (default) | `mlx-community/parakeet-tdt-0.6b-v3` | MLX |
| `canary` | `nvidia/canary-1b-flash` | NeMo |
| `canary-180m` | `nvidia/canary-180m-flash` | NeMo |
| `whisper` | `openai/whisper-large-v3` | Transformers |

## Release Workflow

Update version in three files, then tag:
```bash
# 1. Edit: VERSION, __init__.py, pyproject.toml
# 2. Commit and tag
git add VERSION __init__.py pyproject.toml
git commit -m "chore: bump version to X.Y.Z"
git tag vX.Y.Z && git push && git push origin vX.Y.Z

# 3. Update Homebrew tap with SHA256
curl -sL https://github.com/patelnav/ctrlspeak/archive/refs/tags/vX.Y.Z.tar.gz | shasum -a 256
```

## Key Files for Specific Features

- **Triple-tap detection**: `utils/keyboard_shortcuts.py`
- **Audio recording/segmentation**: `utils/audio.py`
- **Model loading**: `model_loader.py`
- **Transcription worker**: `transcription.py`
- **Hotkey activation logic**: `hotkeys.py`
- **UI state management**: `ui/state.py`

