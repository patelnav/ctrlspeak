# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**ctrlSPEAK** is a macOS-native command-line speech-to-text utility that converts voice input to text via a triple-tap global hotkey. The application automatically pastes transcribed text at the cursor position.

**Requirements:** macOS 12.3+

## Environment Setup

### Python & Virtual Environment
- **Python Version:** 3.13.3 (managed via `pyenv`)
- **Virtual Environment:** `.venv` in project root (required)
- **Creation:** `python -m venv .venv` or `uv venv`
- **Activation:** `source .venv/bin/activate` (always before package operations)

### Package Management
- **Tool:** Use `uv` exclusively (not `pip`)
- **Install deps:** `uv pip install -r requirements.txt`
- **Add package:** `uv pip install <package_name>`
- **Update lock:** `uv pip freeze > requirements.txt` after changes
- **Code formatting:** `uv format .` (Black style)
- **Linting:** `uv lint .` (Ruff style)

## Common Development Commands

### Running the Application
```bash
# After activating .venv
python ctrlspeak.py                    # Run main application
python ctrlspeak.py --model parakeet   # Use specific model (default)
python ctrlspeak.py --model canary     # Multilingual NVIDIA model
python ctrlspeak.py --model whisper    # OpenAI Whisper
python ctrlspeak.py --list-models      # Show available models
python ctrlspeak.py --check-only       # Verify config without running
python ctrlspeak.py --debug            # Enable debug logging
```

### Testing
```bash
# Run test suite
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_ctrlspeak.py -v

# Run single test function
python -m pytest tests/test_ctrlspeak.py::test_name -v
```

### Code Quality
```bash
uv format .                            # Format code (Black)
uv lint .                              # Lint code (Ruff)
```

## Project Architecture

Main entry point: `ctrlspeak.py`

Key directories:
- `models/` - Model implementations
- `utils/` - Utility modules
- `tests/` - Test suite

## Model Management

### Available Models

Aliases and their corresponding HuggingFace models (from README):
- `parakeet` - `mlx-community/parakeet-tdt-0.6b-v3` (MLX, default for Apple Silicon)
- `canary` - `nvidia/canary-1b-flash` (NVIDIA NeMo, multilingual)
- `canary-180m` - `nvidia/canary-180m-flash` (NVIDIA NeMo, smaller)
- `whisper` - `openai/whisper-large-v3` (OpenAI Whisper)

Performance comparison is available in the README.md

### Dependency Installation

Optional requirements files for additional model support:
```bash
source .venv/bin/activate
uv pip install -r requirements.txt              # Core
uv pip install -r requirements-nvidia.txt       # Optional: NVIDIA models
uv pip install -r requirements-whisper.txt      # Optional: Whisper model
```

## Project Structure

```
ctrlspeak/
├── ctrlspeak.py
├── cli.py
├── state.py
├── hotkeys.py
├── transcription.py
├── model_loader.py
├── permissions.py
├── logging_config.py
├── environment.py
├── __init__.py
├── models/              (model implementations)
├── utils/               (utility modules)
├── tests/               (test suite)
├── test_*.py            (standalone test/debug scripts)
├── on.wav, off.wav      (audio cue resources)
└── requirements*.txt    (dependency specifications)
```

## macOS Permissions

The app requires (prompted on first run):
- **Microphone access** - For recording audio
- **Accessibility permissions** - For global keyboard shortcuts

## Additional Commands

```bash
python ctrlspeak.py --debug                # Enable debug logging
python ctrlspeak.py --list-models          # List available models
python ctrlspeak.py --check-only           # Verify config without running
```

## Git & Release Workflow

### Version Updates
When preparing a release, update version in three files:
1. `VERSION` - Version string only (e.g., `1.4.0`)
2. `__init__.py` - `__version__ = "1.4.0"`
3. `pyproject.toml` - `version = "1.4.0"`

### Release Steps
```bash
# 1. Update versions in VERSION, __init__.py, pyproject.toml
# 2. Commit version changes
git add VERSION __init__.py pyproject.toml
git commit -m "chore: bump version to X.Y.Z"

# 3. Tag and push
git tag vX.Y.Z
git push && git push origin vX.Y.Z

# 4. Update Homebrew tap (homebrew-ctrlspeak repo)
#    Calculate SHA256 of tarball and update Formula/ctrlspeak.rb
curl -sL https://github.com/patelnav/ctrlspeak/archive/refs/tags/vX.Y.Z.tar.gz | shasum -a 256
```

