# 🎙️ ctrlSPEAK  

[![Homebrew](https://img.shields.io/badge/Homebrew-Install-orange)](https://github.com/patelnav/homebrew-ctrlspeak)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Turn your voice into text with a triple-tap — minimal, fast, and macOS-native.**

## 🚀 Overview

**ctrlSPEAK** is your *set-it-and-forget-it* speech-to-text companion. Triple-tap `Ctrl`, speak your mind, and watch your words appear wherever your cursor blinks — effortlessly copied and pasted. Built for macOS, it's lightweight, low-overhead, and stays out of your way until you call it.

![ctrlSPEAK Demo](ctrlspeak-demo.gif)

## ✨ Features

- 🖥️ **Minimal Interface**: Runs quietly in the background via the command line  
- ⚡ **Triple-Tap Magic**: Start/stop recording with a quick `Ctrl` triple-tap  
- 📋 **Auto-Paste**: Text lands right where you need it, no extra clicks  
- 🔊 **Audio Cues**: Hear when recording begins and ends  
- 🍎 **Mac Optimized**: Harnesses Apple Silicon's MPS for blazing performance  
- 🌟 **Top-Tier Models**: Powered by NVIDIA NeMo and OpenAI Whisper  

## 🛠️ Get Started

- **System**: macOS 12.3+ (MPS acceleration supported)  
- **Python**: 3.11+  
- **Permissions**:  
  - 🎤 Microphone (for recording)  
  - ⌨️ Accessibility (for shortcuts)  
*Grant these on first launch and you're good to go!*

### 📦 Installation

#### Using Homebrew (Recommended)

```bash
# Install ctrlSPEAK using Homebrew
brew tap patelnav/ctrlspeak
brew install ctrlspeak
```

For faster package installation:
```bash
# Install with UV support for faster package installation
brew install ctrlspeak --with-uv
```

#### Manual Installation

Clone the repository:
```bash
git clone https://github.com/patelnav/ctrlspeak.git
cd ctrlspeak
```

Create and activate a virtual environment:
```bash
# Create a virtual environment
python -m venv .venv

# Activate it on macOS/Linux
source .venv/bin/activate
```

Install dependencies (recommended with UV for faster installation):
```bash
# Install UV first if you don't have it
pip install uv

# Then install dependencies with UV
uv pip install -r requirements.txt

# Or use traditional pip (slower)
pip install -r requirements.txt
```

For Whisper model support (optional):
```bash
# With UV (recommended)
uv pip install -r requirements-whisper.txt

# Or with traditional pip
pip install -r requirements-whisper.txt
```

## 🧰 Entry Points

- `ctrlspeak.py`: The full-featured star of the show  
- `live_transcribe.py`: Continuous transcription for testing vibes  
- `test_transcription.py`: Debug or benchmark with ease  


### Workflow

1. Run ctrlSPEAK in a terminal window:
   ```bash
   # If installed with Homebrew
   ctrlspeak
   
   # If installed manually (from the project directory with activated venv)
   python ctrlspeak.py
   ```
2. Triple-tap Ctrl to start recording
3. Speak clearly into your microphone
4. Triple-tap Ctrl again to stop recording
5. The transcribed text will be automatically pasted at your cursor position

## Models

ctrlSPEAK uses open-source speech recognition models:

- **Parakeet 0.6B** (default): NVIDIA NeMo's `nvidia/parakeet-tdt-0.6b-v2` model. Good balance of speed, accuracy, punctuation, and capitalization.
- **Parakeet 1.1B**: NVIDIA NeMo's older `nvidia/parakeet-tdt-1.1b` model. Potentially higher accuracy in some cases, but lacks punctuation.
- **Canary**: NVIDIA NeMo's `nvidia/canary-1b` multilingual model (En, De, Fr, Es) with punctuation, but can be slower.
- **Whisper** (optional): OpenAI's `openai/whisper-large-v3-turbo` model. Fast and accurate general-purpose model.
  - To use Whisper, install additional dependencies: `uv pip install -r requirements-whisper.txt`

The models are automatically downloaded from HuggingFace the first time you use them.

### Model Selection

You can specify which model to use with the `--model` flag:

```bash
# Using Homebrew installation
ctrlspeak --model parakeet-0.6b  # Default
ctrlspeak --model parakeet-1.1b  # Older, larger Parakeet
ctrlspeak --model canary         # Multilingual with punctuation
ctrlspeak --model whisper        # OpenAI's model

# Using manual installation
python ctrlspeak.py --model parakeet-0.6b
python ctrlspeak.py --model parakeet-1.1b
python ctrlspeak.py --model canary
python ctrlspeak.py --model whisper
```

For debugging, you can use the `--debug` flag:
```bash
ctrlspeak --debug
```

## Models Tested

1. **Parakeet 0.6B (NVIDIA)** - `nvidia/parakeet-tdt-0.6b-v2` (Default)
2. **Parakeet 1.1B (NVIDIA)** - `nvidia/parakeet-tdt-1.1b`
3. **Canary (NVIDIA)** - `nvidia/canary-1b`
4. **Whisper (OpenAI)** - `openai/whisper-large-v3-turbo`

## Performance Comparison

| Model            | Load Time | Transcription Time | Transcription Quality         | Output Example (test.wav)                  |
|------------------|-----------|-------------------|------------------------------|--------------------------------------------|
| Parakeet 0.6B    | 5.17s     | 0.70s             | Good w/ Punct. & Caps.       | "Well, I don't wish to see it any more, observed Phebe, turning away her eyes. It is certainly very like the old portrait." |
| Parakeet 1.1B    | 10.07s    | 1.08s             | Good, *no* punctuation       | "well i don't wish to see it any more observed phoebe turning away her eyes it is certainly very like the old portrait" |
| Canary           | 8.15s     | 30.82s            | Good w/ Punct. & Caps.       | "Well, I don't wish to see it any more, observed Phoebe, turning away her eyes. It is certainly very like the old portrait." |
| Whisper (large-v3) | 2.41s     | 12.78s            | Good, *no* punctuation       | "well i don't wish to see it any more observed phoebe turning away her eyes it is certainly very like the old portrait" |

## Permissions

The app requires:
- Microphone access (for recording audio)
- Accessibility permissions (for global keyboard shortcuts)

You'll be prompted to grant these permissions on first run.

## Troubleshooting

- **No sound on recording start/stop**: Ensure your system volume is not muted
- **Keyboard shortcuts not working**: Grant accessibility permissions in System Settings
- **Transcription errors**: Try speaking more clearly or using the other model

## Credits

### Sound Effects
- Start sound: ["Notification Pluck On"](https://pixabay.com/sound-effects/notification-pluck-on-269288/) from Pixabay
- Stop sound: ["Notification Pluck Off"](https://pixabay.com/sound-effects/notification-pluck-off-269290/) from Pixabay

## License

[MIT License](LICENSE)

## Release Process

This outlines the steps to create a new release and update the associated Homebrew tap.

**1. Prepare the Release:**

*   Ensure the code is stable and tests pass.
*   Update the version number in the following files:
    *   `VERSION` (e.g., `1.2.0`)
    *   `__init__.py` (`__version__ = "1.2.0"`)
    *   `pyproject.toml` (`version = "1.2.0"`)
*   Commit these version changes:
    ```bash
    git add VERSION __init__.py pyproject.toml
    git commit -m "Bump version to X.Y.Z"
    ```

**2. Tag and Push:**

*   Create a git tag matching the version:
    ```bash
    git tag vX.Y.Z
    ```
*   Push the commits and the tag to the remote repository:
    ```bash
    git push && git push origin vX.Y.Z
    ```

**3. Update Homebrew Tap:**

*   The source code tarball URL is automatically generated based on the tag (usually `https://github.com/<your-username>/ctrlspeak/archive/refs/tags/vX.Y.Z.tar.gz`).
*   Download the tarball using its URL and calculate its SHA256 checksum:
    ```bash
    # Replace URL with the actual tarball link based on the tag
    curl -sL https://github.com/<your-username>/ctrlspeak/archive/refs/tags/vX.Y.Z.tar.gz | shasum -a 256
    ```
*   Clone or navigate to your Homebrew tap repository (e.g., `../homebrew-ctrlspeak`).
*   Edit the formula file (e.g., `Formula/ctrlspeak.rb`):
    *   Update the `url` line with the tag tarball URL.
    *   Update the `sha256` line with the checksum you calculated.
    *   *Optional:* Update the `version` line if necessary (though it's often inferred).
    *   *Optional:* If `requirements.txt` or dependencies changed, update the `depends_on` and `install` steps accordingly.
*   Commit and push the changes in the tap repository:
    ```bash
    cd ../path/to/homebrew-ctrlspeak # Or wherever your tap repo is
    git add Formula/ctrlspeak.rb
    git commit -m "Update ctrlspeak to vX.Y.Z"
    git push
    ```

**4. Verify (Optional):**

*   Run `brew update` locally to fetch the updated formula.
*   Run `brew upgrade ctrlspeak` to install the new version.
*   Test the installed version.