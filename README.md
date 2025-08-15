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
- **Python**: 3.10  
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

- **Parakeet 0.6B** (default): NVIDIA NeMo's `nvidia/parakeet-tdt-0.6b-v3` model. Good balance of speed, accuracy, punctuation, and capitalization.
- **Canary**: NVIDIA NeMo's `nvidia/canary-1b-flash` multilingual model (En, De, Fr, Es) with punctuation, but can be slower.
- **Canary (180M)**: NVIDIA NeMo's `nvidia/canary-180m-flash` multilingual model, smaller and less accurate.
- **Whisper** (optional): OpenAI's `openai/whisper-large-v3` model. A fast, accurate, and powerful model that includes excellent punctuation and capitalization.
  - To use Whisper, install additional dependencies: `uv pip install -r requirements-whisper.txt`

**Note:** The `nvidia/parakeet-tdt-1.1b` model is also available for testing, but it is not recommended for general use as it lacks punctuation and is slower than the `0.6b` model.

The models are automatically downloaded from HuggingFace the first time you use them.

### Listing Supported Models

To see a list of all supported models, use the `--list-models` flag:

```bash
ctrlspeak --list-models
```

This will output a list of the available model aliases and their corresponding Hugging Face model names.

### Apple Silicon (MLX) Acceleration

For users on Apple Silicon (M1/M2/M3 Macs), an optimized version of the Parakeet model is available using Apple's MLX framework. This can provide a significant performance boost.

To use the MLX-accelerated model:

1.  **Install MLX dependencies:**
    ```bash
    uv pip install -r requirements-mlx.txt
    ```

2.  **Run with the MLX model:**
    ```bash
    ctrlspeak --model parakeet-mlx
    ```

## Model Selection

You can select a model using the `--model` flag. You can use either the full model name from HuggingFace or a short alias.

**Short Names:**

*   `parakeet`: NVIDIA's Parakeet TDT 0.6B model (v3).
*   `canary`: NVIDIA's Canary 1B Flash model.
*   `canary-180m`: NVIDIA's Canary 180M Flash model.
*   `parakeet-mlx`: Parakeet 0.6B optimized for Apple Silicon (MLX).
*   `whisper`: OpenAI's Whisper v3 model.

**Full Model URL:**

You can also provide a full model URL from Hugging Face. For example:

```bash
ctrlspeak --model nvidia/parakeet-tdt-1.1b
```

This will download and use the specified model.

```bash
# Using Homebrew installation
ctrlspeak --model parakeet  # Default
ctrlspeak --model canary         # Multilingual with punctuation
ctrlspeak --model canary-180m    # The smaller Canary model
ctrlspeak --model canary-v2
ctrlspeak --model whisper        # OpenAI's model
ctrlspeak --model parakeet-mlx # MLX-accelerated model

# Using manual installation
python ctrlspeak.py --model parakeet
python ctrlspeak.py --model canary
python ctrlspeak.py --model canary-180m
python ctrlspeak.py --model canary-v2
python ctrlspeak.py --model whisper
python ctrlspeak.py --model parakeet-mlx
```

For debugging, you can use the `--debug` flag:

```bash
ctrlspeak --debug
```

## Models Tested

1. **Parakeet 0.6B (NVIDIA)** - `nvidia/parakeet-tdt-0.6b-v3` (Default)
2. **Parakeet 1.1B (NVIDIA)** - `nvidia/parakeet-tdt-1.1b`
3. **Canary (NVIDIA)** - `nvidia/canary-1b-flash`
4. **Canary (NVIDIA)** - `nvidia/canary-180m-flash`
5. **Canary (NVIDIA)** - `nvidia/canary-1b-v2`
6. **Whisper (OpenAI)** - `openai/whisper-large-v3`

## Performance Comparison

| Model                                                    | Load Time (s) | Transcription Time (s) | Notes                                            |
| -------------------------------------------------------- | ------------- | ---------------------- | ------------------------------------------------ |
| **`nvidia/parakeet-tdt-0.6b-v3`** (NeMo/PyTorch)         | 14.83         | 1.44                   | Default model, good balance.                     |
| **`mlx-community/parakeet-tdt-0.6b-v2`** (Apple Silicon) | **0.31**      | **1.14**               | **Recommended for Apple Silicon.** Much faster load. |
| **`nvidia/canary-1b-flash`**                             | 8.09          | 3.30                   | Fast multilingual model.                         |
| **`nvidia/canary-180m-flash`**                           | 2.33          | 3.35                   | Smaller, faster multilingual model.              |
| **`nvidia/canary-1b-v2`**                                | 13.96         | 4.85                   | Newer multilingual model.                        |
| **`openai/whisper-large-v3`**                            | 6.2           | 1.8                    | Very accurate, but slower transcription.         |

*Testing performed on a MacBook Pro (M2 Max) with a 7-second audio file (`test.wav`). Your results may vary.*

**Note:** Whisper model uses translate mode to enable proper punctuation and capitalization for English transcription.

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