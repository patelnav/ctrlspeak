# 🎙️ ctrlSPEAK  
**Turn your voice into text with a triple-tap — minimal, fast, and macOS-native.**

## 🚀 Overview

**ctrlSPEAK** is your *set-it-and-forget-it* speech-to-text companion. Triple-tap `Ctrl`, speak your mind, and watch your words appear wherever your cursor blinks — effortlessly copied and pasted. Built for macOS, it’s lightweight, low-overhead, and stays out of your way until you call it.

## ✨ Features

- 🖥️ **Minimal Interface**: Runs quietly in the background via the command line  
- ⚡ **Triple-Tap Magic**: Start/stop recording with a quick `Ctrl` triple-tap  
- 📋 **Auto-Paste**: Text lands right where you need it, no extra clicks  
- 🔊 **Audio Cues**: Hear when recording begins and ends  
- 🍎 **Mac Optimized**: Harnesses Apple Silicon’s MPS for blazing performance  
- 🌟 **Top-Tier Models**: Powered by NVIDIA NeMo and OpenAI Whisper  

## 🛠️ Get Started

- **System**: macOS 12.3+ (MPS acceleration supported)  
- **Python**: 3.11+  
- **Permissions**:  
  - 🎤 Microphone (for recording)  
  - ⌨️ Accessibility (for shortcuts)  
*Grant these on first launch and you’re good to go!*

## 🧰 Entry Points

- `ctrlspeak.py`: The full-featured star of the show  
- `live_transcribe.py`: Continuous transcription for testing vibes  
- `test_transcription.py`: Debug or benchmark with ease  


### Workflow

1. Run ctrlSPEAK in a terminal window
2. Triple-tap Ctrl to start recording
3. Speak clearly into your microphone
4. Triple-tap Ctrl again to stop recording
5. The transcribed text will be automatically pasted at your cursor position

## Models

ctrlSPEAK uses open-source speech recognition models:

- **Whisper** (default): OpenAI's fast and accurate general-purpose speech recognition model
- **Parakeet**: NVIDIA NeMo's English-only model with high accuracy
- **Canary**: NVIDIA NeMo's multilingual model supporting English, German, French, and Spanish

The models are automatically downloaded from HuggingFace the first time you use them.

## Models Tested

1. **Parakeet (NVIDIA)** - `nvidia/parakeet-tdt-1.1b`
2. **Canary (NVIDIA)** - `nvidia/canary-1b`
3. **Whisper (OpenAI)** - `openai/whisper-large-v3-turbo`

## Performance Comparison

| Model   | Load Time | Transcription Time | Transcription Quality | Output Example |
|---------|-----------|-------------------|----------------------|----------------|
| Parakeet | 14.84s    | 2.21s             | Good, without punctuation | "well i don't wish to see it any more observed phoebe turning away her eyes it is certainly very like the old portrait" |
| Canary   | 8.00s     | 25.45s            | Excellent, with punctuation | "Well, I don't wish to see it any more, observed Phoebe, turning away her eyes. It is certainly very like the old portrait." |
| Whisper  | 1.45s     | 2.23s             | Good, without punctuation | "well i don't wish to see it any more observed phoebe turning away her eyes it is certainly very like the old portrait" |

## Permissions

The app requires:
- Microphone access (for recording audio)
- Accessibility permissions (for global keyboard shortcuts)

You'll be prompted to grant these permissions on first run.

## Troubleshooting

- **No sound on recording start/stop**: Ensure your system volume is not muted
- **Keyboard shortcuts not working**: Grant accessibility permissions in System Settings
- **Transcription errors**: Try speaking more clearly or using the other model

## License

[MIT License](LICENSE)