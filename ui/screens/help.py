"""
Help and information screen for ctrlSPEAK.
"""

import logging
from textual.screen import Screen
from textual.containers import Container, Vertical, ScrollableContainer
from textual.widgets import Static, Label, Markdown
from textual.app import ComposeResult
from textual.binding import Binding

from ..state import AppState

logger = logging.getLogger("ctrlspeak.ui.help")


HELP_TEXT = """
# ctrlSPEAK Help

## Keyboard Shortcuts

### Global Hotkeys
- **Ctrl (triple-tap)** - Start/Stop recording
- **Alt+Esc** - Quit application

### UI Navigation
- **D** - Device selection screen
- **S** - Settings screen
- **H** - Help screen (this screen)
- **Q** or **Ctrl+C** - Quit application
- **Esc** - Go back / Close screen

## How It Works

1. **Start Recording**: Triple-tap the Ctrl key quickly
2. **Speak**: The app detects speech above the RMS threshold
3. **Pause**: When you pause for the configured silence duration, audio is segmented
4. **Transcribe**: Each segment is transcribed and copied to clipboard
5. **Stop**: Triple-tap Ctrl again to stop recording

## Audio Segmentation

The app uses **RMS-based speech detection** to automatically segment your recording:

- **RMS Threshold**: Minimum audio level to detect speech (default: 0.01)
- **Silence Duration**: How long to wait before segmenting (default: 1.0s)
- **Min Chunk Duration**: Minimum length to transcribe (default: 0.5s)

## Waveform Display

The waveform shows real-time audio levels:
- **Green bars** = Speech detected (above threshold)
- **Gray bars** = Silence (below threshold)
- **RMS value** = Current audio level

## Models

ctrlSPEAK supports multiple speech recognition models:

- **parakeet** - Fast, accurate (default)
- **parakeet-mlx** - Apple Silicon optimized
- **canary** - Multilingual support
- **whisper** - OpenAI Whisper models

Use `--model <name>` to select a model at startup.

## Tips

- Adjust the **RMS threshold** if speech detection is too sensitive or not sensitive enough
- Increase **silence duration** for longer pauses between sentences
- Use a quiet environment for best results
- Check device settings (D key) to select the right microphone

## About

ctrlSPEAK is an open-source speech-to-text tool for macOS.

Repository: https://github.com/patelnav/ctrlspeak

---

Press **Esc** to close this help screen.
"""


class HelpScreen(Screen):
    """
    Help and information screen.

    Shows:
    - Keyboard shortcuts
    - Usage instructions
    - Model information
    - Tips and tricks
    """

    BINDINGS = [
        Binding("escape", "dismiss", "Back", show=True),
        Binding("q", "dismiss", "Back", show=False),
    ]

    CSS = """
    HelpScreen {
        align: center middle;
    }

    #help-container {
        width: 90;
        height: 90%;
        border: solid $primary;
        padding: 2;
    }
    """

    def __init__(self, app_state: AppState, **kwargs):
        """
        Initialize the help screen.

        Args:
            app_state: Application state
        """
        super().__init__(**kwargs)
        self.app_state = app_state

    def compose(self) -> ComposeResult:
        """Create child widgets."""
        with Container(id="help-container"):
            with ScrollableContainer():
                yield Markdown(HELP_TEXT)

    def action_dismiss(self) -> None:
        """Dismiss the screen and go back."""
        self.dismiss()

    def on_mount(self) -> None:
        """Called when screen is mounted."""
        logger.info("HelpScreen mounted")
