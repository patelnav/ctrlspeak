"""
Recording status widget for ctrlSPEAK.
"""

import logging
import time
from textual.widgets import Static
from rich.text import Text
from rich.table import Table

from ..state import AppState

logger = logging.getLogger("ctrlspeak.ui.status")


class RecordingStatusWidget(Static):
    """
    Displays current recording status information.

    Shows:
    - Recording indicator (pulsing dot)
    - Timer (MM:SS format)
    - Audio duration accumulated
    - Buffer size
    - Silence counter
    """

    def __init__(self, app_state: AppState, **kwargs):
        """
        Initialize the recording status widget.

        Args:
            app_state: Application state for accessing recording info
        """
        super().__init__(**kwargs)
        self.app_state = app_state

    def render(self) -> Text:
        """Render the recording status display."""
        is_recording = self.app_state.is_recording
        buffer_size = self.app_state.buffer_size_samples

        text = Text()

        if not is_recording:
            text.append("Status: ", style="bold cyan")
            text.append("Ready", style="bold green")
            text.append(" - Triple-tap Ctrl to start recording", style="dim")
            return text

        # Calculate elapsed time
        if self.app_state.recording_start_time:
            elapsed = time.time() - self.app_state.recording_start_time
            minutes, seconds = divmod(int(elapsed), 60)
        else:
            minutes, seconds = 0, 0

        # Pulsing recording indicator
        pulse = "●" if int(time.time() * 2) % 2 == 0 else "○"

        # Build status line
        text.append("Recording ", style="bold cyan")
        text.append(f"{pulse} ", style="bold red")
        text.append(f"{minutes:02d}:{seconds:02d}", style="bold white")

        # Audio duration
        if buffer_size > 0:
            audio_duration = buffer_size / 16000  # Assuming 16kHz sample rate
            text.append(f" | Audio: {audio_duration:.1f}s", style="cyan")
        else:
            text.append(" | Audio: 0.0s", style="dim")

        # Buffer info
        text.append(f" | Buffer: {buffer_size:,} samples", style="dim")

        # Show silence detection status if AudioManager is available
        if hasattr(self.app_state, 'current_silence_s'):
            silence = self.app_state.current_silence_s
            if silence > 0:
                text.append(f" | Silence: {silence:.1f}s", style="yellow")

        return text

    def on_mount(self) -> None:
        """Called when widget is mounted."""
        logger.debug("RecordingStatusWidget mounted")
        # Fast refresh for smooth timer and pulse animation
        self.set_interval(0.1, self.refresh)
