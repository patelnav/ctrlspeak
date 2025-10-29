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
    Displays current recording status.

    Shows:
    - Status (Ready or Recording)
    - Timer (MM:SS format) when recording
    - Pulsing indicator when recording
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
        text = Text()

        if not is_recording:
            text.append("Status: ", style="bold cyan")
            text.append("Ready", style="bold green")
            return text

        # Calculate elapsed time
        if self.app_state.recording_start_time:
            elapsed = time.time() - self.app_state.recording_start_time
            minutes, seconds = divmod(int(elapsed), 60)
        else:
            minutes, seconds = 0, 0

        # Pulsing recording indicator
        pulse = "●" if int(time.time() * 2) % 2 == 0 else "○"

        # Build status line - simplified
        text.append("Recording ", style="bold cyan")
        text.append(f"{pulse} ", style="bold red")
        text.append(f"{minutes:02d}:{seconds:02d}", style="bold white")

        return text

    def on_mount(self) -> None:
        """Called when widget is mounted."""
        logger.debug("RecordingStatusWidget mounted")
        # Fast refresh for smooth timer and pulse animation
        self.set_interval(0.1, self.refresh)
