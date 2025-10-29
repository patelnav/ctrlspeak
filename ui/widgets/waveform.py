"""
Waveform display widget for ctrlSPEAK.
"""

import logging
from textual.widgets import Static
from textual.app import ComposeResult
from rich.text import Text
from rich.style import Style

from ..state import AppState

logger = logging.getLogger("ctrlspeak.ui.waveform")


class WaveformDisplay(Static):
    """
    Displays a real-time waveform visualization based on RMS levels.

    Shows:
    - Bar graph visualization of audio levels
    - Color coding: green (speech detected), gray (silence)
    - Numeric RMS value
    """

    def __init__(self, app_state: AppState, **kwargs):
        """
        Initialize the waveform display.

        Args:
            app_state: Application state for accessing current RMS
        """
        super().__init__(**kwargs)
        self.app_state = app_state
        self.bar_width = 50  # Width of the bar graph

    def render(self) -> Text:
        """Render the waveform display."""
        rms = self.app_state.current_rms
        threshold = self.app_state.rms_threshold
        is_recording = self.app_state.is_recording

        # Create the display text
        text = Text()

        if not is_recording:
            text.append("Waveform: ", style="bold cyan")
            text.append("Not recording", style="dim")
            return text

        # Calculate bar length based on RMS (scale it for visibility)
        # Log scale for better visualization
        import math
        if rms > 0:
            # Scale RMS to 0-100 range (adjust multiplier as needed)
            scaled_rms = min(100, int(math.log10(rms + 1) * 50 + 50))
        else:
            scaled_rms = 0

        bar_length = int((scaled_rms / 100) * self.bar_width)
        bar_length = max(0, min(bar_length, self.bar_width))

        # Determine if speech is detected
        is_speech = rms >= threshold

        # Color coding
        bar_style = "bold green" if is_speech else "dim white"
        empty_style = "dim"

        # Build the bar
        text.append("Level: ", style="bold cyan")
        text.append("█" * bar_length, style=bar_style)
        text.append("░" * (self.bar_width - bar_length), style=empty_style)

        # Show numeric RMS value
        text.append(f" RMS: {rms:.6f}", style="dim")

        # Show speech detection status
        if is_speech:
            text.append(" ", style="")
            text.append("● SPEECH", style="bold green")
        else:
            text.append(" ", style="")
            text.append("○ silence", style="dim")

        return text

    def on_mount(self) -> None:
        """Called when widget is mounted."""
        logger.debug("WaveformDisplay mounted")
        # Set up periodic refresh for animation
        self.set_interval(0.05, self.refresh)  # 20 FPS
