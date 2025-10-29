"""
Waveform display widget for ctrlSPEAK.
"""

import logging
import sounddevice as sd
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

    def _get_device_name(self) -> str:
        """Get the name of the currently loaded (active) device."""
        try:
            # Use loaded_device (actually active device)
            device_id = self.app_state.loaded_device
            if device_id is not None:
                device_info = sd.query_devices(device_id)
                return device_info['name']
        except Exception as e:
            logger.debug(f"Error getting device name: {e}")

        # Fallback to default device
        try:
            default_device = sd.default.device[0]
            if default_device is not None:
                device_info = sd.query_devices(default_device)
                return device_info['name']
        except Exception as e:
            logger.debug(f"Error getting default device: {e}")

        return "Unknown Device"

    def render(self) -> Text:
        """Render the waveform display."""
        rms = self.app_state.current_rms
        threshold = self.app_state.rms_threshold
        is_recording = self.app_state.is_recording

        # Create the display text
        text = Text()

        # Show selected device
        device_name = self._get_device_name()
        text.append("🎤 Device: ", style="dim cyan")
        text.append(device_name, style="bold white")
        text.append("\n", style="")

        if not is_recording:
            text.append("Waveform: ", style="bold cyan")
            text.append("Not recording", style="dim")
            return text

        # Calculate bar length based on RMS (scale it for visibility)
        # Linear scale: typical RMS values range from 0.0001 (very quiet) to 0.1+ (loud)
        # We'll scale to show good range with 0.05 as roughly "full"
        max_rms_for_scale = 0.05  # Adjust this to change sensitivity

        # Calculate percentage (0-100)
        if rms > 0:
            scaled_rms = min(100, int((rms / max_rms_for_scale) * 100))
        else:
            scaled_rms = 0

        # Convert percentage to bar length
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
