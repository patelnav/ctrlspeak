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
    Displays a real-time waveform visualization based on Silero VAD.

    Shows:
    - Bar graph visualization of speech probability
    - Color coding: green (speech detected), gray (silence)
    - VAD probability percentage and RMS value
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
        vad_prob = self.app_state.current_vad_prob
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

        # Calculate bar length based on VAD probability (0-1)
        bar_length = int(vad_prob * self.bar_width)
        bar_length = max(0, min(bar_length, self.bar_width))

        # Determine if speech is detected (VAD threshold is 0.5)
        is_speech = vad_prob >= 0.5

        # Color coding
        bar_style = "bold green" if is_speech else "dim white"
        empty_style = "dim"

        # Build the bar
        text.append("VAD:   ", style="bold cyan")
        text.append("█" * bar_length, style=bar_style)
        text.append("░" * (self.bar_width - bar_length), style=empty_style)

        # Show VAD probability and RMS
        text.append(f" {vad_prob:.0%}", style="bold" if is_speech else "dim")
        text.append(f" (RMS: {rms:.4f})", style="dim")

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
