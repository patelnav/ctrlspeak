"""
Device info widget for ctrlSPEAK.
"""

import logging
import sounddevice as sd
from textual.widgets import Static
from rich.text import Text

from ..state import AppState
from models.factory import ModelFactory

logger = logging.getLogger("ctrlspeak.ui.device_info")


class DeviceInfoWidget(Static):
    """
    Displays information about the current audio input device.

    Shows:
    - Device name and ID
    - Number of channels
    - Sample rate
    """

    def __init__(self, app_state: AppState, audio_manager=None, **kwargs):
        """
        Initialize the device info widget.

        Args:
            app_state: Application state
            audio_manager: AudioManager instance for device info
        """
        super().__init__(**kwargs)
        self.app_state = app_state
        self.audio_manager = audio_manager

    def get_device_info(self) -> tuple:
        """
        Get current device information.

        Returns:
            Tuple of (device_id, device_name, channels, sample_rate)
        """
        try:
            # Use selected device from app_state if available, otherwise use system default
            device_id = self.app_state.selected_device if self.app_state.selected_device is not None else (sd.default.device[0] if sd.default.device else None)

            if device_id is not None:
                device_info = sd.query_devices(device_id)
                if device_info:
                    return (
                        device_id,
                        device_info['name'],
                        device_info['max_input_channels'],
                        int(device_info['default_samplerate'])
                    )
        except Exception as e:
            logger.error(f"Error getting device info: {e}")

        return (None, "Unknown Device", 1, 16000)

    def render(self) -> Text:
        """Render the device and model info display."""
        device_id, device_name, channels, sample_rate = self.get_device_info()

        text = Text()

        # Device info
        text.append("📻 Device: ", style="bold cyan")
        if device_id is not None:
            text.append(device_name, style="bold white")
            text.append(f" (#{device_id})", style="dim")
        else:
            text.append(device_name, style="yellow")

        text.append(" | ", style="dim")
        text.append(f"{channels}ch @ {sample_rate/1000:.1f}kHz", style="cyan")

        # Model info
        text.append(" | ", style="dim")
        text.append("🤖 Model: ", style="bold green")

        model_alias = self.app_state.selected_model
        model_full_name = ModelFactory._DEFAULT_ALIASES.get(model_alias, model_alias)
        text.append(f"{model_full_name}", style="bold white")

        return text

    def on_mount(self) -> None:
        """Called when widget is mounted."""
        logger.debug("DeviceInfoWidget mounted")
        # Refresh periodically in case device changes
        self.set_interval(2.0, self.refresh)
