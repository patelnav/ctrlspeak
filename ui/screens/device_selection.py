"""
Device selection screen for ctrlSPEAK.
"""

import logging
import sounddevice as sd
from textual.screen import Screen
from textual.containers import Container, Vertical, ScrollableContainer
from textual.widgets import Static, Label, ListItem, ListView
from textual.app import ComposeResult
from textual.binding import Binding

from ..state import AppState, DeviceInfo

logger = logging.getLogger("ctrlspeak.ui.device_selection")


class DeviceSelectionScreen(Screen):
    """
    Device selection screen for choosing audio input device.

    Shows list of available devices with their capabilities.
    """

    BINDINGS = [
        Binding("escape", "dismiss", "Back", show=True),
        Binding("q", "dismiss", "Back", show=False),
    ]

    def __init__(self, app_state: AppState, audio_manager=None, **kwargs):
        """
        Initialize the device selection screen.

        Args:
            app_state: Application state
            audio_manager: AudioManager instance
        """
        super().__init__(**kwargs)
        self.app_state = app_state
        self.audio_manager = audio_manager
        self.devices = []

    def compose(self) -> ComposeResult:
        """Create child widgets."""
        with Container():
            yield Label("Audio Input Devices", classes="screen-title")
            yield Label("Select an audio input device:", classes="help-text")

            # Get available devices
            self.devices = self.get_available_devices()

            with ScrollableContainer():
                for device in self.devices:
                    device_text = f"{device.name} (Device #{device.id})"
                    device_specs = f"{device.channels}ch @ {device.sample_rate/1000:.1f}kHz"

                    if device.is_default:
                        device_text += " [DEFAULT]"

                    yield Label(f"{device_text} - {device_specs}")

            yield Label("\nUse ↑↓ to navigate, Enter to select, Esc to go back", classes="help-text")

    def get_available_devices(self) -> list[DeviceInfo]:
        """
        Get list of available audio input devices.

        Returns:
            List of DeviceInfo objects
        """
        devices = []
        try:
            all_devices = sd.query_devices()
            default_device_id = sd.default.device[0] if sd.default.device else None

            for i, device in enumerate(all_devices):
                # Only include input devices (those with input channels)
                if device['max_input_channels'] > 0:
                    devices.append(DeviceInfo(
                        id=i,
                        name=device['name'],
                        channels=device['max_input_channels'],
                        sample_rate=int(device['default_samplerate']),
                        is_default=(i == default_device_id)
                    ))

            logger.info(f"Found {len(devices)} input devices")

        except Exception as e:
            logger.error(f"Error enumerating devices: {e}")

        return devices

    def action_dismiss(self) -> None:
        """Dismiss the screen and go back."""
        self.dismiss()

    def on_mount(self) -> None:
        """Called when screen is mounted."""
        logger.info("DeviceSelectionScreen mounted")
