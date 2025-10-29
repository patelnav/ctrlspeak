"""
Device selection screen for ctrlSPEAK.
"""

import logging
import sounddevice as sd
from textual.screen import Screen
from textual.containers import Container, Vertical
from textual.widgets import Static, Label, ListItem, ListView
from textual.app import ComposeResult
from textual.binding import Binding
from textual import on

from ..state import AppState, DeviceInfo

logger = logging.getLogger("ctrlspeak.ui.device_selection")


class DeviceListItem(ListItem):
    """Custom list item for audio devices."""

    def __init__(self, device: DeviceInfo, is_current: bool = False, **kwargs):
        """Initialize with device info."""
        self.device = device
        device_text = f"{device.name} (Device #{device.id})"
        device_specs = f"{device.channels}ch @ {device.sample_rate/1000:.1f}kHz"

        # Show current selection
        if is_current:
            device_text += " [green][CURRENT][/green]"
        elif device.is_default:
            device_text += " [dim][DEFAULT][/dim]"

        label_text = f"{device_text} - {device_specs}"
        super().__init__(Label(label_text), id=f"device-{device.id}", **kwargs)


class DeviceSelectionScreen(Screen):
    """
    Device selection screen for choosing audio input device.

    Shows list of available devices with their capabilities.
    """

    CSS = """
    ListView {
        height: 1fr;
        border: solid $primary;
    }

    ListItem:focus {
        background: $accent;
        color: $text;
    }
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
            yield Label("🎤 Audio Input Devices", classes="screen-title")
            yield Label("Select an audio input device:", classes="help-text")

            # Get available devices
            self.devices = self.get_available_devices()

            if not self.devices:
                yield Label("[red]No audio input devices found![/red]", classes="help-text")
                yield Label("Press Esc to go back", classes="help-text")
                return

            # Create interactive list view with current selection marked
            device_list = ListView(
                *[
                    DeviceListItem(
                        device,
                        is_current=(device.id == self.app_state.selected_device)
                    )
                    for device in self.devices
                ],
                id="device-list"
            )
            yield device_list

            yield Label("↑↓ Navigate • Enter to Select • Esc to Go Back", classes="help-text")

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

    @on(ListView.Selected)
    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle device selection from ListView."""
        device_list: ListView = event.control

        # Get the index of the selected item
        selected_index = device_list.index

        # Check if index is valid (0 is a valid index!)
        if selected_index is None or selected_index < 0 or selected_index >= len(self.devices):
            logger.warning(f"Invalid device index: {selected_index}")
            return

        selected_device = self.devices[selected_index]
        logger.info(f"Selected device: {selected_device.name} (ID: {selected_device.id})")

        # Update app state
        self.app_state.selected_device = selected_device.id

        # Update audio manager if available
        if self.audio_manager:
            try:
                self.audio_manager.set_input_device(selected_device.id)
                logger.info(f"Audio device set to ID {selected_device.id}")
            except Exception as e:
                logger.error(f"Error setting input device: {e}")

        # Dismiss the screen
        self.dismiss()

    def on_mount(self) -> None:
        """Called when screen is mounted."""
        logger.info("DeviceSelectionScreen mounted")
