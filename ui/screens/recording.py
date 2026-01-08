"""
Recording screen for ctrlSPEAK Textual UI.
"""

import logging
from textual.screen import Screen
from textual.containers import Container, Vertical, Horizontal, ScrollableContainer
from textual.widgets import Static, Label
from textual.app import ComposeResult

from ..state import AppState
from ..widgets.waveform import WaveformDisplay
from ..widgets.device_info import DeviceInfoWidget
from ..widgets.status_bar import RecordingStatusWidget
from ..widgets.accumulated_text import AccumulatedTextWidget

logger = logging.getLogger("ctrlspeak.ui.recording")


class RecordingScreen(Static):
    """
    Main recording screen showing:
    - Device info
    - Waveform display
    - Recording status
    - Hotkey hint
    """

    def __init__(
        self,
        app_state: AppState,
        audio_manager=None,
        **kwargs
    ):
        """
        Initialize the recording screen.

        Args:
            app_state: Application state instance
            audio_manager: AudioManager for recording control
        """
        super().__init__(**kwargs)
        self.app_state = app_state
        self.audio_manager = audio_manager

    def compose(self) -> ComposeResult:
        """Create child widgets."""
        with Vertical(id="recording-layout"):
            # Top section: Device/model info (compact header)
            yield DeviceInfoWidget(
                app_state=self.app_state,
                audio_manager=self.audio_manager,
                classes="device-info-header"
            )

            # Waveform display (VAD probability)
            yield WaveformDisplay(
                app_state=self.app_state
            )

            # Main content: Accumulated transcription text (LARGE, scrollable)
            yield AccumulatedTextWidget(
                app_state=self.app_state,
                classes="accumulated-text-main"
            )

            # Recording status (compact)
            yield RecordingStatusWidget(
                app_state=self.app_state,
                classes="recording-status"
            )

            # Help text
            yield Label(
                "Triple-tap Ctrl to start/stop recording",
                classes="help-text"
            )

    def on_mount(self) -> None:
        """Called when screen is mounted."""
        logger.info("RecordingScreen mounted")
        # Set up periodic refresh for live updates
        self.set_interval(0.1, self.refresh_display)

    def refresh_display(self) -> None:
        """Refresh display with current state."""
        # The individual widgets will handle their own updates
        self.refresh()
