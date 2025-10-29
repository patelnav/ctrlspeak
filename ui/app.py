"""
Main Textual application for ctrlSPEAK.
"""

import logging
from typing import Optional
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical, Horizontal
from textual.widgets import Header, Footer, Static, Label
from textual.reactive import reactive
from textual import on

from .state import AppState
from .screens.recording import RecordingScreen
from .screens.device_selection import DeviceSelectionScreen
from .screens.settings import SettingsScreen
from .screens.help import HelpScreen

logger = logging.getLogger("ctrlspeak.ui")


class CtrlSpeakApp(App):
    """
    Main Textual application for ctrlSPEAK.

    Provides an interactive TUI for speech-to-text with:
    - Real-time waveform visualization
    - Device selection
    - Settings management
    - Keyboard shortcuts
    """

    CSS = """
    Screen {
        background: $surface;
    }

    #main-container {
        height: 100%;
        width: 100%;
    }

    .app-title {
        background: $primary;
        color: $text;
        text-align: center;
        padding: 1;
        text-style: bold;
    }

    .status-bar {
        background: $panel;
        color: $text;
        padding: 0 1;
        height: 1;
    }

    .recording-indicator {
        color: $error;
        text-style: bold;
    }

    .waveform {
        height: 5;
        border: solid $primary;
        padding: 1;
    }

    .device-info {
        height: 3;
        border: solid $accent;
        padding: 1;
    }

    .recording-status {
        height: auto;
        padding: 1;
    }

    .help-text {
        color: $text-muted;
        text-align: center;
        padding: 1;
    }
    """

    BINDINGS = [
        Binding("d", "show_devices", "Devices", show=True),
        Binding("s", "show_settings", "Settings", show=True),
        Binding("h", "show_help", "Help", show=True),
        Binding("q", "quit", "Quit", show=True),
        Binding("ctrl+c", "quit", "Quit", show=False),
    ]

    TITLE = "ctrlSPEAK"
    SUB_TITLE = "Speech-to-Text Transcription"

    def __init__(
        self,
        app_state: Optional[AppState] = None,
        audio_manager=None,
        model_type: str = "parakeet",
        **kwargs
    ):
        """
        Initialize the Textual app.

        Args:
            app_state: Optional AppState instance (creates new if None)
            audio_manager: AudioManager instance for recording control
            model_type: Selected model type
        """
        super().__init__(**kwargs)
        self.app_state = app_state or AppState()
        self.audio_manager = audio_manager
        self.app_state.selected_model = model_type

        # Update interval for live data (in seconds)
        self.update_interval = 0.1

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header(show_clock=True)
        yield Container(
            RecordingScreen(app_state=self.app_state, audio_manager=self.audio_manager),
            id="main-container"
        )
        yield Footer()

    def on_mount(self) -> None:
        """Called when app is mounted."""
        logger.info("CtrlSpeakApp mounted")

        # Set up periodic updates for live recording status
        self.set_interval(self.update_interval, self.update_recording_state)

    def update_recording_state(self) -> None:
        """Periodically update recording state from audio manager."""
        if self.audio_manager:
            self.app_state.update_from_audio_manager(self.audio_manager)
            # Notify the recording screen to refresh
            self.refresh()

    async def action_show_devices(self) -> None:
        """Show device selection screen."""
        logger.info("Device selection requested")
        await self.push_screen(DeviceSelectionScreen(
            app_state=self.app_state,
            audio_manager=self.audio_manager
        ))

    async def action_show_settings(self) -> None:
        """Show settings screen."""
        logger.info("Settings screen requested")
        await self.push_screen(SettingsScreen(
            app_state=self.app_state,
            audio_manager=self.audio_manager
        ))

    async def action_show_help(self) -> None:
        """Show help screen."""
        logger.info("Help screen requested")
        await self.push_screen(HelpScreen(app_state=self.app_state))

    async def action_quit(self) -> None:
        """Quit the application."""
        logger.info("Quit requested")
        self.exit()
