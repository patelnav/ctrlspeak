"""
Main Textual application for ctrlSPEAK.
"""

import logging
from typing import Optional
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Header, Footer, Static, Label
from textual.reactive import reactive
from textual import on
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .state import AppState
import state
from .screens.recording import RecordingScreen
from .screens.device_selection import DeviceSelectionScreen
from .screens.help import HelpScreen
from .screens.model_selection import ModelSelectionScreen
from .screens.log_viewer import LogViewerScreen

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
        height: 1fr;
        width: 100%;
    }

    #recording-layout {
        height: 100%;
        width: 100%;
        border: none;
    }

    RecordingScreen {
        height: 100%;
        width: 100%;
        border: none;
    }

    /* Compact header with device and model info */
    .device-info-header {
        height: auto;
        border: solid $accent;
        padding: 1;
        margin-bottom: 1;
        width: 100%;
    }

    /* Main content area - accumulated text takes up most space */
    .accumulated-text-main {
        height: 1fr;
        width: 100%;
        margin-bottom: 1;
    }

    .recording-status {
        height: auto;
        padding: 1;
        margin-bottom: 1;
    }

    .help-text {
        color: $text-muted;
        text-align: center;
        padding: 1;
        margin: 0;
    }
    """

    BINDINGS = [
        Binding("d", "show_devices", "Devices", show=True),
        Binding("m", "show_models", "Models", show=True),
        Binding("l", "show_logs", "Logs", show=True),
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
        self.last_transcription_count = 0  # Track new transcriptions

        # Update interval for live data (in seconds)
        self.update_interval = 0.1

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header(show_clock=True)
        yield RecordingScreen(
            app_state=self.app_state,
            audio_manager=self.audio_manager,
            id="main-container"
        )
        yield Footer()

    def on_mount(self) -> None:
        """Called when app is mounted."""
        logger.info("CtrlSpeakApp mounted")

        # Set up periodic updates for live recording status
        self.set_interval(self.update_interval, self.update_recording_state)

    def update_recording_state(self) -> None:
        """Periodically update recording state from audio manager and sync transcribed chunks."""
        if self.audio_manager:
            self.app_state.update_from_audio_manager(self.audio_manager)

        # Sync new transcribed chunks into accumulated text
        if len(state.transcribed_chunks) > self.last_transcription_count:
            new_chunks = state.transcribed_chunks[self.last_transcription_count:]
            for chunk_text in new_chunks:
                if chunk_text:
                    # Add space between chunks if text already exists
                    if self.app_state.accumulated_text.strip():
                        self.app_state.accumulated_text += " " + chunk_text.strip()
                    else:
                        self.app_state.accumulated_text = chunk_text.strip()
            self.last_transcription_count = len(state.transcribed_chunks)

        # Notify the recording screen to refresh
        self.refresh()

    async def action_show_devices(self) -> None:
        """Show device selection screen."""
        logger.info("Device selection requested")
        await self.push_screen(DeviceSelectionScreen(
            app_state=self.app_state,
            audio_manager=self.audio_manager
        ))

    async def action_show_models(self) -> None:
        """Show model selection screen."""
        logger.info("Model selection requested")
        await self.push_screen(ModelSelectionScreen(app_state=self.app_state))

    async def action_show_logs(self) -> None:
        """Show log viewer screen."""
        logger.info("Log viewer requested")
        await self.push_screen(LogViewerScreen(app_state=self.app_state))

    async def action_show_help(self) -> None:
        """Show help screen."""
        logger.info("Help screen requested")
        await self.push_screen(HelpScreen(app_state=self.app_state))

    async def action_quit(self) -> None:
        """Quit the application."""
        logger.info("Quit requested")
        self.exit()
