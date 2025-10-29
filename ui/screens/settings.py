"""
Settings screen for ctrlSPEAK.
"""

import logging
from textual.screen import Screen
from textual.containers import Container, Vertical, Horizontal, Grid
from textual.widgets import Static, Label, Input, Button
from textual.app import ComposeResult
from textual.binding import Binding

from ..state import AppState

logger = logging.getLogger("ctrlspeak.ui.settings")


class SettingsScreen(Screen):
    """
    Settings screen for adjusting ctrlSPEAK parameters.

    Allows tuning:
    - RMS threshold for speech detection
    - Silence duration for segmentation
    - Minimum chunk duration
    """

    BINDINGS = [
        Binding("escape", "dismiss", "Back", show=True),
        Binding("q", "dismiss", "Back", show=False),
    ]

    CSS = """
    SettingsScreen {
        align: center middle;
    }

    #settings-container {
        width: 80;
        height: auto;
        border: solid $primary;
        padding: 2;
    }

    .setting-row {
        height: 3;
        margin: 1 0;
    }

    .setting-label {
        width: 30;
        content-align: left middle;
    }

    .setting-value {
        width: 20;
    }

    .setting-help {
        color: $text-muted;
        text-style: italic;
    }
    """

    def __init__(self, app_state: AppState, audio_manager=None, **kwargs):
        """
        Initialize the settings screen.

        Args:
            app_state: Application state
            audio_manager: AudioManager instance
        """
        super().__init__(**kwargs)
        self.app_state = app_state
        self.audio_manager = audio_manager

    def compose(self) -> ComposeResult:
        """Create child widgets."""
        with Container(id="settings-container"):
            yield Label("Settings", classes="screen-title")

            # RMS Threshold
            with Horizontal(classes="setting-row"):
                yield Label("RMS Threshold:", classes="setting-label")
                yield Label(f"{self.app_state.rms_threshold:.4f}", classes="setting-value", id="rms-value")

            yield Label(
                "Speech detection threshold (0.0001 - 0.1)",
                classes="setting-help"
            )

            # Silence Duration
            with Horizontal(classes="setting-row"):
                yield Label("Silence Duration:", classes="setting-label")
                yield Label(f"{self.app_state.silence_duration_s:.1f}s", classes="setting-value", id="silence-value")

            yield Label(
                "Time of silence before segmenting (0.5s - 5.0s)",
                classes="setting-help"
            )

            # Minimum Chunk Duration
            with Horizontal(classes="setting-row"):
                yield Label("Min Chunk Duration:", classes="setting-label")
                yield Label(f"{self.app_state.min_chunk_duration_s:.1f}s", classes="setting-value", id="chunk-value")

            yield Label(
                "Minimum recording length to transcribe (0.1s - 2.0s)",
                classes="setting-help"
            )

            # Model
            with Horizontal(classes="setting-row"):
                yield Label("Model:", classes="setting-label")
                yield Label(f"{self.app_state.selected_model}", classes="setting-value", id="model-value")

            yield Label(
                "Speech recognition model (set with --model flag)",
                classes="setting-help"
            )

            yield Label("\n\nPress Esc to go back", classes="help-text")
            yield Label(
                "Note: Settings editing UI will be fully interactive in a future update",
                classes="setting-help"
            )

    def action_dismiss(self) -> None:
        """Dismiss the screen and go back."""
        self.dismiss()

    def on_mount(self) -> None:
        """Called when screen is mounted."""
        logger.info("SettingsScreen mounted")
        # Set up periodic refresh to show current values
        self.set_interval(1.0, self.refresh_values)

    def refresh_values(self) -> None:
        """Refresh displayed values from app state."""
        try:
            if self.app_state:
                self.query_one("#rms-value", Label).update(f"{self.app_state.rms_threshold:.4f}")
                self.query_one("#silence-value", Label).update(f"{self.app_state.silence_duration_s:.1f}s")
                self.query_one("#chunk-value", Label).update(f"{self.app_state.min_chunk_duration_s:.1f}s")
                self.query_one("#model-value", Label).update(f"{self.app_state.selected_model}")
        except Exception as e:
            logger.debug(f"Error refreshing settings values: {e}")
