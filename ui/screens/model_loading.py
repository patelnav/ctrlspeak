"""
Model loading screen for ctrlSPEAK.
Shows progress while a new model is being loaded.
"""

import logging
from textual.screen import ModalScreen
from textual.containers import Container, Vertical
from textual.widgets import Static, Label
from textual.app import ComposeResult
from rich.text import Text

logger = logging.getLogger("ctrlspeak.ui.model_loading")


class ModelLoadingScreen(ModalScreen):
    """
    Modal dialog showing model loading progress.

    Cannot be dismissed until loading is complete or an error occurs.
    Shows:
    - Model name being loaded
    - Progress spinner
    - Status messages
    - Elapsed time
    """

    CSS = """
    ModelLoadingScreen {
        align: center middle;
    }

    #loading-dialog {
        width: 60;
        height: auto;
        border: thick $primary;
        background: $surface;
        padding: 2;
    }

    #loading-title {
        text-align: center;
        color: $accent;
        margin-bottom: 1;
    }

    #model-name {
        text-align: center;
        color: $text;
        margin-bottom: 2;
    }

    #status-message {
        text-align: center;
        color: $text-muted;
        margin-bottom: 1;
        height: 3;
    }

    #spinner {
        text-align: center;
        color: $accent;
        margin-bottom: 1;
    }
    """

    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the model loading screen.

        Args:
            model_name: Name of the model being loaded
        """
        super().__init__(**kwargs)
        self.model_name = model_name
        self.status_message = "Initializing..."
        self.spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.spinner_index = 0

    def compose(self) -> ComposeResult:
        """Create child widgets."""
        with Vertical(id="loading-dialog"):
            yield Label("🔄 Loading Model", id="loading-title")
            yield Label(self.model_name, id="model-name")
            yield Static("", id="spinner")
            yield Static(self.status_message, id="status-message")

    def on_mount(self) -> None:
        """Called when screen is mounted."""
        logger.info(f"ModelLoadingScreen mounted for {self.model_name}")
        # Update spinner animation every 100ms
        self.set_interval(0.1, self.update_spinner)

    def update_spinner(self) -> None:
        """Update the spinner animation."""
        spinner_widget = self.query_one("#spinner", Static)
        spinner_widget.update(self.spinner_frames[self.spinner_index])
        self.spinner_index = (self.spinner_index + 1) % len(self.spinner_frames)

    def update_status(self, message: str, error: bool = False) -> None:
        """
        Update the status message.

        Args:
            message: Status message to display
            error: Whether this is an error message
        """
        self.status_message = message
        status_widget = self.query_one("#status-message", Static)

        if error:
            text = Text(message, style="bold red")
        else:
            text = Text(message, style="cyan")

        status_widget.update(text)
        logger.debug(f"Status updated: {message} (error={error})")
