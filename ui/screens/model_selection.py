"""
Model selection screen for ctrlSPEAK.
"""

import logging
from textual.screen import Screen
from textual.containers import Container
from textual.widgets import Static, Label, ListItem, ListView
from textual.app import ComposeResult
from textual.binding import Binding
from textual import on

from ..state import AppState
from models.factory import ModelFactory

logger = logging.getLogger("ctrlspeak.ui.model_selection")


class ModelListItem(ListItem):
    """Custom list item for models."""

    def __init__(self, model_alias: str, model_full_name: str, current: bool = False, **kwargs):
        """Initialize with model name."""
        self.model_alias = model_alias
        self.model_full_name = model_full_name

        # Show alias and full name
        model_text = f"[cyan]{model_alias}[/cyan] → {model_full_name}"

        if current:
            model_text += " [green][CURRENT][/green]"

        super().__init__(Label(model_text), id=f"model-{model_alias}", **kwargs)


class ModelSelectionScreen(Screen):
    """
    Model selection screen for choosing the speech-to-text model.

    Shows list of available models. Selected model will be used for
    the next session.
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

    def __init__(self, app_state: AppState, **kwargs):
        """
        Initialize the model selection screen.

        Args:
            app_state: Application state
        """
        super().__init__(**kwargs)
        self.app_state = app_state

    def compose(self) -> ComposeResult:
        """Create child widgets."""
        with Container():
            yield Label("🤖 Speech-to-Text Models", classes="screen-title")
            yield Label("Select a model for the next session:", classes="help-text")

            # Create interactive list view with full model names
            model_list = ListView(
                *[
                    ModelListItem(
                        model_alias,
                        ModelFactory._DEFAULT_ALIASES.get(model_alias, model_alias),
                        current=(model_alias == self.app_state.selected_model)
                    )
                    for model_alias in self.app_state.available_models
                ],
                id="model-list"
            )
            yield model_list

            yield Label("↑↓ Navigate • Enter to Select • Esc to Go Back", classes="help-text")
            yield Label("[yellow]Note: Selected model will be used when the app restarts[/yellow]", classes="help-text")

    def action_dismiss(self) -> None:
        """Dismiss the screen and go back."""
        self.dismiss()

    @on(ListView.Selected)
    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle model selection from ListView."""
        model_list: ListView = event.control

        # Get the index of the selected item
        selected_index = model_list.index

        # Check if index is valid
        if selected_index is None or selected_index < 0 or selected_index >= len(self.app_state.available_models):
            logger.warning(f"Invalid model index: {selected_index}")
            return

        selected_model = self.app_state.available_models[selected_index]
        logger.info(f"Selected model: {selected_model}")

        # Update app state
        self.app_state.selected_model = selected_model

        # Save preference to config
        try:
            from utils.config import set_preferred_model
            set_preferred_model(selected_model)
            logger.info(f"Model preference saved: {selected_model}")
        except Exception as e:
            logger.error(f"Error saving model preference: {e}")

        # Dismiss the screen
        self.dismiss()

    def on_mount(self) -> None:
        """Called when screen is mounted."""
        logger.info("ModelSelectionScreen mounted")
