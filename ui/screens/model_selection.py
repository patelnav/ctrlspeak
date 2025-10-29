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

    def __init__(self, model_alias: str, model_full_name: str,
                 is_loaded: bool = False, is_selected: bool = False, **kwargs):
        """
        Initialize with model name.

        Args:
            model_alias: Short alias for the model (e.g., "parakeet-v3")
            model_full_name: Full model name (e.g., "nvidia/parakeet-tdt-0.6b-v3")
            is_loaded: This model is currently loaded and running
            is_selected: This model is saved as preference for next launch
        """
        self.model_alias = model_alias
        self.model_full_name = model_full_name

        # Show alias and full name
        model_text = f"[cyan]{model_alias}[/cyan] → {model_full_name}"

        if is_loaded:
            model_text += " [green][LOADED][/green]"
        elif is_selected:
            model_text += " [dim][PREFERRED][/dim]"

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
                        is_loaded=(model_alias == self.app_state.loaded_model),  # Actually running
                        is_selected=(model_alias == self.app_state.selected_model)  # Saved preference
                    )
                    for model_alias in self.app_state.available_models
                ],
                id="model-list"
            )
            yield model_list

            yield Label("↑↓ Navigate • Enter to Select • Esc to Go Back", classes="help-text")
            yield Label("[green]Model will be loaded immediately (10-30 seconds)[/green]", classes="help-text")

    def action_dismiss(self) -> None:
        """Dismiss the screen and go back."""
        self.dismiss()

    @on(ListView.Selected)
    async def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle model selection from ListView - trigger hot swap."""
        model_list: ListView = event.control

        # Get the index of the selected item
        selected_index = model_list.index

        # Check if index is valid
        if selected_index is None or selected_index < 0 or selected_index >= len(self.app_state.available_models):
            logger.warning(f"Invalid model index: {selected_index}")
            return

        selected_model = self.app_state.available_models[selected_index]
        logger.info(f"Selected model: {selected_model}")

        # Check if it's already loaded
        if selected_model == self.app_state.loaded_model:
            self.app.notify("This model is already loaded", severity="information")
            self.dismiss()
            return

        # Prevent selection during recording
        if self.app_state.is_recording:
            self.app.notify("Cannot switch models while recording", severity="warning")
            return

        # Prevent selection if already loading a model
        if self.app_state.is_loading_model:
            self.app.notify("Model swap already in progress", severity="warning")
            return

        # Update app state preference
        self.app_state.selected_model = selected_model

        # Save preference to config for future launches
        try:
            from utils.config import set_preferred_model
            set_preferred_model(selected_model)
            logger.info(f"Model preference saved: {selected_model}")
        except Exception as e:
            logger.error(f"Error saving model preference: {e}")

        # Dismiss this screen before showing loading screen
        self.dismiss()

        # Trigger hot swap (runs in background)
        success = await self.app.hot_swap_model(selected_model)

        if success:
            logger.info(f"Hot swap to {selected_model} completed successfully")
        else:
            logger.error(f"Hot swap to {selected_model} failed")

    def on_mount(self) -> None:
        """Called when screen is mounted."""
        logger.info("ModelSelectionScreen mounted")
