"""
History screen for ctrlSPEAK.

Browse, view, copy, and delete past transcriptions.
"""

import logging

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal
from textual.screen import ModalScreen, Screen
from textual.widgets import Button, Label, ListItem, ListView

from utils.clipboard import copy_to_clipboard
from utils.history import HistoryEntry, get_history_manager

from ..state import AppState

logger = logging.getLogger("ctrlspeak.ui.history")


class DeleteConfirmDialog(ModalScreen):
    """Confirmation dialog for deleting history entry."""

    CSS = """
    DeleteConfirmDialog {
        align: center middle;
    }

    #dialog {
        width: 60;
        height: auto;
        border: thick $error;
        background: $surface;
        padding: 1 2;
    }

    #question {
        width: 100%;
        content-align: center middle;
        padding: 1;
    }

    #preview {
        width: 100%;
        content-align: center middle;
        padding: 0 1 1 1;
        color: $text-muted;
    }

    #buttons {
        width: 100%;
        height: auto;
        align: center middle;
        padding: 1 0 0 0;
    }

    Button {
        margin: 0 1;
        min-width: 16;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", show=False),
    ]

    def __init__(self, entry_preview: str, **kwargs):
        super().__init__(**kwargs)
        self.entry_preview = entry_preview

    def compose(self) -> ComposeResult:
        with Container(id="dialog"):
            yield Label("⚠️  Delete this transcription?", id="question")
            yield Label(f'"{self.entry_preview[:50]}..."', id="preview")
            with Horizontal(id="buttons"):
                yield Button("Cancel (Esc)", variant="default", id="cancel")
                yield Button("Delete", variant="error", id="confirm")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "confirm")

    def action_cancel(self) -> None:
        self.dismiss(False)


class HistoryListItem(ListItem):
    """Custom list item for history entries."""

    def __init__(self, entry: HistoryEntry, **kwargs):
        """
        Initialize with history entry.

        Args:
            entry: HistoryEntry object
        """
        self.entry = entry

        # Format: timestamp | preview (first 60 chars) | model | duration
        timestamp_str = entry.formatted_timestamp
        preview = (
            entry.preview[:60] + "..."
            if len(entry.preview) > 60
            else entry.preview
        )
        duration_str = f"{entry.duration_seconds:.1f}s"

        # Build the display text
        entry_text = f"[cyan]{timestamp_str}[/cyan] | [dim]{entry.model}[/dim] | [yellow]{duration_str}[/yellow]\n  {preview}"

        super().__init__(Label(entry_text), id=f"history-{entry.id}", **kwargs)


class HistoryScreen(Screen):
    """
    History screen for viewing past transcriptions.

    Shows list of recent transcriptions with ability to view details,
    copy to clipboard, and delete entries.
    """

    CSS = """
    HistoryScreen {
        padding: 0 1;
    }

    .screen-title {
        text-align: center;
        padding: 1;
        margin-bottom: 1;
        color: $accent;
    }

    ListView {
        height: 1fr;
        border: solid $primary;
        margin-bottom: 1;
    }

    ListItem {
        height: auto;
        padding: 1;
    }

    ListItem:focus {
        background: $accent;
        color: $text;
    }

    .stats-text {
        color: $text-muted;
        text-align: center;
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
        Binding("escape", "dismiss", "Back", show=True),
        Binding("q", "dismiss", "Back", show=False),
        Binding("c", "copy_selected", "Copy", show=True),
        Binding("delete", "delete_selected", "Delete", show=True),
        Binding("d", "delete_selected", "Delete", show=False),
    ]

    def __init__(self, app_state: AppState, **kwargs):
        """
        Initialize the history screen.

        Args:
            app_state: Application state
        """
        super().__init__(**kwargs)
        self.app_state = app_state
        self.entries = []
        self.history_manager = get_history_manager()

    def compose(self) -> ComposeResult:
        """Create child widgets."""
        with Container():
            yield Label("📜 Transcription History", classes="screen-title")

            # Get history entries
            self.entries = self.history_manager.get_recent(limit=100)

            if not self.entries:
                yield Label(
                    "[yellow]No transcription history yet.[/yellow]",
                    classes="help-text",
                )
                yield Label(
                    "Start recording to create history entries!",
                    classes="help-text",
                )
                yield Label("Press Esc to go back", classes="help-text")
                return

            # Show statistics
            stats = self.history_manager.get_stats()
            stats_text = (
                f"Total: {stats['total_entries']} entries | "
                f"~{stats['total_words']:,} words | "
                f"~{stats['total_duration'] / 60:.1f} minutes recorded"
            )
            yield Label(stats_text, classes="stats-text")

            # Create interactive list view
            history_list = ListView(
                *[HistoryListItem(entry) for entry in self.entries],
                id="history-list",
            )
            yield history_list

            yield Label(
                "↑↓ Navigate • Enter/c to Copy • d to Remove • Esc to Go Back",
                classes="help-text",
            )

    def action_dismiss(self) -> None:
        """Dismiss the screen and go back."""
        self.dismiss()

    def action_copy_selected(self) -> None:
        """Copy the selected entry to clipboard."""
        history_list = self.query_one("#history-list", ListView)
        selected_index = history_list.index

        if (
            selected_index is None
            or selected_index < 0
            or selected_index >= len(self.entries)
        ):
            self.app.notify("No entry selected", severity="warning")
            return

        entry = self.entries[selected_index]
        try:
            copy_to_clipboard(entry.text)
            self.app.notify(
                f"Copied to clipboard ({len(entry.text)} chars)",
                severity="information",
            )
            logger.info(f"Copied history entry {entry.id} to clipboard")
        except Exception as e:
            logger.error(f"Failed to copy to clipboard: {e}")
            self.app.notify("Failed to copy to clipboard", severity="error")

    def action_delete_selected(self) -> None:
        """Delete the selected entry after confirmation."""
        history_list = self.query_one("#history-list", ListView)
        selected_index = history_list.index

        if (
            selected_index is None
            or selected_index < 0
            or selected_index >= len(self.entries)
        ):
            self.app.notify("No entry selected", severity="warning")
            return

        entry = self.entries[selected_index]

        # Run async deletion in a worker
        self.run_worker(self._delete_with_confirmation(entry), exclusive=True)

    async def _delete_with_confirmation(self, entry: HistoryEntry) -> None:
        """Show confirmation dialog and delete entry if confirmed."""

        confirmed = await self.app.push_screen_wait(
            DeleteConfirmDialog(entry_preview=entry.preview)
        )

        if not confirmed:
            return

        # Delete from database
        if self.history_manager.delete_entry(entry.id):
            self.app.notify(
                f"Deleted entry from {entry.formatted_timestamp}",
                severity="information",
            )
            await self.refresh_entries()
        else:
            self.app.notify("Failed to delete entry", severity="error")

    async def refresh_entries(self) -> None:
        """Refresh the history list after changes."""
        # Get updated entries from database
        new_entries = self.history_manager.get_recent(limit=100)

        # Get the ListView
        try:
            history_list = self.query_one("#history-list", ListView)
        except Exception:
            return

        if not new_entries:
            # No more entries - close the screen
            self.app.notify("History is now empty", severity="information")
            self.dismiss()
            return

        # Update our local cache
        self.entries = new_entries

        # Clear the list first and wait for it to complete
        await history_list.clear()

        # Add all entries back using extend (more efficient than multiple appends)
        new_items = [HistoryListItem(entry) for entry in self.entries]
        await history_list.extend(new_items)

        # Select first item
        if len(self.entries) > 0:
            history_list.index = 0

    @on(ListView.Selected)
    async def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle entry selection - copy to clipboard."""
        history_list: ListView = event.control
        selected_index = history_list.index

        if (
            selected_index is None
            or selected_index < 0
            or selected_index >= len(self.entries)
        ):
            logger.warning(f"Invalid history index: {selected_index}")
            return

        entry = self.entries[selected_index]

        # Copy to clipboard
        try:
            copy_to_clipboard(entry.text)
            self.app.notify(
                f"Copied to clipboard ({len(entry.text)} chars)",
                severity="information",
            )
            logger.info(f"Copied history entry {entry.id} to clipboard")
        except Exception as e:
            logger.error(f"Failed to copy to clipboard: {e}")
            self.app.notify("Failed to copy to clipboard", severity="error")

    def on_mount(self) -> None:
        """Called when screen is mounted."""
        logger.info("HistoryScreen mounted")
