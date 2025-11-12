"""
History screen for ctrlSPEAK.

Browse, view, copy, and delete past transcriptions.
"""

import logging
from textual.screen import Screen
from textual.containers import Container, Vertical
from textual.widgets import Static, Label, ListItem, ListView
from textual.app import ComposeResult
from textual.binding import Binding
from textual import on

from ..state import AppState
from utils.history import get_history_manager, HistoryEntry
from utils.clipboard import copy_to_clipboard

logger = logging.getLogger("ctrlspeak.ui.history")


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
        preview = entry.preview[:60] + "..." if len(entry.preview) > 60 else entry.preview
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
    ListView {
        height: 1fr;
        border: solid $primary;
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
                yield Label("[yellow]No transcription history yet.[/yellow]", classes="help-text")
                yield Label("Start recording to create history entries!", classes="help-text")
                yield Label("Press Esc to go back", classes="help-text")
                return

            # Show statistics
            stats = self.history_manager.get_stats()
            stats_text = (
                f"Total: {stats['total_entries']} entries | "
                f"~{stats['total_words']:,} words | "
                f"~{stats['total_duration']/60:.1f} minutes recorded"
            )
            yield Label(stats_text, classes="stats-text")

            # Create interactive list view
            history_list = ListView(
                *[HistoryListItem(entry) for entry in self.entries],
                id="history-list"
            )
            yield history_list

            yield Label("↑↓ Navigate • Enter/C to Copy • Delete to Remove • Esc to Go Back", classes="help-text")

    def action_dismiss(self) -> None:
        """Dismiss the screen and go back."""
        self.dismiss()

    def action_copy_selected(self) -> None:
        """Copy the selected entry to clipboard."""
        history_list = self.query_one("#history-list", ListView)
        selected_index = history_list.index

        if selected_index is None or selected_index < 0 or selected_index >= len(self.entries):
            self.app.notify("No entry selected", severity="warning")
            return

        entry = self.entries[selected_index]
        try:
            copy_to_clipboard(entry.text)
            self.app.notify(f"Copied to clipboard ({len(entry.text)} chars)", severity="information")
            logger.info(f"Copied history entry {entry.id} to clipboard")
        except Exception as e:
            logger.error(f"Failed to copy to clipboard: {e}")
            self.app.notify("Failed to copy to clipboard", severity="error")

    def action_delete_selected(self) -> None:
        """Delete the selected entry."""
        history_list = self.query_one("#history-list", ListView)
        selected_index = history_list.index

        if selected_index is None or selected_index < 0 or selected_index >= len(self.entries):
            self.app.notify("No entry selected", severity="warning")
            return

        entry = self.entries[selected_index]

        # Delete from database
        if self.history_manager.delete_entry(entry.id):
            self.app.notify(f"Deleted entry from {entry.formatted_timestamp}", severity="information")
            logger.info(f"Deleted history entry {entry.id}")

            # Refresh the screen by re-mounting
            self.refresh_entries()
        else:
            self.app.notify("Failed to delete entry", severity="error")

    def refresh_entries(self) -> None:
        """Refresh the history list after changes."""
        # Get updated entries
        self.entries = self.history_manager.get_recent(limit=100)

        # Get the ListView and update it
        try:
            history_list = self.query_one("#history-list", ListView)
            history_list.clear()

            if self.entries:
                for entry in self.entries:
                    history_list.append(HistoryListItem(entry))
            else:
                # If no more entries, just notify
                self.app.notify("History is now empty", severity="information")
                self.dismiss()

        except Exception as e:
            logger.error(f"Error refreshing history list: {e}")

    @on(ListView.Selected)
    async def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle entry selection - copy to clipboard."""
        history_list: ListView = event.control
        selected_index = history_list.index

        if selected_index is None or selected_index < 0 or selected_index >= len(self.entries):
            logger.warning(f"Invalid history index: {selected_index}")
            return

        entry = self.entries[selected_index]

        # Copy to clipboard
        try:
            copy_to_clipboard(entry.text)
            self.app.notify(f"Copied to clipboard ({len(entry.text)} chars)", severity="information")
            logger.info(f"Copied history entry {entry.id} to clipboard")
        except Exception as e:
            logger.error(f"Failed to copy to clipboard: {e}")
            self.app.notify("Failed to copy to clipboard", severity="error")

    def on_mount(self) -> None:
        """Called when screen is mounted."""
        logger.info("HistoryScreen mounted")
