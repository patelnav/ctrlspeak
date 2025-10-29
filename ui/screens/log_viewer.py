"""
Log viewer screen for ctrlSPEAK.
"""

import logging
from pathlib import Path
from textual.screen import Screen
from textual.containers import Container, ScrollableContainer
from textual.widgets import Static, Label
from textual.app import ComposeResult
from textual.binding import Binding
from rich.text import Text
from rich.panel import Panel

from ..state import AppState

logger = logging.getLogger("ctrlspeak.ui.log_viewer")


class LogViewerScreen(Screen):
    """
    Displays recent logs from the log file.

    Shows the last N lines from ~/.config/ctrlspeak/logs/ctrlspeak.log
    """

    BINDINGS = [
        Binding("escape", "dismiss", "Back", show=True),
        Binding("q", "dismiss", "Back", show=False),
        Binding("r", "refresh_logs", "Refresh", show=True),
    ]

    def __init__(self, app_state: AppState, **kwargs):
        """
        Initialize the log viewer screen.

        Args:
            app_state: Application state
        """
        super().__init__(**kwargs)
        self.app_state = app_state
        self.log_content = Static(id="log-content")

    def compose(self) -> ComposeResult:
        """Create child widgets."""
        with Container():
            yield Label("📋 Application Logs", classes="screen-title")
            yield Label("Recent logs from ~/.config/ctrlspeak/logs/ctrlspeak.log", classes="help-text")

            with ScrollableContainer(id="log-container"):
                yield self.log_content

            yield Label("↑↓ Scroll • R to Refresh • Esc to Go Back", classes="help-text")

    def get_log_file_path(self) -> Path:
        """Get the path to the log file."""
        return Path.home() / ".config" / "ctrlspeak" / "logs" / "ctrlspeak.log"

    def load_logs(self, lines: int = 50) -> str:
        """
        Load recent logs from the log file.

        Args:
            lines: Number of recent lines to load

        Returns:
            Formatted log text
        """
        log_file = self.get_log_file_path()

        if not log_file.exists():
            return "[yellow]No log file found yet. Logs will appear as you use the application.[/yellow]"

        try:
            # Read all lines
            with open(log_file, 'r') as f:
                all_lines = f.readlines()

            # Get last N lines
            recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines

            if not recent_lines:
                return "[dim]Log file is empty[/dim]"

            # Format logs with styling
            formatted_lines = []
            for line in recent_lines:
                line = line.rstrip('\n')

                # Color code by log level
                if ' - ERROR - ' in line:
                    formatted_lines.append(f"[red]{line}[/red]")
                elif ' - WARNING - ' in line:
                    formatted_lines.append(f"[yellow]{line}[/yellow]")
                elif ' - INFO - ' in line:
                    formatted_lines.append(f"[cyan]{line}[/cyan]")
                elif ' - DEBUG - ' in line:
                    formatted_lines.append(f"[dim]{line}[/dim]")
                else:
                    formatted_lines.append(line)

            return "\n".join(formatted_lines)

        except Exception as e:
            logger.error(f"Error reading log file: {e}")
            return f"[red]Error reading log file: {e}[/red]"

    def render_logs(self) -> None:
        """Render the logs in the log content widget."""
        log_text = self.load_logs(lines=100)
        self.log_content.update(Panel(
            log_text,
            title="Recent Logs",
            border_style="blue",
            expand=True
        ))

    def action_dismiss(self) -> None:
        """Dismiss the screen and go back."""
        self.dismiss()

    def action_refresh_logs(self) -> None:
        """Refresh the log display."""
        logger.debug("Refreshing logs...")
        self.render_logs()

    def on_mount(self) -> None:
        """Called when screen is mounted."""
        logger.info("LogViewerScreen mounted")
        self.render_logs()
