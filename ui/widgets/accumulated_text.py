"""
Widget to display accumulated transcription text as it builds up.
"""

import logging
from textual.widgets import Static
from rich.panel import Panel
from rich.text import Text

from ..state import AppState

logger = logging.getLogger("ctrlspeak.ui.accumulated_text")


class AccumulatedTextWidget(Static):
    """
    Displays the accumulated transcription text buffer.

    Shows text as it accumulates from transcribed segments.
    When user triple-taps Ctrl again, this buffer will be pasted.
    """

    def __init__(self, app_state: AppState, **kwargs):
        """
        Initialize the accumulated text widget.

        Args:
            app_state: Application state instance
        """
        super().__init__(**kwargs)
        self.app_state = app_state

    def render(self):
        """Render the accumulated text in a panel."""
        # Get accumulated text
        text_content = self.app_state.accumulated_text if self.app_state else ""

        logger.debug(f"AccumulatedTextWidget.render() - content length: {len(text_content) if text_content else 0}")

        # If no text yet, show placeholder
        if not text_content or not text_content.strip():
            content = Text(
                "Transcribed text will appear here as segments are captured...",
                style="dim cyan"
            )
        else:
            # Return styled text
            content = Text(text_content, style="white")

        # Wrap in a panel for visibility
        return Panel(
            content,
            title="📝 Accumulated Text",
            border_style="blue",
            expand=True,
            padding=(1, 2)
        )

    def on_mount(self) -> None:
        """Called when widget is mounted."""
        logger.debug("AccumulatedTextWidget mounted")
        # Refresh frequently to show new text immediately
        self.set_interval(0.2, self.refresh)
