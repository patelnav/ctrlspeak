"""
Main Textual application for ctrlSPEAK.
"""

import logging
import asyncio
import threading
import gc
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
from .screens.model_loading import ModelLoadingScreen

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

        # Lock to prevent concurrent model swaps
        self.model_swap_lock = threading.Lock()

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

        # Detect if transcribed_chunks was cleared (new recording started)
        if len(state.transcribed_chunks) < self.last_transcription_count:
            logger.debug(f"Detected chunks cleared, resetting last_transcription_count from {self.last_transcription_count} to 0")
            self.last_transcription_count = 0

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

    async def hot_swap_model(self, new_model_alias: str) -> bool:
        """
        Hot-swap to a new model without restarting the application.

        Steps:
        1. Show loading screen
        2. Resolve model alias to full name
        3. Update state.model_type
        4. Unload old model (free memory)
        5. Load new model in background thread
        6. Update state.stt_model and app_state.loaded_model
        7. Dismiss loading dialog

        Args:
            new_model_alias: Model alias to swap to (e.g., "parakeet-v3")

        Returns:
            True if swap was successful, False otherwise
        """
        from models.factory import ModelFactory
        from model_loader import get_model

        # Acquire lock to prevent concurrent swaps
        if not self.model_swap_lock.acquire(blocking=False):
            logger.warning("Model swap already in progress")
            self.notify("Model swap already in progress", severity="warning")
            return False

        loading_screen = None

        try:
            # Mark as loading
            self.app_state.is_loading_model = True
            self.app_state.model_load_progress = "Initializing..."

            # Show loading screen
            loading_screen = ModelLoadingScreen(new_model_alias)
            await self.push_screen(loading_screen)

            # Update progress: "Resolving model..."
            loading_screen.update_status("Resolving model name...")
            await asyncio.sleep(0.1)  # Give UI time to update

            try:
                full_model_name = ModelFactory.resolve_model_alias(new_model_alias)
                logger.info(f"Resolved {new_model_alias} to {full_model_name}")
            except Exception as e:
                logger.error(f"Error resolving model alias: {e}")
                loading_screen.update_status(f"Error: {e}", error=True)
                await asyncio.sleep(2)
                self.notify(f"Failed to resolve model: {e}", severity="error")
                return False

            # Update global state
            old_model_type = state.model_type
            state.model_type = full_model_name
            logger.info(f"Updated state.model_type from {old_model_type} to {full_model_name}")

            # Update progress: "Unloading old model..."
            loading_screen.update_status("Unloading previous model...")
            self.app_state.model_load_progress = "Unloading previous model..."
            await asyncio.sleep(0.1)

            # Unload old model
            old_model = state.stt_model
            state.stt_model = None
            state.model_loaded = False
            del old_model
            gc.collect()
            logger.info("Old model unloaded and memory freed")

            # Update progress: "Loading new model..."
            loading_screen.update_status(f"Loading {full_model_name}...")
            self.app_state.model_load_progress = f"Loading {full_model_name}..."
            await asyncio.sleep(0.1)

            # Load in background thread to avoid blocking UI
            error_message = None
            def load_model_thread():
                nonlocal error_message
                try:
                    logger.info(f"Loading model in background thread: {full_model_name}")
                    new_model = get_model()  # Uses state.model_type
                    return new_model
                except Exception as e:
                    logger.error(f"Error loading model in thread: {e}", exc_info=True)
                    error_message = str(e)
                    return None

            # Run in executor
            loop = asyncio.get_event_loop()
            new_model = await loop.run_in_executor(None, load_model_thread)

            if not new_model:
                # Show the actual error message to the user
                error_display = error_message or "Unknown error"
                loading_screen.update_status(f"Failed: {error_display}", error=True)
                self.app_state.model_load_progress = f"Failed: {error_display}"
                await asyncio.sleep(3)  # Give more time to read error
                self.notify(f"Model loading failed: {error_display}", severity="error")

                # Restore old model type
                state.model_type = old_model_type
                logger.error("Model loading failed, state.model_type restored")

                return False

            # Success!
            state.stt_model = new_model
            state.model_loaded = True
            self.app_state.loaded_model = new_model_alias
            logger.info(f"Model swap successful: {new_model_alias} is now loaded")

            loading_screen.update_status("Model loaded successfully!")
            self.app_state.model_load_progress = "Model loaded successfully!"
            await asyncio.sleep(1)

            self.notify(f"Switched to {new_model_alias}", severity="information")
            return True

        except Exception as e:
            logger.error(f"Hot swap failed: {e}", exc_info=True)
            self.notify(f"Failed to switch model: {e}", severity="error")
            return False

        finally:
            # Dismiss loading screen
            if loading_screen and loading_screen in self.screen_stack:
                loading_screen.dismiss()

            # Mark loading as complete
            self.app_state.is_loading_model = False
            self.app_state.model_load_progress = ""

            # Release lock
            self.model_swap_lock.release()
            logger.info("Model swap lock released")

    async def hot_swap_device(self, new_device_id: int) -> bool:
        """
        Hot-swap to a new audio input device without restarting the application.

        Steps:
        1. Validate device exists
        2. Stop current audio stream
        3. Start new stream with new device
        4. Update app_state.loaded_device

        Args:
            new_device_id: Device ID to swap to

        Returns:
            True if swap was successful, False otherwise
        """
        logger.info(f"Hot swapping to device {new_device_id}...")

        try:
            # Validate device exists
            try:
                import sounddevice as sd
                device_info = sd.query_devices(new_device_id)
                if device_info['max_input_channels'] <= 0:
                    raise ValueError(f"Device {new_device_id} has no input channels")
                logger.info(f"Target device: {device_info['name']}")
            except Exception as e:
                logger.error(f"Invalid device {new_device_id}: {e}")
                self.notify(f"Invalid device: {e}", severity="error")
                return False

            # Restart the audio stream with the new device
            try:
                logger.info("Restarting audio stream...")
                self.audio_manager.restart_input_stream(new_device_id)
                logger.info("Audio stream restarted successfully")
            except Exception as e:
                logger.error(f"Failed to restart audio stream: {e}")
                self.notify(f"Failed to switch device: {e}", severity="error")
                return False

            # Update app state
            self.app_state.loaded_device = new_device_id
            logger.info(f"Device swap successful: now using device {new_device_id}")

            self.notify(f"Switched to {device_info['name']}", severity="information")
            return True

        except Exception as e:
            logger.error(f"Hot device swap failed: {e}", exc_info=True)
            self.notify(f"Failed to switch device: {e}", severity="error")
            return False
