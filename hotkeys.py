"""
Hotkey activation handler for ctrlSPEAK.

Handles triple-tap Ctrl activation and routes to either:
- Streaming mode: For models that support real-time streaming (e.g., Nemotron)
- Queue mode: For models that use batch transcription (e.g., Parakeet, Canary)
"""

import logging
import time
import state
from utils.clipboard import copy_to_clipboard, paste_from_clipboard
from utils.player import play_start_beep, play_stop_beep
from utils.history import get_history_manager
import streaming

logger = logging.getLogger("ctrlspeak.hotkeys")

# Track which mode we're in for the current recording session
_current_session_streaming = False

# Track recording start time for duration calculation
_recording_start_time = None


def _start_queue_recording():
    """Start recording in queue-based mode (original behavior)."""
    logger.info("Starting queue-based recording session...")

    state.transcribed_chunks.clear()
    logger.info("Cleared previous transcribed chunks.")

    # Reset accumulated text for UI
    if hasattr(state, 'app_state_ref') and state.app_state_ref:
        state.app_state_ref.accumulated_text = ""

    state.audio_manager.start_recording()


def _stop_queue_recording():
    """Stop queue-based recording and wait for transcription."""
    logger.info("Stopping queue-based recording session...")

    state.audio_manager.stop_recording()

    logger.info("Waiting for transcription worker to finish processing queue...")
    state.transcription_queue.join()
    logger.info("Transcription queue processed.")

    final_text = " ".join(state.transcribed_chunks).strip()
    return final_text


def on_activate():
    """Handle global hotkey activation.

    Routes to streaming or queue-based mode depending on model capabilities.
    """
    global _current_session_streaming, _recording_start_time

    if not state.audio_manager.is_collecting:
        # =================================================================
        # START RECORDING
        # =================================================================

        # Check if model is being swapped
        if hasattr(state, 'app_state_ref') and state.app_state_ref:
            if state.app_state_ref.is_loading_model:
                logger.warning("Cannot record while model is loading")
                state.console.print("[yellow]Please wait for model to finish loading...[/yellow]")
                return

        if not state.model_loaded:
            state.console.print("[bold yellow]Model is still loading. Please wait...[/bold yellow]")
            return

        # Play start beep
        play_start_beep()

        # Track recording start time for history
        _recording_start_time = time.time()

        # Determine if we should use streaming mode
        if streaming.is_model_streaming_capable():
            logger.info("Using STREAMING mode (model supports streaming)")
            _current_session_streaming = True
            streaming.start_streaming()
        else:
            logger.info("Using QUEUE mode (batch transcription)")
            _current_session_streaming = False
            _start_queue_recording()

    else:
        # =================================================================
        # STOP RECORDING
        # =================================================================

        logger.info("Stop activated. Stopping audio recording...")
        play_stop_beep()

        # Use the mode we started with
        if _current_session_streaming:
            final_text = streaming.stop_streaming()
        else:
            final_text = _stop_queue_recording()

        # Handle final text
        if final_text:
            # Calculate recording duration
            duration_seconds = 0.0
            if _recording_start_time:
                duration_seconds = time.time() - _recording_start_time

            logger.info(f"Final text ({len(final_text)} chars): {final_text[:100]}...")
            copy_to_clipboard(final_text)
            paste_from_clipboard()

            state.console.print("\n[bold cyan]Transcription:[/bold cyan]")
            state.console.print(final_text)

            # Save to history (if enabled)
            if state.history_enabled:
                try:
                    history = get_history_manager()
                    if state.history_db_path:
                        history = get_history_manager(state.history_db_path)
                    history.add_entry(
                        text=final_text,
                        model=state.model_type,
                        duration_seconds=duration_seconds,
                        language=state.source_lang
                    )
                except Exception as e:
                    logger.error(f"Failed to save to history: {e}", exc_info=True)
        else:
            state.console.print("[yellow]No transcription result[/yellow]")

        # Reset session state
        _current_session_streaming = False
        _recording_start_time = None
