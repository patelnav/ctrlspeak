
import logging
import time
import state
from utils.clipboard import copy_to_clipboard, paste_from_clipboard
from utils.player import play_start_beep, play_stop_beep
from utils.history import get_history_manager

logger = logging.getLogger("ctrlspeak")

def on_activate():
    """Handle global hotkey activation"""
    if not state.audio_manager.is_collecting:
        # Check if model is being swapped
        if hasattr(state, 'app_state_ref') and state.app_state_ref:
            if state.app_state_ref.is_loading_model:
                logger.warning("Cannot record while model is loading")
                state.console.print("[yellow]Please wait for model to finish loading...[/yellow]")
                return

        if not state.model_loaded:
            state.console.print("[bold yellow]Model is still loading. Please wait...[/bold yellow]")
            return

        state.transcribed_chunks.clear()
        logger.info("Cleared previous transcribed chunks.")

        # Reset accumulated text for UI
        if hasattr(state, 'app_state_ref') and state.app_state_ref:
            state.app_state_ref.accumulated_text = ""
            logger.debug("Reset accumulated text for new recording.")

        logger.debug("Playing start beep...")
        play_start_beep()
        logger.debug("Calling audio_manager.start_recording()...")
        state.recording_start_time = time.time()
        state.audio_manager.start_recording()
    else:
        logger.info("Stop activated. Stopping audio recording...")
        play_stop_beep()
        state.audio_manager.stop_recording()
        
        logger.info("Waiting for transcription worker to finish processing queue...")
        state.transcription_queue.join() 
        logger.info("Transcription queue processed.")
        
        final_text = " ".join(state.transcribed_chunks).strip()
        if final_text:
            # Calculate recording duration
            duration_seconds = 0.0
            if state.recording_start_time:
                duration_seconds = time.time() - state.recording_start_time
                state.recording_start_time = None  # Reset for next recording

            logger.info(f"Final text (len {len(final_text)} chars): {final_text[:100]}...")
            copy_to_clipboard(final_text)
            paste_from_clipboard()

            state.console.print("\n[bold cyan]Transcription:[/bold cyan]")
            state.console.print(final_text)

            # Save to history
            try:
                history = get_history_manager()
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

