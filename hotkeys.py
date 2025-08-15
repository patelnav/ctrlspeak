
import logging
import state
from utils.clipboard import copy_to_clipboard, paste_from_clipboard
from utils.player import play_start_beep, play_stop_beep

logger = logging.getLogger("ctrlspeak")

def on_activate():
    """Handle global hotkey activation"""
    if not state.audio_manager.is_collecting:
        if not state.model_loaded:
            state.console.print("[bold yellow]Model is still loading. Please wait...[/bold yellow]")
            return
            
        state.transcribed_chunks.clear()
        logger.info("Cleared previous transcribed chunks.")

        logger.debug("Playing start beep...")
        play_start_beep()
        logger.debug("Calling audio_manager.start_recording()...")
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
            logger.info(f"Final text (len {len(final_text)} chars): {final_text[:100]}...")
            copy_to_clipboard(final_text)
            paste_from_clipboard()
            
            state.console.print("\n[bold cyan]Transcription:[/bold cyan]")
            state.console.print(final_text)
        else:
            state.console.print("[yellow]No transcription result[/yellow]")

