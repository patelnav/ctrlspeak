#!/usr/bin/env python3
"""
Audio functionality for ctrlSPEAK.
"""
import sounddevice as sd
import numpy as np
import time
import logging
from queue import Queue
import soundfile as sf
from rich.console import Console
import time as timer_module
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.style import Style
from rich.live import Live
from rich.text import Text
from rich import print as rprint
# Phase 1: Remove direct import from ctrlspeak
# from ctrlspeak import transcription_queue 

# Set up logger
logger = logging.getLogger("ctrlspeak.audio")

# Audio settings
SAMPLE_RATE = 16000  # NeMo expects 16kHz
CHANNELS = 1

class AudioManager:
    """Class to manage audio recording and state"""
    
    def __init__(self, transcription_queue, debug_mode=False):
        """Initialize the audio manager"""
        self.is_running = True
        self.is_collecting = False
        self.audio_buffer = [] 
        self.model_loaded = False # This might be managed elsewhere
        self.console = Console()
        self.recording_start_time = None
        self.recording_duration = 0
        self.live_display = None
        self.debug_mode = debug_mode
        # Store the queue passed from ctrlspeak
        self.transcription_queue = transcription_queue 
        
        # Phase 2: Add state for RMS detection
        self.RMS_THRESHOLD = 0.01 # EXAMPLE VALUE - NEEDS TUNING LATER!
        # Phase 5 Tuning: Reduce silence duration based on user feedback
        self.SILENCE_DURATION_S = 1.0 
        self.MIN_CHUNK_DURATION_S = 0.5 # Avoid transcribing tiny blips
        self.current_silence_s = 0.0
        self.is_potentially_speaking = False # Track if we heard sound recently
    
    def set_debug_mode(self, debug_mode):
        """Set debug mode"""
        self.debug_mode = debug_mode
    
    def set_is_collecting(self, value):
        """Set the is_collecting flag"""
        self.is_collecting = value
        if value:
            self.recording_start_time = time.time()
            # Start the live display for recording status
            if self.live_display is None:
                self.live_display = Live(
                    self._render_recording_status(),
                    refresh_per_second=4,  # Update 4 times per second
                    console=self.console
                )
                self.live_display.start()
        else:
            # Stop the live display when recording ends
            if self.live_display:
                self.live_display.stop()
                self.live_display = None
    
    def _render_recording_status(self):
        """Render the recording status for the live display"""
        if not self.is_collecting:
            return Text("")
        
        # Calculate duration
        self.recording_duration = time.time() - self.recording_start_time
        minutes, seconds = divmod(int(self.recording_duration), 60)
        
        # Create pulsing animation for recording indicator
        pulse = "●" if int(time.time() * 2) % 2 == 0 else "○"
        
        # Create status text
        status = Text()
        status.append("Recording ", style="bold cyan")
        status.append(f"{minutes:02d}:{seconds:02d}", style="bold white")
        status.append(f" {pulse} ", style="bold red")
        status.append(f"({len(self.audio_buffer) / SAMPLE_RATE:.1f}s of audio)", style="dim")
        
        return status
    
    def reset_collected_audio(self):
        """Reset the collected audio buffer and silence state"""
        self.audio_buffer = []
        # Phase 3: Reset silence tracking state as well
        self.current_silence_s = 0.0
        self.is_potentially_speaking = False
        logger.debug("AudioManager: Cleared audio buffer and silence state.")
    
    def is_collecting_func(self):
        """Returns whether we're collecting audio"""
        return self.is_collecting
    
    def is_running_func(self):
        """Returns whether the app is still running"""
        return self.is_running
    
    def set_is_running(self, value):
        """Set the is_running flag"""
        self.is_running = value
    
    def audio_callback(self, indata, frames, time_info, status):
        """Callback function to process audio input (Phase 3: RMS Segmentation)"""
        if not self.is_collecting:
            # Optimization: Don't process if not collecting
            return 
            
        if status:
            logger.warning(f"Audio callback status: {status}")
            # Potentially return or handle specific statuses

        # Always make a copy
        chunk = indata.copy()
        current_chunk_duration_s = float(frames) / SAMPLE_RATE
        
        # Calculate RMS
        try:
            rms = np.sqrt(np.mean(chunk**2))
            logger.debug(f"RMS: {rms:.6f}") 
        except Exception as e:
            logger.error(f"Error calculating RMS: {e}")
            rms = 0 # Assign a default value on error
            
        # --- Phase 3: Segmentation Logic ---
        is_speech_chunk = rms >= self.RMS_THRESHOLD

        if is_speech_chunk:
            # Speech or noise detected
            self.audio_buffer.append(chunk) # Buffer this chunk
            if not self.is_potentially_speaking:
                 logger.debug("Speech detected (RMS threshold crossed).")
            # Reset silence duration because we heard sound
            self.current_silence_s = 0.0 
            self.is_potentially_speaking = True
            
        elif self.is_potentially_speaking:
            # Silence detected *after* we were potentially speaking
            self.current_silence_s += current_chunk_duration_s
            logger.debug(f"Silence accumulating: {self.current_silence_s:.2f}s / {self.SILENCE_DURATION_S}s")
            
            # Check if silence duration threshold is met
            if self.current_silence_s >= self.SILENCE_DURATION_S:
                logger.info(f"Silence threshold reached ({self.current_silence_s:.2f}s). Segmenting audio.")
                
                # We have enough silence. Process the buffered speech *before* this silence.
                if self.audio_buffer:
                    # Concatenate all buffered chunks (which includes speech and the initial <2s silence)
                    segment_data = np.concatenate(self.audio_buffer)
                    segment_duration_s = len(segment_data) / SAMPLE_RATE

                    # Check minimum length for transcription
                    if segment_duration_s >= self.MIN_CHUNK_DURATION_S:
                         logger.info(f"Queueing segment of {segment_duration_s:.2f}s for transcription.")
                         try:
                             self.transcription_queue.put(segment_data)
                         except Exception as q_e:
                             logger.error(f"Error putting segment onto queue: {q_e}")
                    else:
                         logger.info(f"Skipping short segment ({segment_duration_s:.2f}s) below minimum duration {self.MIN_CHUNK_DURATION_S}s.")
                else:
                    logger.debug("Silence threshold reached, but audio buffer was empty. Skipping queue.")

                # Reset buffer and state for the next speech segment
                self.audio_buffer = []
                self.current_silence_s = 0.0
                self.is_potentially_speaking = False # Stay silent until speech is detected again
            else:
                # Silence continues but hasn't reached threshold yet.
                # Keep buffering the silence chunks in case speech resumes quickly.
                 self.audio_buffer.append(chunk)
                 
        # else (is_speech_chunk is False AND self.is_potentially_speaking is False):
            # This means silence continues after a segment was already cut, 
            # or it's initial silence before any speech. 
            # Do nothing - don't buffer this silence, don't reset anything.
            pass

        # Update live display (if using)
        if self.live_display and int(timer_module.time() * 4) % 2 == 0: 
             try:
                 self.live_display.update(self._render_recording_status())
             except Exception as e:
                 logger.error(f"Error updating live display: {e}", exc_info=False) 
                 self.live_display = None
    
    def start_recording(self):
        """Start recording audio (Phase 3)"""
        self.console.line()
        self.console.print(Panel.fit(
            "[bold green]Started recording...[/bold green]\n[dim]Tap Ctrl key 3 times quickly to stop[/dim]",
            border_style="green"
        ))
        
        # Phase 3: Reset buffer and silence state
        self.reset_collected_audio() # Now resets buffer and silence state
        self.set_is_collecting(True) # This also starts the timer/live display
        # Beep now played in ctrlspeak.py
    
    def stop_recording(self):
        """Stop recording and queue the FINAL collected audio segment (Phase 3)"""
        if not self.is_collecting:
            logger.warning("stop_recording called but not collecting.")
            return
            
        logger.info("AudioManager: Stopping recording. Processing final segment...")
        self.set_is_collecting(False) 
        self.console.line()
        
        # Process any remaining audio in the buffer as the last segment
        if self.audio_buffer:
            logger.info(f"AudioManager: Concatenating final {len(self.audio_buffer)} audio chunks...")
            try:
                segment_data = np.concatenate(self.audio_buffer)
                segment_duration_s = len(segment_data) / SAMPLE_RATE
                logger.info(f"AudioManager: Final segment duration: {segment_duration_s:.2f}s")

                # Check minimum length for the final segment
                if segment_duration_s >= self.MIN_CHUNK_DURATION_S:
                    logger.info(f"Queueing final segment of {segment_duration_s:.2f}s for transcription.")
                    self.transcription_queue.put(segment_data)
                else:
                    logger.info(f"Skipping short final segment ({segment_duration_s:.2f}s) below minimum duration {self.MIN_CHUNK_DURATION_S}s.")
                    
            except Exception as e:
                logger.error(f"AudioManager: Error concatenating or queueing final audio segment: {e}", exc_info=True)
        else:
            logger.warning("AudioManager: No final audio segment in buffer to process.")

        # Clear buffer and reset state after stopping
        self.reset_collected_audio()
    
    def start_input_stream(self):
        """Start the audio input stream using the instance method as callback"""
        logger.info("AudioManager: Starting audio input stream...")
        # Use self.audio_callback directly
        return sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=self.audio_callback)

# Remove standalone functions if they are no longer used elsewhere
# def audio_callback(indata, frames, time, status, audio_queue): ...
# def process_audio(audio_queue, ...): ...
# def start_recording(is_collecting_setter, ...): ...
# def stop_recording(is_collecting_setter, ...): ...
# def start_input_stream(callback, audio_queue): ...
# def check_microphone_permissions(): ... # Keep if used elsewhere or move logic

def check_microphone_permissions():
    """Check microphone permissions"""
    console = Console()
    
    try:
        logger.debug("Checking microphone access...")
        with sd.InputStream(channels=1, callback=lambda *args: None):
            pass
        console.print("[green]✓[/green] Microphone access granted")
        return True
    except sd.PortAudioError as e:
        if "Permission Denied" in str(e):
            logger.error("Error: Microphone access denied!")
            console.print("[bold red]✗ Error: Microphone access denied![/bold red]")
            console.print("[yellow]Please enable microphone access in System Settings -> Privacy & Security -> Microphone[/yellow]")
            return False
        else:
            logger.error(f"Error accessing microphone: {e}")
            console.print(f"[bold red]✗ Error accessing microphone: {e}[/bold red]")
            return False 