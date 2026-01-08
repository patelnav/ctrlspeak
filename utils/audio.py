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
    """Class to manage audio recording and state.

    Supports two modes:
    1. Queue mode (default): RMS-based silence detection, queues segments for batch transcription
    2. Streaming mode: Fixed-interval chunks, calls callback for real-time transcription
    """

    def __init__(self, transcription_queue, debug_mode=False, app_state=None):
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
        # Optional app_state for Textual UI integration
        self.app_state = app_state

        # Audio device selection
        self.input_device = None  # None means use default device
        self.current_stream = None  # Store active stream for hot swapping

        # Phase 2: Add state for RMS detection
        self.RMS_THRESHOLD = 0.01 # EXAMPLE VALUE - NEEDS TUNING LATER!
        # Phase 5 Tuning: Reduce silence duration based on user feedback
        self.SILENCE_DURATION_S = 1.0
        self.MIN_CHUNK_DURATION_S = 0.5 # Avoid transcribing tiny blips
        self.current_silence_s = 0.0
        self.is_potentially_speaking = False # Track if we heard sound recently
        self.last_rms = 0.0 # Store last RMS value for app_state updates

        # Streaming mode state
        self._streaming_mode = False
        self._streaming_callback = None  # Function to call with each chunk
        self._streaming_chunk_size_samples = 0  # Samples per chunk
        self._streaming_buffer = []  # Buffer for accumulating streaming chunks
    
    def set_debug_mode(self, debug_mode):
        """Set debug mode"""
        self.debug_mode = debug_mode

    def set_input_device(self, device_id):
        """Set the audio input device by ID."""
        self.input_device = device_id
        logger.info(f"Audio input device set to: {device_id}")

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
        """Callback function to process audio input.

        Routes to either:
        - Streaming mode: Fixed-interval chunks for real-time transcription
        - Queue mode: RMS-based silence detection for batch transcription
        """
        if not self.is_collecting:
            # Optimization: Don't process if not collecting
            return

        if status:
            logger.warning(f"Audio callback status: {status}")
            # Potentially return or handle specific statuses

        # Always make a copy
        chunk = indata.copy()

        # Route to appropriate mode
        if self._streaming_mode:
            self._streaming_audio_callback(chunk, frames)
            return

        # Otherwise, use the original RMS-based segmentation logic
        current_chunk_duration_s = float(frames) / SAMPLE_RATE
        
        # Calculate RMS
        try:
            rms = np.sqrt(np.mean(chunk**2))
            logger.debug(f"RMS: {rms:.6f}")
            self.last_rms = rms

            # Update app_state if available (for Textual UI)
            if self.app_state:
                self.app_state.current_rms = rms
                self.app_state.buffer_size_samples = len(self.audio_buffer)
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

            # Update app_state with current silence
            if self.app_state:
                self.app_state.current_silence_s = self.current_silence_s
            
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

    # =========================================================================
    # Streaming Mode Methods
    # =========================================================================

    def start_streaming(self, chunk_size_ms: int = 560, on_chunk_callback=None):
        """Start recording in streaming mode with fixed-interval chunks.

        In streaming mode, audio is collected in fixed-size chunks (not based on
        silence detection) and passed to the callback for real-time transcription.

        Args:
            chunk_size_ms: Size of each chunk in milliseconds (default 560ms).
                          Valid values: 80, 160, 560, 1120 (aligned with model frames)
            on_chunk_callback: Function to call with each chunk.
                              Signature: callback(audio_samples: np.ndarray) -> str
                              The callback receives float32 audio at 16kHz.
        """
        if self.is_collecting:
            logger.warning("Already collecting audio. Stop first before starting streaming.")
            return

        logger.info(f"Starting streaming mode (chunk size: {chunk_size_ms}ms)")

        # Configure streaming
        self._streaming_mode = True
        self._streaming_callback = on_chunk_callback
        self._streaming_chunk_size_samples = int(SAMPLE_RATE * chunk_size_ms / 1000)
        self._streaming_buffer = []

        # Reset standard buffer and state
        self.reset_collected_audio()

        # Start collection (reuses existing live display logic)
        self.set_is_collecting(True)

        self.console.line()
        self.console.print(Panel.fit(
            "[bold green]Started streaming recording...[/bold green]\n"
            f"[dim]Chunk size: {chunk_size_ms}ms | Tap Ctrl key 3 times quickly to stop[/dim]",
            border_style="green"
        ))

    def stop_streaming(self) -> None:
        """Stop streaming recording and process any remaining audio.

        Returns any text from processing the final buffer.
        """
        if not self.is_collecting:
            logger.warning("stop_streaming called but not collecting.")
            return

        if not self._streaming_mode:
            logger.warning("stop_streaming called but not in streaming mode. Use stop_recording() instead.")
            self.stop_recording()
            return

        logger.info("Stopping streaming recording...")
        self.set_is_collecting(False)
        self.console.line()

        # Process any remaining audio in the streaming buffer
        if self._streaming_buffer and self._streaming_callback:
            remaining_samples = sum(len(c) for c in self._streaming_buffer)
            if remaining_samples > 0:
                # Concatenate remaining buffer
                remaining_audio = np.concatenate(self._streaming_buffer).flatten()
                duration_ms = len(remaining_audio) / SAMPLE_RATE * 1000
                logger.info(f"[AUDIO_STOP] Final buffer: {len(remaining_audio)} samples ({duration_ms:.0f}ms)")
                try:
                    logger.debug(f"[AUDIO_STOP] Calling streaming callback with final buffer (is_final=True)...")
                    self._streaming_callback(remaining_audio, is_final=True)
                    logger.debug(f"[AUDIO_STOP] Final streaming callback completed")
                except Exception as e:
                    logger.error(f"Error in final streaming callback: {e}")
            else:
                logger.info("[AUDIO_STOP] No remaining samples in buffer")
        else:
            logger.info(f"[AUDIO_STOP] No final buffer to process (buffer={bool(self._streaming_buffer)}, callback={bool(self._streaming_callback)})")

        # Reset streaming state
        self._streaming_mode = False
        self._streaming_callback = None
        self._streaming_buffer = []
        self._streaming_chunk_size_samples = 0

        # Clear standard buffer and state
        self.reset_collected_audio()

    def _streaming_audio_callback(self, chunk: np.ndarray, frames: int):
        """Internal callback for streaming mode audio processing.

        Accumulates audio until chunk_size is reached, then calls the streaming callback.

        Args:
            chunk: Audio data from sounddevice (shape: [frames, channels])
            frames: Number of frames in chunk
        """
        # Flatten if needed (sounddevice gives [frames, channels])
        if chunk.ndim > 1:
            chunk = chunk.flatten()

        # Add to streaming buffer
        self._streaming_buffer.append(chunk)

        # Calculate total samples in buffer
        total_samples = sum(len(c) for c in self._streaming_buffer)

        # Calculate RMS for UI feedback
        try:
            rms = np.sqrt(np.mean(chunk**2))
            self.last_rms = rms
            if self.app_state:
                self.app_state.current_rms = rms
        except Exception as e:
            logger.debug(f"Error calculating RMS in streaming mode: {e}")

        # Check if we have enough samples for a chunk
        if total_samples >= self._streaming_chunk_size_samples:
            # Concatenate buffer
            audio_data = np.concatenate(self._streaming_buffer).flatten()

            # Take exactly chunk_size samples
            chunk_samples = audio_data[:self._streaming_chunk_size_samples]

            # Keep remainder in buffer
            remainder = audio_data[self._streaming_chunk_size_samples:]
            if len(remainder) > 0:
                self._streaming_buffer = [remainder]
            else:
                self._streaming_buffer = []

            # Ensure float32 in range [-1, 1]
            if chunk_samples.dtype != np.float32:
                chunk_samples = chunk_samples.astype(np.float32)

            # Call the streaming callback
            if self._streaming_callback:
                try:
                    duration_ms = len(chunk_samples) / SAMPLE_RATE * 1000
                    logger.debug(f"[AUDIO_CHUNK] Sending chunk: {len(chunk_samples)} samples ({duration_ms:.0f}ms)")
                    self._streaming_callback(chunk_samples, is_final=False)
                except Exception as e:
                    logger.error(f"Error in streaming callback: {e}")

        # Update live display
        if self.live_display and int(timer_module.time() * 4) % 2 == 0:
            try:
                self.live_display.update(self._render_recording_status())
            except Exception as e:
                logger.error(f"Error updating live display: {e}", exc_info=False)
                self.live_display = None

    @property
    def is_streaming(self) -> bool:
        """Check if currently in streaming mode."""
        return self._streaming_mode and self.is_collecting

    def start_input_stream(self):
        """Start the audio input stream using the instance method as callback"""
        logger.info("AudioManager: Starting audio input stream...")
        # Use self.audio_callback directly
        device = self.input_device if self.input_device is not None else None
        if device is not None:
            logger.info(f"Using audio device: {device}")
        stream = sd.InputStream(device=device, samplerate=SAMPLE_RATE, channels=CHANNELS, callback=self.audio_callback)
        self.current_stream = stream
        return stream

    def restart_input_stream(self, new_device_id):
        """
        Restart the audio input stream with a new device.
        Used for hot-swapping audio devices without restarting the app.

        Args:
            new_device_id: ID of the new audio input device (or None for default)

        Returns:
            The new InputStream object

        Raises:
            Exception: If stream restart fails
        """
        logger.info(f"AudioManager: Restarting input stream with device {new_device_id}...")

        try:
            # Stop current stream if it exists
            if self.current_stream is not None:
                try:
                    logger.debug("Stopping current audio stream...")
                    self.current_stream.stop()
                    self.current_stream.close()
                    logger.info("Current audio stream stopped and closed")
                except Exception as e:
                    logger.warning(f"Error stopping current stream (continuing anyway): {e}")

            # Update device
            self.input_device = new_device_id

            # Create and start new stream
            logger.info(f"Creating new audio stream with device: {new_device_id}")
            device = self.input_device if self.input_device is not None else None
            new_stream = sd.InputStream(
                device=device,
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                callback=self.audio_callback
            )

            # Start the new stream
            new_stream.start()
            self.current_stream = new_stream

            # If device was None, resolve to actual default device for state tracking
            if new_device_id is None:
                actual_device_id = sd.default.device[0] if sd.default.device else None
                logger.info(f"Audio stream successfully restarted with default device (resolved to {actual_device_id})")
            else:
                logger.info(f"Audio stream successfully restarted with device {new_device_id}")

            return new_stream

        except Exception as e:
            logger.error(f"Failed to restart audio stream: {e}", exc_info=True)
            # Try to restore a working stream with the old device or default
            logger.warning("Attempting to restore previous audio stream...")
            try:
                fallback_stream = sd.InputStream(
                    device=None,  # Use default device as fallback
                    samplerate=SAMPLE_RATE,
                    channels=CHANNELS,
                    callback=self.audio_callback
                )
                fallback_stream.start()
                self.current_stream = fallback_stream
                self.input_device = None
                logger.info("Restored audio stream with default device")
            except Exception as fallback_error:
                logger.error(f"Failed to restore audio stream: {fallback_error}")

            raise e

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