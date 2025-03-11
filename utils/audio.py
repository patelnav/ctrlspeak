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
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.style import Style
from rich.live import Live
from rich.text import Text
from rich import print as rprint

# Set up logger
logger = logging.getLogger("ctrlspeak.audio")

# Audio settings
SAMPLE_RATE = 16000  # NeMo expects 16kHz
CHANNELS = 1

class AudioManager:
    """Class to manage audio recording and state"""
    
    def __init__(self, debug_mode=False):
        """Initialize the audio manager"""
        self.audio_queue = Queue()
        self.is_running = True
        self.is_collecting = False
        self.collected_audio = np.array([], dtype=np.float32)
        self.model_loaded = False
        self.console = Console()
        self.recording_start_time = None
        self.recording_duration = 0
        self.live_display = None
        self.debug_mode = debug_mode
    
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
        status.append(f"({len(self.collected_audio) / SAMPLE_RATE:.1f}s of audio)", style="dim")
        
        return status
    
    def reset_collected_audio(self):
        """Reset the collected audio"""
        self.collected_audio = np.array([], dtype=np.float32)
    
    def get_collected_audio(self):
        """Get the collected audio"""
        return self.collected_audio
    
    def is_collecting_func(self):
        """Returns whether we're collecting audio"""
        return self.is_collecting
    
    def is_running_func(self):
        """Returns whether the app is still running"""
        return self.is_running
    
    def set_is_running(self, value):
        """Set the is_running flag"""
        self.is_running = value
    
    def add_to_collected_audio(self, chunk):
        """Adds a chunk to the collected audio"""
        self.collected_audio = np.append(self.collected_audio, chunk)
        # Update the live display if it exists
        if self.live_display:
            self.live_display.update(self._render_recording_status())
    
    def audio_callback(self, indata, frames, time, status):
        """Callback function to process audio input"""
        if status:
            if self.debug_mode:
                self.console.print(f"[bold red]Status: {status}[/bold red]")
            else:
                # Only log significant errors in non-debug mode
                if "overflow" not in str(status).lower() and "underflow" not in str(status).lower():
                    logger.warning(f"Audio status issue: {status}")
        # Convert to float32 and reshape
        audio_chunk = indata[:, 0].copy()
        self.audio_queue.put(audio_chunk)
    
    def process_audio(self, get_model_func=None):
        """Process audio chunks from the queue"""
        # Initialize model if provided
        if get_model_func:
            if self.debug_mode:
                self.console.print(Panel.fit("[bold yellow]Initializing model...[/bold yellow]", 
                                         title="ctrlSPEAK", 
                                         border_style="green"))
            get_model_func()
        
        if self.debug_mode:
            self.console.print("[dim]Ready and waiting for audio input...[/dim]")
        
        while self.is_running:
            if not self.audio_queue.empty():
                chunk = self.audio_queue.get()
                if self.is_collecting:
                    self.add_to_collected_audio(chunk)
            else:
                time.sleep(0.1)  # Small sleep to prevent busy waiting
    
    def start_recording(self, play_start_beep_func):
        """Start recording audio"""
        self.console.line()
        self.console.print(Panel.fit(
            "[bold green]Started recording...[/bold green]\n[dim]Tap Ctrl key 3 times quickly to stop[/dim]",
            border_style="green"
        ))
        
        # Reset and start collecting
        self.reset_collected_audio()
        play_start_beep_func()
        self.set_is_collecting(True)
    
    def stop_recording(self, play_stop_beep_func):
        """Stop recording and return the collected audio"""
        self.set_is_collecting(False)
        play_stop_beep_func()
        self.console.line()
        
        if len(self.collected_audio) > 0:
            # Save the complete recording without progress display
            temp_file = "temp_recording.wav"
            sf.write(temp_file, self.collected_audio, SAMPLE_RATE)
            return temp_file
        else:
            self.console.print("[bold red]No audio was recorded![/bold red]")
            return None
    
    def start_input_stream(self):
        """Start the audio input stream"""
        # Create a wrapper function that includes the audio_queue parameter
        def callback_wrapper(indata, frames, time, status):
            self.audio_callback(indata, frames, time, status)
        
        return sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=callback_wrapper)

# Keep the standalone functions for backward compatibility
def audio_callback(indata, frames, time, status, audio_queue):
    """Callback function to process audio input"""
    if status:
        logger.debug(f"Status: {status}")
    # Convert to float32 and reshape
    audio_chunk = indata[:, 0].copy()
    audio_queue.put(audio_chunk)

def process_audio(audio_queue, is_collecting_func, add_to_collected_audio_func, is_running_func, get_model_func=None):
    """
    Process audio chunks from the queue
    
    Args:
        audio_queue: Queue containing audio chunks
        is_collecting_func: Function that returns whether we're collecting audio
        add_to_collected_audio_func: Function to add a chunk to collected audio
        is_running_func: Function that returns whether the app is still running
        get_model_func: Optional function to initialize the model
    """
    # Initialize model if provided
    if get_model_func:
        get_model_func()
    
    while is_running_func():
        if not audio_queue.empty():
            chunk = audio_queue.get()
            if is_collecting_func():
                add_to_collected_audio_func(chunk)
                print(".", end="", flush=True)  # Show recording progress
        else:
            time.sleep(0.1)  # Small sleep to prevent busy waiting

def start_recording(is_collecting_setter, play_start_beep_func, reset_collected_audio_func):
    """
    Start recording audio
    
    Args:
        is_collecting_setter: Function to set the is_collecting flag
        play_start_beep_func: Function to play the start beep
        reset_collected_audio_func: Function to reset the collected audio
    """
    is_collecting_setter(True)
    reset_collected_audio_func()
    play_start_beep_func()
    print("\nStarted recording... (Tap Ctrl key 3 times quickly to stop)")

def stop_recording(is_collecting_setter, play_stop_beep_func, get_collected_audio_func):
    """
    Stop recording audio and save the recording
    
    Args:
        is_collecting_setter: Function to set the is_collecting flag
        play_stop_beep_func: Function to play the stop beep
        get_collected_audio_func: Function to get the collected audio
    
    Returns:
        The path to the saved audio file, or None if no audio was collected
    """
    is_collecting_setter(False)
    play_stop_beep_func()
    print("\nProcessing recording...")
    
    collected_audio = get_collected_audio_func()
    if len(collected_audio) > 0:
        # Save the complete recording
        temp_file = "temp_recording.wav"
        sf.write(temp_file, collected_audio, SAMPLE_RATE)
        return temp_file
    return None

def start_input_stream(callback, audio_queue):
    """
    Start the audio input stream
    
    Args:
        callback: Function to call when audio is received
        audio_queue: Queue to put audio chunks in
    
    Returns:
        The audio input stream
    """
    # Create a wrapper function that includes the audio_queue parameter
    def callback_wrapper(indata, frames, time, status):
        callback(indata, frames, time, status, audio_queue)
    
    return sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=callback_wrapper)

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