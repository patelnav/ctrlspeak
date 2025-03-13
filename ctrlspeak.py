#!/usr/bin/env python3
"""
ctrlSPEAK - A speech-to-text utility that runs in the background.
Triple-tap Ctrl to start/stop recording.
"""
import time
startup_time = time.time()

# Import our config module first
print(f"Starting imports... {time.time() - startup_time:.2f}s")
from utils.config import is_first_run, mark_first_run_complete, get_preferred_model, set_preferred_model

# Store first run flag
show_first_run_message = is_first_run()

# Import standard libraries
import os
import sys
import threading
import argparse
import logging
import warnings
import io
print(f"Basic imports done... {time.time() - startup_time:.2f}s")

# Early import of Rich console for first run message
from rich.console import Console
from rich.panel import Panel
# Initialize Rich console early
console = Console()
print = console.print  # Override print with Rich console.print

# Show first run message as early as possible
if show_first_run_message:
    console.print(Panel.fit(
        "[bold yellow]First time running ctrlSPEAK - optimizing libraries...[/bold yellow]\n"
        "This may take a minute, but future starts will be faster",
        title="First Run",
        border_style="blue"
    ))

# Import UI and system libraries
from AppKit import NSWorkspace
import subprocess
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.logging import RichHandler
from rich.text import Text
print(f"UI imports done... {time.time() - startup_time:.2f}s")

# Import ML libraries
print(f"Starting torch import... {time.time() - startup_time:.2f}s")
import torch
import numpy as np
print(f"Torch imported... {time.time() - startup_time:.2f}s")

# Import our modules
print(f"Starting app imports... {time.time() - startup_time:.2f}s")
from models.factory import ModelFactory
from utils.keyboard_shortcuts import KeyboardShortcutManager
from utils.clipboard import copy_to_clipboard, paste_from_clipboard
from utils.player import play_start_beep, play_stop_beep
from utils.audio import AudioManager, check_microphone_permissions, SAMPLE_RATE, CHANNELS
from utils import permission_manager
print(f"All imports done... {time.time() - startup_time:.2f}s")

# Debug flag - will be set from command line
DEBUG_MODE = False

# Silence all warnings from the beginning
warnings.filterwarnings("ignore")

# Configure logging aggressively before any imports
# This is needed because some libraries configure logging on import
class NullHandler(logging.Handler):
    def emit(self, record):
        pass

# Add null handler to root logger
logging.getLogger().addHandler(NullHandler())
logging.getLogger().setLevel(logging.CRITICAL)

# Configure NeMo and other framework logging before imports
os.environ["NEMO_LOGGING_LEVEL"] = "ERROR"  # Configure NeMo logging level (ERROR, WARNING, INFO, DEBUG)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Silence TensorFlow logging (0=all, 1=no INFO, 2=no WARNING, 3=no ERROR)

# Custom filter to block NeMo warnings
class FilterNemoWarnings(logging.Filter):
    def filter(self, record):
        # Allow only in debug mode, otherwise filter out NeMo warnings
        if DEBUG_MODE:
            return True
        if "nemo" in record.name.lower():
            return False
        return True

# Install the filter on the root logger
logging.getLogger().addFilter(FilterNemoWarnings())

# Now import the rest
from models.factory import ModelFactory
from utils.keyboard_shortcuts import KeyboardShortcutManager
from utils.clipboard import copy_to_clipboard, paste_from_clipboard
from utils.player import play_start_beep, play_stop_beep
from utils.audio import AudioManager, check_microphone_permissions, SAMPLE_RATE, CHANNELS
from utils import permission_manager

# Configure NeMo's logger if available
for logger_name in ['nemo', 'nemo_logger', 'nemo.collections']:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.CRITICAL)
    logger.addFilter(FilterNemoWarnings())

# Set up logging - need to do this early
logging.basicConfig(
    level=logging.WARNING,  # Root logger level (only show warnings and above by default)
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, console=console)]
)
logger = logging.getLogger("ctrlspeak")

# Silence all the noisy libraries
for lib in ['matplotlib', 'numba', 'urllib3', 'nemo', 'nemo_logger', 
           'nemo.collections', 'pytorch_lightning', 'filelock', 
           'huggingface_hub', 'transformers', 'sound_player']:
    logging.getLogger(lib).setLevel(logging.CRITICAL)
    # Also apply our filter to each logger
    logging.getLogger(lib).addFilter(FilterNemoWarnings())

def save_environment_variables():
    """Save current environment variables for later restoration"""
    return {
        "NEMO_LOGGING_LEVEL": os.environ.get("NEMO_LOGGING_LEVEL", ""),
        "TF_CPP_MIN_LOG_LEVEL": os.environ.get("TF_CPP_MIN_LOG_LEVEL", ""),
        "PYTORCH_ENABLE_MPS_FALLBACK": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "")
    }

def restore_environment_variables(saved_vars):
    """Restore environment variables from saved state"""
    for key, value in saved_vars.items():
        if value:
            os.environ[key] = value
        elif key in os.environ:
            del os.environ[key]

def setup_logging_for_mode(debug_mode):
    """Configure logging based on debug mode"""
    if debug_mode:
        # Debug mode - more verbose
        logger.setLevel(logging.DEBUG)
        os.environ["NEMO_LOGGING_LEVEL"] = "INFO"  # Allow NeMo INFO logs
        
        # Allow warnings in debug mode
        warnings.filterwarnings("default")
        
        # Configure library loggers for more info
        for lib in ['nemo', 'pytorch_lightning']:
            logging.getLogger(lib).setLevel(logging.INFO)
    else:
        # Regular mode - minimal output
        logger.setLevel(logging.INFO)
        os.environ["NEMO_LOGGING_LEVEL"] = "CRITICAL"  # Only critical NeMo logs
        
        # Silence common noisy loggers
        for lib in ['nemo', 'nemo_logger', 'nemo.collections', 'pytorch_lightning']:
            logging.getLogger(lib).setLevel(logging.CRITICAL)
        
        # Suppress warnings
        warnings.filterwarnings("ignore")

def quiet_function(func, *args, **kwargs):
    """Run a function with stdout and stderr redirected to /dev/null"""
    if DEBUG_MODE:
        # In debug mode, just run the function normally
        return func(*args, **kwargs)
    
    # In non-debug mode, redirect stdout and stderr
    null_file = open(os.devnull, 'w')
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    
    # Temporarily make logging even more quiet
    old_levels = {}
    for name, logger in logging.root.manager.loggerDict.items():
        if isinstance(logger, logging.Logger):
            old_levels[name] = logger.level
            logger.setLevel(logging.CRITICAL)
    
    try:
        sys.stdout = null_file
        sys.stderr = null_file
        return func(*args, **kwargs)
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        null_file.close()
        
        # Restore logging levels
        for name, level in old_levels.items():
            logging.getLogger(name).setLevel(level)

def print_startup_info():
    """Print startup information"""
    if DEBUG_MODE:
        logger.debug(f"PyTorch version: {torch.__version__}")
        logger.debug(f"CUDA available: {torch.cuda.is_available()}")
        logger.debug(f"MPS available: {torch.backends.mps.is_available()}")
        logger.debug(f"MPS backend enabled: {torch.backends.mps.is_built()}")
        logger.debug(f"Default PyTorch threads: {torch.get_num_threads()}")
    else:
        # For non-debug mode, just show a cleaner version
        console.print("[bold]ctrlSPEAK[/bold] - Speech recognition ready")

def check_permissions():
    """Check and request necessary permissions"""
    
    # Test microphone access by trying to open a stream
    try:
        console.print("\n[bold]Step 1 of 2: Checking microphone access...[/bold]")
        if not permission_manager.check_microphone_permissions(verbose=True, console=console):
            # Open System Settings
            console.print(Panel.fit(
                "[bold red]Microphone access required[/bold red]\n\n"
                "ctrlSPEAK needs microphone access to record your speech.\n"
                "Without this permission, the app cannot transcribe audio.\n\n"
                "[yellow]Opening System Settings → Privacy & Security → Microphone...[/yellow]\n"
                "Please add and enable this application in the list.",
                title="Permission Required",
                border_style="red"
            ))
            subprocess.run(["open", "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone"])
            console.print("\nPlease restart the application after granting permission.")
            sys.exit(1)
        else:
            console.print("[bold green]✓ Microphone access is granted.[/bold green]")
    except Exception as e:
        console.print(f"[bold red]Error accessing microphone: {e}[/bold red]")
        sys.exit(1)

    # Check keyboard accessibility permissions
    console.print("\n[bold]Step 2 of 2: Checking keyboard monitoring permissions...[/bold]")
    if not permission_manager.check_keyboard_permissions(verbose=True, console=console):
        console.print("\nPlease restart the application after granting permission.")
        sys.exit(1)
    
    console.print("\n[bold green]All required permissions are granted! Starting ctrlSPEAK...[/bold green]")
    
    return True

# Enable MPS (Metal) acceleration if available
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
logger.debug(f"Using device: {device}")

# Initialize audio manager
audio_manager = None  # Will be initialized in main()
stt_model = None  # Global model variable
model_type = "parakeet"  # Default model type
model_loaded = False  # Track whether the model has been loaded

def get_model():
    """Load model with progress tracking"""
    global stt_model, model_loaded
    
    if stt_model is not None:
        return stt_model
    
    console.print("\n[bold yellow]Loading model... please wait[/bold yellow]")
    start_time = time.time()
    
    # DIAGNOSTIC STEP 1: Simplified model loading approach from test_transcription.py
    # DIAGNOSTIC STEP 2: Enhanced exception handling with detailed logging
    try:
        # Step 1: Create model instance
        logger.info(f"Step 1: Creating {model_type} model instance...")
        try:
            stt_model = ModelFactory.get_model(model_type=model_type, device=device, verbose=DEBUG_MODE)
            logger.info("Model instance created successfully")
        except Exception as e:
            logger.error(f"Error creating model instance: {str(e)}")
            console.print(f"[bold red]Error creating model instance: {str(e)}[/bold red]")
            if DEBUG_MODE:
                import traceback
                traceback.print_exc()
            return None
        
        # Step 2: Load model weights
        logger.info("Step 2: Loading model weights...")
        try:
            stt_model.load_model()
            logger.info("Model weights loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model weights: {str(e)}")
            console.print(f"[bold red]Error loading model weights: {str(e)}[/bold red]")
            if DEBUG_MODE:
                import traceback
                traceback.print_exc()
            return None
        
        end_time = time.time()
        model_loaded = True
        console.print(f"[bold green]Model loaded in {end_time - start_time:.2f} seconds. Ready to record![/bold green]")
        
        return stt_model
    except Exception as e:
        logger.error(f"Unexpected error in get_model: {str(e)}")
        console.print(f"[bold red]Unexpected error in get_model: {str(e)}[/bold red]")
        if DEBUG_MODE:
            import traceback
            traceback.print_exc()
        return None

def process_audio_thread():
    """Thread function to process audio"""
    audio_manager.process_audio(get_model_func=get_model)

def on_activate():
    """Handle global hotkey activation"""
    global stt_model, model_loaded
    
    # Don't allow recording to start if model isn't loaded
    if not audio_manager.is_collecting:
        if not model_loaded:
            console.print("[bold yellow]Model is still loading. Please wait...[/bold yellow]")
            return
            
        # Start recording
        audio_manager.start_recording(play_start_beep_func=play_start_beep)
    else:
        # Stop recording
        audio_file = audio_manager.stop_recording(play_stop_beep_func=play_stop_beep)
        
        if audio_file:
            # Transcribe the complete recording
            try:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[bold green]Transcribing audio...[/bold green]"),
                ) as progress:
                    task = progress.add_task("Processing")
                    
                    start_time = time.time()
                    
                    # Special handling for transcription to block all messages
                    for name, logger in logging.root.manager.loggerDict.items():
                        if isinstance(logger, logging.Logger):
                            logger.setLevel(logging.CRITICAL)
                    logging.getLogger().setLevel(logging.CRITICAL)
                    os.environ["NEMO_LOGGING_LEVEL"] = "CRITICAL"
                    
                    # Use the clean single-file API directly
                    text = quiet_function(stt_model.transcribe, audio_file)
                    
                    # Debug logging
                    if DEBUG_MODE:
                        logger.debug(f"Transcription result: {text}")
                    
                    # Restore logging if in debug mode
                    if DEBUG_MODE:
                        setup_logging_for_mode(True)
                            
                    end_time = time.time()
                    progress.stop()
                
                # Copy to clipboard and simulate paste if we got a result
                if text:
                    copy_to_clipboard(text)
                    paste_from_clipboard()
                    
                    # Display transcription without a panel for easy copy-paste
                    console.print("\n[bold cyan]Transcription:[/bold cyan]")
                    # Display text as-is, no need for extra formatting
                    console.print(text)
                    
                    console.print(f"[dim]Completed in [bold]{end_time - start_time:.2f}[/bold] seconds[/dim]")
                else:
                    console.print("[yellow]No transcription result[/yellow]")
            except Exception as e:
                console.print(f"[bold red]Error during transcription: {e}[/bold red]")
                # If error occurs, at least show what's in clipboard
                console.print("[yellow]Text copied to clipboard - you can paste manually with Command+V[/yellow]")

def exit_app():
    """Exit the application"""
    global audio_manager
    
    # Stop the audio manager
    if audio_manager:
        audio_manager.set_is_running(False)
        # If we're recording, stop it
        if audio_manager.is_collecting:
            audio_manager.set_is_collecting(False)
    
    console.print("[bold yellow]Shutting down ctrlSPEAK...[/bold yellow]")
    
    # Force exit the program
    os._exit(0)  # Use os._exit to ensure immediate termination
    
    return False  # This won't be reached, but kept for clarity

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="ctrlSPEAK - Speech-to-text transcription tool")
    parser.add_argument("--model", type=str, choices=["parakeet", "canary", "whisper"], 
                        default=get_preferred_model(),
                        help="Speech recognition model to use")
    parser.add_argument("--debug", action="store_true", 
                        help="Enable debug mode with verbose logging")
    return parser.parse_args()

def main():
    """Main application entry point"""
    global model_type, DEBUG_MODE, audio_manager
    
    # DIAGNOSTIC STEP 3: Save environment variables at the start
    saved_env_vars = save_environment_variables()
    
    try:
        # Parse command-line arguments
        args = parse_arguments()
        DEBUG_MODE = args.debug
        model_type = args.model
        
        # Setup logging
        setup_logging_for_mode(DEBUG_MODE)
        
        # Check permissions first
        if not check_permissions():
            return 1
        
        # Update configuration after first run
        if is_first_run():
            mark_first_run_complete()
        
        # Save the selected model as preferred
        set_preferred_model(model_type)
        
        # Initialize audio manager with debug mode
        audio_manager = AudioManager(debug_mode=DEBUG_MODE)
        
        # Print startup info based on debug mode
        print_startup_info()
        
        # Create a welcome banner
        console.print(Panel.fit(
            "[bold cyan]ctrlSPEAK[/bold cyan] - A speech-to-text utility that runs in the background",
            title="Welcome",
            subtitle=f"Using {model_type} model",
            border_style="blue"
        ))
        
        # DIAGNOSTIC STEP 3: Restore environment variables before loading model
        restore_environment_variables(saved_env_vars)
        
        # Load model immediately on startup
        get_model()
        
        # Now show the controls after model is loaded
        table = Table(title="Controls", show_header=True, header_style="bold magenta")
        table.add_column("Action", style="green")
        table.add_column("Description", style="cyan")
        table.add_row("Triple-tap Ctrl", "Start/stop recording")
        table.add_row("Option+Esc", "Quit the application")
        console.print(table)
        
        # Set up keyboard shortcut manager
        keyboard_manager = KeyboardShortcutManager()
        keyboard_manager.register_triple_ctrl_tap(on_activate)
        keyboard_manager.register_shortcut('<alt>+<esc>', exit_app)
        
        # Start the processing thread
        process_thread = threading.Thread(target=process_audio_thread)
        process_thread.daemon = True  # Make thread daemon so it exits when main thread exits
        process_thread.start()
        
        # Start listening for hotkeys
        keyboard_manager.start_listening()
        
        # Start recording
        with audio_manager.start_input_stream():
            keyboard_manager.join()
    
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Stopping recording...[/bold yellow]")
    finally:
        # DIAGNOSTIC STEP 3: Restore environment variables at the end
        restore_environment_variables(saved_env_vars)
        
        if audio_manager:
            audio_manager.set_is_running(False)
        if 'process_thread' in locals() and process_thread.is_alive():
            process_thread.join(timeout=1.0)  # Wait up to 1 second for thread to finish
        if 'keyboard_manager' in locals():
            keyboard_manager.stop_listening()
        console.print("[bold green]ctrlSPEAK stopped.[/bold green]")
        os._exit(0)  # Ensure clean exit

if __name__ == "__main__":
    main() 