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
# import io # Unused
# Added imports for Phase 0
import queue
import numpy as np
import time as timer_module # Use alias to avoid conflict with time module used elsewhere
# Phase 4: Imports needed for temp file handling in worker
import tempfile
import soundfile as sf
from pathlib import Path # <<< Ensure this import is present
from huggingface_hub import constants as hf_constants # Added for cache location
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
# from AppKit import NSWorkspace # Unused
import subprocess # Keep for permission opening
# from rich.table import Table # Unused
# from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn # Unused
from rich.logging import RichHandler
# from rich.text import Text # Unused
print(f"UI imports done... {time.time() - startup_time:.2f}s")

# Import ML libraries
print(f"Starting torch import... {time.time() - startup_time:.2f}s")
import torch
print(f"Torch imported... {time.time() - startup_time:.2f}s")

# Import our modules
print(f"Starting app imports... {time.time() - startup_time:.2f}s")
# <<< Keep only this first set of app imports >>>
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

# Define known model mappings (directory prefix -> CLI name)
MODEL_CACHE_MAP = {
    "models--nvidia--parakeet-tdt-0.6b-v3": "nvidia/parakeet-tdt-0.6b-v3",
    "models--nvidia--canary-1b": "nvidia/canary-1b",
    "models--openai--whisper-large-v3-turbo": "openai/whisper-large-v3"
}

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
        # Phase 2 Fix: Explicitly set audio logger level
        logging.getLogger("ctrlspeak.audio").setLevel(logging.DEBUG)
        os.environ["NEMO_LOGGING_LEVEL"] = "INFO"  # Allow NeMo INFO logs
        
        # Allow warnings in debug mode
        warnings.filterwarnings("default")
        
        # Configure library loggers for more info
        for lib in ['nemo', 'pytorch_lightning']:
            logging.getLogger(lib).setLevel(logging.INFO)
    else:
        # Regular mode - minimal output
        logger.setLevel(logging.INFO)
        # Ensure audio logger is also set appropriately for non-debug
        logging.getLogger("ctrlspeak.audio").setLevel(logging.WARNING) # Or INFO?
        os.environ["NEMO_LOGGING_LEVEL"] = "CRITICAL"  # Only critical NeMo logs
        
        # Silence common noisy loggers
        for lib in ['nemo', 'nemo_logger', 'nemo.collections', 'pytorch_lightning']:
            logging.getLogger(lib).setLevel(logging.CRITICAL)
        
        # Suppress warnings
        warnings.filterwarnings("ignore")

    # --- Phase 0: Run Test ---
    # REMOVED FROM HERE
    # --- End Phase 0 Test Call ---

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

# --- Start: Async Infrastructure ---
# Global variables for transcription worker
transcribed_chunks = [] # Stores results from worker
transcription_queue = queue.Queue() # Channel to send audio data to worker
# transcription_active = threading.Event() # Obsolete
transcription_worker_thread = None # Holds the worker thread object
# Flag to control main loop
main_loop_active = True

def transcription_worker(model, work_queue, results_list):
    """
    Pulls audio data from queue, transcribes using the real model (via temp file),
    adds text to results_list. Runs in a separate thread until None is received.
    """
    logger.debug("Transcription worker thread started.")
    global SAMPLE_RATE 

    # Phase 5 Lifecycle Change: Loop until None sentinel
    while True: 
        audio_data = None
        temp_file_path = None
        try:
            # Block indefinitely until an item is available
            audio_data = work_queue.get() 
            
            # Phase 5 Lifecycle Change: Check for sentinel
            if audio_data is None:
                logger.info("Worker received None sentinel. Exiting loop.")
                work_queue.task_done() # Mark sentinel as processed
                # Shutdown Log: Add log before break
                logger.debug("Worker thread loop terminating.")
                break # Exit the while loop
            
            logger.debug(f"Worker received chunk of type {type(audio_data)} and shape {getattr(audio_data, 'shape', 'N/A')}")

            # --- Phase 4: Real Transcription via Temp File ---
            if len(audio_data) == 0:
                 logger.warning("Worker received empty audio data array, skipping.")
                 work_queue.task_done()
                 continue

            # 1. Create temp file
            # Use delete=False so we control deletion after transcription
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                temp_file_path = tmp.name
            logger.debug(f"Worker created temp file: {temp_file_path}")

            # 2. Write NumPy array to WAV file
            try:
                # Ensure data is float32 or int16 as expected by soundfile
                if audio_data.dtype != np.float32:
                    # Attempt conversion if needed, log warning
                    logger.warning(f"Audio data was {audio_data.dtype}, attempting conversion to float32 for sf.write")
                    audio_data = audio_data.astype(np.float32)
                
                sf.write(temp_file_path, audio_data, SAMPLE_RATE)
                logger.debug(f"Worker successfully wrote {len(audio_data)} samples to {temp_file_path}")
            except Exception as write_e:
                logger.error(f"Worker failed to write temp WAV file {temp_file_path}: {write_e}", exc_info=True)
                # Skip transcription if file writing failed
                work_queue.task_done()
                # Clean up temp file if write failed but path exists
                if temp_file_path and os.path.exists(temp_file_path):
                     try: os.unlink(temp_file_path)
                     except Exception: pass
                continue

            # 3. Transcribe using the model (expects a list of paths)
            logger.debug(f"Worker calling model.transcribe() for {temp_file_path}...")
            transcription_start_time = timer_module.time()
            # The transcribe method might be transcribe_batch, check model interface
            # Assuming model object has a method like transcribe that takes a list
            try:
                 # Use transcribe_batch as found in ParakeetModel
                 results = model.transcribe_batch([temp_file_path])
                 if results and isinstance(results, list):
                      text = results[0] # Take the first result for the single file
                 else:
                      text = None
                      logger.warning(f"Worker received unexpected result type from transcribe_batch: {type(results)}")
                 transcription_duration = timer_module.time() - transcription_start_time
                 logger.info(f"Worker transcribed chunk in {transcription_duration:.2f}s: {text[:30]}...")
            except Exception as transcribe_e:
                 logger.error(f"Worker: Error during model transcription: {transcribe_e}", exc_info=True)
                 text = None
            # --- End Phase 4 Transcription ---

            if text:
                # Print the intermediate result (dimmed, no prefix) with a preceding newline
                console.print(f"\n[dim]{text}[/dim]")
                results_list.append(text)

            # 4. Clean up temp file (now happens in finally)
            work_queue.task_done() # Signal this task is done

        except Exception as e:
            # General error handling for the loop (e.g., unexpected queue issue)
            logger.error(f"Unexpected error in transcription worker loop (before finally): {e}", exc_info=True)
            # Ensure task is marked done if we successfully got an item but failed later
            if audio_data is not None: # Check if we actually got an item
                try:
                    work_queue.task_done()
                except ValueError: # If already marked done in inner try
                    pass 
            # Avoid tight loop on persistent errors
            timer_module.sleep(0.1)
        finally:
            # 5. Ensure temp file deletion 
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                    logger.debug(f"Worker deleted temp file: {temp_file_path}")
                except Exception as del_e:
                     logger.error(f"Worker failed to delete temp file {temp_file_path}: {del_e}")

    logger.info("Transcription worker thread finished normally.")

def get_model():
    """Load model with progress tracking"""
    # model_type is accessed globally here, as set/resolved in main()
    global stt_model, model_loaded, model_type 
    
    if stt_model is not None:
        return stt_model
    
    console.print("\n[bold yellow]Loading model... please wait[/bold yellow]")
    start_time = time.time()
    
    try:
        # Step 1: Create model instance using the resolved global model_type
        logger.info(f"Step 1: Creating {model_type} model instance...")
        try:
            # Pass the resolved model_type to the factory method
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
    # Remove obsolete comments/code
    global stt_model, model_loaded, audio_manager, transcribed_chunks, transcription_queue, transcription_worker_thread
    
    if not audio_manager.is_collecting:
        # --- Start Recording --- 
        if not model_loaded:
            console.print("[bold yellow]Model is still loading. Please wait...[/bold yellow]")
            return
            
        transcribed_chunks.clear()
        logger.info("Cleared previous transcribed chunks.")

        # Remove obsolete worker restart logic
        # if transcription_worker_thread is None or not transcription_worker_thread.is_alive():
        #    ...
        
        # logger.debug("Setting transcription_active event.") # Obsolete
        # transcription_active.set() # Obsolete

        # ... (Play beep)
        logger.debug("Playing start beep...")
        play_start_beep()
        logger.debug("Calling audio_manager.start_recording()...")
        audio_manager.start_recording()
    else:
        # --- Stop Recording --- 
        logger.info("Stop activated. Stopping audio recording...")
        play_stop_beep() # Play beep immediately on stop trigger
        audio_manager.stop_recording() # Queues audio data
        
        # Wait for queue processing
        logger.info("Waiting for transcription worker to finish processing queue...")
        # transcription_active.clear() # Obsolete
        transcription_queue.join() 
        logger.info("Transcription queue processed.")
        
        # ... (Concatenate results, play beep, paste)
        final_text = " ".join(transcribed_chunks).strip()
        # play_stop_beep() # Moved earlier
        if final_text:
            logger.info(f"Final text (len {len(final_text)} chars): {final_text[:100]}...")
            copy_to_clipboard(final_text)
            paste_from_clipboard()
            
            # Display transcription without a panel for easy copy-paste
            console.print("\n[bold cyan]Transcription:[/bold cyan]")
            # Display text as-is, no need for extra formatting
            console.print(final_text)
        else:
            console.print("[yellow]No transcription result[/yellow]")

def exit_app():
    """Initiates the application shutdown sequence."""
    # Use flag to break main loop
    global audio_manager, transcription_worker_thread, transcription_queue, keyboard_manager, main_loop_active
    
    logger.info("Shutdown requested.")
    console.print("[bold yellow]Shutting down ctrlSPEAK...[/bold yellow]")

    # 1. Stop audio recording if active
    if audio_manager and audio_manager.is_collecting:
        logger.info("Stopping active recording during exit...")
        audio_manager.stop_recording() 
    
    # 2. Signal transcription worker to exit by putting None on queue
    logger.info("Signaling transcription worker to exit...")
    transcription_queue.put(None) # Sentinel value
    
    # 3. Stop the keyboard listener
    if 'keyboard_manager' in globals() and keyboard_manager is not None:
        logger.info("Stopping keyboard listener...")
        try:
             keyboard_manager.stop_listening()
             # Shutdown Log: Add log after stop_listening
             logger.info("Keyboard listener stop signaled.")
        except Exception as e_stop_kb:
             logger.error(f"Error stopping keyboard listener in exit_app: {e_stop_kb}")
    
    # 4. Signal main loop to exit
    logger.info("Signaling main loop to exit...")
    main_loop_active = False
    
    logger.info("Exit_app finished signaling components.")

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="ctrlSPEAK - Speech-to-text transcription tool")
    parser.add_argument("--model", type=str, 
                        default=get_preferred_model(),
                        help="Speech recognition model to use (default: %(default)s)")
    parser.add_argument("--debug", action="store_true", 
                        help="Enable debug mode with verbose logging")
    parser.add_argument("--check-only", action="store_true", # <<< Add check-only flag
                        help="Check model cache and configuration, then exit.")
    parser.add_argument("--list-models", action="store_true",
                        help="List all supported models and exit.")
    return parser.parse_args()

def find_cached_models(): # Takes no arguments
    """Scans the Hugging Face cache directory for known ctrlspeak models."""
    cached = set()
    try:
        # Use huggingface_hub to find the cache directory reliably
        cache_dir = Path(hf_constants.HF_HUB_CACHE)
        logger.info(f"Checking Hugging Face cache at: {cache_dir}")

        if not cache_dir.is_dir():
            logger.warning(f"Hugging Face cache directory not found: {cache_dir}")
            return cached

        for item in cache_dir.iterdir():
            # Check if it's a directory and matches one of our known model prefixes
            if item.is_dir() and item.name in MODEL_CACHE_MAP:
                cached.add(MODEL_CACHE_MAP[item.name])
                logger.debug(f"Found cached model directory: {item.name} -> {MODEL_CACHE_MAP[item.name]}")

    except Exception as e:
        logger.error(f"Error scanning Hugging Face cache: {e}", exc_info=DEBUG_MODE)
        console.print(f"[yellow]Warning: Could not scan Hugging Face cache ({e})[/yellow]")

    logger.info(f"Found cached models: {cached}")
    return cached

def main():
    """Main application entry point"""
    # Use flag to control main loop
    global model_type, DEBUG_MODE, audio_manager, stt_model, transcription_worker_thread, transcribed_chunks, transcription_queue, main_loop_active
    
    # Save environment variables at the start
    saved_env_vars = save_environment_variables()
    
    try:
        # Parse command-line arguments
        args = parse_arguments()

        if args.list_models:
            console.print("\n[bold]Supported Models:[/bold]")
            for alias, model_name in ModelFactory._DEFAULT_ALIASES.items():
                console.print(f"  - [cyan]{alias}[/cyan]: {model_name}")
            sys.exit(0)

        DEBUG_MODE = args.debug
        model_type_arg = args.model # Get the raw argument

        # --- Resolve potential model alias using the factory --- 
        # This updates the global model_type variable used by get_model()
        model_type = ModelFactory.resolve_model_alias(model_type_arg)
        # --- End resolution ---

        # Setup logging
        setup_logging_for_mode(DEBUG_MODE)

        # --- Check Cache EARLY ---
        logger.info("Scanning for cached models...")
        # Call find_cached_models without arguments
        cached_models = find_cached_models() 
        # --- End Cache Check ---

        # --- Display Model Selection Info BEFORE Loading ---
        console.print("\n[bold]Model Configuration:[/bold]")
        # Display the original argument and the resolved type
        if model_type_arg.lower() != model_type.lower():
             console.print(f"  Selected (alias): [cyan]{model_type_arg}[/cyan] -> Resolved: [cyan]{ModelFactory.resolve_model_alias(model_type)}[/cyan]")
        else:
             console.print(f"  Selected: [cyan]{ModelFactory.resolve_model_alias(model_type)}[/cyan]")
        
        if cached_models:
            # Check cache using the *resolved* global model_type
            other_cached = sorted(list(cached_models - {ModelFactory.resolve_model_alias(model_type)})) 
            if model_type in cached_models:
                 console.print(f"  Status: [green]Found in cache[/green]")
            else:
                 console.print(f"  Status: [yellow]Not found in cache (will be downloaded)[/yellow]")

            if other_cached:
                console.print(f"  Other cached models available: {', '.join(other_cached)}")
        else:
            console.print("  [yellow]Cache status unknown (or cache empty/inaccessible)[/yellow]")
        # --- End Model Selection Info ---

        # --- Handle --check-only flag ---
        if args.check_only:
            console.print("\n[bold cyan]--check-only specified. Exiting now.[/bold cyan]")
            sys.exit(0)
        # --- End --check-only handling ---

        # Check permissions first
        logger.debug("Checking permissions...") # Keep useful debug logs
        if not check_permissions():
            logger.warning("Permission check failed.")
            return 1
        logger.debug("Permission check passed.")

        # Update configuration after first run
        logger.debug("Checking if first run...")
        if is_first_run():
            logger.info("First run detected, marking complete.")
            mark_first_run_complete()
        logger.debug("First run check complete.")

        # Save the selected model as preferred
        set_preferred_model(model_type)
        
        # Pass transcription_queue to AudioManager
        logger.debug("Initializing AudioManager...")
        audio_manager = AudioManager(transcription_queue=transcription_queue, debug_mode=DEBUG_MODE)
        
        # Print startup info based on debug mode
        print_startup_info()
        
        # --- Initialize Keyboard Listener BEFORE Model Loading ---
        logger.info("Initializing keyboard listener BEFORE model...")
        keyboard_manager = KeyboardShortcutManager()
        keyboard_manager.register_triple_ctrl_tap(on_activate)
        keyboard_manager.register_shortcut('<alt>+<esc>', exit_app)
        logger.debug("Keyboard shortcuts registered. Listener thread will start later.")
        
        # Remove diagnostic step comment
        restore_environment_variables(saved_env_vars)
        
        # Load model immediately on startup
        logger.debug("Calling get_model()...")
        # get_model() will use the resolved global model_type 
        stt_model = get_model() 
        logger.debug(f"get_model() returned: {type(stt_model)}") # Log type
        if not stt_model:
             console.print("[bold red]Failed to load STT model. Exiting.[/bold red]")
             return 1 # Exit if model loading failed
        logger.info("Model loaded successfully.")

        # Create a welcome banner (using selected type, actual details later)
        # Remove commented out banner details
        console.print(Panel.fit(
            "[bold cyan]ctrlspeak[/bold cyan] - Ready to transcribe.\nTriple-tap [bold]Ctrl[/bold] to start/stop recording.",
            title="Welcome",
            border_style="blue"
        ))

        # Remove commented out banner update
        # # --- Update Welcome Banner AFTER Model Load ---
        # # Optionally, update the banner if you want more specific info like the exact HF path
        # # This might require modifying how the Panel is stored or re-printing it.
        # # For simplicity, we'll skip re-printing the banner for now. The initial one shows the type.
        # # --- End Banner Update ---

        # --- Start Transcription Worker Thread ---
        logger.info("Starting transcription worker thread...")
        transcription_worker_thread = threading.Thread(
            target=transcription_worker,
            args=(stt_model, transcription_queue, transcribed_chunks),
            daemon=True, # Allow exit even if worker is blocked
            name="TranscriptionWorker"
        )
        transcription_worker_thread.start()
        logger.debug("Transcription worker thread started.")

        # --- Start Keyboard Listeners AFTER Model Loading ---
        logger.info("Starting keyboard listener threads...")
        keyboard_manager.start_listening() # Actually start the threads now
        logger.debug("Keyboard listener threads started.")
        
        # Use context manager for audio stream 
        logger.info("Starting audio stream context...")
        with audio_manager.start_input_stream():
             logger.info("Audio stream started successfully.")
             logger.info("Application ready. Waiting for keyboard events (or Ctrl+C)...")
             
             # Use controlled loop for keep-alive
             logger.debug("Entering main keep-alive loop...")
             while main_loop_active:
                 timer_module.sleep(0.5) # Keep main thread alive but idle
             logger.info("Main keep-alive loop exited. Proceeding to shutdown.")

        logger.info("Exited audio stream context. Main thread proceeding.")
    
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Ctrl+C detected. Shutting down...[/bold yellow]")
        exit_app() # Graceful shutdown via exit_app
    finally:
        # --- Definitive Cleanup --- 
        # Remove shutdown log comment
        logger.info("Executing main finally block for cleanup...")
        
        # 1. Ensure keyboard listener is stopped
        if 'keyboard_manager' in globals() and keyboard_manager is not None:
             try:
                 keyboard_manager.stop_listening()
                 logger.debug("Finally: Keyboard listener stopped.")
             except Exception as e_kb:
                  logger.error(f"Error stopping keyboard manager in finally: {e_kb}")

        # 2. Ensure Audio Manager is stopped completely
        logger.debug("Finally: Stopping audio manager...")
        if audio_manager:
            if audio_manager.is_collecting:
                 logger.warning("Audio was still collecting in finally block. Stopping recording.")
                 try:
                      audio_manager.stop_recording() # Queue final audio if needed
                 except Exception as e_aud_stop:
                      logger.error(f"Error stopping recording in finally: {e_aud_stop}")
            try:
                audio_manager.set_is_running(False) # Signal any internal loops
                logger.debug("Finally: Audio manager set_is_running(False).")
            except Exception as e_aud_run:
                 logger.error(f"Error setting audio manager not running in finally: {e_aud_run}")
        
        # 3. Signal (if needed) and Join Worker Thread
        logger.debug("Finally: Signaling worker thread with None...")
        transcription_queue.put(None) 
        logger.debug("Finally: None sentinel placed on queue.")
        
        if transcription_worker_thread and transcription_worker_thread.is_alive():
            # Remove shutdown log comment
            logger.info("Finally: Waiting for transcription worker thread to join...")
            transcription_worker_thread.join(timeout=3.0) 
            # Remove shutdown log comment
            if transcription_worker_thread.is_alive():
                logger.warning("Finally: Transcription worker thread did NOT join after timeout.")
            else:
                 logger.info("Finally: Transcription worker thread joined successfully.")
        elif transcription_worker_thread:
             logger.debug("Finally: Transcription worker thread was already stopped.")
        else:
             logger.debug("Finally: No transcription worker thread object found.")

        # 4. Restore environment variables
        if 'saved_env_vars' in locals(): # Check if it was defined
             logger.debug("Finally: Restoring environment variables.")
             restore_environment_variables(saved_env_vars)

        console.print("[bold green]ctrlSPEAK stopped.[/bold green]")
        # Allow natural exit instead of os._exit()
        # Phase 5 Shutdown Fix: Explicitly exit if main thread reaches here
        # Only exit if not handling an exception that already caused an exit
        if 'args' in locals() and not args.check_only: # Avoid double exit if check_only was used
             sys.exit(0)

if __name__ == "__main__":
    # Remove commented out test call
    # # --- Phase 0 Test Call ---
    # # Uncomment the line below to run the test *before* the main application logic
    # # test_async_infra()
    # # --- End Phase 0 Test Call ---

    main() 