#!/usr/bin/env python3
"""
ctrlspeak - A speech-to-text utility that runs in the background.
Triple-tap Ctrl to start/stop recording.
"""
import sys
import os
from utils.tqdm_lock import ensure_tqdm_thread_lock  # lightweight import

# Apply immediately at import time to preempt tqdm's default lock behavior
ensure_tqdm_thread_lock()

import time
import threading
import logging
from rich.console import Console
from rich.panel import Panel

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cli import parse_args_only
from utils.config import is_first_run, mark_first_run_complete, set_preferred_model

import state
from state import console, KNOWN_MODELS
from logging_config import setup_logging, setup_logging_for_mode
from environment import save_environment_variables, restore_environment_variables
from permissions import check_permissions

logger = logging.getLogger("ctrlspeak")


def find_cached_models():
    """Scans the Hugging Face cache directory for known ctrlspeak models."""
    cached = set()
    try:
        from huggingface_hub import scan_cache_dir

        cache_info = scan_cache_dir()
        logger.info(f"Scanning Hugging Face cache ({cache_info.size_on_disk_str} total)")

        for repo in cache_info.repos:
            # Only check models (not datasets/spaces) that are in our known set
            if repo.repo_type == "model" and repo.repo_id in KNOWN_MODELS:
                cached.add(repo.repo_id)
                logger.debug(f"Found cached model: {repo.repo_id}")

    except Exception as e:
        logger.error(f"Error scanning Hugging Face cache: {e}", exc_info=state.DEBUG_MODE)
        console.print(f"[yellow]Warning: Could not scan Hugging Face cache ({e})[/yellow]")

    logger.info(f"Found cached models: {cached}")
    return cached


def run_app(args):
    """Run application with Textual UI"""
    import threading
    import torch
    from models.factory import ModelFactory
    from models.registry import get_model_metadata
    from utils.keyboard_shortcuts import KeyboardShortcutManager
    from utils.audio import AudioManager
    from model_loader import get_model
    from transcription import transcription_worker
    from hotkeys import on_activate
    from ui import CtrlSpeakApp, AppState

    state.startup_time = time.time()
    setup_logging()

    saved_env_vars = save_environment_variables()

    try:
        if not check_permissions():
            logger.warning("Permission check failed.")
            return 1

        state.DEBUG_MODE = args.debug
        model_type_arg = args.model
        state.source_lang = args.source_lang
        state.target_lang = args.target_lang

        # History configuration
        state.history_enabled = not args.no_history
        if args.history_db:
            from pathlib import Path
            state.history_db_path = Path(args.history_db)

        state.model_type = ModelFactory.resolve_model_alias(model_type_arg)

        state.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        logger.debug(f"Using device: {state.device}")

        setup_logging_for_mode(state.DEBUG_MODE)

        cached_models = find_cached_models()

        console.print("\n[bold]Model Configuration:[/bold]")
        if model_type_arg.lower() != state.model_type.lower():
            console.print(
                f"  Selected (alias): [cyan]{model_type_arg}[/cyan] -> Resolved: [cyan]{ModelFactory.resolve_model_alias(state.model_type)}[/cyan]"
            )
        else:
            console.print(f"  Selected: [cyan]{ModelFactory.resolve_model_alias(state.model_type)}[/cyan]")

        if cached_models:
            selected_meta = get_model_metadata(state.model_type)
            selected_repo_id = selected_meta.repo_id if selected_meta else state.model_type
            other_cached = sorted(list(cached_models - {selected_repo_id}))
            if selected_repo_id in cached_models:
                console.print(f"  Status: [green]Found in cache[/green]")
            else:
                console.print(f"  Status: [yellow]Not found in cache (will be downloaded)[/yellow]")

            if other_cached:
                console.print(f"  Other cached models available: {', '.join(other_cached)}")
        else:
            console.print("  [yellow]Cache status unknown (or cache empty/inaccessible)[/yellow]")

        if args.check_only:
            console.print("\n[bold cyan]--check-only specified. Exiting now.[/bold cyan]")
            sys.exit(0)

        if is_first_run():
            mark_first_run_complete()

        set_preferred_model(state.model_type)

        # Create app state for Textual UI
        app_state = AppState()
        app_state.selected_model = model_type_arg  # Store the alias, not the full name
        state.app_state_ref = app_state  # Store reference for hotkeys to access

        state.audio_manager = AudioManager(
            transcription_queue=state.transcription_queue,
            debug_mode=state.DEBUG_MODE,
            app_state=app_state
        )

        state.keyboard_manager = KeyboardShortcutManager()
        state.keyboard_manager.register_triple_ctrl_tap(on_activate)
        state.keyboard_manager.register_shortcut('<alt>+<esc>', exit_app)

        restore_environment_variables(saved_env_vars)

        try:
            state.stt_model = get_model()
        except Exception as e:
            console.print(f"[bold red]Failed to load STT model: {e}[/bold red]")
            return 1

        # Sync loaded model state after successful load
        app_state.loaded_model = model_type_arg  # Store the alias that was actually loaded

        console.print(
            Panel.fit(
                "[bold cyan]ctrlspeak[/bold cyan] - Ready to transcribe.\nTriple-tap [bold]Ctrl[/bold] to start/stop recording.",
                title="Welcome",
                border_style="blue",
            )
        )

        state.transcription_worker_thread = threading.Thread(
            target=transcription_worker,
            args=(state.stt_model, state.transcription_queue, state.transcribed_chunks, state.source_lang, state.target_lang),
            daemon=True,
            name="TranscriptionWorker",
        )
        state.transcription_worker_thread.start()

        state.keyboard_manager.start_listening()

        # Start audio stream
        with state.audio_manager.start_input_stream():
            logger.info("Starting Textual UI...")

            # Sync loaded device state after stream starts
            # If input_device is None, resolve to actual default device ID
            if state.audio_manager.input_device is None:
                import sounddevice as sd
                app_state.loaded_device = sd.default.device[0] if sd.default.device else None
            else:
                app_state.loaded_device = state.audio_manager.input_device

            # Create and run Textual app
            app = CtrlSpeakApp(
                app_state=app_state,
                audio_manager=state.audio_manager,
                model_type=model_type_arg  # Pass the alias, not the resolved full name
            )

            # Run the app (this blocks until app exits)
            app.run()

            logger.info("Textual UI exited, cleaning up...")

    except KeyboardInterrupt:
        console.print("\n[bold yellow]Ctrl+C detected. Shutting down...[/bold yellow]")
        exit_app()
    finally:
        logger.info("Executing main finally block for cleanup...")

        if state.keyboard_manager is not None:
            try:
                state.keyboard_manager.stop_listening()
            except Exception as e_kb:
                logger.error(f"Error stopping keyboard manager in finally: {e_kb}")

        if state.audio_manager:
            if state.audio_manager.is_collecting:
                try:
                    state.audio_manager.stop_recording()
                except Exception as e_aud_stop:
                    logger.error(f"Error stopping recording in finally: {e_aud_stop}")
            try:
                state.audio_manager.set_is_running(False)
            except Exception as e_aud_run:
                logger.error(f"Error setting audio manager not running in finally: {e_aud_run}")

        state.transcription_queue.put(None)

        if state.transcription_worker_thread and state.transcription_worker_thread.is_alive():
            state.transcription_worker_thread.join(timeout=3.0)
            if state.transcription_worker_thread.is_alive():
                logger.warning("Finally: Transcription worker thread did NOT join after timeout.")

        if 'saved_env_vars' in locals():
            restore_environment_variables(saved_env_vars)

        console.print("[bold green]ctrlspeak stopped.[/bold green]")
        if 'args' in locals() and not args.check_only:
            sys.exit(0)


def exit_app():
    """Initiates the application shutdown sequence."""
    logger.info("Shutdown requested.")
    console.print("[bold yellow]Shutting down ctrlspeak...")

    if state.audio_manager and state.audio_manager.is_collecting:
        logger.info("Stopping active recording during exit...")
        state.audio_manager.stop_recording()

    logger.info("Signaling transcription worker to exit...")
    state.transcription_queue.put(None)

    if state.keyboard_manager is not None:
        logger.info("Stopping keyboard listener...")
        try:
            state.keyboard_manager.stop_listening()
            logger.info("Keyboard listener stop signaled.")
        except Exception as e_stop_kb:
            logger.error(f"Error stopping keyboard listener in exit_app: {e_stop_kb}")

    logger.info("Signaling main loop to exit...")
    state.main_loop_active = False

    logger.info("Exit_app finished signaling components.")


def main():
    """Main application entry point"""
    args = parse_args_only()

    if args.check_compatibility:
        from models.compatibility import CompatibilityChecker
        CompatibilityChecker.print_report()
        sys.exit(0)

    if args.list_models:
        from models.registry import MODEL_REGISTRY
        console = Console()
        console.print("\n[bold]Supported Models:[/bold]")
        
        # Check what dependencies are available
        nemo_available = False
        transformers_available = False
        mlx_available = False
        
        try:
            import nemo.collections.asr as nemo_asr
            nemo_available = True
        except ImportError:
            pass
            
        try:
            import transformers
            transformers_available = True
        except ImportError:
            pass
            
        try:
            import mlx
            mlx_available = True
        except ImportError:
            pass
        
        for alias, meta in MODEL_REGISTRY.items():
            status = ""
            note = ""
            
            if "mlx" in meta.requires:
                if mlx_available:
                    status = " [green]✓ Available[/green]"
                    note = " (Apple Silicon / MLX)"
                else:
                    status = " [red]✗ Requires MLX[/red]"
                    note = " (Apple Silicon / MLX - install MLX dependencies)"
            elif "nemo" in meta.requires:
                if nemo_available:
                    status = " [green]✓ Available[/green]"
                else:
                    status = " [red]✗ Requires NVIDIA support[/red]"
                    note = " (install with: brew reinstall ctrlspeak --with-nvidia)"
            elif "transformers" in meta.requires:
                if transformers_available:
                    status = " [green]✓ Available[/green]"
                else:
                    status = " [red]✗ Requires transformers[/red]"
                    note = " (install with: uv pip install -r requirements-whisper.txt)"
            elif meta.requires == "cohere-mlx":
                if mlx_available:
                    status = " [green]✓ Available[/green]"
                    note = " (Apple Silicon / MLX, ~4.1GB download)"
                else:
                    status = " [red]✗ Requires MLX[/red]"
                    note = " (install with: uv pip install -r requirements-cohere-mlx.txt)"
            elif meta.requires == "cohere":
                if transformers_available:
                    status = " [green]✓ Available[/green]"
                    note = " (CPU-first, ~4.1GB download, multilingual only)"
                else:
                    status = " [red]✗ Requires transformers[/red]"
                    note = " (install with: uv pip install -r requirements-cohere.txt)"
            else:
                status = " [green]✓ Available[/green]"
                
            console.print(f"  - [cyan]{alias}[/cyan]: {meta.repo_id}{note}{status}")
        
        # Show installation recommendations
        if not nemo_available:
            console.print(f"\n[yellow]💡 Tip:[/yellow] Install NVIDIA model support with:")
            console.print(f"  [cyan]brew reinstall ctrlspeak --with-nvidia[/cyan]")
        
        if not transformers_available:
            console.print(f"\n[yellow]💡 Tip:[/yellow] Install Whisper model support with:")
            console.print(f"  [cyan]uv pip install -r requirements-whisper.txt[/cyan]")
            
        sys.exit(0)

    # Run the Textual UI application
    run_app(args)


if __name__ == "__main__":
    main()
