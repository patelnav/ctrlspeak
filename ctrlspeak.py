#!/usr/bin/env python3
"""
ctrlspeak - A speech-to-text utility that runs in the background.
Triple-tap Ctrl to start/stop recording.
"""
import sys
import os
import time
import threading
import logging
from rich.console import Console
from rich.panel import Panel

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cli import parse_args_only
from models.factory import ModelFactory
from utils.config import is_first_run, mark_first_run_complete, set_preferred_model

import state
from state import console, MODEL_CACHE_MAP
from logging_config import setup_logging, setup_logging_for_mode
from environment import save_environment_variables, restore_environment_variables
from permissions import check_permissions

logger = logging.getLogger("ctrlspeak")


def find_cached_models():
    """Scans the Hugging Face cache directory for known ctrlspeak models."""
    cached = set()
    try:
        from huggingface_hub import constants as hf_constants
        from pathlib import Path
        cache_dir = Path(hf_constants.HF_HUB_CACHE)
        logger.info(f"Checking Hugging Face cache at: {cache_dir}")

        if not cache_dir.is_dir():
            logger.warning(f"Hugging Face cache directory not found: {cache_dir}")
            return cached

        for item in cache_dir.iterdir():
            if item.is_dir() and item.name in MODEL_CACHE_MAP:
                cached.add(MODEL_CACHE_MAP[item.name])
                logger.debug(f"Found cached model directory: {item.name} -> {MODEL_CACHE_MAP[item.name]}")

    except Exception as e:
        logger.error(f"Error scanning Hugging Face cache: {e}", exc_info=state.DEBUG_MODE)
        console.print(f"[yellow]Warning: Could not scan Hugging Face cache ({e})[/yellow]")

    logger.info(f"Found cached models: {cached}")
    return cached


def run_app(args):
    """Main application entry point"""
    # Lazy imports to avoid heavy deps during --list-models
    from utils.keyboard_shortcuts import KeyboardShortcutManager
    from utils.audio import AudioManager
    from model_loader import get_model
    from transcription import transcription_worker
    from hotkeys import on_activate
    import torch

    state.startup_time = time.time()
    setup_logging()

    saved_env_vars = save_environment_variables()

    try:
        if not check_permissions():
            logger.warning("Permission check failed.")
            return 1

        state.DEBUG_MODE = args.debug
        model_type_arg = args.model

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
            other_cached = sorted(list(cached_models - {ModelFactory.resolve_model_alias(state.model_type)}))
            if state.model_type in cached_models:
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

        state.audio_manager = AudioManager(
            transcription_queue=state.transcription_queue, debug_mode=state.DEBUG_MODE
        )

        state.keyboard_manager = KeyboardShortcutManager()
        state.keyboard_manager.register_triple_ctrl_tap(on_activate)
        state.keyboard_manager.register_shortcut('<alt>+<esc>', exit_app)

        restore_environment_variables(saved_env_vars)

        state.stt_model = get_model()
        if not state.stt_model:
            console.print("[bold red]Failed to load STT model. Exiting.[/bold red]")
            return 1

        console.print(
            Panel.fit(
                "[bold cyan]ctrlspeak[/bold cyan] - Ready to transcribe.\nTriple-tap [bold]Ctrl[/bold] to start/stop recording.",
                title="Welcome",
                border_style="blue",
            )
        )

        state.transcription_worker_thread = threading.Thread(
            target=transcription_worker,
            args=(state.stt_model, state.transcription_queue, state.transcribed_chunks),
            daemon=True,
            name="TranscriptionWorker",
        )
        state.transcription_worker_thread.start()

        state.keyboard_manager.start_listening()

        with state.audio_manager.start_input_stream():
            logger.info("Application ready. Waiting for keyboard events (or Ctrl+C)...")

            while state.main_loop_active:
                time.sleep(0.5)

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

    if args.list_models:
        console = Console()
        console.print("\n[bold]Supported Models:[/bold]")
        for alias, model_name in ModelFactory._DEFAULT_ALIASES.items():
            console.print(f"  - [cyan]{alias}[/cyan]: {model_name}")
        sys.exit(0)

    run_app(args)


if __name__ == "__main__":
    main()