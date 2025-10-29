
import time
import traceback
import logging
import state
from models.factory import ModelFactory

logger = logging.getLogger("ctrlspeak")

class ModelLoadError(Exception):
    """Exception raised when model loading fails with user-friendly message."""
    pass


def get_model():
    """
    Load model with progress tracking.

    Returns:
        The loaded model instance

    Raises:
        ModelLoadError: If model fails to load, with user-friendly error message
    """

    if state.stt_model is not None:
        return state.stt_model

    state.console.print("\n[bold yellow]Loading model... please wait[/bold yellow]")
    start_time = time.time()

    try:
        logger.info(f"Step 1: Creating {state.model_type} model instance...")
        try:
            state.stt_model = ModelFactory.get_model(model_type=state.model_type, device=state.device, verbose=state.DEBUG_MODE)
            logger.info("Model instance created successfully")
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error creating model instance: {error_msg}")
            state.console.print(f"[bold red]Error creating model instance: {error_msg}[/bold red]")
            if state.DEBUG_MODE:
                import traceback
                traceback.print_exc()
            raise ModelLoadError(error_msg)

        logger.info("Step 2: Loading model weights...")
        try:
            state.stt_model.load_model()
            logger.info("Model weights loaded successfully")
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error loading model weights: {error_msg}")
            state.console.print(f"[bold red]Error loading model weights: {error_msg}[/bold red]")
            if state.DEBUG_MODE:
                import traceback
                traceback.print_exc()
            raise ModelLoadError(error_msg)

        end_time = time.time()
        state.model_loaded = True
        state.console.print(f"[bold green]Model loaded in {end_time - start_time:.2f} seconds. Ready to record![/bold green]")

        return state.stt_model
    except ModelLoadError:
        # Re-raise ModelLoadError as-is
        raise
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Unexpected error in get_model: {error_msg}")
        state.console.print(f"[bold red]Unexpected error in get_model: {error_msg}[/bold red]")
        if state.DEBUG_MODE:
            import traceback
            traceback.print_exc()
        raise ModelLoadError(error_msg)
