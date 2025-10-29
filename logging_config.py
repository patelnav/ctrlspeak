
import logging
import os
import warnings
from pathlib import Path
from rich.logging import RichHandler
import state

class NullHandler(logging.Handler):
    def emit(self, record):
        pass

def _get_log_file():
    """Get the path to the log file, creating parent directories if needed."""
    config_dir = Path.home() / ".config" / "ctrlspeak"
    log_dir = config_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / "ctrlspeak.log"

def setup_logging():
    warnings.filterwarnings("ignore")

    os.environ["NEMO_LOGGING_LEVEL"] = "ERROR"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    class FilterNemoWarnings(logging.Filter):
        def filter(self, record):
            if state.DEBUG_MODE:
                return True
            if "nemo" in record.name.lower():
                return False
            return True

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Set root to DEBUG to capture everything

    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Set up both console and file logging
    console_handler = RichHandler(rich_tracebacks=True, console=state.console)
    console_handler.setLevel(logging.INFO)  # Console shows INFO and above
    handlers = [console_handler]

    # Add file handler for persistent logging
    try:
        log_file = _get_log_file()
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        file_handler.setLevel(logging.DEBUG)
        handlers.append(file_handler)
    except Exception as e:
        # If file logging fails, just log to console
        state.console.print(f"[yellow]Warning: Could not set up file logging: {e}[/yellow]")

    # Add handlers to root logger
    for handler in handlers:
        root_logger.addHandler(handler)

    # Add filter to root
    root_logger.addFilter(FilterNemoWarnings())

    # Configure specific loggers
    for logger_name in ['nemo', 'nemo_logger', 'nemo.collections']:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.CRITICAL)
        logger.addFilter(FilterNemoWarnings())

    # Suppress verbose libraries
    for lib in ['matplotlib', 'numba', 'urllib3', 'nemo', 'nemo_logger',
               'nemo.collections', 'pytorch_lightning', 'filelock',
               'huggingface_hub', 'transformers', 'sound_player']:
        logging.getLogger(lib).setLevel(logging.CRITICAL)
        logging.getLogger(lib).addFilter(FilterNemoWarnings())

def setup_logging_for_mode(debug_mode):
    logger = logging.getLogger("ctrlspeak")
    audio_logger = logging.getLogger("ctrlspeak.audio")

    if debug_mode:
        logger.setLevel(logging.DEBUG)
        audio_logger.setLevel(logging.DEBUG)
        os.environ["NEMO_LOGGING_LEVEL"] = "INFO"
        warnings.filterwarnings("default")
        for lib in ['nemo', 'pytorch_lightning']:
            logging.getLogger(lib).setLevel(logging.INFO)
    else:
        # In normal mode, still show INFO level logs for audio recording
        logger.setLevel(logging.INFO)
        audio_logger.setLevel(logging.INFO)  # Changed from WARNING to INFO to capture audio recording logs
        os.environ["NEMO_LOGGING_LEVEL"] = "CRITICAL"
        for lib in ['nemo', 'nemo_logger', 'nemo.collections', 'pytorch_lightning']:
            logging.getLogger(lib).setLevel(logging.CRITICAL)
        warnings.filterwarnings("ignore")
