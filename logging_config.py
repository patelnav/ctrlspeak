
import logging
import os
import warnings
from rich.logging import RichHandler
import state

class NullHandler(logging.Handler):
    def emit(self, record):
        pass

def setup_logging():
    warnings.filterwarnings("ignore")

    logging.getLogger().addHandler(NullHandler())
    logging.getLogger().setLevel(logging.CRITICAL)

    os.environ["NEMO_LOGGING_LEVEL"] = "ERROR"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    class FilterNemoWarnings(logging.Filter):
        def filter(self, record):
            if state.DEBUG_MODE:
                return True
            if "nemo" in record.name.lower():
                return False
            return True

    logging.getLogger().addFilter(FilterNemoWarnings())

    for logger_name in ['nemo', 'nemo_logger', 'nemo.collections']:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.CRITICAL)
        logger.addFilter(FilterNemoWarnings())

    logging.basicConfig(
        level=logging.WARNING,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, console=state.console)]
    )
    logger = logging.getLogger("ctrlspeak")

    for lib in ['matplotlib', 'numba', 'urllib3', 'nemo', 'nemo_logger', 
               'nemo.collections', 'pytorch_lightning', 'filelock', 
               'huggingface_hub', 'transformers', 'sound_player']:
        logging.getLogger(lib).setLevel(logging.CRITICAL)
        logging.getLogger(lib).addFilter(FilterNemoWarnings())

def setup_logging_for_mode(debug_mode):
    logger = logging.getLogger("ctrlspeak")
    if debug_mode:
        logger.setLevel(logging.DEBUG)
        logging.getLogger("ctrlspeak.audio").setLevel(logging.DEBUG)
        os.environ["NEMO_LOGGING_LEVEL"] = "INFO"
        warnings.filterwarnings("default")
        for lib in ['nemo', 'pytorch_lightning']:
            logging.getLogger(lib).setLevel(logging.INFO)
    else:
        logger.setLevel(logging.INFO)
        logging.getLogger("ctrlspeak.audio").setLevel(logging.WARNING)
        os.environ["NEMO_LOGGING_LEVEL"] = "CRITICAL"
        for lib in ['nemo', 'nemo_logger', 'nemo.collections', 'pytorch_lightning']:
            logging.getLogger(lib).setLevel(logging.CRITICAL)
        warnings.filterwarnings("ignore")
