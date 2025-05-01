"""
Factory for creating speech-to-text models.
"""
from models.parakeet import ParakeetModel
from models.canary import CanaryModel
import logging
import importlib.util
import sys

# Configure logging
logger = logging.getLogger("model_factory")

class ModelFactory:
    """Factory for creating speech-to-text models."""
    
    @staticmethod
    def get_model(model_type, verbose=False, **kwargs):
        """Get a speech-to-text model.
        
        Args:
            model_type: The type of model to create.
            verbose: Whether to enable verbose logging.
            **kwargs: Additional arguments to pass to the model constructor.
            
        Returns:
            A speech-to-text model.
            
        Raises:
            ValueError: If the model type is not supported.
            ImportError: If Whisper dependencies are not installed.
        """
        model_type = model_type.lower()
        
        # Configure logging
        if verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
        
        logger.debug(f"Creating model of type: {model_type}")
        
        # Include verbose parameter in kwargs
        kwargs['verbose'] = verbose
        
        if model_type == "parakeet":
            logger.warning("Using generic 'parakeet' type. Defaulting to nvidia/parakeet-tdt-0.6b-v2.")
            logger.warning("Please use 'parakeet-0.6b' or 'parakeet-1.1b' in the future.")
            logger.debug("Initializing ParakeetModel with nvidia/parakeet-tdt-0.6b-v2")
            return ParakeetModel(model_name="nvidia/parakeet-tdt-0.6b-v2", **kwargs)
        elif model_type == "parakeet-0.6b":
            logger.debug("Initializing ParakeetModel with nvidia/parakeet-tdt-0.6b-v2")
            return ParakeetModel(model_name="nvidia/parakeet-tdt-0.6b-v2", **kwargs)
        elif model_type == "parakeet-1.1b":
            logger.debug("Initializing ParakeetModel with nvidia/parakeet-tdt-1.1b")
            return ParakeetModel(model_name="nvidia/parakeet-tdt-1.1b", **kwargs)
        elif model_type == "canary":
            logger.debug("Initializing CanaryModel")
            return CanaryModel(**kwargs)
        elif model_type == "whisper":
            # Check if Whisper dependencies are installed
            if importlib.util.find_spec("transformers") is None:
                raise ImportError(
                    "Whisper dependencies not found. Please install them using:\n"
                    "pip install -r requirements-whisper.txt"
                )
            
            # Dynamically import WhisperModel
            try:
                from models.whisper import WhisperModel
                logger.debug("Initializing WhisperModel")
                return WhisperModel(**kwargs)
            except ImportError as e:
                raise ImportError(
                    "Failed to import Whisper model. Please install dependencies using:\n"
                    "pip install -r requirements-whisper.txt"
                ) from e
        else:
            logger.error(f"Unsupported model type: {model_type}")
            raise ValueError(f"Unsupported model type: {model_type}") 