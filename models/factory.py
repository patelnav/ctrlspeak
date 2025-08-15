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
    
    # Mapping from user-friendly aliases to specific model identifiers
    _DEFAULT_ALIASES = {
        "parakeet": "parakeet-0.6b" # Default parakeet maps to 0.6b
    }

    @classmethod
    def resolve_model_alias(cls, model_name: str) -> str:
        """Resolves a potential model alias to its specific model name."""
        name_lower = model_name.lower()
        resolved_name = cls._DEFAULT_ALIASES.get(name_lower, model_name)
        if resolved_name != model_name:
            logger.info(f"Resolved model alias ''{model_name}'' to ''{resolved_name}''.")
        return resolved_name

    @staticmethod
    def get_model(model_type, verbose=False, **kwargs):
        """Get a speech-to-text model.
        
        Args:
            model_type: The specific type of model to create (aliases should be resolved beforehand).
            verbose: Whether to enable verbose logging.
            **kwargs: Additional arguments to pass to the model constructor.
            
        Returns:
            A speech-to-text model.
            
        Raises:
            ValueError: If the model type is not supported.
            ImportError: If Whisper dependencies are not installed.
        """
        # Alias resolution should happen *before* calling this method.
        # The input model_type is expected to be specific now.
        model_type = model_type.lower()
        
        # Configure logging
        if verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
        
        logger.debug(f"Creating model of type: {model_type}")
        
        # Include verbose parameter in kwargs
        kwargs['verbose'] = verbose
        
        if model_type == "parakeet-0.6b":
            logger.debug("Initializing ParakeetModel with nvidia/parakeet-tdt-0.6b-v3")
            return ParakeetModel(model_name="nvidia/parakeet-tdt-0.6b-v3", **kwargs)
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