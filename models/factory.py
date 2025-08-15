"""
Factory for creating speech-to-text models.
"""
import logging
import importlib.util
import platform
import sys

# Configure logging
logger = logging.getLogger("model_factory")

class ModelFactory:
    """Factory for creating speech-to-text models."""
    
    # Mapping from user-friendly aliases to specific model identifiers
    _DEFAULT_ALIASES = {
        "parakeet": "nvidia/parakeet-tdt-0.6b-v3",
        "parakeet-mlx": "mlx-community/parakeet-tdt-0.6b-v2",
        "canary": "nvidia/canary-1b-flash",
        "canary-180m": "nvidia/canary-180m-flash",
        "canary-v2": "nvidia/canary-1b-v2",
        "whisper": "openai/whisper-large-v3"
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
        
        # Check for MLX model
        if "mlx" in model_type:
            if sys.platform != "darwin" or platform.machine() != "arm64":
                logger.error("MLX models are only supported on Apple Silicon (macOS arm64).")
                raise ValueError("MLX models are only supported on Apple Silicon (macOS arm64).")
            
            try:
                from models.parakeet_mlx import ParakeetMLXModel
                logger.debug("Initializing ParakeetMLXModel")
                return ParakeetMLXModel(model_name=model_type, **kwargs)
            except ImportError:
                logger.error("MLX dependencies not found. Please install them using:\n"
                             "uv pip install -r requirements-mlx.txt")
                raise ImportError("MLX dependencies not found. Please install them using:\n"
                                  "uv pip install -r requirements-mlx.txt")

        # Configure logging
        if verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
        
        logger.debug(f"Creating model of type: {model_type}")
        
        # Include verbose parameter in kwargs
        kwargs['verbose'] = verbose
        
        if "canary" in model_type:
            from models.canary import CanaryModel
            logger.debug(f"Initializing CanaryModel with {model_type}")
            return CanaryModel(model_name=model_type, **kwargs)
        elif "parakeet" in model_type:
            from models.parakeet import ParakeetModel
            logger.debug(f"Initializing ParakeetModel with {model_type}")
            return ParakeetModel(model_name=model_type, **kwargs)
        elif "whisper" in model_type:
            logger.debug(f"Initializing WhisperModel with {model_type}")
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
                return WhisperModel(model_name=model_type, **kwargs)
            except ImportError as e:
                raise ImportError(
                    "Failed to import Whisper model. Please install dependencies using:\n"
                    "pip install -r requirements-whisper.txt"
                ) from e
        else:
            logger.error(f"Unsupported model type: {model_type}")
            raise ValueError(f"Unsupported model type: {model_type}")