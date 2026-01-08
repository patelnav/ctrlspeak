"""
Factory for creating speech-to-text models.
"""
import logging
import importlib.util
import platform
import sys

# Configure logging
logger = logging.getLogger("model_factory")

# Import model identifiers from state.py
from state import (
    MLX_PARAKEET_V3,
    MLX_PARAKEET_V2,
    NVIDIA_PARAKEET_V3,
    NVIDIA_PARAKEET_V2,
    NVIDIA_CANARY_1B_FLASH,
    NVIDIA_CANARY_180M,
    NVIDIA_CANARY_V2,
    NVIDIA_NEMOTRON_STREAMING,
    OPENAI_WHISPER_V3,
)

class ModelFactory:
    """Factory for creating speech-to-text models."""

    # Mapping from user-friendly aliases to specific model identifiers
    _DEFAULT_ALIASES = {
        "parakeet-v3-mlx": MLX_PARAKEET_V3,
        "parakeet-v3": NVIDIA_PARAKEET_V3,
        "parakeet-v2-mlx": MLX_PARAKEET_V2,
        "parakeet-v2": NVIDIA_PARAKEET_V2,
        "parakeet": MLX_PARAKEET_V3,  # Default to v3 MLX
        "canary": NVIDIA_CANARY_1B_FLASH,
        "canary-180m": NVIDIA_CANARY_180M,
        "canary-v2": NVIDIA_CANARY_V2,
        "nemotron": NVIDIA_NEMOTRON_STREAMING,
        "whisper": OPENAI_WHISPER_V3
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
        
        if "nemotron" in model_type:
            # Check if nemo is available for Nemotron models
            try:
                import nemo.collections.asr as nemo_asr
            except ImportError:
                raise ImportError(
                    "Nemotron models require nemo-toolkit. Please install with:\n"
                    "brew reinstall ctrlspeak --with-nvidia"
                )

            from models.nemotron import NemotronModel
            logger.debug(f"Initializing NemotronModel with {model_type}")
            return NemotronModel(model_name=model_type, **kwargs)
        elif "canary" in model_type:
            # Check if nemo is available for Canary models
            try:
                import nemo.collections.asr as nemo_asr
            except ImportError:
                raise ImportError(
                    "Canary models require nemo-toolkit. Please install with:\n"
                    "brew reinstall ctrlspeak --with-nvidia"
                )

            from models.canary import CanaryModel
            logger.debug(f"Initializing CanaryModel with {model_type}")
            return CanaryModel(model_name=model_type, **kwargs)
        elif "parakeet" in model_type:
            # Check if this is an NVIDIA Parakeet model that needs nemo
            if "nvidia" in model_type:
                try:
                    import nemo.collections.asr as nemo_asr
                except ImportError:
                    raise ImportError(
                        "NVIDIA Parakeet models require nemo-toolkit. Please install with:\n"
                        "brew reinstall ctrlspeak --with-nvidia"
                    )
            
            from models.parakeet import ParakeetModel
            logger.debug(f"Initializing ParakeetModel with {model_type}")
            return ParakeetModel(model_name=model_type, **kwargs)
        elif "whisper" in model_type:
            logger.debug(f"Initializing WhisperModel with {model_type}")
            # Check if Whisper dependencies are installed
            if importlib.util.find_spec("transformers") is None:
                raise ImportError(
                    "Whisper models require transformers. Please install with:\n"
                    "brew reinstall ctrlspeak --with-whisper"
                )
            
            # Dynamically import WhisperModel
            try:
                from models.whisper import WhisperModel
                logger.debug("Initializing WhisperModel")
                return WhisperModel(model_name=model_type, **kwargs)
            except ImportError as e:
                raise ImportError(
                    "Failed to import Whisper model. Please install with:\n"
                    "brew reinstall ctrlspeak --with-whisper"
                ) from e
        else:
            logger.error(f"Unsupported model type: {model_type}")
            raise ValueError(f"Unsupported model type: {model_type}")