"""
Factory for creating speech-to-text models.
"""
import logging
import importlib.util
import platform
import sys

# Configure logging
logger = logging.getLogger("model_factory")

from models.registry import MODEL_REGISTRY, get_model_metadata

class ModelFactory:
    """Factory for creating speech-to-text models."""

    # Backward compatibility mapping
    _DEFAULT_ALIASES = {alias: meta.repo_id for alias, meta in MODEL_REGISTRY.items()}

    @classmethod
    def resolve_model_alias(cls, model_name: str) -> str:
        """Resolves a potential model alias to its specific model name."""
        name_lower = model_name.lower()
        meta = get_model_metadata(name_lower)
        if meta:
            resolved_name = meta.alias
            if resolved_name != model_name:
                logger.info(f"Resolved model string '{model_name}' to canonical alias '{resolved_name}' ({meta.repo_id}).")
            return resolved_name
        return model_name

    @staticmethod
    def get_model(model_type, verbose=False, **kwargs):
        """Get a speech-to-text model based on the backend defined in registry."""
        model_type = model_type.lower()
        meta = get_model_metadata(model_type)
        
        if not meta:
            logger.error(f"Unsupported model type: {model_type}")
            raise ValueError(f"Unsupported model type: {model_type}")
            
        backend = meta.backend
        
        # Configure logging
        if verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
            
        logger.debug(f"Creating model of type: {meta.repo_id} using backend {backend}")
        kwargs['verbose'] = verbose
        
        if backend == "mlx_parakeet":
            if sys.platform != "darwin" or platform.machine() != "arm64":
                logger.error("MLX models are only supported on Apple Silicon (macOS arm64).")
                raise ValueError("MLX models are only supported on Apple Silicon (macOS arm64).")
            try:
                from models.parakeet_mlx import ParakeetMLXModel
                logger.debug("Initializing ParakeetMLXModel")
                return ParakeetMLXModel(model_name=meta.repo_id, **kwargs)
            except ImportError:
                raise ImportError("MLX dependencies not found. Please install them using:\n"
                                  "uv pip install -r requirements-mlx.txt")
                                  
        elif backend == "nemo_nemotron":
            try:
                import nemo.collections.asr as nemo_asr
            except ImportError:
                raise ImportError("Nemotron models require nemo-toolkit.")
            from models.nemotron import NemotronModel
            return NemotronModel(model_name=meta.repo_id, **kwargs)
            
        elif backend == "nemo_canary":
            try:
                import nemo.collections.asr as nemo_asr
            except ImportError:
                raise ImportError("Canary models require nemo-toolkit.")
            from models.canary import CanaryModel
            return CanaryModel(model_name=meta.repo_id, **kwargs)
            
        elif backend == "nemo_parakeet":
            try:
                import nemo.collections.asr as nemo_asr
            except ImportError:
                raise ImportError("NVIDIA Parakeet models require nemo-toolkit.")
            from models.parakeet import ParakeetModel
            return ParakeetModel(model_name=meta.repo_id, **kwargs)
            
        elif backend == "transformers_whisper":
            if importlib.util.find_spec("transformers") is None:
                raise ImportError("Whisper models require transformers.")
            try:
                from models.whisper import WhisperModel
                return WhisperModel(model_name=meta.repo_id, **kwargs)
            except ImportError as e:
                raise ImportError("Failed to import Whisper model.") from e
                
        elif backend == "transformers_cohere":
            if importlib.util.find_spec("transformers") is None:
                raise ImportError("Cohere models require transformers.")
            try:
                from models.cohere import CohereModel
                return CohereModel(model_name=meta.repo_id, **kwargs)
            except ImportError as e:
                raise ImportError("Failed to import Cohere model.") from e
                
        elif backend == "mlx_cohere":
            if sys.platform != "darwin" or platform.machine() != "arm64":
                logger.error("MLX models are only supported on Apple Silicon (macOS arm64).")
                raise ValueError("MLX models are only supported on Apple Silicon (macOS arm64).")
            try:
                from models.cohere_mlx import CohereMLXModel
                return CohereMLXModel(model_name=meta.repo_id, **kwargs)
            except ImportError as e:
                raise ImportError("Failed to import Cohere MLX model.") from e
                
        else:
            logger.error(f"Unsupported backend: {backend}")
            raise ValueError(f"Unsupported backend: {backend}")