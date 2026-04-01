import logging
import torch
from typing import List
from pathlib import Path
from models.base_model import BaseSTTModel
from models.cohere_mlx_model import Model

logger = logging.getLogger("cohere_mlx_backend")

class CohereMLXModel(BaseSTTModel):
    SUPPORTED_LANGUAGES = {
        "ar", "de", "el", "en", "es", "fr", "it", 
        "ja", "ko", "nl", "pl", "pt", "vi", "zh"
    }

    def __init__(self, model_name="CohereLabs/cohere-transcribe-03-2026", device=None, verbose=False, **kwargs):
        # Force MPS device usage since this is the specific Apple Silicon MLX backend
        # Note: MLX operations default to the globally set device, usually GPU on Apple Silicon.
        super().__init__(device=torch.device("mps"), verbose=verbose)
        self.model_name = model_name
        self.backend_model = None

    def load_model(self):
        try:
            import mlx.core as mx
            from huggingface_hub import snapshot_download
        except ImportError:
            raise ImportError(
                "Cohere MLX models require mlx, safetensors, and huggingface_hub. "
                "Please install with:\\nuv pip install -e '.[cohere-mlx]'"
            )
            
        logger.info(f"Downloading/Locating Cohere MLX model weights from {self.model_name}...")
        model_path = Path(snapshot_download(repo_id=self.model_name))
        
        logger.info(f"Loading Cohere MLX model from {model_path}...")
        
        import json
        with open(model_path / "config.json", "r") as f:
            config = json.load(f)
            
        self.backend_model = Model(config)
        
        from mlx.utils import tree_unflatten
        from safetensors import safe_open
        
        weights = {}
        for sf_file in model_path.glob("model*.safetensors"):
            with safe_open(str(sf_file), framework="pt") as f:
                for k in f.keys():
                    weights[k] = mx.array(f.get_tensor(k))
                    
        weights = self.backend_model.sanitize(weights)
        self.backend_model.update(tree_unflatten(list(weights.items())))
        
        self.backend_model = Model.post_load_hook(self.backend_model, model_path)
        
        mx.eval(self.backend_model.parameters())
        logger.info("Cohere MLX model loaded successfully.")

    def transcribe_batch(self, audio_paths: List[str], source_lang: str = "en", target_lang: str = "en", **kwargs) -> List[str]:
        if not audio_paths:
            return []
            
        source_lang = source_lang.lower()
        target_lang = target_lang.lower()
        
        if source_lang not in self.SUPPORTED_LANGUAGES:
            raise ValueError(f"Language '{source_lang}' is not supported by Cohere Transcribe. Supported languages: {', '.join(sorted(self.SUPPORTED_LANGUAGES))}")
            
        if source_lang != target_lang:
            raise ValueError("Cohere Transcribe does not support translation. target_lang must match source_lang.")

        if self.backend_model is None:
            raise RuntimeError("Model must be loaded before transcribing.")

        try:
            # The MLX reference Model class processes the list of paths batch-wise natively
            transcriptions = self.backend_model.transcribe(
                language=source_lang,
                audio_files=audio_paths,
                punctuation=True
            )
            
            return [self._clean_text(t) for t in transcriptions]
        except Exception as e:
            logger.error(f"Error transcribing batch with Cohere MLX: {e}")
            # If batch fails due to an obscure error, just return empty results as fallback.
            return [""] * len(audio_paths)
