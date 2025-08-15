"""
Parakeet MLX model implementation for speech-to-text on Apple Silicon.
"""
import time
import logging
from typing import List

from models.base_model import BaseSTTModel

# Configure logging
logger = logging.getLogger("parakeet_mlx_model")

class ParakeetMLXModel(BaseSTTModel):
    """Parakeet MLX model for speech-to-text on Apple Silicon."""

    def __init__(self, model_name="mlx-community/parakeet-tdt-0.6b-v2", device=None, verbose=False):
        """Initialize the Parakeet MLX model."""
        super().__init__(device=None, verbose=verbose)  # MLX handles device automatically
        self.model_name = model_name
        self.model = None

    def load_model(self):
        """Load the model from the pretrained checkpoint."""
        if self.model is not None:
            return self.model
        
        logger.info(f"Loading {self.model_name}...")
        start_time = time.time()
        
        try:
            from parakeet_mlx import from_pretrained
            self.model = from_pretrained(self.model_name)
            load_time = time.time() - start_time
            logger.info(f"Model loaded in {load_time:.2f} seconds")
            return self.model
        except ImportError:
            logger.error("parakeet_mlx or mlx not installed. Please run: pip install -r requirements-mlx.txt")
            raise
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def transcribe_batch(self, audio_paths: List[str], **kwargs) -> List[str]:
        """Transcribe multiple audio files in batch."""
        # The `parakeet-mlx` model does not support language arguments,
        # so we accept and ignore them for compatibility with the transcription worker.
        _ = kwargs.get("source_lang")
        _ = kwargs.get("target_lang")

        if self.model is None:
            self.load_model()
        
        if not audio_paths:
            return []
        
        try:
            transcriptions = []
            for audio_path in audio_paths:
                result = self.model.transcribe(audio_path)
                transcriptions.append(self._clean_text(result.text))
            
            return transcriptions
        except Exception as e:
            logger.error(f"Error during transcription: {str(e)}")
            raise

    def name(self):
        """Return the name of the model."""
        simple_name = self.model_name.split('/')[-1]
        return f"ParakeetMLX-{simple_name}"
