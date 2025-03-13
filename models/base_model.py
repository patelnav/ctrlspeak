"""
Base model interface for speech-to-text models.
"""
from abc import ABC, abstractmethod
import torch
import logging
from typing import List, Optional, Union, Any

# Configure base logger
logger = logging.getLogger("base_stt_model")

class BaseSTTModel(ABC):
    """Base class for all speech-to-text models."""
    
    def __init__(self, device=None, verbose=False):
        """Initialize the model.
        
        Args:
            device: The device to run the model on. If None, will use MPS if available, otherwise CPU.
            verbose: Whether to enable verbose logging.
        """
        # Set device
        if device is None:
            self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        else:
            self.device = device
        
        self.model = None
        self.verbose = verbose
        
        # Configure logging
        if self.verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
        
        logger.debug(f"Initialized {self.__class__.__name__} with device {self.device}")
    
    @abstractmethod
    def load_model(self):
        """Load the model from the pretrained checkpoint."""
        pass
    
    def transcribe(self, audio_path: str) -> str:
        """Transcribe a single audio file to text.
        
        This is the main API method that should be used by applications.
        
        Args:
            audio_path: Path to the audio file to transcribe.
            
        Returns:
            A clean string transcription.
        """
        if not audio_path:
            logger.warning("No audio path provided")
            return ""
            
        results = self.transcribe_batch([audio_path])
        
        if not results:
            logger.warning("No transcription returned from model")
            return ""
            
        return results[0]
    
    def transcribe_batch(self, audio_paths: List[str]) -> List[str]:
        """Transcribe multiple audio files in batch.
        
        Args:
            audio_paths: List of paths to audio files.
            
        Returns:
            List of clean string transcriptions.
        """
        raise NotImplementedError(
            "Child classes must implement either transcribe_batch or override transcribe"
        )
    
    def _clean_text(self, text: Any) -> str:
        """Internal method to clean text output.
        
        Args:
            text: Raw text to clean, could be string, list, or other format.
            
        Returns:
            Cleaned string.
        """
        if text is None:
            return ""
            
        # Handle list/array type outputs
        if isinstance(text, list):
            if not text:  # Empty list
                return ""
                
            # Process each item in the list, recursively if needed
            cleaned_items = [self._clean_text(item) for item in text]
            # Filter out empty strings
            cleaned_items = [item for item in cleaned_items if item]
            
            if not cleaned_items:
                return ""
                
            # Join multiple items with newlines
            return "\n".join(cleaned_items)
            
        # Handle dict outputs (like from Whisper)
        if isinstance(text, dict) and "text" in text:
            return self._clean_text(text["text"])
            
        # Ensure string type and strip whitespace
        return str(text).strip()
    
    @property
    def name(self):
        """Return the name of the model."""
        return self.__class__.__name__
    
    def __str__(self):
        return f"{self.name} (device: {self.device})" 