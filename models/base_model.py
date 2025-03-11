"""
Base model interface for speech-to-text models.
"""
from abc import ABC, abstractmethod
import torch
import logging

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
    
    @abstractmethod
    def transcribe(self, audio_paths):
        """Transcribe audio files.
        
        Args:
            audio_paths: List of paths to audio files.
            
        Returns:
            List of transcriptions.
        """
        pass
    
    @property
    def name(self):
        """Return the name of the model."""
        return self.__class__.__name__
    
    def __str__(self):
        return f"{self.name} (device: {self.device})" 