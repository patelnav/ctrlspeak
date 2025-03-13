"""
Parakeet model implementation for speech-to-text.
"""
import time
import nemo.collections.asr as nemo_asr
from models.base_model import BaseSTTModel
import torch
import logging
import os
from typing import List

# Configure logging
logger = logging.getLogger("parakeet_model")

class ParakeetModel(BaseSTTModel):
    """Parakeet TDT 1.1B model for speech-to-text."""
    
    def __init__(self, model_name="nvidia/parakeet-tdt-1.1b", device=None, verbose=False):
        """Initialize the Parakeet model.
        
        Args:
            model_name: The name of the pretrained model to load.
            device: The device to run the model on (handled automatically by NeMo).
            verbose: Whether to enable verbose logging.
        """
        super().__init__(device)
        self.model_name = model_name
        self.model = None
        self.verbose = verbose
        
        if self.verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
    
    def load_model(self):
        """Load the model from the pretrained checkpoint."""
        if self.model is not None:
            return self.model
        
        logger.info(f"Loading {self.model_name}...")
        start_time = time.time()
        
        try:
            # Load model using NeMo's built-in functionality
            logger.debug("Initializing EncDecRNNTBPEModel...")
            self.model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(self.model_name)
            
            load_time = time.time() - start_time
            logger.info(f"Model loaded in {load_time:.2f} seconds")
            return self.model
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def transcribe_batch(self, audio_paths: List[str]) -> List[str]:
        """Transcribe multiple audio files in batch.
        
        Args:
            audio_paths: List of paths to audio files.
            
        Returns:
            List of clean string transcriptions.
        """
        if self.model is None:
            self.load_model()
        
        if not audio_paths:
            return []
        
        try:
            start_time = time.time()
            
            # Log audio file details
            for i, path in enumerate(audio_paths):
                if os.path.exists(path):
                    file_size = os.path.getsize(path) / 1024
                    logger.debug(f"Audio file {i+1}: {path} ({file_size:.2f} KB)")
                else:
                    logger.warning(f"Audio file {i+1}: {path} does not exist")
            
            # Transcribe using NeMo's direct transcribe method
            logger.debug("Starting transcription...")
            transcribe_start = time.time()
            
            # Let NeMo handle the transcription
            raw_transcriptions = self.model.transcribe(audio_paths)
            
            # Clean up each result
            transcriptions = [self._clean_text(text) for text in raw_transcriptions]
            
            # Log results if in debug mode
            if self.verbose:
                logger.debug("=== Transcription Results ===")
                for i, text in enumerate(transcriptions):
                    logger.debug(f"Transcription {i+1}: {text}")
            
            end_time = time.time()
            logger.info(f"Transcription completed in {end_time - start_time:.2f} seconds")
            
            return transcriptions
            
        except Exception as e:
            logger.error(f"Error during transcription: {str(e)}")
            raise
    
    @property
    def name(self):
        """Return the name of the model."""
        return f"Parakeet-{self.model_name.split('/')[-1]}" 