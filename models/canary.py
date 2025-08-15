"""
Canary model implementation for speech-to-text.
"""
import time
import nemo.collections.asr as nemo_asr
from models.base_model import BaseSTTModel
import torch
import logging
import os
from typing import List

# Configure logging
logger = logging.getLogger("canary_model")

class CanaryModel(BaseSTTModel):
    """Canary model for multilingual speech-to-text and translation.
    Defaults to 'nvidia/canary-1b-flash' if no model_name is specified."""

    def __init__(self, model_name="nvidia/canary-1b-flash", device=None, verbose=False):
        """Initialize the Canary model.
        
        Args:
            model_name: The name of the pretrained model to load.
            device: The device to run the model on.
            verbose: Whether to enable verbose logging.
        """
        # Setup device with MPS support
        if device is None:
            if torch.cuda.is_available():
                device = torch.device('cuda:0')
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                logger.info("Using MPS device (Apple Silicon GPU)")
                device = torch.device('mps')
                # Set MPS fallback for compatibility
                os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
            else:
                device = torch.device('cpu')
        
        super().__init__(device)
        self.model_name = model_name
        self.model = None
        self.verbose = verbose
        self.amp_dtype = torch.float16  # Can be float16 or bfloat16
        self.use_amp = True if device.type in ['cuda', 'mps'] else False
        
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
            # Load model on CPU first
            logger.debug("Initializing EncDecMultiTaskModel on CPU...")
            self.model = nemo_asr.models.EncDecMultiTaskModel.from_pretrained(
                self.model_name,
                map_location='cpu'
            )
            
            # Update decoding parameters with optimal settings
            logger.debug("Configuring decoding strategy...")
            decode_cfg = self.model.cfg.decoding
            decode_cfg.strategy = 'beam'
            decode_cfg.beam.beam_size = 1
            self.model.change_decoding_strategy(decode_cfg)
            
            # Convert model to float32 before device placement
            logger.debug("Converting model to float32...")
            self.model = self.model.to(torch.float32)
            
            # Move model to target device with AMP support
            if self.device.type in ['cuda', 'mps']:
                logger.debug(f"Moving model to {self.device} with AMP...")
                try:
                    with torch.amp.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                        self.model = self.model.to(self.device)
                except TypeError as e:
                    if "float64" in str(e) and self.device.type == 'mps':
                        logger.warning("MPS doesn't support float64. Falling back to CPU...")
                        self.device = torch.device("cpu")
                        self.model = self.model.to(self.device)
                        self.use_amp = False
                    else:
                        raise
            else:
                self.model = self.model.to(self.device)
            
            load_time = time.time() - start_time
            logger.info(f"Model loaded in {load_time:.2f} seconds on {self.device}")
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
            
            # Transcribe with AMP support
            logger.debug("Starting transcription...")
            transcribe_start = time.time()
            
            with torch.amp.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                with torch.no_grad():
                    raw_result = self.model.transcribe(audio_paths)
            
            transcribe_time = time.time() - transcribe_start
            logger.debug(f"Transcription completed in {transcribe_time:.2f} seconds")
            
            # Clean up each result
            transcriptions = [self._clean_text(h.text) for h in raw_result]
            
            end_time = time.time()
            logger.info(f"Transcription completed in {end_time - start_time:.2f} seconds")
            return transcriptions
            
        except Exception as e:
            logger.error(f"Error during transcription: {str(e)}")
            raise
    
    @property
    def name(self):
        """Return the name of the model."""
        return f"Canary-{self.model_name.split('/')[-1]}" 