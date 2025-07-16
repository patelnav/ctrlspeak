"""
Whisper model implementation for speech-to-text.
"""
import time
import os
import torch
import logging
from models.base_model import BaseSTTModel
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, WhisperProcessor
from typing import List

# Configure logging
logger = logging.getLogger("whisper_model")

class WhisperModel(BaseSTTModel):
    """Whisper model for speech-to-text with accurate word timestamps."""
    
    def __init__(self, model_name="openai/whisper-large-v3", device=None, verbose=False):
        """Initialize the Whisper model.
        
        Args:
            model_name: The name of the pretrained model to load.
            device: The device to run the model on.
            verbose: Whether to enable verbose logging.
        """
        # Setup device with MPS support
        if device is None:
            if torch.cuda.is_available():
                device = torch.device('cuda:0')
                logger.info("Using CUDA device")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                logger.info("Using MPS device (Apple Silicon GPU)")
                device = torch.device('mps')
                # Set MPS fallback for compatibility
                os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
            else:
                logger.info("Using CPU device")
                device = torch.device('cpu')
        
        super().__init__(device)
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.pipe = None
        self.verbose = verbose
        self.amp_dtype = torch.float16 if device.type in ['cuda', 'mps'] else torch.float32
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
            # Load model and processor
            logger.debug("Initializing AutoModelForSpeechSeq2Seq and AutoProcessor...")
            
            self.processor = WhisperProcessor.from_pretrained(self.model_name, language="en")
            
            # Load model with appropriate dtype
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_name,
                torch_dtype=self.amp_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                attn_implementation="eager"  # Use eager implementation to avoid MPS compatibility issues
            )
            
            # Move model to device
            logger.debug(f"Moving model to {self.device}...")
            self.model = self.model.to(self.device)
            
            # Create pipeline with simpler settings
            logger.debug("Creating ASR pipeline...")
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                torch_dtype=self.amp_dtype,
                device=self.device
            )
            
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
            
            transcriptions = []
            for audio_path in audio_paths:
                with torch.amp.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                    with torch.no_grad():
                        # Process the audio with simpler parameters
                        result = self.pipe(audio_path, generate_kwargs={"language": "<|en|>", "task": "translate"})
                        # Clean up the result
                        transcriptions.append(self._clean_text(result["text"]))
            
            transcribe_time = time.time() - transcribe_start
            logger.debug(f"Transcription completed in {transcribe_time:.2f} seconds")
            
            end_time = time.time()
            logger.info(f"Transcription completed in {end_time - start_time:.2f} seconds")
            return transcriptions
            
        except Exception as e:
            logger.error(f"Error during transcription: {str(e)}")
            raise
    
    @property
    def name(self):
        """Return the name of the model."""
        return f"Whisper-{self.model_name.split('/')[-1]}" 