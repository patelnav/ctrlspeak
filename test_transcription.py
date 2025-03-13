#!/usr/bin/env python3
"""
ctrlSPEAK Test - A script for testing transcription with detailed logging.
"""
import torch
import sys
import os
import time
import argparse
import logging
from models.factory import ModelFactory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ctrlspeak-test")

def transcribe_audio(audio_file, model_type, verbose=False):
    """
    Transcribe an audio file using the specified model.
    
    Args:
        audio_file: Path to the audio file to transcribe.
        model_type: Type of model to use (parakeet, canary, or whisper).
        verbose: Whether to enable verbose logging.
    
    Returns:
        The transcription result.
    """
    if verbose:
        logger.setLevel(logging.DEBUG)
    
    # Check if file exists
    if not os.path.exists(audio_file):
        logger.error(f"Audio file not found: {audio_file}")
        return None
    
    # Enable MPS (Metal) acceleration if available
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info(f"Loading {model_type} model...")
    start_time = time.time()
    model = ModelFactory.get_model(model_type=model_type, device=device, verbose=verbose)
    model.load_model()
    load_time = time.time() - start_time
    logger.info(f"Model loaded in {load_time:.2f} seconds")
    
    # Transcribe audio
    logger.info(f"Transcribing {audio_file}...")
    start_time = time.time()
    
    # Transcribe using our simplified API
    result = model.transcribe(audio_file)
    
    end_time = time.time()
    transcribe_time = end_time - start_time
    
    # Print results
    logger.info(f"Transcription completed in {transcribe_time:.2f} seconds")
    
    if result:
        logger.info("-" * 50)
        logger.info(f"Transcription: {result}")
        logger.info("-" * 50)
    else:
        logger.warning("No transcription result")
    
    return result

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="ctrlSPEAK Test - Test transcription on an audio file")
    parser.add_argument("audio_file", help="Path to the audio file to transcribe")
    parser.add_argument("--model", choices=["parakeet", "canary", "whisper"], default="parakeet",
                        help="Model to use for transcription (default: parakeet)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    
    # Print PyTorch info
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"MPS available: {torch.backends.mps.is_available()}")
    logger.info(f"MPS backend enabled: {torch.backends.mps.is_built()}")
    
    # Transcribe
    transcribe_audio(args.audio_file, args.model, args.verbose)

if __name__ == "__main__":
    main() 