#!/usr/bin/env python3
"""
Test script to verify RMS calculation and logging from microphone input.
Runs for 5 seconds and prints RMS values to debug log.
"""

import sounddevice as sd
import numpy as np
import logging
import time
import sys

# --- Configuration ---
SAMPLE_RATE = 16000  # Match the main app's sample rate
CHANNELS = 1
DURATION = 5  # seconds
DEVICE = None # Use default input device

# --- Logging Setup ---
# Configure logging to show DEBUG messages on the console for this test
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)] # Print logs to stdout
)
logger = logging.getLogger("test_rms_logging")

# --- Callback Function ---
def rms_log_callback(indata, frames, time_info, status):
    """Calculates and logs the RMS of the incoming audio chunk."""
    if status:
        logger.warning(f"Callback status: {status}")
    
    try:
        # Calculate RMS for the current chunk
        # indata is likely float32 for sounddevice
        rms = np.sqrt(np.mean(indata**2))
        logger.debug(f"RMS: {rms:.6f}")
    except Exception as e:
        logger.error(f"Error calculating RMS in callback: {e}", exc_info=True)

# --- Main Test Logic ---
if __name__ == "__main__":
    logger.info(f"Starting RMS logging test for {DURATION} seconds...")
    logger.info(f"Using Sample Rate: {SAMPLE_RATE} Hz, Channels: {CHANNELS}")
    
    stream = None
    try:
        # Create and start the input stream
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype='float32', # Explicitly set dtype
            callback=rms_log_callback,
            device=DEVICE
        )
        stream.start()
        logger.info("Microphone stream started. Speak or make noise!")

        # Keep the stream running for the specified duration
        time.sleep(DURATION)

        logger.info("Test duration finished.")

    except Exception as e:
        logger.error(f"An error occurred during the test: {e}", exc_info=True)
    finally:
        # Ensure the stream is stopped and closed properly
        if stream:
            try:
                if not stream.stopped:
                    stream.stop()
                    logger.info("Microphone stream stopped.")
                if not stream.closed:
                    stream.close()
                    logger.info("Microphone stream closed.")
            except Exception as e_close:
                logger.error(f"Error closing stream: {e_close}")
        
        logger.info("RMS logging test finished.") 