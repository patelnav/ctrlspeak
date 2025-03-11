#!/usr/bin/env python3
"""
ctrlSPEAK Live - A continuous speech-to-text utility that transcribes in real-time.
Press Ctrl+C to exit.
"""
import nemo.collections.asr as nemo_asr
import torch
import sounddevice as sd
import numpy as np
import time
import threading
from queue import Queue
import soundfile as sf
from utils import audio

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS backend enabled: {torch.backends.mps.is_built()}")
print(f"Default PyTorch threads: {torch.get_num_threads()}")

# Enable MPS (Metal) acceleration if available
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Audio settings
SAMPLE_RATE = 16000  # NeMo expects 16kHz
CHANNELS = 1
CHUNK_DURATION = 2  # Process 2 seconds of audio at a time
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)

# Create a queue to store audio chunks
audio_queue = Queue()
is_recording = True

def get_model():
    """Load model with progress tracking"""
    print("\nLoading model...")
    start_time = time.time()
    
    try:
        model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained("nvidia/parakeet-tdt-1.1b")
        if device.type == "mps":
            model = model.to(device)
        
        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f} seconds")
        return model
    
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def audio_callback(indata, frames, time, status):
    """Callback function to process audio input"""
    if status:
        print(f"Status: {status}")
    # Convert to float32 and reshape
    audio_chunk = indata[:, 0].copy()
    audio_queue.put(audio_chunk)

def process_audio():
    """Process audio chunks from the queue"""
    model = get_model()
    accumulated_audio = np.array([], dtype=np.float32)
    
    while is_recording:
        if not audio_queue.empty():
            chunk = audio_queue.get()
            accumulated_audio = np.append(accumulated_audio, chunk)
            
            # Process when we have enough samples
            if len(accumulated_audio) >= CHUNK_SAMPLES:
                # Save the audio chunk temporarily
                temp_file = "temp_chunk.wav"
                sf.write(temp_file, accumulated_audio[:CHUNK_SAMPLES], SAMPLE_RATE)
                
                # Transcribe
                try:
                    start_time = time.time()
                    transcription = model.transcribe([temp_file])[0]
                    end_time = time.time()
                    if transcription and transcription[0]:  # Only print non-empty transcriptions
                        print(f"\nTranscription ({end_time - start_time:.2f}s): {transcription[0]}")
                except Exception as e:
                    print(f"Error during transcription: {e}")
                
                # Keep the remainder
                accumulated_audio = accumulated_audio[CHUNK_SAMPLES:]
        else:
            time.sleep(0.1)  # Small sleep to prevent busy waiting

try:
    print("\nStarting ctrlSPEAK Live transcription... (Press Ctrl+C to stop)")
    
    # Start the processing thread
    process_thread = threading.Thread(target=process_audio)
    process_thread.start()
    
    # Start recording
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=audio_callback):
        while True:
            time.sleep(0.1)

except KeyboardInterrupt:
    print("\nStopping recording...")
    is_recording = False
    process_thread.join()
    print("Recording stopped.") 