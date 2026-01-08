#!/usr/bin/env python3
"""
Parallel model comparison test - BOTH in real-time mode.

Runs Nemotron (streaming) and Parakeet (RMS-based chunking) simultaneously
on the same live microphone input to compare transcription quality.

Usage:
    python test_parallel_models.py

Press Enter to start recording, Enter again to stop.
"""

import sys
import threading
import queue
import time
import numpy as np
import sounddevice as sd
import tempfile
import soundfile as sf

SAMPLE_RATE = 16000
CHUNK_SIZE_MS = 1120  # Nemotron streaming chunk size
CHUNK_SIZE_SAMPLES = int(SAMPLE_RATE * CHUNK_SIZE_MS / 1000)

# RMS detection settings (same as main app)
RMS_THRESHOLD = 0.01
SILENCE_DURATION_S = 1.0
MIN_CHUNK_DURATION_S = 0.5

# Shared state
nemotron_queue = queue.Queue()  # For Nemotron streaming
parakeet_queue = queue.Queue()  # For Parakeet batch chunks
is_recording = False
stop_event = threading.Event()

# RMS-based chunking state for Parakeet
rms_audio_buffer = []
rms_silence_s = 0.0
rms_is_speaking = False


def load_models():
    """Load both models."""
    print("Loading models...")

    # Load Nemotron
    print("  Loading Nemotron (streaming)...")
    from models.nemotron import NemotronModel
    nemotron = NemotronModel()
    nemotron.load_model()

    # Load Parakeet MLX
    print("  Loading Parakeet MLX (batch with RMS)...")
    from models.parakeet_mlx import ParakeetMLXModel
    parakeet = ParakeetMLXModel()
    parakeet.load_model()

    print("Both models loaded!\n")
    return nemotron, parakeet


def audio_callback(indata, frames, time_info, status):
    """Audio callback - feeds both pipelines."""
    global rms_audio_buffer, rms_silence_s, rms_is_speaking

    if not is_recording:
        return

    chunk = indata.copy().flatten()
    chunk_duration_s = float(frames) / SAMPLE_RATE

    # === NEMOTRON: Fixed-interval streaming ===
    nemotron_queue.put(chunk)

    # === PARAKEET: RMS-based silence detection ===
    rms = np.sqrt(np.mean(chunk**2))
    is_speech = rms >= RMS_THRESHOLD

    if is_speech:
        rms_audio_buffer.append(chunk)
        rms_silence_s = 0.0
        rms_is_speaking = True
    elif rms_is_speaking:
        rms_silence_s += chunk_duration_s

        if rms_silence_s >= SILENCE_DURATION_S:
            # Silence threshold reached - queue segment for Parakeet
            if rms_audio_buffer:
                segment = np.concatenate(rms_audio_buffer)
                segment_duration = len(segment) / SAMPLE_RATE
                if segment_duration >= MIN_CHUNK_DURATION_S:
                    parakeet_queue.put(segment)
            rms_audio_buffer = []
            rms_silence_s = 0.0
            rms_is_speaking = False
        else:
            rms_audio_buffer.append(chunk)


def nemotron_worker(model, results):
    """Worker thread for Nemotron streaming."""
    accumulated_samples = []
    chunk_count = 0

    while not stop_event.is_set() or not nemotron_queue.empty():
        try:
            chunk = nemotron_queue.get(timeout=0.1)
            accumulated_samples.append(chunk)

            # Process when we have enough samples
            total_samples = sum(len(c) for c in accumulated_samples)
            if total_samples >= CHUNK_SIZE_SAMPLES:
                audio_data = np.concatenate(accumulated_samples).flatten()
                chunk_to_process = audio_data[:CHUNK_SIZE_SAMPLES]
                remainder = audio_data[CHUNK_SIZE_SAMPLES:]

                if len(remainder) > 0:
                    accumulated_samples = [remainder]
                else:
                    accumulated_samples = []

                chunk_count += 1
                text = model.stream_chunk(chunk_to_process)
                if text:
                    results['nemotron'] = text
                    # Live render Nemotron output with chunk count and text length
                    display = text[-70:] if len(text) > 70 else text  # Show END of text
                    print(f"\n  [N#{chunk_count} len={len(text)}] ...{display}", end='', flush=True)

            nemotron_queue.task_done()
        except queue.Empty:
            continue

    # Process any remaining audio with is_final=True
    if accumulated_samples:
        remaining = np.concatenate(accumulated_samples).flatten()
        if len(remaining) > 0:
            chunk_count += 1
            text = model.stream_chunk(remaining, is_final=True)
            if text:
                results['nemotron'] = text

    # Finalize
    final_text = model.finalize_streaming()
    results['nemotron'] = final_text
    results['nemotron_chunks'] = chunk_count


def parakeet_worker(model, results):
    """Worker thread for Parakeet with RMS-based real-time chunking."""
    transcriptions = []
    segment_count = 0

    while not stop_event.is_set() or not parakeet_queue.empty():
        try:
            segment = parakeet_queue.get(timeout=0.1)
            segment_count += 1

            # Save to temp file and transcribe
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                sf.write(f.name, segment, SAMPLE_RATE)
                temp_path = f.name

            result = model.transcribe_batch([temp_path])
            if result and result[0]:
                transcriptions.append(result[0])
                # Show running transcription
                current_text = " ".join(transcriptions)
                display = current_text[:70] + "..." if len(current_text) > 70 else current_text
                print(f"\n  [P] {display}", end='', flush=True)

            # Cleanup
            import os
            os.unlink(temp_path)

            parakeet_queue.task_done()
        except queue.Empty:
            continue

    results['parakeet'] = " ".join(transcriptions)
    results['parakeet_segments'] = segment_count


def main():
    global is_recording, rms_audio_buffer, rms_silence_s, rms_is_speaking

    print("=" * 60)
    print("PARALLEL MODEL COMPARISON TEST (Both Real-Time)")
    print("=" * 60)
    print()
    print("Nemotron: Fixed 1120ms chunks (streaming API)")
    print("Parakeet: RMS-based silence detection (batch per segment)")
    print()

    # Load models
    nemotron, parakeet = load_models()

    # Results storage
    results = {
        'nemotron': '',
        'parakeet': '',
        'nemotron_chunks': 0,
        'parakeet_segments': 0
    }

    input("Press ENTER to start recording...")
    print("\nRecording... (press ENTER to stop)\n")

    # Initialize Nemotron streaming
    nemotron.init_streaming()

    # Reset state
    rms_audio_buffer = []
    rms_silence_s = 0.0
    rms_is_speaking = False
    stop_event.clear()
    is_recording = True

    # Start workers
    nemotron_thread = threading.Thread(target=nemotron_worker, args=(nemotron, results))
    parakeet_thread = threading.Thread(target=parakeet_worker, args=(parakeet, results))
    nemotron_thread.start()
    parakeet_thread.start()

    # Start audio stream
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        callback=audio_callback,
        dtype=np.float32
    )
    stream.start()

    # Wait for user to stop
    input()

    # Stop recording
    print("\n\nStopping...")
    is_recording = False

    # Queue any remaining audio for Parakeet
    if rms_audio_buffer:
        segment = np.concatenate(rms_audio_buffer)
        if len(segment) / SAMPLE_RATE >= MIN_CHUNK_DURATION_S:
            parakeet_queue.put(segment)

    stop_event.set()
    stream.stop()
    stream.close()

    # Wait for workers
    print("  Waiting for Nemotron...")
    nemotron_thread.join(timeout=10)
    print("  Waiting for Parakeet...")
    parakeet_thread.join(timeout=30)

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\n[NEMOTRON - Streaming] ({results['nemotron_chunks']} chunks)")
    print(f"  \"{results['nemotron']}\"")

    print(f"\n[PARAKEET - RMS Chunked] ({results['parakeet_segments']} segments)")
    print(f"  \"{results['parakeet']}\"")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
