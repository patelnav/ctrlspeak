"""
Streaming transcription module for ctrlSPEAK.

Handles real-time streaming transcription for models that support it
(e.g., Nemotron). Uses a worker thread to process audio chunks without
blocking the audio callback.
"""

import logging
import threading
import queue
import numpy as np
import state

logger = logging.getLogger("ctrlspeak.streaming")

# Streaming worker thread state
_streaming_queue = None
_streaming_worker_thread = None
_streaming_stop_event = None


def _streaming_worker():
    """Worker thread for processing streaming audio chunks.

    Runs in a separate thread to avoid blocking the audio callback.
    Pulls chunks from the queue and processes them through the model.
    """
    global _streaming_queue, _streaming_stop_event

    logger.info("Streaming worker thread started")

    chunk_count = 0
    while not _streaming_stop_event.is_set():
        try:
            # Wait for a chunk with timeout to allow checking stop event
            queue_item = _streaming_queue.get(timeout=0.1)

            if queue_item is None:
                # Sentinel value - stop processing
                logger.debug("[WORKER] Received stop sentinel")
                break

            # Unpack tuple (audio_samples, is_final)
            audio_samples, is_final = queue_item

            chunk_count += 1
            duration_ms = len(audio_samples) / 16000 * 1000
            # Calculate RMS to verify audio level
            rms = np.sqrt(np.mean(audio_samples**2))
            max_amp = np.abs(audio_samples).max()
            logger.debug(f"[WORKER] Processing chunk #{chunk_count}: {len(audio_samples)} samples ({duration_ms:.0f}ms), RMS={rms:.4f}, max={max_amp:.4f}, is_final={is_final}")

            if state.stt_model is None:
                logger.warning("Streaming chunk received but no model loaded")
                _streaming_queue.task_done()
                continue

            # Process chunk through model's streaming API
            text = state.stt_model.stream_chunk(audio_samples, is_final=is_final)

            # Update accumulated text (streaming returns cumulative text)
            if text:
                # For streaming, the model returns cumulative text, not deltas
                # So we replace rather than append
                if state.transcribed_chunks:
                    state.transcribed_chunks[-1] = text
                else:
                    state.transcribed_chunks.append(text)

                # Update UI if available
                if hasattr(state, 'app_state_ref') and state.app_state_ref:
                    state.app_state_ref.accumulated_text = text

                if len(text) > 50:
                    logger.debug(f"[WORKER] Chunk #{chunk_count} result: \"{text[:50]}...\"")
                else:
                    logger.debug(f"[WORKER] Chunk #{chunk_count} result: \"{text}\"")

            logger.debug(f"[WORKER] Chunk #{chunk_count} done, calling task_done()")
            _streaming_queue.task_done()

        except queue.Empty:
            # Timeout - just continue to check stop event
            continue
        except Exception as e:
            logger.error(f"Error in streaming worker: {e}")
            try:
                _streaming_queue.task_done()
            except ValueError:
                pass

    logger.info(f"[WORKER] Streaming worker thread stopped after processing {chunk_count} chunks")


def on_streaming_chunk(audio_samples, is_final=False):
    """Callback for streaming mode - queues audio chunk for processing.

    Called by AudioManager with fixed-interval audio chunks.
    Queues the chunk for the streaming worker thread to process.

    Args:
        audio_samples: numpy array of float32 audio samples (16kHz mono)
        is_final: if True, this is the last chunk and decoder should flush
    """
    global _streaming_queue

    if _streaming_queue is not None:
        try:
            qsize = _streaming_queue.qsize()
            duration_ms = len(audio_samples) / 16000 * 1000
            logger.debug(f"[QUEUE_ADD] Queueing chunk: {len(audio_samples)} samples ({duration_ms:.0f}ms), is_final={is_final}, queue size: {qsize}")
            # Queue as tuple (audio, is_final)
            _streaming_queue.put_nowait((audio_samples, is_final))
        except queue.Full:
            logger.warning("Streaming queue full, dropping chunk")
    else:
        logger.warning("Streaming chunk received but queue not initialized")


def start_streaming():
    """Start streaming transcription session.

    Initializes the model's streaming state, starts the worker thread,
    and begins audio collection in streaming mode.
    """
    global _streaming_queue, _streaming_worker_thread, _streaming_stop_event

    logger.info("Starting streaming recording session...")

    # Initialize model's streaming state
    state.stt_model.init_streaming()

    # Get chunk size from model (if available)
    # 1120ms (14 frames) gives best accuracy, 560ms is faster but lower quality
    chunk_size_ms = 1120  # Default to best accuracy
    if hasattr(state.stt_model, 'chunk_size_ms'):
        chunk_size_ms = state.stt_model.chunk_size_ms

    # Initialize transcription storage with empty string for streaming
    state.transcribed_chunks.clear()
    state.transcribed_chunks.append("")  # Placeholder for cumulative text

    # Reset accumulated text for UI
    if hasattr(state, 'app_state_ref') and state.app_state_ref:
        state.app_state_ref.accumulated_text = ""

    # Initialize streaming queue and worker thread
    _streaming_queue = queue.Queue(maxsize=50)  # Buffer up to 50 chunks
    _streaming_stop_event = threading.Event()
    _streaming_worker_thread = threading.Thread(
        target=_streaming_worker,
        name="StreamingWorker",
        daemon=True
    )
    _streaming_worker_thread.start()

    # Start streaming audio collection
    state.audio_manager.start_streaming(
        chunk_size_ms=chunk_size_ms,
        on_chunk_callback=on_streaming_chunk
    )


def stop_streaming():
    """Stop streaming transcription and return final text.

    Stops audio collection, waits for worker thread to finish,
    and finalizes the model's streaming state.

    Returns:
        Final transcribed text from the streaming session.
    """
    global _streaming_queue, _streaming_worker_thread, _streaming_stop_event

    logger.info("[STOP] Stopping streaming recording session...")

    # Stop audio collection (processes remaining buffer and queues final chunk)
    logger.debug("[STOP] Calling audio_manager.stop_streaming()...")
    state.audio_manager.stop_streaming()
    logger.debug("[STOP] audio_manager.stop_streaming() completed")

    # Wait for all queued chunks to be processed BEFORE stopping worker
    # This fixes the race condition where final chunk wasn't transcribed
    if _streaming_queue:
        qsize = _streaming_queue.qsize()
        logger.info(f"[STOP] Waiting for queue to drain ({qsize} items remaining)...")
        try:
            _streaming_queue.join()
            logger.info("[STOP] Queue drained successfully - all chunks processed")
        except Exception as e:
            logger.warning(f"[STOP] Error waiting for queue: {e}")

    # Now stop the streaming worker thread
    if _streaming_stop_event:
        _streaming_stop_event.set()

    # Send sentinel to wake up worker if waiting
    if _streaming_queue:
        try:
            _streaming_queue.put_nowait(None)
        except queue.Full:
            pass

    # Wait for worker thread to finish
    if _streaming_worker_thread and _streaming_worker_thread.is_alive():
        logger.debug("Waiting for streaming worker thread to finish...")
        _streaming_worker_thread.join(timeout=2.0)
        if _streaming_worker_thread.is_alive():
            logger.warning("Streaming worker thread did not stop in time")

    # Finalize model's streaming state and get final text
    logger.debug("[STOP] Calling model.finalize_streaming()...")
    final_text = state.stt_model.finalize_streaming()

    if final_text:
        logger.info(f"[STOP] Final text ({len(final_text)} chars): \"{final_text[:80]}{'...' if len(final_text) > 80 else ''}\"")
    else:
        logger.warning("[STOP] finalize_streaming returned empty text")

    # Cleanup
    _streaming_queue = None
    _streaming_worker_thread = None
    _streaming_stop_event = None

    return final_text


def is_model_streaming_capable():
    """Check if the current model supports streaming transcription.

    Returns:
        True if model supports streaming, False otherwise.
    """
    return (
        state.stt_model is not None and
        hasattr(state.stt_model, 'supports_streaming') and
        state.stt_model.supports_streaming
    )
