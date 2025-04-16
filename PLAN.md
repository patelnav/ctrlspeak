# ctrlSPEAK Enhancement Plan: RMS-Based Async Transcription

## Goal

Modify `ctrlspeak` to detect speech pauses (silence) using audio energy (RMS). Upon detecting a significant pause (~2 seconds), the preceding speech segment is sent for transcription in the background using a worker thread. Recording continues uninterrupted. When the user stops recording (second triple-tap), any remaining audio is transcribed, and all transcribed segments are concatenated and pasted. This aims to improve perceived transcription speed by parallelizing transcription during pauses.

## Core Design Changes (Minimal Approach)

1.  **RMS Silence Detection:** Use Root Mean Square (RMS) energy of incoming audio chunks to detect periods of silence, replacing the need for a dedicated VAD library initially.
2.  **Audio Buffering:** `AudioManager` will buffer chunks of audio *during* speech and short silences.
3.  **Asynchronous Transcription Queue:** When a sufficiently long silence is detected via RMS, the accumulated audio buffer (as a NumPy array or bytes) is placed onto a `queue.Queue`.
4.  **Transcription Worker:** A dedicated `threading.Thread` monitors the queue, retrieves audio segments, transcribes them using the STT model, and stores the resulting text.
5.  **State Management:** `ctrlspeak.py` manages the worker thread lifecycle, the queue, and the list collecting transcribed text segments.
6.  **Final Concatenation:** The 'stop' action triggers transcription of any final audio segment and concatenates all stored text segments before pasting.

## Phased Implementation Plan (RMS-Based, Iterative)

This plan focuses on incremental changes with verification at each step.

### Phase 0: Basic Setup & Async Infrastructure Test

1.  **Goal:** Verify the core async components (queue, worker thread, result collection) work reliably in isolation using a mock transcriber.
2.  **Implementation (`ctrlspeak.py`):**
    *   Add imports: `import queue`, `import threading`, `import numpy as np`, `import time`.
    *   Add global variables: `transcribed_chunks = []`, `transcription_queue = queue.Queue()`, `transcription_active = threading.Event()`, `transcription_worker_thread = None`.
    *   Implement `transcription_worker` function: Takes args `(model, work_queue, results_list, active_event)`. Uses loop `while active_event.is_set() or not work_queue.empty():`. Gets items from `work_queue` (timeout 0.5), calls `model.transcribe()`, appends to `results_list`, calls `work_queue.task_done()`. Include basic `try...except queue.Empty` and general `Exception` logging.
    *   **Mock Transcriber:** Inside `transcription_worker` initially, *replace* `model.transcribe(audio_data)` with a mock call, e.g., `text = f"mock_transcription_for_chunk_len_{len(audio_data)}"` or similar. This avoids needing the real model for this phase.
    *   Add a simple test function (e.g., `test_async_infra()`) callable perhaps via a temporary command-line flag or just run manually in `main` initially:
```python
        def test_async_infra():
            print("Testing async infrastructure...")
            # Start worker with mock model (e.g., a simple lambda/function)
            mock_model = lambda data: f"mock_{len(data)}"
            test_active = threading.Event()
            test_active.set()
            test_queue = queue.Queue()
            test_results = []
            worker = threading.Thread(target=transcription_worker, args=(mock_model, test_queue, test_results, test_active), daemon=True)
            worker.start()

            # Put dummy data
            print("Putting dummy data onto queue...")
            test_queue.put(np.zeros(100))
            test_queue.put(np.zeros(200))
            print("Waiting for worker to process...")
            # Give worker time to process - join() is better
            # time.sleep(0.2)

            # Signal stop and wait
            print("Signaling worker stop...")
            test_active.clear()
            print("Waiting for queue to join...")
            test_queue.join() # Wait for task_done calls
            print("Queue joined. Waiting for worker thread to join...")
            worker.join(timeout=1.0) # Wait for thread itself

            # Verify
            print(f"Test results: {test_results}")
            assert len(test_results) == 2, f"Expected 2 results, got {len(test_results)}"
            # Order might not be guaranteed, sort for assertion
            test_results.sort(key=len)
            assert test_results[0] == "mock_100", f"Unexpected result[0]: {test_results[0]}"
            assert test_results[1] == "mock_200", f"Unexpected result[1]: {test_results[1]}"
            print("Async infrastructure test PASSED.")

        # Example call early in main() during development:
        # if __name__ == "__main__":
        #     # ... setup logger ...
        #     test_async_infra()
        #     # ... rest of main ...
        ```
3.  **Verification:** Run the `test_async_infra()` function (e.g., by modifying `main` temporarily). Confirm it prints "PASSED" and the mock results are as expected. This validates the basic threading, queue, and result collection logic.

### Phase 1: Integrate Basic Async Structure into Start/Stop Flow

1.  **Goal:** Connect the verified async infrastructure to the application's lifecycle (`main`, `on_activate` start/stop, `exit_app`) without implementing RMS segmentation yet.
2.  **Implementation (`ctrlspeak.py`):**
    *   In `main()`: Remove the test call. Start the *real* `transcription_worker_thread` using the structure from Phase 0 (but still passing a **mock** model initially, e.g., `mock_model = lambda data: f"mock_{len(data)}"`).
    *   In `on_activate` (Start): `transcribed_chunks.clear()`. Call `audio_manager.start_recording()`.
    *   In `on_activate` (Stop): Call `audio_manager.stop_recording()`. Signal worker `transcription_active.clear()`. Log message "Waiting for transcription queue...". **Wait for worker `transcription_queue.join()`**. Log message "Queue processed." Concatenate `transcribed_chunks` (should contain one mock result). Paste result. Ensure beep timing is appropriate (likely *after* processing).
    *   In `exit_app`: Add `transcription_active.clear()` and `transcription_worker_thread.join(timeout=2.0)`.
    *   **Implementation (`utils/audio.py` - AudioManager):**
        *   Modify `stop_recording`: Instead of saving a file, get the entire recorded buffer (e.g., `all_data = np.concatenate(self.audio_buffer)` assuming `self.audio_buffer` is a list of numpy chunks collected in `audio_callback`). Log the size of `all_data`. Put this single `all_data` chunk onto the *global* `transcription_queue` (needs import: `from ctrlspeak import transcription_queue`). **Remove the file path return value.** `stop_recording` now performs an action but returns nothing.
        *   Modify `audio_callback`: Ensure it appends `indata.copy()` to `self.audio_buffer`.
        *   Modify `start_recording`: Ensure `self.audio_buffer = []` is called.
3.  **Verification:** Manually run `ctrlspeak.py`.
    *   Triple-tap start, speak briefly, triple-tap stop.
    *   Check logs/console output:
        *   Worker thread started (using mock model).
        *   `stop_recording` logged putting one large chunk onto the queue.
        *   Worker log shows processing one chunk (mock transcription).
        *   Stop logic logged waiting ("Waiting for transcription queue...") and completion ("Queue processed.").
        *   The final pasted text is the single mock transcription result.
        *   Application exits cleanly via Ctrl+C (`exit_app` logs worker join).
    *   **Verify Bluetooth audio profile remains in high-quality mode (e.g., A2DP) during recording.**

### Phase 2: Implement RMS Calculation & Basic Logging in Callback

1.  **Goal:** Calculate and observe RMS values for incoming audio to inform threshold selection, without changing segmentation logic yet.
2.  **Implementation (`utils/audio.py` - AudioManager):**
    *   In `__init__`: Add placeholder state variables: `self.RMS_THRESHOLD = 0.01 # Needs tuning`, `self.SILENCE_DURATION_S = 2.0`, `self.MIN_CHUNK_DURATION_S = 0.5`, `self.current_speech_buffer = []`, `self.current_silence_s = 0.0`, `self.is_potentially_speaking = False`. (Rename `self.audio_buffer` from Phase 1 to `self.current_speech_buffer` if needed).
    *   Modify `audio_callback`:
        *   Keep appending audio: `self.current_speech_buffer.append(indata.copy())`.
        *   Calculate RMS: `rms = np.sqrt(np.mean(indata**2))`.
        *   **Log the RMS value:** `logger.debug(f"RMS: {rms:.4f}")`.
        *   **Do not add segmentation logic yet.**
    *   Modify `stop_recording`: Concatenate `self.current_speech_buffer` and put the single chunk on the queue (as in Phase 1).
3.  **Verification:** Manually run `ctrlspeak.py`.
    *   Triple-tap start. Speak normally, speak softly, stay silent for periods. Triple-tap stop.
    *   Observe the `DEBUG` level logs.
    *   Verify that RMS values are consistently low during silence (e.g., < 0.005) and significantly higher during speech (e.g., > 0.01 or 0.02). Note approximate values for your setup. This helps choose a sensible initial `RMS_THRESHOLD`.

### Phase 3: Implement RMS Thresholding & Segmentation Logic

1.  **Goal:** Implement the core silence detection logic using RMS and verify that audio is segmented and queued during pauses.
2.  **Implementation (`utils/audio.py` - AudioManager):**
    *   Modify `audio_callback` to fully implement the RMS segmentation logic:
```python
        # (Inside audio_callback)
        chunk = indata.copy()
        current_chunk_duration_s = float(frames) / self.sample_rate # Calculate duration of this chunk

        rms = np.sqrt(np.mean(chunk**2))
        logger.debug(f"RMS: {rms:.4f}") # Keep logging RMS

        is_speech_chunk = rms >= self.RMS_THRESHOLD

        if is_speech_chunk:
            # Append speech (or silence that broke a potential silence period) to buffer
            self.current_speech_buffer.append(chunk)
            if not self.is_potentially_speaking:
                 logger.debug("Speech detected (RMS above threshold).")
            self.current_silence_s = 0.0 # Reset silence duration
            self.is_potentially_speaking = True
        elif self.is_potentially_speaking:
            # Silence detected after potential speech - Start accumulating silence duration
            self.current_silence_s += current_chunk_duration_s
            logger.debug(f"Silence accumulating: {self.current_silence_s:.2f}s / {self.SILENCE_DURATION_S}s")

            # Check if silence duration threshold is met
            if self.current_silence_s >= self.SILENCE_DURATION_S:
                logger.info(f"Silence threshold reached ({self.current_silence_s:.2f}s). Attempting to segment.")
                # We have enough silence. Process the buffered speech *before* this silence.
                if self.current_speech_buffer:
                    # Concatenate all buffered chunks
                    segment_data = np.concatenate(self.current_speech_buffer)
                    segment_duration_s = len(segment_data) / self.sample_rate

                    # Check minimum length for transcription
                    if segment_duration_s >= self.MIN_CHUNK_DURATION_S:
                         logger.info(f"Queueing segment of {segment_duration_s:.2f}s for transcription.")
                         # Access global queue (HACK - improve later by passing queue)
                         from ctrlspeak import transcription_queue
                         transcription_queue.put(segment_data)
                    else:
                         logger.info(f"Skipping short segment ({segment_duration_s:.2f}s).")

                # Reset buffer and state for the next speech segment
                self.current_speech_buffer = []
                self.current_silence_s = 0.0
                self.is_potentially_speaking = False # Stay silent until speech is detected again
else:
                # Silence continues but hasn't reached threshold yet. Keep the chunk
                # in case speech resumes shortly (avoids splitting mid-word pauses).
                 self.current_speech_buffer.append(chunk)

        # Else (is_speech_chunk is False AND self.is_potentially_speaking is False):
        # This means silence continues after a segment was already cut, or it's initial silence.
        # Do nothing - don't buffer silence between segments.
        ```
    *   Modify `start_recording`: Ensure all relevant state (`current_speech_buffer`, `current_silence_s`, `is_potentially_speaking`) is reset.
    *   Modify `stop_recording`: Ensure any final audio in `current_speech_buffer` is concatenated and queued (checking `MIN_CHUNK_DURATION_S`).
3.  **Verification:** Manually run `ctrlspeak.py` (still using **mock** transcriber).
    *   Triple-tap start. Speak a sentence, pause for > 2 seconds, speak another sentence, pause, stop.
    *   Observe logs:
        *   Confirm `DEBUG` logs show RMS values and "Silence accumulating..." messages during pauses.
        *   Confirm `INFO` logs show "Silence threshold reached" and "Queueing segment..." messages trigger after ~2 seconds of silence *following speech*.
        *   Confirm worker log shows processing *multiple* chunks.
        *   Confirm final pasted text contains multiple mock transcription results concatenated (e.g., "mock_16000 mock_24000").

### Phase 4: Switch to Real Transcription Model

1.  **Goal:** Verify the complete end-to-end system works using the actual STT model for transcription.
2.  **Implementation:**
    *   **Worker Input:** Determine if the `stt_model.transcribe()` method used by `ctrlspeak` accepts a NumPy array directly. Get the expected `SAMPLE_RATE` (likely `16000`).
        *   **If YES:** In `ctrlspeak.py`/`main`, when creating the `transcription_worker_thread`, pass the *real* loaded `stt_model` object. The worker function should now work as is.
        *   **If NO (Requires File Path):** Modify the `transcription_worker` function:
            *   Import `tempfile`, `soundfile` (or `wave`), `os`.
            *   Define `SAMPLE_RATE = 16000` (or get from config/audio_manager).
            *   Inside the `try...except queue.Empty` block:
```python
                audio_data = work_queue.get(timeout=0.5)
                temp_file_path = None
                try:
                    # Create temp file
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        temp_file_path = tmp.name
                    # Write data using soundfile
                    sf.write(temp_file_path, audio_data, SAMPLE_RATE)
                    logger.debug(f"Saved segment to temp file: {temp_file_path}")

                    # Transcribe from file path
                    text = model.transcribe(temp_file_path)

                    if text:
                        logger.info(f"Worker transcribed chunk: {text[:50]}...")
                        results_list.append(text)

                except Exception as e:
                    logger.error(f"Error during transcription or file handling: {e}", exc_info=True)
                finally:
                    # Clean up temp file
                    if temp_file_path and os.path.exists(temp_file_path):
                        try:
                            os.unlink(temp_file_path)
                            logger.debug(f"Deleted temp file: {temp_file_path}")
                        except Exception as e_del:
                            logger.error(f"Failed to delete temp file {temp_file_path}: {e_del}")
                work_queue.task_done() # Ensure task_done is called even on error
                ```
    *   Adjust logging in worker if needed.
3.  **Verification:** Manually run `ctrlspeak.py`.
    *   Perform several realistic test cases:
        *   Short utterance, stop.
        *   Long utterance, stop.
        *   Utterance, pause > 2s, utterance, stop.
        *   Multiple pauses.
    *   Verify:
        *   The final pasted text is accurate and correctly concatenates segments separated by pauses (check spacing).
        *   Transcription speed feels subjectively faster due to parallel processing (especially after pauses). Check logs to see worker activity during recording.
        *   **Re-verify Bluetooth audio profile remains in high-quality mode.**
        *   Resource usage (CPU) is acceptable during recording and transcription.

### Phase 5: Tuning & Refinement

1.  **Goal:** Improve the reliability and usability of the RMS detection and the overall system.
2.  **Implementation:**
    *   **Tune Parameters:** Experiment with `RMS_THRESHOLD`, `SILENCE_DURATION_S`, `MIN_CHUNK_DURATION_S` based on results from Phase 4 in different noise conditions. The threshold is key.
    *   **Configuration:** Make `RMS_THRESHOLD` (and potentially others) user-configurable (e.g., load from a simple config file or add a command-line argument). Provide guidance on tuning it.
    *   **Refine Segmentation:** Review the logic in Phase 3 for exactly how the buffer is split. Does it clip speech or include too much silence? Adjust if necessary (e.g., how the buffer is handled when silence starts/ends).
    *   **Error Handling:** Add more specific error handling in the worker (e.g., what happens if transcription fails for one chunk? Log and continue? Add placeholder text?). Improve error reporting to the user.
    *   **Code Cleanup:** Refactor access to the global queue (e.g., pass `transcription_queue` to `AudioManager` instance). Improve logging clarity. Use constants for thresholds.
    *   **(Optional) Memory Limits:** Add `maxsize` to `queue.Queue()` if needed. Consider limits on `current_speech_buffer` length to prevent unbounded growth on extremely long utterances without pauses.
3.  **Verification:** Test extensively in target environments (quiet, noisy). Ensure configurability works. Confirm robustness against edge cases (e.g., starting with silence, very short speech).

## Implementation Considerations (Adapted for RMS)

*   **RMS Threshold Tuning:** This is critical. A fixed threshold won't work everywhere. Needs to be adjustable and ideally calibrated per environment/mic setup. Initial value ~0.01 is a guess.
*   **Minimum Chunk Duration:** Prevents processing tiny audio blips between words that might dip below RMS threshold momentarily. `MIN_CHUNK_DURATION_S = 0.5` seems reasonable.
*   **Transcription Input:** Handling NumPy arrays directly in the worker is cleaner than temp files if the model supports it. Verify model capabilities. If using files, ensure robust cleanup.
*   **Concatenation:** Ensure segments are joined with appropriate spacing (a single space is usually fine).
*   **Thread Safety:** Using `queue.Queue` handles thread safety between `AudioManager` (producer) and `transcription_worker` (consumer). Access to the shared `transcribed_chunks` list *might* need a `threading.Lock` if order is critical or more complex operations than `append` are done, but for simple appending by a single worker, the GIL often provides practical safety (though explicit lock is technically safer).
*   **Bluetooth HFP Mode:** Still requires careful management of audio stream lifecycles, especially ensuring input stream is closed before output stream (for beeps) is opened. Check timing in `on_activate`.

---

This updated plan uses the RMS approach, breaks it down into verifiable steps, and prioritizes getting the async structure working before tackling the nuances of RMS tuning and segmentation logic.


