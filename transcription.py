
import time
import os
import tempfile
import logging
import numpy as np
import soundfile as sf
import state
from utils.audio import SAMPLE_RATE

logger = logging.getLogger("ctrlspeak")

def transcription_worker(model, work_queue, results_list, source_lang, target_lang):
    """
    Pulls audio data from queue, transcribes using the real model (via temp file),
    adds text to results_list. Runs in a separate thread until None is received.
    """
    logger.debug("Transcription worker thread started.")

    while True: 
        audio_data = None
        temp_file_path = None
        try:
            audio_data = work_queue.get() 
            
            if audio_data is None:
                logger.info("Worker received None sentinel. Exiting loop.")
                work_queue.task_done()
                logger.debug("Worker thread loop terminating.")
                break
            
            logger.debug(f"Worker received chunk of type {type(audio_data)} and shape {getattr(audio_data, 'shape', 'N/A')}")

            if len(audio_data) == 0:
                 logger.warning("Worker received empty audio data array, skipping.")
                 work_queue.task_done()
                 continue

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                temp_file_path = tmp.name
            logger.debug(f"Worker created temp file: {temp_file_path}")

            try:
                if audio_data.dtype != np.float32:
                    logger.warning(f"Audio data was {audio_data.dtype}, attempting conversion to float32 for sf.write")
                    audio_data = audio_data.astype(np.float32)
                
                sf.write(temp_file_path, audio_data, SAMPLE_RATE)
                logger.debug(f"Worker successfully wrote {len(audio_data)} samples to {temp_file_path}")
            except Exception as write_e:
                logger.error(f"Worker failed to write temp WAV file {temp_file_path}: {write_e}", exc_info=True)
                work_queue.task_done()
                if temp_file_path and os.path.exists(temp_file_path):
                     try: os.unlink(temp_file_path)
                     except Exception: pass
                continue

            logger.debug(f"Worker calling model.transcribe() for {temp_file_path}...")
            transcription_start_time = time.time()
            try:
                 results = model.transcribe_batch([temp_file_path], source_lang=source_lang, target_lang=target_lang)
                 if results and isinstance(results, list):
                      text = results[0]
                 else:
                      text = None
                      logger.warning(f"Worker received unexpected result type from transcribe_batch: {type(results)}")
                 transcription_duration = time.time() - transcription_start_time
                 logger.info(f"Worker transcribed chunk in {transcription_duration:.2f}s: {text[:30]}...")
            except Exception as transcribe_e:
                 logger.error(f"Worker: Error during model transcription: {transcribe_e}", exc_info=True)
                 text = None

            if text:
                state.console.print(f"\n[dim]{text}[/dim]")
                results_list.append(text)

            work_queue.task_done()

        except Exception as e:
            logger.error(f"Unexpected error in transcription worker loop (before finally): {e}", exc_info=True)
            if audio_data is not None:
                try:
                    work_queue.task_done()
                except ValueError:
                    pass 
            time.sleep(0.1)
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                    logger.debug(f"Worker deleted temp file: {temp_file_path}")
                except Exception as del_e:
                     logger.error(f"Worker failed to delete temp file {temp_file_path}: {del_e}")

    logger.info("Transcription worker thread finished normally.")
