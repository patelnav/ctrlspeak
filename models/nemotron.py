"""
Nemotron Speech Streaming model implementation for speech-to-text.

This model uses NVIDIA's FastConformer-CacheAware-RNNT architecture
optimized for streaming transcription with native punctuation.

Supports two modes:
1. Batch transcription: Traditional file-based transcription via transcribe_batch()
2. Streaming transcription: Real-time incremental transcription via streaming API
"""
import time
import nemo.collections.asr as nemo_asr
from models.base_model import BaseSTTModel
import torch
import numpy as np
import logging
import os
from typing import List, Optional, Any

# Configure logging
logger = logging.getLogger("nemotron_model")

# Streaming configuration defaults
DEFAULT_CHUNK_SIZE_MS = 1120  # 14 frames at 80ms each - best accuracy
SAMPLE_RATE = 16000  # Required by NeMo models


class NemotronModel(BaseSTTModel):
    """Nemotron Speech Streaming model for speech-to-text.

    Uses FastConformer-CacheAware-RNNT architecture with native
    punctuation and capitalization support.
    """

    def __init__(self, model_name="nvidia/nemotron-speech-streaming-en-0.6b", device=None, verbose=False):
        """Initialize the Nemotron model.

        Args:
            model_name: The name of the pretrained model to load.
            device: The device to run the model on (handled automatically by NeMo).
            verbose: Whether to enable verbose logging.
        """
        super().__init__(device)
        self.model_name = model_name
        self.model = None
        self.verbose = verbose

        # Streaming state (initialized in init_streaming)
        self._streaming_active = False
        self._cache_last_channel: Optional[torch.Tensor] = None
        self._cache_last_time: Optional[torch.Tensor] = None
        self._cache_last_channel_len: Optional[torch.Tensor] = None
        self._previous_hypotheses: Optional[Any] = None
        self._pred_out_stream: Optional[Any] = None
        self._accumulated_text: str = ""
        self._chunk_size_ms: int = DEFAULT_CHUNK_SIZE_MS
        self._step_num: int = 0  # Track step number for drop_extra_pre_encoded

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
            # Load model using NeMo's generic ASR model loader
            # Nemotron uses ASRModel.from_pretrained() unlike Parakeet's EncDecRNNTBPEModel
            logger.debug("Initializing ASRModel (FastConformer-CacheAware-RNNT)...")
            self.model = nemo_asr.models.ASRModel.from_pretrained(self.model_name)

            load_time = time.time() - start_time
            logger.info(f"Model loaded in {load_time:.2f} seconds")
            return self.model

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def transcribe_batch(self, audio_paths: List[str], **kwargs) -> List[str]:
        """Transcribe multiple audio files in batch.

        Args:
            audio_paths: List of paths to audio files.
            **kwargs: Additional arguments (ignored for compatibility).

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

            # Transcribe using NeMo's transcribe method
            logger.debug("Starting transcription...")
            transcribe_start = time.time()

            # Pass verbose=False to disable tqdm progress bar which causes multiprocessing issues
            raw_transcriptions = self.model.transcribe(audio_paths, verbose=False)

            # NeMo 2.2.0+ returns Hypothesis objects, so we need special handling
            is_hypothesis = False
            if raw_transcriptions and isinstance(raw_transcriptions, list):
                if len(raw_transcriptions) > 0 and hasattr(raw_transcriptions[0], 'text'):
                    is_hypothesis = True

            # Handle the Hypothesis objects if needed
            if is_hypothesis:
                logger.debug("Handling Hypothesis objects from NeMo 2.2.0+")
                transcriptions = [hyp.text for hyp in raw_transcriptions if hasattr(hyp, 'text')]
            else:
                # Clean up each result using the standard method
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
            # Suppress tqdm multiprocessing errors (non-fatal)
            if "fds_to_keep" not in str(e):
                logger.error(f"Error during transcription: {str(e)}")
            raise

    def name(self):
        """Return the name of the model."""
        simple_name = self.model_name.split('/')[-1]
        return f"Nemotron-{simple_name}"

    # =========================================================================
    # Streaming Interface Implementation
    # =========================================================================

    @property
    def supports_streaming(self) -> bool:
        """Nemotron supports streaming transcription."""
        return True

    @property
    def chunk_size_ms(self) -> int:
        """Get the configured chunk size in milliseconds."""
        return self._chunk_size_ms

    @chunk_size_ms.setter
    def chunk_size_ms(self, value: int) -> None:
        """Set the chunk size in milliseconds.

        Valid values align with model's attention context sizes:
        - 80ms (1 frame) - ultra-low latency
        - 160ms (2 frames)
        - 560ms (7 frames) - default, good balance
        - 1120ms (14 frames) - best accuracy
        """
        if value not in [80, 160, 560, 1120]:
            logger.warning(f"Non-standard chunk size {value}ms. Recommended: 80, 160, 560, 1120")
        self._chunk_size_ms = value

    def init_streaming(self) -> None:
        """Initialize streaming state for a new recording session.

        Sets up encoder cache state for incremental processing.
        Must be called before the first stream_chunk() call.
        """
        if self.model is None:
            self.load_model()

        logger.info("Initializing streaming transcription...")

        try:
            # Put model in eval mode
            self.model.eval()

            # Get initial cache state from encoder
            (
                self._cache_last_channel,
                self._cache_last_time,
                self._cache_last_channel_len,
            ) = self.model.encoder.get_initial_cache_state(batch_size=1)

            # Reset streaming state
            self._previous_hypotheses = None
            self._pred_out_stream = None
            self._accumulated_text = ""
            self._streaming_active = True
            self._step_num = 0  # Reset step counter

            # Configure attention context based on chunk size
            # att_context_size = [left_context, right_context]
            # right_context determines chunk size: chunk_frames = right_context + 1
            chunk_frames = self._chunk_size_ms // 80
            right_context = chunk_frames - 1
            left_context = 70  # Standard left context

            if hasattr(self.model.encoder, 'set_default_att_context_size'):
                self.model.encoder.set_default_att_context_size([left_context, right_context])
                logger.debug(f"Set attention context: [{left_context}, {right_context}]")

            # Configure preprocessor for streaming (disable dither, pad_to, normalize)
            if hasattr(self.model, 'preprocessor') and self.model.preprocessor is not None:
                if hasattr(self.model.preprocessor, 'featurizer'):
                    self.model.preprocessor.featurizer.dither = 0.0
                    self.model.preprocessor.featurizer.pad_to = 0
                    logger.debug("Configured preprocessor for streaming (dither=0, pad_to=0)")

            # Log streaming config if available
            if hasattr(self.model.encoder, 'streaming_cfg') and self.model.encoder.streaming_cfg is not None:
                cfg = self.model.encoder.streaming_cfg
                logger.debug(f"Streaming config: drop_extra_pre_encoded={getattr(cfg, 'drop_extra_pre_encoded', 'N/A')}")

            logger.info(f"Streaming initialized (chunk size: {self._chunk_size_ms}ms)")

        except Exception as e:
            logger.error(f"Failed to initialize streaming: {e}")
            self._streaming_active = False
            raise

    def stream_chunk(self, audio_samples: np.ndarray, is_final: bool = False) -> str:
        """Process a chunk of audio and return incremental transcription.

        Args:
            audio_samples: Audio samples as float32 numpy array (16kHz mono).
                          Should be approximately chunk_size_ms worth of audio.
            is_final: If True, this is the last chunk - enables keep_all_outputs
                     for proper flushing of the decoder state.

        Returns:
            Transcribed text from this chunk. May be empty if no speech detected.
            Returns cumulative transcription, not just the delta.
        """
        if not self._streaming_active:
            raise RuntimeError("Streaming not initialized. Call init_streaming() first.")

        if audio_samples is None or len(audio_samples) == 0:
            return self._accumulated_text

        try:
            # For final chunks, pad short audio to minimum size for better decoding
            # Minimum ~500ms (8000 samples) helps the model have enough context
            MIN_FINAL_SAMPLES = 8000
            if is_final and len(audio_samples) < MIN_FINAL_SAMPLES:
                padding_needed = MIN_FINAL_SAMPLES - len(audio_samples)
                audio_samples = np.pad(audio_samples, (0, padding_needed), mode='constant', constant_values=0)
                logger.debug(f"[STREAM] Padded final chunk to {len(audio_samples)} samples")

            # Preprocess audio to model input format
            processed_signal, processed_signal_length = self._preprocess_audio(audio_samples)

            # Log cache state for debugging
            cache_shape = self._cache_last_channel.shape if self._cache_last_channel is not None else "None"
            logger.debug(f"[STREAM] Step {self._step_num}, cache shape: {cache_shape}")

            # Calculate drop_extra_pre_encoded based on step number
            # First step: 0, subsequent steps: use model's config value
            if self._step_num == 0:
                drop_extra = 0
            else:
                # Get from model's streaming config if available
                if hasattr(self.model.encoder, 'streaming_cfg') and self.model.encoder.streaming_cfg is not None:
                    drop_extra = getattr(self.model.encoder.streaming_cfg, 'drop_extra_pre_encoded', 0)
                else:
                    drop_extra = 0

            logger.debug(f"[STREAM] drop_extra_pre_encoded={drop_extra}, is_final={is_final}")

            # Run streaming inference step
            # keep_all_outputs=True for final chunk to flush decoder state
            with torch.no_grad():
                (
                    self._pred_out_stream,
                    transcribed_texts,
                    self._cache_last_channel,
                    self._cache_last_time,
                    self._cache_last_channel_len,
                    self._previous_hypotheses,
                ) = self.model.conformer_stream_step(
                    processed_signal=processed_signal,
                    processed_signal_length=processed_signal_length,
                    cache_last_channel=self._cache_last_channel,
                    cache_last_time=self._cache_last_time,
                    cache_last_channel_len=self._cache_last_channel_len,
                    keep_all_outputs=is_final,  # True for final chunk to flush
                    previous_hypotheses=self._previous_hypotheses,
                    previous_pred_out=self._pred_out_stream,
                    drop_extra_pre_encoded=drop_extra,
                    return_transcription=True,
                )

            # Increment step counter
            self._step_num += 1

            # Log cache state after for debugging
            cache_shape_after = self._cache_last_channel.shape if self._cache_last_channel is not None else "None"
            logger.debug(f"[STREAM] Cache shape after: {cache_shape_after}")

            # Extract text from result
            # conformer_stream_step returns Hypothesis objects, not strings
            if transcribed_texts and len(transcribed_texts) > 0:
                hypothesis = transcribed_texts[0]
                # Handle Hypothesis object
                if hasattr(hypothesis, 'text'):
                    new_text = hypothesis.text
                elif isinstance(hypothesis, str):
                    new_text = hypothesis
                else:
                    new_text = str(hypothesis)

                if new_text:
                    self._accumulated_text = new_text
                    logger.debug(f"Streaming transcription: {new_text}")

            return self._accumulated_text

        except Exception as e:
            logger.error(f"Error in stream_chunk: {e}")
            # Don't raise - return what we have so far
            return self._accumulated_text

    def finalize_streaming(self) -> str:
        """Finalize streaming session and return final transcription.

        Cleans up streaming state and returns the accumulated text.
        Should be called when recording stops.

        Returns:
            Complete transcribed text from the streaming session.
        """
        if not self._streaming_active:
            logger.warning("finalize_streaming called but streaming not active")
            return ""

        logger.info("Finalizing streaming transcription...")

        final_text = self._accumulated_text

        # Reset streaming state
        self._streaming_active = False
        self._cache_last_channel = None
        self._cache_last_time = None
        self._cache_last_channel_len = None
        self._previous_hypotheses = None
        self._pred_out_stream = None
        self._accumulated_text = ""

        if final_text and len(final_text) > 50:
            logger.info(f"Streaming finalized. Result: {final_text[:50]}...")
        else:
            logger.info(f"Streaming finalized. Result: {final_text}")

        return final_text

    def _preprocess_audio(self, audio_samples: np.ndarray) -> tuple:
        """Preprocess audio samples for model input.

        Args:
            audio_samples: Raw audio as float32 numpy array (16kHz mono).

        Returns:
            Tuple of (processed_signal, processed_signal_length) tensors.
        """
        # Ensure float32
        if audio_samples.dtype != np.float32:
            audio_samples = audio_samples.astype(np.float32)

        # Normalize if needed (expecting -1.0 to 1.0 range)
        if np.abs(audio_samples).max() > 1.0:
            audio_samples = audio_samples / 32768.0

        # Convert to tensor [batch, time]
        audio_tensor = torch.from_numpy(audio_samples).unsqueeze(0)

        # Get signal length
        audio_length = torch.tensor([audio_tensor.shape[1]], dtype=torch.long)

        # Use model's preprocessor if available
        if hasattr(self.model, 'preprocessor') and self.model.preprocessor is not None:
            processed_signal, processed_signal_length = self.model.preprocessor(
                input_signal=audio_tensor,
                length=audio_length,
            )
        else:
            # Fallback: use raw audio
            processed_signal = audio_tensor
            processed_signal_length = audio_length

        return processed_signal, processed_signal_length
