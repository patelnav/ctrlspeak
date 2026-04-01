import torch
import logging
import librosa
from typing import List
from importlib.metadata import version, PackageNotFoundError
from models.base_model import BaseSTTModel

logger = logging.getLogger("cohere_model")

class CohereModel(BaseSTTModel):
    MIN_TRANSFORMERS_VERSION = (5, 4, 0)
    MIN_MISTRAL_COMMON_VERSION = (1, 10, 0)
    SUPPORTED_LANGUAGES = {
        "ar", "de", "el", "en", "es", "fr", "it", 
        "ja", "ko", "nl", "pl", "pt", "vi", "zh"
    }

    def __init__(self, model_name="CohereLabs/cohere-transcribe-03-2026", device=None, verbose=False, **kwargs):
        # Override device preference to CPU first, even if MPS was explicitly passed
        passed_device = getattr(device, 'type', str(device)) if device else 'cpu'
        if passed_device == "mps":
            logger.warning("MPS was passed, but Cohere Transcribe intentionally runs on CPU because MPS is not validated for this model.")
            self.device = torch.device("cpu")
        else:
            self.device = device if device is not None else torch.device("cpu")
            
        super().__init__(device=self.device, verbose=verbose)
        self.model_name = model_name
        self.processor = None
        self.backend_model = None

    def load_model(self):
        try:
            from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
        except ImportError:
            raise ImportError("Cohere model requires transformers. Please install it.")

        try:
            installed_version = version("transformers")
        except PackageNotFoundError as exc:
            raise ImportError("Cohere model requires transformers>=5.4.0.") from exc

        version_parts = tuple(int(part) for part in installed_version.split(".")[:3])
        if version_parts < self.MIN_TRANSFORMERS_VERSION:
            raise ImportError(
                f"Cohere Transcribe requires transformers>=5.4.0 for native loading without trust_remote_code. "
                f"Found transformers=={installed_version}. Upgrade with: uv pip install 'transformers>=5.4.0'"
            )

        try:
            mistral_common_version = version("mistral-common")
        except PackageNotFoundError as exc:
            raise ImportError(
                "Cohere Transcribe requires mistral-common>=1.10.0. "
                "Install with: uv pip install 'mistral-common[image]>=1.10.0'"
            ) from exc

        mistral_common_parts = tuple(int(part) for part in mistral_common_version.split(".")[:3])
        if mistral_common_parts < self.MIN_MISTRAL_COMMON_VERSION:
            raise ImportError(
                f"Cohere Transcribe requires mistral-common>=1.10.0 for tokenizer compatibility. "
                f"Found mistral-common=={mistral_common_version}. "
                f"Upgrade with: uv pip install 'mistral-common[image]>=1.10.0'"
            )
            
        logger.info(f"Loading Cohere processor from {self.model_name}...")
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=False
        )
        
        logger.info(f"Loading Cohere model from {self.model_name} on {self.device}...")
        self.backend_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_name,
            trust_remote_code=False
        ).to(self.device)
        
        self.backend_model.eval()
        logger.info("Cohere model loaded successfully.")

    def transcribe_batch(self, audio_paths: List[str], source_lang: str = "en", target_lang: str = "en", **kwargs) -> List[str]:
        if not audio_paths:
            return []
            
        source_lang = source_lang.lower()
        target_lang = target_lang.lower()
        
        if source_lang not in self.SUPPORTED_LANGUAGES:
            raise ValueError(f"Language '{source_lang}' is not supported by Cohere Transcribe. Supported languages: {', '.join(sorted(self.SUPPORTED_LANGUAGES))}")
            
        if source_lang != target_lang:
            raise ValueError("Cohere Transcribe does not support translation. target_lang must match source_lang.")

        if self.processor is None or self.backend_model is None:
            raise RuntimeError("Model and processor must be loaded before transcribing.")

        results = []
        for path in audio_paths:
            try:
                # Load audio to waveform at 16kHz
                waveform, sr = librosa.load(path, sr=16000)
                
                inputs = self.processor(
                    waveform, 
                    sampling_rate=16000, 
                    return_tensors="pt", 
                    language=source_lang, 
                    punctuation=True
                )
                
                audio_chunk_index = inputs.pop("audio_chunk_index", None)

                # Only tensor model inputs should move onto the device.
                model_inputs = {
                    key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                    for key, value in inputs.items()
                }
                
                if "input_features" in model_inputs and self.backend_model.dtype != model_inputs["input_features"].dtype:
                    model_inputs["input_features"] = model_inputs["input_features"].to(self.backend_model.dtype)

                with torch.no_grad():
                    predicted_ids = self.backend_model.generate(
                        **model_inputs,
                        max_new_tokens=256
                    )

                decode_kwargs = {
                    "skip_special_tokens": True,
                    "language": source_lang
                }
                
                if audio_chunk_index is not None:
                    chunk_idx = audio_chunk_index
                    if isinstance(chunk_idx, torch.Tensor):
                        chunk_idx = chunk_idx.flatten().tolist()
                    elif hasattr(chunk_idx, "tolist"):
                        chunk_idx = chunk_idx.tolist()
                    decode_kwargs["audio_chunk_index"] = chunk_idx
                    transcription = self.processor.decode(predicted_ids[0], **decode_kwargs)
                    if not self._clean_text(transcription):
                        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                else:
                    transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                results.append(self._clean_text(transcription))
            except Exception as e:
                logger.error(f"Error transcribing {path} with Cohere: {e}")
                results.append("")
                
        return results
