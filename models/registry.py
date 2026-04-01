from dataclasses import dataclass
from typing import Dict, List, Optional
import state

@dataclass
class ModelMetadata:
    alias: str
    repo_id: str
    backend: str
    description: str
    requires: str
    supports_translation: bool
    supports_streaming: bool
    preferred_device_strategy: str

MODEL_REGISTRY: Dict[str, ModelMetadata] = {
    "parakeet-v3-mlx": ModelMetadata("parakeet-v3-mlx", state.MLX_PARAKEET_V3, "mlx_parakeet", "Apple Silicon optimized", "mlx", False, False, "mps"),
    "parakeet-v3": ModelMetadata("parakeet-v3", state.NVIDIA_PARAKEET_V3, "nemo_parakeet", "NVIDIA server-grade", "nemo", False, False, "cuda/mps"),
    "parakeet-v2-mlx": ModelMetadata("parakeet-v2-mlx", state.MLX_PARAKEET_V2, "mlx_parakeet", "Apple Silicon optimized", "mlx", False, False, "mps"),
    "parakeet-v2": ModelMetadata("parakeet-v2", state.NVIDIA_PARAKEET_V2, "nemo_parakeet", "NVIDIA server-grade", "nemo", False, False, "cuda/mps"),
    "parakeet": ModelMetadata("parakeet", state.MLX_PARAKEET_V3, "mlx_parakeet", "Default Parakeet", "mlx", False, False, "mps"),
    "canary": ModelMetadata("canary", state.NVIDIA_CANARY_1B_FLASH, "nemo_canary", "Canary 1B Flash", "nemo", True, False, "cuda/mps"),
    "canary-180m": ModelMetadata("canary-180m", state.NVIDIA_CANARY_180M, "nemo_canary", "Canary 180M", "nemo", True, False, "cuda/mps"),
    "canary-v2": ModelMetadata("canary-v2", state.NVIDIA_CANARY_V2, "nemo_canary", "Canary V2", "nemo", True, False, "cuda/mps"),
    "nemotron": ModelMetadata("nemotron", state.NVIDIA_NEMOTRON_STREAMING, "nemo_nemotron", "Nemotron Streaming", "nemo", False, True, "cuda/mps"),
    "whisper": ModelMetadata("whisper", state.OPENAI_WHISPER_V3, "transformers_whisper", "Whisper Large V3", "transformers", True, False, "mps/cuda/cpu"),
    "cohere": ModelMetadata("cohere", state.COHERE_TRANSCRIBE, "mlx_cohere", "Cohere Transcribe 03-2026 (Apple Silicon)", "cohere-mlx", False, False, "mps"),
}

def get_model_metadata(alias_or_repo_id: str) -> Optional[ModelMetadata]:
    alias_lower = alias_or_repo_id.lower()
    if alias_lower in MODEL_REGISTRY:
        return MODEL_REGISTRY[alias_lower]
        
    for meta in MODEL_REGISTRY.values():
        if meta.repo_id.lower() == alias_lower:
            return meta
            
    return None
