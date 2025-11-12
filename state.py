
import queue
from rich.console import Console

# Global variables
startup_time = 0
console = Console()
print = console.print

DEBUG_MODE = False

# Model identifiers - source of truth for all model references in the codebase
MLX_PARAKEET_V3 = "mlx-community/parakeet-tdt-0.6b-v3"
MLX_PARAKEET_V2 = "mlx-community/parakeet-tdt-0.6b-v2"
NVIDIA_PARAKEET_V3 = "nvidia/parakeet-tdt-0.6b-v3"
NVIDIA_PARAKEET_V2 = "nvidia/parakeet-tdt-0.6b-v2"
NVIDIA_CANARY_1B_FLASH = "nvidia/canary-1b-flash"
NVIDIA_CANARY_180M = "nvidia/canary-180m-flash"
NVIDIA_CANARY_V2 = "nvidia/canary-1b-v2"
OPENAI_WHISPER_V3 = "openai/whisper-large-v3"

# Set of all known/supported models
KNOWN_MODELS = {
    MLX_PARAKEET_V3,
    MLX_PARAKEET_V2,
    NVIDIA_PARAKEET_V3,
    NVIDIA_PARAKEET_V2,
    NVIDIA_CANARY_1B_FLASH,
    NVIDIA_CANARY_180M,
    NVIDIA_CANARY_V2,
    OPENAI_WHISPER_V3,
}

audio_manager = None
stt_model = None
model_type = "parakeet"
model_loaded = False

transcribed_chunks = []
transcription_queue = queue.Queue()
transcription_worker_thread = None
main_loop_active = True

keyboard_manager = None

device = None

source_lang = "en"
target_lang = "en"

# Recording timing
recording_start_time = None
