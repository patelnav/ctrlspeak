
import queue
from rich.console import Console

# Global variables
startup_time = 0
console = Console()
print = console.print

DEBUG_MODE = False

MODEL_CACHE_MAP = {
    "models--nvidia--parakeet-tdt-0.6b-v3": "nvidia/parakeet-tdt-0.6b-v3",
    "models--nvidia--canary-1b": "nvidia/canary-1b",
    "models--openai--whisper-large-v3-turbo": "openai/whisper-large-v3"
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
