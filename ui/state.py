"""
Centralized application state for ctrlSPEAK Textual UI.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from textual.reactive import Reactive


@dataclass
class DeviceInfo:
    """Information about an audio device."""
    id: int
    name: str
    channels: int
    sample_rate: int
    is_default: bool = False


class AppState:
    """
    Centralized, reactive application state for the Textual UI.
    Uses Textual's reactive system for automatic UI updates.
    """

    def __init__(self):
        # Recording state
        self.is_recording: bool = False
        self.audio_duration_s: float = 0.0
        self.buffer_size_samples: int = 0
        self.current_rms: float = 0.0
        self.recording_start_time: Optional[float] = None
        self.current_silence_s: float = 0.0

        # Device state
        self.selected_device: Optional[int] = None
        self.available_devices: List[DeviceInfo] = []

        # Settings state
        self.rms_threshold: float = 0.01
        self.silence_duration_s: float = 1.0
        self.min_chunk_duration_s: float = 0.5
        self.selected_model: str = "parakeet"
        self.source_lang: str = "en"
        self.target_lang: str = "en"

        # UI state
        self.current_screen: str = "recording"
        self.transcription_text: str = ""
        self.last_transcription: str = ""

        # Statistics
        self.total_transcriptions: int = 0
        self.total_recording_time_s: float = 0.0

    def reset_recording_state(self):
        """Reset recording-specific state."""
        self.is_recording = False
        self.audio_duration_s = 0.0
        self.buffer_size_samples = 0
        self.current_rms = 0.0
        self.recording_start_time = None

    def update_from_audio_manager(self, audio_manager):
        """Update state from AudioManager instance."""
        self.is_recording = audio_manager.is_collecting
        self.buffer_size_samples = len(audio_manager.audio_buffer) if audio_manager.audio_buffer else 0
        self.rms_threshold = audio_manager.RMS_THRESHOLD
        self.silence_duration_s = audio_manager.SILENCE_DURATION_S
        self.min_chunk_duration_s = audio_manager.MIN_CHUNK_DURATION_S

        if audio_manager.recording_start_time:
            self.recording_start_time = audio_manager.recording_start_time
