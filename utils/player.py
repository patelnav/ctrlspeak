"""
Sound player utility for playing audio feedback.
"""
import os
import threading
import sounddevice as sd
import soundfile as sf
import numpy as np
from pydub import AudioSegment
import io
import tempfile
import logging

# Configure logger
logger = logging.getLogger("sound_player")

class SoundPlayer:
    """Class for playing sound effects."""
    
    def __init__(self):
        """Initialize the sound player."""
        self.sounds = {}
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self._sounds_loaded = False
    
    def _load_sounds(self):
        """Load all sound files."""
        if self._sounds_loaded:
            return
            
        # Use the sound.mp3 file for both start and stop (reversed for stop)
        sound_path = os.path.join(self.base_dir, "sound.mp3")
        
        if os.path.exists(sound_path) and os.path.getsize(sound_path) > 200:
            try:
                # Load the MP3 file using pydub
                audio = AudioSegment.from_mp3(sound_path)
                
                # Convert to numpy array for start sound
                samples = np.array(audio.get_array_of_samples())
                if audio.channels == 2:
                    samples = samples.reshape((-1, 2))
                
                # Store the start sound
                self.sounds['start'] = (samples, audio.frame_rate)
                logger.debug("Loaded sound.mp3 for start sound")
                
                # Create reversed version for stop sound
                reversed_audio = audio.reverse()
                reversed_samples = np.array(reversed_audio.get_array_of_samples())
                if reversed_audio.channels == 2:
                    reversed_samples = reversed_samples.reshape((-1, 2))
                
                # Store the stop sound (reversed)
                self.sounds['stop'] = (reversed_samples, reversed_audio.frame_rate)
                logger.debug("Created reversed version for stop sound")
                
            except Exception as e:
                logger.error(f"Error loading sound.mp3: {e}")
                self._create_fallback_sound('start')
                self._create_fallback_sound('stop')
        else:
            logger.debug(f"Sound file not found or empty: {sound_path}")
            self._create_fallback_sound('start')
            self._create_fallback_sound('stop')
            
        self._sounds_loaded = True
    
    def _create_fallback_sound(self, name):
        """Create a fallback sound if the sound file cannot be loaded."""
        # Create a simple beep sound
        samplerate = 44100
        duration = 0.2  # seconds
        frequency = 1000 if name == 'start' else 800  # Hz
        
        t = np.linspace(0, duration, int(samplerate * duration), False)
        # Apply a simple envelope to avoid clicks
        envelope = np.ones_like(t)
        # Apply fade in and fade out (10% of the duration)
        fade_samples = int(duration * samplerate * 0.1)
        envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
        envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
        
        data = 0.5 * np.sin(2 * np.pi * frequency * t) * envelope
        self.sounds[name] = (data, samplerate)
        logger.debug(f"Created fallback sound for: {name}")
    
    def play(self, sound_name):
        """Play a sound by name.
        
        Args:
            sound_name: Name of the sound to play ('start' or 'stop').
        """
        # Ensure sounds are loaded before playing
        self._load_sounds()
        
        if sound_name not in self.sounds:
            logger.error(f"Sound '{sound_name}' not found")
            return
        
        # Play sound in a separate thread to avoid blocking
        threading.Thread(target=self._play_sound, args=(sound_name,), daemon=True).start()
    
    def _play_sound(self, sound_name):
        """Internal method to play a sound.
        
        Args:
            sound_name: Name of the sound to play.
        """
        data, samplerate = self.sounds[sound_name]
        try:
            sd.play(data, samplerate)
            sd.wait()  # Wait until sound has finished playing
        except Exception as e:
            logger.error(f"Error playing sound {sound_name}: {e}")

# Singleton instance
player = SoundPlayer()

# Convenience functions
def play_start_beep():
    """Play the start recording beep."""
    player.play('start')

def play_stop_beep():
    """Play the stop recording beep."""
    player.play('stop') 