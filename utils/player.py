"""
Sound player utility for playing audio feedback.
"""
import os
import threading
import sounddevice as sd
import soundfile as sf
import numpy as np
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
        """Load sound files."""
        if self._sounds_loaded:
            return
            
        # Load start and stop sounds
        start_path = os.path.join(self.base_dir, "on.wav")
        stop_path = os.path.join(self.base_dir, "off.wav")
        
        try:
            # Load start sound
            if os.path.exists(start_path):
                data, samplerate = sf.read(start_path)
                self.sounds['start'] = (data, samplerate)
                logger.debug("Loaded on.wav for start sound")
            else:
                logger.error(f"Start sound file not found: {start_path}")
                return
            
            # Load stop sound
            if os.path.exists(stop_path):
                data, samplerate = sf.read(stop_path)
                self.sounds['stop'] = (data, samplerate)
                logger.debug("Loaded off.wav for stop sound")
            else:
                logger.error(f"Stop sound file not found: {stop_path}")
                return
                
        except Exception as e:
            logger.error(f"Error loading sound files: {e}")
            return
            
        self._sounds_loaded = True
    
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