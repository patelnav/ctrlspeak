#!/usr/bin/env python3
"""
Test for MLX models on Apple Silicon.
"""
import unittest
import sys
import platform
import os
from test_transcription import transcribe_audio

class TestMLXTranscription(unittest.TestCase):
    @unittest.skipUnless(
        sys.platform == "darwin" and platform.machine() == "arm64",
        "MLX tests are only supported on Apple Silicon (macOS arm64)"
    )
    def test_parakeet_mlx_transcription(self):
        """Test transcription with the Parakeet MLX model."""
        audio_file = "test.wav"
        self.assertTrue(os.path.exists(audio_file), "Test audio file not found.")
        
        try:
            import mlx
        except ImportError:
            self.fail("MLX dependencies not installed. Please run: uv pip install -r requirements-mlx.txt")
            
        result = transcribe_audio(audio_file, "parakeet-mlx", verbose=True)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        self.assertIsInstance(result[0], str)

if __name__ == "__main__":
    unittest.main()
