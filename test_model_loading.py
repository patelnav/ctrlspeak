"""
Unit tests for model loading and error handling.
"""
import pytest
import sys
from unittest.mock import Mock, patch
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_nvidia_model_without_nemo():
    """Test that NVIDIA models fail gracefully without nemo-toolkit installed."""
    from models.factory import ModelFactory

    # Mock the nemo import to fail
    with patch.dict('sys.modules', {'nemo.collections.asr': None}):
        with pytest.raises(ImportError) as exc_info:
            model = ModelFactory.get_model(
                model_type="nvidia/parakeet-tdt-0.6b-v3",
                device=None,
                verbose=False
            )

        # Check that error message is helpful
        assert "nemo-toolkit" in str(exc_info.value).lower()
        assert "reinstall" in str(exc_info.value).lower() or "install" in str(exc_info.value).lower()


def test_mlx_model_loads():
    """Test that MLX models can be created (not loaded, just instantiated)."""
    from models.factory import ModelFactory
    import platform

    # Only run on Apple Silicon
    if sys.platform != "darwin" or platform.machine() != "arm64":
        pytest.skip("MLX only supported on Apple Silicon")

    # This should not raise
    model = ModelFactory.get_model(
        model_type="mlx-community/parakeet-tdt-0.6b-v3",
        device=None,
        verbose=False
    )

    assert model is not None
    assert hasattr(model, 'load_model')


def test_model_alias_resolution():
    """Test that model aliases resolve correctly."""
    from models.factory import ModelFactory
    from state import NVIDIA_PARAKEET_V3, MLX_PARAKEET_V3

    # Test alias resolution
    assert ModelFactory.resolve_model_alias("parakeet") == MLX_PARAKEET_V3
    assert ModelFactory.resolve_model_alias("parakeet-v3") == NVIDIA_PARAKEET_V3
    assert ModelFactory.resolve_model_alias("parakeet-v3-mlx") == MLX_PARAKEET_V3

    # Test that full model names pass through
    assert ModelFactory.resolve_model_alias(NVIDIA_PARAKEET_V3) == NVIDIA_PARAKEET_V3


def test_get_model_raises_on_import_error():
    """Test that get_model() raises ModelLoadError when dependencies are missing."""
    import state
    from model_loader import get_model, ModelLoadError

    # Set up state for a NVIDIA model
    state.model_type = "nvidia/parakeet-tdt-0.6b-v3"
    state.device = None
    state.DEBUG_MODE = False
    state.stt_model = None

    # Mock the nemo import to fail
    with patch('models.factory.ModelFactory.get_model', side_effect=ImportError("nemo-toolkit not found")):
        with pytest.raises(ModelLoadError) as exc_info:
            get_model()

        # Check that error message is passed through
        assert "nemo-toolkit" in str(exc_info.value).lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
