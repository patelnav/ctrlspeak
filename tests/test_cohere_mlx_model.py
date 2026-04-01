import pytest
import sys
import json
import tempfile
from unittest.mock import patch, MagicMock
from pathlib import Path
import torch

# Mock MLX globally
mock_mx = MagicMock()
sys.modules['mlx'] = mock_mx
sys.modules['mlx.core'] = mock_mx
sys.modules['mlx.nn'] = mock_mx

from models.cohere_mlx import CohereMLXModel

def test_cohere_mlx_target_source_lang_mismatch():
    model = CohereMLXModel()
    with pytest.raises(ValueError, match="does not support translation"):
        model.transcribe_batch(["test.wav"], source_lang="en", target_lang="de")

@patch("models.cohere_mlx.Model")
def test_cohere_mlx_mock_transcribe(mock_model_class):
    mock_backend = MagicMock()
    mock_model_class.return_value = mock_backend
    
    mock_backend.transcribe.return_value = [" mock transcription "]
    
    model = CohereMLXModel()
    model.backend_model = mock_backend
    
    results = model.transcribe_batch(["dummy.wav"], source_lang="en", target_lang="en")
    
    assert results == ["mock transcription"]
    mock_backend.transcribe.assert_called_once_with(
        language="en",
        audio_files=["dummy.wav"],
        punctuation=True
    )

@patch("models.cohere_mlx.Model")
def test_cohere_mlx_load_uses_pt_safe_open(mock_model_class):
    mock_backend = MagicMock()
    mock_backend.sanitize.return_value = {"weight": MagicMock()}
    mock_backend.parameters.return_value = []
    mock_model_class.return_value = mock_backend
    mock_model_class.post_load_hook.return_value = mock_backend

    mock_safe_open_cm = MagicMock()
    mock_safe_open_cm.__enter__.return_value.keys.return_value = ["weight"]
    mock_safe_open_cm.__enter__.return_value.get_tensor.return_value = torch.ones((1,), dtype=torch.bfloat16)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        (tmp_path / "config.json").write_text(json.dumps({}))
        (tmp_path / "model.safetensors").write_bytes(b"")

        with patch("huggingface_hub.snapshot_download", return_value=str(tmp_path)):
            with patch.dict(sys.modules, {"mlx.utils": MagicMock(tree_unflatten=lambda items: dict(items))}):
                with patch("safetensors.safe_open", return_value=mock_safe_open_cm) as mock_safe_open:
                    model = CohereMLXModel()
                    model.load_model()

    mock_safe_open.assert_called_once_with(str(tmp_path / "model.safetensors"), framework="pt")
