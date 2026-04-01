import pytest
import sys
from unittest.mock import patch, MagicMock

# Mock transformers globally so it doesn't fail import if run without extras
mock_transformers = MagicMock()
sys.modules['transformers'] = mock_transformers

from models.registry import get_model_metadata
from models.factory import ModelFactory
from models.cohere import CohereModel
import torch

def test_cohere_registry_entry():
    meta = get_model_metadata("cohere")
    assert meta is not None
    assert meta.repo_id == "CohereLabs/cohere-transcribe-03-2026"
    assert meta.backend == "mlx_cohere"

def test_cohere_factory_dispatch():
    with patch("sys.platform", "darwin"), patch("platform.machine", return_value="arm64"):
        with patch("models.cohere_mlx.CohereMLXModel") as mock_mlx_class:
            ModelFactory.get_model("cohere")
            mock_mlx_class.assert_called_once()

def test_cohere_target_source_lang_mismatch():
    model = CohereModel()
    with pytest.raises(ValueError, match="Cohere Transcribe does not support translation"):
        model.transcribe_batch(["test.wav"], source_lang="en", target_lang="de")

@patch.object(mock_transformers, 'AutoProcessor')
@patch.object(mock_transformers, 'AutoModelForSpeechSeq2Seq')
def test_cohere_mocked_backend_load(mock_model_seq, mock_proc):
    mock_backend_model = MagicMock()
    mock_model_seq.from_pretrained.return_value = mock_backend_model
    mock_backend_model.to.return_value = mock_backend_model

    model = CohereModel()
    model.load_model()
    
    mock_proc.from_pretrained.assert_called_with("CohereLabs/cohere-transcribe-03-2026", trust_remote_code=False)

@patch("models.cohere.librosa.load")
@patch.object(mock_transformers, 'AutoProcessor')
@patch.object(mock_transformers, 'AutoModelForSpeechSeq2Seq')
def test_cohere_mocked_backend_transcribe(mock_model_seq, mock_proc, mock_librosa):
    mock_processor = MagicMock()
    mock_proc.from_pretrained.return_value = mock_processor
    
    mock_inputs = {"input_features": torch.tensor([1.0, 2.0]), "audio_chunk_index": torch.tensor([0])}
    mock_processor.return_value = mock_inputs
    mock_processor.decode.return_value = "mock trans"
    
    mock_backend_model = MagicMock()
    mock_backend_model.dtype = torch.float32
    mock_backend_model.to.return_value = mock_backend_model
    mock_model_seq.from_pretrained.return_value = mock_backend_model
    mock_backend_model.generate.return_value = torch.tensor([[1, 2, 3]])
    
    mock_waveform = torch.zeros(16000).numpy()
    mock_librosa.return_value = (mock_waveform, 16000)

    model = CohereModel()
    model.load_model()
    
    results = model.transcribe_batch(["test.wav"], source_lang="en", target_lang="en")
    assert results == ["mock trans"]
    mock_processor.assert_called_with(mock_waveform, sampling_rate=16000, return_tensors="pt", language="en", punctuation=True)
    mock_processor.batch_decode.assert_not_called()
    decode_args, decode_kwargs = mock_processor.decode.call_args
    assert torch.equal(decode_args[0], torch.tensor([1, 2, 3]))
    assert decode_kwargs == {
        "skip_special_tokens": True,
        "language": "en",
        "audio_chunk_index": [0],
    }
