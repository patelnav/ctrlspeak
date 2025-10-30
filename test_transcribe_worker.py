"""Test transcription in worker thread (simulates real usage)."""
import sys
import threading
import queue
import torch
sys.path.insert(0, '/Users/navpatel/Developer/ctrlspeak')

from models.factory import ModelFactory

def worker_transcribe(model_type, audio_file, result_queue):
    """Transcribe in a worker thread (simulates transcription worker)."""
    try:
        device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
        model = ModelFactory.get_model(model_type=model_type, device=device, verbose=False)
        model.load_model()
        print(f'Worker: Model loaded')

        result = model.transcribe(audio_file)
        result_queue.put(('success', result))
        print(f'Worker: Transcription successful')
    except Exception as e:
        result_queue.put(('error', str(e)))
        print(f'Worker: Error - {e}')

# Test with NVIDIA model
print('Testing NVIDIA parakeet in worker thread...')
result_q = queue.Queue()
worker = threading.Thread(target=worker_transcribe, args=('nvidia/parakeet-tdt-0.6b-v3', 'test.wav', result_q))
worker.start()
worker.join(timeout=60)

if worker.is_alive():
    print('ERROR: Worker timed out!')
else:
    status, result = result_q.get()
    if status == 'success':
        print(f'✓ SUCCESS: {result}')
    else:
        print(f'✗ FAILED: {result}')
