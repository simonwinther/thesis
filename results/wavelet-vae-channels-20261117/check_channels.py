"""Small CPU checks for configurable latent width and checkpoint roundtrips."""
import json
import sys
import tempfile
from pathlib import Path
import torch
sys.path.insert(0, sys.argv[1])
from checks import check_models
from model import build_model, model_config, reconstruct
from train import checkpoint
from evaluate import load_model

torch.set_num_threads(2)
results = {}
for width in (2, 4, 8):
    result = check_models(width)
    model = build_model('lf', width).eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    config = {'band': 'lf', 'latent_channels': width}
    x = torch.randn(1, 1, 16, 24, 16)
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / 'checkpoint.pt'
        checkpoint(path, model, optimizer, 1, config, 'fixture', 1.)
        loaded, state = load_model(path, 'lf', 'cpu', 'fixture')
        assert state['model_config'] == model_config(width)
        with torch.no_grad():
            torch.testing.assert_close(reconstruct(model, x), reconstruct(loaded, x), rtol=0, atol=0)
        if width == 2:
            del state['config']['latent_channels']
            torch.save(state, path)
            load_model(path, 'lf', 'cpu', 'fixture')
        state['config']['latent_channels'] = width + 1
        torch.save(state, path)
        try:
            load_model(path, 'lf', 'cpu', 'fixture')
        except ValueError:
            pass
        else:
            raise AssertionError('Corrupt width metadata accepted')
    results[str(width)] = result
print(json.dumps({'passed': True, 'widths': results}, indent=2))
