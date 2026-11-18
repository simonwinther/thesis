"""Check encoder-only ablations, gradients and checkpoint provenance on CPU."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import torch
sys.path.insert(0, sys.argv[1])
from model import build_model, model_config, reconstruct, vae_loss
from train import checkpoint
from evaluate import load_model

torch.set_num_threads(2)
spec=importlib.util.spec_from_file_location('previous_model', Path(sys.argv[2])/'model.py')
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
results={}
for band, inputs in (('lf',1),('hf',7)):
    torch.manual_seed(43); reference=previous.build_model(band,8)
    torch.manual_seed(43); baseline=build_model(band,8); baseline_rng=torch.get_rng_state()
    assert all(torch.equal(v,reference.state_dict()[k]) for k,v in baseline.state_dict().items())
    baseline_types=[type(m).__name__ for m in baseline.encoder.modules()]
    for widths in (None,(64,128,128),(128,256,256)):
        torch.manual_seed(43); model=build_model(band,8,widths)
        assert torch.equal(torch.get_rng_state(),baseline_rng)
        assert [type(m).__name__ for m in model.encoder.modules()]==baseline_types
        assert all(torch.equal(v,baseline.state_dict()[k]) for k,v in model.state_dict().items() if not k.startswith('encoder.'))
        assert model.encoder.blocks[0].conv.out_channels==(32 if widths is None else widths[0])
        x=torch.randn(1,inputs,16,24,16)
        loss,pred,mu,std,rec,kl=vae_loss(model,x)
        assert pred.shape==x.shape and mu.shape==(1,8,4,6,4)
        assert torch.isfinite(loss) and torch.isfinite(kl) and (std>0).all()
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        for component in (model.encoder,model.decoder,model.quant_conv_mu,model.quant_conv_log_sigma):
            assert sum(float(p.grad.abs().sum()) for p in component.parameters())>0
        optimizer=torch.optim.Adam(model.parameters(),lr=1e-4);optimizer.step();model.eval()
        config={'band':band,'latent_channels':8,'encoder_channels':widths}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'checkpoint.pt'
            checkpoint(path,model,optimizer,1,config,'fixture',1.)
            loaded,state=load_model(path,band,'cpu','fixture')
            assert state['model_config']==model_config(8,widths)
            with torch.no_grad():
                torch.testing.assert_close(reconstruct(model,x),reconstruct(loaded,x),rtol=0,atol=0)
            if widths is None:
                del state['config']['encoder_channels'];torch.save(state,path)
                load_model(path,band,'cpu','fixture')
            state['config']['encoder_channels']=(256,512,512)
            torch.save(state,path)
            try: load_model(path,band,'cpu','fixture')
            except ValueError: pass
            else: raise AssertionError('Contradictory encoder metadata accepted')
        key=band+':'+str(widths)
        results[key]={'encoder_parameters':sum(p.numel() for p in model.encoder.parameters()),
                      'decoder_parameters':sum(p.numel() for p in model.decoder.parameters()),
                      'total_parameters':sum(p.numel() for p in model.parameters()),'initial_nonencoder_weights_identical':True,
                      'depth_identical':True,'latent_shape':[8,4,6,4],'checkpoint_roundtrip':True}
print(json.dumps({'passed':True,'results':results},indent=2))
