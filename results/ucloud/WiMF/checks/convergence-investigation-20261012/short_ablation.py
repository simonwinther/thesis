"""Bounded full-volume diagnostic: weights-only initialization, NOT resume.

All variants start from the same step12626 model with a fresh Adam optimizer.
Train on one fixed real subject, one row per target modality [B=4], for150
updates. This deliberately tests recovery of endpoint synthesis, not general
convergence. Evaluate a separate fixed validation subject before/after; never
optimize it. Saved old checkpoints keep their original time scale unchanged.
"""
import gc
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0,'/home/ucloud/test/WiMF')
from guided_diffusion.wimf import WaveletImprovedMeanFlow, apply_idwt
from guided_diffusion.wimf_model import create_wimf_model
from guided_diffusion.wimf_metrics import reconstruction_metrics
from scripts.wimf_data import WiMFDataset, ALL_MODALITIES

ROOT = Path(__file__).resolve().parent
CKPT = Path('/work/WMF/WiMF/runs/brats-gli-2023-fast-20261012-175231/checkpoints/wimf_00012626.pt')
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def prepare(dataset,index):
    """Create four independent target/condition rows from one registered MRI."""
    record = dataset[index]
    images = {m:record[m].unsqueeze(0).cuda() for m in ALL_MODALITIES}
    flow = WaveletImprovedMeanFlow()
    tuples = [flow.prepare_batch(images,m) for m in ALL_MODALITIES]
    tensors = [torch.cat([row[k] for row in tuples]) for k in range(4)]
    return record['subject'],images,tensors


@torch.no_grad()
def scores(model,flow,images,tensors,full=False):
    """Evaluate deterministic fixed-noise 1NFE and true brain reconstruction."""
    x,c,source,y = tensors
    generator = torch.Generator(device='cuda').manual_seed(1009)
    noise = torch.randn((4,1,224,224,160),generator=generator,device='cuda')
    output = apply_idwt(flow.sample(model,c,source,y,noise=noise))
    result = {}
    for k,m in enumerate(ALL_MODALITIES):
        first = next(v for v in ALL_MODALITIES if v != m)
        prediction = output[k:k+1].clamp(0,1).masked_fill(images[first]==0,0)
        target = images[m]
        mask = target>0
        error = (prediction-target).square()
        item = dict(mse=error.mean().item(),brain_mse=error[mask].mean().item())
        if full:
            item = reconstruction_metrics(prediction.cpu().numpy().squeeze(),target.cpu().numpy().squeeze())
        result[m] = item
    return result


def main():
    checkpoint = torch.load(CKPT,map_location='cpu',weights_only=False)
    train = prepare(WiMFDataset('/work/datasets/brats-gli-2023/raw/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData'),411)
    val = prepare(WiMFDataset('/work/datasets/brats-gli-2023/raw/ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData'),46)
    report = dict(initialization=str(CKPT),optimizer='fresh Adam(.9,.95), lr1e-5, clip1',
                  train_subject=train[0],validation_subject=val[0],steps=150,
                  warning='One-subject bounded diagnostic; not evidence of dataset-wide convergence.',variants=[])
    for name,sigma,scale in [('control',0.,1000.),('noise',.1,1000.),('noise_unit_time',.1,1.)]:
        torch.manual_seed(701)
        config = dict(checkpoint['model_config'],time_feature_scale=scale)
        model = create_wimf_model(**config).cuda()
        model.load_state_dict(checkpoint['model'])
        flow = WaveletImprovedMeanFlow(**dict(checkpoint['flow_config'],prior_sigma=sigma))
        optimizer = torch.optim.Adam(model.parameters(),lr=1e-5,betas=(.9,.95),eps=1e-8)
        model.eval()
        entry = dict(name=name,prior_sigma=sigma,time_feature_scale=scale,
                     before={split:scores(model,flow,*case[1:],full=True) for split,case in [('train',train),('validation',val)]},updates=[])
        print(json.dumps(dict(event='begin',**{k:v for k,v in entry.items() if k!='updates'})),flush=True)
        # Restore a common random stream after evaluation. Every training call
        # uses genuine Algorithm1 and the same logical batch/learned JVP tangent.
        torch.manual_seed(1701)
        begin = time.monotonic()
        model.train()
        x,c,source,y = train[2]
        for step in range(1,151):
            optimizer.zero_grad(set_to_none=True)
            losses = flow.training_losses(model,x,c,source,y,efficient=True)
            losses['loss'].backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            if step%25==0:
                model.eval()
                item = dict(step=step,loss=losses['loss'].item(),velocity_mse=losses['mse'].item(),
                            grad_norm=float(norm),train=scores(model,flow,*train[1:]))
                entry['updates'].append(item)
                print(json.dumps(dict(event='update',variant=name,**item)),flush=True)
                model.train()
            del losses
        model.eval()
        entry['seconds'] = time.monotonic()-begin
        entry['after'] = {split:scores(model,flow,*case[1:],full=True) for split,case in [('train',train),('validation',val)]}
        report['variants'].append(entry)
        (ROOT/'short-ablation.json').write_text(json.dumps(report,indent=2,allow_nan=False))
        print(json.dumps(dict(event='finished',variant=name,seconds=entry['seconds'],after=entry['after'])),flush=True)
        # Save weights/config only, explicitly marked as diagnostic rather than
        # a resumable training checkpoint. Never overwrite original run files.
        torch.save(dict(kind='WiMF-diagnostic-weights-only',model_config=config,
                        flow_config=flow.config(),model={k:v.cpu() for k,v in model.state_dict().items()}),ROOT/f'{name}-150.pt')
        del optimizer,model
        gc.collect()
        torch.cuda.empty_cache()
    print('Saved',ROOT/'short-ablation.json',flush=True)


if __name__=='__main__':
    main()
