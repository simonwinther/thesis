"""Probe saved WiMF weights, without changing weights, optimizer, or Git files.

Use full 224x224x160 real MRI inputs. Four rows request the four missing
modalities of one subject, with each row conditioned only on its other three.
Compare interior interpolation predictions with synthesis at the prior endpoint,
and split the Algorithm 1 JVP into state and explicit-time contributions.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, '/home/ucloud/test/WiMF')
from guided_diffusion.wimf import WaveletImprovedMeanFlow, apply_idwt
from guided_diffusion.wimf_model import create_wimf_model
from scripts.wimf_data import WiMFDataset, ALL_MODALITIES
from guided_diffusion.wimf_metrics import reconstruction_metrics

ROOT = Path(__file__).resolve().parent
RUN = Path('/work/WMF/WiMF/runs/brats-gli-2023-fast-20261012-175231')
CHECKPOINTS = [
    Path('/work/WMF/WiMF/runs/brats-gli-2023-tqdm-20261012-145638/checkpoints/wimf_00000854.pt'),
    RUN/'checkpoints/wimf_00005000.pt',
    RUN/'checkpoints/wimf_00012626.pt',
]
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.manual_seed(42)


def stats(prediction, target):
    """Return per-row coefficient MSE, RMS, alignment and norm ratios."""
    a, b = prediction.flatten(1).double(), target.flatten(1).double()
    return dict(
        mse=(a-b).square().mean(1).tolist(),
        rms=a.square().mean(1).sqrt().tolist(),
        target_rms=b.square().mean(1).sqrt().tolist(),
        relative_norm=(a.norm(dim=1)/b.norm(dim=1).clamp_min(1e-12)).tolist(),
        cosine=torch.nn.functional.cosine_similarity(a,b,dim=1).tolist(),
    )


@torch.no_grad()
def evaluate(model, flow, sample, label, step, save_images=False, derivative=True):
    """Inspect fixed-time learned velocities and unclamped full-volume outputs."""
    prepared = [flow.prepare_batch(sample, modality) for modality in ALL_MODALITIES]
    x, c, e, y = [torch.cat([row[k] for row in prepared]) for k in range(4)]
    velocity = e-x
    report = dict(case=label, step=step, modalities=list(ALL_MODALITIES), sweep=[])

    # Test true interpolated states versus the inference state, at the same times.
    for time in (.25, .5, .75, .9, .95, .99, 1.):
        t = torch.full((4,),time,device='cuda')
        r = torch.zeros_like(t)
        z = (1-time)*x+time*e
        boundary = model(torch.cat((z,c),1),t,t,y=y)
        average = model(torch.cat((z,c),1),t,r,y=y)
        endpoint = model(torch.cat((e,c),1),t,r,y=y)
        report['sweep'].append(dict(t=time,boundary=stats(boundary,velocity),
                                   average=stats(average,velocity),
                                   endpoint=stats(endpoint,velocity)))

    # State and explicit time JVPs add to the full derivative; conditions stay fixed.
    if derivative:
        report['derivative'] = []
        for time, lower in ((.5,.2),(.9,.2),(.99,0.)):
            t = torch.full((4,),time,device='cuda')
            r = torch.full_like(t,lower)
            z = (1-time)*x+time*e
            def fn(state, lo, up):
                return model(torch.cat((state,c),1),up,lo,y=y)
            v = fn(z,t,t)
            u, spatial = torch.func.jvp(fn,(z,r,t),(v,torch.zeros_like(r),torch.zeros_like(t)))
            _, temporal = torch.func.jvp(fn,(z,r,t),(torch.zeros_like(z),torch.zeros_like(r),torch.ones_like(t)))
            correction = (time-lower)*(spatial+temporal)
            report['derivative'].append(dict(t=time,r=lower,
                spatial=stats(spatial,velocity),temporal=stats(temporal,velocity),
                correction=stats(correction,velocity),primal=stats(u,velocity),
                compound=stats(u+correction,velocity),
                squared_error_sum=(u+correction-velocity).square().flatten(1).sum(1).tolist()))
            del spatial, temporal, correction, u, v

    # Apply exactly the monitoring clamp and first-condition foreground mask.
    synth = flow.sample(model,c,e,y)
    images = apply_idwt(synth)
    report['reconstruction'] = {}
    for index,modality in enumerate(ALL_MODALITIES):
        first = next(name for name in ALL_MODALITIES if name != modality)
        image = images[index:index+1].clamp(0,1).masked_fill(sample[first]==0,0)
        target = sample[modality].cpu().numpy().squeeze()
        image = image.cpu().numpy().squeeze()
        report['reconstruction'][modality] = reconstruction_metrics(image,target)
        if save_images:
            np.save(ROOT/f'{label}-{step}-{modality}-prediction.npy',image)
            np.save(ROOT/f'{label}-{modality}-target.npy',target)
    return report


def main():
    dataset = WiMFDataset('/work/datasets/brats-gli-2023/raw/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData')
    case = dataset[411]
    sample = {name:case[name].unsqueeze(0).cuda() for name in ALL_MODALITIES}
    output = dict(subject=case['subject'],checkpoints=[])
    model = None
    for path in CHECKPOINTS:
        checkpoint = torch.load(path,map_location='cpu',weights_only=False)
        if model is None:
            model = create_wimf_model(**checkpoint['model_config']).cuda().eval()
        model.load_state_dict(checkpoint['model'])
        flow = WaveletImprovedMeanFlow(**checkpoint['flow_config'])
        result = evaluate(model,flow,sample,'train411',checkpoint['step'],save_images=True)
        result['checkpoint'] = str(path)
        output['checkpoints'].append(result)
        (ROOT/'checkpoint-probes.json').write_text(json.dumps(output,indent=2,allow_nan=False))
        print(json.dumps(dict(step=checkpoint['step'],reconstruction=result['reconstruction'],
                             derivatives=result['derivative'])),flush=True)
        del checkpoint
    print('Saved',ROOT/'checkpoint-probes.json',flush=True)


if __name__ == '__main__':
    main()
