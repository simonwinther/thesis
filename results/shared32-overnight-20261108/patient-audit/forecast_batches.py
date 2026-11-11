"""Replay CPU sampling only, verifying its prefix against actual training audits."""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import torch


def main():
    root = Path(sys.argv[1])
    data = json.loads(Path(sys.argv[2]).read_text())
    sys.path[:0] = [str(root), str(root / 'scripts')]
    import train_wfm

    args = train_wfm.create_argparser().parse_args(data['commands'][3:])
    torch.manual_seed(args.seed)
    model_keys = ['image_size', 'num_channels', 'num_res_blocks', 'channel_mult',
                  'learn_sigma', 'class_cond', 'use_checkpoint', 'attention_resolutions',
                  'num_heads', 'num_head_channels', 'num_heads_upsample',
                  'use_scale_shift_norm', 'dropout', 'resblock_updown', 'use_fp16',
                  'use_new_attention_order', 'dims', 'num_groups', 'in_channels',
                  'out_channels', 'bottleneck_attention', 'resample_2d', 'additive_skips', 'use_freq']
    model = train_wfm.create_model(**train_wfm.args_to_dict(args, model_keys))
    before = torch.get_rng_state().clone()
    source = train_wfm.StartingSource(mode=args.source, init=args.source_init,
                                    init_std=args.source_init_std,
                                    hidden_channels=args.source_hidden_channels,
                                    seed=args.seed, architecture=args.source_architecture,
                                    biased_modality=args.source_biased_modality,
                                    shared_bias=args.source_shared_bias)
    assert torch.equal(before, torch.get_rng_state()), 'Source consumes global CPU RNG'
    train_wfm.WaveletFlowMatching(num_timesteps=args.diffusion_steps,
                                 sigma_max=args.sigma_max, mode='i2i')
    corrected = '--patient-aware' in sys.argv[3:]
    if corrected:
        from patient_batching import DistinctPatientBatchSampler, patient_id
        loader = torch.utils.data.DataLoader(
            data['case_order'], num_workers=0,
            batch_sampler=DistinctPatientBatchSampler(
                torch.utils.data.RandomSampler(data['case_order']),
                [patient_id(case) for case in data['case_order']], 4))
    else:
        loader = torch.utils.data.DataLoader(data['case_order'], batch_size=4,
                                             shuffle=True, drop_last=True, num_workers=0)
    iterator = iter(loader)
    forecast, collisions = [], []
    for step in range(1, 10001):
        try:
            cases = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            cases = next(iterator)
        cases = list(cases)
        forecast.append(cases)
        if len({case.rsplit('-', 1)[0] for case in cases}) != 4:
            collisions.append({'step': step, 'cases': cases})
    observed = data['observed_batches']
    mismatches = [i + 1 for i, cases in enumerate(observed) if cases != forecast[i]]
    result = {'checked_utc': datetime.now(timezone.utc).isoformat(),
              'torch': torch.__version__, 'cuda_visible': torch.cuda.is_available(),
              'seed': args.seed, 'batch_size': 4, 'patient_aware': corrected,
              'observed_batches': len(observed),
              'prefix_matches': not mismatches, 'first_mismatch_steps': mismatches[:5],
              'patient_collisions': collisions if not mismatches else None,
              'interpretation': 'Forecast is usable only if actual prefix matches exactly. '
                                'No training, forward/backward passes, MRI reads or GPU use.'}
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
