# Servers and environment

## GPU selection for submissions

- Last inventory check: **2026-11-11**. Recheck before the next submission once
  30 days have passed, or earlier on an invalid-node/GPU error or reported
  hardware change. No scheduled polling or check on every chat turn.
- On both hosts, inspect `sinfo -N -o '%N|%P|%t|%G|%f'` and node details;
  verify GPU model/VRAM for new or mixed-type nodes. Update this date and all
  three model copies of `scripts/slurm/submit_gpu.sh` together after review.
- Mostafa: all four GPUs eligible (3 RTX 6000 Ada + 1 RTX A6000, 48 GB each).
- Hendrix: A100 (40/80 GB), L40S (48 GB), A40 (48 GB), H100. H100 is listed
  by Slurm but its exact VRAM and this training environment are not yet tested.
  Features are unset; exclude nodes 05,06,09–13,17–20 to omit the current
  Titan/Quadro nodes. The helper passes full comma-separated names: Hendrix's
  Slurm 20.11 rejects `hendrixgpu[05-06,09-13,17-20]fl` (suffix after brackets).
  Recheck this mapping, not just GPU availability.
- Keep busy suitable nodes eligible: Slurm may allocate them when resources
  become available, subject to priority, reservations, CPU/RAM and time limits.
  Each job requests one GPU. No GPU preference is implied by listing order.
- From a model directory, run GPU `.sbatch` files with `bash` to self-submit.
  For job names, arrays, dependencies or other sbatch options, use
  `bash scripts/slurm/submit_gpu.sh [sbatch options] script.sbatch [script args]`.
  Calling plain `sbatch` directly bypasses host selection. Existing allocations
  run the script body; non-Slurm execution must explicitly set `MRI_DIRECT_RUN=1`
  when no job ID exists. UCloud evaluation callers already do this.
- GPU job scripts and GPU submissions in the pilot/shared launchers use this
  policy. CPU-only launch/report callbacks do not request a GPU. The diagnostic
  `check_shared_memory.sbatch` deliberately retains its A100-only request.
- Frozen experiment scripts under `results/` and `runs/` remain unchanged.
  GPU selection changes do not authorize rerunning cancelled sweeps or altering
  batch size, seed, source, noise, checkpointing or other experiment settings.
- Live test 2026-11-11: Mostafa1366 ran on RTX 6000 Ada; Hendrix3625 was pending
  and cancelled. Repeat3627 ran on hendrixgpu03fl with one A40, automatically
  selected by Slurm; identical test script and corrected wrapper hashes verified.
  All test jobs confirmed CANCELLED; no training ran. [Evidence](../results/gpu-selection-smoke-20261111-184134/).

## mostafa

- `ssh mostafa`, user `si-wi`. Professor's Slurm server, 4 GPUs
  (3× RTX 6000 Ada, 1× A6000), partition `gpu`. Reachable only via Twingate.
- Conda: `source /opt/anaconda3/etc/profile.d/conda.sh && conda activate wmf`.
- Use bash batch scripts; `sbatch --wrap` uses sh and breaks conda.
- sacct is disabled; check jobs with squeue.
- Canonical code: `/home/si-wi/dev/thesis` (synced from local).
  Older copies `~/thesis` and `~/thesis-old-oct19`: keep, do not consolidate.
- Data: `~/datasets/raw/ASNR-MICCAI-BraTS2023-GLI-Challenge-{TrainingData,ValidationData}`
  (1,251 / 219 cases).
- `~/datasets/subsets/val50/{data,split.txt}`: 50 fixed validation patients,
  real dirs of symlinked files (the loader's os.walk doesn't follow dir symlinks).
- Final 10k checkpoints of all reported pilot runs (36 UCloud + 5 seed-42) are
  under `.../models/WFM-InformedSource/runs`. Audit:
  `results/transfers/ucloud-to-mostafa-20261107/`.

## hendrix

- `ssh hendrix`, user `zlp616`. Slurm, A100 GPUs (40 GB variant needs
  `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128` and checkpointing for B=4).
- Keep ≤10 queued/running jobs including callbacks.
- Training data `~/datasets/raw`, validation `~/datasets/raW` (capital W).
  SHA256-verified equal to mostafa.

## UCloud (B200)

- `ssh -p <port> ucloud@ssh.cloud.sdu.dk`; port changes per job.
- All storage, including `/work/WMF`, is lost when the allocation ends.
- `source /work/WMF/ucloud-env.sh` first; venv `/work/WMF/venv`. No Slurm:
  run commands directly with `CUDA_VISIBLE_DEVICES`.

## Twingate (local NixOS laptop)

- `sudo systemctl start twingate`, sign in via URL from `twingate status`.
- It breaks DNS. Fix after every start:
  `sudo resolvectl default-route sdwan0 false` and `sudo resolvectl dns sdwan0 ''`.
- Permanent fix (`twingate-dns-guard`) in `~/nixos-config/modules/nixos/vpn.nix`,
  uncommitted and not activated; needs the user's approval.
