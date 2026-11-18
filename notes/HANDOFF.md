# Handoff

Current state, 18 November 2026. History: EXPERIMENTS.md and archive/.

## Current AE work
- ACTIVE full-data run: results/wavelet-ae-full-seed43-20261118/run_registry.json.
  Mostafa1551 LF and1552 HF RUNNING on TWO separate RTX6000Ada GPUs.
  CPU monitor1553 evaluates matched checkpoint pairs; no other GPU jobs started.
  Fresh seed43, ALL1251 train cases/5004 scans, BS1,50000 updates (~10 epochs).
  Same encoder128/256/256,decoder32/64/64,latent8x28x28x20 per branch,
  deterministic z=mu,beta0,Adam1e-4,FP32/TF32off. Save/evaluate every1000.
- Fixed monitoring:8 train scans +8 disjoint validation scans;2 per modality,
  different patient identities within each subset. Manifest records exact IDs.
  TensorBoard overlays train/val coefficient MSE and paired MRI MSE/SSIM/PSNR.
  Online training MSE logs every20 updates; score curves use fixed subsets.
- Remote: ~/dev/thesis/models/WaveletAutoencoder/runs/ae-full-seed43-20261118.
  TensorBoard binds Mostafa127.0.0.1:6007; local SSH tunnel127.0.0.1:16007.
  Server/tunnel PID files recorded under remote/local experiment roots.
  GPU jobs run independently of this computer; tunnel is needed only for viewing.
- CPU split/loss/TensorBoard tests passed; frozen code hashes verified remotely.
  Checked18Nov15:02CET:LF14720/HF15460,finite losses,~30%complete.
  Latest paired14k validationSSIM0.74827/PSNR21.6424; CPU monitor healthy.
- Previous four-scan overfit COMPLETE at50k: Hendrix2658/2659/report2660.
  SSIM0.9527663/PSNR34.74645, report completed18Nov10:01CET.
  Evidence: results/wavelet-vae-encoder128-resume50k-20261117/.
  Encoder64 remains at4k. No latent export or WFM integration started.

## GPU policy and completed experiments
- Mostafa: MAXIMUM TWO concurrent GPU jobs (user update13Nov2026).
  Existing dependencies formed two afterany chains; preserve this policy for
  any future work. Evidence: results/mostafa-two-gpu-limit-20261113/plan.json.
- All current seed43 comparison experiments are complete: 90 result cells,
  six sources, sigma0/0.5/1/1.5/2, BS1/10k, BS4/2.5k and BS4/10k.
- Last comparison run1405 complete; no comparison cells remain missing.
- Registries retain historical submission states; use actual completion
  files and the final verified snapshot rather than stale status fields.
  Sweeps: seed43-fill-gaps-20261113, conditional-highsigma-seed43-b4-20261113,
  mlp8-sigma1-seed43-b4-20261112, all under results/.
- MLP72 Random,sigma0,BS4/10k uses the previously selected L40S replica and
  evaluation6317. Original1782 also finished; never pool replicas or choose
  by score. Cancelled40-run sweep stays cancelled; heartbeat stays paused.

## Completed comparison figures
- All90 seed43 result cells are complete and verified; old sweeps remain stopped.
  Final figures: results/seed43-final-comparison-20261116/.
  Full provenance: results/seed43-ssim-comparison-20261116/verified-results.json.
- Earlier89-cell independent audit: results/seed43-overview-20261115-evening/.
  Eight legacy cells use rounded text; final digit can differ. See REPOSITORY.md.
- Never launch from old plans. Work locally then rsync. Size-check lightweight
  downloads; no checkpoint/data transfers without an explicit request.
