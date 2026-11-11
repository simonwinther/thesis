# Project guidance

## Notes

- `notes/` is the shared agent log. Read `notes/AGENTS.md` and `notes/HANDOFF.md` first.
- Then read `notes/REPOSITORY.md` for lasting repository context and WFM / WFM-InformedSource run caveats.
- Never write to the user's personal Obsidian vault (`~/Documents/Uni Notes`).

## Models

- cWDM (`models/cwdm`): Conditional Wavelet Diffusion Models.
- WFM (`models/WaveletFlowMatching`): Wavelet Flow Matching.
- WFM-InformedSource (`models/WFM-InformedSource`): WFM with a learned/informed starting source.
- WMF (`models/WaveletMeanFlows`): Wavelet Mean Flows, our main project adapting WFM to MeanFlows.

WFM and WMF are different projects. Check the abbreviation and folder carefully.

## Working style

The user prefers to make code changes themselves. Default to short, simple explanations of what to change, where, and why.

Only edit files or run state-changing commands when explicitly asked. Questions, suggestions, and "why not?" are not permission. Keep authorized changes small and easy to understand; avoid large rewrites.

With read-only permissions, never edit, write, delete, or request escalation, even if asked to make changes. Tell the user to change permissions first, and wait until they do.

## Code and result syncing

- Make code changes locally in `/home/simon/dev/thesis`, then rsync them to the servers before running experiments. The local copy is the source of truth; do not edit code only on a server.
- Copy lightweight experiment results back locally after experiments and whenever practical: metrics, summaries, plots, configurations, and small logs. Check transfer size first. Full run directories, checkpoints, datasets, and generated volumes require an explicit request.
- Treat all storage on the B200 allocation as temporary. Copy results and any explicitly selected checkpoint backups to the local machine before the allocation shuts down.
- Keep code uploads separate from result downloads. Exclude datasets, runs, results, environments, caches, and Git metadata from code uploads, and use rsync without deletion flags.
