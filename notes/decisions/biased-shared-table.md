# Biased init dropped for the shared 8×4 table (2026-11-08)

The historical partner rule (T1n↔T1c, T2w↔T2f, partner weight 0.6) cannot
initialize a single shared 8×4 table:

- T1n held out → T1c = 3·T2f
- T2w held out → T2f = 3·T1c

Both cannot hold, so no shared table satisfies the rule for all four targets.

Decision: biased is excluded from the shared-table comparison. Compare
uniform and random against the mean and Gaussian baselines.

Rejected alternatives:

- `--source_shared_bias partner`: adds a fixed target-dependent offset, so the
  source is no longer one shared table.
- `--source_shared_bias fixed` (default, boosts `source_biased_modality`, t1n):
  a different hypothesis, not comparable with historical biased. When T1n is
  the target it starts identical to uniform.
