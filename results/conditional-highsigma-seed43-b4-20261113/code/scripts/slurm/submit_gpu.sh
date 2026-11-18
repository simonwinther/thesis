#!/usr/bin/env bash
# Run from the model directory. Accepts ordinary sbatch options, script, and args.
# GPU inventory checked 2026-11-11; review after 30 days before new submissions.
# See notes/SERVERS.md in the thesis repository. Keep copies in the three models
# identical so each model can also be uploaded/frozen independently.
set -euo pipefail

if [[ $# == 0 ]]; then
    echo "Usage: bash scripts/slurm/submit_gpu.sh [sbatch options] script [script args]" >&2
    exit 2
fi

gpu_options=()
case "$(hostname -s)" in
    mostafa*)
        # All four registered GPUs are suitable: RTX 6000 Ada / RTX A6000.
        ;;
    hendrix*)
        # Leave A100, L40S, A40 and H100 nodes eligible, even when currently busy.
        # Slurm 20.11 rejects bracket ranges followed by the "fl" suffix.
        excluded_nodes=(hendrixgpu{05,06,09,10,11,12,13,17,18,19,20}fl)
        excluded_csv=$(IFS=,; printf '%s' "${excluded_nodes[*]}")
        gpu_options+=("--exclude=$excluded_csv")
        ;;
    *)
        echo "Unknown GPU cluster: $(hostname -s); check its GPU inventory first." >&2
        exit 1
        ;;
esac

# Several existing job scripts use this relative Slurm output directory.
mkdir -p runs/slurm
exec sbatch "${gpu_options[@]}" "$@"
