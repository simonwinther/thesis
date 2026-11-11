#!/usr/bin/env bash
set -euo pipefail
source scripts/slurm/env.sh
activate_env
exec python scripts/summarize_shared_sweep.py "$1"
