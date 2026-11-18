#!/usr/bin/env bash
set -euo pipefail
source scripts/slurm/env.sh
activate_env
exec python scripts/submit_shared_remaining.py "$1"
