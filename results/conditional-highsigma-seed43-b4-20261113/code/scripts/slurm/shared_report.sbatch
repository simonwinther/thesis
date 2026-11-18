#!/usr/bin/env bash
set -euo pipefail
source scripts/slurm/env.sh
activate_env
exec python scripts/finish_shared_host.py "$1"
