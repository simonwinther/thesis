"""Small, deterministic manifests for reviewed and smoke-tested source code."""
import hashlib
from pathlib import Path

EXCLUDED = {'runs', 'results', 'datasets', 'assets', '.git', '__pycache__', '.venv', 'venv', 'envs', '.cache'}
RUNTIME_SCRIPTS = {'train_wfm.py', 'sample_wfm.py', 'stream_metrics.py', 'evaluate.py',
                   'check_mixed_batch.py', 'check_shared_pair.py', 'experiment_manifest.py'}


def code_manifest(root):
    root = Path(root)
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob('*'))
            if p.is_file() and p.suffix in ('.py', '.sh', '.sbatch')
            and not set(p.relative_to(root).parts) & EXCLUDED}


def training_manifest(root):
    return {name: digest for name, digest in code_manifest(root).items()
            if name.startswith(('DWT_IDWT/', 'guided_diffusion/'))
            or name in ('scripts/slurm/check_mixed.sbatch', 'scripts/slurm/check_shared_pair.sbatch', 'scripts/slurm/env.sh')
            or (name.startswith('scripts/') and Path(name).name in RUNTIME_SCRIPTS)}
