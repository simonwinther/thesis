"""Save small runtime and split manifests without loading images or checkpoints."""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
args = parser.parse_args()
packages = {}
for name in ("torch", "torchvision", "numpy", "scipy", "scikit-image", "SimpleITK", "nibabel", "PyWavelets", "tensorboard"):
    try:
        packages[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        packages[name] = None

splits = {}
for split, variable, expected in (("train", "TRAIN_DIR", 1251), ("val50", "VAL50_DIR", 50), ("valfull", "VALFULL_DIR", 219)):
    directory = Path(os.environ[variable])
    cases = sorted(p.name for p in directory.iterdir() if p.is_dir() and p.name.startswith("BraTS-"))
    if len(cases) != expected:
        raise ValueError(f"Unexpected {split} size: {len(cases)} instead of {expected}")
    splits[split] = {"path": str(directory), "cases": len(cases), "case_ids": cases,
                     "case_ids_sha256": hashlib.sha256("\n".join(cases).encode()).hexdigest()}
train, validation, subset = (set(splits[name]["case_ids"]) for name in ("train", "valfull", "val50"))
if train & validation or not subset.issubset(validation):
    raise ValueError("Training/validation case overlap or inconsistent val50 subset")

gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,uuid,driver_version,memory.total", "--format=csv"], text=True)
pids = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).splitlines()
processes = []
for pid in pids:
    proc = Path("/proc") / pid.strip()
    try:
        argv = [value.decode() for value in (proc / "cmdline").read_bytes().split(b"\0") if value]
        if not any("scripts/train_wfm.py" in value or "scripts/sample_wfm.py" in value for value in argv):
            continue
        environment = {}
        for item in (proc / "environ").read_bytes().split(b"\0"):
            name, separator, value = item.partition(b"=")
            if separator and name.decode() in {"CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "SIGMA_MAX"}:
                environment[name.decode()] = value.decode()
        processes.append({"pid": int(pid), "argv": argv, "selected_environment": environment})
    except (FileNotFoundError, PermissionError):
        continue

snapshot = {"captured_at_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
            "python_executable": sys.executable, "platform": platform.platform(), "cpu_count": os.cpu_count(),
            "packages": packages, "gpus_csv": gpu, "splits": splits, "train_val_case_overlap": 0,
            "live_training_or_sampling_commands": processes,
            "scope": "Metadata only: no image volumes, checkpoints or unrestricted environment variables were read."}
path = Path(args.output)
path.write_text(json.dumps(snapshot, indent=2) + "\n")
print(f"Saved {path} ({path.stat().st_size} bytes); splits verified and train/validation case IDs are disjoint.")
