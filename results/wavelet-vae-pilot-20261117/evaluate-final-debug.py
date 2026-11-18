"""Evaluate the final bounded debug pair after both successful Slurm jobs."""
import json
from pathlib import Path
import subprocess
import sys

root, code = map(Path, sys.argv[1:3])
paths = {}
for band in ("lf", "hf"):
    folder = root / f"overfit-{band}"
    if (folder / "status.txt").read_text().strip() != "complete":
        raise RuntimeError(f"Debug branch incomplete: {band}")
    paths[band] = folder / json.loads((folder / "latest.json").read_text())["checkpoint"]
subprocess.run([sys.executable, str(code / "evaluate.py"), "--manifest", str(root / "manifest.json"),
                "--lf", str(paths["lf"]), "--hf", str(paths["hf"]), "--subset", "debug",
                "--output", str(root / "debug-report-final")], check=True)
