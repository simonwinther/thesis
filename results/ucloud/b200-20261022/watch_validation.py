"""Evaluate completed 10k checkpoints and record the best validation SSIM."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("model", choices=("wfm", "wmf"))
args = parser.parse_args()
root = Path("/work/WMF/b200-20261022")
training = root / "runs" / args.model
output = root / "validation" / args.model
output.mkdir(parents=True, exist_ok=True)
os.environ["CUDA_VISIBLE_DEVICES"] = "0" if args.model == "wfm" else "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
print(f"Watching {training}; 20 fixed validation subjects, all modalities, 1 sampling step", flush=True)
while True:
    waiting = False
    for checkpoint in sorted((training / "checkpoints").glob(f"{args.model}_unified_*.pt")):
        step = int(checkpoint.stem.rsplit("_", 1)[1])
        if step % 10000:
            continue
        folder = output / f"{step:06d}"
        result = folder / "metrics.json"
        failure = folder / "failed.json"
        if result.exists() or failure.exists():
            continue
        optimizer = checkpoint.with_name(f"opt_{args.model}_{step:06d}.pt")
        if not optimizer.exists() or time.time() - optimizer.stat().st_mtime < 15:
            waiting = True
            continue
        folder.mkdir(parents=True, exist_ok=True)
        print(f"Evaluating {checkpoint}", flush=True)
        command = [sys.executable, "-B", str(root / "evaluate_checkpoint.py"),
                   "--model", args.model, "--checkpoint", str(checkpoint),
                   "--output", str(result), "--subjects", "20", "--sampling-steps", "1"]
        with (folder / "console.log").open("a") as log:
            status = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
        if status:
            failure.write_text(json.dumps({"exit_code": status, "checkpoint": str(checkpoint)}) + "\n")
            print(f"Validation failed: {folder / 'console.log'}", flush=True)
            continue
        results = [json.loads(path.read_text()) for path in output.glob("*/metrics.json")]
        best = max(results, key=lambda item: item["overall"]["ssim"])
        (output / "best_checkpoint.json").write_text(json.dumps({
            "checkpoint": best["checkpoint"], "checkpoint_step": best["checkpoint_step"],
            "overall": best["overall"], "subjects": best["subjects"],
            "sampling_steps": best["sampling_steps"], "selection_metric": "mean brain-masked SSIM",
        }, indent=2) + "\n")
        print(f"Finished step {step}; best step {best['checkpoint_step']}", flush=True)
    if (training / "exit_code").exists() and not waiting:
        print("Training exited; checkpoint validation complete", flush=True)
        break
    time.sleep(30)
