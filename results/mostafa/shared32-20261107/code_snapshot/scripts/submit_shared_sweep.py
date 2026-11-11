"""Submit the predefined 50-run shared-source comparison through Slurm."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("--check", required=True, type=Path)
    args = parser.parse_args()
    root = Path.cwd()
    assert (root / "guided_diffusion/source.py").is_file()
    assert args.name.replace("-", "").replace("_", "").isalnum()
    assert (args.check / "status.txt").read_text().strip() == "complete", "Smoke check incomplete"
    sweep = root / "runs" / args.name
    sweep.mkdir()  # Refuse to overwrite or resubmit an existing sweep.
    code = sweep / "code"
    shutil.copytree(root, code, ignore=shutil.ignore_patterns(
        "runs", "results", "datasets", "assets", ".git", "__pycache__", "*.pyc",
        ".venv", "venv", "envs", ".cache"))
    manifest = {str(p.relative_to(code)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(code.rglob("*")) if p.is_file()}
    (sweep / "code_sha256.json").write_text(json.dumps(manifest, indent=2) + "\n")
    jobs = []
    for sigma in (0.0, 0.5):
        for seed in (42, 43, 44, 45, 2026):
            for arm in ("mean", "uniform", "random", "biased", "gaussian"):
                name = f"{arm}-s{str(sigma).replace('.', '')}-seed{seed}"
                jobs.append(dict(index=len(jobs), arm=arm, seed=seed, sigma=sigma,
                                 directory=str(sweep / "runs" / name)))
    plan = dict(created_utc=datetime.now(timezone.utc).isoformat(), name=args.name,
                model_root=str(root), sweep=str(sweep), code=str(code), jobs=jobs,
                seeds=[42, 43, 44, 45, 2026], sigmas=[0.0, 0.5],
                iterations=10000, batch_size=1, unet_lr=1e-5, source_lr=1e-3,
                evaluation_seed=42, sampling_steps=1, validation_cases=219,
                num_available=3, source_architecture="shared", shared_bias="partner",
                biased_note="Fixed target-dependent partner factor 3 remains active throughout training; "
                            "this arm extends the pure shared-table formula.",
                historical_note="Seed 42 was on Mostafa; other historical seeds were on B200. "
                                "Historical noisy biased/Gaussian only have seeds 43 and 44.",
                storage="Final model/source only; no optimizer, intermediate checkpoints or generated volumes.")
    for job in jobs:
        directory = Path(job["directory"])
        directory.mkdir(parents=True)
        (directory / "status.txt").write_text("queued\n")
        (directory / "config.json").write_text(json.dumps(job, indent=2) + "\n")
    plan_path = sweep / "run_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    (sweep / "status.txt").write_text("submitted\n")
    array = subprocess.check_output([
        "sbatch", "--parsable", f"--job-name={args.name}", "--array=0-49%2",
        f"--output={sweep}/array-%A_%a.log", f"--chdir={code}",
        str(code / "scripts/slurm/shared_arm.sbatch"), str(plan_path),
    ], text=True).strip()
    report = subprocess.check_output([
        "sbatch", "--parsable", "--partition=gpu", "--cpus-per-task=1", "--mem=2G",
        "--time=00:10:00", f"--job-name={args.name}-report", f"--dependency=afterany:{array}",
        f"--output={sweep}/report.log", f"--chdir={code}",
        str(code / "scripts/slurm/shared_report.sbatch"), str(plan_path),
    ], text=True).strip()
    plan.update(array_job_id=array, report_job_id=report)
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(dict(sweep=str(sweep), array_job_id=array, report_job_id=report, runs=len(jobs))))


if __name__ == "__main__":
    main()
