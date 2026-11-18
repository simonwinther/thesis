"""Submit the approved fresh LF/HF pair and its CPU curve monitor once."""
import datetime
import json
import shlex
import subprocess
from pathlib import Path

local = Path(__file__).resolve().parent
registry_path = local / "run_registry.json"
assert not registry_path.exists(), "Inspect existing jobs; never submit duplicates"
plan = json.loads((local / "plan.json").read_text())
assert plan["start"] == "fresh" and plan["max_steps"] == 50000 and plan["beta"] == 0
queue = subprocess.check_output(["ssh", "mostafa", 'squeue -h -u "$USER" -o "%i %T %j %b"'], text=True)
(local / "queue-before.txt").write_text(queue)
assert not queue.strip(), "Inspect existing allocations before using two GPU slots"
root = plan["remote_root"]
work = str(Path(root).parent.parent)
registry = {"plan": plan, "jobs": []}


def submit(arguments, kind, band=None):
    command = "cd " + shlex.quote(work) + " && " + shlex.join(arguments)
    result = subprocess.run(["ssh", "mostafa", command], text=True, capture_output=True, check=True)
    job = result.stdout.strip().split(";")[0]
    assert job.isdigit(), result.stdout
    registry["jobs"].append({"host": "mostafa", "job_id": job, "kind": kind, "band": band,
                             "command": command, "submitted_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()})
    registry_path.write_text(json.dumps(registry, indent=2) + "\n")
    print(job, kind, band, flush=True)


for band in ("lf", "hf"):
    submit(["bash", "scripts/slurm/submit_gpu.sh", "--parsable", "--job-name=ae-full-" + band,
            "--time=48:00:00", "--cpus-per-task=8", "--mem=32G", root + "/code-v1/run.sbatch",
            root + "/code-v1/train.py", "--manifest", root + "/manifest.json", "--band", band,
            "--latent-channels", "8", "--encoder-channels", "128", "256", "256", "--stage", "full",
            "--output", root + "/" + band, "--max-steps", "50000", "--max-train-seconds", "150000",
            "--save-every", "1000", "--seed", "43", "--workers", "4", "--tensorboard", root + "/tensorboard"],
           "training", band)
submit(["sbatch", "--parsable", "--job-name=ae-full-monitor", root + "/monitor.sbatch",
        root + "/code-v1/monitor.py", "--manifest", root + "/manifest.json", "--lf", root + "/lf",
        "--hf", root + "/hf", "--output", root + "/monitor", "--tensorboard", root + "/tensorboard",
        "--max-steps", "50000"], "cpu_monitor")
