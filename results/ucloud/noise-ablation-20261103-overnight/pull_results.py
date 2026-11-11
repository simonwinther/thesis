"""Copy this pilot's lightweight results every five minutes, for up to 36 hours."""
from pathlib import Path
import datetime
import json
import os
import re
import shlex
import subprocess
import sys
import time

archive = Path(__file__).resolve().parent
plan = json.loads((archive / "run_plan.json").read_text())
remote_root = plan["remote_model"] + "/runs"
host = "ucloud@ssh.cloud.sdu.dk"
ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=12", "-o", "StrictHostKeyChecking=yes", "-p", "2742"]
sweep = plan["sweep_name"]
completion_sweep = plan.get("completion_sweep_name", sweep)
completion_sweeps = [completion_sweep, *plan.get("completion_dependency_sweeps", [])]
names = plan["run_names"]
destination = archive / "runs"
destination.mkdir(exist_ok=True)
command = ["rsync", "-azm", "--stats", "--timeout=60", "--max-size=5m"]
command += [f"--include=/{name}/" for name in names]
command += ["--exclude=/*/", "--include=*/", "--include=*.txt", "--include=*.json", "--include=*.csv", "--include=*.log", "--exclude=*", "-e", shlex.join(ssh), f"{host}:{remote_root}/", str(destination) + "/"]
environment = dict(os.environ, LC_ALL="C")
deadline = time.monotonic() + 36 * 3600
status_file = archive / "backup_status.txt"
status_file.write_text("running\n")
while time.monotonic() < deadline:
    try:
        result = subprocess.run(ssh + [host, "cat", *[f"{remote_root}/{name}/status.txt" for name in completion_sweeps]], capture_output=True, text=True, check=True, timeout=30)
        statuses = result.stdout.splitlines()
        if len(statuses) != len(completion_sweeps):
            raise RuntimeError("Missing completion status")
        status = statuses[0]
        if status == "complete" and any(value != "complete" for value in statuses[1:]):
            status = "finished-with-errors" if any("failed" in value or value == "finished-with-errors" for value in statuses[1:]) else "waiting-diagnostics"
        preview = subprocess.run(command[:1] + ["--dry-run"] + command[1:], capture_output=True, text=True, check=True, timeout=90, env=environment)
        match = re.search(r"Total transferred file size: ([\d,]+) bytes", preview.stdout)
        if match is None:
            raise RuntimeError("Could not read the transfer size")
        size = int(match.group(1).replace(",", ""))
        if size > 50 * 1024 * 1024:
            raise RuntimeError(f"Refusing a large result transfer: {size} bytes")
        subprocess.run(command, capture_output=True, text=True, check=True, timeout=120, env=environment)
        terminal = status in {"complete", "finished-with-errors"}
        if terminal:
            status_file.write_text(f"final results copied; pilot={status}\n")
        review = archive / "build_review.py"
        if review.exists():
            subprocess.run([sys.executable, str(review)], check=True, timeout=30)
        print(datetime.datetime.now(datetime.timezone.utc).isoformat(), f"copied {size} bytes; pilot={status}", flush=True)
        if terminal:
            break
    except (subprocess.SubprocessError, RuntimeError) as error:
        print(datetime.datetime.now(datetime.timezone.utc).isoformat(), f"backup retry: {error}", flush=True)
    time.sleep(300)
else:
    status_file.write_text("stopped after 36 hours; check pilot status and any missing results\n")
