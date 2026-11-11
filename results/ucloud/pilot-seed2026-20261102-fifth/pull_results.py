"""Copy this pilot's lightweight results every five minutes, for up to 36 hours."""
from pathlib import Path
import datetime
import json
import os
import re
import shlex
import subprocess
import time

archive = Path(__file__).resolve().parent
plan = json.loads((archive / "run_plan.json").read_text())
remote_root = plan["remote_model"] + "/runs"
host = "ucloud@ssh.cloud.sdu.dk"
ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=12", "-o", "StrictHostKeyChecking=yes", "-p", "2742"]
seed, run_id = plan["seed"], plan["run_id"]
sweep = f"pilot-seed{seed}-{run_id}"
names = [sweep] + [f"is-{arm}-s0-10k-seed{seed}-{run_id}" for arm in ("mean", "uniform", "random", "biased", "gaussian")]
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
        result = subprocess.run(ssh + [host, "cat", f"{remote_root}/{sweep}/status.txt"], capture_output=True, text=True, check=True, timeout=30)
        status = result.stdout.strip()
        preview = subprocess.run(command[:1] + ["--dry-run"] + command[1:], capture_output=True, text=True, check=True, timeout=90, env=environment)
        match = re.search(r"Total transferred file size: ([\d,]+) bytes", preview.stdout)
        if match is None:
            raise RuntimeError("Could not read the transfer size")
        size = int(match.group(1).replace(",", ""))
        if size > 50 * 1024 * 1024:
            raise RuntimeError(f"Refusing a large result transfer: {size} bytes")
        subprocess.run(command, capture_output=True, text=True, check=True, timeout=120, env=environment)
        print(datetime.datetime.now(datetime.timezone.utc).isoformat(), f"copied {size} bytes; pilot={status}", flush=True)
        if status in {"complete", "finished-with-errors"}:
            status_file.write_text(f"final results copied; pilot={status}\n")
            break
    except (subprocess.SubprocessError, RuntimeError) as error:
        print(datetime.datetime.now(datetime.timezone.utc).isoformat(), f"backup retry: {error}", flush=True)
    time.sleep(300)
else:
    status_file.write_text("stopped after 36 hours; check pilot status and any missing results\n")
