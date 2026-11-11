"""Back up only this sweep's small results, with a size check before each copy."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import os
import re
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("remote_sweep")
    parser.add_argument("destination", type=Path)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--host", default="mostafa", choices=("mostafa", "hendrix"))
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    ssh_transport = "ssh" if args.host == "hendrix" else "ssh -o BatchMode=yes -o ConnectTimeout=15"
    command = ["rsync", "-azm", "--stats", "--timeout=60", "--max-size=5m",
               "--exclude=code/", "--exclude=checkpoints/", "--exclude=__pycache__/",
               "--include=*/", "--include=*.txt", "--include=*.json", "--include=*.csv",
               "--include=*.log", "--include=*.md", "--exclude=*",
               "-e", ssh_transport,
               f"{args.host}:{args.remote_sweep}/", str(args.destination) + "/"]
    deadline = time.monotonic() + 14 * 86400
    while time.monotonic() < deadline:
        try:
            preview = subprocess.run(command[:1] + ["--dry-run"] + command[1:], check=True,
                                     capture_output=True, text=True, timeout=120, env={**os.environ, "LC_ALL": "C"})
            match = re.search(r"Total transferred file size: ([\d,]+) bytes", preview.stdout)
            if not match:
                raise RuntimeError("Transfer size unavailable")
            size = int(match[1].replace(",", ""))
            if size > 250 * 1024**2:
                raise RuntimeError(f"Refusing result transfer of {size} bytes")
            subprocess.run(command, check=True, capture_output=True, timeout=180)
            status_path = args.destination / "status.txt"
            status = status_path.read_text().strip() if status_path.exists() else "unknown"
            print(datetime.now(timezone.utc).isoformat(), f"copied {size} bytes; sweep={status}", flush=True)
            if status in ("complete", "finished-with-errors", "cancelled") or args.once:
                (args.destination / "backup_status.txt").write_text(f"copied; sweep={status}\n")
                return
        except (subprocess.SubprocessError, RuntimeError) as error:
            print(datetime.now(timezone.utc).isoformat(), f"backup retry: {error}", flush=True)
            if args.once:
                raise
        time.sleep(600)
    (args.destination / "backup_status.txt").write_text("stopped after 14 days; verify remaining results\n")


if __name__ == "__main__":
    main()
