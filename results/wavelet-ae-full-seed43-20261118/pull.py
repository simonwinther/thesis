"""Size-check and copy lightweight monitoring evidence, never checkpoints."""
import re
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parent
destination = root / "download" / "mostafa"
destination.mkdir(parents=True, exist_ok=True)
command = ["rsync", "-az", "--prune-empty-dirs", "--exclude=code-v1/", "--include=*/",
           "--include=*.json", "--include=*.csv", "--include=*.txt", "--include=*.png",
           "--include=*.log", "--include=events.out.tfevents.*", "--exclude=*",
           "mostafa:dev/thesis/models/WaveletAutoencoder/runs/ae-full-seed43-20261118/", str(destination) + "/"]
size = subprocess.check_output(command[:1] + ["--dry-run", "--stats"] + command[1:], text=True)
(root / "transfer-size.txt").write_text(size)
match = re.search(r"Total transferred file size: ([\d,]+) bytes", size)
if not match or int(match[1].replace(",", "")) > 20_000_000:
    raise SystemExit("Inspect transfer size before copying: " + size)
print(match[0], flush=True)
subprocess.run(command, check=True)
