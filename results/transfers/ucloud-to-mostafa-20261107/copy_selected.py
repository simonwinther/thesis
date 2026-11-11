"""Copy the explicit final-checkpoint manifest directly between servers."""
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REMOTE_META = "/home/si-wi/dev/.ucloud-transfer-20261107"
DESTINATION = "/home/si-wi/dev/thesis/models/WFM-InformedSource/runs/"
runtime = json.loads((HERE / "transfer_runtime.json").read_text())
agent = json.loads(Path(runtime["agent_state_file"]).read_text())
source_shell = shlex.join([
    "ssh", "-p", "2980", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
    "-o", "UserKnownHostsFile=" + REMOTE_META + "/ucloud_known_hosts",
    "-o", "GlobalKnownHostsFile=/dev/null", "-o", "HostKeyAlgorithms=ssh-ed25519",
    "-o", "IdentitiesOnly=yes", "-i", REMOTE_META + "/ucloud_identity.pub",
    "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=3",
])
outer_ssh = [
    "ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
    "-o", "ForwardAgent=" + agent["agent_socket"], "-o", "ServerAliveInterval=30",
    "-o", "ServerAliveCountMax=3", "mostafa",
]

def execute(argv, log_name):
    print("Starting", log_name, flush=True)
    with (HERE / log_name).open("w") as log:
        result = subprocess.run(outer_ssh + [shlex.join(argv)], stdout=log, stderr=subprocess.STDOUT)
    print(log_name, "exit", result.returncode, flush=True)
    if result.returncode:
        print((HERE/log_name).read_text()[-4000:])
        raise SystemExit(result.returncode)

mode = sys.argv[1]
if mode == "check":
    execute(shlex.split(source_shell) + ["ucloud@ssh.cloud.sdu.dk", "test -d /work/WMF/thesis/models/WFM-InformedSource/runs && printf 'source authenticated successfully\\n'"], "authentication.log")
    execute(["rsync", "-an", "--stats", "--files-from=" + REMOTE_META + "/ucloud_files.txt", "-e", source_shell, "ucloud@ssh.cloud.sdu.dk:/work/WMF/thesis/models/WFM-InformedSource/runs/", DESTINATION], "ucloud_dry_run.log")
elif mode == "copy":
    execute(["rsync", "-a", "--partial", "--partial-dir=.rsync-partial", "--info=progress2", "--stats", "--files-from=" + REMOTE_META + "/ucloud_files.txt", "-e", source_shell, "ucloud@ssh.cloud.sdu.dk:/work/WMF/thesis/models/WFM-InformedSource/runs/", DESTINATION], "ucloud_copy.log")
    runtime.update(status="ucloud-copy-complete", copied_at_utc=datetime.now(timezone.utc).isoformat())
    (HERE / "transfer_runtime.json").write_text(json.dumps(runtime,indent=2)+"\n")
    execute(["rsync", "-a", "--info=progress2", "--stats", "--files-from=" + REMOTE_META + "/mostafa_seed42_files.txt", "/home/si-wi/thesis/models/WFM-InformedSource/runs/", DESTINATION], "seed42_copy.log")
    runtime.update(status="both-copies-complete", copied_at_utc=datetime.now(timezone.utc).isoformat())
    (HERE / "transfer_runtime.json").write_text(json.dumps(runtime,indent=2)+"\n")
else:
    raise SystemExit("Use check or copy")
