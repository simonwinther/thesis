"""Keep transient Slurm teardown states from causing an unnecessary relaunch."""
import getpass
import subprocess
import time


def wait_for_queue_room(additional=9, limit=10, timeout=180):
    deadline = time.monotonic() + timeout
    while True:
        queued = subprocess.check_output(
            ['squeue', '-h', '-r', '-u', getpass.getuser(), '-o', '%A'], text=True).splitlines()
        if len(queued) + additional <= limit:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(f'Submitting would exceed {limit} queued/running jobs; no jobs submitted')
        time.sleep(5)
