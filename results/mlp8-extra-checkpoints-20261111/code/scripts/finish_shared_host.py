"""Validate completion of one host's assigned runs, without pooling partial seeds."""
import json
from pathlib import Path
import sys

plan = json.loads(Path(sys.argv[1]).read_text())
states = {Path(job['directory']).name: (Path(job['directory']) / 'status.txt').read_text().strip()
          for job in plan['jobs']}
complete = all(state == 'complete' for state in states.values())
root = Path(plan['sweep'])
(root / 'completion.json').write_text(json.dumps(states, indent=2) + '\n')
(root / 'status.txt').write_text('complete\n' if complete else 'finished-with-errors\n')
if not complete:
    raise RuntimeError('One or more assigned runs did not complete; see completion.json')
