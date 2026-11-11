"""Stream only the approved missing dataset files; never stage data on the PC."""
import json
from pathlib import Path
import subprocess

archive = Path(__file__).resolve().parent
inventories = {host: json.loads((archive / f'{host}_dataset_inventory.json').read_text())
               for host in ('mostafa', 'hendrix')}
missing = json.loads((archive / 'hendrix_missing_files.json').read_text())
for split, selection in missing.items():
    if not selection['files']:
        continue
    source_root = inventories['mostafa'][split]['paths'][0]
    target_root = inventories['hendrix'][split]['paths'][0]
    assert all('\n' not in name and '..' not in Path(name).parts for name in selection['files'])
    print(f"Streaming {len(selection['files'])} files, {selection['bytes']} bytes: {source_root} -> {target_root}", flush=True)
    source = subprocess.Popen(['ssh', '-o', 'BatchMode=yes', 'mostafa',
                               'tar', '-C', source_root, '-cf', '-', '-T', '-'],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    target = subprocess.Popen(['ssh', '-o', 'BatchMode=yes', 'hendrix',
                               'tar', '-C', target_root, '--skip-old-files', '-xf', '-'], stdin=source.stdout)
    source.stdout.close()
    source.stdin.write(('\n'.join(selection['files']) + '\n').encode())
    source.stdin.close()
    source_code, target_code = source.wait(), target.wait()
    if source_code or target_code:
        raise RuntimeError(f'Transfer failed: source={source_code}, target={target_code}')
    print(f'Completed {split}', flush=True)
(archive / 'dataset_transfer_status.txt').write_text('transfer complete; SHA256 verification required\n')
