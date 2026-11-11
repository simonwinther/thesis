"""Resume only the approved file list over four independent streaming transfers."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess

archive = Path(__file__).resolve().parent
source = json.loads((archive / 'mostafa_dataset_inventory.json').read_text())['TrainingData']
target = json.loads((archive / 'hendrix_dataset_inventory.json').read_text())['TrainingData']
selection = json.loads((archive / 'hendrix_missing_files.json').read_text())['TrainingData']['files']
expected = {name: source['files'][name] for name in selection}
program = '''from pathlib import Path
import json
root=Path(ROOT)
expected=EXPECTED
print(json.dumps([name for name,size in expected.items() if not (root/name).exists() or (root/name).stat().st_size!=size]))
'''.replace('ROOT', repr(target['paths'][0])).replace('EXPECTED', repr(expected))
r = subprocess.run(['ssh', '-o', 'BatchMode=yes', 'hendrix', 'python3 -'],
                   input=program, text=True, capture_output=True, check=True)
remaining = json.loads(r.stdout)
bins, sizes = [[] for _ in range(4)], [0] * 4
for name in sorted(remaining, key=lambda n: expected[n], reverse=True):
    index = sizes.index(min(sizes))
    bins[index].append(name)
    sizes[index] += expected[name]
print(json.dumps(dict(remaining_files=len(remaining), remaining_bytes=sum(sizes), stream_bytes=sizes)), flush=True)


def copy_files(index):
    names = bins[index]
    if not names:
        return index
    a = subprocess.Popen(['ssh', '-o', 'BatchMode=yes', 'mostafa', 'tar', '-C', source['paths'][0], '-cf', '-', '-T', '-'],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    b = subprocess.Popen(['ssh', '-o', 'BatchMode=yes', 'hendrix', 'tar', '-C', target['paths'][0], '-xf', '-'], stdin=a.stdout)
    a.stdout.close()
    a.stdin.write(('\n'.join(names) + '\n').encode())
    a.stdin.close()
    rc1, rc2 = a.wait(), b.wait()
    if rc1 or rc2:
        raise RuntimeError(f'Stream {index} failed: source={rc1}, target={rc2}')
    return index


with ThreadPoolExecutor(max_workers=4) as pool:
    for future in as_completed([pool.submit(copy_files, i) for i in range(4)]):
        print(f'Stream {future.result()} complete', flush=True)
(archive / 'dataset_transfer_status.txt').write_text('transfer complete; SHA256 verification required\n')
print('All approved files transferred; waiting for full SHA256 audit.', flush=True)
