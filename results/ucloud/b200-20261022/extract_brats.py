from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json,time,zipfile
raw=Path('/home/ucloud/datasets/raw')
dest=Path('/home/ucloud/datasets/brats2023')
dest.mkdir(parents=True,exist_ok=True)

def extract(name):
    archive=raw/('ASNR-MICCAI-BraTS2023-GLI-Challenge-'+name+'Data.zip')
    start=time.monotonic()
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            if not (dest/info.filename).resolve().is_relative_to(dest.resolve()):
                raise ValueError('Archive path escapes extraction directory')
        print('Extracting',name,len(z.infolist()),'entries',flush=True)
        z.extractall(dest)
    subjects=list((dest/archive.stem).glob('BraTS-GLI-*'))
    result={'archive':str(archive),'directory':str(dest/archive.stem),'subjects':len(subjects),'seconds':time.monotonic()-start}
    print(json.dumps(result),flush=True)
    return result

with ThreadPoolExecutor(max_workers=2) as pool:
    results=list(pool.map(extract,['Training','Validation']))
Path('/work/WMF/b200-20261022/dataset_extraction.json').write_text(json.dumps(results,indent=2))
