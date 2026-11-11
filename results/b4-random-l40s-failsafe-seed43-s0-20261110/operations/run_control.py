"""Execute one immutable sweep entry; keep small results and final weights."""

from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

import torch


def publish_complete(plan, out):
    sweep = Path(plan['sweep'])
    (sweep / 'completion.json').write_text(json.dumps({out.name: 'complete'}, indent=2) + '\n')
    (sweep / 'status.txt').write_text('complete\n')


def main():
    plan = json.loads(Path(sys.argv[1]).read_text())
    job = plan["jobs"][int(sys.argv[2])]
    out = Path(job["directory"])
    status = out / "status.txt"
    preflight_only = '--preflight-only' in sys.argv[3:]
    restart = int(os.environ.get('SLURM_RESTART_COUNT', '0'))
    previous_status = status.read_text().strip()
    if not preflight_only and previous_status == 'complete':
        publish_complete(plan, out)
        return
    if previous_status != 'queued' and (preflight_only or restart == 0):
        raise RuntimeError('Only a Slurm requeue may resume an entry that already started')
    try:
        if shutil.disk_usage(out).free < 30 * 1024**3:
            raise RuntimeError("Less than 30 GiB free; refusing to fill shared storage")
        manifest = json.loads((Path(plan["sweep"]) / "code_sha256.json").read_text())
        for name, digest in manifest.items():
            assert hashlib.sha256((Path(plan["code"]) / name).read_bytes()).hexdigest() == digest, name
        operation_manifest = json.loads((Path(plan["sweep"]) / "operations_sha256.json").read_text())
        for name, digest in operation_manifest.items():
            assert hashlib.sha256((Path(plan["sweep"]) / "operations" / name).read_bytes()).hexdigest() == digest, name
        if not preflight_only and not torch.cuda.is_available():
            raise RuntimeError('Training requires a GPU allocation')
        batch_size = int(plan.get('batch_size', 4))
        mixed_targets = plan.get('mixed_targets', True)
        use_checkpoint = plan.get('use_checkpoint', True)
        if batch_size not in (1, 4) or mixed_targets != (batch_size == 4):
            raise ValueError('Expected batch 1 with one target or batch 4 with four mixed targets')
        runtime = dict(started_utc=datetime.now(timezone.utc).isoformat(),
                       python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
                       gpu=torch.cuda.get_device_name(0) if not preflight_only and torch.cuda.is_available() else None,
                       visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                       slurm_job=os.environ.get("SLURM_JOB_ID"), restart_count=restart, job=job,
                       batch_size=batch_size, mixed_targets=mixed_targets,
                       use_checkpoint=use_checkpoint,
                       cuda_allocator=os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'default'))
        expected_allocator = 'max_split_size_mb:128' if batch_size == 4 else 'default'
        if runtime['cuda_allocator'] != expected_allocator:
            raise RuntimeError('CUDA allocator differs from the approved GPU check')
        runtime["packages"] = {name: metadata.version(name) for name in
                               ("numpy", "scipy", "scikit-image", "nibabel", "PyWavelets")}
        if not preflight_only:
            (out / f"runtime-attempt-{restart}.json").write_text(json.dumps(runtime, indent=2) + "\n")
        (out / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
        # Paths from the established Mostafa layout. Verify case counts before training.
        train = Path(plan['train_dir'])
        val = Path(plan['val_dir'])
        splits = {}
        for name, directory, count in (("train", train, 1251), ("valfull", val, 219)):
            cases = sorted(p.name for p in directory.iterdir() if p.is_dir() and p.name.startswith("BraTS-"))
            assert len(cases) == count, (name, len(cases))
            splits[name] = cases
        assert not {x.rsplit("-", 1)[0] for x in splits["train"]} & {x.rsplit("-", 1)[0] for x in splits["valfull"]}
        (out / "split_manifest.json").write_text(json.dumps(splits, indent=2) + "\n")
        if preflight_only:
            subprocess.run([sys.executable, str(Path(plan["sweep"]) / "operations/numerical_check.py"), str(out / "numerical_validation.json")], check=True)
            (out / 'preflight_passed.json').write_text(json.dumps(runtime, indent=2) + '\n')
            print('PASS: entrypoint manifests, packages, storage and full split checks')
            return
        flags = ["--source", job["arm"]] if job["arm"] in ("mean", "gaussian") else [
            "--source", "learned", "--source_init", job["arm"]]
        train_cmd = [sys.executable, "-u", "scripts/train_wfm.py", "--data_dir", str(train),
                     "--contr", "all", "--batch_size", str(batch_size), "--mixed_targets", str(mixed_targets),
                     "--use_checkpoint", str(use_checkpoint), "--num_workers", "4", "--lr", "1e-5", "--source_lr", "1e-3",
                     "--sigma_max", str(job["sigma"]), "--seed", str(job["seed"]),
                     "--max_iterations", "10000", "--save_interval", "10000", "--save_optimizer", "False",
                     "--recovery_interval", str(plan["recovery_interval"]), "--log_interval", "10", "--tensorboard_path", str(out), "--use_tensorboard", "False",
                     "--source_architecture", job["source_architecture"], "--source_shared_bias", "fixed", *flags]
        if job['arm'] == 'biased':
            raise RuntimeError('Biased is pending; it is not authorized in this sweep')
        (out / 'commands.json').write_text(json.dumps(dict(train=train_cmd), indent=2) + '\n')
        status.write_text("training\n")
        with (out / "train.log").open("a") as log:
            subprocess.run(train_cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
        subprocess.run([sys.executable, str(Path(plan['sweep']) / 'operations/check_control.py'), str(out), job['arm'], job['source_architecture']], check=True)
        status.write_text("evaluating-valfull\n")
        eval_cmd = [sys.executable, "-u", "scripts/sample_wfm.py", "--model_path",
                    str(out / "checkpoints/wfm_unified_010000.pt"), "--data_dir", str(val),
                    "--output_dir", str(out / "eval-valfull-euler1-avail3-10000"),
                    "--contr", "all", "--sampling_steps", "1", "--num_samples", "-1", "--seed", "42",
                    "--sigma_max", str(job["sigma"]), "--num_available", "3", "--metrics_only", "True"]
        (out / "commands.json").write_text(json.dumps(dict(train=train_cmd, evaluation=eval_cmd), indent=2) + "\n")
        with (out / "eval-valfull.log").open("a") as log:
            subprocess.run(eval_cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
        rows = json.loads((out / "eval-valfull-euler1-avail3-10000/metrics.json").read_text())
        assert set(rows) == {"t1n", "t1c", "t2w", "t2f"}
        for modality, values in rows.items():
            assert len(values) == 219 and {r["subject"] for r in values} == set(splits["valfull"]), modality
            assert all(math.isfinite(r["masked"][key]) for r in values for key in ("ssim", "psnr"))
        runtime["finished_utc"] = datetime.now(timezone.utc).isoformat()
        runtime['host'] = plan['host']
        (out / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
        status.write_text("complete\n")
        publish_complete(plan, out)
    except BaseException as error:
        status.write_text(f"failed: {type(error).__name__}: {error}\n")
        raise


if __name__ == "__main__":
    main()
