"""Execute one immutable sweep entry; keep small results and final weights."""

from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import torch


def main():
    plan = json.loads(Path(sys.argv[1]).read_text())
    job = plan["jobs"][int(sys.argv[2])]
    out = Path(job["directory"])
    status = out / "status.txt"
    try:
        if status.read_text().strip() != "queued":
            raise RuntimeError("Refusing to rerun an entry that has already started")
        if shutil.disk_usage(out).free < 30 * 1024**3:
            raise RuntimeError("Less than 30 GiB free; refusing to fill shared storage")
        manifest = json.loads((Path(plan["sweep"]) / "code_sha256.json").read_text())
        for name, digest in manifest.items():
            assert hashlib.sha256((Path(plan["code"]) / name).read_bytes()).hexdigest() == digest, name
        runtime = dict(started_utc=datetime.now(timezone.utc).isoformat(),
                       python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
                       gpu=torch.cuda.get_device_name(0),
                       visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                       slurm_job=os.environ.get("SLURM_JOB_ID"), job=job)
        runtime["packages"] = {name: metadata.version(name) for name in
                               ("numpy", "scipy", "scikit-image", "nibabel", "PyWavelets")}
        (out / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
        # Paths from the established Mostafa layout. Verify case counts before training.
        data = Path.home() / "datasets/raw"
        train = data / "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
        val = data / "ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData"
        splits = {}
        for name, directory, count in (("train", train, 1251), ("valfull", val, 219)):
            cases = sorted(p.name for p in directory.iterdir() if p.is_dir() and p.name.startswith("BraTS-"))
            assert len(cases) == count, (name, len(cases))
            splits[name] = cases
        assert not set(splits["train"]) & set(splits["valfull"])
        (out / "split_manifest.json").write_text(json.dumps(splits, indent=2) + "\n")
        flags = ["--source", job["arm"]] if job["arm"] in ("mean", "gaussian") else [
            "--source", "learned", "--source_init", job["arm"]]
        train_cmd = [sys.executable, "-u", "scripts/train_wfm.py", "--data_dir", str(train),
                     "--contr", "all", "--batch_size", "1", "--lr", "1e-5", "--source_lr", "1e-3",
                     "--sigma_max", str(job["sigma"]), "--seed", str(job["seed"]),
                     "--max_iterations", "10000", "--save_interval", "10000", "--save_optimizer", "False",
                     "--log_interval", "10", "--tensorboard_path", str(out), "--use_tensorboard", "False",
                     "--source_architecture", "shared", "--source_shared_bias", "partner", *flags]
        status.write_text("training\n")
        with (out / "train.log").open("w") as log:
            subprocess.run(train_cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
        status.write_text("evaluating-valfull\n")
        eval_cmd = [sys.executable, "-u", "scripts/sample_wfm.py", "--model_path",
                    str(out / "checkpoints/wfm_unified_010000.pt"), "--data_dir", str(val),
                    "--output_dir", str(out / "eval-valfull-euler1-avail3-10000"),
                    "--contr", "all", "--sampling_steps", "1", "--num_samples", "-1", "--seed", "42",
                    "--sigma_max", str(job["sigma"]), "--num_available", "3", "--metrics_only", "True"]
        (out / "commands.json").write_text(json.dumps(dict(train=train_cmd, evaluation=eval_cmd), indent=2) + "\n")
        with (out / "eval-valfull.log").open("w") as log:
            subprocess.run(eval_cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
        rows = json.loads((out / "eval-valfull-euler1-avail3-10000/metrics.json").read_text())
        assert set(rows) == {"t1n", "t1c", "t2w", "t2f"}
        for modality, values in rows.items():
            assert len(values) == 219 and {r["subject"] for r in values} == set(splits["valfull"]), modality
        runtime["finished_utc"] = datetime.now(timezone.utc).isoformat()
        (out / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
        status.write_text("complete\n")
    except BaseException as error:
        status.write_text(f"failed: {type(error).__name__}: {error}\n")
        raise


if __name__ == "__main__":
    main()
