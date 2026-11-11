"""Audit completed runs against their configuration, case manifests and B200 files."""
from pathlib import Path
import argparse
import ast
import csv
import datetime
import hashlib
import json
import math
import re
import shlex
import subprocess

archive = Path(__file__).resolve().parent
plan = json.loads((archive / "run_plan.json").read_text())
parser = argparse.ArgumentParser()
parser.add_argument("--require-complete", action="store_true")
args = parser.parse_args()
modalities = ("t1n", "t1c", "t2w", "t2f")
environment = json.loads((archive / "runs" / plan["sweep_name"] / "runtime_environment.json").read_text())
patients = {50: set(environment["splits"]["val50"]["case_ids"]),
            219: set(environment["splits"]["valfull"]["case_ids"])}
jobs = []
for sources, seeds, primary in ((plan["overnight_sources"], plan["overnight_seeds"], True),
                                (plan["additional_sources"], plan["additional_seeds"], False)):
    for source in sources:
        for seed in seeds:
            jobs.append({"name": f"is-{source}-s05-10k-seed{seed}-{plan['run_id']}",
                         "source": source, "seed": seed, "noise": 0.5, "steps": 1,
                         "training": True, "splits": [("val50", 50), ("valfull", 219)] if primary else [("valfull", 219)]})
for noise, sources, seeds in ((0, ("mean", "uniform", "gaussian"), (43, 44)),
                              (0.5, plan["noisy_diagnostics"]["sources"], plan["noisy_diagnostics"]["seeds"])):
    tag = "s0" if noise == 0 else "s05"
    for source in sources:
        for seed in seeds:
            jobs.append({"name": f"eval-{source}-{tag}-seed{seed}-euler10-val50-{plan['run_id']}",
                         "source": source, "seed": seed, "noise": noise, "steps": 10,
                         "training": False, "splits": [("val50", 50)]})
for seed in plan["gaussian_noisy_diagnostics"]["seeds"]:
    for steps in plan["gaussian_noisy_diagnostics"]["sampling_steps"]:
        jobs.append({"name": f"eval-gaussian-s05-seed{seed}-euler{steps}-val50-{plan['run_id']}",
                     "source": "gaussian", "seed": seed, "noise": 0.5, "steps": steps,
                     "training": False, "splits": [("val50", 50)]})

files, completed, pending = [], [], []
for job in jobs:
    run = archive / "runs" / job["name"]
    status = (run / "status.txt").read_text().strip() if (run / "status.txt").exists() else "not started"
    if status != "complete":
        pending.append({"run": job["name"], "status": status})
        continue
    source = job["source"]
    expected_mode = "learned" if source in ("uniform", "random", "biased") else source
    expected_init = source if expected_mode == "learned" else "uniform"
    if job["training"]:
        train_log = (run / "train.log").read_text()
        for text in ("Training mode: Unified (all modalities)", "Class conditioning: True",
                     "Sigma max: 0.5", "Available conditions: 3", "Batch size: 1",
                     "Learning rate: 1e-05", "Max iterations: 10000"):
            assert text in train_log, (run, text)
        assert f"Source: {expected_mode} (init={expected_init}, init_std=0.1, lr=0.001)" in train_log, run
    else:
        config = dict(line.split("=", 1) for line in (run / "config.txt").read_text().splitlines())
        for key, expected in {"source": source, "training_seed": str(job["seed"]),
                              "sigma_max": str(job["noise"]), "sampling_steps": str(job["steps"]),
                              "eval_seed": "42", "validation_cases": "50"}.items():
            assert config[key] == expected, (run, key)
        tag = "s0" if job["noise"] == 0 else "s05"
        identifier = {43: "20261102-repeat", 44: "20261102-third"}[job["seed"]] if job["noise"] == 0 else plan["run_id"]
        assert config["checkpoint"] == f"runs/is-{source}-{tag}-10k-seed{job['seed']}-{identifier}/checkpoints/wfm_unified_010000.pt", run
    for split, count in job["splits"]:
        directory = run / f"eval-{split}-euler1-avail3-10000" if job["training"] else run
        sampling_log = (run / (f"eval-{split}.log" if job["training"] else "sampling.log")).read_text()
        match = re.search(r"Source config: (\{[^\n]+\})", sampling_log)
        assert match, run
        source_config = ast.literal_eval(match.group(1))
        assert source_config == {"mode": expected_mode, "init": expected_init,
                                 "init_std": 0.1, "hidden_channels": 128, "seed": job["seed"]}, (run, source_config)
        assert "Available conditions: 3" in sampling_log, run
        timing = (directory / "timing.txt").read_text()
        assert f"Sampling Steps: {job['steps']}\n" in timing, directory
        assert f"Total Samples: {count * 4}\n" in timing, directory
        for modality in modalities:
            for prefix in ("metrics-", "metrics-full-"):
                file = directory / f"{prefix}{modality}.txt"
                data = file.read_bytes()
                text = data.decode()
                assert re.search(r"^Contrast:\s*" + modality + r"\s*$", text, re.M), file
                assert re.search(r"^Number of samples:\s*" + str(count) + r"\s*$", text, re.M), file
                ids = re.findall(r"^(BraTS-[^:\n]+): PSNR=", text, re.M)
                assert len(ids) == len(set(ids)) == count and set(ids) == patients[count], file
                files.append({"path": str(file.relative_to(archive / "runs")),
                              "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "cases": count})
    completed.append(job)

weight_summaries, pending_weights = [], []
for item in plan.get("weight_summary_files", []):
    file = archive / item["local"]
    if not file.exists():
        pending_weights.append(item["local"])
        continue
    rows = list(csv.DictReader(file.open()))
    assert len(rows) == 480 and all(int(row["patients"]) == 219 for row in rows), file
    totals = {}
    for row in rows:
        mean, sd, low, high = (float(row[key]) for key in ("mean_weight", "patient_sample_sd", "min_weight", "max_weight"))
        assert all(math.isfinite(v) for v in (mean, sd, low, high)), file
        assert 0 <= low <= mean <= high <= 1 and sd >= 0, file
        key = (row["step"], row["target"], row["subband"])
        totals.setdefault(key, []).append(mean)
    assert len(totals) == 160 and all(len(v) == 3 and abs(sum(v) - 1) < 1e-5 for v in totals.values()), file
    weight_summaries.append({**item, "rows": len(rows), "bytes": file.stat().st_size,
                             "sha256": hashlib.sha256(file.read_bytes()).hexdigest()})

expected_code = {**plan["code_sha256"], **plan["followup_code_sha256"], **plan.get("operational_code_sha256", {})}
request = {"files": [file["path"] for file in files] + [item["remote"] for item in weight_summaries], "runs": [job["name"] for job in completed],
           "training_runs": [job["name"] for job in completed if job["training"]],
           "code": list(expected_code),
           "sweeps": [plan["sweep_name"], plan["primary_completion_sweep_name"],
                      plan["completion_sweep_name"], plan["idle_diagnostics_sweep"],
                      *plan["completion_dependency_sweeps"]]}
remote_code = '''from pathlib import Path
import hashlib,json
root=Path(REMOTE_ROOT)
request=REQUEST
result={"hashes":{p:hashlib.sha256((root/"runs"/p).read_bytes()).hexdigest() for p in request["files"]},
"code":{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in request["code"]},
"statuses":{p:(root/"runs"/p/"status.txt").read_text().strip() for p in request["runs"]+request["sweeps"]},
"checkpoints":{}}
for name in request["training_runs"]:
 result["checkpoints"][name]={}
 for prefix in ("wfm", "source"):
  file=root/"runs"/name/"checkpoints"/(prefix+"_unified_010000.pt")
  result["checkpoints"][name][prefix]=file.stat().st_size
print(json.dumps(result))
'''.replace("REMOTE_ROOT", repr(plan["remote_model"])).replace("REQUEST", repr(request))
response = subprocess.run(["ssh", "-o", "ConnectTimeout=15", "-p", "2742", "ucloud@ssh.cloud.sdu.dk",
                           "/work/WMF/venv/bin/python -c " + shlex.quote(remote_code)],
                          capture_output=True, text=True, check=True, timeout=90)
remote = json.loads(response.stdout)
for file in files:
    assert file["sha256"] == remote["hashes"][file["path"]], file["path"]
for item in weight_summaries:
    assert item["sha256"] == remote["hashes"][item["remote"]], item["local"]
assert remote["code"] == expected_code, "Remote code differs from recorded code snapshot"
for name in expected_code:
    assert hashlib.sha256((archive / "code_snapshot" / name).read_bytes()).hexdigest() == expected_code[name], name
for name, digest in plan.get("historical_operational_code", {}).items():
    assert hashlib.sha256((archive / "code_history" / name).read_bytes()).hexdigest() == digest, name
for job in completed:
    assert remote["statuses"][job["name"]] == "complete", job["name"]
for sizes in remote["checkpoints"].values():
    assert sizes["wfm"] > 0 and sizes["source"] > 0
backup = (archive / "backup_status.txt").read_text().strip()
all_complete = not pending and not pending_weights and all(remote["statuses"][s] == "complete" for s in request["sweeps"]) and backup == "final results copied; pilot=complete"
report = {"verified_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
          "all_complete": all_complete, "expected_tasks": len(jobs), "completed_tasks": len(completed),
          "pending": pending, "backup_status": backup, "remote_sweep_statuses": {s: remote["statuses"][s] for s in request["sweeps"]},
          "weight_summaries_verified": len(weight_summaries), "pending_weight_summaries": pending_weights,
          "weight_summaries": weight_summaries,
          "code_files_verified": len(expected_code), "historical_code_files_verified": len(plan.get("historical_operational_code", {})),
          "metric_files_verified": len(files),
          "total_metric_bytes": sum(file["bytes"] for file in files),
          "checks": ["training settings and actual loaded source configuration", "sampling step counts and sample counts",
                     "masked and unmasked metrics match actual dataset case IDs", "all local metric SHA256 match B200",
                     "remote code and local source snapshot match recorded hashes", "final model/source checkpoints exist remotely; none downloaded"],
          "remote_checkpoint_sizes": remote["checkpoints"], "metric_files": files}
(archive / "archive_verification.json").write_text(json.dumps(report, indent=2) + "\n")
print(f"Audited {len(completed)}/{len(jobs)} tasks, {len(files)} metric files, {len(expected_code)} source files; all_complete={all_complete}.")
if args.require_complete and not all_complete:
    raise SystemExit("Full overnight scope or final backup is incomplete.")
