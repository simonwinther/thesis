"""Render the ten completed seed43 BS4 runs directly from verified metrics."""
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SWEEP = ROOT / "results/mlp8-sigma1-seed43-b4-20261112"
TARGETS = ("t1n", "t1c", "t2w", "t2f")
ORDER = ["mlp8-uniform-s0", "mlp8-random-s0", "mlp8-uniform-s05", "mlp8-random-s05",
         "mean-s1", "old-uniform-s1", "old-random-s1", "mlp8-uniform-s1", "mlp8-random-s1", "gaussian-s1"]
NAMES = {"mlp8-uniform": "MLP8 uniform", "mlp8-random": "MLP8 random",
         "old-uniform": "Old uniform (MLP72)", "old-random": "Old random (MLP72)",
         "mean": "Mean", "gaussian": "Gaussian"}
registry = json.loads((SWEEP / "run_registry.json").read_text())
jobs = {j["name"]: j for j in registry["jobs"]}
assert set(jobs) == set(ORDER) and len(jobs) == 10
records, reference_ids = [], None
for name in ORDER:
    job = jobs[name]
    assert job["seed"] == 43 and job["batch_size"] == 4
    assert job["source_architecture"] == "conditional"
    assert job["use_statistics"] == (not name.startswith("mlp8"))
    directory = SWEEP / job["host"] / "runs" / Path(job["directory"]).name
    summary = json.loads((directory / "training_summary.json").read_text())
    assert summary["optimizer_updates"] == 10000
    command = json.loads((directory / "commands.json").read_text())
    for updates in (2500, 10000):
        evaluation = command["evaluation"][str(updates)]
        for flag, expected in (("--sampling_steps", "1"), ("--seed", "42"), ("--num_available", "3")):
            assert evaluation[evaluation.index(flag) + 1] == expected
        path = directory / f"eval-valfull-euler1-avail3-{updates}/metrics.json"
        metrics = json.loads(path.read_text())
        assert set(metrics) == set(TARGETS)
        per_target = {}
        for target, rows in metrics.items():
            ids = {r["subject"] for r in rows}
            assert len(rows) == len(ids) == 219
            reference_ids = ids if reference_ids is None else reference_ids
            assert ids == reference_ids
            assert all(math.isfinite(r["masked"][m]) for r in rows for m in ("ssim", "psnr"))
            per_target[target] = {m: mean(r["masked"][m] for r in rows) for m in ("ssim", "psnr")}
        records.append({"run": name, "sigma": job["sigma"], "seed": 43, "batch_size": 4,
                        "updates": updates, "host": job["host"], "per_target": per_target,
                        **{m: mean(per_target[t][m] for t in TARGETS) for m in ("ssim", "psnr")},
                        "metrics_path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})

assert len(records) == 20
audit = {"verified_utc": datetime.now(timezone.utc).isoformat(), "training_runs": 10,
         "checkpoint_evaluations": 20, "same_219_case_ids": True, "sampling_steps": 1,
         "averaging": "Brain-masked case means, averaged equally over four target modalities.",
         "limitations": "One seed; GPUs differ. Two checkpoints from each training run, not separate runs.",
         "results": records}
(OUT / "verification.json").write_text(json.dumps(audit, indent=2) + "\n")
with (OUT / "scores.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["run", "sigma", "seed", "batch_size", "updates", "ssim", "psnr", "host"])
    writer.writeheader()
    writer.writerows({k: r[k] for k in writer.fieldnames} for r in records)

lookup = {(r["run"], r["updates"]): r for r in records}
labels = [NAMES[name.rsplit("-s", 1)[0]] + "  ·  σ=" + f"{jobs[name]['sigma']:g}" for name in ORDER]
BLUE, GREY = "#17679b", "#545b62"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "svg.fonttype": "none", "pdf.fonttype": 42})
fig, axes = plt.subplots(1, 2, figsize=(14.5, 10.2), sharey=True)
fig.patch.set_facecolor("white")
for ax, metric in zip(axes, ("ssim", "psnr")):
    for y, name in enumerate(ORDER):
        early, final = lookup[name, 2500][metric], lookup[name, 10000][metric]
        ax.plot((early, final), (y, y), color="#b7bdc2", lw=1.8, zorder=2)
        for updates, value in ((2500, early), (10000, final)):
            is_final = updates == 10000
            ax.plot(value, y, "s" if is_final else "D", ms=7.5,
                    mfc=BLUE if is_final else "white", mec=BLUE if is_final else GREY, mew=1.5, zorder=3)
            ax.annotate(f"{value:.4f}" if metric == "ssim" else f"{value:.2f}",
                        (value, y), xytext=(0, -8 if is_final else 7), textcoords="offset points",
                        ha="center", va="top" if is_final else "bottom", fontsize=9.5,
                        color=BLUE if is_final else GREY,
                        bbox=dict(facecolor="white", edgecolor="none", pad=0.3), zorder=4)
    ax.set_yticks(range(10), labels)
    ax.set_ylim(9.6, -0.6)
    ax.set_xlim((0.05, 0.72) if metric == "ssim" else (8.7, 19.1))
    ax.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7] if metric == "ssim" else [10, 12, 14, 16, 18])
    ax.set_title("SSIM ↑" if metric == "ssim" else "PSNR (dB) ↑", fontsize=15, pad=15)
    ax.grid(axis="x", color="#e9ebed", lw=0.7)
    for separator in (1.5, 3.5):
        ax.axhline(separator, color="#e3e6e8", lw=0.8)
    ax.tick_params(axis="y", length=0, pad=14)
    ax.tick_params(axis="x", length=3, color="#a0a5a9")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#bbc0c4")

handles = [Line2D([], [], marker="D", linestyle="none", mfc="white", mec=GREY, mew=1.5, ms=7, label="2,500 updates"),
           Line2D([], [], marker="s", linestyle="none", color=BLUE, ms=7, label="10,000 updates")]
fig.suptitle("Seed 43 · BS4", x=0.5, y=0.98, fontsize=20, weight="bold")
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=2, frameon=False, fontsize=12)
fig.subplots_adjust(left=0.26, right=0.985, top=0.85, bottom=0.13, wspace=0.16)
fig.text(0.26, 0.066, "MLP72: original inputs  ·  MLP8: statistics removed; same MLP", fontsize=10, color=GREY)
fig.text(0.26, 0.034, "219 validation cases · 4 targets · brain-masked means · 1 sampling step", fontsize=10, color=GREY)
for extension in ("png", "pdf", "svg"):
    fig.savefig(OUT / f"seed43-checkpoint-comparison.{extension}", dpi=240, facecolor="white")
plt.close(fig)
print("Verified20 evaluations from10 runs; identical219-case sets;40 plotted metric values.")
print(OUT / "seed43-checkpoint-comparison.png")
