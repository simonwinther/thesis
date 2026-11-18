"""Draw the historical comparison and an extension with sigma1 checkpoints."""
import csv
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
PRIOR = ROOT / "results/b4-controls-seed43-s05-20261109/seed43-b1-b4-verification.json"
ADDED = ROOT / "results/b4-controls-seed43-s0-20261110/seed43-b1-b4-comparison-updated.json"
TARGETS = ("t1n", "t1c", "t2w", "t2f")
SOURCES = ("mean", "old_uniform", "old_random", "new_uniform", "new_random")
LABELS = ("Mean", "Old learned uniform", "Old learned random", "New learned uniform", "New learned random")
specs = json.loads(PRIOR.read_text())["runs"] + json.loads(ADDED.read_text())["added_runs"]
verified, reference_ids = [], None

for spec in specs:
    assert spec["seed"] == 43
    directory = (Path(spec["metric_path"]).parent if "metric_path" in spec else
                 Path(spec["run_path"]) / "eval-valfull-euler1-avail3-10000")
    hashes, per_target = {}, {}
    for filename, expected in spec.get("metric_file_sha256", {}).items():
        actual = hashlib.sha256((directory / filename).read_bytes()).hexdigest()
        assert actual == expected, (directory, filename)
        hashes[filename] = actual
    metric_json = directory / "metrics.json"
    data = json.loads(metric_json.read_text()) if metric_json.exists() else None
    if data is not None:
        assert set(data) == set(TARGETS)
        hashes["metrics.json"] = hashlib.sha256(metric_json.read_bytes()).hexdigest()
        if "sha256" in spec:
            assert hashes["metrics.json"] == spec["sha256"]
    for target in TARGETS:
        if data is not None:
            rows = [(r["subject"], r["masked"]["psnr"], r["masked"]["ssim"]) for r in data[target]]
        else:
            rows = [(m[0], float(m[1]), float(m[2])) for m in re.findall(
                r"^(BraTS-\S+): PSNR=([-\d.eE+]+), SSIM=([-\d.eE+]+)",
                (directory / f"metrics-{target}.txt").read_text(), re.M)]
        ids = {r[0] for r in rows}
        assert len(rows) == len(ids) == 219, (directory, target)
        assert all(math.isfinite(v) for r in rows for v in r[1:])
        reference_ids = ids if reference_ids is None else reference_ids
        assert ids == reference_ids, (directory, target)
        per_target[target] = {"psnr": mean(r[1] for r in rows), "ssim": mean(r[2] for r in rows)}
    macro = {m: mean(per_target[t][m] for t in TARGETS) for m in ("ssim", "psnr")}
    assert all(abs(macro[m] - spec["macro"][m]) < 1e-10 for m in macro), directory
    verified.append({"batch": spec["batch"], "source": spec["source"], "sigma": spec["sigma"],
                     "seed": 43, "updates": 10000, **macro, "per_target": per_target,
                     "metric_directory": str(directory), "sha256": hashes,
                     "precision": "full_precision_json" if data is not None else "rounded_per_case_text"})

lookup = {(r["source"], r["sigma"], r["batch"]): r for r in verified}
expected = {(s, n, b) for s in SOURCES for n in (0, 0.5) for b in (1, 4)}
assert len(lookup) == len(verified) == 19
assert expected - set(lookup) == {("old_random", 0, 4)}
audit = {"verified_utc": datetime.now(timezone.utc).isoformat(), "runs_verified": 19,
         "validation_cases": 219, "identical_case_sets": True, "sampling_steps": 1,
         "averaging": "Brain-masked per-case scores; equal mean across four target modalities.",
         "limitations": ["One seed; no across-seed uncertainty estimates.",
                         "BS1 sees10000 examples; BS4 sees40000. Execution environments differ.",
                         "Historical B1 text exports contain rounded per-case scores."],
         "missing": {"source": "old_random", "batch": 4, "sigma": 0,
                     "reason": "No completed full-validation metrics. Hendrix jobs1782/1807 pending on2026-11-12."},
         "runs": verified}
(OUT / "verification.json").write_text(json.dumps(audit, indent=2) + "\n")
with (OUT / "scores.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["source", "batch", "sigma", "seed", "updates", "ssim", "psnr", "status"])
    writer.writeheader()
    for source in SOURCES:
        for sigma in (0, 0.5):
            for batch in (1, 4):
                r = lookup.get((source, sigma, batch))
                writer.writerow(dict(source=source, batch=batch, sigma=sigma, seed=43, updates=10000,
                                     ssim=r["ssim"] if r else "", psnr=r["psnr"] if r else "",
                                     status="complete" if r else "pending"))

sigma1 = {}
for source in ("mean", "gaussian"):
    run = ROOT / "results/mlp8-sigma1-seed43-b4-20261112/hendrix/runs" / f"{source}-s1-seed43"
    command = json.loads((run / "commands.json").read_text())["training"]
    for flag, value in (("--source", source), ("--sigma_max", "1.0"), ("--batch_size", "4"),
                        ("--seed", "43"), ("--max_iterations", "10000"), ("--extra_save_steps", "2500")):
        assert command[command.index(flag) + 1] == value, (source, flag)
    assert json.loads((run / "training_summary.json").read_text())["optimizer_updates"] == 10000
    for updates in (2500, 10000):
        path = run / f"eval-valfull-euler1-avail3-{updates}/metrics.json"
        data = json.loads(path.read_text())
        assert set(data) == set(TARGETS)
        per_target = {}
        for target, rows in data.items():
            assert len(rows) == 219 and {r["subject"] for r in rows} == reference_ids
            assert all(math.isfinite(r["masked"][m]) for r in rows for m in ("ssim", "psnr"))
            per_target[target] = {m: mean(r["masked"][m] for r in rows) for m in ("ssim", "psnr")}
        sigma1[source, updates] = {"source": source, "batch": 4, "seed": 43, "sigma": 1,
                                  "updates": updates, "per_target": per_target,
                                  **{m: mean(per_target[t][m] for t in TARGETS) for m in ("ssim", "psnr")},
                                  "metric_path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
extended_audit = {**audit, "verified_utc": datetime.now(timezone.utc).isoformat(),
                  "runs_verified": 21, "checkpoint_results_verified": 23,
                  "runs": verified + list(sigma1.values()),
                  "sigma1_note": "BS4 at2500/10000 updates sees10000/40000 examples; same training run per source."}
(OUT / "verification-with-sigma1.json").write_text(json.dumps(extended_audit, indent=2) + "\n")
with (OUT / "scores.csv").open(newline="") as f:
    csv_rows = list(csv.DictReader(f))
with (OUT / "scores-with-sigma1.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
    writer.writeheader()
    writer.writerows(csv_rows)
    for r in sigma1.values():
        writer.writerow({k: r[k] for k in writer.fieldnames if k != "status"} | {"status": "complete"})

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 14,
                     "axes.labelsize": 11, "svg.fonttype": "none", "pdf.fonttype": 42})
BLUE, GREY = "#17679b", "#545b62"
legend = [Line2D([], [], marker="o", linestyle="none", markerfacecolor="white", markeredgecolor=GREY,
                 markeredgewidth=1.7, markersize=7, label="BS1"),
          Line2D([], [], marker="s", linestyle="none", color=BLUE, markersize=7, label="BS4")]

def draw_panel(ax, metric, sigma):
    limits, ticks = ((0.44, 0.68), [0.45, 0.50, 0.55, 0.60, 0.65]) if metric == "ssim" else ((15.5, 18.3), [15.5, 16, 16.5, 17, 17.5, 18])
    for y, source in enumerate(SOURCES):
        b1, b4 = lookup.get((source, sigma, 1)), lookup.get((source, sigma, 4))
        if b1 and b4:
            ax.plot([b1[metric], b4[metric]], [y, y], color="#b7bdc2", lw=1.8, zorder=2)
        for batch, r in ((1, b1), (4, b4)):
            if r:
                ax.plot(r[metric], y, "o" if batch == 1 else "s", ms=7.5,
                        mfc="white" if batch == 1 else BLUE, mec=GREY if batch == 1 else BLUE,
                        mew=1.6 if batch == 1 else 1, zorder=3)
                ax.annotate(f"{r[metric]:.4f}" if metric == "ssim" else f"{r[metric]:.2f}",
                            (r[metric], y), xytext=(0, 7 if batch == 1 else -8),
                            textcoords="offset points", ha="center", va="bottom" if batch == 1 else "top",
                            fontsize=9, color=GREY if batch == 1 else BLUE,
                            bbox=dict(facecolor="white", edgecolor="none", pad=0.3), zorder=4)
        if not b4:
            ax.text(0.57 if metric == "ssim" else 16.9, y, "BS4 pending", va="center",
                    fontsize=10, color=GREY, style="italic")
    ax.set(xlim=limits, xticks=ticks, yticks=range(5), yticklabels=LABELS, ylim=(4.55, -0.55))
    ax.set_title(r"$\sigma = " + str(sigma) + "$", pad=12)
    ax.set_xlabel("SSIM ↑" if metric == "ssim" else "PSNR (dB) ↑", labelpad=9)
    ax.grid(axis="x", color="#e9ebed", lw=0.7, zorder=0)
    ax.tick_params(axis="y", length=0, pad=10)
    ax.tick_params(axis="x", length=3, color="#9a9fa3", labelsize=10)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#bbc0c4")

def draw_sigma1_panel(ax, metric):
    # Separate source labels and explicit wider scale keep the historical panels readable.
    for y, source in enumerate(("mean", "gaussian")):
        early, final = sigma1[source, 2500], sigma1[source, 10000]
        ax.plot([early[metric], final[metric]], [y, y], color="#b7bdc2", lw=1.8, zorder=2)
        for updates, r in ((2500, early), (10000, final)):
            ax.plot(r[metric], y, "D" if updates == 2500 else "s", ms=7.5,
                    mfc="white" if updates == 2500 else BLUE, mec=BLUE, mew=1.6, zorder=3)
            ax.annotate(f"{r[metric]:.4f}" if metric == "ssim" else f"{r[metric]:.2f}",
                        (r[metric], y), xytext=(0, 7 if updates == 2500 else -8),
                        textcoords="offset points", ha="center", va="bottom" if updates == 2500 else "top",
                        fontsize=9, color=BLUE, bbox=dict(facecolor="white", edgecolor="none", pad=0.3), zorder=4)
    limits, ticks = ((0.04, 0.74), [0.1, 0.3, 0.5, 0.7]) if metric == "ssim" else ((8.7, 19.5), [10, 12, 14, 16, 18])
    ax.set(xlim=limits, xticks=ticks, yticks=(0, 1), yticklabels=("Mean", "Gaussian"), ylim=(1.55, -0.55))
    ax.set_title(r"$\sigma = 1$ · BS4", pad=12)
    ax.set_xlabel("SSIM ↑" if metric == "ssim" else "PSNR (dB) ↑", labelpad=9)
    ax.grid(axis="x", color="#e9ebed", lw=0.7, zorder=0)
    ax.tick_params(axis="y", length=0, pad=10)
    ax.tick_params(axis="x", length=3, color="#9a9fa3", labelsize=10)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#bbc0c4")

def make_figure(metrics, filename, include_sigma1=False):
    fig, axes = plt.subplots(len(metrics), 3 if include_sigma1 else 2,
                             figsize=(18 if include_sigma1 else 13, 10.4 if len(metrics) == 2 else 5.0),
                             squeeze=False)
    fig.patch.set_facecolor("white")
    for i, metric in enumerate(metrics):
        for j, sigma in enumerate((0, 0.5)):
            draw_panel(axes[i, j], metric, sigma)
            if j:
                axes[i, j].tick_params(axis="y", labelleft=False)
        if include_sigma1:
            draw_sigma1_panel(axes[i, 2], metric)
    fig.suptitle("Seed 43" if include_sigma1 else "Seed 43 · 10,000 updates",
                 x=0.5, y=0.98, fontsize=19, weight="bold")
    handles = legend if not include_sigma1 else [
        Line2D([], [], marker="o", linestyle="none", mfc="white", mec=GREY, mew=1.7, ms=7, label="BS1 · 10,000 updates"),
        Line2D([], [], marker="s", linestyle="none", color=BLUE, ms=7, label="BS4 · 10,000 updates"),
        Line2D([], [], marker="D", linestyle="none", mfc="white", mec=BLUE, mew=1.7, ms=7, label="BS4 · 2,500 updates")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.925), frameon=False, ncol=len(handles))
    left = 0.14 if include_sigma1 else 0.19
    fig.subplots_adjust(left=left, right=0.98, top=0.82, bottom=0.18 if include_sigma1 else (0.16 if len(metrics) == 2 else 0.23),
                        wspace=0.38 if include_sigma1 else 0.20, hspace=0.52)
    fig.text(left, 0.066 if len(metrics) == 2 else 0.10,
             "Old = conditional MLP    ·    New = shared 8 × 4 table", fontsize=10, color=GREY)
    fig.text(left, 0.034 if len(metrics) == 2 else 0.052,
             ("Examples: BS1/10k = BS4/2.5k = 10,000; BS4/10k = 40,000  |  219 cases · 4 targets · 1 step" if include_sigma1 else
              "BS1: 10,000 examples  ·  BS4: 40,000 examples  |  219 validation cases · 4 targets · 1 sampling step"),
             fontsize=9, color=GREY)
    if include_sigma1:
        fig.text(left, 0.105, "σ = 1 uses wider x-axes to include Gaussian.", fontsize=10, color=GREY)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{filename}.{ext}", dpi=240, facecolor="white")
    plt.close(fig)

make_figure(("ssim", "psnr"), "seed43-comparison")
make_figure(("ssim",), "seed43-ssim")
make_figure(("psnr",), "seed43-psnr")
make_figure(("ssim", "psnr"), "seed43-comparison-with-sigma1-values", include_sigma1=True)
print("Recomputed19 runs / 76 target evaluations; identical219-case sets; prior audited scores reproduced.")
print(OUT / "seed43-comparison.png")
print("Added four verified sigma1 checkpoint results:")
for result in sigma1.values():
    print(result["source"], result["updates"], result["ssim"], result["psnr"])
