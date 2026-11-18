"""Show seed43 results and explicit gaps, excluding the shared32 table."""
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
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
TARGETS = ("t1n", "t1c", "t2w", "t2f")
SOURCES = ("mean", "gaussian", "old_uniform", "old_random", "mlp8_uniform", "mlp8_random")
LABELS = ("Mean", "Gaussian", "MLP 72 Uniform", "MLP 72 Random", "MLP 8 Uniform", "MLP 8 Random")
specs = []
historical = json.loads((ROOT / "results/seed43-comparison-20261112/verification.json").read_text())
for row in historical["runs"]:
    if row["source"] in ("mean", "old_uniform", "old_random"):
        specs.append({**row, "directory": row["metric_directory"], "expected": {m: row[m] for m in ("ssim", "psnr")}})
recent = json.loads((ROOT / "results/seed43-mlp-checkpoints-20261113/verification.json").read_text())
for row in recent["results"]:
    source = row["run"].rsplit("-s", 1)[0].replace("-", "_")
    specs.append({**row, "source": source, "batch": row["batch_size"],
                  "directory": str(Path(row["metrics_path"]).parent),
                  "sha256": {"metrics.json": row["sha256"]},
                  "expected": {m: row[m] for m in ("ssim", "psnr")}})
for sigma, folder, run in (
    (0, "pilot-seed43-20261102-repeat", "is-gaussian-s0-10k-seed43-20261102-repeat"),
    (0.5, "noise-ablation-20261103-overnight", "is-gaussian-s05-10k-seed43-20261103-overnight"),
):
    specs.append({"source": "gaussian", "batch": 1, "sigma": sigma, "seed": 43, "updates": 10000,
                  "directory": str(ROOT / "results/ucloud" / folder / "runs" / run / "eval-valfull-euler1-avail3-10000")})

records, reference_ids = [], None
for spec in specs:
    assert spec["seed"] == 43 and spec["source"] in SOURCES
    directory = Path(spec["directory"])
    for filename, expected in spec.get("sha256", {}).items():
        assert hashlib.sha256((directory / filename).read_bytes()).hexdigest() == expected
    json_path = directory / "metrics.json"
    data = json.loads(json_path.read_text()) if json_path.exists() else None
    hashes, per_target = {}, {}
    if data is not None:
        assert set(data) == set(TARGETS)
        hashes["metrics.json"] = hashlib.sha256(json_path.read_bytes()).hexdigest()
    for target in TARGETS:
        if data is not None:
            rows = [(r["subject"], r["masked"]["psnr"], r["masked"]["ssim"]) for r in data[target]]
        else:
            path = directory / f"metrics-{target}.txt"
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            rows = [(m[0], float(m[1]), float(m[2])) for m in re.findall(
                r"^(BraTS-\S+): PSNR=([-\d.eE+]+), SSIM=([-\d.eE+]+)", path.read_text(), re.M)]
        ids = {r[0] for r in rows}
        assert len(rows) == len(ids) == 219
        reference_ids = ids if reference_ids is None else reference_ids
        assert ids == reference_ids
        assert all(math.isfinite(v) for r in rows for v in r[1:])
        per_target[target] = {"psnr": mean(r[1] for r in rows), "ssim": mean(r[2] for r in rows)}
    macro = {m: mean(per_target[t][m] for t in TARGETS) for m in ("ssim", "psnr")}
    if "expected" in spec:
        assert all(abs(macro[m] - spec["expected"][m]) < 1e-10 for m in macro)
    records.append({k: spec[k] for k in ("source", "batch", "sigma", "seed", "updates")} | macro |
                   {"per_target": per_target, "metric_directory": str(directory), "sha256": hashes,
                    "precision": "full_precision_json" if data is not None else "rounded_per_case_text"})

lookup = {(r["source"], r["sigma"], r["batch"], r["updates"]): r for r in records}
assert len(lookup) == len(records) == 33
columns = [(sigma, batch, steps) for sigma in (0, 0.5, 1, 1.5, 2)
           for batch, steps in ((1, 10000), (4, 2500), (4, 10000))]
status_snapshot = json.loads((OUT / "status_snapshot.json").read_text())
active = {(r["source"], r["sigma"], r["batch"], steps): r["display_status"]
          for r in status_snapshot["experiments"] for steps in r["evaluation_steps"]}
assert all(key not in lookup for key in active)
assert all(key[0] in SOURCES and key[1:] in columns for key in active)
audit = {"verified_utc": datetime.now(timezone.utc).isoformat(), "seed": 43,
         "checkpoint_results": len(records), "same_219_case_ids": True, "sampling_steps": 1,
         "scope": "Mean, Gaussian, original MLP72 uniform/random, MLP8 uniform/random; BS1/BS4.",
         "excluded": "Shared32 uniform/random and biased variants; other seeds. Unfinished evaluations have status labels only.",
         "columns": [{"sigma": s, "batch": b, "updates": u} for s, b, u in columns],
         "status_snapshot": status_snapshot,
         "coverage": {"complete": len(lookup), "in_progress_or_queued": len(active),
                      "missing_not_active": len(SOURCES) * len(columns) - len(lookup) - len(active)},
         "averaging": "Brain-masked case means averaged equally over four targets; three available inputs.",
         "limitations": ["Different example budgets and execution environments; single-seed comparison.",
                         "Historical text exports contain rounded per-case values.",
                         "Dashes mean neither a completed evaluation nor an active/queued run in this comparison; never zero.",
                         "Running cells refer to the parent run: checkpoint evaluations occur after training finishes."],
         "historical_display_notes": ["Gaussian BS1 sigma0 PSNR displays13.52 from per-case text means versus13.53 in older rounded-header tables.",
                                      "Gaussian BS1 sigma0.5 SSIM displays0.2104 versus0.2103 from rounded headers. Full-precision historical values are unavailable."],
         "results": records}
(OUT / "verification.json").write_text(json.dumps(audit, indent=2) + "\n")
with (OUT / "scores.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["source", "sigma", "batch", "updates", "seed", "ssim", "psnr"])
    writer.writeheader()
    writer.writerows({k: r[k] for k in writer.fieldnames} for r in records)
with (OUT / "coverage.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["source", "sigma", "batch", "updates", "seed", "status", "ssim", "psnr"])
    writer.writeheader()
    for source in SOURCES:
        for sigma, batch, steps in columns:
            key = source, sigma, batch, steps
            result = lookup.get(key, {})
            writer.writerow({"source": source, "sigma": sigma, "batch": batch, "updates": steps,
                             "seed": 43, "status": "Complete" if result else active.get(key, "Missing"),
                             "ssim": result.get("ssim", ""), "psnr": result.get("psnr", "")})

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "svg.fonttype": "none", "pdf.fonttype": 42})
colour = LinearSegmentedColormap.from_list("scores", ["#f4f8fc", "#b6d3e7", "#5594bf", "#125681"])
colour.set_bad("#eeeeee")

def panel(ax, metric):
    matrix = np.array([[lookup[(s, sigma, batch, steps)][metric] if (s, sigma, batch, steps) in lookup
                        else np.nan for sigma, batch, steps in columns] for s in SOURCES])
    norm = Normalize(0, 0.70) if metric == "ssim" else Normalize(9, 19)
    image = ax.imshow(np.ma.masked_invalid(matrix), cmap=colour, norm=norm, aspect="auto")
    decimals = 4 if metric == "ssim" else 2
    for col in range(len(columns)):
        best = max((round(v, decimals) for v in matrix[:, col] if math.isfinite(v)), default=None)
        for row in range(len(SOURCES)):
            value = matrix[row, col]
            if not math.isfinite(value):
                status = active.get((SOURCES[row], *columns[col]))
                ax.text(col, row, status or "—", ha="center", va="center",
                        color="#454d55" if status else "#999999", fontsize=10 if status else 12,
                        style="italic" if status else "normal")
                continue
            is_best = round(value, decimals) == best
            ax.text(col, row, f"{value:.{decimals}f}", ha="center", va="center", fontsize=11,
                    color="white" if norm(value) > 0.68 else "#202a33",
                    weight="bold" if is_best else "normal")
            if is_best:
                ax.add_patch(Rectangle((col-0.46, row-0.44), 0.92, 0.88, fill=False,
                                       edgecolor="#111111", linewidth=1.8))
    ax.set_yticks(range(len(SOURCES)), LABELS)
    ax.set_xticks(range(len(columns)), [f"BS{batch}\n{steps/1000:g}k updates" for _, batch, steps in columns])
    ax.xaxis.tick_top()
    ax.tick_params(axis="both", length=0, pad=9)
    ax.set_xticks(np.arange(-0.5, len(columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(SOURCES), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)
    for sigma in sorted({c[0] for c in columns}):
        indices = [i for i,c in enumerate(columns) if c[0] == sigma]
        ax.text(mean(indices), 1.29, f"σ = {sigma:g}", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=14)
        if indices[0]:
            ax.axvline(indices[0]-0.5, color="white", linewidth=5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("SSIM ↑" if metric == "ssim" else "PSNR (dB) ↑", loc="left", fontsize=17, pad=78, weight="bold")
    bar = ax.figure.colorbar(image, ax=ax, fraction=0.026, pad=0.018)
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=9, length=3)

def draw(metrics, name):
    combined = len(metrics) == 2
    fig, axes = plt.subplots(len(metrics), 1, figsize=(22, 11.8 if combined else 7.0), squeeze=False)
    fig.subplots_adjust(left=0.14, right=0.97, top=0.81 if combined else 0.68,
                        bottom=0.16 if combined else 0.24, hspace=0.95)
    for ax, metric in zip(axes[:,0], metrics):
        panel(ax, metric)
    fig.suptitle("Seed 43 · Source comparison", fontsize=20, weight="bold", y=0.98)
    fig.text(0.14, 0.112 if combined else 0.17,
             "Darker = higher score. Outline = column best. Running = run active; result pending. Queued = waiting. — = no result or active/queued run.",
             fontsize=10, color="#454d55")
    fig.text(0.14, 0.080 if combined else 0.12,
             "MLP 72: original inputs · MLP 8: same MLP with statistics removed", fontsize=10, color="#454d55")
    fig.text(0.14, 0.048 if combined else 0.07,
             "Examples: BS1/10k = BS4/2.5k = 10,000; BS4/10k = 40,000", fontsize=10, color="#454d55")
    fig.text(0.14, 0.019 if combined else 0.025,
             "219 validation cases · 4 targets · brain-masked means · 1 sampling step · GPUs differ"
             f" · Status: {status_snapshot['checked_local']}", fontsize=9, color="#606870")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT/f"{name}.{ext}", dpi=240, facecolor="white")
    plt.close(fig)

draw(("ssim", "psnr"), "seed43-full-overview")
draw(("ssim",), "seed43-ssim-overview")
draw(("psnr",), "seed43-psnr-overview")
print(f"Verified {len(records)} checkpoint results, {len(SOURCES)} model variants, {len(columns)} budget/noise columns; shared32 excluded.")
print(audit["coverage"])
print(OUT / "seed43-full-overview.png")
