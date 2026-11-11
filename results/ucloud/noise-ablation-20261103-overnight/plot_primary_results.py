"""Plot the complete four-seed primary cohort, with matched seeds visible."""
from pathlib import Path
from decimal import Decimal
import csv
import datetime
import hashlib
import json
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

archive = Path(__file__).resolve().parent
plan = json.loads((archive / "run_plan.json").read_text())
seeds, sources = plan["overnight_seeds"], plan["overnight_sources"]
labels = {"mean": "Mean", "uniform": "Learned\nuniform", "random": "Learned\nrandom"}
data = {}
with (archive / "results.csv").open() as stream:
    for row in csv.DictReader(stream):
        if row["scope"] != "primary-full219":
            continue
        if row["status"] != "complete":
            raise SystemExit("Primary cohort is incomplete; no partial plot generated.")
        key = (row["source"], Decimal(row["sigma_max"]), int(row["training_seed"]))
        assert key not in data and int(row["cases"]) == 219 and row["sampling_steps"] == "1"
        data[key] = {metric: Decimal(row[metric]) for metric in ("SSIM", "PSNR")}
expected = {(source, noise, seed) for source in sources for noise in (Decimal(0), Decimal("0.5")) for seed in seeds}
assert set(data) == expected and all(value.is_finite() for scores in data.values() for value in scores.values())

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})
figure, axes = plt.subplots(1, 2, figsize=(12, 5.6))
figure.subplots_adjust(left=0.075, right=0.98, bottom=0.22, top=0.80, wspace=0.25)
colors = {Decimal(0): "#737d8c", Decimal("0.5"): "#2166ac"}
offsets = {Decimal(0): -0.18, Decimal("0.5"): 0.18}
jitter = (-0.055, -0.018, 0.018, 0.055)
stats = []
for axis, metric in zip(axes, ("SSIM", "PSNR")):
    for index, source in enumerate(sources):
        for seed_index, seed in enumerate(seeds):
            axis.plot([index - 0.18 + jitter[seed_index], index + 0.18 + jitter[seed_index]],
                      [float(data[source, noise, seed][metric]) for noise in (Decimal(0), Decimal("0.5"))],
                      color="#bac1ca", linewidth=0.9, alpha=0.8, zorder=1)
        for noise in (Decimal(0), Decimal("0.5")):
            values = [data[source, noise, seed][metric] for seed in seeds]
            mean, sd = statistics.mean(values), statistics.stdev(values)
            axis.scatter([index + offsets[noise] + j for j in jitter], [float(v) for v in values],
                         color=colors[noise], s=33, alpha=0.95, edgecolors="white", linewidths=0.4, zorder=3)
            axis.errorbar(index + offsets[noise], float(mean), yerr=float(sd), fmt="D", color="#171c23",
                          markerfacecolor="white", markersize=5, capsize=4, linewidth=1.4, zorder=4)
            stats.append({"metric": metric, "source": source, "sigma_max": str(noise), "seeds": seeds,
                          "seed_values": [str(v) for v in values], "mean": str(mean), "sample_sd": str(sd)})
    axis.set_xticks(range(len(sources)), [labels[source] for source in sources])
    axis.set_ylabel("SSIM ↑" if metric == "SSIM" else "PSNR (dB) ↑")
    axis.yaxis.set_major_formatter(FormatStrFormatter("%.2f" if metric == "SSIM" else "%.1f"))
    axis.yaxis.set_major_locator(MultipleLocator(0.05 if metric == "SSIM" else 0.5))
    axis.set_xlim(-0.55, len(sources) - 0.45)
    axis.grid(axis="y", color="#dde1e6", linewidth=0.7)
    axis.set_axisbelow(True)
    axis.margins(y=0.13)
figure.suptitle("Primary source comparison · four training seeds", y=0.98, fontsize=16)
legend = [Line2D([], [], marker="o", color="none", markerfacecolor=colors[Decimal(0)], markeredgecolor="none", label="Training sigma_max = 0"),
          Line2D([], [], marker="o", color="none", markerfacecolor=colors[Decimal("0.5")], markeredgecolor="none", label="Training sigma_max = 0.5"),
          Line2D([], [], marker="D", color="#171c23", markerfacecolor="white", label="Mean ± sample SD")]
figure.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.53, 0.93), ncol=3, frameon=False)
figure.text(0.075, 0.075, "Brain-masked validation: 219 cases, four targets averaged equally; one sampling step; 10k training steps, batch size 1.\nDots are training seeds 43, 44, 45 and 2026. Lines connect the same seed across training-noise settings.", fontsize=10, color="#454c57")
for extension in ("png", "svg"):
    figure.savefig(archive / f"primary_source_comparison.{extension}", dpi=180, facecolor="white")
plt.close(figure)
payload = json.dumps(stats, sort_keys=True)
metadata = {"generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "data_source": "results.csv primary-full219 rows only", "primary_data_sha256": hashlib.sha256(payload.encode()).hexdigest(),
            "statistic": "Mean and sample SD across four training seeds, using saved target-summary precision", "groups": stats}
(archive / "primary_source_comparison.meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
print("Saved complete four-seed primary comparison as PNG, SVG and data metadata.")
