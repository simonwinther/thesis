"""Validate full-219 metrics and report seed variation and historical comparisons."""

import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics as stats
import sys

MODALITIES = ("t1n", "t1c", "t2w", "t2f")
ARMS = ("mean", "uniform", "random", "gaussian")
LABELS = dict(mean="Mean", uniform="Learned uniform", random="Learned random",
              biased="Learned biased + fixed partner", gaussian="Gaussian")
EVAL = "eval-valfull-euler1-avail3-10000"


def historical(roots, arm, seed, sigma, expected_cases):
    if isinstance(roots, Path):
        roots = [roots]
    matches = []
    for root in roots:
        found = historical_matches(Path(root), arm, seed, sigma)
        if found:
            matches = found
            break
    if not matches:
        return None
    assert len(matches) == 1, matches
    values = {"ssim": [], "psnr": []}
    for modality in MODALITIES:
        path = matches[0] / EVAL / f"metrics-{modality}.txt"
        text = path.read_text()
        assert re.search(rf"^Contrast:\s*{modality}\s*$", text, re.M), path
        assert re.search(r"^Number of samples:\s*219\s*$", text, re.M), path
        cases = re.findall(r"^(BraTS-[^:\n]+): PSNR=", text, re.M)
        assert len(cases) == 219 and set(cases) == expected_cases, path
        for metric in values:
            value = float(re.search(rf"^{metric.upper()}:\s*([\d.eE+-]+)", text, re.M)[1])
            assert math.isfinite(value), path
            values[metric].append(value)
    return {key: stats.mean(numbers) for key, numbers in values.items()}


def historical_matches(root, arm, seed, sigma):
    if seed == 42 and sigma == 0:
        suffix = dict(mean=981, uniform=984, random=987, biased=990, gaussian=993)[arm]
        matches = list(root.glob(f"is-{arm}-s0-10k-{suffix}"))
    else:
        tag = "0" if sigma == 0 else "05"
        matches = list(root.glob(f"is-{arm}-s{tag}-10k-seed{seed}-*"))
    return matches


def main():
    plan = json.loads(Path(sys.argv[1]).read_text())
    sweep = Path(plan["sweep"])
    arms = plan.get('arms', list(ARMS))
    if len(plan['jobs']) != len(arms) * len(plan['seeds']) * len(plan['sigmas']):
        raise ValueError('Incomplete or unexpected run grid')
    keys = [(job['sigma'], job['arm'], job['seed']) for job in plan['jobs']]
    expected = {(sigma, arm, seed) for sigma in plan['sigmas'] for arm in arms for seed in plan['seeds']}
    if len(set(keys)) != len(keys) or set(keys) != expected:
        raise ValueError('Duplicate or missing run-grid combinations')
    history_roots = plan.get('historical_roots')
    if history_roots is None:
        history_roots = [str(Path(plan['model_root']) / 'runs')]
    incomplete = []
    for job in plan["jobs"]:
        path = Path(job["directory"]) / "status.txt"
        state = path.read_text().strip() if path.exists() else "missing"
        if state != "complete":
            incomplete.append(f"{Path(job['directory']).name}: {state}")
    if incomplete:
        (sweep / "REPORT.md").write_text("Incomplete sweep; no final pooled comparison.\n\n" + "\n".join(incomplete) + "\n")
        (sweep / "status.txt").write_text("finished-with-errors\n")
        raise RuntimeError(f"{len(incomplete)} entries did not complete; see REPORT.md")
    scores, old, records = {}, {}, []
    for job in plan["jobs"]:
        out = Path(job["directory"])
        cases = set(json.loads((out / "split_manifest.json").read_text())["valfull"])
        raw = json.loads((out / EVAL / "metrics.json").read_text())
        assert set(raw) == set(MODALITIES)
        values = {key: [] for key in ("ssim", "psnr")}
        for modality in MODALITIES:
            rows = raw[modality]
            assert len(rows) == 219 and {r["subject"] for r in rows} == cases
            for metric in values:
                numbers = [r["masked"][metric] for r in rows]
                assert all(math.isfinite(n) for n in numbers)
                values[metric].append(stats.mean(numbers))
        key = job["sigma"], job["arm"], job["seed"]
        scores[key] = {name: stats.mean(numbers) for name, numbers in values.items()}
        old[key] = historical(history_roots, job["arm"], job["seed"], job["sigma"], cases)
        expected_old_seeds = plan.get('historical_expected', {}).get(str(job['sigma']), {}).get(job['arm'], [])
        if job['seed'] in expected_old_seeds and old[key] is None:
            raise FileNotFoundError(f'Missing expected historical comparison: {key}')
        record = dict(host=job.get('host', 'unspecified'), sigma=job["sigma"], source=job["arm"], seed=job["seed"], **scores[key])
        for metric in values:
            record[f"historical_{metric}"] = old[key][metric] if old[key] else ""
            record[f"delta_historical_{metric}"] = scores[key][metric] - old[key][metric] if old[key] else ""
        records.append(record)
    for record in records:
        for metric in ("ssim", "psnr"):
            record[f"delta_mean_{metric}"] = record[metric] - scores[record["sigma"], "mean", record["seed"]][metric]
    if plan.get('verify_batch_audits', False):
        for seed in plan['seeds']:
            paths = [Path(job['directory']) / 'batch_targets.csv' for job in plan['jobs'] if job['seed'] == seed]
            fingerprints = {hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
            if len(fingerprints) != 1:
                raise ValueError(f'Patient/target assignments differ between matched arms/noise levels for seed {seed}')
    hosts = {}
    for job in plan['jobs']:
        hosts.setdefault(job.get('host', 'unspecified'), set()).add(job['seed'])
    host_note = '; '.join(f"{host}: seeds {', '.join(map(str, sorted(seeds)))}" for host, seeds in sorted(hosts.items()))
    lines = ["# Shared source table comparison", "",
             "Full BraTS 2023 GLI training set; 10,000 optimizer updates, physical batch size 4, with one distinct "
             "held-out target per sample. One batched forward/backward per update; 40,000 training samples per run. "
             "Full validation: 219 cases, "
             "one Euler step, global evaluation seed 42, three conditioning modalities. The Gaussian source keeps "
             "its historical private seed training_seed+2, so its across-seed variation includes evaluation noise. Brain-masked metrics; "
             "four target modalities averaged equally. Mean ± sample SD across training seeds.", "",
             "Uniform/random use the pure shared 8×4 table with normalization across only each sample's available inputs. "
             "Biased is pending and excluded. " + host_note + ". "
             "All source/noise combinations for a given seed run on the same server. The recorded GPU models may "
             "differ within a server; consult per-run runtime.json.", ""]
    for sigma in plan["sigmas"]:
        lines.extend([f"## sigma_max = {sigma}", ""])
        for metric, digits in (("ssim", 4), ("psnr", 2)):
            lines.extend([f"### {metric.upper()} ↑" + (" (dB)" if metric == "psnr" else ""), "",
                          "| Source | " + " | ".join(f"Seed {seed}" for seed in plan["seeds"]) + " | Mean ± sample SD |",
                          "|---|" + "---:|" * (len(plan["seeds"]) + 1)])
            for arm in arms:
                numbers = [scores[sigma, arm, seed][metric] for seed in plan["seeds"]]
                lines.append("| " + LABELS[arm] + " | " + " | ".join(f"{n:.{digits}f}" for n in numbers)
                             + f" | {stats.mean(numbers):.{digits}f} ± {stats.stdev(numbers):.{digits}f} |")
            lines.append("")
        lines.extend(["### Comparison with historical runs (matching seeds only)", "",
                      "| Source | Matched seeds | Mean ΔSSIM | Mean ΔPSNR (dB) |",
                      "|---|---|---:|---:|"])
        for arm in arms:
            seeds = [s for s in plan["seeds"] if old[sigma, arm, s] is not None]
            if seeds:
                deltas = {m: stats.mean(scores[sigma, arm, s][m] - old[sigma, arm, s][m] for s in seeds)
                          for m in ("ssim", "psnr")}
                lines.append(f"| {LABELS[arm]} | {', '.join(map(str, seeds))} | {deltas['ssim']:+.4f} | {deltas['psnr']:+.2f} |")
        lines.extend(["", "Learned sources compared with the NEW mean baseline within each seed:", ""])
        for arm in (a for a in arms if a not in ('mean', 'gaussian')):
            deltas = [scores[sigma, arm, s]["ssim"] - scores[sigma, "mean", s]["ssim"] for s in plan["seeds"]]
            lines.append(f"- {LABELS[arm]}: mean ΔSSIM {stats.mean(deltas):+.4f}; higher SSIM in {sum(d > 0 for d in deltas)}/{len(plan['seeds'])} seeds.")
        lines.append("")
    lines.extend(["Historical pilots used batch size 1: the new batch size 4 has four times the sample exposure per 10k "
                  "optimizer updates. Historical learned uniform/random used an image/target-conditioned MLP plus "
                  "per-target prior logits; the new versions use only a shared 32-entry table. The initializer labels "
                  "therefore do not denote identical source architectures. Activation checkpointing and hardware/software "
                  "also differ. Changes from historical runs cannot be attributed solely to the source table. "
                  "Historical seed 42 ran on Mostafa; other historical seeds ran on B200, with different software/hardware. "
                  "Historical noisy Gaussian results cover only seeds 43 and 44. Historical means come from rounded "
                  "metric summaries; new metrics retain full precision. Cross-experiment changes include environment effects. "
                  "Within-sweep comparisons against mean are the primary comparison. These are descriptive pilot results; "
                  "mean ± SD does not establish statistical significance.", ""])
    (sweep / "REPORT.md").write_text("\n".join(lines))
    with (sweep / "scores.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (sweep / "scores.json").write_text(json.dumps(records, indent=2) + "\n")
    (sweep / "status.txt").write_text("complete\n")
    print(f"Validated {len(records)} runs; wrote {sweep / 'REPORT.md'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        plan = json.loads(Path(sys.argv[1]).read_text())
        sweep = Path(plan["sweep"])
        (sweep / "status.txt").write_text("finished-with-errors\n")
        report = sweep / "REPORT.md"
        if not report.exists() or not report.read_text().startswith("Incomplete sweep"):
            report.write_text(f"Result validation failed; no final comparison is valid.\n\n{type(error).__name__}: {error}\n")
        raise
