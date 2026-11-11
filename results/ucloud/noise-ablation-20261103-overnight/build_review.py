"""Build a local review using validated, brain-masked metric summaries only."""
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime
from zoneinfo import ZoneInfo
import csv
import hashlib
import json
import re
import statistics

ARCHIVE = Path(__file__).resolve().parent
PLAN = json.loads((ARCHIVE / "run_plan.json").read_text())
SEEDS = PLAN["overnight_seeds"]
SOURCES = PLAN["overnight_sources"]
LABELS = {"mean": "Mean", "uniform": "Learned uniform", "random": "Learned random", "biased": "Learned biased", "gaussian": "Gaussian"}
MODALITIES = ("t1n", "t1c", "t2w", "t2f")
verified, patient_sets, primary, secondary, noisy_secondary, additional, jobs = [], {}, {}, {}, {}, {}, []
target_rows = []
gaussian_secondary = {}


def state(run):
    path = run / "status.txt"
    return path.read_text().strip() if path.exists() else "not started"


def metrics(directory, count):
    totals = {name: [] for name in ("SSIM", "PSNR")}
    targets = {}
    for modality in MODALITIES:
        path = directory / f"metrics-{modality}.txt"
        text = path.read_text()
        assert re.search(r"^Contrast:\s*" + modality + r"\s*$", text, re.M), path
        assert re.search(r"^Number of samples:\s*" + str(count) + r"\s*$", text, re.M), path
        patients = re.findall(r"^(BraTS-[^:\n]+): PSNR=", text, re.M)
        assert len(patients) == len(set(patients)) == count, path
        reference = patient_sets.setdefault(count, set(patients))
        assert set(patients) == reference, f"Patient sets differ: {path}"
        for name in totals:
            match = re.search(r"^" + name + r":\s*([0-9.eE+-]+)", text, re.M)
            assert match, path
            value = Decimal(match.group(1))
            assert value.is_finite(), path
            totals[name].append(value)
            targets.setdefault(modality, {})[name] = value
        verified.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "cases": count})
    return {**{name: sum(values) / 4 for name, values in totals.items()}, "targets": targets}


def control(source, seed):
    base = Path(PLAN["noise_free_controls"][str(seed)])
    matches = list(base.glob(f"is-{source}-s0-10k-seed{seed}-*"))
    assert len(matches) == 1, f"Ambiguous control: {source}, {seed}"
    assert state(matches[0]) == "complete", matches[0]
    return matches[0]


def record(scope, source, seed, sigma, steps, run, directory, cases):
    status = state(run)
    values = metrics(directory, cases) if status == "complete" else None
    jobs.append({"scope": scope, "source": source, "training_seed": seed, "sigma_max": sigma,
                 "sampling_steps": steps, "cases": cases, "status": status, "run": str(run),
                 "SSIM": values["SSIM"] if values else "", "PSNR": values["PSNR"] if values else ""})
    if values:
        for modality, scores in values["targets"].items():
            target_rows.append({**jobs[-1], "target": modality, **scores,
                                "metric_path": str(directory / f"metrics-{modality}.txt")})
    return values


for sigma in (0, 0.5):
    for source in SOURCES:
        for seed in SEEDS:
            run = control(source, seed) if sigma == 0 else ARCHIVE / "runs" / f"is-{source}-s05-10k-seed{seed}-{PLAN['run_id']}"
            directory = run / "eval-valfull-euler1-avail3-10000"
            primary[sigma, source, seed] = record("primary-full219", source, seed, sigma, 1, run, directory, 219)

for sigma in (0, 0.5):
    for source in PLAN["additional_sources"]:
        for seed in PLAN["additional_seeds"]:
            run = control(source, seed) if sigma == 0 else ARCHIVE / "runs" / f"is-{source}-s05-10k-seed{seed}-{PLAN['run_id']}"
            directory = run / "eval-valfull-euler1-avail3-10000"
            additional[sigma, source, seed] = record("additional-full219", source, seed, sigma, 1, run, directory, 219)

for source in ("mean", "uniform", "gaussian"):
    for seed in (43, 44):
        run = control(source, seed)
        secondary[1, source, seed] = record("diagnostic-val50", source, seed, 0, 1, run, run / "eval-val50-euler1-avail3-10000", 50)
        run = ARCHIVE / "runs" / f"eval-{source}-s0-seed{seed}-euler10-val50-{PLAN['run_id']}"
        secondary[10, source, seed] = record("diagnostic-val50", source, seed, 0, 10, run, run, 50)
for source in PLAN["noisy_diagnostics"]["sources"]:
    for seed in PLAN["noisy_diagnostics"]["seeds"]:
        run = ARCHIVE / "runs" / f"is-{source}-s05-10k-seed{seed}-{PLAN['run_id']}"
        noisy_secondary[1, source, seed] = record("diagnostic-noisy-val50", source, seed, 0.5, 1, run, run / "eval-val50-euler1-avail3-10000", 50)
        run = ARCHIVE / "runs" / f"eval-{source}-s05-seed{seed}-euler10-val50-{PLAN['run_id']}"
        noisy_secondary[10, source, seed] = record("diagnostic-noisy-val50", source, seed, 0.5, 10, run, run, 50)
assert patient_sets[50].issubset(patient_sets[219]), "val50 is not a subset of full validation"
for seed in PLAN["gaussian_noisy_diagnostics"]["seeds"]:
    for steps in PLAN["gaussian_noisy_diagnostics"]["sampling_steps"]:
        run = ARCHIVE / "runs" / f"eval-gaussian-s05-seed{seed}-euler{steps}-val50-{PLAN['run_id']}"
        gaussian_secondary[steps, "gaussian", seed] = record("diagnostic-gaussian-noisy-val50", "gaussian", seed, 0.5, steps, run, run, 50)
environment_path = ARCHIVE / "runs" / PLAN["sweep_name"] / "runtime_environment.json"
environment = json.loads(environment_path.read_text()) if environment_path.exists() else None
if environment:
    assert patient_sets[219] == set(environment["splits"]["valfull"]["case_ids"])
    assert patient_sets[50] == set(environment["splits"]["val50"]["case_ids"])
    assert environment["train_val_case_overlap"] == 0

noise_done = sum(primary[0.5, a, s] is not None for a in SOURCES for s in SEEDS)
eval_done = sum(secondary[10, a, s] is not None for a in ("mean", "uniform", "gaussian") for s in (43, 44))
extra_total = len(PLAN["additional_sources"]) * len(PLAN["additional_seeds"])
extra_done = sum(additional[0.5, a, s] is not None for a in PLAN["additional_sources"] for s in PLAN["additional_seeds"])
noisy_eval_done = sum(noisy_secondary[10, a, s] is not None for a in PLAN["noisy_diagnostics"]["sources"] for s in PLAN["noisy_diagnostics"]["seeds"])
gaussian_eval_done = sum(value is not None for value in gaussian_secondary.values())
overall = state(ARCHIVE / "runs" / PLAN["completion_sweep_name"])
backup = (ARCHIVE / "backup_status.txt").read_text().strip()
finished = noise_done == 12 and extra_done == extra_total and eval_done == 6 and noisy_eval_done == 4 and gaussian_eval_done == 4 and overall == "complete" and backup == "final results copied; pilot=complete"


def fmt(value, digits, signed=False):
    if value is None:
        return "—"
    rounded = Decimal(value).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP)
    return format(rounded, ("+" if signed else "") + f".{digits}f")


def aggregate(values, digits):
    if any(value is None for value in values):
        return "pending"
    return f"{fmt(statistics.mean(values), digits)} ± {fmt(statistics.stdev(values), digits)}"


lines = ["# Overnight B200 review — 2026-11-03", "",
         f"**{'VALIDATED RESULTS COMPLETE' if finished else 'IN PROGRESS — do not treat pending cells as results'}**", "",
         f"Updated: {datetime.now(ZoneInfo('Europe/Copenhagen')).isoformat(timespec='seconds')}", "",
         f"Noise-enabled trials validated: **{noise_done}/12**. Ten-step diagnostics validated: **{eval_done}/6**.",
         f"Additional two-seed source checks validated: **{extra_done}/{extra_total}**.",
         f"Additional noise-enabled ten-step diagnostics validated: **{noisy_eval_done}/4**.",
         f"Noise-enabled Gaussian one/ten-step diagnostics validated: **{gaussian_eval_done}/4**.",
         f"Overall queue: `{overall}`. Backup: `{backup}`.", "",
         "All primary comparisons use the same four B200 training seeds, 1,251 training cases, 10k training steps, batch size 1, UNet lr 1e-5, learned-source lr 1e-3 and three available conditioning modalities. Evaluation uses seed 42, one sampling step and all 219 validation cases. Metrics are brain-masked and average the four target-modality summary means equally. Sample SD is across the four training seeds, computed before display rounding. No pooled estimate is shown for an incomplete row.", "",
         "Noise level is the training coefficient `sigma_max`; the two levels are 0 and 0.5. The implemented schedule is `sigma(t) = sigma_max * sqrt(t * (1-t) + 1e-8)`, so a coefficient of 0.5 gives a peak of approximately 0.25 at t = 0.5. This is a matched noise ablation, not the full paper's 50k-step, batch-size-4 protocol. These summaries are descriptive and do not claim statistical significance.", "",
         "Saved target-modality means have four decimal places for SSIM and two for PSNR. All averages and sample SDs use those saved means before final display rounding. The SD shown here measures variation between training seeds; it is distinct from the variation between validation patients reported inside each metric file."]
lines += ["", "Mean fixes each available modality's weight at 1/3 in every wavelet band. The learned variants share the same source MLP: uniform starts at equal weights, random starts with logit perturbations of standard deviation 0.1 (weights near 1/3), and biased starts at 0.6 for the paired modality and 0.2 for each other modality (T1n–T1c and T2w–T2f pairs). Learning can change these weights. Gaussian starts from independent standard-normal wavelet coefficients at both noise levels. Every variant keeps the same three MRI scans as network conditioning; the noise column controls the separate training-path noise term."]
if environment:
    lines += ["", f"Runtime captured: Python {environment['python'].split()[0]}, PyTorch {environment['packages']['torch']}. The manifest verifies 1,251 training cases, 219 full-validation cases, the fixed val50 subset and disjoint training/validation case IDs. GPU UUIDs, driver/package versions and actual initial training commands are saved in runs/{PLAN['sweep_name']}/runtime_environment.json."]
if noise_done == 12 and (ARCHIVE / "primary_source_comparison.png").exists():
    lines += ["", f"Primary-cohort plot with individual seeds and mean ± sample SD: [PNG]({ARCHIVE / 'primary_source_comparison.png'}), [SVG]({ARCHIVE / 'primary_source_comparison.svg'})."]
if finished:
    averages = {source: {metric: statistics.mean(primary[0.5, source, seed][metric] for seed in SEEDS) for metric in ("SSIM", "PSNR")} for source in SOURCES}
    lines[6:6] = [f"The learned-source SSIM advantage seen without training noise did not persist in the noise-enabled primary cohort. At sigma_max = 0.5, average SSIM is {fmt(averages['mean']['SSIM'], 4)} for mean, {fmt(averages['uniform']['SSIM'], 4)} for learned uniform and {fmt(averages['random']['SSIM'], 4)} for learned random. Uniform beats mean in one of four seeds; random in two of four. Random has a small average PSNR edge ({fmt(averages['random']['PSNR'], 2)} versus {fmt(averages['mean']['PSNR'], 2)} dB), with mixed directions across seeds. These are descriptive pilot results.", "",
                  "The sampling diagnostic also depends on training noise: ten steps improve Gaussian SSIM at sigma_max = 0 in both seeds, but worsen both SSIM and PSNR at sigma_max = 0.5 in both seeds. This finding uses the fixed 50-case subset and must stay separate from the 219-case primary results.", ""]

for name, digits in (("SSIM", 4), ("PSNR", 2)):
    lines += ["", f"**Primary {name}{' (dB)' if name == 'PSNR' else ''} ↑ — full 219 cases, one step**", "",
              "| Source | Training noise | " + " | ".join(f"Seed {s}" for s in SEEDS) + " | Mean ± sample SD |",
              "|---|---:|" + "---:|" * (len(SEEDS) + 1)]
    for sigma in (0, 0.5):
        for source in SOURCES:
            values = [primary[sigma, source, s][name] if primary[sigma, source, s] else None for s in SEEDS]
            lines.append(f"| {LABELS[source]} | {sigma} | " + " | ".join(fmt(v, digits) for v in values) + f" | {aggregate(values, digits)} |")
    lines += ["", f"**Paired {name} difference: learned source minus mean**", "",
              "| Source | Training noise | " + " | ".join(f"Seed {s}" for s in SEEDS) + " | Mean difference |",
              "|---|---:|" + "---:|" * (len(SEEDS) + 1)]
    for sigma in (0, 0.5):
        for source in ("uniform", "random"):
            deltas = []
            for seed in SEEDS:
                a, b = primary[sigma, source, seed], primary[sigma, "mean", seed]
                deltas.append(a[name] - b[name] if a and b else None)
            mean = statistics.mean(deltas) if all(d is not None for d in deltas) else None
            lines.append(f"| {LABELS[source]} | {sigma} | " + " | ".join(fmt(v, digits, True) for v in deltas) + f" | {fmt(mean, digits, True)} |")

lines += ["", "**Separate exploratory source comparison — two training seeds, full 219 cases, one step**", "",
          "Gaussian and learned biased were added after the first noisy result. Keep this two-seed cohort separate from the four-seed primary cohort. Mean rows use the same two seeds as controls. One additional training process per GPU may overlap the primary queue, with a 64 GiB free-memory admission check; this execution-scheduling difference is recorded in the run plan."]
for name, digits in (("SSIM", 4), ("PSNR", 2)):
    lines += ["", f"**Exploratory {name}{' (dB)' if name == 'PSNR' else ''} ↑**", "",
              "| Source | Training noise | " + " | ".join(f"Seed {s}" for s in PLAN["additional_seeds"]) + " | Mean ± sample SD |",
              "|---|---:|" + "---:|" * (len(PLAN["additional_seeds"]) + 1)]
    for sigma in (0, 0.5):
        for source in ("mean", *PLAN["additional_sources"]):
            cohort = primary if source == "mean" else additional
            values = [cohort[sigma, source, s][name] if cohort[sigma, source, s] else None for s in PLAN["additional_seeds"]]
            lines.append(f"| {LABELS[source]} | {sigma} | " + " | ".join(fmt(v, digits) for v in values) + f" | {aggregate(values, digits)} |")

lines += ["", "**Separate diagnostic — fixed val50, sigma-zero checkpoints, one versus ten sampling steps**", "",
          "These 50-case results must not be mixed with the full-validation tables. The same existing 10k-step checkpoints and evaluation seed are used at both sampling budgets. No new training is performed.", "",
          "| Source | Training seed | SSIM, 1 step | SSIM, 10 steps | PSNR, 1 step | PSNR, 10 steps |",
          "|---|---:|---:|---:|---:|---:|"]
for source in ("mean", "uniform", "gaussian"):
    for seed in (43, 44):
        one, ten = secondary[1, source, seed], secondary[10, source, seed]
        lines.append(f"| {LABELS[source]} | {seed} | {fmt(one['SSIM'], 4)} | {fmt(ten['SSIM'] if ten else None, 4)} | {fmt(one['PSNR'], 2)} | {fmt(ten['PSNR'] if ten else None, 2)} |")

lines += ["", "**Separate diagnostic — fixed val50, noise-enabled checkpoints, one versus ten sampling steps**", "",
          "These exploratory checks use checkpoints trained with sigma_max = 0.5 at seeds 43 and 44, using the same validation patients and evaluation seed. Gaussian one/ten-step checks complete the sampling comparison for all three diagnostic sources at both noise levels. Keep this two-seed comparison separate from the primary four-seed results on 219 cases.", "",
          "| Source | Training seed | SSIM, 1 step | SSIM, 10 steps | PSNR, 1 step | PSNR, 10 steps |",
          "|---|---:|---:|---:|---:|---:|"]
for source in [*PLAN["noisy_diagnostics"]["sources"], "gaussian"]:
    for seed in PLAN["noisy_diagnostics"]["seeds"]:
        cohort = gaussian_secondary if source == "gaussian" else noisy_secondary
        one, ten = cohort[1, source, seed], cohort[10, source, seed]
        lines.append(f"| {LABELS[source]} | {seed} | {fmt(one['SSIM'] if one else None, 4)} | {fmt(ten['SSIM'] if ten else None, 4)} | {fmt(one['PSNR'] if one else None, 2)} | {fmt(ten['PSNR'] if ten else None, 2)} |")

lines += ["", "**Observations**", ""]
for source in SOURCES:
    complete = [(s, primary[0.5, source, s], primary[0, source, s]) for s in SEEDS if primary[0.5, source, s]]
    if complete:
        changes = ", ".join(f"seed {s}: {fmt(a['SSIM'] - b['SSIM'], 4, True)} SSIM / {fmt(a['PSNR'] - b['PSNR'], 2, True)} dB" for s, a, b in complete)
        lines.append(f"- Adding training noise, {LABELS[source].lower()}: {changes}. " + ("All four seeds are complete." if len(complete) == 4 else "Interim; remaining seeds are pending."))
for source in ("uniform", "random"):
    paired = [(s, primary[0.5, source, s], primary[0.5, "mean", s]) for s in SEEDS]
    complete = [(s, a, b) for s, a, b in paired if a and b]
    if complete:
        wins = sum(a["SSIM"] > b["SSIM"] for _, a, b in complete)
        lines.append(f"- With noise enabled, {LABELS[source].lower()} beats mean in SSIM in {wins}/{len(complete)} validated matched seeds. " + ("All four seeds are complete." if len(complete) == 4 else "Remaining seeds are pending; this is an interim observation."))
for source in PLAN["additional_sources"]:
    complete = [(s, additional[0.5, source, s], primary[0.5, "mean", s]) for s in PLAN["additional_seeds"] if additional[0.5, source, s] and primary[0.5, "mean", s]]
    if complete:
        changes = ", ".join(f"seed {s}: {fmt(a['SSIM'] - b['SSIM'], 4, True)} SSIM / {fmt(a['PSNR'] - b['PSNR'], 2, True)} dB" for s, a, b in complete)
        lines.append(f"- Exploratory two-seed comparison, {LABELS[source].lower()} minus mean with noise enabled: {changes}. " + ("Both seeds are complete." if len(complete) == 2 else "The other seed is pending."))
if noise_done == 0:
    lines.append("- No noise-enabled full-validation result is complete yet; no overnight result conclusions are available.")
if eval_done == 6:
    for source in ("mean", "uniform", "gaussian"):
        ds = [secondary[10, source, s]["SSIM"] - secondary[1, source, s]["SSIM"] for s in (43, 44)]
        lines.append(f"- For sigma_max = 0 checkpoints, ten-step versus one-step {LABELS[source].lower()} SSIM changes on val50: seed 43 {fmt(ds[0], 4, True)}, seed 44 {fmt(ds[1], 4, True)}.")
if noisy_eval_done == 4:
    for source in PLAN["noisy_diagnostics"]["sources"]:
        ds = [noisy_secondary[10, source, s]["SSIM"] - noisy_secondary[1, source, s]["SSIM"] for s in PLAN["noisy_diagnostics"]["seeds"] if noisy_secondary[1, source, s]]
        if len(ds) == 2:
            lines.append(f"- For sigma_max = 0.5 checkpoints, ten-step versus one-step {LABELS[source].lower()} SSIM changes on val50: seed 43 {fmt(ds[0], 4, True)}, seed 44 {fmt(ds[1], 4, True)}.")
if gaussian_eval_done == 4:
    ds = [gaussian_secondary[10, "gaussian", s]["SSIM"] - gaussian_secondary[1, "gaussian", s]["SSIM"] for s in PLAN["gaussian_noisy_diagnostics"]["seeds"]]
    lines.append(f"- For sigma_max = 0.5 checkpoints, ten-step versus one-step Gaussian SSIM changes on val50: seed 43 {fmt(ds[0], 4, True)}, seed 44 {fmt(ds[1], 4, True)}.")
lines += ["", "**Run status and provenance**", "", "| Task | Source | Seed | Noise | Steps | Status |", "|---|---|---:|---:|---:|---|"]
for job in jobs:
    if (job["scope"] in {"primary-full219", "additional-full219"} and job["sigma_max"] == 0.5) or (job["scope"] in {"diagnostic-val50", "diagnostic-noisy-val50"} and job["sampling_steps"] == 10) or job["scope"] == "diagnostic-gaussian-noisy-val50":
        lines.append(f"| {job['scope']} | {LABELS[job['source']]} | {job['training_seed']} | {job['sigma_max']} | {job['sampling_steps']} | {job['status']} |")
lines += ["", "See run_plan.json, initial_core_run_plan.json and OBSERVATIONS.md for decisions, exact settings, code hashes and the original scope. results.csv records every macro metric's run path; per_target_results.csv preserves all four target means separately for exploratory checks. verified_metric_files.json records the local SHA256 and case count of each parsed file. archive_verification.json audits configurations, code and remote/local result agreement; its all_complete flag must be true before declaring the night finished. Lightweight backups exclude checkpoints, generated volumes and TensorBoard events.", ""]
(ARCHIVE / "MORNING_REVIEW.md").write_text("\n".join(lines))
with (ARCHIVE / "results.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(jobs[0]))
    writer.writeheader()
    writer.writerows(jobs)
with (ARCHIVE / "per_target_results.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(target_rows[0]))
    writer.writeheader()
    writer.writerows(target_rows)
(ARCHIVE / "verified_metric_files.json").write_text(json.dumps(verified, indent=2) + "\n")
print(f"Morning review updated: {noise_done}/12 primary noisy trials, {extra_done}/{extra_total} additional noisy trials, {eval_done}/6 original, {noisy_eval_done}/4 noisy and {gaussian_eval_done}/4 Gaussian-budget diagnostics; {len(verified)} metric files verified.")
