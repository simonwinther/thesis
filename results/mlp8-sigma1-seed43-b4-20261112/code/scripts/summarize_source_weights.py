"""Summarize checkpoint weights across patients without copying large CSV dumps."""
from pathlib import Path
import argparse
import csv
import json
import math
import statistics
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--cases", type=int, default=219)
    args = parser.parse_args()
    source = args.input
    if source.name != "source_weights_checkpoints.csv":
        raise ValueError("Expected a checkpoint-weight CSV, not a model checkpoint")
    if (source.parent.parent / "status.txt").read_text().strip() != "complete":
        raise ValueError("Only summarize a completed run")
    groups = defaultdict(dict)
    subjects = set()
    with source.open() as stream:
        for row in csv.DictReader(stream):
            key = (int(row["step"]), row["target"], row["subband"], row["modality"])
            value = float(row["weight"])
            assert math.isfinite(value) and 0 <= value <= 1, (key, value)
            assert row["target"] != row["modality"]
            assert row["subject"] not in groups[key], (key, row["subject"])
            groups[key][row["subject"]] = value
            subjects.add(row["subject"])
    assert len(subjects) == args.cases and groups
    assert 10000 in {key[0] for key in groups}
    steps = sorted({key[0] for key in groups})
    assert len(groups) == len(steps) * 4 * 8 * 3
    output = source.with_name("source_weights_checkpoint_summary.csv")
    rows = []
    for (step, target, band, modality), values in sorted(groups.items()):
        assert set(values) == subjects
        weights = list(values.values())
        rows.append({"step": step, "target": target, "subband": band, "modality": modality,
                     "patients": len(weights), "mean_weight": f"{statistics.mean(weights):.8f}",
                     "patient_sample_sd": f"{statistics.stdev(weights):.8f}",
                     "min_weight": f"{min(weights):.8f}", "max_weight": f"{max(weights):.8f}"})
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"input": str(source), "input_bytes": source.stat().st_size,
                      "output": str(output), "output_bytes": output.stat().st_size,
                      "patients": args.cases, "checkpoint_steps": steps, "summary_rows": len(rows)}))


if __name__ == "__main__":
    main()
