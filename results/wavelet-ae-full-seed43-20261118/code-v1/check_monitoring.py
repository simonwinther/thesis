"""Check split isolation, loss scaling and paired TensorBoard curves on CPU."""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

from data import balanced_order, records
from evaluate import coefficient_metrics
from model import build_model
from monitor import check_monitor_split, coefficient_mean
from train import validate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    manifest = json.loads(args.manifest.read_text())
    check_monitor_split(manifest)
    assert len(records(manifest, "train")) == 5004
    assert all(len(records(manifest, s)) == 8 for s in ("monitor_train", "monitor_validation"))
    for epoch in (0, 1):
        order = balanced_order(1251, 43, epoch)
        assert sorted(order) == list(range(5004))
        assert all({i % 4 for i in order[j:j+4]} == {0,1,2,3} for j in range(0,5004,4))
        assert order[2500:] == balanced_order(1251, 43, epoch)[2500:]
    broken = {**manifest, "validation": [manifest["train"][0], *manifest["validation"]]}
    try:
        check_monitor_split(broken)
    except ValueError:
        pass
    else:
        raise AssertionError("Patient leakage was accepted")
    rng = np.random.default_rng(43)
    for band, channels in (("lf", 1), ("hf", 7)):
        target = rng.normal(size=(channels,16,16,16)).astype(np.float32)
        prediction = target + .125
        rows = coefficient_metrics(prediction, target, np.ones((16,16,16), dtype=bool), band)
        np.testing.assert_allclose(coefficient_mean(rows,band), np.mean((prediction-target)**2), rtol=1e-6)
        model = build_model(band, 8, (128,256,256)).train()
        before = {k:v.clone() for k,v in model.state_dict().items()}
        state = torch.get_rng_state().clone()
        result = validate(model, [{"coefficients":torch.from_numpy(target),
                                  "interior":torch.ones(16,16,16,dtype=torch.bool),
                                  "case":"test", "modality":"t1n"}], "cpu")
        assert model.training and result and torch.equal(state,torch.get_rng_state())
        assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
        assert all(p.grad is None for p in model.parameters())
    with tempfile.TemporaryDirectory() as tmp:
        writer = SummaryWriter(tmp)
        writer.add_scalars("coefficient_MSE", {"Training (fixed 8)":.1,"Validation (fixed 8)":.2}, 1000)
        writer.close()
        events = []
        for path in Path(tmp).rglob("events.out.tfevents.*"):
            accumulator = EventAccumulator(str(path)).Reload()
            if "coefficient_MSE" in accumulator.Tags()["scalars"]:
                event = accumulator.Scalars("coefficient_MSE")[0]
                assert event.step == 1000
                events.append(round(event.value,6))
        assert sorted(events) == [.1,.2]
    print("PASS: 5004 records; 8+8 balanced disjoint scans; modality-balanced epochs; MSE scaling; eval leaves weights/RNG untouched; TensorBoard overlays")


if __name__ == "__main__":
    main()
