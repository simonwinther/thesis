"""Check the shared table against explicit sums, masking, and gradient updates."""

import itertools
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch as th

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from guided_diffusion.source import ALL_MODALITIES, BIASED_PARTNER, StartingSource, load_source, save_source
from stream_metrics import StreamingMetrics, normalized_array
from evaluate import load_nifti, compute_metrics_masked
import nibabel as nib


def main():
    state = th.get_rng_state()
    for init in ("uniform", "random", "biased"):
        source = StartingSource("learned", init, architecture="shared")
        assert sum(p.numel() for p in source.parameters()) == 32
        assert th.equal(state, th.get_rng_state()), "Initialization changed the training RNG"
    source = StartingSource("learned", "random", architecture="shared")
    assert th.equal(source.table_logits, StartingSource("learned", "random", architecture="shared").table_logits)
    assert not th.equal(source.table_logits, StartingSource("learned", "random", seed=43, architecture="shared").table_logits)

    # Distinct values expose transposed bands, modality mixups and summing bands.
    generator = th.Generator().manual_seed(123)
    coefficients = th.randn(2, 4, 8, 3, 3, 3, generator=generator)
    with th.no_grad():
        source.table_logits.copy_(th.arange(32).reshape(8, 4) / 17)
    for target in ALL_MODALITIES:
        names = [m for m in ALL_MODALITIES if m != target]
        ids = [ALL_MODALITIES.index(m) for m in names]
        cond = coefficients[:, ids].flatten(1, 2)
        for mask in itertools.product((False, True), repeat=3):
            if not any(mask):
                continue
            available = th.tensor([mask, mask[::-1]])
            result, weights = source(cond, names, target, available)
            assert result.shape == (2, 8, 3, 3, 3)
            for batch in range(2):
                for band in range(8):
                    raw = [source.table_logits[band, i].exp() if available[batch, j] else th.tensor(0.)
                           for j, i in enumerate(ids)]
                    expected_weights = th.stack(raw) / sum(raw)
                    expected = sum(expected_weights[j] * coefficients[batch, i, band]
                                   for j, i in enumerate(ids))
                    th.testing.assert_close(result[batch, band], expected)
                    th.testing.assert_close(weights[batch, :, band], expected_weights)
            altered = cond * 7 + 23
            assert th.equal(weights, source(altered, names, target, available)[1])

        uniform = StartingSource("learned", architecture="shared")
        available = th.ones(2, 3, dtype=th.bool)
        th.testing.assert_close(uniform(cond, names, target, available)[0], coefficients[:, ids].mean(1))
        biased = StartingSource("learned", "biased", architecture="shared")
        weights = biased(cond, names, target, available)[1]
        for j, name in enumerate(names):
            expected = 1/3 if target == "t1n" else (0.6 if name == "t1n" else 0.2)
            th.testing.assert_close(weights[:, j], th.full((2, 8), expected))
        partner_bias = StartingSource("learned", "biased", architecture="shared", shared_bias="partner")
        assert sum(p.numel() for p in partner_bias.parameters()) == 32
        weights = partner_bias(cond, names, target, available)[1]
        for j, name in enumerate(names):
            expected = 0.6 if name == BIASED_PARTNER[target] else 0.2
            th.testing.assert_close(weights[:, j], th.full((2, 8), expected))

    # Accumulate four different held-out targets into exactly the SAME table.
    trainable = StartingSource("learned", architecture="shared")
    optimizer = th.optim.AdamW(trainable.parameters(), lr=1e-3)
    loss = 0
    for target_idx, target in enumerate(ALL_MODALITIES):
        names = [m for m in ALL_MODALITIES if m != target]
        ids = [ALL_MODALITIES.index(m) for m in names]
        start, _ = trainable(coefficients[:, ids].flatten(1, 2), names, target,
                             th.ones(2, 3, dtype=th.bool))
        target_coeff = coefficients[:, target_idx]
        path = 0.7 * start + 0.3 * target_coeff
        velocity_target = target_coeff - start
        loss = loss + (path.square() - velocity_target).square().mean()
    loss.backward()
    assert th.isfinite(trainable.table_logits.grad).all()
    assert (trainable.table_logits.grad.abs() > 0).all()
    optimizer.step()
    assert (trainable.table_logits != 0).any()
    with tempfile.TemporaryDirectory() as folder:
        checkpoint = Path(folder) / "source.pt"
        save_source(trainable, checkpoint)
        restored = load_source(checkpoint)
        assert restored.config() == trainable.config()
        assert th.equal(restored.table_logits, trainable.table_logits)
        # A real NIfTI round trip must give the same numbers as streaming.
        rng = np.random.default_rng(42)
        pred = rng.uniform(-0.1, 1, (16, 16, 16)).astype(np.float32)
        gt = rng.uniform(0, 1, pred.shape).astype(np.float32)
        gt[:3] = 0
        paths = [Path(folder) / name for name in ("pred.nii.gz", "gt.nii.gz")]
        for path, array in zip(paths, (pred, gt)):
            nib.save(nib.Nifti1Image(array, np.eye(4)), path)
            np.testing.assert_array_equal(load_nifti(path), normalized_array(array))
        expected = compute_metrics_masked(*(load_nifti(p) for p in paths))
        metrics = StreamingMetrics()
        metrics.add("case", "t1n", pred, gt)
        for key, value in expected.items():
            assert metrics.rows["t1n"][0]["masked"][key] == value
        metrics.save(folder)
    if len(sys.argv) > 1:
        old = load_source(sys.argv[1])
        assert old.architecture == "conditional"
        assert "architecture" not in old.config()
    print("PASS: 32 parameters, all targets/subsets, explicit weighted sums, RNG isolation, "
          "uniform/biased/random initialization, shared gradients, checkpoint round trip, "
          "NIfTI/streaming metric equivalence and historical checkpoint loading.")


if __name__ == "__main__":
    main()
