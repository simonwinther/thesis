"""BraTS preprocessing and fixed Haar transforms, compatible with WFM."""

import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pywt
import torch
from torch.utils.data import Dataset

MODALITIES = ("t1n", "t1c", "t2w", "t2f")
BANDS = ("LLL", "LLH", "LHL", "LHH", "HLL", "HLH", "HHL", "HHH")
KEYS = ("aaa", "aad", "ada", "add", "daa", "dad", "dda", "ddd")
IMAGE_SHAPE = (224, 224, 160)
LATENT_SPATIAL_SHAPE = (28, 28, 20)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_scan(path):
    # Match the baseline's float64 normalization followed by float32 storage.
    image = nib.load(str(path)).get_fdata()
    if image.shape != (240, 240, 155) or not np.isfinite(image).all():
        raise ValueError(f"Invalid BraTS image: {path}, shape={image.shape}")
    image = np.clip(image, np.quantile(image, 0.001), np.quantile(image, 0.999))
    low, high = image.min(), image.max()
    image = (image - low) / (high - low) if high > low else np.zeros_like(image)
    result = np.zeros(IMAGE_SHAPE, dtype=np.float32)
    result[:, :, :155] = image[8:-8, 8:-8, :]
    return result


def wavelet_encode(image):
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 3 or any(n % 2 for n in image.shape):
        raise ValueError("Haar input must have three even spatial dimensions")
    coefficients = pywt.dwtn(image, "haar", mode="periodization", axes=(0, 1, 2))
    result = np.stack([coefficients[key] for key in KEYS]).astype(np.float32)
    result[0] /= np.float32(3)
    return result


def wavelet_decode(coefficients):
    if coefficients.ndim != 4 or coefficients.shape[0] != 8:
        raise ValueError("Inverse Haar expects eight ordered coefficient channels")
    values = {key: np.asarray(coefficients[i], dtype=np.float32)
              for i, key in enumerate(KEYS)}
    values["aaa"] = values["aaa"] * np.float32(3)
    return pywt.idwtn(values, "haar", mode="periodization", axes=(0, 1, 2)).astype(np.float32)


def interior_mask(image):
    x, y, z = image.shape
    return (image > 0).reshape(x // 2, 2, y // 2, 2, z // 2, 2).all(axis=(1, 3, 5))


def band_slice(band):
    if band not in ("lf", "hf"):
        raise ValueError(band)
    return slice(0, 1) if band == "lf" else slice(1, 8)


def discover(root):
    root = Path(root).expanduser().resolve()
    cases = []
    for first in sorted(root.rglob("*-t1n.nii.gz")):
        case = first.name.removesuffix("-t1n.nii.gz")
        paths = {m: str(first.parent / f"{case}-{m}.nii.gz") for m in MODALITIES}
        if not all(Path(p).is_file() for p in paths.values()):
            raise ValueError(f"Incomplete case: {case}")
        cases.append({"case": case, "patient": case.rsplit("-", 1)[0], "paths": paths})
    if len({c["case"] for c in cases}) != len(cases):
        raise ValueError(f"Duplicate case IDs under {root}")
    return cases


def distinct_cases(cases, count, seed):
    chosen, seen = [], set()
    for i in np.random.default_rng(seed).permutation(len(cases)):
        case = cases[i]
        if case["patient"] not in seen:
            chosen.append(case)
            seen.add(case["patient"])
        if len(chosen) == count:
            return chosen
    raise ValueError("Not enough distinct patients")


def make_manifest(train_root, val_root, seed=43):
    train, val = discover(train_root), discover(val_root)
    if (len(train), len(val)) != (1251, 219):
        raise ValueError(f"Expected 1251/219 cases, got {len(train)}/{len(val)}")
    if {c["patient"] for c in train} & {c["patient"] for c in val}:
        raise ValueError("Training/validation patient identities overlap")
    debug = [{"case": c["case"], "modality": m}
             for c, m in zip(distinct_cases(train, 4, seed), MODALITIES)]
    return {"seed": seed, "train": train, "validation": val,
            "development": [c["case"] for c in distinct_cases(val, 8, seed)],
            "debug": debug, "modalities": MODALITIES,
            "preprocessing": "percentiles0.1/99.9; [0,1]; pad155->160; xy8:-8",
            "wavelet": "haar; LLL/3; " + ",".join(BANDS)}


def records(manifest, subset):
    if subset in ("monitor_train", "monitor_validation"):
        split = "train" if subset == "monitor_train" else "validation"
        cases = {c["case"]: c for c in manifest[split]}
        return [(cases[d["case"]], d["modality"]) for d in manifest[subset]]
    if subset == "debug":
        cases = {c["case"]: c for c in manifest["train"]}
        return [(cases[d["case"]], d["modality"]) for d in manifest["debug"]]
    cases = manifest["train" if subset == "train" else "validation"]
    if subset == "development":
        cases = [c for c in cases if c["case"] in manifest["development"]]
    elif subset not in ("train", "validation"):
        raise ValueError(subset)
    return [(c, m) for c in cases for m in MODALITIES]


class WaveletDataset(Dataset):
    def __init__(self, manifest, subset, band):
        self.items = records(manifest, subset)
        self.selection = band_slice(band)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        case, modality = self.items[index]
        image = load_scan(case["paths"][modality])
        return {"coefficients": torch.from_numpy(wavelet_encode(image)[self.selection].copy()),
                "interior": torch.from_numpy(interior_mask(image)),
                "case": case["case"], "modality": modality}


def balanced_order(case_count, seed, epoch):
    # Exactly one of each modality per four examples, each case/modality once/epoch.
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
    columns = np.stack([4 * rng.permutation(case_count) + m for m in range(4)], axis=1)
    return np.concatenate([row[rng.permutation(4)] for row in columns]).tolist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--val-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    write_json(args.output, make_manifest(args.train_root, args.val_root))
    print(f"Verified 1251 train / 219 validation cases; manifest: {args.output}")
