"""Prepare aligned HR/label HDF5 volumes and degraded LR images."""
import argparse
import json
import random
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from basicsr.utils.matlab_functions import imresize
from skimage.transform import rescale

SEED = 0


def degrade(image, mode, scale, rng):
    """Preserve the degradation operations and random draw order."""
    if mode == "blur_shot":
        image = image.astype(np.float32) / 255.0
        image = cv2.GaussianBlur(image, (0, 0), sigmaX=1.5, sigmaY=1.5)
        lr = np.clip(rescale(image, 1.0 / scale, order=1, anti_aliasing=False,
                             preserve_range=True), 0, 1).astype(np.float32)
        lr = rng.poisson(lr * 60.0).astype(np.float32) / 60.0
        return np.clip(lr * 255.0, 0, 255).round().astype(np.uint8)
    image = torch.from_numpy(image[None].copy()) / 255.0
    lr = imresize(image, 1 / scale) * 255.0
    if mode == "bicubic_noisy":
        lr = lr + torch.randn_like(lr) * random.uniform(10, 25)
    return lr.round().clamp(0, 255).squeeze(0).numpy().astype(np.uint8)


def read_volume(path, key):
    with h5py.File(path, "r") as stream:
        volume = stream[key][:]
    if volume.ndim != 3:
        raise ValueError("Input must have shape [Z,H,W]")
    return volume


def write_volume(path, volume):
    """Save the main dataset without overwriting existing files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "x") as stream:
        stream.create_dataset("main", data=volume)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="HR intensity HDF5")
    parser.add_argument("--input-key", default="main")
    parser.add_argument("--output", required=True, help="LR HDF5")
    parser.add_argument("--labels", help="可选的源标签 HDF5")
    parser.add_argument("--labels-key", default="main")
    parser.add_argument("--labels-output", help="裁剪后标签 HDF5")
    parser.add_argument("--hr-output", help="裁剪后 HR HDF5")
    parser.add_argument("--mode", choices=["bicubic", "bicubic_noisy", "blur_shot"],
                        required=True)
    parser.add_argument("--scale", type=int, choices=[4, 8, 16], required=True)
    parser.add_argument("--crop", type=int, default=1024)
    args = parser.parse_args()
    if bool(args.labels) != bool(args.labels_output):
        parser.error("--labels and --labels-output must be supplied together")
    if args.mode == "blur_shot" and args.scale != 4:
        raise ValueError("Published blur/shot setting uses x4")
    if args.mode == "bicubic_noisy" and args.scale != 8:
        raise ValueError("Published noisy CREMI-B setting uses x8")
    if args.crop <= 0 or args.crop % args.scale:
        raise ValueError("Crop size must be positive and divisible by scale")

    output = Path(args.output).resolve()
    metadata = output.with_suffix(".json")
    targets = [output, metadata]
    targets += [Path(p).resolve() for p in (args.hr_output, args.labels_output) if p]
    sources = [Path(p).resolve() for p in (args.input, args.labels) if p]
    if len(set(targets)) != len(targets) or set(targets) & set(sources):
        raise ValueError("Input and output paths must be distinct")
    for path in targets:
        if path.exists():
            raise FileExistsError(path)

    volume = read_volume(args.input, args.input_key)
    if (volume.dtype.kind not in "uif" or not np.isfinite(volume).all()
            or np.any(volume < 0) or np.any(volume > 255)):
        raise ValueError("HR intensities must be finite numeric values in [0,255]")
    volume = volume.astype(np.uint8)
    if min(volume.shape[1:]) < args.crop:
        raise ValueError("Input is smaller than the requested crop")
    labels = read_volume(args.labels, args.labels_key) if args.labels else None
    if labels is not None:
        if labels.shape != volume.shape:
            raise ValueError("HR images and instance labels must have identical shapes")
        if not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0):
            raise ValueError("Instance labels must be nonnegative integers")
    y = (volume.shape[1] - args.crop) // 2
    x = (volume.shape[2] - args.crop) // 2
    volume = volume[:, y:y + args.crop, x:x + args.crop]
    if labels is not None:
        labels = labels[:, y:y + args.crop, x:x + args.crop]

    random.seed(SEED)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    lr = np.stack([degrade(image, args.mode, args.scale, rng) for image in volume])
    if args.hr_output:
        write_volume(Path(args.hr_output), volume)
    if args.labels_output:
        write_volume(Path(args.labels_output), labels)
    write_volume(output, lr)
    with metadata.open("x", encoding="utf-8") as stream:
        json.dump(vars(args), stream, indent=2)


if __name__ == "__main__":
    main()
