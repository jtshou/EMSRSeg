"""Compute metrics from saved SR images and instance labels."""
import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
from basicsr.metrics import calculate_psnr, calculate_ssim
from skimage.metrics import adapted_rand_error, variation_of_information


def read_volume(path, key):
    """Read a [Z,H,W] HDF5 volume without changing label precision."""
    with h5py.File(path, "r") as stream:
        volume = stream[key][:]
    if volume.ndim != 3:
        raise ValueError("Expected a [Z,H,W] HDF5 volume: " + str(path))
    return volume


def select_ground_truth(volume, start, count, crop):
    """Select sections and apply a spatial center crop."""
    stop = len(volume) if count is None else start + count
    if start < 0 or stop > len(volume) or stop <= start:
        raise ValueError("Invalid ground-truth section range")
    volume = volume[start:stop]
    if crop:
        if crop < 0 or min(volume.shape[1:]) < crop:
            raise ValueError("Ground truth is smaller than the requested crop")
        y = (volume.shape[1] - crop) // 2
        x = (volume.shape[2] - crop) // 2
        volume = volume[:, y:y + crop, x:x + crop]
    return volume


def read_sr(directory):
    """Read grayscale images in section order without requantization."""
    paths = [p for p in Path(directory).iterdir()
             if p.is_file() and p.suffix.lower() in (".tif", ".tiff", ".png")]
    paths.sort(key=lambda p: (0, int(p.stem)) if p.stem.isdigit()
               else (1, p.name))
    if not paths:
        raise ValueError("No SR TIFF/PNG sections found")
    images = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None or image.ndim != 2 or image.dtype != np.uint8:
            raise ValueError("Expected a grayscale uint8 SR section: " + str(path))
        images.append(image)
    return np.stack(images)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sr-dir", help="已保存的灰度 SR TIFF/PNG 目录")
    parser.add_argument("--hr", help="HR intensity HDF5")
    parser.add_argument("--segmentation", help="整数实例分割 HDF5")
    parser.add_argument("--labels", help="GT 实例标签 HDF5")
    parser.add_argument("--hr-key", default="main")
    parser.add_argument("--labels-key", default="main")
    parser.add_argument("--segmentation-key", default="main")
    parser.add_argument("--z-start", type=int, default=0)
    parser.add_argument("--z-count", type=int)
    parser.add_argument("--crop", type=int, default=1024, help="中心裁剪大小；0 表示不裁剪")
    parser.add_argument("--output", help="可选的新 JSON 文件，默认仅打印")
    args = parser.parse_args()
    if bool(args.sr_dir) != bool(args.hr):
        parser.error("--sr-dir and --hr must be supplied together")
    if bool(args.segmentation) != bool(args.labels):
        parser.error("--segmentation and --labels must be supplied together")
    if not args.sr_dir and not args.segmentation:
        parser.error("Supply SR/HR or segmentation/labels, or both")
    output = Path(args.output) if args.output else None
    if output is not None and output.exists():
        raise FileExistsError(output)

    scores = {}
    if args.sr_dir:
        sr = read_sr(args.sr_dir)
        hr = select_ground_truth(read_volume(args.hr, args.hr_key),
                                 args.z_start, args.z_count, args.crop)
        if hr.dtype != np.uint8 or sr.shape != hr.shape:
            raise ValueError("SR and HR must have matching [Z,H,W] shapes and uint8 intensities")
        scores["psnr"] = float(np.mean([
            calculate_psnr(x, y, crop_border=0) for x, y in zip(sr, hr)]))
        scores["ssim"] = float(np.mean([
            calculate_ssim(x, y, crop_border=0) for x, y in zip(sr, hr)]))
    if args.segmentation:
        prediction = read_volume(args.segmentation, args.segmentation_key)
        labels = select_ground_truth(read_volume(args.labels, args.labels_key),
                                     args.z_start, args.z_count, args.crop)
        if prediction.shape != labels.shape:
            raise ValueError("Prediction and GT labels must have identical shapes")
        for name, volume in (("prediction", prediction), ("GT", labels)):
            if not np.issubdtype(volume.dtype, np.integer) or np.any(volume < 0):
                raise ValueError(name + " labels must be nonnegative integers")
        split, merge = variation_of_information(labels, prediction, ignore_labels=(0,))
        scores.update(voi_split=float(split), voi_merge=float(merge),
                      voi=float(split + merge),
                      arand=float(adapted_rand_error(labels, prediction,
                                                     ignore_labels=(0,))[0]))
    text = json.dumps(scores, indent=2)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            stream.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
