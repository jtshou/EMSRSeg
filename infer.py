"""Full-volume SR and segmentation inference."""
import argparse
import json
from pathlib import Path
import cv2
import h5py
import torch
from emsrseg.config import load_config
from emsrseg.models import build_model
from emsrseg.checkpoints import load_checkpoint
from emsrseg.evaluation import seed_everything, infer_volume, sr_metrics, segment_and_score, to_uint8, color_labels
from data.provider_test_3d import Provider_valid

SEED = 1001

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--volume", choices=["all","wafer4","wafer36_2"], default="all")
    parser.add_argument("--save-affinities", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config, args.data_root)
    if cfg["stage"] != 2:
        raise ValueError("Use a Stage II config: inference must not depend on HR priors.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA with MMCV deformable convolution is required.")
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    seed_everything(SEED)
    model = build_model(cfg)
    checkpoint = load_checkpoint(model, args.checkpoint or cfg["checkpoint"])
    model.cuda().eval()
    torch.backends.cudnn.benchmark = False
    ds = cfg["datasets"]["val"]
    if ds["dataset_name"] == "wafer":
        volumes = [(0,"wafer4"),(1,"wafer36_2")]
        if args.volume != "all":
            volumes = [(i,n) for i,n in volumes if n == args.volume]
    else:
        if args.volume != "all":
            raise ValueError("Wafer volume selector is not valid for CREMI.")
        volumes = [(0,ds["dataset_name"])]
    all_scores = []
    for index, name in volumes:
        dataset = Provider_valid(ds, cfg["scale"], index)
        seed_everything(SEED)
        affinities, sr = infer_volume(model, dataset, "cuda")
        labels = dataset.get_gt_lb().astype("uint32")
        segmentation, scores = segment_and_score(affinities, labels)
        scores.update(sr_metrics(sr, dataset.get_hr_data(), quantization="truncate"))
        scores.update(volume=name, checkpoint=checkpoint)
        directory = root/name
        directory.mkdir(parents=True)
        for folder in ("sr","seg"):
            (directory/folder).mkdir()
        offset = 100 if name.startswith("cremi") else 0
        sr_u8 = to_uint8(sr, quantization="truncate")
        display_seg = segmentation.copy()
        display_seg[labels == 0] = 0
        colors = color_labels(display_seg)
        gt_colors = color_labels(labels)
        for z in range(len(sr_u8)):
            cv2.imwrite(str(directory/"sr"/f"{z+offset:03d}.tif"), sr_u8[z])
            pair = __import__("numpy").concatenate((colors[z],gt_colors[z]),axis=1)
            cv2.imwrite(str(directory/"seg"/f"{z+offset:03d}.png"), pair)
        with h5py.File(directory/"segmentation.h5", "x") as stream:
            stream.create_dataset("main", data=segmentation, compression="gzip")
        if args.save_affinities:
            with h5py.File(directory/"affinities.h5", "x") as stream:
                stream.create_dataset("main", data=affinities, compression="gzip")
        (directory/"scores.json").write_text(json.dumps(scores,indent=2),encoding="utf-8")
        all_scores.append(scores)
        print(json.dumps(scores), flush=True)
    (root/"scores.json").write_text(json.dumps(all_scores,indent=2),encoding="utf-8")

if __name__ == "__main__":
    main()
