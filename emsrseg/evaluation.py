"""BasicSR metrics and Waterz segmentation evaluation."""
import random
import numpy as np
import torch

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def to_uint8(volume, quantization="round"):
    values = np.clip(np.asarray(volume) * 255.0, 0, 255)
    if quantization == "round":
        values = values.round()
    elif quantization != "truncate":
        raise ValueError("quantization must be round or truncate")
    return values.astype(np.uint8)

def sr_metrics(sr, hr, quantization="round"):
    from basicsr.metrics import calculate_psnr, calculate_ssim
    a, b = to_uint8(sr, quantization), to_uint8(hr)
    if a.shape != b.shape:
        raise ValueError(f"SR/HR shape mismatch: {a.shape} vs {b.shape}")
    return {
        "psnr": float(np.mean([calculate_psnr(x, y, crop_border=0) for x, y in zip(a,b)])),
        "ssim": float(np.mean([calculate_ssim(x, y, crop_border=0) for x, y in zip(a,b)])),
        "sr_protocol": "BasicSR; SR " + quantization + " uint8; GT round uint8; crop_border=0; slice mean",
    }

def segment_and_score(affinities, labels, threshold=0.5):
    import waterz
    from third_party.fragment import watershed, relabel
    from skimage.metrics import adapted_rand_error, variation_of_information
    fragments = watershed(affinities, "maxima_distance", use_new=False)
    scoring = "OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256>>"
    segmentation = list(waterz.agglomerate(
        affinities, [threshold], fragments=fragments,
        scoring_function=scoring, discretize_queue=256))[0]
    segmentation = relabel(segmentation).astype(np.uint64)
    split, merge = variation_of_information(labels, segmentation, ignore_labels=(0,))
    arand = adapted_rand_error(labels, segmentation, ignore_labels=(0,))[0]
    return segmentation, dict(voi_split=float(split), voi_merge=float(merge),
                              voi=float(split + merge), arand=float(arand),
                              waterz_threshold=threshold)

@torch.no_grad()
def infer_volume(model, dataset, device, stage=2):
    dataset.reset_output()
    
    for sample in torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0):
        lr = sample["lr"].to(device)
        if stage == 1:
            _, sr, affinity = model(lr, sample["hr"].to(device))
        else:
            sr, affinity = model(lr)
        dataset.add_vol(affinity[0].cpu().numpy(), sr[0,0].clamp(0,1).cpu().numpy())
    return dataset.get_results()

def color_labels(labels, seed=0):
    ids, inverse = np.unique(labels, return_inverse=True)
    colors = np.random.default_rng(seed).integers(0, 255, (len(ids),3), dtype=np.uint8)
    colors[ids == 0] = 0
    return colors[inverse].reshape(*labels.shape, 3)
