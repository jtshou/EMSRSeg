"""Training losses with the original normalization."""
from functools import reduce
from operator import mul
import torch

def affinity_loss(prediction, target, weight):
    
    norm = prediction.shape[0] * reduce(mul, prediction.shape[2:], 1)
    return (weight * (prediction - target).square()).sum() / norm

def training_loss(outputs, batch, stage, prior_weight=0.1):
    affinity = batch["lb_affs"][:, :3]
    weight = batch["weightmap"][:, :3]
    if stage == 1:
        auxiliary, sr, segmentation = outputs
        seg_aux = affinity_loss(auxiliary, affinity, weight)
        seg_main = affinity_loss(segmentation, affinity, weight)
        
        section = torch.randint(0, sr.shape[2], (1,))
        sr_loss = torch.nn.functional.l1_loss(sr[:, :, section], batch["hr"][:, :, section])
        return seg_aux + seg_main + sr_loss, dict(sr=sr_loss, aux=seg_aux, seg=seg_main)
    segmentation, hr_prior, structural_prior = outputs
    seg_main = affinity_loss(segmentation, affinity, weight)
    prior = torch.nn.functional.mse_loss(structural_prior, hr_prior)
    return seg_main + prior_weight * prior, dict(seg=seg_main, prior=prior)
