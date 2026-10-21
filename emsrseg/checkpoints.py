"""Load historical checkpoints without silently dropping active parameters."""
from pathlib import Path
import hashlib
import torch

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def read_state(path):
    
    state = torch.load(Path(path), map_location="cpu")
    for key in ("state_dict", "model_weights", "model"):
        if isinstance(state, dict) and isinstance(state.get(key), dict):
            state = state[key]
            break
    normalized = {}
    for key, value in state.items():
        name = ".".join(part for part in key.split(".") if part != "module")
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Not a model tensor: {key}")
        if name in normalized and not torch.equal(value, normalized[name]):
            raise ValueError(f"Conflicting checkpoint aliases: {name}")
        normalized[name] = value
    return normalized

def load_checkpoint(model, path):
    state = read_state(path)
    
    for key, value in state.items():
        alias = key.replace("diffusion_model.condition.", "diff_encoder.", 1)
        alias = alias.replace("diffusion_model.model.", "diff_backbone.", 1)
        if alias != key and alias in state and not torch.equal(value, state[alias]):
            raise ValueError(f"Conflicting shared weights: {key} / {alias}")
    expected = model.state_dict()
    unused = sorted(set(state) - set(expected))
    
    
    if unused and model.__class__.__name__ in ("EMSRSegCNNStage2", "EMSRSegTransformerStage2") and hasattr(model, "diffusion_model") and model.upscale_rate == 8:
        channels = 28 if model.__class__.__name__ == "EMSRSegTransformerStage2" else 64
        widths = [(channels,4*channels)] * 3 + [(channels,channels),(channels,3)]
        allowed = {}
        for index, (incoming, outgoing) in enumerate(widths, 1):
            allowed[f"out_conv{index}.weight"] = (outgoing,incoming,3,3)
            allowed[f"out_conv{index}.bias"] = (outgoing,)
        if set(unused) != set(allowed):
            raise ValueError(f"Unrecognized historical auxiliary keys: {unused}")
        for key in unused:
            if tuple(state[key].shape) != allowed[key]:
                raise ValueError(f"Historical auxiliary shape mismatch: {key}")
            del state[key]
    elif unused:
        raise ValueError(f"Unexpected checkpoint keys: {unused}")
    model.load_state_dict(state, strict=True)
    return {"path": str(path), "sha256": sha256(path), "loaded_tensors": len(state),
            "unused_legacy_auxiliary_keys": unused}

def initialize_stage2(model, stage1_model, checkpoint):
    """Validate Stage I weights, copy shared parameters, and freeze SR/HR-prior modules."""
    load_checkpoint(stage1_model, checkpoint)
    source = stage1_model.state_dict()
    current = model.state_dict()
    shared = set(source) & set(current)
    for key in shared:
        if source[key].shape != current[key].shape:
            raise ValueError(f"Stage transition shape mismatch: {key}")
        current[key] = source[key]
    model.load_state_dict(current, strict=True)
    frozen_names = {
        key for key in shared
        if not any(name in key for name in ("seg_head", "seg_emb", "prior_fusion_1"))
    }
    
    frozen_modules = sorted({key.rsplit(".", 1)[0] for key in frozen_names})
    for name in frozen_modules:
        for param in model.get_submodule(name).parameters():
            param.requires_grad_(False)
    return frozen_modules

def set_training_mode(model, frozen_modules):
    model.train()
    for name in frozen_modules:
        model.get_submodule(name).eval()
