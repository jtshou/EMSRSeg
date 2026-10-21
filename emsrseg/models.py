"""Model factory preserving registered parameter names."""
from importlib import import_module

def build_model(config, stage=None):
    stage = int(stage or config["stage"])
    backbone = config["backbone"]
    scale = config["scale"]
    if stage not in (1, 2):
        raise ValueError("Stage must be 1 or 2")
    if backbone == "cnn":
        if scale not in (4, 8):
            raise ValueError("Published CNN settings use x4 or x8")
        suffix = "_x4" if scale == 4 else ""
        module = import_module(f"model.emsrseg_cnn_stage{stage}{suffix}")
        cls = getattr(module, f"EMSRSegCNNStage{stage}")
    elif backbone == "transformer":
        if scale not in (8, 16):
            raise ValueError("Published Transformer settings use x8 or x16")
        module = import_module(f"model.emsrseg_transformer_stage{stage}")
        cls = getattr(module, f"EMSRSegTransformerStage{stage}")
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    kwargs = dict(upscale_rate=scale, w_att=config["network"]["w_att"])
    if stage == 2:
        kwargs["timestep_l"] = config["network"]["timestep_l"]
    return cls(**kwargs)
