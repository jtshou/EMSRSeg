"""Resolve configuration paths using an explicit data root."""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]

def load_config(path, data_root):
    with open(path, encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    data_root = Path(data_root).expanduser().resolve()
    for phase, dataset in cfg["datasets"].items():
        dataset["phase"] = phase
        dataset["scale"] = cfg["scale"]
        for key in ("data_folder_gt", "data_folder_lr"):
            dataset[key] = str(data_root / dataset[key])
    for key, value in cfg["pretrain"].items():
        if isinstance(value, str):
            cfg["pretrain"][key] = str(ROOT / value)
    cfg["checkpoint"] = str(ROOT / cfg["checkpoint"])
    return cfg
