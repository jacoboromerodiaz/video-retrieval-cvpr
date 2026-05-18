"""Load VJEPA-2.1 models and processor"""

import torch


def load_model(mode: str = "prod"):
    if mode == "dev":
        name = "vjepa2_1_vit_base_384"
        device = "cpu"
    else:
        name = "vjepa2_1_vit_gigantic_384"
        device = "cuda"

    model = torch.hub.load("facebookresearch/vjepa2", name, trust_repo=True)
    return model[0].eval().to(device), device


def load_processor():
    processor = torch.hub.load(
        "facebookresearch/vjepa2",
        "vjepa2_preprocessor",
        crop_size=384,
    )
    return processor
