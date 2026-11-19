"""Strict LeWM architecture/weight loader from the research execution.

Only path defaults and argument normalization differ from the original loader;
external source and model directories are explicit. No download/audit driver.
"""

from pathlib import Path
import json
import sys


def load_model(device="cpu", *, source, model_dir):
    source, model_dir = Path(source), Path(model_dir)
    import torch
    from transformers import ViTConfig, ViTModel
    sys.path.insert(0, str(source))
    from jepa import JEPA
    from module import ARPredictor, Embedder, MLP
    if any(Path(sys.modules[name].__file__).resolve().parent != Path(source).resolve()
           for name in ("jepa", "module")):
        raise RuntimeError("A different LeWM source is already loaded in this Python process")
    cfg = json.loads((model_dir / "config.json").read_text())
    enc_cfg = cfg["encoder"]
    if enc_cfg["size"] != "tiny" or enc_cfg["pretrained"]:
        raise ValueError("Only the verified ViT tiny architecture is accepted")
    # Literal tiny branch of stable_pretraining.backbone.utils.vit_hf.
    encoder = ViTModel(ViTConfig(
        hidden_size=192, num_hidden_layers=12, num_attention_heads=3,
        intermediate_size=768, image_size=enc_cfg["image_size"],
        patch_size=enc_cfg["patch_size"]),
        add_pooling_layer=False, use_mask_token=enc_cfg["use_mask_token"])
    encoder.config.interpolate_pos_encoding = True
    kwargs = lambda name: {k: v for k, v in cfg[name].items()
                           if k not in ("_target_", "norm_fn")}
    model = JEPA(encoder=encoder, predictor=ARPredictor(**kwargs("predictor")),
                 action_encoder=Embedder(**kwargs("action_encoder")),
                 projector=MLP(**kwargs("projector"), norm_fn=torch.nn.BatchNorm1d),
                 pred_proj=MLP(**kwargs("pred_proj"), norm_fn=torch.nn.BatchNorm1d))
    state = torch.load(model_dir / "weights.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.eval().to(device)
