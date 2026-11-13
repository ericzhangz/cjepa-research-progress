"""Fetch pinned public assets; load official LeWM state_dict without pickle fallback.

Run with the existing loco Python. No dependencies are installed.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import urllib.request

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "lewm"
MODEL_REV = "22b330c28c27ead4bfd1888615af1340e3fe9052"
DATA_REV = "7628202a2180972f291ba1bc6723834921e72c19"
SOURCE_REV = "8edfeb336732b5f3ce7b8b210d0ba370a09e2cac"


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url, destination, expected=None):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        partial = destination.with_suffix(destination.suffix + ".partial")
        with urllib.request.urlopen(url, timeout=90) as src, partial.open("wb") as dst:
            while True:
                block = src.read(1024 * 1024)
                if not block:
                    break
                dst.write(block)
        partial.replace(destination)
    digest = sha256(destination)
    if expected and digest != expected:
        raise RuntimeError(f"SHA256 mismatch for {destination}")
    return {"url": url, "path": str(destination), "bytes": destination.stat().st_size,
            "sha256": digest}


def load_model(device="cpu"):
    import torch
    from transformers import ViTConfig, ViTModel
    sys.path.insert(0, str(SOURCE))
    from jepa import JEPA
    from module import ARPredictor, Embedder, MLP
    cfg = json.loads((ROOT / "model/config.json").read_text())
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
    state = torch.load(ROOT / "model/weights.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.eval().to(device)


def main():
    entries = []
    model_base = f"https://huggingface.co/quentinll/lewm-pusht/resolve/{MODEL_REV}/"
    entries.append(fetch(model_base + "config.json", ROOT / "model/config.json"))
    entries.append(fetch(model_base + "weights.pt", ROOT / "model/weights.pt",
                         "48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607"))
    data_base = f"https://huggingface.co/datasets/lerobot/pusht/resolve/{DATA_REV}/"
    data_files = ["meta/info.json", "meta/stats.json", "README.md",
                  "data/chunk-000/file-000.parquet",
                  "meta/episodes/chunk-000/file-000.parquet",
                  "videos/observation.image/chunk-000/file-000.mp4"]
    for name in data_files:
        entries.append(fetch(data_base + name, ROOT / "lerobot_pusht" / name))
    source_base = f"https://raw.githubusercontent.com/lucas-maes/le-wm/{SOURCE_REV}/"
    for name in ["train.py", "config/train/data/pusht.yaml", "config/train/model/lewm.yaml"]:
        entries.append(fetch(source_base + name, ROOT / "reference" / name))
    import torch
    torch.set_num_threads(1)
    model = load_model()
    with torch.inference_mode():
        encoded = model.encode({"pixels": torch.zeros(1, 4, 3, 224, 224),
                                "action": torch.zeros(1, 4, 10)})
        pred = model.predict(encoded["emb"][:, :3], encoded["act_emb"][:, :3])
    if pred.shape != (1, 3, 192) or not torch.isfinite(pred).all():
        raise RuntimeError("Native forward validation failed")
    import cv2
    capture = cv2.VideoCapture(str(ROOT / "lerobot_pusht/videos/observation.image/chunk-000/file-000.mp4"))
    ok, frame = capture.read()
    video_metadata = {"decoder_open": capture.isOpened(), "first_frame_decoded": ok,
                      "frame_count": capture.get(cv2.CAP_PROP_FRAME_COUNT),
                      "fps": capture.get(cv2.CAP_PROP_FPS),
                      "shape": list(frame.shape) if ok else None}
    capture.release()
    manifest = {"source_commit": SOURCE_REV, "model_revision": MODEL_REV,
                "dataset_revision": DATA_REV, "files": entries,
                "downloaded_asset_bytes": sum(x["bytes"] for x in entries),
                "strict_state_dict_load": True, "weights_only": True,
                "parameters": sum(p.numel() for p in model.parameters()),
                "torch_version": str(torch.__version__),
                "prediction_shape": list(pred.shape), "video": video_metadata,
                "scope": "Shape/load check only. LeRobot data are transfer diagnostics, not LeWM paper reproduction. Encoder pretraining overlap is unknown."}
    (ROOT / "ASSET_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
