"""Evaluate the *same* saved Phase 1 futures with an independently calibrated XY readout.

Extract uses the base HDF5/Blosc environment. Encode and score use the LeWM
environment. The fixed calibration split is from the author's training archive.
"""
import argparse
import ctypes
import ctypes.util
import os
import json
from pathlib import Path
import shutil
import tarfile
import tempfile

import numpy as np
from .prbd_paths import ROOT, output_path


OUT = ROOT / "runs/prbd_phase1_20261121"



def extract():
    import h5py
    import zstandard
    library = os.environ.get("PRBD_BLOSC_LIBRARY") or ctypes.util.find_library("blosc")
    if not library:
        raise OSError("Set PRBD_BLOSC_LIBRARY to the local Blosc shared library")
    lib = ctypes.CDLL(library)
    lib.blosc_decompress.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t)
    lib.blosc_decompress.restype = ctypes.c_int
    with tempfile.TemporaryDirectory() as temp:
        target = Path(temp) / "tworoom.h5"
        with Path(os.environ["PRBD_TWOROOM_TRAIN_ARCHIVE"]).open("rb") as archive:
            with zstandard.ZstdDecompressor().stream_reader(archive) as reader:
                with tarfile.open(fileobj=reader, mode="r|") as tar:
                    member = tar.next()
                    if member.name != target.name or not member.isfile():
                        raise ValueError("Expected official TwoRoom HDF5 file")
                    with tar.extractfile(member) as src, target.open("wb") as dst:
                        shutil.copyfileobj(src, dst, length=8 << 20)
        with h5py.File(target, "r") as data:
            starts = data["ep_offset"][:2304].reshape(-1)
            lengths = data["ep_len"][:2304].reshape(-1)
            if len(starts) != 2304 or (lengths < 1).any():
                raise ValueError("Expected 2,304 official calibration/check episodes")
            # Four fixed phases per episode, no target/evaluation split consulted.
            rows = (starts[:, None] + lengths[:, None] * np.arange(4) // 4).ravel().astype('int64')
            xy = np.concatenate([data["proprio"][int(row):int(row)+1] for row in rows])
            shape = data["pixels"].chunks
            if shape != (100, 224, 224, 3):
                raise ValueError("Unexpected official pixel storage")
            decoded = ctypes.create_string_buffer(int(np.prod(shape)))
            pixels = np.frombuffer(decoded, dtype=np.uint8).reshape(shape)
            output = np.lib.format.open_memmap(output_path(OUT / "official_xy_rgb.npy"), mode="w+", dtype="uint8",
                                               shape=(len(rows), *shape[1:]))
            for chunk in np.unique(rows // 100):
                mask, value = data["pixels"].id.read_direct_chunk((int(chunk * 100), 0, 0, 0))
                if mask or lib.blosc_decompress(value, decoded, len(decoded)) != len(decoded):
                    raise ValueError("Unexpected filtered RGB chunk")
                selected = np.flatnonzero((rows // 100) == chunk)
                output[selected] = pixels[rows[selected] % 100]
            output.flush()
            np.savez(output_path(OUT / "official_xy_labels.npz"), xy=xy, rows=rows, episodes=np.repeat(np.arange(2304),4))
    print("saved 9,216 published calibration/check frames", flush=True)


def encode(device):
    import torch
    from .prbd_tworoom_speed_fit import load_host
    torch.set_num_threads(3)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    host = load_host(device).eval()
    rgb = np.load(output_path(OUT / "official_xy_rgb.npy"), mmap_mode="r")
    labels = np.load(output_path(OUT / "official_xy_labels.npz"))
    xy = torch.tensor(labels["xy"], dtype=torch.float64)
    mean = torch.tensor([.485,.456,.406],device=device)[None,:,None,None]
    std = torch.tensor([.229,.224,.225],device=device)[None,:,None,None]
    z = []
    with torch.inference_mode():
        for begin in range(0, len(rgb), 32):
            img = torch.from_numpy(rgb[begin:begin+32].copy()).to(device).permute(0,3,1,2).float()/255
            z.append(host.encode({"pixels": ((img-mean)/std)[:,None]})["emb"][:,0].cpu())
    z = torch.cat(z).double()
    design = torch.cat((z, torch.ones(len(z),1,dtype=z.dtype)), -1)
    fit = torch.from_numpy(labels["episodes"] < 2048)
    solution = torch.linalg.lstsq(design[fit], xy[fit], driver="gelsd")
    coeff = solution.solution
    error = (design @ coeff - xy).square()
    report = {"fit_episodes":2048,"check_episodes":256,"frames_per_episode":4,
              "rank":int(solution.rank),"fit_coordinate_rmse":float(error[fit].mean().sqrt()),
              "check_coordinate_rmse":float(error[~fit].mean().sqrt()),
              "labels":"proprio; four fixed time quartiles; unchanged OLS + intercept; no capacity labels or dev/fresh positions used"}
    torch.save({"coef":coeff, "z":z.float(), "report":report,
                "record_namespace":"author_tworoom_training_hdf5",
                "training_episode_ids":list(range(2048)), "training_frames":int(fit.sum())},
               output_path(OUT / "official_xy_readout.pt"))
    output_path(OUT / "official_xy_readout.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(report,flush=True)


def score_file(checkpoint):
    from .prbd_capacity_evaluate import rescore_readout
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(path)
    return rescore_readout(path, OUT / "official_xy_readout.pt",
                           OUT / (path.stem + "_official_xy.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("extract","encode","score"))
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--records",nargs="+")
    args = parser.parse_args()
    if args.stage == "extract":
        extract()
    elif args.stage == "encode":
        encode(args.device)
    else:
        if not args.records:
            parser.error("score requires --records from the saved capacity runs")
        for record in args.records:
            score_file(record)
