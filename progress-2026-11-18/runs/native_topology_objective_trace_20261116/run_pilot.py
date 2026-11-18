"""One matched continuation, reusing the current Arm 2 fitting path."""
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.p1a.native_topology_condition import load_readout, RUN
from research.p1a.native_topology_prefix_fit import read_live_panels, fit


def main():
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    source = ROOT / "runs/native_topology_accumulated_20261116/corrected/round3"
    factor = torch.load(source.parents[1] / "calibration/xy_cholesky.pt",
                        map_location=device, weights_only=True)
    decode = load_readout(device, RUN)
    for name in ("control", "factual"):
        output = Path(__file__).parent / "pilot" / name
        output.mkdir(parents=True, exist_ok=True)
        main_panel, sources, live = read_live_panels(source / "data_train_live.npz", device,
                                                    accumulated=True, factual_targets=name == "factual")
        fit("arm2", output, main_panel, sources, device,
            start_run=source, fixed_xy=decode, live=live, accumulated_round=1,
            xy_factor=factor, objective_pilot=name)
        del main_panel, sources, live
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
