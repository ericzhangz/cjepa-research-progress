"""Continue one existing predictor for 160 updates per matched coverage arm."""

import argparse

import torch

from .native_topology_prefix_fit import RUN as PREFIX_RUN, fit, read_fan, read_prefix_sources


RUN = PREFIX_RUN.with_name("native_topology_paired_20261115")
ARMS = ("control", "paired")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", choices=ARMS, nargs="+", default=ARMS)
    options = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    paths = {"control": PREFIX_RUN / "data_train_coverage.npz", "paired": RUN / "data_train_paired.npz"}
    panels = {arm: read_fan(path, device) for arm, path in paths.items()}
    for control, paired in zip(panels["control"][:2], panels["paired"][:2], strict=True):
        torch.testing.assert_close(control, paired, atol=0, rtol=0)
    source = tuple(torch.cat(samples) for samples in zip(
        *(read_prefix_sources(paths[arm], device) for arm in ARMS), strict=True,
    ))
    for arm in options.arms:
        fit(arm, RUN, panels[arm], source, device, start_run=PREFIX_RUN / "prefix")


if __name__ == "__main__":
    main()
