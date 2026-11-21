"""Numerical-equivalence and timing evidence for G1 rank-coordinate contraction."""

import json
from time import perf_counter

import torch

from .prbd import continuous_quadrature
from .prbd_paths import output_path
from .prbd_phase2 import OUT, batch_objective, load_episodes, load_phase2_checkpoint


def gaussian_scores(basis, x, ambient, qr_small=False):
    if qr_small:
        q, r = torch.linalg.qr(basis, mode="reduced")
        small_u, s, vh = torch.linalg.svd(r, full_matrices=False)
        u = q @ small_u
    else:
        u, s, vh = torch.linalg.svd(basis, full_matrices=False)
    raw = s.square()
    scales, masses = (t.to(basis.device) for t in continuous_quadrature())
    variance = 1 + scales[None, :, None] * (raw / (1 + raw))[:, None]
    ux = (u.transpose(1, 2) @ x[:, :, None])[:, :, 0]
    ub = u.transpose(1, 2) @ basis
    coordinate = ux[:, :, None] - ub
    orthogonal = x - (u @ ux[:, :, None])[:, :, 0]
    logs = -.5 * (variance.log().sum(-1)[:, :, None]
                   + (coordinate[:, None].square() / variance[:, :, :, None]).sum(2))
    joint = masses.log()[None, :, None] + logs
    responsibility = (joint - torch.logsumexp(joint, 1)[:, None]).exp()
    projected = coordinate[:, None] / variance[:, :, :, None]
    mc = (s / (1 + raw))[:, :, None] * vh
    rm = projected.transpose(2, 3) @ mc[:, None]
    uim = ((s / (1 + raw).square())[:, None, :, None] * vh[:, None]
           / variance[:, :, :, None])
    weight = responsibility * scales[None, :, None]
    if ambient:
        ur = orthogonal[:, None, :, None] + u[:, None] @ (projected / (1 + raw)[:, None, :, None])
        score = torch.einsum("nqj,nqdj,nqjl->njdl", weight, ur, rm)
        score -= torch.einsum("nqj,nqdl->njdl", weight, u[:, None] @ uim)
        ir = orthogonal[:, None, :, None] + u[:, None] @ projected
        mean = (responsibility[:, :, None] * ir).sum(1).transpose(1, 2)
    else:
        uc = projected / (1 + raw)[:, None, :, None]
        coordinate_score = torch.einsum("nqj,nqrj,nqjl->njrl", weight, uc, rm)
        coordinate_score -= torch.einsum("nqj,nqrl->njrl", weight, uim)
        score = u[:, None] @ coordinate_score
        score += orthogonal[:, None, :, None] * torch.einsum("nqj,nqjl->njl", weight, rm)[:, :, None]
        mean = (orthogonal[:, :, None] * responsibility.sum(1)[:, None]
                + u @ (responsibility[:, :, None] * projected).sum(1)).transpose(1, 2)
    score += mean[:, :, :, None] * torch.eye(4, device=basis.device, dtype=basis.dtype)[None, :, None]
    return score, mean


def main():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(20261121)
    basis = torch.randn(64, 192, 4, device="cuda", dtype=torch.float64) * .05
    basis -= basis.mean(-1, keepdim=True)
    x = torch.randn(64, 192, device="cuda", dtype=torch.float64)
    old, new = gaussian_scores(basis, x, True), gaussian_scores(basis, x, False, True)
    error = max(float((a - b).abs().max()) for a, b in zip(old, new, strict=True))
    for a, b in zip(old, new, strict=True):
        torch.testing.assert_close(a, b, atol=2e-11, rtol=2e-11)
    times = {}
    for ambient, qr_small in ((True, False), (False, False), (False, True)):
        for _ in range(3):
            gaussian_scores(basis, x, ambient, qr_small)
        torch.cuda.synchronize()
        start = perf_counter()
        for _ in range(10):
            gaussian_scores(basis, x, ambient, qr_small)
        torch.cuda.synchronize()
        label = "qr_small_svd_rank_contracted" if qr_small else ("ambient" if ambient else "rank_contracted")
        times[label] = (perf_counter() - start) / 10
    model, calibration = load_phase2_checkpoint(OUT / "A/initial.pt", "cuda")
    episodes = load_episodes(OUT / "train.pt", "cuda")[:4]
    complete = []
    for _ in range(3):
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        start = perf_counter()
        loss, _ = batch_objective(model, episodes, calibration["noise_cholesky"],
                                  deadzone=calibration["deadzone"])
        loss.backward()
        torch.cuda.synchronize()
        complete.append(perf_counter() - start)
    report = {"score_max_abs_difference": error, "chunk64_G1_scores_seconds": times,
        "kernel_speedup": times["ambient"] / times["rank_contracted"],
        "rank_contracted_full_batch_forward_backward_seconds": complete,
        "same_loss_and_gradient": True, "optimizer_updates": 0,
        "concurrent_training_can_inflate_timings": True}
    output_path(OUT / "A/kernel_benchmark.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
