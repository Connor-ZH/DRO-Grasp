import typing
from typing import Optional

import torch


@typing.no_type_check
def estimate_p(
    P: torch.FloatTensor, R: torch.FloatTensor, W: Optional[torch.FloatTensor] = None
) -> torch.FloatStorage:
    assert P.ndim == 3  # N x D x 1
    assert R.ndim == 1  # N
    assert P.shape[0] == R.shape[0]
    assert P.shape[1] in {2, 3}

    N, D, _ = P.shape

    if W is None:
        W = torch.ones(N, device=P.device)
    assert W.ndim == 1  # N
    W = W[:, None, None]

    # Shared stuff.
    Pt = P.permute(0, 2, 1)
    PPt = P @ Pt
    PtP = (Pt @ P).squeeze()
    I = torch.eye(D, device=P.device)
    NI = I[None].repeat(N, 1, 1)
    PtP_minus_r2 = (PtP - R**2)[:, None, None]

    # These are ripped straight from the paper, with weighting passed through.
    a = (W * (PtP_minus_r2 * P)).mean(dim=0)
    B = (W * (-2 * PPt - PtP_minus_r2 * NI)).mean(dim=0)
    c = (W * P).mean(dim=0)
    f = a + B @ c + 2 * c @ c.T @ c
    H = -2 * PPt.mean(dim=0) + 2 * c @ c.T
    q = -torch.linalg.inv(H) @ f
    p = q + c

    return p


def multilateration(dro, fixed_pc):
    """
    Compute the target point cloud described by D(R,O) matrix & fixed_pc

    :param dro: (B, N, N), point-wise relative distance matrix between target point cloud & fixed point cloud
    :param fixed_pc: (B, N, 3), point cloud as a reference for relative distance
    :return: (B, N, 3), the target point cloud
    """
    assert dro.ndim == 3 and fixed_pc.ndim == 3, "multilateration() requires batch data."
    v_est_p = torch.vmap(torch.vmap(estimate_p, in_dims=(None, 0)))
    target_pc = v_est_p(fixed_pc.unsqueeze(-1), dro)[..., 0]
    return target_pc


def estimate_p_batch(P, R, W=None):
    """
    P: (B, N, D, 1)
    R: (B, N)
    """
    B, N, D, _ = P.shape

    if W is None:
        W = torch.ones(B, N, device=P.device)
    W = W[..., None, None]  # (B, N, 1, 1)

    Pt = P.transpose(-2, -1)                    # (B, N, 1, D)
    PPt = P @ Pt                                # (B, N, D, D)
    PtP = (Pt @ P).squeeze(-1).squeeze(-1)      # (B, N)
    I = torch.eye(D, device=P.device).expand(B, N, D, D)
    PtP_minus_r2 = (PtP - R**2)[..., None, None]  # (B, N, 1, 1)

    a = (W * (PtP_minus_r2 * P)).mean(dim=1)          # (B, D, 1)
    Bmat = (W * (-2 * PPt - PtP_minus_r2 * I)).mean(dim=1)  # (B, D, D)
    c = (W * P).mean(dim=1)                           # (B, D, 1)
    f = a + Bmat @ c + 2 * c @ c.transpose(-2, -1) @ c  # (B, D, 1)
    H = -2 * PPt.mean(dim=1) + 2 * c @ c.transpose(-2, -1)  # (B, D, D)

    q = torch.linalg.solve(-H, f)  # (B, D, 1)
    p = q + c
    return p


def multilateration_new(dro, fixed_pc):
    """
    dro: (B, N, N)
    fixed_pc: (B, N, 3)
    """
    B, N, D = fixed_pc.shape
    P = fixed_pc.unsqueeze(-1)  # (B, N, D, 1)

    # Each row of dro is the R vector for one point
    # Reshape to (B*N, N)
    R = dro.reshape(B * N, N)

    # Expand P to match (B*N, N, D, 1)
    P_expanded = P.unsqueeze(1).expand(B, N, N, D, 1).reshape(B * N, N, D, 1)

    # Solve all at once
    p = estimate_p_batch(P_expanded, R)  # (B*N, D, 1)

    target_pc = p.view(B, N, D)  # (B, N, D)
    return target_pc

if __name__ == "__main__":
    import time

    B, N = 32, 512
    dro = torch.rand(B, N, N, device="cuda")
    fixed_pc = torch.rand(B, N, 3, device="cuda")

    # Warmup
    _ = multilateration(dro, fixed_pc)
    _ = multilateration_new(dro, fixed_pc)

    # Measure ref
    start = time.time()
    for _ in range(100):
        _ = multilateration(dro, fixed_pc)
    torch.cuda.synchronize()
    print("Ref time:", (time.time() - start)/10)

    # Measure fast
    start = time.time()
    for _ in range(100):
        _ = multilateration_new(dro, fixed_pc)
    torch.cuda.synchronize()
    print("Fast time:", (time.time() - start)/10)
