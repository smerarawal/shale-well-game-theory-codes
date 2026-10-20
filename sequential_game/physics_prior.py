"""
physics_prior.py -- batched torch version of solver_v3's IMPLICIT PRESSURE step, used as the
physics half of the hybrid stage surrogate.

solver_v3 solves, every pressure step, the linear system (compressible, closed boundary)

    phi*ct*(p_new - p_old)/dt = div(lambda_t(Sw) * k * grad p_new) + source,

with harmonic face averages of k and of lambda_t, injector sources and (game setting) the
gravity source built from the phase mobilities. For a FROZEN saturation field this is a
linear system, so a whole stage can be advanced with a few backward-Euler substeps on the
true k / phi: the dependence of pressure on the permeability geometry (channels, barriers)
is then exact, which is what the purely learned stage network gets wrong out of distribution.

What is approximate: lambda_t is frozen over the stage (the solver re-evaluates it every
~0.015 time units as the front moves) and the substep is coarser than the solver's CFL-derived
dt. The learned network is trained to supply the remaining residual.

    p = pressure_prior(k, phi, sw, well_rate, p_in, stage_time=1.0)       # all [B,n,n]
"""
import torch

SWC, SOR = 0.2, 0.2


def _relperm(sw, krw_max=0.8, krn_max=1.0, nw=2, nn=2):
    se = ((sw - SWC) / (1 - SWC - SOR)).clamp(0, 1)
    return krw_max * se ** nw, krn_max * (1 - se) ** nn


def _harm(a, b):
    return 2.0 * a * b / (a + b + 1e-10)


def _faces(f):
    """f:[B,n,n] -> (Fx [B,n-1,n] between (i,j),(i+1,j); Fy [B,n,n-1] between (i,j),(i,j+1))."""
    return _harm(f[:, :-1, :], f[:, 1:, :]), _harm(f[:, :, :-1], f[:, :, 1:])


def _idx(n, device):
    i, j = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    m = (i * n + j).to(device)
    return m[:-1, :].reshape(-1), m[1:, :].reshape(-1), m[:, :-1].reshape(-1), m[:, 1:].reshape(-1)


def _assemble(k, phi, sw, q, n, dt, ct, mu_w, mu_n, gravity, rho_w, rho_n, g, idx):
    """-> (A [B,N,N], rhs_source [B,N], acc [B,N]) for one backward-Euler step with lambda(sw) frozen."""
    B = k.shape[0]
    N = n * n
    krw, krn = _relperm(sw)
    lam_w, lam_n = krw / mu_w, krn / mu_n
    lam = lam_w + lam_n
    kx, ky = _faces(k)
    lx, ly = _faces(lam)
    Tx, Ty = kx * lx, ky * ly                                        # dx = 1
    mx0, mx1, my0, my1 = idx
    src = q.clone()
    if gravity:                                                        # solver: Q_face on j-faces, source += Q_south - Q
        _, lwy = _faces(lam_w)
        _, lny = _faces(lam_n)
        Q = ky * (lwy * rho_w + lny * rho_n) * g                       # [B,n,n-1]; last column = 0 in the solver
        Qf = torch.zeros_like(k)
        Qf[:, :, :-1] = Q
        Qs = torch.zeros_like(k)
        Qs[:, :, 1:] = Q
        src = src + (Qs - Qf)
    acc = (phi * ct / dt).reshape(B, N)
    A = torch.zeros(B, N, N, dtype=k.dtype, device=k.device)
    tx, ty = Tx.reshape(B, -1), Ty.reshape(B, -1)
    for m0, m1, t in ((mx0, mx1, tx), (my0, my1, ty)):
        A[:, m0, m1] -= t
        A[:, m1, m0] -= t
        A[:, m0, m0] += t
        A[:, m1, m1] += t
    return A + torch.diag_embed(acc), src.reshape(B, N), acc


def pressure_prior(k, phi, sw, well_rate, p_in, stage_time=1.0, n_sub=8, sw_end=None, ct=1.0, mu_w=1.0, mu_n=5.0,
                   gravity=True, rho_w=1.0, rho_n=0.6, g=1.5, dtype=torch.float64):
    """Backward-Euler pressure advance over `stage_time` in `n_sub` substeps.
    sw_end=None: lambda frozen at `sw`.  sw_end given: lambda follows Sw(t) = sw + (t/T)(sw_end - sw)
    (midpoint of each substep), i.e. the saturation path is assumed linear in time.
    Differentiable w.r.t. sw / sw_end (and every other input). Returns p [B,n,n] in the input dtype."""
    odt = p_in.dtype
    k, phi, sw, q, p = (a.to(dtype) for a in (k, phi, sw, well_rate, p_in))
    sw_end = None if sw_end is None else sw_end.to(dtype)
    B, n, _ = k.shape
    idx = _idx(n, k.device)
    dt = stage_time / n_sub
    pp = p.reshape(B, n * n)
    LU = piv = None
    for m in range(n_sub):
        if m == 0 or sw_end is not None:
            s_m = sw if sw_end is None else sw + (m + 0.5) / n_sub * (sw_end - sw)
            A, s, acc = _assemble(k, phi, s_m, q, n, dt, ct, mu_w, mu_n, gravity, rho_w, rho_n, g, idx)
            LU, piv = torch.linalg.lu_factor(A)
        pp = torch.linalg.lu_solve(LU, piv, (acc * pp + s).unsqueeze(-1)).squeeze(-1)
    return pp.reshape(B, n, n).to(odt)
