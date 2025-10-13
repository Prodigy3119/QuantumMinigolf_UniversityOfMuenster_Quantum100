from __future__ import annotations
import numpy as np
import math

# NOTE: scipy.ndimage.uniform_filter may be available via backends import, but we keep a safe local fallback
try:
    from scipy.ndimage import uniform_filter as _scipy_uniform_filter
except Exception:
    _scipy_uniform_filter = None

def step_wave(psi, expV_half, expK, fft2, ifft2, inplace=True):
    if inplace:
        psi *= expV_half
        psi_k = fft2(psi)
        psi_k *= expK
        psi = ifft2(psi_k)
        psi *= expV_half
        return psi.astype(np.complex64, copy=False)
    else:
        psi = expV_half * psi
        psi_k = fft2(psi)
        psi_k *= expK
        psi = ifft2(psi_k)
        psi = expV_half * psi
        return psi.astype(np.complex64, copy=False)

def mean_pool2(A, s, xp):
    if s <= 1:
        return A
    ny, nx = A.shape
    ny2 = (ny // s) * s
    nx2 = (nx // s) * s
    A = A[:ny2, :nx2]
    return A.reshape(ny2 // s, s, nx2 // s, s).mean(axis=(1, 3))

def fast_box_blur(F, passes, xp, use_gpu=False):
    if passes <= 0:
        return F
    if not use_gpu and _scipy_uniform_filter is not None:
        for _ in range(passes):
            F = _scipy_uniform_filter(F, size=3, mode='nearest')
        return F
    # roll-based 3x3 blur
    for _ in range(passes):
        F = (F +
             xp.roll(F,  1, 0) + xp.roll(F, -1, 0) +
             xp.roll(F,  1, 1) + xp.roll(F, -1, 1) +
             xp.roll(xp.roll(F, 1, 0),  1, 1) +
             xp.roll(xp.roll(F, 1, 0), -1, 1) +
             xp.roll(xp.roll(F,-1, 0),  1, 1) +
             xp.roll(xp.roll(F,-1, 0), -1, 1)
            ) / 9.0
    return F

# physics.py
def prepare_frame(
    dens_xp,
    xp,
    use_gpu,
    flags,
    smooth_passes,
    ds_factor,
    to_cpu,
    *,
    encode_uint8: bool = False,
    norm_min: float = 0.0,
    norm_max: float = 0.04,
    return_device_array: bool = False,
):
    """
    Convert |psi|^2 to a display-ready scalar field.
    GPU path stays on xp; CPU path copies once and does the rest locally.
    """
    norm_range = float(max(norm_max - norm_min, 1e-6))

    if use_gpu:
        F = xp.sqrt(dens_xp)  # nicer dynamic range
        if flags.fast_blur:
            F = fast_box_blur(F, smooth_passes, xp, use_gpu=True)
        if flags.display_downsample and ds_factor > 1:
            F = mean_pool2(F, ds_factor, xp)
        if encode_uint8:
            if norm_min != 0.0:
                F = F - norm_min
            F = xp.clip(F, 0.0, norm_range)
            F = xp.rint(F * (255.0 / norm_range))
            F = F.astype(xp.uint8, copy=False)
            if return_device_array:
                return F
            return to_cpu(F)
        if return_device_array:
            return F
        return to_cpu(F).astype(np.float32, copy=False)

    # CPU path: single device->host copy, then all ops here
    F = to_cpu(dens_xp).astype(np.float32, copy=False)
    np.sqrt(F, out=F)
    if flags.fast_blur and _scipy_uniform_filter is not None and smooth_passes > 0:
        for _ in range(smooth_passes):
            F = _scipy_uniform_filter(F, size=3, mode='nearest')
    if flags.display_downsample and ds_factor > 1:
        ny, nx = F.shape
        ny2 = (ny // ds_factor) * ds_factor
        nx2 = (nx // ds_factor) * ds_factor
        F = F[:ny2, :nx2].reshape(ny2 // ds_factor, ds_factor, nx2 // ds_factor, ds_factor).mean(axis=(1, 3))

    if encode_uint8:
        if norm_min != 0.0:
            F -= norm_min
        np.clip(F, 0.0, norm_range, out=F)
        F *= (255.0 / norm_range)
        return F.astype(np.uint8, copy=False)

    return F.astype(np.float32, copy=False)


def compute_expectation(X, Y, dens_xp, xp, to_cpu):
    denom = xp.sum(dens_xp) + 1e-12
    Ex = xp.sum(X * dens_xp) / denom
    Ey = xp.sum(Y * dens_xp) / denom
    return float(to_cpu(Ex)), float(to_cpu(Ey))

def covariance_ellipse(X, Y, dens_xp, xp, to_cpu):
    denom = xp.sum(dens_xp) + 1e-12
    Ex = xp.sum(X * dens_xp) / denom
    Ey = xp.sum(Y * dens_xp) / denom
    Ex2 = xp.sum((X * X) * dens_xp) / denom
    Ey2 = xp.sum((Y * Y) * dens_xp) / denom
    Exy = xp.sum((X * Y) * dens_xp) / denom

    ex = float(to_cpu(Ex)); ey = float(to_cpu(Ey))
    ex2 = float(to_cpu(Ex2)); ey2 = float(to_cpu(Ey2))
    exy = float(to_cpu(Exy))

    varx = max(0.0, ex2 - ex * ex)
    vary = max(0.0, ey2 - ey * ey)
    cov = exy - ex * ey

    S = np.array([[varx, cov], [cov, vary]], dtype=float)
    w, v = np.linalg.eigh(S)
    w = np.clip(w, 0.0, None)
    order = np.argsort(w)[::-1]
    w = w[order]; v = v[:, order]
    angle_deg = math.degrees(math.atan2(v[1, 0], v[0, 0]))
    a1 = 2.0 * math.sqrt(w[0] + 1e-12)  # 1-sigma diameter
    b1 = 2.0 * math.sqrt(w[1] + 1e-12)
    return ex, ey, a1, b1, angle_deg

def sample_from_density(dens_cpu: np.ndarray,
                        gamma: float = 1.0,
                        hole_mask_cpu: np.ndarray | None = None,
                        boost_beta: float = 0.0,
                        rng: np.random.Generator | None = None) -> tuple[float, float]:
    if rng is None:
        rng = np.random.default_rng()
    f = np.clip(dens_cpu.astype(np.float64), 0.0, None)
    if gamma != 1.0:
        f = np.power(f, gamma, out=f)
    if hole_mask_cpu is not None and boost_beta > 0.0:
        m = hole_mask_cpu
        hole_count = int(m.sum())
        if hole_count > 0:
            h = np.zeros_like(f)
            h[m] = 1.0 / hole_count
            beta = float(np.clip(boost_beta, 0.0, 1.0))
            f = (1.0 - beta) * f + beta * h

    s = f.sum()
    if not np.isfinite(s) or s <= 0.0:
        Ny, Nx = f.shape
        return float(Nx / 2), float(Ny / 2)

    cdf = np.cumsum(f.ravel())
    u = (rng.random() * cdf[-1])
    idx = int(np.searchsorted(cdf, u, side='right'))
    Ny, Nx = f.shape
    iy, ix = divmod(idx, Nx)
    x = ix + rng.random()
    y = iy + rng.random()
    x = float(np.clip(x, 0, Nx - 1e-6))
    y = float(np.clip(y, 0, Ny - 1e-6))
    return x, y
