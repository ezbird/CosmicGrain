#!/usr/bin/env python3
"""
Make the Zhang & Green R(V) resolution-convergence power-spectrum figure.

Input
-----
A native 50-pc products NPZ from zhang_green_rv_power_structure.py.

Method
------
1. Start from the SAME native 50-pc projected R(V) map for every curve.
2. Degrade to 50, 100, 200, and 250 pc using the saved extinction weight:
       Rv_coarse = sum(Rv * W) / sum(W)
3. Keep pixels inside R <= 2.5 kpc.
4. Apply the same 0.20-kpc cosine taper at every resolution.
5. Compute the masked 2-D pseudo-power spectrum and plot k^2 P(k).

Nothing is interpolated between resolutions.
"""

import math
import sys
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("cosmicgrain.mplstyle")

RESOLUTIONS = [0.05, 0.10, 0.20, 0.25]  # kpc
RADIUS = 2.5                              # kpc
TAPER = 0.20                              # kpc
NBINS = 35


def block_sum(a, factor):
    """Sum factor x factor blocks."""
    ny, nx = a.shape
    if ny % factor or nx % factor:
        raise ValueError(f"{a.shape} is not divisible by factor {factor}")
    return a.reshape(ny//factor, factor, nx//factor, factor).sum((1, 3))


def degrade(rv, weight, valid, factor):
    """Extinction-weighted spatial degradation."""
    good = valid & np.isfinite(rv) & np.isfinite(weight) & (weight > 0)

    w = np.where(good, weight, 0.0)
    rw = np.where(good, rv * weight, 0.0)

    wc = block_sum(w, factor)
    rwc = block_sum(rw, factor)

    out = np.full_like(wc, np.nan, dtype=float)
    ok = wc > 0
    out[ok] = rwc[ok] / wc[ok]
    return out, wc, ok


def radius_map(n, dx):
    """Distance of every pixel center from map center."""
    x = (np.arange(n) - 0.5*(n - 1)) * dx
    xx, yy = np.meshgrid(x, x, indexing="xy")
    return np.hypot(xx, yy)


def cosine_window(rr):
    """Circular R<=2.5 kpc window with a 0.20-kpc cosine edge taper."""
    w = np.zeros_like(rr)
    inner = RADIUS - TAPER
    w[rr <= inner] = 1.0

    edge = (rr > inner) & (rr < RADIUS)
    x = (rr[edge] - inner) / TAPER
    w[edge] = 0.5 * (1.0 + np.cos(np.pi*x))
    return w


def power_spectrum(rv, valid, rr, dx):
    """Azimuthally averaged masked pseudo-power spectrum."""
    win = valid.astype(float) * cosine_window(rr)

    # Subtract the window-weighted mean before the FFT.
    mean = np.sum(rv[valid] * win[valid]) / np.sum(win[valid])
    field = np.zeros_like(rv)
    field[valid] = rv[valid] - mean

    F = np.fft.fftshift(np.fft.fft2(field * win)) * dx**2
    area = np.sum(win**2) * dx**2
    P2 = np.abs(F)**2 / area

    n = rv.shape[0]
    kval = 2*np.pi*np.fft.fftshift(np.fft.fftfreq(n, d=dx))
    kx, ky = np.meshgrid(kval, kval, indexing="xy")
    kk = np.hypot(kx, ky)

    kfund = 2*np.pi/(n*dx)
    knyq = np.pi/dx
    edges = np.geomspace(kfund, knyq, NBINS + 1)

    k, P = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        q = (kk >= lo) & (kk < hi)
        if np.any(q):
            k.append(np.sqrt(lo*hi))
            P.append(np.mean(P2[q]))

    return np.asarray(k), np.asarray(P), knyq


def main():
    if len(sys.argv) != 2:
        sys.exit(f"Usage: python3 {sys.argv[0]} zhang_green_rv_products.npz")

    dat = np.load(sys.argv[1])
    rv0 = np.asarray(dat["rv"], float)
    weight0 = np.asarray(dat["column_weight"], float)
    valid0 = np.asarray(dat["valid_mask"], bool)
    dx0 = float(np.asarray(dat["pixel_kpc"]).ravel()[0])

    fig, ax = plt.subplots(figsize=(7.5, 6))

    for target_dx in RESOLUTIONS:
        factor = int(round(target_dx / dx0))
        if not np.isclose(target_dx, factor*dx0):
            raise ValueError(f"{target_dx} kpc is not an integer multiple of {dx0}")

        rv, weight, has_data = degrade(rv0, weight0, valid0, factor)
        dx = factor * dx0
        rr = radius_map(rv.shape[0], dx)

        valid = has_data & (rr <= RADIUS) & np.isfinite(rv) & (weight > 0)

        k, P, knyq = power_spectrum(rv, valid, rr, dx)
        k2P = k*k*P

        ax.loglog(
            k, k2P, marker="o", ms=4,
            label=f"{dx*1000:.0f} pc"
        )

        print(
            f"{dx*1000:4.0f} pc: "
            f"{valid.sum()} valid pixels, kNyq={knyq:.2f} kpc^-1"
        )

    ax.set_xlabel(r"$k\ (\mathrm{kpc}^{-1})$")
    ax.set_ylabel(r"$k^2P_{R(V)}(k)$")
    ax.set_title("Zhang & Green R(V)")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()

    fig.savefig("zhang_green_rv_resolution_convergence_power_spectrum.png", dpi=220)
    fig.savefig("zhang_green_rv_resolution_convergence_power_spectrum.pdf")
    print("Saved Zhang & Green resolution-convergence figure.")


if __name__ == "__main__":
    main()
