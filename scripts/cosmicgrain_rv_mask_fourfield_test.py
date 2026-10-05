#!/usr/bin/env python3
"""
CosmicGrain four-field mask diagnostic.

At each requested map pixel size, construct the normal R_V validity mask and
apply THAT EXACT SAME mask/window to:
  1) R_V
  2) dust surface density
  3) tau_V
  4) tau_B

This tests whether the resolution-dependent R_V mask/window itself can produce
the vertical power-spectrum shift.

Input NPZ must contain:
tau_b, tau_v, dust_mass_code, dust_count, pixel_size_pkpc
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt

def block_sum(a, f):
    ny, nx = a.shape
    if ny % f or nx % f:
        raise ValueError(f"Shape {a.shape} is not divisible by rebin factor {f}")
    return a.reshape(ny//f, f, nx//f, f).sum(axis=(1,3))

def make_rv(tb, tv):
    den = tb - tv
    rv = np.full_like(tv, np.nan, dtype=float)
    q = np.isfinite(tb) & np.isfinite(tv) & (tv > 0) & (den > 0)
    rv[q] = tv[q] / den[q]
    return rv

def tapered_window(mask, taper_pixels):
    if taper_pixels <= 0:
        return mask.astype(float)
    d = distance_transform_edt(mask)
    u = np.clip(d / float(taper_pixels), 0.0, 1.0)
    w = np.sin(0.5*np.pi*u)**2
    w[~mask] = 0.0
    return w

def isotropic_ps(field, dx, mask, taper_pixels):
    mask = np.asarray(mask, bool) & np.isfinite(field)
    w = tapered_window(mask, taper_pixels)
    if np.sum(w) <= 0 or np.sum(w*w) <= 0:
        return np.array([]), np.array([]), np.array([])

    vals = np.where(mask, field, 0.0)
    mu = np.sum(w * vals) / np.sum(w)

    delta = np.zeros_like(field, dtype=float)
    delta[mask] = field[mask] - mu

    F = dx**2 * np.fft.fftshift(np.fft.fft2(w * delta))
    Aeff = dx**2 * np.sum(w*w)
    P2 = np.abs(F)**2 / Aeff

    ny, nx = field.shape
    kx = 2*np.pi*np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    ky = 2*np.pi*np.fft.fftshift(np.fft.fftfreq(ny, d=dx))
    KX, KY = np.meshgrid(kx, ky)
    K = np.hypot(KX, KY)

    kfund = 2*np.pi / min(nx*dx, ny*dx)
    knyq = np.pi/dx
    edges = np.arange(0.5*kfund, knyq + 1.5*kfund, kfund)

    kk, pp, nn = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        q = (K >= lo) & (K < hi)
        if np.any(q):
            kk.append(np.mean(K[q]))
            pp.append(np.mean(P2[q]))
            nn.append(np.sum(q))
    return np.asarray(kk), np.asarray(pp), np.asarray(nn)

def dxlabel(dx):
    return f"{dx*1000:.0f} pc" if dx < 1 else f"{dx:g} kpc"

p = argparse.ArgumentParser()
p.add_argument("npz")
p.add_argument("--pixel-sizes", nargs="+", type=float,
               default=[0.125, 0.25, 0.5, 1.0, 2.0])
p.add_argument("--min-count", type=int, default=1)
p.add_argument("--signal-frac", type=float, default=1e-3)
p.add_argument("--taper-pixels", type=int, default=2)
p.add_argument("--min-modes", type=int, default=8)
p.add_argument("--prefix", default="halo441_rv_mask_fourfield_test")
a = p.parse_args()

z = np.load(a.npz)
tb0 = np.asarray(z["tau_b"], float)
tv0 = np.asarray(z["tau_v"], float)
dm0 = np.asarray(z["dust_mass_code"], float)
nc0 = np.asarray(z["dust_count"], int)
dx0 = float(np.asarray(z["pixel_size_pkpc"]).ravel()[0])

names = ["rv", "dust", "tauv", "taub"]
titles = [r"$R_V$", "Dust surface density", r"$\tau_V$", r"$\tau_B$"]
ylabs = [
    r"$k^2P_{R_V}(k)$",
    r"$k^2P_{\Sigma_{\rm dust}}(k)$",
    r"$k^2P_{\tau_V}(k)$",
    r"$k^2P_{\tau_B}(k)$"
]
spectra = {name: [] for name in names}

print("="*84)
print("SAME R_V MASK APPLIED TO FOUR FIELDS")
print("="*84)
print("At each resolution all four fields use the identical R_V mask/window.\n")

for dx in a.pixel_sizes:
    f = int(round(dx/dx0))
    if f < 1 or not np.isclose(dx/dx0, f):
        raise ValueError(f"{dx} kpc is not an integer multiple of native {dx0} kpc")

    tb = block_sum(tb0, f)
    tv = block_sum(tv0, f)
    dm = block_sum(dm0, f)
    nc = block_sum(nc0, f).astype(int)

    rv = make_rv(tb, tv)
    tau_threshold = a.signal_frac * np.nanmax(tv)
    mask = np.isfinite(rv) & (nc >= a.min_count) & (tv >= tau_threshold)

    sigma_dust = dm / dx**2
    fields = {"rv": rv, "dust": sigma_dust, "tauv": tv, "taub": tb}

    print(f"{dxlabel(dx):>6s}: mask={mask.sum():6d}/{mask.size:<6d} "
          f"({100*mask.mean():6.2f}%), tauV threshold={tau_threshold:.5e}")

    for name in names:
        k, P, nmodes = isotropic_ps(fields[name], dx, mask, a.taper_pixels)
        spectra[name].append((dx, k, k*k*P, nmodes))

print("\nPower ratios relative to 125 pc")
print("(p16 / median / p84 for k <= 0.5 kNyq of each coarse map)")
for name in names:
    print(f"\n{name}:")
    _, kref, yref, _ = spectra[name][0]
    for dx, k, y, nmodes in spectra[name][1:]:
        knyq = np.pi/dx
        use = ((k > 0) & (k <= 0.5*knyq) & (nmodes >= a.min_modes)
               & np.isfinite(y) & (y > 0))
        ref = np.interp(k[use], kref, yref, left=np.nan, right=np.nan)
        ratio = y[use]/ref
        ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
        if ratio.size:
            q16, q50, q84 = np.percentile(ratio, [16,50,84])
            print(f"  {dxlabel(dx):>6s}: {q16:.3f} / {q50:.3f} / {q84:.3f}")
        else:
            print(f"  {dxlabel(dx):>6s}: insufficient common modes")

fig, axs = plt.subplots(2, 2, figsize=(11.5, 8.5))
axs = axs.ravel()

for ax, name, title, ylab in zip(axs, names, titles, ylabs):
    for dx, k, y, nmodes in spectra[name]:
        use = ((nmodes >= a.min_modes) & np.isfinite(k) & np.isfinite(y)
               & (k > 0) & (y > 0))
        ax.loglog(k[use], y[use], lw=1.7, label=dxlabel(dx))
    ax.set_title(title)
    ax.set_xlabel(r"$k$ (kpc$^{-1}$)")
    ax.set_ylabel(ylab)
    ax.grid(True, which="both", alpha=0.18)
    ax.tick_params(direction="in", which="both", top=True, right=True)

fig.suptitle("halo441 power spectrum — identical $R_V$ mask", fontsize=15)
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
fig.tight_layout(rect=(0, 0.06, 1, 0.95))

fig.savefig(a.prefix + ".png", dpi=300, bbox_inches="tight")
fig.savefig(a.prefix + ".pdf", bbox_inches="tight")
plt.close(fig)

out = {}
for name in names:
    for dx, k, y, nmodes in spectra[name]:
        tag = f"{round(dx*1000):.0f}pc"
        out[f"k_{name}_{tag}"] = k
        out[f"k2p_{name}_{tag}"] = y
        out[f"Nmodes_{name}_{tag}"] = nmodes
np.savez_compressed(a.prefix + "_spectra.npz", **out)

print("\nSaved:")
print(" ", a.prefix + ".png")
print(" ", a.prefix + ".pdf")
print(" ", a.prefix + "_spectra.npz")
