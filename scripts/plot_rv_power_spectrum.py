#!/usr/bin/env python3
"""
plot_rv_power_spectrum.py

Power-spectrum analysis of a face-on CosmicGrain R(V) map saved by
halo569_rv_sfr_triptych.py.

Computes two 2-D pseudo-power spectra:
  1) total R(V) fluctuations:       R(V) - <R(V)>
  2) radial-detrended fluctuations: R(V) - <R(V)>_r

The validity mask is apodized before the FFT. The reported power normalization
corrects the overall suppression by <W^2>, but does NOT deconvolve Fourier
mode-mixing caused by the mask/window. A normalized window spectrum is saved
as a diagnostic.

Angular-wavenumber convention:
    k = 2*pi*f       [pkpc^-1]
    lambda = 2*pi/k [pkpc]

For the adopted continuous-transform normalization,
    variance per d ln k = k^2 P(k) / (2*pi)
for an isotropic 2-D field.

Requires: numpy, matplotlib, scipy
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy.ndimage import distance_transform_edt
except ImportError as exc:
    raise SystemExit("ERROR: scipy is required (scipy.ndimage.distance_transform_edt).") from exc


def scalar_from_npz(d, key, default=None):
    if key not in d:
        return default
    x = np.asarray(d[key])
    if x.size != 1:
        return default
    return float(x.reshape(-1)[0])


def build_apodized_window(mask, taper_pixels):
    """Smoothly taper valid pixels to zero near every mask boundary."""
    mask = np.asarray(mask, dtype=bool)
    if taper_pixels <= 0:
        return mask.astype(float)

    # Padding makes the outer array boundary behave as invalid.
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    dist = distance_transform_edt(padded)[1:-1, 1:-1]

    u = np.clip(dist / float(taper_pixels), 0.0, 1.0)
    # sin^2 taper: zero at boundary, unity beyond taper width.
    w = np.sin(0.5 * np.pi * u) ** 2
    w[~mask] = 0.0
    return w


def radial_profile_trend(rv, mask, radius, bin_width):
    """Mean R(V) in radial annuli, interpolated back onto the 2-D field."""
    rmax = np.nanmax(radius[mask])
    edges = np.arange(0.0, rmax + bin_width, bin_width)
    if edges[-1] <= rmax:
        edges = np.append(edges, edges[-1] + bin_width)

    rc = 0.5 * (edges[:-1] + edges[1:])
    prof = np.full(len(rc), np.nan)
    n = np.zeros(len(rc), dtype=int)

    for i in range(len(rc)):
        m = mask & (radius >= edges[i]) & (radius < edges[i + 1])
        n[i] = np.count_nonzero(m)
        if n[i] > 0:
            prof[i] = np.mean(rv[m])

    good = np.isfinite(prof)
    if np.count_nonzero(good) < 2:
        raise RuntimeError("Too few populated radial bins to construct radial trend.")

    # Constant extrapolation at the ends is intentional.
    trend = np.interp(radius, rc[good], prof[good],
                      left=prof[good][0], right=prof[good][-1])
    return trend, rc, prof, n


def power2d(delta, window, dx):
    """
    Continuous-FT-like pseudo-spectrum:
      F = dx^2 FFT(W delta)
      P = |F|^2 / (A <W^2>)

    This makes the integrated spectral variance equal
      sum(W^2 delta^2) / sum(W^2)
    up to floating-point precision.
    """
    ny, nx = delta.shape
    area = (nx * dx) * (ny * dx)
    mean_w2 = np.mean(window ** 2)
    if mean_w2 <= 0:
        raise RuntimeError("Window has zero weight.")

    g = window * delta
    F = (dx ** 2) * np.fft.fft2(g)
    P = np.abs(F) ** 2 / (area * mean_w2)

    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dx)
    KX, KY = np.meshgrid(kx, ky)
    kmag = np.sqrt(KX ** 2 + KY ** 2)

    dkx = 2.0 * np.pi / (nx * dx)
    dky = 2.0 * np.pi / (ny * dx)
    spectral_var = np.sum(P) * dkx * dky / (2.0 * np.pi) ** 2
    real_var = np.sum((window * delta) ** 2) / np.sum(window ** 2)

    return P, kmag, kx, ky, real_var, spectral_var


def annular_average(P2, kmag, kfund, knyq):
    """Average P(kx,ky) in annuli of width one fundamental mode."""
    # Restrict to k <= axis Nyquist so annuli retain full angular coverage.
    nmax = int(np.floor(knyq / kfund))
    edges = (np.arange(nmax + 1) + 0.5) * kfund
    # First edge is 0.5*kfund; bins center close to integer*kfund.
    kval = []
    pval = []
    nmodes = []

    for i in range(len(edges) - 1):
        m = (kmag >= edges[i]) & (kmag < edges[i + 1]) & (kmag <= knyq)
        nn = np.count_nonzero(m)
        if nn:
            kval.append(np.mean(kmag[m]))
            pval.append(np.mean(P2[m]))
            nmodes.append(nn)

    return np.asarray(kval), np.asarray(pval), np.asarray(nmodes, dtype=int)


def window_spectrum(window, dx, kmag, kfund, knyq):
    """Power-spectrum shape of the apodized validity window."""
    ny, nx = window.shape
    q = window - np.mean(window)
    F = dx ** 2 * np.fft.fft2(q)
    area = (nx * dx) * (ny * dx)
    P = np.abs(F) ** 2 / area
    k, p, n = annular_average(P, kmag, kfund, knyq)
    k2p = k ** 2 * p
    mx = np.nanmax(k2p) if np.any(np.isfinite(k2p)) else np.nan
    shape = k2p / mx if np.isfinite(mx) and mx > 0 else np.full_like(k2p, np.nan)
    return P, k, p, shape, n


def save_map(path, data, extent, title, cbar_label, cmap=None, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(data, origin="lower", extent=extent, interpolation="nearest",
                   aspect="equal", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xlabel("x' [pkpc]")
    ax.set_ylabel("y' [pkpc]")
    ax.set_title(title)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description="Compute k^2 P(k) for a face-on CosmicGrain R(V) map."
    )
    ap.add_argument("maps_npz", help="NPZ from halo569_rv_sfr_triptych.py")
    ap.add_argument("--prefix", default="rv_power_spectrum")
    ap.add_argument("--radial-bin-width", type=float, default=1.0,
                    help="Radial trend bin width [pkpc] (default: 1)")
    ap.add_argument("--taper-pixels", type=float, default=2.0,
                    help="Mask apodization width [pixels] (default: 2)")
    ap.add_argument("--max-k-fraction", type=float, default=0.8,
                    help="Upper k/k_Nyq for the conservative peak diagnostic (default: 0.8)")
    args = ap.parse_args()

    d = np.load(args.maps_npz, allow_pickle=False)

    if "rv" not in d or "rv_mask" not in d:
        raise KeyError("NPZ must contain 'rv' and 'rv_mask'.")

    rv = np.asarray(d["rv"], dtype=float)
    mask = np.asarray(d["rv_mask"], dtype=bool) & np.isfinite(rv)

    if rv.ndim != 2 or rv.shape[0] != rv.shape[1]:
        raise ValueError(f"Expected square 2-D R(V) map; got {rv.shape}")

    n = rv.shape[0]
    dx = scalar_from_npz(d, "pixel_size_pkpc")
    radius_map = scalar_from_npz(d, "plot_radius_pkpc")
    z = scalar_from_npz(d, "snapshot_redshift")

    if dx is None:
        if radius_map is None:
            raise KeyError("Need 'pixel_size_pkpc' or 'plot_radius_pkpc' in NPZ.")
        dx = 2.0 * radius_map / n

    if "centers_pkpc" in d:
        centers = np.asarray(d["centers_pkpc"], dtype=float).reshape(-1)
        if len(centers) != n:
            raise ValueError("centers_pkpc length does not match map dimensions.")
    else:
        centers = (np.arange(n) + 0.5 - 0.5 * n) * dx

    xx, yy = np.meshgrid(centers, centers)
    rr = np.sqrt(xx ** 2 + yy ** 2)

    valid = np.count_nonzero(mask)
    if valid < 20:
        raise RuntimeError(f"Only {valid} valid pixels; insufficient for spectrum.")

    L = n * dx
    kfund = 2.0 * np.pi / L
    knyq = np.pi / dx

    mean_rv = np.mean(rv[mask])
    delta_raw = np.zeros_like(rv)
    delta_raw[mask] = rv[mask] - mean_rv

    trend, rcent, rprof, rn = radial_profile_trend(
        rv, mask, rr, args.radial_bin_width
    )
    delta_det = np.zeros_like(rv)
    delta_det[mask] = rv[mask] - trend[mask]
    # Remove any tiny residual weighted monopole after radial subtraction.
    # The exact weighted mean is removed after constructing the window below.

    window = build_apodized_window(mask, args.taper_pixels)
    sw = np.sum(window)
    if sw <= 0:
        raise RuntimeError("Apodized window contains no non-zero pixels.")

    raw_wmean = np.sum(window * delta_raw) / sw
    det_wmean = np.sum(window * delta_det) / sw
    delta_raw[mask] -= raw_wmean
    delta_det[mask] -= det_wmean

    P2_raw, kmag, kx, ky, var_raw_real, var_raw_spec = power2d(delta_raw, window, dx)
    P2_det, _, _, _, var_det_real, var_det_spec = power2d(delta_det, window, dx)

    k_raw, p_raw, nm_raw = annular_average(P2_raw, kmag, kfund, knyq)
    k_det, p_det, nm_det = annular_average(P2_det, kmag, kfund, knyq)

    if len(k_raw) != len(k_det) or not np.allclose(k_raw, k_det):
        raise RuntimeError("Raw/detrended radial k bins do not match.")

    P2_win, k_win, p_win, win_shape, nm_win = window_spectrum(
        window, dx, kmag, kfund, knyq
    )

    k = k_raw
    k2p_raw = k ** 2 * p_raw
    k2p_det = k ** 2 * p_det

    # Conservative peak: avoid sparsely sampled bins and the last 20% toward Nyquist.
    good_peak = (nm_raw >= 8) & (k <= args.max_k_fraction * knyq)
    if np.any(good_peak):
        inds = np.where(good_peak)[0]
        iraw = inds[np.nanargmax(k2p_raw[inds])]
        idet = inds[np.nanargmax(k2p_det[inds])]
    else:
        iraw = int(np.nanargmax(k2p_raw))
        idet = int(np.nanargmax(k2p_det))

    lam_raw = 2.0 * np.pi / k[iraw]
    lam_det = 2.0 * np.pi / k[idet]

    prefix = Path(args.prefix)

    # ----- CSV -----
    csv_path = Path(str(prefix) + "_spectrum.csv")
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "k_pkpc^-1", "lambda_pkpc",
            "P_raw_Rv2_pkpc2", "k2P_raw_Rv2",
            "P_detrended_Rv2_pkpc2", "k2P_detrended_Rv2",
            "window_k2P_normalized", "N_modes"
        ])
        for i in range(len(k)):
            ws = win_shape[i] if i < len(win_shape) else np.nan
            w.writerow([
                k[i], 2.0 * np.pi / k[i],
                p_raw[i], k2p_raw[i],
                p_det[i], k2p_det[i],
                ws, nm_raw[i]
            ])

    # ----- NPZ -----
    npz_path = Path(str(prefix) + "_spectrum.npz")
    np.savez_compressed(
        npz_path,
        rv=rv,
        rv_mask=mask,
        window=window,
        radial_trend=trend,
        delta_rv_raw=delta_raw,
        delta_rv_detrended=delta_det,
        P2d_raw=P2_raw,
        P2d_detrended=P2_det,
        P2d_window=P2_win,
        k_pkpc_inv=k,
        lambda_pkpc=2.0 * np.pi / k,
        P_raw=p_raw,
        k2P_raw=k2p_raw,
        P_detrended=p_det,
        k2P_detrended=k2p_det,
        window_k2P_normalized=win_shape,
        n_modes=nm_raw,
        radial_centers_pkpc=rcent,
        radial_profile_rv=rprof,
        radial_profile_n=rn,
        pixel_size_pkpc=dx,
        map_width_pkpc=L,
        k_fund_pkpc_inv=kfund,
        k_nyq_pkpc_inv=knyq,
        mean_rv=mean_rv,
        valid_pixels=valid,
        taper_pixels=args.taper_pixels,
        peak_k_raw=k[iraw],
        peak_lambda_raw_pkpc=lam_raw,
        peak_k_detrended=k[idet],
        peak_lambda_detrended_pkpc=lam_det,
    )

    extent = [
        centers[0] - 0.5 * dx, centers[-1] + 0.5 * dx,
        centers[0] - 0.5 * dx, centers[-1] + 0.5 * dx
    ]

    # ----- Individual diagnostic figures -----
    rv_plot = np.where(mask, rv, np.nan)
    save_map(
        str(prefix) + "_rv_map.png", rv_plot, extent,
        "Valid face-on $R_V$ field", "$R_V$"
    )
    save_map(
        str(prefix) + "_window.png", window, extent,
        f"Apodized validity window ({args.taper_pixels:g}-pixel taper)",
        "W"
    )
    save_map(
        str(prefix) + "_delta_detrended.png",
        np.where(mask, delta_det, np.nan), extent,
        r"Radial-detrended $\delta R_V$", r"$\delta R_V$"
    )

    # 2-D Fourier power, raw.
    pshift = np.fft.fftshift(P2_raw)
    kxshift = np.fft.fftshift(kx)
    kyshift = np.fft.fftshift(ky)
    eps = np.nanmax(pshift) * 1e-12 if np.nanmax(pshift) > 0 else 1e-30
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(
        np.log10(pshift + eps),
        origin="lower",
        extent=[kxshift[0], kxshift[-1], kyshift[0], kyshift[-1]],
        aspect="equal",
        interpolation="nearest"
    )
    ax.set_xlabel(r"$k_x$ [pkpc$^{-1}$]")
    ax.set_ylabel(r"$k_y$ [pkpc$^{-1}$]")
    ax.set_title(r"2-D power: total $\delta R_V$")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(r"$\log_{10} P(k_x,k_y)$")
    fig.tight_layout()
    fig.savefig(str(prefix) + "_power2d_raw.png", dpi=220)
    plt.close(fig)

    # Main science spectrum.
    fig, ax = plt.subplots(figsize=(8.0, 5.8))
    ax.loglog(k, k2p_raw, marker="o", label=r"Total: $R_V-\langle R_V\rangle$")
    ax.loglog(k, k2p_det, marker="s",
              label=r"Radial detrended: $R_V-\langle R_V\rangle_r$")
    ax.axvline(kfund, linestyle=":", linewidth=1.2, label=r"$k_{\rm fund}$")
    ax.axvline(knyq, linestyle="--", linewidth=1.2, label=r"$k_{\rm Nyq}$")
    ax.scatter([k[iraw]], [k2p_raw[iraw]], s=55, zorder=5)
    ax.scatter([k[idet]], [k2p_det[idet]], s=55, zorder=5)
    ax.set_xlim(kfund * 0.9, knyq * 1.05)
    ax.set_xlabel(r"$k$ [pkpc$^{-1}$]")
    ax.set_ylabel(r"$k^2 P_{R_V}(k)$ [$R_V^2$]")
    ztxt = "" if z is None else f", z={z:.3f}"
    ax.set_title(r"$R_V$ spatial power" + ztxt)
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=9)

    def k_to_lam(x):
        x = np.asarray(x)
        return 2.0 * np.pi / x

    def lam_to_k(x):
        x = np.asarray(x)
        return 2.0 * np.pi / x

    sec = ax.secondary_xaxis("top", functions=(k_to_lam, lam_to_k))
    sec.set_xlabel(r"Spatial wavelength $\lambda=2\pi/k$ [pkpc]")
    fig.tight_layout()
    fig.savefig(str(prefix) + "_k2Pk.png", dpi=240)
    fig.savefig(str(prefix) + "_k2Pk.pdf")
    plt.close(fig)

    # Window spectral shape -- deliberately separate because its amplitude is arbitrary.
    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    ax.semilogx(k_win, win_shape, marker="o")
    ax.axvline(kfund, linestyle=":", linewidth=1.2)
    ax.axvline(knyq, linestyle="--", linewidth=1.2)
    ax.set_xlim(kfund * 0.9, knyq * 1.05)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r"$k$ [pkpc$^{-1}$]")
    ax.set_ylabel(r"Normalized $k^2P_W(k)$")
    ax.set_title("Validity-window spectral shape")
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(str(prefix) + "_window_spectrum.png", dpi=220)
    plt.close(fig)

    # Radial trend diagnostic.
    goodr = np.isfinite(rprof)
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.plot(rcent[goodr], rprof[goodr], marker="o")
    ax.set_xlabel("Projected radius [pkpc]")
    ax.set_ylabel(r"$\langle R_V\rangle_r$")
    ax.set_title("Radial trend removed for patchiness spectrum")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(str(prefix) + "_radial_profile.png", dpi=220)
    plt.close(fig)

    print("=" * 72)
    print("R(V) POWER-SPECTRUM ANALYSIS")
    print("=" * 72)
    print(f"Input                  : {args.maps_npz}")
    print(f"Map                    : {n} x {n} pixels")
    print(f"Pixel size             : {dx:.6f} pkpc")
    print(f"Map width              : {L:.6f} pkpc")
    print(f"Valid pixels           : {valid:,} / {n*n:,} ({100*valid/(n*n):.2f}%)")
    print(f"Mean R(V), valid       : {mean_rv:.6f}")
    print(f"Apodization            : {args.taper_pixels:g} pixels")
    print()
    print(f"k_fund                 : {kfund:.6f} pkpc^-1")
    print(f"k_Nyquist              : {knyq:.6f} pkpc^-1")
    print(f"Nominal wavelength     : {2*np.pi/kfund:.3f} -> {2*np.pi/knyq:.3f} pkpc")
    print()
    print("Parseval / normalization checks:")
    print(f"  raw:       real={var_raw_real:.8e}  spectral={var_raw_spec:.8e}  "
          f"ratio={var_raw_spec/var_raw_real if var_raw_real > 0 else np.nan:.8f}")
    print(f"  detrended: real={var_det_real:.8e}  spectral={var_det_spec:.8e}  "
          f"ratio={var_det_spec/var_det_real if var_det_real > 0 else np.nan:.8f}")
    print()
    print(f"Conservative raw peak       : k={k[iraw]:.6f} pkpc^-1  "
          f"lambda={lam_raw:.3f} pkpc  Nmodes={nm_raw[iraw]}")
    print(f"Conservative detrended peak : k={k[idet]:.6f} pkpc^-1  "
          f"lambda={lam_det:.3f} pkpc  Nmodes={nm_det[idet]}")
    print()
    print("NOTE: these are apodized pseudo-spectra. <W^2> corrects the overall")
    print("      amplitude suppression, but mask-induced Fourier mode mixing is")
    print("      not deconvolved. Use the saved window spectrum as a diagnostic.")
    print()
    print("Saved:")
    for p in [
        csv_path, npz_path,
        Path(str(prefix) + "_rv_map.png"),
        Path(str(prefix) + "_window.png"),
        Path(str(prefix) + "_delta_detrended.png"),
        Path(str(prefix) + "_power2d_raw.png"),
        Path(str(prefix) + "_k2Pk.png"),
        Path(str(prefix) + "_k2Pk.pdf"),
        Path(str(prefix) + "_window_spectrum.png"),
        Path(str(prefix) + "_radial_profile.png"),
    ]:
        print(f"  {p}")


if __name__ == "__main__":
    main()
