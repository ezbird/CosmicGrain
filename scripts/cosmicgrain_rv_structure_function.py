#!/usr/bin/env python3
"""
CosmicGrain R(V) second-order structure function.

Computes
    S2(r) = < [R_V(x+r) - R_V(x)]^2 >
using ONLY pairs for which both pixels have valid R(V). Missing pixels are
never filled with zero and there is no FFT window.

The script starts from the native tau_B/tau_V maps and repeats the calculation
at requested map pixel sizes. R(V) is always recomputed after optical-depth
rebinning.

Outputs:
  *_structure_function.png/.pdf
  *_structure_function.csv

The CSV contains separation, S2, sqrt(S2), pair count, and valid-pixel
statistics for every map pixel size.
"""

import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt

def block_sum(a, f):
    ny, nx = a.shape
    if ny % f or nx % f:
        raise ValueError(f"shape {a.shape} not divisible by factor {f}")
    return a.reshape(ny//f, f, nx//f, f).sum((1,3))

def make_rv(tb, tv):
    d = tb - tv
    rv = np.full_like(tv, np.nan, dtype=float)
    q = np.isfinite(tb) & np.isfinite(tv) & (tv > 0) & (d > 0)
    rv[q] = tv[q] / d[q]
    return rv

def pixlabel(dx):
    return f"{dx*1000:.0f} pc" if dx < 1 else f"{dx:g} kpc"

def structure_function(rv, mask, dx, rmin=None, rmax=None, nbins=22):
    """
    Exact pair accumulation by integer pixel displacement.

    Each unordered pair is counted exactly once by using dy>0 plus dy=0,dx>0.
    Pairs are binned by their true 2-D separation sqrt(dx^2+dy^2)*pixel_size.
    """
    ny, nx = rv.shape
    if rmin is None:
        rmin = dx
    if rmax is None:
        rmax = 0.5 * min(nx*dx, ny*dx)

    edges = np.logspace(np.log10(rmin), np.log10(rmax), nbins+1)
    sum_d2 = np.zeros(nbins, dtype=float)
    sum_d4 = np.zeros(nbins, dtype=float)
    npair = np.zeros(nbins, dtype=np.int64)
    sum_r = np.zeros(nbins, dtype=float)

    max_sy = min(ny-1, int(np.floor(rmax/dx)))
    max_sx = min(nx-1, int(np.floor(rmax/dx)))

    for sy in range(0, max_sy+1):
        sx0 = 1 if sy == 0 else -max_sx
        for sx in range(sx0, max_sx+1):
            if sy == 0 and sx <= 0:
                continue
            rr = np.hypot(sx, sy) * dx
            if rr < rmin or rr > rmax:
                continue
            b = np.searchsorted(edges, rr, side="right") - 1
            if b < 0 or b >= nbins:
                continue

            # slices for A and shifted B, allowing positive/negative sx
            if sx >= 0:
                xa0, xa1 = 0, nx-sx
                xb0, xb1 = sx, nx
            else:
                xa0, xa1 = -sx, nx
                xb0, xb1 = 0, nx+sx

            ya0, ya1 = 0, ny-sy
            yb0, yb1 = sy, ny

            ma = mask[ya0:ya1, xa0:xa1]
            mb = mask[yb0:yb1, xb0:xb1]
            q = ma & mb
            n = int(q.sum())
            if n == 0:
                continue

            a = rv[ya0:ya1, xa0:xa1][q]
            bb = rv[yb0:yb1, xb0:xb1][q]
            d2 = (bb-a)**2

            npair[b] += n
            sum_d2[b] += d2.sum()
            sum_d4[b] += np.square(d2).sum()
            sum_r[b] += rr*n

    good = npair > 0
    r = np.full(nbins, np.nan)
    s2 = np.full(nbins, np.nan)
    err = np.full(nbins, np.nan)
    r[good] = sum_r[good] / npair[good]
    s2[good] = sum_d2[good] / npair[good]

    # Scatter-based standard error of pair contributions. Pairs are not fully
    # independent, so treat this as a diagnostic rather than a formal error bar.
    g2 = npair > 1
    var = np.zeros(nbins)
    var[g2] = (sum_d4[g2] / npair[g2] - s2[g2]**2)
    var[var < 0] = 0
    err[g2] = np.sqrt(var[g2] / npair[g2])

    return r, s2, err, npair, edges

ap = argparse.ArgumentParser()
ap.add_argument("npz")
ap.add_argument("--pixel-sizes", nargs="+", type=float,
                default=[0.125,0.25,0.5,1.0,2.0])
ap.add_argument("--signal-frac", type=float, default=1e-3)
ap.add_argument("--square-size", type=float, default=None,
                help="optional centered square side length in kpc; default full map")
ap.add_argument("--rmin", type=float, default=None)
ap.add_argument("--rmax", type=float, default=None)
ap.add_argument("--nbins", type=int, default=22)
ap.add_argument("--min-pairs", type=int, default=100)
ap.add_argument("--prefix", default="halo441_rv_structure_function")
a = ap.parse_args()

z = np.load(a.npz)
tb0 = np.asarray(z["tau_b"], float)
tv0 = np.asarray(z["tau_v"], float)
nc0 = np.asarray(z["dust_count"], int)
dx0 = float(np.asarray(z["pixel_size_pkpc"]).ravel()[0])

if tb0.shape[0] != tb0.shape[1]:
    raise ValueError("expected square native map")

# Optional exact centered physical crop.
if a.square_size is not None:
    N = tb0.shape[0]
    nn = int(round(a.square_size/dx0))
    if not np.isclose(nn*dx0, a.square_size):
        raise ValueError("square size does not align with native grid")
    if nn > N:
        raise ValueError("square exceeds native map")
    lo = (N-nn)//2
    hi = lo+nn
    tb0 = tb0[lo:hi,lo:hi]
    tv0 = tv0[lo:hi,lo:hi]
    nc0 = nc0[lo:hi,lo:hi]

L = tb0.shape[0]*dx0
rmax = a.rmax if a.rmax is not None else 0.5*L

print("="*94)
print("COSMICGRAIN R(V) REAL-SPACE STRUCTURE FUNCTION")
print("="*94)
print(f"region: {L:g} x {L:g} kpc")
print(f"native dx: {dx0:g} kpc")
print(f"S2(r) = <[R_V(x+r)-R_V(x)]^2>, valid pairs only")
print(f"maximum separation analyzed: {rmax:g} kpc")

allres = []
rows = []

for dx in a.pixel_sizes:
    f = int(round(dx/dx0))
    if not np.isclose(f*dx0, dx):
        raise ValueError(f"{dx:g} kpc is not an integer multiple of native dx")
    if tb0.shape[0] % f:
        raise ValueError(f"region is not divisible by factor {f}")

    tb = block_sum(tb0, f)
    tv = block_sum(tv0, f)
    nc = block_sum(nc0, f)
    rv = make_rv(tb, tv)

    # Same natural measurement criterion used in the power-spectrum work.
    mask = np.isfinite(rv) & (nc >= 1) & (tv >= a.signal_frac*np.nanmax(tv))
    vals = rv[mask]

    this_rmin = a.rmin if a.rmin is not None else dx
    r, s2, err, npair, edges = structure_function(
        rv, mask, dx, this_rmin, rmax, a.nbins
    )

    use = np.isfinite(r) & np.isfinite(s2) & (npair >= a.min_pairs)
    allres.append((dx,r,s2,err,npair,use))

    print(f"\n{pixlabel(dx)}")
    print(f"  grid          : {rv.shape[0]} x {rv.shape[1]}")
    print(f"  valid pixels  : {mask.sum()}/{mask.size} ({100*mask.mean():.2f}%)")
    print(f"  Rv p16/50/84 : {np.percentile(vals,[16,50,84])}")
    print(f"  Rv std        : {np.std(vals):.5f}")
    print(f"  SF bins kept  : {use.sum()}/{len(use)} (min pairs={a.min_pairs})")
    if use.any():
        print(f"  pair counts   : {npair[use].min():,} ... {npair[use].max():,}")
        print(f"  r range kept  : {r[use].min():.3f} ... {r[use].max():.3f} kpc")

    for i in range(len(r)):
        if np.isfinite(r[i]) and np.isfinite(s2[i]):
            rows.append({
                "pixel_size_kpc":dx,
                "valid_pixels":int(mask.sum()),
                "total_pixels":int(mask.size),
                "valid_fraction":float(mask.mean()),
                "r_kpc":float(r[i]),
                "S2_Rv2":float(s2[i]),
                "sqrt_S2_Rv":float(np.sqrt(max(s2[i],0))),
                "S2_pair_scatter_SE":float(err[i]) if np.isfinite(err[i]) else np.nan,
                "N_pairs":int(npair[i]),
                "passes_min_pairs":int(npair[i] >= a.min_pairs)
            })

# Two panels: S2 and sqrt(S2), same pair calculation.
fig, axs = plt.subplots(1,2,figsize=(11.5,4.8))

for dx,r,s2,err,npair,use in allres:
    if use.any():
        axs[0].loglog(r[use],s2[use],lw=1.8,label=pixlabel(dx))
        axs[1].loglog(r[use],np.sqrt(s2[use]),lw=1.8,label=pixlabel(dx))

axs[0].set_title(r"Second-order structure function")
axs[0].set_xlabel(r"Separation $r$ (kpc)")
axs[0].set_ylabel(r"$S_2(r)$")
axs[1].set_title(r"RMS $R(V)$ difference")
axs[1].set_xlabel(r"Separation $r$ (kpc)")
axs[1].set_ylabel(r"$\sqrt{S_2(r)}$")

for ax in axs:
    ax.grid(True,which="both",alpha=.18)
    ax.tick_params(direction="in",which="both",top=True,right=True)

region_title = f"{L:g} x {L:g} kpc" if a.square_size is not None else f"full {L:g} x {L:g} kpc map"
fig.suptitle(f"halo441 R(V) spatial structure - {region_title}",fontsize=15)
H,Lb = axs[0].get_legend_handles_labels()
fig.legend(H,Lb,loc="lower center",ncol=len(Lb),frameon=False)
fig.tight_layout(rect=(0,.08,1,.94))
fig.savefig(a.prefix+"_structure_function.png",dpi=300,bbox_inches="tight")
fig.savefig(a.prefix+"_structure_function.pdf",bbox_inches="tight")
plt.close(fig)

csvname = a.prefix+"_structure_function.csv"
with open(csvname,"w",newline="") as fp:
    fields=["pixel_size_kpc","valid_pixels","total_pixels","valid_fraction",
            "r_kpc","S2_Rv2","sqrt_S2_Rv","S2_pair_scatter_SE",
            "N_pairs","passes_min_pairs"]
    w=csv.DictWriter(fp,fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

print("\nSaved:")
print(" ",a.prefix+"_structure_function.png")
print(" ",a.prefix+"_structure_function.pdf")
print(" ",csvname)
print("\nMissing pixels were never set to zero; only valid-valid pixel pairs contribute.")
