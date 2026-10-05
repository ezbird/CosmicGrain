#!/usr/bin/env python3
"""
cosmicgrain_morphology_triptych_v2.py

Actual-particle morphology diagnostic:
    Stars face-on | Gas face-on | Stars edge-on

Corrections relative to v1:
  * Reads HubbleParam from Parameters first (Header fallback).
  * No silent h=1 fallback: aborts if HubbleParam cannot be found.
  * Separate stellar and gas map pixel sizes (defaults 0.25 and 0.50 pkpc).
  * Correct ckpc/h -> physical kpc and 1e10 Msun/h -> Msun conversions.
  * Reports 30-pkpc masses/counts for direct comparison with the audit.
  * Uses the same stellar-angular-momentum orientation for all three panels.
  * No artificial image smoothing: maps are mass-weighted 2-D histograms.
"""

import argparse, glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def scalar(x):
    a = np.asarray(x)
    return float(a.reshape(-1)[0])


def header_info(fn):
    with h5py.File(fn, "r") as f:
        H = f["Header"].attrs
        a = scalar(H["Time"])
        box = scalar(H["BoxSize"])

        h = None
        h_source = None
        if "Parameters" in f and "HubbleParam" in f["Parameters"].attrs:
            h = scalar(f["Parameters"].attrs["HubbleParam"])
            h_source = "Parameters/HubbleParam"
        elif "HubbleParam" in H:
            h = scalar(H["HubbleParam"])
            h_source = "Header/HubbleParam"

        if h is None or not np.isfinite(h) or h <= 0:
            raise RuntimeError(
                "Could not find a valid HubbleParam in Parameters or Header. "
                "Refusing to assume h=1."
            )
    return a, 1.0/a - 1.0, box, h, h_source


def load(files, ptype, fields):
    out = {k: [] for k in fields}
    for fn in files:
        with h5py.File(fn, "r") as f:
            name = f"PartType{ptype}"
            if name not in f:
                continue
            g = f[name]
            n = len(g["Coordinates"])
            for k in fields:
                if k in g:
                    out[k].append(np.asarray(g[k]))
                elif k == "Masses":
                    mt = np.asarray(f["Header"].attrs["MassTable"])
                    if mt[ptype] <= 0:
                        raise KeyError(f"{name}/Masses missing and MassTable[{ptype}] <= 0")
                    out[k].append(np.full(n, mt[ptype], dtype=float))
                else:
                    raise KeyError(f"{name}/{k} missing in {fn}")
    if not out[fields[0]]:
        raise RuntimeError(f"No PartType{ptype} particles found")
    return {k: np.concatenate(v, axis=0) for k, v in out.items()}


def periodic_delta(x, c, L):
    d = np.asarray(x, float) - np.asarray(c, float)
    d -= L * np.rint(d / L)
    return d


def faceon_rotation(j):
    zhat = np.asarray(j, float)
    zhat /= np.linalg.norm(zhat)
    ref = np.array([0., 0., 1.])
    if abs(np.dot(ref, zhat)) > 0.9:
        ref = np.array([0., 1., 0.])
    xhat = np.cross(ref, zhat)
    xhat /= np.linalg.norm(xhat)
    yhat = np.cross(zhat, xhat)
    yhat /= np.linalg.norm(yhat)
    return np.vstack((xhat, yhat, zhat))


def hist_surface_density(x, y, mass, xlim, ylim, pix):
    nx = int(round((xlim[1]-xlim[0])/pix))
    ny = int(round((ylim[1]-ylim[0])/pix))
    H, _, _ = np.histogram2d(
        y, x, bins=(ny, nx), range=(ylim, xlim), weights=mass
    )
    dx = (xlim[1]-xlim[0])/nx
    dy = (ylim[1]-ylim[0])/ny
    return H/(dx*dy), dx, dy


def display_limits(M, low=2.0, high=99.7):
    q = M[np.isfinite(M) & (M > 0)]
    if q.size == 0:
        return 1., 10.
    lo, hi = np.percentile(q, [low, high])
    lo = max(lo, q.min())
    hi = max(hi, lo*1.01)
    return lo, hi


def main():
    p = argparse.ArgumentParser()
    p.add_argument("snapshot")
    p.add_argument("--center", nargs=3, type=float, required=True,
                   metavar=("X","Y","Z"), help="center in ckpc/h")
    p.add_argument("--radius", type=float, default=25.,
                   help="face-on half-width (physical kpc)")
    p.add_argument("--edge-z", type=float, default=10.,
                   help="edge-on vertical half-height (physical kpc)")
    p.add_argument("--orientation-radius", type=float, default=20.,
                   help="stellar angular momentum aperture (physical kpc)")
    p.add_argument("--star-pixel-size", type=float, default=0.25,
                   help="stellar map pixel size (physical kpc)")
    p.add_argument("--gas-pixel-size", type=float, default=0.50,
                   help="gas map pixel size (physical kpc)")
    p.add_argument("--audit-radius", type=float, default=30.,
                   help="3-D physical radius for audit comparison")
    p.add_argument("--prefix", default="morphology")
    p.add_argument("--dpi", type=int, default=250)
    args = p.parse_args()

    files = sorted(glob.glob(args.snapshot))
    if not files:
        raise FileNotFoundError(f"No files match {args.snapshot}")

    a, z, box, h, hsrc = header_info(files[0])
    c = np.asarray(args.center, float)

    stars = load(files, 4, ["Coordinates", "Velocities", "Masses"])
    gas   = load(files, 0, ["Coordinates", "Masses"])

    # GADGET coordinates: comoving ckpc/h -> physical kpc.
    rs = periodic_delta(stars["Coordinates"], c, box) * a / h
    rg = periodic_delta(gas["Coordinates"], c, box) * a / h

    # GADGET masses: 1e10 Msun/h -> Msun.
    ms = stars["Masses"].astype(float) * 1.0e10 / h
    mg = gas["Masses"].astype(float) * 1.0e10 / h

    rstar = np.linalg.norm(rs, axis=1)
    rgas  = np.linalg.norm(rg, axis=1)

    ori = rstar < args.orientation_radius
    if ori.sum() < 10:
        raise RuntimeError("Too few stars in orientation aperture")

    v = stars["Velocities"].astype(float)
    vbulk = np.average(v[ori], axis=0, weights=ms[ori])
    J = np.sum(np.cross(rs[ori], v[ori]-vbulk) * ms[ori,None], axis=0)
    if not np.isfinite(J).all() or np.linalg.norm(J) == 0:
        raise RuntimeError("Invalid stellar angular momentum")
    jhat = J/np.linalg.norm(J)
    R = faceon_rotation(jhat)

    sr = rs @ R.T
    gr = rg @ R.T

    rad = args.radius
    sf = ((abs(sr[:,0]) < rad) & (abs(sr[:,1]) < rad) & (abs(sr[:,2]) < rad))
    gf = ((abs(gr[:,0]) < rad) & (abs(gr[:,1]) < rad) & (abs(gr[:,2]) < rad))
    se = ((abs(sr[:,0]) < rad) & (abs(sr[:,1]) < rad) & (abs(sr[:,2]) < args.edge_z))

    S, sdx, _ = hist_surface_density(
        sr[sf,0], sr[sf,1], ms[sf], (-rad,rad), (-rad,rad), args.star_pixel_size
    )
    G, gdx, _ = hist_surface_density(
        gr[gf,0], gr[gf,1], mg[gf], (-rad,rad), (-rad,rad), args.gas_pixel_size
    )
    E, edx, _ = hist_surface_density(
        sr[se,0], sr[se,2], ms[se], (-rad,rad), (-args.edge_z,args.edge_z),
        args.star_pixel_size
    )

    sa = rstar < args.audit_radius
    ga = rgas  < args.audit_radius

    print("="*78)
    print("COSMICGRAIN MORPHOLOGY TRIPTYCH v2")
    print("="*78)
    print(f"a={a:.8f}, z={z:.8f}, h={h:.8f}  [{hsrc}]")
    print(f"center = {c} ckpc/h")
    print(f"orientation stars (<{args.orientation_radius:g} pkpc): {ori.sum():,}")
    print("stellar bulk velocity:", vbulk)
    print("j_hat:", jhat)
    print("rotation matrix:\n", R)
    print()
    print(f"3-D audit aperture <{args.audit_radius:g} pkpc:")
    print(f"  stars: N={sa.sum():,}, M={ms[sa].sum():.6e} Msun")
    print(f"  gas:   N={ga.sum():,}, M={mg[ga].sum():.6e} Msun")
    print()
    print(f"plotted stellar face-on volume: N={sf.sum():,}, M={ms[sf].sum():.6e} Msun")
    print(f"plotted gas face-on volume:     N={gf.sum():,}, M={mg[gf].sum():.6e} Msun")
    print(f"stellar pixel size = {sdx:.3f} pkpc")
    print(f"gas pixel size     = {gdx:.3f} pkpc")

    fig, axs = plt.subplots(1, 3, figsize=(17.5, 6.0))
    entries = [
        (S, (-rad,rad), (-rad,rad), "magma",
         "Stars — Face-on", r"$\Sigma_\star$ ($M_\odot$ kpc$^{-2}$)"),
        (G, (-rad,rad), (-rad,rad), "viridis",
         "Gas — Face-on", r"$\Sigma_{\rm gas}$ ($M_\odot$ kpc$^{-2}$)"),
        (E, (-rad,rad), (-args.edge_z,args.edge_z), "magma",
         "Stars — Edge-on", r"$\Sigma_\star$ ($M_\odot$ kpc$^{-2}$)")
    ]

    for i,(M,xlim,ylim,cmap,title,cblab) in enumerate(entries):
        lo,hi = display_limits(M)
        im = axs[i].imshow(
            np.ma.masked_less_equal(M,0), origin="lower",
            extent=[xlim[0],xlim[1],ylim[0],ylim[1]],
            cmap=cmap, norm=LogNorm(vmin=lo,vmax=hi),
            interpolation="nearest", aspect="equal"
        )
        axs[i].set_title(title)
        axs[i].set_xlabel(r"$x'$ (kpc)")
        axs[i].set_ylabel(r"$y'$ (kpc)" if i < 2 else r"$z'$ (kpc)")
        cb = fig.colorbar(im, ax=axs[i], orientation="horizontal",
                          pad=0.13, fraction=0.055)
        cb.set_label(cblab)

    fig.tight_layout()
    png = args.prefix+"_triptych.png"
    pdf = args.prefix+"_triptych.pdf"
    npz = args.prefix+"_maps.npz"
    fig.savefig(png, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    np.savez_compressed(
        npz, stellar_faceon=S, gas_faceon=G, stellar_edgeon=E,
        star_pixel_size_pkpc=sdx, gas_pixel_size_pkpc=gdx,
        face_radius_pkpc=rad, edge_z_pkpc=args.edge_z,
        center_ckpch=c, j_hat=jhat, rotation_matrix=R,
        stellar_bulk_velocity=vbulk, a=a, z=z, h=h, boxsize_ckpch=box
    )
    print("\nSaved:")
    print(" ", png)
    print(" ", pdf)
    print(" ", npz)

if __name__ == "__main__":
    main()
