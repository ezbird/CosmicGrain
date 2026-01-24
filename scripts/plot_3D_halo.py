#!/usr/bin/env python3
import os, glob, h5py, numpy as np
import matplotlib
matplotlib.use("Agg")              # headless-safe
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# -------------------- appearance per type (tweak freely) -------------------- #
PART_STYLE = {
    "PartType0": {"name": "Gas",           "color": "tab:blue",  "size": 0.2, "alpha": 0.25},
    "PartType1": {"name": "DM (HR)",       "color": "black",     "size": 0.2, "alpha": 0.15},
    "PartType2": {"name": "DM (LR shell)", "color": "0.55",      "size": 0.15,"alpha": 0.10},
    "PartType3": {"name": "DM (LR shell)", "color": "0.70",      "size": 0.15,"alpha": 0.10},
    "PartType4": {"name": "Stars",         "color": "tab:red",   "size": 1.0, "alpha": 0.85},
    "PartType5": {"name": "BH",            "color": "purple",    "size": 1.5, "alpha": 0.90},
    "PartType6": {"name": "Dust",          "color": "tab:green", "size": 0.8, "alpha": 0.85},
}

ALIAS = {
    "gas":"PartType0","dm":"PartType1","stars":"PartType4","bh":"PartType5","dust":"PartType6",
    "pt0":"PartType0","pt1":"PartType1","pt2":"PartType2","pt3":"PartType3","pt4":"PartType4","pt5":"PartType5","pt6":"PartType6"
}

CM_PER_KPC = 3.085678e21
CM_PER_MPC = 3.085678e24

# ----------------------------- helpers --------------------------------- #
def parse_types_list(types_list):
    """Accepts: gas dm stars, PartType0 PartType1, 0,1,4,6, or 'all'."""
    if not types_list:
        return ["PartType0","PartType1","PartType4"]  # default: gas+DM+stars
    toks = []
    for t in types_list:
        toks.extend(t.replace(",", " ").split())
    if any(t.lower() in ("all","*") for t in toks):
        return list(PART_STYLE.keys())
    out, seen = [], set()
    for t in toks:
        key = t.strip()
        if key.isdigit():
            pt = f"PartType{key}"
        else:
            pt = ALIAS.get(key.lower(), key if key.startswith("PartType") else None)
        if pt in PART_STYLE and pt not in seen:
            out.append(pt); seen.add(pt)
    if not out:
        print("WARNING: --types matched nothing; using default gas+dm+stars")
        return ["PartType0","PartType1","PartType4"]
    return out

def unit_scale_and_label(header, units="auto"):
    """Return scale so x_plot = x_code*scale, and text label 'kpc'/'Mpc'/'code'."""
    ul = float(header.get("UnitLength_in_cm", 0.0))
    if units == "kpc":
        scale = (ul/CM_PER_KPC) if ul>0 else 1.0
        return scale, "kpc"
    if units == "mpc":
        scale = (ul/CM_PER_MPC) if ul>0 else 1.0
        return scale, "Mpc"
    if units == "code" or ul == 0.0:
        return 1.0, "code"
    # auto: pick whichever is closer
    if abs(ul-CM_PER_KPC) < abs(ul-CM_PER_MPC):
        return ul/CM_PER_KPC, "kpc"
    return ul/CM_PER_MPC, "Mpc"

def wrap_min_image_1d(x, center, box):
    y = x - center
    y -= np.round(y/box)*box
    return y

def apply_periodic_roi(coords, center, width, box):
    """Return coords wrapped so that center is in the middle, and mask inside cube of width^3."""
    c = np.asarray(center, float)
    shifted = coords - c
    shifted -= np.round(shifted/box)*box
    half = 0.5*width
    m = (np.abs(shifted[:,0])<=half) & (np.abs(shifted[:,1])<=half) & (np.abs(shifted[:,2])<=half)
    return (shifted[m] + c), m  # return back in original coordinate frame (centered ROI)

def read_singlefile(fname, ptypes):
    with h5py.File(fname, "r") as f:
        header = dict(f["Header"].attrs)
        data = {}
        for pt in ptypes:
            if pt in f:
                data[pt] = f[pt]["Coordinates"][:]
    return data, header

def read_multifile_dir(snapdir, ptypes):
    files = sorted(glob.glob(os.path.join(snapdir, "*.hdf5")))
    if not files: return None, None
    acc = {pt:[] for pt in ptypes}
    header = None
    for fp in files:
        with h5py.File(fp, "r") as f:
            if header is None: header = dict(f["Header"].attrs)
            for pt in ptypes:
                if pt in f:
                    acc[pt].append(f[pt]["Coordinates"][:])
    data = {pt: np.concatenate(v, axis=0) for pt,v in acc.items() if v}
    return data, header

# ------------------------------ plotting -------------------------------- #
def plot_3d(data_scaled, header, box_plot, unit_label, out_png,
            elev=25, azim=45, ortho=False, legend=True, bg="white",
            counts_note=True, title_extra=""):
    fig = plt.figure(figsize=(7.8,7.6), dpi=140)
    ax = fig.add_subplot(111, projection="3d")
    if ortho and hasattr(ax, "set_proj_type"):
        ax.set_proj_type("ortho")

    # aesthetics
    ax.set_facecolor(bg)
    fig.patch.set_facecolor(bg)
    for spine in ax.w_xaxis.get_ticklines()+ax.w_yaxis.get_ticklines()+ax.w_zaxis.get_ticklines():
        spine.set_markeredgewidth(0.6)
    # draw each requested type
    handles = []
    for pt, xyz in data_scaled.items():
        st = PART_STYLE[pt]
        h = ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2],
                       s=st["size"], c=st["color"], alpha=st["alpha"],
                       depthshade=True, linewidths=0)
        handles.append((h, st["name"]))

    ax.set_xlim(0, box_plot); ax.set_ylim(0, box_plot); ax.set_zlim(0, box_plot)
    ax.set_xlabel(f"X ({unit_label})"); ax.set_ylabel(f"Y ({unit_label})"); ax.set_zlabel(f"Z ({unit_label})")
    ax.view_init(elev=elev, azim=azim)

    time = float(header.get("Time", 1.0))
    try:
        z = 1.0/time - 1.0
    except Exception:
        z = float(header.get("Redshift", 0.0))

    n_stars = data_scaled.get("PartType4", np.zeros((0,3))).shape[0]
    n_dust  = data_scaled.get("PartType6", np.zeros((0,3))).shape[0]

    title = f"3D particles  z={z:.2f}  box={box_plot:.3g} {unit_label}"
    if title_extra:
        title += f"  {title_extra}"
    ax.set_title(title, pad=12)

    if counts_note:
        txt = f"Stars: {n_stars:,}   Dust: {n_dust:,}"
        ax.text2D(0.02, 0.98, txt, transform=ax.transAxes, ha="left", va="top",
                  fontsize=9, color="0.1", bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))

    if legend and handles:
        ax.legend([h for h,_ in handles], [n for _,n in handles],
                  loc="upper left", bbox_to_anchor=(0.02,0.98), fontsize=8, frameon=False)

    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

def main():
    import argparse
    ap = argparse.ArgumentParser(description="3D scatter of Gadget-4 snapshot with selectable PartTypes")
    ap.add_argument("--path", required=True,
                    help="Path to single snapshot file (snapshot_###.hdf5) or to snapdir_### directory")
    ap.add_argument("--out", default="snapshot_3d.png", help="Output PNG (or prefix if --spin > 0)")
    ap.add_argument("--types", nargs="*", default=None,
                    help="Which particle types to plot (e.g., gas dm stars dust OR 0,1,4,6 OR PartTypeX). 'all' for everything.")
    ap.add_argument("--units", choices=["auto","kpc","mpc","code"], default="auto", help="Axis units (auto from header)")
    ap.add_argument("--sample", type=int, default=500000,
                    help="Random subsample per type (0 = plot all). 3D scatter gets slow above ~1e6 points.")
    ap.add_argument("--center", type=float, nargs=3, metavar=("X","Y","Z"), default=None,
                    help="Optional ROI center (same units as axes). Enables periodic crop if --width given.")
    ap.add_argument("--width", type=float, default=None,
                    help="Optional ROI cube width (same units as axes). If omitted -> plot full box.")
    ap.add_argument("--elev", type=float, default=25.0, help="Camera elevation (deg)")
    ap.add_argument("--azim", type=float, default=45.0, help="Camera azimuth (deg)")
    ap.add_argument("--ortho", action="store_true", help="Use orthographic projection")
    ap.add_argument("--spin", type=int, default=0,
                    help="If >0, save a spin of this many frames (PNG sequence: out_000.png, ...).")
    ap.add_argument("--bg", default="white", help="Background color (e.g. 'black')")
    args = ap.parse_args()

    # read snapshot
    ptypes = parse_types_list(args.types)
    if os.path.isdir(args.path):
        data, header = read_multifile_dir(args.path, ptypes)
    else:
        data, header = read_singlefile(args.path, ptypes)

    if not data:
        print("No requested PartTypes present — nothing to plot.")
        return

    scale, unit_label = unit_scale_and_label(header, args.units)
    box_plot = float(header["BoxSize"]) * scale

    # scale + optional ROI + subsample
    rng = np.random.default_rng(7)
    data_scaled = {}
    for pt, arr in data.items():
        arr = arr.astype(np.float64) * scale

        # optional ROI crop around center with periodic wrapping
        if args.center is not None and args.width:
            arr, _ = apply_periodic_roi(arr, np.array(args.center, float), args.width, box_plot)

        # subsample per type
        if args.sample and args.sample > 0 and arr.shape[0] > args.sample:
            idx = rng.choice(arr.shape[0], size=args.sample, replace=False)
            arr = arr[idx]

        if arr.size:
            data_scaled[pt] = arr

    if not data_scaled:
        print("After ROI/subsampling, nothing to plot.")
        return

    # single image or spin
    if args.spin <= 0:
        plot_3d(data_scaled, header, box_plot, unit_label, args.out,
                elev=args.elev, azim=args.azim, ortho=args.ortho, bg=args.bg)
        print(f"Saved {args.out}")
    else:
        base, ext = os.path.splitext(args.out)
        for i in range(args.spin):
            az = args.azim + 360.0 * i/args.spin
            out_png = f"{base}_{i:03d}.png"
            plot_3d(data_scaled, header, box_plot, unit_label, out_png,
                    elev=args.elev, azim=az, ortho=args.ortho, bg=args.bg,
                    title_extra=f"(view {i+1}/{args.spin})")
        print(f"Saved {args.spin} frames with prefix: {base}_###.png")

if __name__ == "__main__":
    main()
