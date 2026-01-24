#!/usr/bin/env python3
import argparse, h5py, numpy as np, matplotlib.pyplot as plt

# ---------- periodic wrap + ROI ----------
def wrap_min_image_1d(x, center, box):
    y = x - center; y -= np.round(y / box) * box; return y
def select_roi_2d(x, y, cx, cy, w, box):
    X = wrap_min_image_1d(x, cx, box); Y = wrap_min_image_1d(y, cy, box)
    half = 0.5*w; m = (np.abs(X) <= half) & (np.abs(Y) <= half)
    return m, X[m] + cx, Y[m] + cy

# ---------- I/O ----------
def load_parttype(f, ptype):
    g = f.get(f"PartType{ptype}")
    if g is None: return None, None
    pos = g["Coordinates"][:]
    if "Masses" in g: mass = g["Masses"][:]
    else:
        mt = f["Header"].attrs.get("MassTable")
        mass = None if (mt is None or mt[ptype]==0) else np.full(pos.shape[0], mt[ptype], np.float64)
    return pos, mass
def load_snapshot(fn):
    with h5py.File(fn, "r") as f:
        hdr = dict(f["Header"].attrs)
        unit_len = float(hdr.get("UnitLength_in_cm", 3.085678e24)) # Mpc default
        unit_mass= float(hdr.get("UnitMass_in_g", 1.989e43))       # 1e10 Msun default
        to_Mpc = unit_len/3.085678e24
        box = float(hdr["BoxSize"])*to_Mpc; z = float(hdr.get("Redshift",0.0))
        p1p, p1m = load_parttype(f,1); p2p, p2m = load_parttype(f,2)
        def pos_mpc(p): return None if p is None else p*to_Mpc
        def m_msun(m): 
            if m is None: return None
            return m*1e10 if (np.nanmax(m)<1e8) else m*(unit_mass/1.98847e33)
        return dict(
            p1_pos=pos_mpc(p1p), p1_mass=m_msun(p1m),
            p2_pos=pos_mpc(p2p), p2_mass=m_msun(p2m),
            box=box, z=z
        )

# ---------- deposition ----------
def _hist2d(x, y, w, x_edges, y_edges):
    H,_,_ = np.histogram2d(x, y, bins=[x_edges, y_edges], weights=w); return H.T
def _cic_deposit(x, y, w, x_edges, y_edges):
    nx = len(x_edges)-1; ny = len(y_edges)-1
    x0, x1 = x_edges[0], x_edges[-1]; y0, y1 = y_edges[0], y_edges[-1]
    gx = (x - x0) * nx / (x1 - x0); gy = (y - y0) * ny / (y1 - y0)
    # keep strictly inside grid
    eps = 1e-6
    m = (gx>=0)&(gx<nx-eps)&(gy>=0)&(gy<ny-eps)
    gx, gy, w = gx[m], gy[m], w[m]
    i0 = np.floor(gx).astype(int); j0 = np.floor(gy).astype(int)
    tx, ty = gx - i0, gy - j0; i1 = i0+1; j1 = j0+1
    grid = np.zeros((ny, nx), np.float64)
    np.add.at(grid, (j0, i0), w*(1-tx)*(1-ty))
    np.add.at(grid, (j0, i1), w*(   tx)*(1-ty))
    np.add.at(grid, (j1, i0), w*(1-tx)*(   ty))
    np.add.at(grid, (j1, i1), w*(   tx)*(   ty))
    return grid
def box_smooth(grid, passes=1):
    if passes<=0: return grid
    g = grid.copy()
    for _ in range(passes):
        pad = np.pad(g, 1, mode="edge")
        g = (pad[:-2, :-2]+pad[:-2,1:-1]+pad[:-2,2:]+
             pad[1:-1, :-2]+pad[1:-1,1:-1]+pad[1:-1,2:]+
             pad[2:,  :-2]+pad[2:, 1:-1]+pad[2:, 2:]) / 9.0
    return g

# ---------- slicing ----------
def make_slice_for_type(pos, mass, box, center, axis, width, thickness, x_edges, y_edges,
                        method="cic", weights="mass", smooth=0):
    if pos is None or pos.size==0: return None
    axd = {"x":0,"y":1,"z":2}; ax = axd[axis]
    zc = center[ax]
    m_th = np.abs(wrap_min_image_1d(pos[:,ax], zc, box)) <= 0.5*thickness
    if not np.any(m_th): return None
    pos = pos[m_th]; mass = None if mass is None else mass[m_th]
    perp = [i for i in range(3) if i!=ax]
    x = pos[:, perp[0]]; y = pos[:, perp[1]]
    cx, cy = center[perp[0]], center[perp[1]]
    m_roi, x, y = select_roi_2d(x, y, cx, cy, width, box)
    if weights=="counts" or mass is None: w = np.ones_like(x[m_roi])
    else: w = mass[m_roi]
    x = x; y = y
    if method=="cic": grid = _cic_deposit(x, y, w, x_edges, y_edges)
    else: grid = _hist2d(x, y, w, x_edges, y_edges)
    # convert to surface density (Msun / Mpc^2) if using mass weights
    dx = x_edges[1]-x_edges[0]; dy = y_edges[1]-y_edges[0]; area = dx*dy
    grid = grid/area if weights=="mass" else grid/(area)  # counts per Mpc^2 also ok
    if smooth>0: grid = box_smooth(grid, smooth)
    return grid

# ---------- scaling + plot ----------
def smart_log_clipping(surf, vminp=5.0, vmaxp=90.0):
    good = (surf>0)&np.isfinite(surf)
    if not np.any(good):
        return np.full_like(surf,-12.0), -12.0, -10.0
    vals = np.log10(surf[good])
    vmin = np.percentile(vals, vminp); vmax = np.percentile(vals, vmaxp)
    if vmax<=vmin: vmax = vals.max(); vmin = vmax-4.0
    return np.log10(np.clip(surf,10**vmin,10**vmax)), vmin, vmax

def plot_overlay(s1, s2, x_edges, y_edges, center, axis, width, thickness, z,
                 cmap1="bone", alpha1=0.9, vminp1=70, vmaxp1=99.8, label1="PartType1",
                 cmap2="magma", alpha2=0.6, vminp2=40, vmaxp2=99,   label2="PartType2",
                 output="slice_zoom_1_2.png"):
    extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
    fig, ax = plt.subplots(figsize=(8.6,7.2))
    if s1 is not None:
        l1,v1min,v1max = smart_log_clipping(s1, vminp1, vmaxp1)
        im1 = ax.imshow(l1, origin="lower", extent=extent, cmap=cmap1, alpha=alpha1)
        c1 = fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
        c1.set_label(f"log10 Σ [{label1}]")
    if s2 is not None:
        l2,v2min,v2max = smart_log_clipping(s2, vminp2, vmaxp2)
        im2 = ax.imshow(l2, origin="lower", extent=extent, cmap=cmap2, alpha=alpha2)
        c2 = fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.12)
        c2.set_label(f"log10 Σ [{label2}]")
    ax.set_xlabel("Mpc"); ax.set_ylabel("Mpc")
    ax.set_title(f"DM slice around {np.array2string(center, precision=3)} "
                 f"(axis={axis}, w={width} Mpc, t={thickness} Mpc, z={z:.2f})")
    # crosshair
    cx = 0.5*(x_edges[0]+x_edges[-1]); cy = 0.5*(y_edges[0]+y_edges[-1]); L = 0.05*(x_edges[-1]-x_edges[0])
    ax.plot([cx-L, cx+L], [cy, cy], color="white", lw=1.2, alpha=0.9)
    ax.plot([cx, cx], [cy-L, cy+L], color="white", lw=1.2, alpha=0.9)
    plt.tight_layout(); plt.savefig(output, dpi=200, bbox_inches="tight"); print(f"Saved {output}")

def main():
    ap = argparse.ArgumentParser(description="Zoomed DM slice overlaying PartType1 and PartType2.")
    ap.add_argument("snapshot"); ap.add_argument("x", type=float); ap.add_argument("y", type=float); ap.add_argument("z", type=float)
    ap.add_argument("-a","--axis", choices=["x","y","z"], default="z")
    ap.add_argument("-w","--width", type=float, default=5.0)
    ap.add_argument("-t","--thickness", type=float, default=1.0)
    ap.add_argument("-r","--res", type=int, default=1024)
    ap.add_argument("--weights", choices=["mass","counts"], default="mass", help="Per-particle weight")
    ap.add_argument("--method", choices=["hist","cic"], default="cic", help="Deposition method")
    ap.add_argument("--smooth", type=int, default=0, help="3x3 box smoothing passes")
    ap.add_argument("--cmap1", default="bone");  ap.add_argument("--alpha1", type=float, default=0.9)
    ap.add_argument("--cmap2", default="magma"); ap.add_argument("--alpha2", type=float, default=0.6)
    ap.add_argument("--vminp1", type=float, default=70.0); ap.add_argument("--vmaxp1", type=float, default=99.8)
    ap.add_argument("--vminp2", type=float, default=40.0); ap.add_argument("--vmaxp2", type=float, default=99.0)
    ap.add_argument("--label1", default="PartType1"); ap.add_argument("--label2", default="PartType2")
    ap.add_argument("-o","--output", default="slice_zoom_1_2.png")
    args = ap.parse_args()

    data = load_snapshot(args.snapshot); center = np.array([args.x, args.y, args.z], float)
    # build one common grid
    x_edges = np.linspace(center[0]-0.5*args.width, center[0]+0.5*args.width, args.res+1)
    y_edges = np.linspace(center[1]-0.5*args.width, center[1]+0.5*args.width, args.res+1)

    s1 = make_slice_for_type(data["p1_pos"], data["p1_mass"], data["box"], center, args.axis,
                             args.width, args.thickness, x_edges, y_edges,
                             method=args.method, weights=args.weights, smooth=args.smooth)
    s2 = make_slice_for_type(data["p2_pos"], data["p2_mass"], data["box"], center, args.axis,
                             args.width, args.thickness, x_edges, y_edges,
                             method=args.method, weights=args.weights, smooth=args.smooth)
    if s1 is None and s2 is None:
        raise SystemExit("No particles in slice/ROI for either PartType1 or PartType2.")
    plot_overlay(s1, s2, x_edges, y_edges, center, args.axis, args.width, args.thickness, data["z"],
                 cmap1=args.cmap1, alpha1=args.alpha1, vminp1=args.vminp1, vmaxp1=args.vmaxp1, label1=args.label1,
                 cmap2=args.cmap2, alpha2=args.alpha2, vminp2=args.vminp2, vmaxp2=args.vmaxp2, label2=args.label2,
                 output=args.output)

if __name__ == "__main__": main()
