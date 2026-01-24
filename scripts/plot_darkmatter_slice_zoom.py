#!/usr/bin/env python3
import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt

def wrap_min_image_1d(x, center, box):
    """Shift x so that values are within [-box/2, box/2) relative to center."""
    y = x - center
    y -= np.round(y / box) * box
    return y

def select_roi_2d(x, y, cx, cy, w, box):
    """Periodic ROI selection around (cx, cy) with width w (square window)."""
    X = wrap_min_image_1d(x, cx, box)
    Y = wrap_min_image_1d(y, cy, box)
    half = 0.5 * w
    mask = (np.abs(X) <= half) & (np.abs(Y) <= half)
    return mask, X[mask] + cx, Y[mask] + cy

def load_snapshot(filename):
    f = h5py.File(filename, "r")
    header = dict(f["Header"].attrs)
    boxsize = header["BoxSize"]
    coords = f["PartType1/Coordinates"][:]  # DM only
    massarr = header["MassTable"]
    if massarr[1] > 0:
        mass = np.full(coords.shape[0], massarr[1], dtype=np.float64)
    else:
        mass = f["PartType1/Masses"][:]
    f.close()
    return coords, mass, boxsize, header

def make_slice(coords, mass, box, center, axis="z", width=5.0, thickness=1.0, res=512):
    """Make a 2D density slice centered at (x,y,z)."""
    axes = {"x":0, "y":1, "z":2}
    ax = axes[axis]

    # slice center along projection axis
    slice_center = center[ax]

    # select slice thickness
    mask = np.abs(wrap_min_image_1d(coords[:,ax], slice_center, box)) <= 0.5*thickness
    coords = coords[mask]
    mass   = mass[mask]

    # choose perpendicular axes
    perp = [i for i in range(3) if i != ax]
    x = coords[:, perp[0]]
    y = coords[:, perp[1]]

    cx, cy = center[perp[0]], center[perp[1]]

    mask, x, y = select_roi_2d(x, y, cx, cy, width, box)
    mass = mass[mask]

    edges = [np.linspace(cx-0.5*width, cx+0.5*width, res+1),
             np.linspace(cy-0.5*width, cy+0.5*width, res+1)]

    H, _, _ = np.histogram2d(x, y, bins=edges, weights=mass)
    pixel_area = (edges[0][1]-edges[0][0])*(edges[1][1]-edges[1][0])
    surf = (H.T) / pixel_area
    return surf, edges

def plot_slice(surf, edges, axis, center, width, thickness, output):
    plt.figure(figsize=(6,6))
    extent = [edges[0][0], edges[0][-1], edges[1][0], edges[1][-1]]
    plt.imshow(np.log10(surf+1e-12), origin="lower", extent=extent, cmap="magma")
    plt.colorbar(label="log10 Surface Density [Msun/Mpc^2]")
    plt.xlabel("Mpc")
    plt.ylabel("Mpc")
    plt.title(f"DM slice around {center} (axis={axis}, w={width} Mpc, t={thickness} Mpc)")
    plt.savefig(output, dpi=200, bbox_inches="tight")
    print(f"Saved {output}")

def main():
    ap = argparse.ArgumentParser(description="Zoomed dark matter slice around given (x,y,z).")
    ap.add_argument("snapshot", help="HDF5 snapshot filename")
    ap.add_argument("x", type=float, help="x center (Mpc)")
    ap.add_argument("y", type=float, help="y center (Mpc)")
    ap.add_argument("z", type=float, help="z center (Mpc)")
    ap.add_argument("-a","--axis", choices=["x","y","z"], default="z", help="Projection axis")
    ap.add_argument("-w","--width", type=float, default=5.0, help="ROI width (Mpc)")
    ap.add_argument("-t","--thickness", type=float, default=1.0, help="Slice thickness (Mpc)")
    ap.add_argument("-r","--res", type=int, default=512, help="Grid resolution")
    ap.add_argument("-o","--output", default="slice_zoom.png", help="Output image filename")
    args = ap.parse_args()

    coords, mass, box, header = load_snapshot(args.snapshot)
    center = np.array([args.x, args.y, args.z])
    surf, edges = make_slice(coords, mass, box, center,
                             axis=args.axis, width=args.width,
                             thickness=args.thickness, res=args.res)
    plot_slice(surf, edges, args.axis, center, args.width, args.thickness, args.output)

if __name__ == "__main__":
    main()
