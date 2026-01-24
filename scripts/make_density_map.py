#!/usr/bin/env python3
import argparse
import h5py, numpy as np
import matplotlib.pyplot as plt
import os

def read_header(h):
    hdr = h['Header'].attrs
    d = dict(
        BoxSize=float(hdr['BoxSize']),
        Time=float(hdr.get('Time', 0.0)),
        Redshift=float(hdr.get('Redshift', -1.0)),
        HubbleParam=float(hdr.get('HubbleParam', 1.0)),
        NumPart_ThisFile=np.array(hdr['NumPart_ThisFile']),
        MassTable=np.array(hdr['MassTable'])
    )
    return d

def periodic_wrap(delta, box):
    # wrap into [-box/2, box/2)
    return (delta + 0.5*box) % box - 0.5*box

def slice_mask(pos, center, axis_index, half_thickness, box):
    # shift positions relative to center with periodic wrap
    d = periodic_wrap(pos - center, box)
    # axis selection
    coord = d[:, axis_index]
    return (coord >= -half_thickness) & (coord <= half_thickness), d

def load_coords(h, ptype):
    key = f'PartType{ptype}'
    if key not in h:
        return None, None, None
    g = h[key]
    if 'Coordinates' not in g:
        return None, None, None
    pos = np.asarray(g['Coordinates'])
    masses = None
    if 'Masses' in g:
        masses = np.asarray(g['Masses'])
    return pos, masses, g

def estimate_hr_dm_mass(h):
    pos, masses, g = load_coords(h, 1)
    if masses is not None and masses.size > 0:
        sample = masses[:min(200000, masses.size)]
        return np.median(sample)
    # fallback to MassTable if per-particle masses absent
    mt = h['Header'].attrs['MassTable']
    if mt[1] > 0:
        return float(mt[1])
    return None

def auto_center_hr_dm(h, box, frac=0.05):
    pos, masses, g = load_coords(h, 1)
    if pos is None or pos.size == 0:
        return np.array([0.5*box, 0.5*box, 0.5*box])
    # If we have masses, bias the COM; otherwise treat equal
    w = masses if masses is not None else np.ones(pos.shape[0], dtype=np.float64)
    # crude: find a dense patch by taking median position of nearest quantile by mass
    # We'll pick a random subset to avoid memory blowups
    n = pos.shape[0]
    idx = np.random.RandomState(0).choice(n, size=min(500000, n), replace=False)
    p = pos[idx]
    wsub = w[idx]
    # Center-of-mass in periodic domain: shift to unit cube around an initial guess, iterate
    c = np.array([0.5*box, 0.5*box, 0.5*box], dtype=np.float64)
    for _ in range(4):
        d = periodic_wrap(p - c, box)
        c = (c + (wsub[:,None]*d).sum(axis=0)/wsub.sum()) % box
    return c

def hist2d_density(d, axes=(0,1), weights=None, bins=800, extent=None):
    x = d[:, axes[0]]
    y = d[:, axes[1]]
    H, xedges, yedges = np.histogram2d(x, y, bins=bins, range=extent, weights=weights)
    return H.T, xedges, yedges

def make_layers(h, center, axis='z', half_thickness=0.0005, bins=800, extent=None, mass_weight=False,
                include_types=(2,3,5,1,0,4,6), lr_alpha=0.6, hi_alpha=0.9):
    header = read_header(h)
    box = header['BoxSize']
    axis_idx = {'x':0, 'y':1, 'z':2}[axis]

    # extent in wrapped coordinates around center
    if extent is None:
        # default: square region of +/- 0.25 box around center in the two plot axes
        s = 0.25*box
        if axis_idx == 2:
            extent = [[-s, s], [-s, s]]
        elif axis_idx == 1:
            extent = [[-s, s], [-s, s]]
        else:
            extent = [[-s, s], [-s, s]]

    # Axis labels / orientation mapping
    # When axis='z', we show X vs Y of wrapped coords
    show_axes = [i for i in [0,1,2] if i != axis_idx]
    layers = []

    for ptype in include_types:
        pos, masses, g = load_coords(h, ptype)
        if pos is None:
            continue

        mask, dwrap = slice_mask(pos, np.array(center), axis_idx, half_thickness, box)
        if not np.any(mask):
            continue

        dsl = dwrap[mask]

        # choose weights
        w = None
        if mass_weight and masses is not None:
            w = masses[mask]

        # Build histogram in the plane
        # extent uses wrapped coordinates; we compute x/y in that wrapped frame
        H, xedges, yedges = hist2d_density(dsl, axes=(show_axes[0], show_axes[1]), weights=w, bins=bins, extent=extent)

        # log scale for view (keep raw counts for possible post-processing)
        Hplot = np.log10(H + 1.0)

        # Choose alpha: LR shells fainter
        if ptype in (2,3,5):
            alpha = lr_alpha
            zorder = 1
        else:
            alpha = hi_alpha
            zorder = 2

        layers.append(dict(
            ptype=ptype,
            H=Hplot,
            xedges=xedges,
            yedges=yedges,
            alpha=alpha,
            zorder=zorder
        ))
    return layers, header, show_axes

def plot_layers(layers, header, show_axes, out_png, title=None):
    fig, ax = plt.subplots(figsize=(6,6))
    for L in layers:
        extent = [L['xedges'][0], L['xedges'][-1], L['yedges'][0], L['yedges'][-1]]
        im = ax.imshow(L['H'], origin='lower', extent=extent, aspect='equal', alpha=L['alpha'], zorder=L['zorder'])
    ax.set_xlabel(['x','y','z'][show_axes[0]] + ' (box units)')
    ax.set_ylabel(['x','y','z'][show_axes[1]] + ' (box units)')
    if title is None:
        title = f"Density map @ z={header.get('Redshift',-1):.2f} (box={header['BoxSize']:.3f})"
    ax.set_title(title)
    plt.tight_layout()
    if out_png:
        plt.savefig(out_png, dpi=200)
    else:
        plt.show()

def parse_types(s):
    # comma-separated integers, e.g. "2,3,5,1,0,4"
    return tuple(int(x.strip()) for x in s.split(','))

def main():
    ap = argparse.ArgumentParser(description="2D density map for Gadget-4 snapshot (zoom-friendly).")
    ap.add_argument('--snap', required=True, help='Path to snapshot_XXX.hdf5')
    ap.add_argument('--center', nargs=3, type=float, help='Center (box units). If omitted and --auto-center hr_dm_com is set, that is used.')
    ap.add_argument('--auto-center', choices=['none','hr_dm_com'], default='none', help='How to auto center if --center not given.')
    ap.add_argument('--axis', choices=['x','y','z'], default='z', help='Slice normal axis (default z → show x vs y).')
    ap.add_argument('--half-thickness', type=float, default=0.0005, help='Half-thickness of slice (box units).')
    ap.add_argument('--bins', type=int, default=800, help='2D histogram bins per axis.')
    ap.add_argument('--mass-weight', action='store_true', help='Use particle masses as weights.')
    ap.add_argument('--types', type=parse_types, default=parse_types('2,3,5,1,0,4,6'),
                    help='Comma-separated PartTypes to include, rendered as layered density fields. Default "2,3,5,1,0,4,6".')
    ap.add_argument('--region', nargs=2, type=float, metavar=('DX','DY'),
                    help='Plot half-widths (box units) in the two displayed axes; default 0.25 0.25.')
    ap.add_argument('--out', default=None, help='Output PNG filename. If omitted, shows interactively.')
    args = ap.parse_args()

    with h5py.File(args.snap, 'r') as h:
        header = read_header(h)
        box = header['BoxSize']

        if args.center is not None:
            center = np.array(args.center, dtype=np.float64)
        elif args.auto_center == 'hr_dm_com':
            center = auto_center_hr_dm(h, box)
            print(f"[auto-center] HR DM COM ~ {center}")
        else:
            center = np.array([0.5*box, 0.5*box, 0.5*box])
            print(f"[center] default box center {center}")

        # extent (in wrapped coordinates around center), e.g., [-dx, dx] × [-dy, dy]
        if args.region is not None:
            dx, dy = args.region
            extent = [[-dx, dx], [-dy, dy]]
        else:
            s = 0.25*box
            extent = [[-s, s], [-s, s]]

        layers, header, show_axes = make_layers(
            h, center, axis=args.axis, half_thickness=args.half_thickness,
            bins=args.bins, extent=extent, mass_weight=args.mass_weight,
            include_types=args.types
        )

    title = os.path.basename(args.snap)
    plot_layers(layers, header, show_axes, args.out, title=title)

if __name__ == '__main__':
    main()