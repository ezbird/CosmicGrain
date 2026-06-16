#!/usr/bin/env python3
"""
plot_halo_projection.py  –  CosmicGrain Figure 1
=========================================================
Three-panel projection map of Halo 569 at a chosen redshift.

    Left   : gas column density          Σ_gas  [M☉/pc²]
    Centre : mass-weighted temperature   ⟨T⟩_mw [K]
    Right  : dust surface density        Σ_dust [M☉/pc²]

SPH projection is done with an adaptive Gaussian kernel per
particle (vectorised by smoothing-length bin) so it runs in
a reasonable time on a laptop even for 1024³ particle counts.

Usage
-----
python plot_halo_projection.py \
    --snap ../S10_output_2048/snapdir_015 \
    --groups ../S10_output_2048/groups_015/fof_subhalo_tab_015.0.hdf5 \
    --half-width 120 --npix 1024 \
    --dust-smooth 3 \
    --rung "S10 2048³ corrected z=3.53" \
    --out corrected_snap15_wide.png

        [--halo-id 0]           # FoF halo index (default 0 = primary)
        [--half-width 300]      # half-width of projection box in pkpc
                                # (default: R_200c from group catalog)
        [--depth-frac 1.0]      # projection depth = depth_frac × half_width
        [--npix 1024]           # pixels per side
        [--axis z]              # projection axis: x | y | z
        [--smooth-fac 1.0]      # extra Gaussian smoothing (1.0 = none)
        [--dust-smooth 0]       # Gaussian smoothing for dust panel in pkpc
        [--rung S10]            # label for figure title / filename
        [--no-circle]           # suppress R_200 circle
        [--fourth-panel dz]     # add 4th panel: 'dz' (D/Z) or 'abar' (mean grain size)

Use this to see which halos are biggest etc. can you --halo-id to switch centering:
python3 - << 'EOF'
import h5py, numpy as np

cat = '../S10_output_2048/groups_015/fof_subhalo_tab_015.0.hdf5'
with h5py.File(cat, 'r') as f:
    pos  = f['Group/GroupPos'][:]
    mass = f['Group/Group_M_Crit200'][:]
    r200 = f['Group/Group_R_Crit200'][:]

h = 0.6732
a = 0.2207
for i in range(min(5, len(mass))):
    pos_pkpc = pos[i] * a / h
    print(f"Group[{i}]: M200={mass[i]:.3e}, R200={r200[i]*a/h:.1f} pkpc, "
          f"pos=[{pos[i][0]:.1f}, {pos[i][1]:.1f}, {pos[i][2]:.1f}] ckpc/h")
EOF

Notes
-----
* HubbleParam is read from Parameters.attrs (not Header.attrs).
* Dust particles are PartType6.
* Coordinates are comoving kpc/h internally; all display units are physical.
* Temperature uses a fixed mean molecular weight μ = 0.6.  If your snapshot
  has 'ElectronAbundance', pass --use-xe to compute μ per-particle.
"""

import argparse
import os
import sys

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
import numpy as np
from scipy.ndimage import gaussian_filter

# ── Physical constants ────────────────────────────────────────────────────────
PROTONMASS = 1.6726e-24   # g
BOLTZMANN  = 1.3806e-16   # erg/K
GAMMA      = 5.0 / 3.0
MU_FIXED   = 0.6          # mean molecular weight (primordial, ~ionised)
KPC_TO_PC  = 1.0e3
MSUN       = 1.989e33     # g

# ── CosmicGrain palette ───────────────────────────────────────────────────────
TEAL = '#2a9d8f'

def teal_cmap():
    """Custom sequential colormap for dust panel (black → teal → white)."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        'dust_teal',
        ['#000000', '#0d4f47', '#2a9d8f', '#85d3cb', '#ffffff'],
    )

# ── Unit helpers ──────────────────────────────────────────────────────────────
def code_to_physical_kpc(x_ckpc_h, a, h):
    """Comoving kpc/h  →  physical kpc."""
    return x_ckpc_h * a / h

def code_mass_to_msun(m_code, h):
    """Code mass (1e10 Msun/h)  →  M☉."""
    return m_code * 1.0e10 / h

def internal_energy_to_K(u_code, xe=None):
    """
    Gadget internal energy (km/s)^2  →  temperature [K].
    xe : electron abundance per hydrogen atom (array or None)
    """
    u_cgs = u_code * 1.0e10
    if xe is not None:
        xH = 0.76
        mu = 4.0 / (1.0 + 3.0 * xH + 4.0 * xH * xe)
    else:
        mu = MU_FIXED
    return u_cgs * (GAMMA - 1.0) * mu * PROTONMASS / BOLTZMANN

# ── I/O ───────────────────────────────────────────────────────────────────────
def snap_file_list(snap_arg):
    import glob
    base = snap_arg.rstrip('/')
    for pat in (r'.0.hdf5', r'.hdf5'):
        if base.endswith(pat):
            base = base[: -len(pat)]
            break
    if os.path.isdir(base) or os.path.isdir(snap_arg):
        d = snap_arg if os.path.isdir(snap_arg) else base
        candidates = sorted(glob.glob(os.path.join(d, '*.0.hdf5')))
        if not candidates:
            candidates = sorted(glob.glob(os.path.join(d, '*.hdf5')))
        if not candidates:
            sys.exit(f'ERROR: No HDF5 files found in {d}')
        first = candidates[0]
        base  = first[:-len('.0.hdf5')] if first.endswith('.0.hdf5') else first[:-len('.hdf5')]
    first_try = base + '.0.hdf5'
    single    = base + '.hdf5'
    if os.path.exists(first_try):
        first_file = first_try
    elif os.path.exists(single):
        return [single]
    else:
        sys.exit(f'ERROR: Cannot find snapshot at {base}[.0].hdf5')
    with h5py.File(first_file, 'r') as f:
        n_files = int(f['Header'].attrs.get('NumFilesPerSnapshot', 1))
    paths = [f'{base}.{i}.hdf5' for i in range(n_files)]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        sys.exit(f'ERROR: Missing snapshot file(s): {missing}')
    return paths


def read_header(snap_files):
    with h5py.File(snap_files[0], 'r') as f:
        h = float(f['Parameters'].attrs['HubbleParam'])
        a = float(f['Header'].attrs['Time'])
        z = float(f['Header'].attrs['Redshift'])
    return dict(h=h, a=a, z=z)


def read_halo_center(group_file, halo_id, hdr):
    with h5py.File(group_file, 'r') as gf:
        pos   = gf['Group/GroupPos'][halo_id]
        r200c = gf['Group/Group_R_Crit200'][halo_id]
    r200c_pkpc = code_to_physical_kpc(r200c, hdr['a'], hdr['h'])
    return pos, r200c_pkpc


def _box_mask(pos_ckpc_h, center_ckpc_h, half_ckpc_h, depth_ckpc_h, axis):
    axes_map = {'x': 0, 'y': 1, 'z': 2}
    los  = axes_map[axis]
    perp = [a for a in range(3) if a != los]
    dx = pos_ckpc_h - center_ckpc_h
    in_perp = np.all(np.abs(dx[:, perp]) < half_ckpc_h, axis=1)
    in_los  = np.abs(dx[:, los]) < depth_ckpc_h
    return in_perp & in_los


def _read_ptype_fields(snap_files, ptype, fields):
    bufs = {name: [] for name in fields}
    found_ptype = False
    for fpath in snap_files:
        with h5py.File(fpath, 'r') as f:
            if ptype not in f:
                continue
            found_ptype = True
            grp = f[ptype]
            for name, optional in fields.items():
                if name in grp:
                    bufs[name].append(grp[name][:])
                elif not optional:
                    sys.exit(f'ERROR: Required field {ptype}/{name} missing in {fpath}')
    if not found_ptype:
        return None
    out = {}
    for name in fields:
        if bufs[name]:
            out[name] = np.concatenate(bufs[name], axis=0)
        else:
            out[name] = None
    return out


def read_gas(snap_files, center_ckpc_h, half_ckpc_h, depth_ckpc_h,
             axis, hdr, use_xe=False):
    fields = {
        'Coordinates':      False,
        'Masses':           False,
        'InternalEnergy':   False,
        'SmoothingLength':  False,
        'ElectronAbundance': True,
        'Metallicity':      True,
    }
    data = _read_ptype_fields(snap_files, 'PartType0', fields)
    if data is None:
        sys.exit('ERROR: No PartType0 (gas) found in snapshot.')
    pos   = data['Coordinates']
    mass  = data['Masses']
    u     = data['InternalEnergy']
    hsml  = data['SmoothingLength']
    xe    = data['ElectronAbundance'] if use_xe else None
    metals = data['Metallicity']
    mask = _box_mask(pos, center_ckpc_h, half_ckpc_h, depth_ckpc_h, axis)
    pos  = pos[mask];  mass = mass[mask]
    u    = u[mask];    hsml = hsml[mask]
    if xe     is not None: xe     = xe[mask]
    if metals is not None: metals = metals[mask]
    a, h = hdr['a'], hdr['h']
    pos_pkpc  = code_to_physical_kpc(pos - center_ckpc_h, a, h)
    hsml_pkpc = code_to_physical_kpc(hsml, a, h)
    mass_msun = code_mass_to_msun(mass, h)
    T_K       = internal_energy_to_K(u, xe)
    axes_map = {'x': 0, 'y': 1, 'z': 2}
    los  = axes_map[axis]
    perp = [a2 for a2 in range(3) if a2 != los]
    pos2d = pos_pkpc[:, perp]
    return pos2d, mass_msun, T_K, hsml_pkpc, metals


def read_dust(snap_files, center_ckpc_h, half_ckpc_h, depth_ckpc_h,
              axis, hdr):
    fields = {
        'Coordinates': False,
        'Masses':      False,
        'GrainRadius': True,
    }
    data = _read_ptype_fields(snap_files, 'PartType6', fields)
    if data is None:
        return None, None, None
    pos  = data['Coordinates']
    mass = data['Masses']
    abar = data['GrainRadius']
    mask = _box_mask(pos, center_ckpc_h, half_ckpc_h, depth_ckpc_h, axis)
    pos  = pos[mask];  mass = mass[mask]
    if abar is not None: abar = abar[mask]
    a, h = hdr['a'], hdr['h']
    pos_pkpc  = code_to_physical_kpc(pos - center_ckpc_h, a, h)
    mass_msun = code_mass_to_msun(mass, h)
    axes_map = {'x': 0, 'y': 1, 'z': 2}
    los  = axes_map[axis]
    perp = [a2 for a2 in range(3) if a2 != los]
    pos2d = pos_pkpc[:, perp]
    return pos2d, mass_msun, abar

# ── SPH projection ────────────────────────────────────────────────────────────
def project_sph(pos2d_pkpc, mass_msun, hsml_pkpc, npix, half_width_pkpc,
                quantity=None, smooth_fac=1.0, max_stamp=48):
    pixel_kpc = 2.0 * half_width_pkpc / npix
    px = (pos2d_pkpc[:, 0] + half_width_pkpc) / pixel_kpc
    py = (pos2d_pkpc[:, 1] + half_width_pkpc) / pixel_kpc
    ph = np.maximum(0.5, hsml_pkpc / pixel_kpc)
    margin = max_stamp
    in_view = ((px >= -margin) & (px < npix + margin) &
               (py >= -margin) & (py < npix + margin))
    px, py, ph = px[in_view], py[in_view], ph[in_view]
    mass = mass_msun[in_view]
    if quantity is not None:
        qty = quantity[in_view]
    mass_map = np.zeros((npix, npix), dtype=np.float64)
    qty_map  = np.zeros((npix, npix), dtype=np.float64) if quantity is not None else None
    ph_int  = np.clip(ph.astype(int), 1, max_stamp)
    sort_idx = np.argsort(ph_int)
    px, py, ph_int, mass = px[sort_idx], py[sort_idx], ph_int[sort_idx], mass[sort_idx]
    if quantity is not None:
        qty = qty[sort_idx]
    ix = np.round(px).astype(int)
    iy = np.round(py).astype(int)
    for h_px in np.unique(ph_int):
        sel = ph_int == h_px
        _ix = ix[sel]; _iy = iy[sel]; _m = mass[sel]
        _q  = qty[sel] if quantity is not None else None
        r  = min(int(np.ceil(2.5 * h_px)), max_stamp)
        xx = np.arange(-r, r + 1, dtype=np.float64)
        K  = np.exp(-0.5 * (xx[:, None]**2 + xx[None, :]**2) / h_px**2)
        K /= K.sum()
        for i in range(len(_ix)):
            x0, y0 = _ix[i], _iy[i]
            x_lo = x0 - r;  x_hi = x0 + r + 1
            y_lo = y0 - r;  y_hi = y0 + r + 1
            gx_lo = max(0, x_lo);  gx_hi = min(npix, x_hi)
            gy_lo = max(0, y_lo);  gy_hi = min(npix, y_hi)
            if gx_lo >= gx_hi or gy_lo >= gy_hi:
                continue
            kx_lo = gx_lo - x_lo;  kx_hi = kx_lo + (gx_hi - gx_lo)
            ky_lo = gy_lo - y_lo;  ky_hi = ky_lo + (gy_hi - gy_lo)
            mass_map[gx_lo:gx_hi, gy_lo:gy_hi] += _m[i] * K[kx_lo:kx_hi, ky_lo:ky_hi]
            if qty_map is not None:
                qty_map[gx_lo:gx_hi, gy_lo:gy_hi] += _m[i] * _q[i] * K[kx_lo:kx_hi, ky_lo:ky_hi]
    if smooth_fac > 1.0:
        sigma = smooth_fac
        mass_map = gaussian_filter(mass_map, sigma=sigma)
        if qty_map is not None:
            qty_map = gaussian_filter(qty_map, sigma=sigma)
    return mass_map, qty_map


def project_points(pos2d_pkpc, mass_msun, npix, half_width_pkpc,
                   quantity=None, smooth_pkpc=None):
    pixel_kpc = 2.0 * half_width_pkpc / npix
    edges = np.linspace(-half_width_pkpc, half_width_pkpc, npix + 1)
    mass_map, _, _ = np.histogram2d(
        pos2d_pkpc[:, 0], pos2d_pkpc[:, 1],
        bins=[edges, edges], weights=mass_msun)
    qty_map = None
    if quantity is not None:
        w_map, _, _ = np.histogram2d(
            pos2d_pkpc[:, 0], pos2d_pkpc[:, 1],
            bins=[edges, edges], weights=mass_msun * quantity)
        qty_map = w_map
    if smooth_pkpc is not None and smooth_pkpc > 0:
        sigma_px = smooth_pkpc / pixel_kpc
        mass_map = gaussian_filter(mass_map, sigma=sigma_px)
        if qty_map is not None:
            qty_map = gaussian_filter(qty_map, sigma=sigma_px)
    return mass_map, qty_map


def to_surf_dens(mass_map_msun_pkpc2, half_width_pkpc, npix):
    pixel_kpc = 2.0 * half_width_pkpc / npix
    pixel_pc2 = (pixel_kpc * KPC_TO_PC)**2
    return mass_map_msun_pkpc2 / pixel_pc2


# ── Figure ────────────────────────────────────────────────────────────────────
def make_figure(panels, half_width_pkpc, r200_pkpc, z, rung,
                axis, show_r200_circle, out_path, fourth_panel_label=None):
    n_panels = len(panels)
    # Extra right margin so the last colorbar label doesn't get clipped
    fig_w = 5.2 * n_panels + 1.2
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, 5.8),
                             facecolor='black')
    if n_panels == 1:
        axes = [axes]

    # Leave generous right margin for the rightmost colorbar label
    fig.subplots_adjust(left=0.06, right=0.91, top=0.93, bottom=0.08,
                        wspace=0.08)

    for idx, (ax, (img, title, cmap, norm, cbar_label, _)) in enumerate(zip(axes, panels)):
        im = ax.imshow(
            img.T,
            origin='lower',
            extent=[-half_width_pkpc, half_width_pkpc,
                    -half_width_pkpc, half_width_pkpc],
            cmap=cmap, norm=norm,
            interpolation='lanczos',
            aspect='equal',
        )
        ax.set_facecolor('black')
        ax.tick_params(colors='white', labelsize=9, direction='in',
                       top=True, right=True)
        for spine in ax.spines.values():
            spine.set_edgecolor('white')
            spine.set_linewidth(0.6)
        ax.set_title(title, color='white', fontsize=11, pad=4)

        if show_r200_circle:
            theta = np.linspace(0, 2 * np.pi, 360)
            ax.plot(r200_pkpc * np.cos(theta), r200_pkpc * np.sin(theta),
                    '--', color='white', lw=0.7, alpha=0.65)
            ax.text(r200_pkpc * 0.72, r200_pkpc * 0.72,
                    r'$R_{200c}$', color='white', fontsize=7.5, alpha=0.75)

        ax.set_xlim(-half_width_pkpc, half_width_pkpc)
        ax.set_ylim(-half_width_pkpc, half_width_pkpc)

        ax_label = {'x': ('y [pkpc]', 'z [pkpc]'),
                    'y': ('x [pkpc]', 'z [pkpc]'),
                    'z': ('x [pkpc]', 'y [pkpc]')}[axis]
        if ax is axes[0]:
            ax.set_xlabel(ax_label[0], color='white', fontsize=9)
            ax.set_ylabel(ax_label[1], color='white', fontsize=9)
        else:
            ax.set_yticklabels([])

        is_last = (idx == n_panels - 1)

        # Colorbar — placed to the right of each panel.
        # For non-last panels use a narrow bar with the label rotated
        # vertically ON the bar (not spilling into the next panel).
        # For the last panel a slightly wider bar is fine.
        cbar_frac    = 0.035 if not is_last else 0.046
        cbar_pad     = 0.01
        cbar = fig.colorbar(im, ax=ax, pad=cbar_pad,
                            fraction=cbar_frac, aspect=25)
        cbar.ax.yaxis.set_tick_params(color='white', labelsize=7.5)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')
        cbar.outline.set_edgecolor('white')

        # Label: rotated vertically, placed on the colorbar axis itself
        # so it never overlaps the adjacent panel.
        cbar.set_label(cbar_label, color='white', fontsize=8,
                       rotation=270, labelpad=12)

    # Scale bar on leftmost panel
    scalebar_pkpc = _nice_scalebar(half_width_pkpc)
    ax0 = axes[0]
    x0  = -half_width_pkpc + 0.07 * 2 * half_width_pkpc
    y0  = -half_width_pkpc + 0.07 * 2 * half_width_pkpc
    ax0.plot([x0, x0 + scalebar_pkpc], [y0, y0], '-', color='white', lw=2)
    ax0.text(x0 + scalebar_pkpc / 2, y0 + 0.03 * 2 * half_width_pkpc,
             f'{scalebar_pkpc:.0f} pkpc', ha='center', va='bottom',
             color='white', fontsize=8)

    # Redshift + rung label on rightmost panel
    axes[-1].text(0.97, 0.97,
                  rf'$z = {z:.2f}$' + (f'\n{rung}' if rung else ''),
                  transform=axes[-1].transAxes,
                  ha='right', va='top', color='white',
                  fontsize=9, linespacing=1.6)

    fig.suptitle('CosmicGrain — Halo 569', color='white', fontsize=13,
                 y=0.98)

    plt.savefig(out_path, dpi=200, bbox_inches='tight',
                facecolor='black', edgecolor='none')
    plt.close(fig)
    print(f'  Saved → {out_path}')


def _nice_scalebar(half_width_pkpc):
    for s in [500, 200, 100, 50, 20, 10, 5]:
        if s < half_width_pkpc * 1.0:
            return float(s)
    return max(1.0, round(half_width_pkpc * 0.2, -1))


# ── Main ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--snap',        required=True)
    p.add_argument('--groups',      default=None)
    p.add_argument('--halo-id',     type=int, default=0)
    p.add_argument('--center',      type=float, nargs=3, default=None,
                   metavar=('X','Y','Z'))
    p.add_argument('--r200',        type=float, default=None)
    p.add_argument('--half-width',  type=float, default=None)
    p.add_argument('--depth-frac',  type=float, default=1.0)
    p.add_argument('--npix',        type=int,   default=1024)
    p.add_argument('--axis',        choices=['x','y','z'], default='z')
    p.add_argument('--smooth-fac',  type=float, default=1.0)
    p.add_argument('--dust-smooth', type=float, default=0.0)
    p.add_argument('--rung',        default='')
    p.add_argument('--no-circle',   action='store_true')
    p.add_argument('--fourth-panel', choices=['dz','abar'], default=None)
    p.add_argument('--use-xe',      action='store_true')
    p.add_argument('--out',         default='figure1_projection.png')
    p.add_argument('--vmin-gas',    type=float, default=None)
    p.add_argument('--vmax-gas',    type=float, default=None)
    p.add_argument('--vmin-temp',   type=float, default=None)
    p.add_argument('--vmax-temp',   type=float, default=None)
    p.add_argument('--vmin-dust',   type=float, default=None)
    p.add_argument('--vmax-dust',   type=float, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    snap_files = snap_file_list(args.snap)
    n = len(snap_files)
    print(f'Snapshot: {snap_files[0]}' +
          (f'  (+{n-1} more files)' if n > 1 else ''))
    hdr = read_header(snap_files)
    print(f'  z = {hdr["z"]:.4f},  a = {hdr["a"]:.4f},  h = {hdr["h"]:.4f}')

    if args.center is not None:
        center_ckpc_h = np.array(args.center)
        print(f'  Using manual centre: {center_ckpc_h} ckpc/h')
    elif args.groups is not None:
        center_ckpc_h, r200_pkpc_cat = read_halo_center(
            args.groups, args.halo_id, hdr)
        print(f'  Halo {args.halo_id} centre: {center_ckpc_h} ckpc/h')
        print(f'  R_200c = {r200_pkpc_cat:.1f} pkpc')
    else:
        sys.exit('ERROR: Provide --groups or --center.')

    if args.r200 is not None:
        r200_pkpc = args.r200
    elif args.groups is not None:
        r200_pkpc = r200_pkpc_cat
    else:
        r200_pkpc = 200.0
        print(f'  WARNING: No R_200c available, using {r200_pkpc} pkpc.')

    half_width = args.half_width if args.half_width is not None else r200_pkpc
    depth      = half_width * args.depth_frac
    print(f'  Projection box: ±{half_width:.1f} pkpc, depth ±{depth:.1f} pkpc')

    half_ckpc_h  = half_width  / hdr['a'] * hdr['h']
    depth_ckpc_h = depth       / hdr['a'] * hdr['h']

    print('Reading gas particles…')
    gas_pos2d, gas_mass, gas_T, gas_hsml, gas_metals = read_gas(
        snap_files, center_ckpc_h, half_ckpc_h, depth_ckpc_h,
        args.axis, hdr, use_xe=args.use_xe)
    print(f'  {len(gas_mass):,} gas particles in slab')

    print('Projecting gas…')
    gas_mass_map, gas_temp_map = project_sph(
        gas_pos2d, gas_mass, gas_hsml, args.npix, half_width,
        quantity=gas_T, smooth_fac=args.smooth_fac)

    pixel_kpc = 2.0 * half_width / args.npix
    pixel_pc2 = (pixel_kpc * KPC_TO_PC)**2
    gas_sigma = gas_mass_map / pixel_pc2

    T_mw = np.where(gas_mass_map > 0, gas_temp_map / gas_mass_map, np.nan)

    dz_map = None
    dz_num = None
    if args.fourth_panel == 'dz' and gas_metals is not None:
        _, dz_num = project_sph(
            gas_pos2d, gas_mass, gas_hsml, args.npix, half_width,
            quantity=gas_metals, smooth_fac=args.smooth_fac)

    del gas_pos2d, gas_mass, gas_T, gas_hsml

    print('Reading dust particles (PartType6)…')
    dust_pos2d, dust_mass, dust_abar = read_dust(
        snap_files, center_ckpc_h, half_ckpc_h, depth_ckpc_h,
        args.axis, hdr)

    if dust_pos2d is not None:
        print(f'  {len(dust_mass):,} dust particles in slab')
        print('Projecting dust…')
        dust_mass_map, dust_abar_map = project_points(
            dust_pos2d, dust_mass, args.npix, half_width,
            quantity=dust_abar,
            smooth_pkpc=args.dust_smooth if args.dust_smooth > 0 else None)
        dust_sigma = dust_mass_map / pixel_pc2
        with np.errstate(invalid='ignore'):
            abar_mw = (np.where(dust_mass_map > 0,
                                dust_abar_map / dust_mass_map, np.nan)
                       if dust_abar_map is not None else None)
    else:
        print('  WARNING: No PartType6 found – dust panel will be empty.')
        dust_sigma = np.zeros((args.npix, args.npix))
        dust_mass_map = dust_sigma
        abar_mw = None

    if args.fourth_panel == 'dz' and dz_num is not None and dust_pos2d is not None:
        gas_metal_sigma = np.where(gas_mass_map > 0, dz_num / pixel_pc2, np.nan)
        dz_map = np.where(gas_metal_sigma > 0,
                          dust_sigma / gas_metal_sigma, np.nan)

    def safe_logbounds(arr):
        pos = arr[arr > 0]
        if len(pos) == 0:
            return 1e-5, 1.0
        return np.nanpercentile(pos, 0.5), np.nanpercentile(pos, 99.9)

    sg_lo, sg_hi = safe_logbounds(gas_sigma)
    gas_vmin = args.vmin_gas  if args.vmin_gas  is not None else sg_lo
    gas_vmax = args.vmax_gas  if args.vmax_gas  is not None else sg_hi

    T_flat = T_mw[np.isfinite(T_mw)]
    t_lo = np.nanpercentile(T_flat, 1.0)  if len(T_flat) else 1e3
    t_hi = np.nanpercentile(T_flat, 99.5) if len(T_flat) else 1e8
    temp_vmin = args.vmin_temp if args.vmin_temp is not None else max(t_lo, 10.0)
    temp_vmax = args.vmax_temp if args.vmax_temp is not None else min(t_hi, 1e8)

    sd_lo, sd_hi = safe_logbounds(dust_sigma)
    dust_vmin = args.vmin_dust if args.vmin_dust is not None else sd_lo
    dust_vmax = args.vmax_dust if args.vmax_dust is not None else sd_hi

    norm_gas  = LogNorm(vmin=gas_vmin,  vmax=gas_vmax)
    norm_temp = LogNorm(vmin=temp_vmin, vmax=temp_vmax)
    norm_dust = LogNorm(vmin=dust_vmin, vmax=dust_vmax)

    panels = [
        (gas_sigma,
         r'Gas column density $\Sigma_\mathrm{gas}$',
         'cividis', norm_gas,
         r'$\Sigma_\mathrm{gas}\ [\mathrm{M}_\odot\,\mathrm{pc}^{-2}]$',
         False),
        (T_mw,
         r'Mass-weighted temperature $\langle T \rangle_\mathrm{mw}$',
         'inferno', norm_temp,
         r'$\langle T \rangle_\mathrm{mw}\ \mathrm{[K]}$',
         False),
        (dust_sigma,
         r'Dust surface density $\Sigma_\mathrm{dust}$',
         teal_cmap(), norm_dust,
         r'$\Sigma_\mathrm{dust}\ [\mathrm{M}_\odot\,\mathrm{pc}^{-2}]$',
         False),
    ]

    if args.fourth_panel == 'dz' and dz_map is not None:
        dz_vmin = np.nanpercentile(dz_map[dz_map > 0], 1)
        dz_vmax = np.nanpercentile(dz_map[dz_map > 0], 99)
        panels.append((
            dz_map,
            r'Dust-to-metal ratio $\mathcal{D}/Z$',
            'magma', LogNorm(vmin=dz_vmin, vmax=dz_vmax),
            r'$\mathcal{D}/Z$',
            True,
        ))
    elif args.fourth_panel == 'abar' and abar_mw is not None:
        a_vmin = np.nanpercentile(abar_mw[abar_mw > 0], 1)
        a_vmax = np.nanpercentile(abar_mw[abar_mw > 0], 99)
        panels.append((
            abar_mw,
            r'Mean grain radius $\langle a \rangle_\mathrm{mw}$',
            'plasma', LogNorm(vmin=a_vmin, vmax=a_vmax),
            r'$\langle a \rangle_\mathrm{mw}\ \mathrm{[nm]}$',
            True,
        ))

    print('Rendering figure…')
    make_figure(
        panels, half_width, r200_pkpc,
        z=hdr['z'],
        rung=args.rung,
        axis=args.axis,
        show_r200_circle=not args.no_circle,
        out_path=args.out,
    )

    print('\n── Projection summary ──────────────────────────────────────────')
    print(f'  z                  = {hdr["z"]:.4f}')
    print(f'  R_200c             = {r200_pkpc:.1f} pkpc')
    print(f'  Half-width         = {half_width:.1f} pkpc')
    print(f'  Grid               = {args.npix}² px  ({pixel_kpc*1e3:.1f} pc/px)')
    if gas_mass_map[gas_mass_map > 0].size:
        print(f'  Σ_gas  range       = {gas_vmin:.2e} – {gas_vmax:.2e} M☉/pc²')
        print(f'  T_mw   range       = {temp_vmin:.2e} – {temp_vmax:.2e} K')
    if dust_pos2d is not None:
        print(f'  M_dust in slab     = {dust_mass_map.sum():.3e} M☉')
        print(f'  Σ_dust range       = {dust_vmin:.2e} – {dust_vmax:.2e} M☉/pc²')
    print('────────────────────────────────────────────────────────────────')


if __name__ == '__main__':
    main()
