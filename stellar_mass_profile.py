#!/usr/bin/env python3
"""
stellar_mass_profile.py
-----------------------
Computes the cumulative stellar mass profile for a single Gadget-4 run
at z~0 and finds the radius enclosing a given fraction of total stellar
mass within R200.  Use this to set R_ISM_PKPC in compare_grid_dust.py.

Usage:
    python stellar_mass_profile.py --run S10 --res 1024
    python stellar_mass_profile.py --run S10 --res 1024 --fractions 0.80 0.90 0.95
    python stellar_mass_profile.py --run S10 --res 1024 --out stellar_profile_S10.png

Output:
    - Printed table of enclosed-fraction radii
    - Plot: cumulative stellar mass fraction vs physical kpc
           with vertical lines at requested fractions
"""

import re
import glob
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py


# ─────────────────────────────────────────────────────────────────────────────
# Helpers (self-contained — no dependency on compare_grid_dust.py)
# ─────────────────────────────────────────────────────────────────────────────

def find_z0_snapshot(run, resolution):
    """Return (snap_base, catalog_path) for the snapshot nearest z=0."""
    output_dir = f'{run}_output_{resolution}'
    if not os.path.isdir(output_dir):
        raise FileNotFoundError(f'Output directory not found: {output_dir}')

    best_base, best_cat, best_z = None, None, 1e30
    for snapdir in sorted(glob.glob(os.path.join(output_dir, 'snapdir_*'))):
        snap_num_m = re.search(r'snapdir_(\d+)', snapdir)
        if not snap_num_m:
            continue
        snap_num = snap_num_m.group(1)

        # snapshot base (multi-file or single)
        cands = sorted(glob.glob(os.path.join(snapdir, 'snapshot_*.0.hdf5')))
        if not cands:
            cands = sorted(glob.glob(os.path.join(snapdir, 'snapshot_*.hdf5')))
        if not cands:
            continue
        snap_base = re.sub(r'(\.0)?\.hdf5$', '', cands[0])

        # catalog
        groups_dir = os.path.join(output_dir, f'groups_{snap_num}')
        cat_cands  = sorted(glob.glob(
            os.path.join(groups_dir, f'fof_subhalo_tab_{snap_num}.*.hdf5')))
        if not cat_cands:
            continue
        cat = cat_cands[0]

        # redshift
        try:
            with h5py.File(cands[0], 'r') as f:
                z = float(f['Header'].attrs['Redshift'])
        except Exception:
            continue

        if abs(z) < abs(best_z):
            best_z    = z
            best_base = snap_base
            best_cat  = cat

    if best_base is None:
        raise RuntimeError('No snapshot with catalog found.')
    print(f'  Using snapshot: {best_base}  (z={best_z:.4f})')
    print(f'  Catalog:        {best_cat}')
    return best_base, best_cat, best_z


def read_header(snap_base):
    """Read h, scale factor, UnitMass, UnitLength from snapshot."""
    for suffix in ['.0.hdf5', '.hdf5']:
        f = snap_base + suffix
        if not os.path.exists(f):
            continue
        with h5py.File(f, 'r') as hf:
            h   = float(hf['Parameters'].attrs['HubbleParam'])
            a   = float(hf['Header'].attrs['Time'])
            um  = float(hf['Parameters'].attrs.get('UnitMass_in_g',     1.989e43))
            ul  = float(hf['Parameters'].attrs.get('UnitLength_in_cm',  3.085678e21))
            return dict(h=h, a=a, um=um, ul=ul)
    raise RuntimeError(f'Cannot read header from {snap_base}')


def get_halo_center_and_r200(cat_path, hdr):
    """
    Read GroupPos and Group_R_Crit200 for the most massive group.
    Returns center_ckpc_h (comoving kpc/h), r200_pkpc (physical kpc).
    """
    # collect all chunks
    stem = re.sub(r'\.\d+\.hdf5$', '', cat_path)
    chunks = sorted(glob.glob(stem + '.*.hdf5'))
    if not chunks:
        chunks = [cat_path]

    for chunk in chunks:
        with h5py.File(chunk, 'r') as hf:
            if 'Group' not in hf:
                continue
            grp = hf['Group']
            if 'GroupPos' not in grp or grp['GroupPos'].shape[0] == 0:
                continue

            center_ckpc_h = grp['GroupPos'][0].astype(float)   # comoving kpc/h

            if 'Group_R_Crit200' in grp:
                r200_ckpc_h = float(grp['Group_R_Crit200'][0])
            elif 'Group_R_Mean200' in grp:
                r200_ckpc_h = float(grp['Group_R_Mean200'][0])
            else:
                raise KeyError('No R200 key found in catalog Group.')

            a, h = hdr['a'], hdr['h']
            r200_pkpc = r200_ckpc_h * a / h
            print(f'  Halo center (ckpc/h): {center_ckpc_h}')
            print(f'  R200 = {r200_ckpc_h:.1f} ckpc/h  =  {r200_pkpc:.1f} pkpc')
            return center_ckpc_h, r200_pkpc

    raise RuntimeError('No valid group found in catalog.')


def load_stars_within_r200(snap_base, center_ckpc_h, r200_pkpc, hdr, cat_path):
    """
    Load only PartType4 belonging to the central subhalo (SubFind subhalo 0
    of group 0).  This excludes satellite galaxies and the stellar halo,
    which would otherwise drag the cumulative profile out to CGM radii.
    """
    a, h  = hdr['a'], hdr['h']
    um    = hdr['um']
    msun_per_code = um / 1.989e33
    to_pkpc       = a / h

    # ── Read subhalo 0 membership from catalog ────────────────────────────
    stem   = re.sub(r'\.\d+\.hdf5$', '', cat_path)
    chunks = sorted(glob.glob(stem + '.*.hdf5'))
    if not chunks:
        chunks = [cat_path]

    offset, length = None, None
    for chunk in chunks:
        with h5py.File(chunk, 'r') as hf:
            if 'Subhalo' not in hf:
                continue
            sub = hf['Subhalo']
            if 'SubhaloLenType' not in sub or 'SubhaloOffsetType' not in sub:
                print('  WARNING: SubhaloLenType/OffsetType not found — '
                      'falling back to all stars within R200')
                break
            offset = int(sub['SubhaloOffsetType'][0, 4])  # subhalo 0, PartType4
            length = int(sub['SubhaloLenType'][0, 4])
            print(f'  Central subhalo: {length:,} star particles '
                  f'(SubFind offset {offset})')
            break

    # ── Load particles ────────────────────────────────────────────────────
    subfiles = sorted(glob.glob(snap_base + '.*.hdf5'))
    if not subfiles:
        single = snap_base + '.hdf5'
        subfiles = [single] if os.path.exists(single) else []

    # Collect ALL PartType4 in SubFind order first
    all_pos, all_mass = [], []
    for fname in subfiles:
        with h5py.File(fname, 'r') as hf:
            if 'PartType4' not in hf:
                continue
            all_pos.append(hf['PartType4']['Coordinates'][:])
            all_mass.append(hf['PartType4']['Masses'][:])

    if not all_pos:
        raise RuntimeError('No PartType4 found.')

    pos  = np.concatenate(all_pos)
    mass = np.concatenate(all_mass)

    # Apply SubFind membership cut if available
    if offset is not None and length is not None and length > 0:
        pos  = pos [offset : offset + length]
        mass = mass[offset : offset + length]
        print(f'  After SubFind cut: {len(pos):,} star particles')
    else:
        # Fallback: radial cut only
        r200_ckpc_h = r200_pkpc / to_pkpc
        r = np.linalg.norm(pos - center_ckpc_h, axis=1)
        mask = r < r200_ckpc_h
        pos  = pos[mask]
        mass = mass[mask]
        print(f'  Fallback radial cut: {len(pos):,} star particles')

    r_pkpc    = np.linalg.norm(pos - center_ckpc_h, axis=1) * to_pkpc
    mass_msun = mass * msun_per_code

    print(f'  Total stellar mass = {mass_msun.sum():.3e} M_sun')
    return r_pkpc, mass_msun


# ─────────────────────────────────────────────────────────────────────────────
# Profile + plot
# ─────────────────────────────────────────────────────────────────────────────

def compute_profile(r_pkpc, mass_msun, r200_pkpc, n_bins=200):
    """
    Return (r_centers, enclosed_fraction, shell_surface_density).
    r_centers in physical kpc; enclosed_fraction in [0,1].
    """
    r_edges   = np.linspace(0, r200_pkpc, n_bins + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])

    shell_mass, _ = np.histogram(r_pkpc, bins=r_edges, weights=mass_msun)
    enclosed      = np.cumsum(shell_mass)
    total         = mass_msun.sum()
    enclosed_frac = enclosed / total if total > 0 else np.zeros_like(enclosed)

    # Surface density: M_sun / kpc^2, projected annulus area
    annulus_area  = np.pi * (r_edges[1:]**2 - r_edges[:-1]**2)
    surf_dens     = shell_mass / np.where(annulus_area > 0, annulus_area, 1.0)

    return r_centers, enclosed_frac, surf_dens


def find_enclosed_radii(r_centers, enclosed_frac, fractions):
    """
    For each requested fraction, interpolate to find the enclosing radius.
    Returns dict fraction -> radius_pkpc.
    """
    result = {}
    for f in fractions:
        idx = np.searchsorted(enclosed_frac, f)
        if idx == 0:
            result[f] = r_centers[0]
        elif idx >= len(r_centers):
            result[f] = r_centers[-1]
        else:
            # linear interpolation between bracketing bins
            r0, r1 = r_centers[idx-1], r_centers[idx]
            f0, f1 = enclosed_frac[idx-1], enclosed_frac[idx]
            t = (f - f0) / (f1 - f0) if (f1 - f0) > 0 else 0.0
            result[f] = r0 + t * (r1 - r0)
    return result


def make_plot(run, resolution, z_snap, r200_pkpc,
              r_centers, enclosed_frac, surf_dens,
              fraction_radii, out_path):

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(7, 8),
        gridspec_kw=dict(hspace=0.08, height_ratios=[1.2, 1]))

    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(fraction_radii)))

    # ── Top: cumulative fraction ──────────────────────────────────────────────
    ax_top.plot(r_centers, enclosed_frac, color='#1f77b4', lw=2.2)
    ax_top.axvline(r200_pkpc, color='gray', ls=':', lw=1.2,
                   label=f'$R_{{200}}$ = {r200_pkpc:.0f} pkpc')

    for (frac, r_enc), c in zip(sorted(fraction_radii.items()), colors):
        ax_top.axvline(r_enc, color=c, ls='--', lw=1.5,
                       label=f'{frac*100:.0f}% → {r_enc:.1f} pkpc')
        ax_top.axhline(frac, color=c, ls='--', lw=0.8, alpha=0.5)
        ax_top.annotate(f'{r_enc:.1f} pkpc',
                        xy=(r_enc, frac),
                        xytext=(r_enc + r200_pkpc * 0.03, frac - 0.04),
                        fontsize=8, color=c,
                        arrowprops=dict(arrowstyle='->', color=c, lw=0.8))

    ax_top.set_ylabel('Enclosed stellar mass fraction', fontsize=11)
    ax_top.set_xlim(0, r200_pkpc)
    ax_top.set_ylim(0, 1.05)
    ax_top.grid(True, alpha=0.3)
    ax_top.legend(fontsize=8.5, loc='lower right')
    ax_top.tick_params(labelbottom=False)
    ax_top.set_title(
        f'{run} ({resolution}$^3$) — Stellar mass profile at $z={z_snap:.3f}$\n'
        f'$R_{{200}}$ = {r200_pkpc:.1f} pkpc, '
        f'Total $M_\\star$ = {enclosed_frac[-1]:.3f} (normalised)',
        fontsize=10)

    # ── Bottom: surface density ───────────────────────────────────────────────
    ax_bot.semilogy(r_centers, np.where(surf_dens > 0, surf_dens, np.nan),
                    color='#1f77b4', lw=2.2)
    ax_bot.axvline(r200_pkpc, color='gray', ls=':', lw=1.2)
    for (frac, r_enc), c in zip(sorted(fraction_radii.items()), colors):
        ax_bot.axvline(r_enc, color=c, ls='--', lw=1.5)

    ax_bot.set_xlabel('Galactocentric radius (physical kpc)', fontsize=11)
    ax_bot.set_ylabel(r'Stellar surface density ($M_\odot\,{\rm kpc}^{-2}$)',
                      fontsize=11)
    ax_bot.set_xlim(0, r200_pkpc)
    ax_bot.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\n  Plot saved: {out_path}')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Cumulative stellar mass profile to determine R_ISM')
    parser.add_argument('--run',  default='S10',
                        help='Run label, e.g. S10 (default: S10)')
    parser.add_argument('--res',  type=int, default=1024,
                        help='Resolution, e.g. 1024 (default: 1024)')
    parser.add_argument('--fractions', nargs='+', type=float,
                        default=[0.80, 0.90, 0.95],
                        help='Enclosed-mass fractions to mark (default: 0.80 0.90 0.95)')
    parser.add_argument('--n-bins', type=int, default=200,
                        help='Number of radial bins (default: 200)')
    parser.add_argument('--out', default=None,
                        help='Output plot path (default: stellar_profile_{run}_{res}.png)')
    args = parser.parse_args()

    if args.out is None:
        args.out = f'stellar_profile_{args.run}_{args.res}.png'

    print(f'\n=== Stellar mass profile: {args.run} at {args.res}^3 ===\n')

    snap_base, cat_path, z_snap = find_z0_snapshot(args.run, args.res)
    hdr                          = read_header(snap_base)
    center_ckpc_h, r200_pkpc     = get_halo_center_and_r200(cat_path, hdr)
    r_pkpc, mass_msun = load_stars_within_r200(snap_base, center_ckpc_h, r200_pkpc, hdr, cat_path)

    r_centers, enclosed_frac, surf_dens = compute_profile(
        r_pkpc, mass_msun, r200_pkpc, n_bins=args.n_bins)

    fraction_radii = find_enclosed_radii(r_centers, enclosed_frac, args.fractions)

    print('\n  ┌─────────────────────────────────────┐')
    print(  '  │  Enclosed stellar mass fractions    │')
    print(  '  ├──────────────┬──────────────────────┤')
    print(  '  │   Fraction   │   Radius (pkpc)      │')
    print(  '  ├──────────────┼──────────────────────┤')
    for frac in sorted(fraction_radii):
        r = fraction_radii[frac]
        print(f'  │    {frac*100:5.1f}%    │    {r:8.2f} pkpc         │')
    print(  '  └──────────────┴──────────────────────┘')
    print(f'\n  R200 = {r200_pkpc:.1f} pkpc')
    print(f'\n  Suggested R_ISM_PKPC values:')
    for frac in sorted(fraction_radii):
        r = fraction_radii[frac]
        print(f'    {frac*100:.0f}% enclosed → R_ISM_PKPC = {r:.1f}  '
              f'({r/r200_pkpc*100:.1f}% of R200)')

    make_plot(args.run, args.res, z_snap, r200_pkpc,
              r_centers, enclosed_frac, surf_dens,
              fraction_radii, args.out)

    print('\nDone.')


if __name__ == '__main__':
    main()
