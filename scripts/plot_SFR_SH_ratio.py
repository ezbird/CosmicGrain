#!/usr/bin/env python3
"""
Two-panel plot: Stellar-to-Halo Mass Ratio and Absolute SFR
- Panel 1: M*/M_halo evolution with Behroozi+2013 comparison
- Panel 2: Absolute SFR of main halo vs observations of MW-mass progenitors

Uses halo_utils for proper halo identification at each snapshot via the
subfind group catalogs, rather than a static particle ID list.

Usage:
  python plot_SFR_SH_ratio.py <output_dir> [options]

Examples:
  python plot_SFR_SH_ratio.py ../2_output_1024/
  python plot_SFR_SH_ratio.py ../2_output_1024/ --rmax 300
  python plot_SFR_SH_ratio.py ../2_output_1024/ --rmax 300 --verbose
"""

import numpy as np
import h5py
import glob
import os
import sys
import argparse
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.integrate import quad

try:
    from halo_utils import load_target_halo, compute_radial_distance
except ImportError:
    print("ERROR: Cannot import halo_utils. Make sure halo_utils.py is in your Python path.")
    sys.exit(1)


# ============================================================
# Cosmology helpers
# ============================================================

def age_at_redshift(z, H0=67.4, Om=0.315, OL=0.685):
    """Calculate age of universe at redshift z in Gyr."""
    def integrand(zp):
        return 1.0 / ((1.0 + zp) * np.sqrt(Om * (1.0 + zp)**3 + OL))
    age_Gyr = (1.0 / H0) * quad(integrand, z, np.inf)[0] * 977.8
    return age_Gyr


def behroozi_2013_shmr(M_halo, z):
    """
    Stellar-to-halo mass ratio from Behroozi et al. 2013.

    Parameters
    ----------
    M_halo : float or array   Halo mass in M_sun
    z      : float or array   Redshift

    Returns
    -------
    M_star : stellar mass in M_sun
    ratio  : M_star / M_halo
    """
    a = 1.0 / (1.0 + z)

    log_M1  = 11.514 + (-1.793 * (a - 1.0)) + (-0.251 * z)
    epsilon = -1.777 + (-0.006 * (a - 1.0)) + (-0.000 * z) + (-0.119 * (a - 1.0))
    alpha   = -1.412 + (0.731  * (a - 1.0))
    delta   =  3.508 + (2.608  * (a - 1.0)) + (-0.043 * z)
    gamma   =  0.316 + (1.319  * (a - 1.0)) + ( 0.279 * z)

    x   = np.log10(M_halo) - log_M1
    f_x = (-np.log10(10.0 ** (alpha * x) + 1.0)
           + delta * (np.log10(1.0 + np.exp(x))) ** gamma
           / (1.0 + np.exp(10.0 ** (-x))))

    log_M_star = epsilon + log_M1 + f_x - np.log10(2.0)
    M_star = 10.0 ** log_M_star
    return M_star, M_star / M_halo


# ============================================================
# Catalog / snapshot discovery
# ============================================================

def find_group_catalog_for_snapshot(snap_dir, output_dir, catalog_prefix='fof_subhalo_tab'):
    """
    Given snapdir_NNN/, locate the matching group catalog file.
    Tries common Gadget-4 directory naming conventions.
    Returns path to first catalog file found, or None.
    """
    snap_num = ''.join(filter(str.isdigit, os.path.basename(os.path.normpath(snap_dir))))
    if not snap_num:
        return None

    candidates = [
        os.path.join(output_dir, f'groups_{snap_num}'),
        os.path.join(output_dir, f'group_tab_{snap_num}'),
        os.path.join(output_dir, f'fof_tab_{snap_num}'),
        output_dir,
    ]
    for cat_dir in candidates:
        if not os.path.exists(cat_dir):
            continue
        pattern = os.path.join(cat_dir, f'{catalog_prefix}_{snap_num}.*.hdf5')
        files = sorted(glob.glob(pattern))
        if files:
            return files[0]
    return None


def find_snapshot_base(snap_dir):
    """
    Return base path that load_target_halo expects (it appends .N.hdf5).
    E.g. .../snapdir_026/ -> .../snapdir_026/snapshot_026
    """
    snap_num = ''.join(filter(str.isdigit, os.path.basename(os.path.normpath(snap_dir))))
    return os.path.join(snap_dir, f'snapshot_{snap_num}')


# ============================================================
# Per-snapshot computation
# ============================================================

def process_snapshot(snap_dir, output_dir, rmax_kpc=None,
                     catalog_prefix='fof_subhalo_tab', verbose=False):
    """
    Load halo properties + SFR for a single snapshot using halo_utils.

    rmax_kpc is applied as a post-filter on particle coordinates after
    load_target_halo returns, so it works regardless of what kwargs
    load_target_halo accepts internally.

    Returns dict or None on failure.
    """
    catalog_file = find_group_catalog_for_snapshot(snap_dir, output_dir, catalog_prefix)
    if catalog_file is None:
        if verbose:
            print(f"  [SKIP] No group catalog found for {snap_dir}")
        return None

    snap_base  = find_snapshot_base(snap_dir)
    snap_files = sorted(glob.glob(snap_base + '.*.hdf5'))
    if not snap_files:
        snap_files = sorted(glob.glob(snap_base + '.hdf5'))
    if not snap_files:
        if verbose:
            print(f"  [SKIP] No snapshot files found at {snap_base}")
        return None

    # Redshift from header
    with h5py.File(snap_files[0], 'r') as f:
        redshift = float(f['Header'].attrs['Redshift'])

    try:
        # Load without passing rmax — apply as post-filter below.
        # This avoids TypeError if load_target_halo does not accept rmax as a kwarg.
        halo = load_target_halo(
            catalog_file,
            snap_base,
            particle_types=[0, 1, 4],   # Gas, DM, Stars
            verbose=verbose,
        )
    except Exception as e:
        if verbose:
            print(f"  [WARN] load_target_halo failed for {snap_dir}: {e}")
        return None

    halo_info = halo.get('halo_info', {})
    halo_pos  = halo_info.get('position', None)

    # ---- Apply rmax as post-filter on coordinates ----
    if rmax_kpc is not None and halo_pos is not None:
        for ptype_key in ('gas', 'stars', 'dm'):
            pdata = halo.get(ptype_key)
            if pdata is None:
                continue
            coords = pdata.get('Coordinates')
            if coords is None or len(coords) == 0:
                continue
            r    = compute_radial_distance(coords, halo_pos)
            mask = r <= rmax_kpc
            halo[ptype_key] = {
                k: v[mask]
                for k, v in pdata.items()
                if hasattr(v, '__len__') and len(v) == len(r)
            }

    # ---- Stellar mass (code units × 1e10 -> M_sun) ----
    stellar_mass = 0.0
    if halo.get('stars') is not None:
        star_masses  = halo['stars'].get('Masses', np.array([]))
        stellar_mass = float(np.sum(star_masses)) * 1e10

    # ---- Halo mass from catalog ----
    raw_halo_mass = halo_info.get('mass', 0.0)
    halo_mass     = float(raw_halo_mass) * 1e10

    # ---- SFR from star-forming gas ----
    sfr = 0.0
    if halo.get('gas') is not None:
        gas_sfr = halo['gas'].get('StarFormationRate', np.array([]))
        sfr     = float(np.sum(gas_sfr[gas_sfr > 0]))

    ratio = stellar_mass / halo_mass if halo_mass > 0 else 0.0

    return {
        'redshift':     redshift,
        'stellar_mass': stellar_mass,
        'halo_mass':    halo_mass,
        'ratio':        ratio,
        'sfr':          sfr,
    }


def process_all_snapshots(output_dir, rmax_kpc=None,
                          catalog_prefix='fof_subhalo_tab', verbose=True):
    """
    Process all snapdir_NNN/ directories. Returns data dict sorted by
    descending redshift (high-z first).
    """
    snap_dirs = sorted(glob.glob(os.path.join(output_dir, 'snapdir_*')))
    if not snap_dirs:
        print(f"ERROR: No snapdir_* directories found in {output_dir}")
        sys.exit(1)

    print(f"\nProcessing {len(snap_dirs)} snapshots via halo_utils...")

    records = []
    for snap_dir in snap_dirs:
        result = process_snapshot(snap_dir, output_dir,
                                  rmax_kpc=rmax_kpc,
                                  catalog_prefix=catalog_prefix,
                                  verbose=verbose)
        if result is None:
            continue

        records.append(result)
        print(f"  z={result['redshift']:.3f} | "
              f"M*={result['stellar_mass']:.2e} Msun | "
              f"Mhalo={result['halo_mass']:.2e} Msun | "
              f"SFR={result['sfr']:.2f} Msun/yr")

    if not records:
        print("ERROR: No snapshots could be processed.")
        sys.exit(1)

    records.sort(key=lambda r: r['redshift'], reverse=True)

    return {
        'redshift':     np.array([r['redshift']     for r in records]),
        'stellar_mass': np.array([r['stellar_mass'] for r in records]),
        'halo_mass':    np.array([r['halo_mass']    for r in records]),
        'ratio':        np.array([r['ratio']        for r in records]),
        'sfr':          np.array([r['sfr']          for r in records]),
    }


# ============================================================
# Plotting
# ============================================================

def plot_two_panel(data, output_file='halo_evolution.png', run_name='', rmax_kpc=None):
    """
    Two-panel figure sharing the redshift x-axis.
      Top    : Stellar-to-halo mass ratio
      Bottom : Absolute SFR vs MW-mass progenitor observations

    Panel titles appear inside each panel at the bottom-right.
    Panels are drawn close together since they share the x-axis.

    Observational MW-mass SFR range based on:
      van Dokkum et al. (2013), ApJ 771, L35
      Patel et al. (2013), ApJ 766, 15
    """
    fig = plt.figure(figsize=(9, 9))
    gs  = GridSpec(2, 1, hspace=0.05,
                   top=0.91, bottom=0.09,
                   left=0.12, right=0.96,
                   figure=fig)

    if run_name:
        rmax_label = f'  |  rmax = {rmax_kpc:.0f} kpc' if rmax_kpc else ''
        fig.suptitle(f'{run_name}{rmax_label}',
                     fontsize=13, fontweight='bold', color='green', y=0.975)

    z    = data['redshift']
    z_lo = 0.0
    z_hi = float(np.ceil(z.max()))

    # ----------------------------------------------------------------
    # TOP PANEL — Stellar-to-Halo Mass Ratio
    # ----------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0])

    ax1.semilogy(z, data['ratio'], 'b-', linewidth=2,
                 label='Simulation (Halo 569)')

    _, beh_ratio = behroozi_2013_shmr(data['halo_mass'], z)
    final_mhalo  = data['halo_mass'][-1]
    ax1.semilogy(z, beh_ratio, 'r--', linewidth=2, alpha=0.8,
                 label=(r'Behroozi+2013 '
                        r'($M_h = {:.1f}\times10^{{12}}\ \mathrm{{M}}_\odot$)'
                        .format(final_mhalo / 1e12)))

    ax1.axhline(0.02, color='gray', linestyle=':', alpha=0.6,
                label='Peak efficiency (~2%)')

    ax1.set_ylabel(r'$M_\star / M_{\rm halo}$', fontsize=12)
    ax1.set_xlim(z_hi, z_lo)
    ax1.set_ylim(1e-4, 0.1)
    ax1.legend(fontsize=9, loc='upper left')
    ax1.grid(True, which='both', alpha=0.3)
    ax1.tick_params(axis='x', labelbottom=False)   # shared x-axis: hide tick labels

    # In-panel title — bottom right
    ax1.text(0.98, 0.05, 'Stellar-to-Halo Mass Ratio',
             transform=ax1.transAxes, fontsize=11, fontweight='bold',
             ha='right', va='bottom',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                       alpha=0.75, edgecolor='none'))

    # Age-of-universe axis on top
    ax1_top   = ax1.twiny()
    z_ticks   = np.array([t for t in ax1.get_xticks() if z_lo <= t <= z_hi])
    age_ticks = [age_at_redshift(zt) for zt in z_ticks]
    ax1_top.set_xlim(ax1.get_xlim())
    ax1_top.set_xticks(z_ticks)
    ax1_top.set_xticklabels([f'{a:.1f}' for a in age_ticks], fontsize=9)
    ax1_top.set_xlabel('Age of Universe (Gyr)', fontsize=11)

    # ----------------------------------------------------------------
    # BOTTOM PANEL — Absolute SFR
    # ----------------------------------------------------------------
    ax2 = fig.add_subplot(gs[1])

    ax2.semilogy(z, data['sfr'], 'b-o', linewidth=2, markersize=5,
                 label='Simulation (Halo 569)')

    # MW-mass progenitor SFR range
    # van Dokkum et al. 2013 (ApJ 771, L35) + Patel et al. 2013 (ApJ 766, 15)
    obs_z   = np.array([0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    obs_sfr = np.array([1.5, 3.0, 7.0, 15.0, 25.0, 20.0, 15.0, 10.0, 8.0])
    ax2.fill_between(obs_z, obs_sfr * 0.5, obs_sfr * 2.0,
                     color='gray', alpha=0.3)
    ax2.plot(obs_z, obs_sfr, 'k--', linewidth=2, alpha=0.8,
             label=('MW-mass progenitors\n'
                    'van Dokkum+2013; Patel+2013'))

    ax2.set_xlabel('Redshift', fontsize=13)
    ax2.set_ylabel(r'SFR ($\mathrm{M}_\odot\,\mathrm{yr}^{-1}$)', fontsize=12)
    ax2.set_xlim(z_hi, z_lo)
    ax2.set_ylim(0.1, 200)
    ax2.legend(fontsize=9, loc='upper left')
    ax2.grid(True, which='both', alpha=0.3)

    # In-panel title — bottom right
    ax2.text(0.98, 0.05, 'Absolute Star Formation Rate',
             transform=ax2.transAxes, fontsize=11, fontweight='bold',
             ha='right', va='bottom',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                       alpha=0.75, edgecolor='none'))

    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✅ Plot saved to: {output_file}")
    plt.show()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Two-panel halo evolution plot using halo_utils.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('output_dir',
                        help='Simulation output directory (contains snapdir_* and groups_*)')
    parser.add_argument('--rmax', type=float, default=None,
                        help='Extraction radius in kpc (default: 2 × halfmass_rad from catalog)')
    parser.add_argument('--catalog-prefix', default='fof_subhalo_tab',
                        help='Prefix for group catalog files')
    parser.add_argument('--out', default=None,
                        help='Output figure filename')
    parser.add_argument('--verbose', action='store_true',
                        help='Print per-snapshot halo_utils output')
    args = parser.parse_args()

    output_dir = os.path.normpath(args.output_dir)
    if not os.path.isdir(output_dir):
        print(f"ERROR: Directory not found: {output_dir}")
        sys.exit(1)

    run_name   = os.path.basename(output_dir)
    output_fig = args.out or os.path.join(output_dir, 'halo_evolution.png')

    print("\n" + "=" * 70)
    print("🌌  Halo Evolution Analysis (2-Panel, halo_utils edition) 🌌")
    print("=" * 70)
    print(f"  Output dir      : {output_dir}")
    print(f"  Catalog prefix  : {args.catalog_prefix}")
    rmax_str = f'{args.rmax} kpc' if args.rmax else '2 × halfmass_rad (default)'
    print(f"  Extraction rmax : {rmax_str}")
    print(f"  Figure output   : {output_fig}")
    print("=" * 70)

    data = process_all_snapshots(
        output_dir,
        rmax_kpc=args.rmax,
        catalog_prefix=args.catalog_prefix,
        verbose=args.verbose,
    )

    plot_two_panel(data, output_file=output_fig, run_name=run_name, rmax_kpc=args.rmax)

    print("\n" + "=" * 70)
    print("✅  Done!")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
