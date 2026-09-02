#!/usr/bin/env python3
"""
plot_dust_histograms_sourcecoded.py

Create a grid of histograms showing dust particle properties from a Gadget-4 simulation,
with each histogram overlaid by dust source (SNII, AGB, or LRN).

Usage:
python plot_dust_histograms_sourcecoded.py --catalog ../groups_049/fof_subhalo_tab_049.0.hdf5 --snapshot ../snapdir_049/snapshot_049 --out dust_histograms_sourcecoded.png --rmax 300
"""

import argparse
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import h5py
import glob

try:
    from halo_utils import get_halo569_reference, get_halo569, load_particles_within_radius, convert_code_mass_to_msun
except ImportError:
    print("ERROR: This script requires halo_utils.py in the same directory")
    exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Dust source definitions
#
# CosmicGrain convention:
#   DustSource = 0 : SNII
#   DustSource = 1 : AGB
#   DustSource = 2 : LRN
# ─────────────────────────────────────────────────────────────────────────────
SOURCE_BINS = [
    (0, 'SNII', '#2196F3'),
    (1, 'AGB',  '#4CAF50'),
    (2, 'LRN',  '#FF5722'),
]
SOURCE_ALPHA = 0.55


# Fields to load for PartType6 (dust). load_particles_within_radius's own
# default field list (halo_utils._SPATIAL_FIELDS[6]) does NOT include
# DustFormationTime, which this script needs for the age panel, so it's
# requested explicitly here rather than relying on the default.
DUST_FIELDS = ["Coordinates", "Masses", "Velocities", "GrainRadius",
               "DustSource", "CarbonMassFraction", "DustTemperature",
               "DustFormationTime", "ParticleIDs"]


def get_snapshot_info(snapshot_base):
    files = sorted(glob.glob(f'{snapshot_base}.*.hdf5'))
    if not files:
        files = [f'{snapshot_base}.hdf5']
    with h5py.File(files[0], 'r') as f:
        header = f['Header'].attrs
        info = {}
        for key in ['Time', 'Redshift', 'Omega0', 'OmegaLambda']:
            if key in header:
                info[key] = float(header[key])
        # HubbleParam lives under Parameters, not Header -- see halo_utils
        # module docstring / project convention.
        if 'Parameters' in f and 'HubbleParam' in f['Parameters'].attrs:
            info['HubbleParam'] = float(f['Parameters'].attrs['HubbleParam'])
        if 'PartType6' in f:
            info['dust_fields'] = list(f['PartType6'].keys())
    return info


def scale_factor_to_age(a, h=0.7, Om=0.3, OL=0.7):
    from scipy.integrate import quad
    def integrand(a_prime):
        return 1.0 / (a_prime * np.sqrt(Om / a_prime**3 + OL))
    t_H = 9.778 / h
    age, _ = quad(integrand, 0, a)
    return age * t_H


def compute_velocity_magnitude(velocities):
    return np.sqrt(np.sum(velocities**2, axis=1))


def make_source_coded_histogram(ax, data, dust_source, xlabel, title,
                                  bins=50, log_x=False, color_all='#BBBBBB',
                                  xlim=None, weights=None):
    """
    Plot a histogram with the full dust population as a grey outline and
    SNII / AGB / LRN populations overlaid with shared bin edges.

    Parameters
    ----------
    weights : array or None
        Optional histogram weights. If None, the y axis is particle count.
        Passing dust mass gives a mass-weighted distribution.
    """
    data        = np.asarray(data)
    dust_source = np.asarray(dust_source)

    if weights is not None:
        weights = np.asarray(weights)

    finite = np.isfinite(data) & np.isfinite(dust_source)
    if weights is not None:
        finite &= np.isfinite(weights)

    data        = data[finite]
    dust_source = dust_source[finite]
    if weights is not None:
        weights = weights[finite]

    if len(data) == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes)
        ax.set_title(title)
        return

    if log_x:
        keep = data > 0
        data = data[keep]
        dust_source = dust_source[keep]
        if weights is not None:
            weights = weights[keep]

        if len(data) == 0:
            ax.text(0.5, 0.5, 'No positive data', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(title)
            return

        if xlim is not None:
            bin_edges = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), bins + 1)
        else:
            bin_edges = np.logspace(np.log10(data.min()), np.log10(data.max()), bins + 1)
    else:
        if xlim is not None:
            bin_edges = np.linspace(xlim[0], xlim[1], bins + 1)
        else:
            bin_edges = np.linspace(data.min(), data.max(), bins + 1)

    ax.hist(data, bins=bin_edges, weights=weights, histtype='step',
            color='#999', linewidth=1.2, label='All')

    for source_id, label, color in SOURCE_BINS:
        mask = dust_source == source_id
        if not np.any(mask):
            continue
        source_weights = weights[mask] if weights is not None else None
        ax.hist(data[mask], bins=bin_edges, weights=source_weights,
                color=color, alpha=SOURCE_ALPHA, edgecolor='none',
                label=f'{label}  (N={mask.sum():,})')

    if log_x:
        ax.set_xscale('log')

    if xlim is not None:
        ax.set_xlim(xlim)

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(r'Dust mass (M$_\odot$)' if weights is not None else 'Count',
                  fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)

    stats_text = (f'Median = {np.median(data):.2e}\n'
                  f'Mean   = {np.mean(data):.2e}')
    ax.text(0.03, 0.97, stats_text, transform=ax.transAxes,
            fontsize=9.5, va='top', ha='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))


def main():
    parser = argparse.ArgumentParser(
        description='Plot DustSource-coded dust histograms for target halo')
    parser.add_argument('--catalog',  required=True,
                        help='Path to Subfind catalog (fof_subhalo_tab_*.hdf5) -- kept for '
                             'backward-compatible invocation; the actual catalog directory '
                             'used is derived from --snapshot\'s snapdir_NNN parent.')
    parser.add_argument('--snapshot', required=True,
                        help='Base path to snapshot (e.g., .../snapdir_049/snapshot_049)')
    parser.add_argument('--out',      default='dust_histograms_sourcecoded.png')
    parser.add_argument('--rmax',     type=float, default=None,
                        help='Max radius for dust extraction, in ckpc/h (matching Coordinates '
                             'convention). Default: R200c (spherical-overdensity, from halo_utils).')
    parser.add_argument('--bins',     type=int,   default=50)
    parser.add_argument('--dpi',      type=int,   default=150)
    parser.add_argument('--figsize',  type=float, nargs=2, default=[16, 10])
    args = parser.parse_args()

    print("=" * 60)
    print("DUST HISTOGRAM PLOTTER  (source-coded)")
    print("=" * 60)

    snap_info    = get_snapshot_info(args.snapshot)
    current_time = snap_info.get('Time')
    redshift     = snap_info.get('Redshift')

    print(f"\nSnapshot: a={current_time:.6f}  z={redshift:.4f}" if current_time else "")
    if 'dust_fields' in snap_info:
        print(f"PartType6 fields: {snap_info['dust_fields']}")

    # ── Locate the halo via halo_utils (shrinking-sphere center + true SO R200) ─
    # args.snapshot is normally ".../snapdir_NNN/snapshot_NNN"; derive
    # output_dir (parent of snapdir_NNN/ and groups_NNN/) and snap_num from it,
    # so --catalog no longer needs to be parsed directly.
    print("\nLoading halo info...")
    snapdir_path = Path(args.snapshot).parent
    m = re.search(r'snapdir_(\d+)', snapdir_path.name)
    if not m:
        print(f"ERROR: could not parse a snapshot number from snapdir name "
              f"'{snapdir_path.name}' (expected e.g. 'snapdir_049')")
        return
    snap_num = int(m.group(1))
    output_dir = snapdir_path.parent
    groups_dir = output_dir / f"groups_{snap_num:03d}"

    # refine_center=False: use the frozen FOF/catalog center, matching every
    # other script in this pipeline (run_radial_evolution.py,
    # plot_mdust_mstar_all_halos.py, plot_gsd_comparison.py,
    # plot_radial_dgr.py all do the same). Without this, halo_utils' default
    # shrinking-sphere refinement can wander a large offset from the true
    # center on some snapshots, producing a spuriously small SO R200/M200 --
    # this is exactly what happened here: a 172 ckpc/h refined-center offset
    # at snap 047 dropped R200 from 115.3 pkpc to 66.1 pkpc and M200 from
    # 1.6e11 to 3.0e10 Msun, disagreeing with every other script's value for
    # the same halo at the same snapshot.
    ref = get_halo569_reference(output_dir, verbose=True, refine_center=False)
    halo = get_halo569(groups_dir, snap_num, ref, verbose=True, refine_center=False)
    if halo is None:
        print(f"ERROR: could not identify the target halo at snap {snap_num:03d}")
        return

    halo_pos  = halo["center"]          # ckpc/h
    halo_mass = halo["m200_msun"]       # already fully converted to Msun

    # R200c (ckpc/h) is the new default aperture -- the old default was
    # 2x the Subfind half-mass radius, which is no longer available from
    # halo_utils (which deliberately avoids Subhalo fields; see its module
    # docstring). R200c is the standard aperture used consistently by every
    # other CosmicGrain analysis script.
    rmax = args.rmax if args.rmax is not None else halo["r200_ckpch"]
    print(f"\nExtracting dust within {rmax:.2f} ckpc/h...")

    dust_data = load_particles_within_radius(
        snapdir_path, halo_pos, rmax, part_types=(6,), fields_by_type={6: DUST_FIELDS}
    ).get(6, {})

    if not dust_data or len(dust_data.get('Coordinates', [])) == 0:
        print("ERROR: No dust particles found!")
        return

    # ── Extract fields ───────────────────────────────────────────────────────
    grain_radius = dust_data['GrainRadius']
    carbon_frac  = dust_data['CarbonMassFraction']
    # FIX: the previous version did `dust_data['Masses'] * 1e10`, converting
    # code mass units to Msun but omitting the /h factor -- every mass value
    # (including the M200 shown in the figure subtitle) was off by a factor
    # of 1/h (~1.48x for h~0.6732). convert_code_mass_to_msun() applies the
    # full, correct conversion (* 1e10 / h), consistent with every other
    # script in this pipeline.
    masses       = convert_code_mass_to_msun(dust_data['Masses'], halo["h"])
    velocities   = dust_data['Velocities']
    dust_temp    = dust_data['DustTemperature']
    dust_source  = np.asarray(dust_data['DustSource']).astype(int)
    vel_mag      = compute_velocity_magnitude(velocities)

    # ── Formation time → age ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("COMPUTING DUST AGES")
    print("=" * 60)

    has_formation_time = False
    if 'DustFormationTime' in dust_data:
        dust_formation     = dust_data['DustFormationTime']
        has_formation_time = True
        print("✓ DustFormationTime")
    else:
        dust_formation = np.zeros(len(grain_radius))
        print("✗ No formation time — all ages set to 0")

    if has_formation_time and current_time is not None and np.any(dust_formation > 0):
        try:
            from scipy.integrate import quad  # noqa
            h  = snap_info.get('HubbleParam', halo["h"])
            Om = snap_info.get('Omega0',      0.3158)
            OL = snap_info.get('OmegaLambda', 0.6842)
            print(f"Cosmology: h={h}, Ωm={Om}, ΩΛ={OL}")

            current_age = scale_factor_to_age(current_time, h, Om, OL)
            print(f"Current age of universe: {current_age:.3f} Gyr")

            dust_age_gyr = np.zeros(len(dust_formation))
            for i, a_form in enumerate(dust_formation):
                if 0 < a_form <= current_time:
                    dust_age_gyr[i] = current_age - scale_factor_to_age(a_form, h, Om, OL)

            print(f"Age range: {dust_age_gyr.min():.3f} – {dust_age_gyr.max():.3f} Gyr")
            age_xlabel = 'Age (Gyr)'
            age_title  = 'Age'

        except ImportError:
            print("scipy not available — using linear approximation")
            dust_age_gyr = (current_time - dust_formation) * 13.8
            age_xlabel   = 'Age (approx, Gyr)'
            age_title    = 'Age (approx)'
    else:
        dust_age_gyr = np.zeros(len(grain_radius))
        age_xlabel   = 'Age (Gyr)'
        age_title    = 'Age'

    # ── Dust-source breakdown ────────────────────────────────────────────────
    print("\nDust source breakdown:")
    for source_id, label, color in SOURCE_BINS:
        mask = dust_source == source_id
        n = int(mask.sum())
        m = float(masses[mask].sum()) if n else 0.0
        frac_n = 100.0 * n / len(dust_source) if len(dust_source) else 0.0
        frac_m = 100.0 * m / masses.sum() if masses.sum() > 0 else 0.0
        print(f"  {label:4s}: N={n:,} ({frac_n:5.1f}%)  "
              f"M={m:.4e} Msun ({frac_m:5.1f}%)")

    # AGB-origin radius diagnostics: this directly tests whether 100-nm
    # injection grains have migrated to the expected ~33, 11, 3.6 nm
    # shattering descendants.
    agb = dust_source == 1
    if np.any(agb):
        print("\nAGB-origin grain-radius diagnostics:")
        agb_r = grain_radius[agb]
        agb_m = masses[agb]
        ranges = [
            (80.0, np.inf, "a >= 80 nm"),
            (25.0, 80.0, "25 <= a < 80 nm"),
            (8.0, 25.0, "8 <= a < 25 nm"),
            (0.0, 8.0, "a < 8 nm"),
        ]
        for lo, hi, label in ranges:
            mask_local = (agb_r >= lo) & (agb_r < hi)
            n = int(mask_local.sum())
            m = float(agb_m[mask_local].sum())
            print(f"  {label:18s}: N={n:7d}  "
                  f"({100*n/len(agb_r):6.2f}%)  "
                  f"M={m:.4e} Msun "
                  f"({100*m/agb_m.sum() if agb_m.sum()>0 else 0:6.2f}%)")

    # ── Figure ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=args.figsize)
    gs  = GridSpec(2, 3, figure=fig, hspace=0.30, wspace=0.3, top=0.83)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])

    # xlim=(lo, hi) pins both the bin range and axis limits;
    # None means derive limits from the data.
    panels = [
        (ax1, grain_radius, 'Grain Radius (nm)',   'Grain Radius',       False, None),
        (ax2, carbon_frac,  'Carbon Fraction',      'Carbon Fraction',    False, None),
        (ax3, masses,       'Mass (M$_\\odot$)',     'Mass',               True,  (1e-3, 1e5)),
        (ax4, vel_mag,      'Velocity (km/s)',       'Velocity Magnitude', False, (0, 500)),
        (ax5, dust_temp,    'Temperature (K)',       'Temperature',        False, (16, 22)),
        (ax6, dust_age_gyr, age_xlabel,              age_title,            False, None),
    ]

    for (ax, data, xlabel, title, log_x, xlim) in panels:
        make_source_coded_histogram(
            ax, data, dust_source,
            xlabel=xlabel, title=title,
            bins=args.bins, log_x=log_x, xlim=xlim
        )

    # Mark the AGB injection size and successive x0.33 shattering descendants.
    for radius, label in [
        (100.0, 'AGB birth 100 nm'),
        (33.0,  '1 shatter'),
        (11.0,  '2 shatters'),
        (3.6,   '3 shatters'),
    ]:
        ax1.axvline(radius, color='k', linestyle=':', linewidth=0.9, alpha=0.55)
    ax1.text(100.0, ax1.get_ylim()[1]*0.94, '100', ha='center', va='top', fontsize=7)
    ax1.text(33.0,  ax1.get_ylim()[1]*0.84, '33',  ha='center', va='top', fontsize=7)
    ax1.text(11.0,  ax1.get_ylim()[1]*0.74, '11',  ha='center', va='top', fontsize=7)
    ax1.text(3.6,   ax1.get_ylim()[1]*0.64, '3.6', ha='center', va='top', fontsize=7)

    # ── Shared legend ────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(color='#BBBBBB', alpha=0.55, label='All particles')
    ]
    for source_id, label, color in SOURCE_BINS:
        mask = dust_source == source_id
        legend_handles.append(
            mpatches.Patch(
                color=color, alpha=SOURCE_ALPHA,
                label=f'{label}  (N={mask.sum():,})'
            )
        )

    fig.legend(handles=legend_handles,
               loc='upper center', ncol=len(legend_handles),
               fontsize=9, framealpha=0.9,
               bbox_to_anchor=(0.5, 0.905))

    # ── Supra-figure titles ───────────────────────────────────────────────────
    rmax_pkpc = rmax * halo["a"] / halo["h"]
    fig.text(0.5, 0.985, 'Dust Properties',
             fontsize=16, fontweight='bold', ha='center', va='top')
    fig.text(0.5, 0.958,
             (f'Halo 569  $\\cdot$  '
              f'M$_{{200}}$={halo_mass:.2e} M$_\\odot$  $\\cdot$  '
              f'R$<${rmax_pkpc:.0f} pkpc  $\\cdot$  z={redshift:.2f}'),
             fontsize=10, ha='center', va='top')

    plt.savefig(args.out, dpi=args.dpi, bbox_inches='tight')
    print(f"\nSaved: {args.out}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Total dust particles:  {len(grain_radius):,}")
    print(f"Grain Radius (nm):     min={grain_radius.min():.2e}  max={grain_radius.max():.2e}  median={np.median(grain_radius):.2e}")
    print(f"Carbon Fraction:       min={carbon_frac.min():.3f}   max={carbon_frac.max():.3f}   median={np.median(carbon_frac):.3f}")
    print(f"Masses (Msun):         min={masses.min():.2e}  max={masses.max():.2e}  median={np.median(masses):.2e}")
    print(f"Velocity (km/s):       min={vel_mag.min():.1f}   max={vel_mag.max():.1f}   median={np.median(vel_mag):.1f}")
    print(f"Temperature (K):       min={dust_temp.min():.1f}   max={dust_temp.max():.1f}   median={np.median(dust_temp):.1f}")
    print(f"Age (Gyr):             min={dust_age_gyr.min():.3f}  max={dust_age_gyr.max():.3f}  median={np.median(dust_age_gyr):.3f}")
    for source_id, label, color in SOURCE_BINS:
        mask = dust_source == source_id
        if np.any(mask):
            print(f"{label:4s} radius (nm):        N={mask.sum():,}  "
                  f"median={np.median(grain_radius[mask]):.2f}  "
                  f"mean={np.mean(grain_radius[mask]):.2f}  "
                  f"Mdust={masses[mask].sum():.4e} Msun")


if __name__ == "__main__":
    main()
