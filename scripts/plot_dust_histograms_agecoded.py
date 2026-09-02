#!/usr/bin/env python3
"""
plot_dust_histograms_agecoded.py

Create a grid of histograms showing dust particle properties from a Gadget-4 simulation,
with each histogram overlaid/stacked by dust particle age bin.

Usage:
python plot_dust_histograms_agecoded.py --catalog ../groups_049/fof_subhalo_tab_049.0.hdf5 --snapshot ../snapdir_049/snapshot_049 --out dust_histograms_agecoded.png --rmax 300
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
# Age bin definitions — drawn oldest-first so young sits on top
# ─────────────────────────────────────────────────────────────────────────────
AGE_BINS = [
    (0.0,   5.0,  'Young  (<5 Gyr)',    '#2196F3'),   # blue
    (5.0,  10.0,  'Mid    (5–10 Gyr)',  '#4CAF50'),   # green
    (10.0, 14.0,  'Old    (>10 Gyr)',   '#FF5722'),   # orange-red
]
AGE_BIN_ALPHA = 0.55

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


def make_age_coded_histogram(ax, data, age_gyr, xlabel, title,
                              bins=50, log_x=False, color_all='#BBBBBB',
                              xlim=None):
    """
    Plot a histogram with overlaid age-bin histograms.

    The full population is drawn in light grey first, then age bins are
    overlaid oldest-first so the youngest bin sits on top visually.

    Parameters
    ----------
    xlim : tuple (lo, hi) or None
        If given, fixes both the bin range and the displayed x-axis limits.
        For log_x panels, values are in data units (not log).
    """
    data    = np.asarray(data)
    age_gyr = np.asarray(age_gyr)

    finite  = np.isfinite(data) & np.isfinite(age_gyr)
    data    = data[finite]
    age_gyr = age_gyr[finite]

    if len(data) == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return

    # ── Build shared bin edges ────────────────────────────────────────────────
    if log_x:
        data_pos = data[data > 0]
        if len(data_pos) == 0:
            ax.text(0.5, 0.5, 'No positive data', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(title)
            return
        if xlim is not None:
            bin_edges = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), bins + 1)
        else:
            bin_edges = np.logspace(np.log10(data_pos.min()), np.log10(data_pos.max()), bins + 1)
        data_full = data_pos
        age_full  = age_gyr[data > 0]
    else:
        if xlim is not None:
            bin_edges = np.linspace(xlim[0], xlim[1], bins + 1)
        else:
            bin_edges = np.linspace(data.min(), data.max(), bins + 1)
        data_full = data
        age_full  = age_gyr

    # ── Full population (grey outline, behind everything) ─────────────────────
    ax.hist(data_full, bins=bin_edges, histtype='step',
            color='#999', linewidth=1.2, label='All')

    # ── Draw oldest bin first so younger bins layer on top ────────────────────
    for (a_lo, a_hi, label, color) in reversed(AGE_BINS):
        mask   = (age_full >= a_lo) & (age_full < a_hi)
        subset = data_full[mask]
        if len(subset) == 0:
            continue
        ax.hist(subset, bins=bin_edges, color=color, alpha=AGE_BIN_ALPHA,
                edgecolor='none', label=f'{label}  (N={len(subset):,})')

    if log_x:
        ax.set_xscale('log')

    if xlim is not None:
        ax.set_xlim(xlim)

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)

    # ── Stats box — top-left, larger font ────────────────────────────────────
    stats_text = (f'Median = {np.median(data_full):.2e}\n'
                  f'Mean   = {np.mean(data_full):.2e}')
    ax.text(0.03, 0.97, stats_text, transform=ax.transAxes,
            fontsize=9.5, va='top', ha='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))


def main():
    parser = argparse.ArgumentParser(
        description='Plot age-coded dust histograms for target halo')
    parser.add_argument('--catalog',  required=True,
                        help='Path to Subfind catalog (fof_subhalo_tab_*.hdf5) -- kept for '
                             'backward-compatible invocation; the actual catalog directory '
                             'used is derived from --snapshot\'s snapdir_NNN parent.')
    parser.add_argument('--snapshot', required=True,
                        help='Base path to snapshot (e.g., .../snapdir_049/snapshot_049)')
    parser.add_argument('--out',      default='dust_histograms_agecoded.png')
    parser.add_argument('--rmax',     type=float, default=None,
                        help='Max radius for dust extraction, in ckpc/h (matching Coordinates '
                             'convention). Default: R200c (spherical-overdensity, from halo_utils).')
    parser.add_argument('--bins',     type=int,   default=50)
    parser.add_argument('--dpi',      type=int,   default=150)
    parser.add_argument('--figsize',  type=float, nargs=2, default=[16, 10])
    args = parser.parse_args()

    print("=" * 60)
    print("DUST HISTOGRAM PLOTTER  (age-coded)")
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

    # ── Age bin breakdown ────────────────────────────────────────────────────
    print("\nAge bin breakdown:")
    for (a_lo, a_hi, label, color) in AGE_BINS:
        n = np.sum((dust_age_gyr >= a_lo) & (dust_age_gyr < a_hi))
        print(f"  {label}: {n:,} ({100*n/len(dust_age_gyr):.1f}%)")

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
        (ax5, dust_temp,    'Temperature (K)',       'Temperature',        False, (16, 19)),
        (ax6, dust_age_gyr, age_xlabel,              age_title,            False, None),
    ]

    for (ax, data, xlabel, title, log_x, xlim) in panels:
        make_age_coded_histogram(ax, data, dust_age_gyr,
                                  xlabel=xlabel, title=title,
                                  bins=args.bins, log_x=log_x, xlim=xlim)

    # ── Shared legend — listed young→old to match intuitive reading order ────
    legend_handles = [
        mpatches.Patch(color='#BBBBBB', alpha=0.55, label='All particles')
    ]
    for (a_lo, a_hi, label, color) in AGE_BINS:
        n = np.sum((dust_age_gyr >= a_lo) & (dust_age_gyr < a_hi))
        legend_handles.append(
            mpatches.Patch(color=color, alpha=AGE_BIN_ALPHA,
                           label=f'{label}  (N={n:,})')
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


if __name__ == "__main__":
    main()
