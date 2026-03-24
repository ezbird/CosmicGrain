#!/usr/bin/env python3
"""
plot_dust_budget_pies.py

Parse a Gadget-4 dust log file and plot dust budget pie charts.

Layout:
  Row 1 (always):   Particle-count pies  — sourced from DESTRUCTION AUDIT in log
  Row 2 (optional): Mass-weighted pies   — sourced from a snapshot HDF5

  Left column:  Dust creation by channel (SNII vs AGB)
  Right column: Dust destruction by channel (Astration, Thermal, Shock, Internal)

The main title carries the redshift; individual pies show only channel breakdowns.

For the mass-weighted row the script reads PartType6/Masses + PartType6/GrainType
from the snapshot.  GrainType encoding: 1 → SNII, 2 → AGB.
Destruction mass is estimated as destroyed_count × mean_particle_mass.

Usage:
  python plot_dust_budget_pies.py <logfile> [options]

Options:
  --snapshot SNAP_BASE   Base path to snapshot, e.g. ../snapdir_049/snapshot_049
  --snap-index INT       Which DESTRUCTION AUDIT block to use (default -1 = last)
  --output FILE          Output image filename (default: dust_budget_pies.png)
  --unit-mass FLOAT      Code mass unit in M_sun (default: 1e10)

Examples:
  python plot_dust_budget_pies.py output_512.log
  python plot_dust_budget_pies.py output_512.log \\
      --snapshot ../snapdir_049/snapshot_049 --output dust_budget_z0.png
"""

import re
import argparse
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


# ─────────────────────────────────────────────────────────────────────────────
# Log parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_log(filepath):
    """Return list of dicts, one per DESTRUCTION AUDIT block in the log."""
    snapshots = []
    re_snii      = re.compile(r'Created by SNII:\s+([\d]+)')
    re_agb       = re.compile(r'Created by AGB:\s+([\d]+)')
    re_thermal   = re.compile(r'Thermal sputtering:\s+([\d]+)')
    re_shock     = re.compile(r'Shock destruction:\s+([\d]+)')
    re_astration = re.compile(r'Astration:\s+([\d]+)')
    re_cleanup   = re.compile(r'cleanup_invalid\(\):\s+([\d]+)')
    re_corrupt   = re.compile(r'growth corruption:\s+([\d]+)')
    re_badgas    = re.compile(r'bad gas index.*?:\s+([\d]+)')
    re_live      = re.compile(r'Current live particles:\s+([\d]+)')

    with open(filepath, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i]
        if '=== DESTRUCTION AUDIT' in line:
            mz = re.search(r'z=([\d.]+)', line)
            ma = re.search(r'a=([\d.]+)', line)
            snap = {
                'z': float(mz.group(1)) if mz else 0.0,
                'a': float(ma.group(1)) if ma else 0.0,
                'snii': 0, 'agb': 0,
                'thermal': 0, 'shock': 0, 'astration': 0,
                'cleanup': 0, 'corruption': 0, 'bad_gas': 0,
                'live': 0,
            }
            for j in range(i + 1, min(i + 40, len(lines))):
                l = lines[j]
                for key, pat in [
                    ('snii',       re_snii),  ('agb',        re_agb),
                    ('thermal',    re_thermal), ('shock',      re_shock),
                    ('astration',  re_astration), ('cleanup',    re_cleanup),
                    ('corruption', re_corrupt),  ('bad_gas',    re_badgas),
                    ('live',       re_live),
                ]:
                    m2 = pat.search(l)
                    if m2:
                        snap[key] = int(m2.group(1))
            snapshots.append(snap)
        i += 1
    return snapshots


# ─────────────────────────────────────────────────────────────────────────────
# Snapshot mass extraction
# ─────────────────────────────────────────────────────────────────────────────

def load_dust_masses_by_channel(snapshot_base, unit_mass_msun=1e10):
    if not HAS_H5PY:
        raise ImportError("h5py is required for --snapshot support")

    files = sorted(glob.glob(f'{snapshot_base}.*.hdf5'))
    if not files:
        files = sorted(glob.glob(f'{snapshot_base}.hdf5'))
    if not files:
        raise FileNotFoundError(
            f"No snapshot files found matching '{snapshot_base}[.N].hdf5'")

    print(f"  Found {len(files)} snapshot chunk(s).")
    all_masses, all_graintypes = [], []

    for fname in files:
        with h5py.File(fname, 'r') as f:
            if 'PartType6' not in f:
                continue
            pt6    = f['PartType6']
            masses = pt6['Masses'][:]
            alive  = masses > 0
            if alive.sum() == 0:
                continue
            all_masses.append(masses[alive])
            if 'GrainType' in pt6:
                all_graintypes.append(pt6['GrainType'][:][alive])
            else:
                print("  WARNING: GrainType not found – treating all as SNII.")
                all_graintypes.append(np.ones(alive.sum(), dtype=np.int32))

    if not all_masses:
        raise ValueError("No live PartType6 particles found in snapshot.")

    masses     = np.concatenate(all_masses) * unit_mass_msun
    graintypes = np.concatenate(all_graintypes)
    unique_gt  = np.unique(graintypes)
    print(f"  GrainType values found: {unique_gt}")

    snii_mask  = graintypes == 1
    agb_mask   = graintypes == 2
    other_mask = ~(snii_mask | agb_mask)

    snii_mass = masses[snii_mask].sum()
    agb_mass  = masses[agb_mask].sum()
    if other_mask.sum() > 0:
        print(f"  WARNING: {other_mask.sum()} particles have unexpected GrainType; lumped into AGB.")
        agb_mass += masses[other_mask].sum()

    total_mass = masses.sum()
    mean_pm    = total_mass / len(masses) if len(masses) > 0 else 0.0

    return {
        'snii_mass':          snii_mass,
        'agb_mass':           agb_mass,
        'total_mass':         total_mass,
        'n_live':             len(masses),
        'mean_particle_mass': mean_pm,
        'grain_types_found':  unique_gt,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────────────────

CREATION_COLORS    = ['#4C72B0', '#DD8452']
DESTRUCTION_COLORS = ['#C44E52', '#55A868', '#8172B2', '#937860']


def _fmt_mass(m):
    """
    Scientific notation with one digit before the decimal point.
    e.g.  941,000,000 M_sun  →  '9.4 × 10⁸ M☉'
          1,200,000   M_sun  →  '1.2 × 10⁶ M☉'
    """
    if m == 0:
        return '0 M☉'
    import math
    exp  = int(math.floor(math.log10(abs(m))))
    coeff = m / 10**exp
    # Round to 1 decimal place; if rounding pushes coeff to 10, adjust
    coeff_r = round(coeff, 1)
    if coeff_r >= 10:
        coeff_r /= 10
        exp += 1
    sup = str(exp).translate(str.maketrans('-0123456789', '⁻⁰¹²³⁴⁵⁶⁷⁸⁹'))
    return f'{coeff_r} × 10{sup} M☉'


def _fmt_count(n):
    return f'{int(round(n)):,}'


# ─────────────────────────────────────────────────────────────────────────────
# Pie chart helper
# ─────────────────────────────────────────────────────────────────────────────

def make_pie(ax, values, labels, colors, title, fmt_count=True, footnote=None):
    """
    Draw one pie chart.

    title      : string shown above the pie (one or two lines)
    fmt_count  : True → integer particles;  False → solar masses
    footnote   : optional small italic note placed below the legend
    """
    filtered = [(v, l, c) for v, l, c in zip(values, labels, colors) if v > 0]
    if not filtered:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes, fontsize=13, color='#444444')
        ax.set_title(title, fontsize=13, fontweight='bold', color='#111111')
        ax.axis('off')
        return

    vals, labs, cols = zip(*filtered)
    total = sum(vals)

    if fmt_count:
        autopct       = lambda pct: f'{pct:.0f}%'
        total_str     = f'Total: {_fmt_count(total)} particles'
        legend_labels = [f'{l}   ({_fmt_count(v)})' for l, v in zip(labs, vals)]
    else:
        autopct       = lambda pct: f'{pct:.0f}%'
        total_str     = f'Total: {_fmt_mass(total)}'
        legend_labels = [f'{l}   {_fmt_mass(v)}' for l, v in zip(labs, vals)]

    wedges, _, autotexts = ax.pie(
        vals,
        labels=None,
        colors=cols,
        autopct=autopct,
        startangle=140,
        pctdistance=0.68,
        wedgeprops=dict(linewidth=1.4, edgecolor='white'),
    )
    for at in autotexts:
        at.set_fontsize(14)
        at.set_color('white')
        at.set_fontweight('bold')

    # Title above pie
    ax.set_title(f'{title}\n{total_str}',
                 fontsize=13, fontweight='bold', pad=10, color='#111111')

    # Legend tightly below pie
    leg = ax.legend(
        wedges, legend_labels,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.03),
        fontsize=11,
        framealpha=0.95,
        edgecolor='#cccccc',
        ncol=1,
    )
    leg.get_frame().set_linewidth(0.8)

    # Footnote below legend (e.g. estimation method for mass row)
    if footnote:
        ax.text(0.5, -0.26,
                footnote,
                transform=ax.transAxes,
                ha='center', va='top',
                fontsize=8, style='italic', color='#666666',
                wrap=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_budget(snap, mass_data, output_path):
    nrows = 2 if mass_data is not None else 1

    fig, axes = plt.subplots(
        nrows, 2,
        figsize=(12, 7.5 * nrows),
        gridspec_kw={
            'wspace': 0.04,
            'hspace': 0.50 if nrows > 1 else 0.2,
        }
    )
    fig.patch.set_facecolor('white')

    if nrows == 1:
        axes = axes[np.newaxis, :]

    for row in axes:
        for ax in row:
            ax.set_facecolor('white')

    z = snap['z']

    # ── shared values ────────────────────────────────────────────────────────
    creation_vals    = [snap['snii'], snap['agb']]
    internal         = snap['cleanup'] + snap['corruption'] + snap['bad_gas']
    destruction_vals = [snap['astration'], snap['thermal'], snap['shock'], internal]

    # ── Row 1: particle counts ───────────────────────────────────────────────
    make_pie(axes[0, 0], creation_vals,
             ['Type II SN', 'AGB Stars'],
             CREATION_COLORS,
             'Dust Creation  (# particles)',
             fmt_count=True)

    make_pie(axes[0, 1], destruction_vals,
             ['Astration', 'Thermal Sputtering', 'Shock Destruction', 'Internal / Cleanup'],
             DESTRUCTION_COLORS,
             f'Dust Destruction  (# particles)   Live: {snap["live"]:,}',
             fmt_count=True)

    # ── Row 2: mass-weighted ─────────────────────────────────────────────────
    if mass_data is not None:
        make_pie(axes[1, 0],
                 [mass_data['snii_mass'], mass_data['agb_mass']],
                 ['Type II SN', 'AGB Stars'],
                 CREATION_COLORS,
                 'Dust Creation  (mass, live particles)',
                 fmt_count=False)

        mpm = mass_data['mean_particle_mass']
        dest_masses = [v * mpm for v in destruction_vals]
        make_pie(axes[1, 1], dest_masses,
                 ['Astration', 'Thermal Sputtering', 'Shock Destruction', 'Internal / Cleanup'],
                 DESTRUCTION_COLORS,
                 'Dust Destruction  (mass, estimated)',
                 fmt_count=False,
                 footnote=f'count × mean particle mass  ({_fmt_mass(mpm)})')

    # ── Main title ────────────────────────────────────────────────────────────
    fig.suptitle(f'Dust Budget  —  z = {z:.2f}',
                 fontsize=18, fontweight='bold', color='#111111', y=1.01)

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved: {output_path}')
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Plot dust creation/destruction budget pies')
    parser.add_argument('logfile',
                        help='Gadget-4 log containing DESTRUCTION AUDIT blocks')
    parser.add_argument('--snapshot', default=None,
                        help='Snapshot base path for mass-weighted row '
                             '(e.g. ../snapdir_049/snapshot_049)')
    parser.add_argument('--snap-index', type=int, default=-1,
                        help='Audit block index to plot (default: -1 = last)')
    parser.add_argument('--output', default='dust_budget_pies.png')
    parser.add_argument('--unit-mass', type=float, default=1e10,
                        help='Code mass unit in M_sun (default: 1e10)')
    args = parser.parse_args()

    print(f'Parsing log: {args.logfile}')
    snapshots = parse_log(args.logfile)
    if not snapshots:
        print('ERROR: No DESTRUCTION AUDIT blocks found in log.')
        sys.exit(1)
    print(f'Found {len(snapshots)} audit block(s).')

    snap = snapshots[args.snap_index]
    print(f'Using audit index {args.snap_index}: z={snap["z"]:.3f}')
    print(f'  SNII={snap["snii"]}  AGB={snap["agb"]}')
    print(f'  Astration={snap["astration"]}  Thermal={snap["thermal"]}  '
          f'Shock={snap["shock"]}  '
          f'Internal={snap["cleanup"]+snap["corruption"]+snap["bad_gas"]}')

    mass_data = None
    if args.snapshot:
        if not HAS_H5PY:
            print('WARNING: h5py not available – skipping mass-weighted row.')
        else:
            print(f'Loading dust masses from: {args.snapshot}')
            try:
                mass_data = load_dust_masses_by_channel(
                    args.snapshot, unit_mass_msun=args.unit_mass)
                print(f'  SNII mass : {_fmt_mass(mass_data["snii_mass"])}')
                print(f'  AGB  mass : {_fmt_mass(mass_data["agb_mass"])}')
                print(f'  Total live: {_fmt_mass(mass_data["total_mass"])}  '
                      f'({mass_data["n_live"]:,} particles)')
            except Exception as e:
                print(f'WARNING: Could not load snapshot masses: {e}')
                mass_data = None

    plot_budget(snap, mass_data, args.output)


if __name__ == '__main__':
    main()
