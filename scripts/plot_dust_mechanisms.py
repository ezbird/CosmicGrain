#!/usr/bin/env python3
"""
Two-panel plot: Dust Creation and Destruction Mechanisms vs Redshift

Panel 1: Cumulative dust creation by channel (SNII vs AGB)
Panel 2: Cumulative dust destruction by mechanism (thermal sputtering,
         shock destruction, astration, internal/corruption)

DATA SOURCES (two modes, automatically selected):
  1. Log file parsing  — reads DESTRUCTION AUDIT blocks and DUST_CREATE lines
                         from simulation stdout/log files. Most complete.
  2. Snapshot fallback — reads PartType6 GrainType from HDF5 snapshots to
                         infer the creation channel mix (live population only,
                         not rates).

For full creation channel tracking, add these counters to dust.cc:
    long long NDustCreatedBySNII = 0;
    long long NDustCreatedByAGB  = 0;
and increment them in create_dust_particles_from_feedback().
Then they will be picked up automatically if printed to the log.

Usage:
  python plot_dust_mechanisms.py <output_dir> [options]

Examples:
  python plot_dust_mechanisms.py ../2_output_1024/
  python plot_dust_mechanisms.py ../2_output_1024/ --log stdout.txt
  python plot_dust_mechanisms.py ../2_output_1024/ --log stdout.txt --cumulative
  python plot_dust_mechanisms.py ../2_output_1024/ --log stdout.txt --rates
"""

import numpy as np
import h5py
import glob
import os
import re
import sys
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec


# ============================================================
# Log file parsing
# ============================================================

# Matches lines like:
# [DUST|T=0|a=0.49525 z=1.019]   Thermal sputtering:     2847  (flag=1)
AUDIT_PATTERNS = {
    'z':         re.compile(r'\|a=[\d.]+ z=([\d.]+)\].*DESTRUCTION AUDIT'),
    'thermal':   re.compile(r'Thermal sputtering:\s+(\d+)'),
    'shock':     re.compile(r'Shock destruction:\s+(\d+)'),
    'astration': re.compile(r'Astration:\s+(\d+)'),
    'cleanup':   re.compile(r'cleanup_invalid\(\):\s+(\d+)'),
    'corruption':re.compile(r'growth corruption:\s+(\d+)'),
    'bad_gas':   re.compile(r'bad gas index.*?:\s+(\d+)'),
    'created':   re.compile(r'Total created:\s+(\d+)'),
    'live':      re.compile(r'Current live particles:\s+(\d+)'),
}

# Optional: if you add explicit SN/AGB creation counters to print_dust_statistics
CREATION_PATTERNS = {
    'snii': re.compile(r'Created by SNII:\s+(\d+)'),
    'agb':  re.compile(r'Created by AGB:\s+(\d+)'),
}


def parse_log_file(log_path, verbose=False):
    """
    Parse simulation log for DESTRUCTION AUDIT blocks.
    Returns list of dicts, one per audit block found, sorted by ascending z.
    Each dict has keys: z, thermal, shock, astration, cleanup, corruption,
                        bad_gas, created, live, [snii, agb if present]
    """
    if not os.path.exists(log_path):
        print(f"ERROR: Log file not found: {log_path}")
        return []

    print(f"Parsing log file: {log_path}")

    records = []
    current = None

    with open(log_path, 'r', errors='replace') as fh:
        for line in fh:

            # Start of a new audit block
            if 'DESTRUCTION AUDIT' in line:
                m = AUDIT_PATTERNS['z'].search(line)
                if m:
                    current = {'z': float(m.group(1))}
                continue

            if current is None:
                continue

            # Check end of block
            if '=======================================' in line and 'z' in current:
                # Check we have at minimum the required fields
                if 'thermal' in current:
                    records.append(current)
                    if verbose:
                        print(f"  Parsed audit block at z={current['z']:.3f}: "
                              f"thermal={current.get('thermal',0)} "
                              f"shock={current.get('shock',0)} "
                              f"astration={current.get('astration',0)} "
                              f"created={current.get('created',0)}")
                current = None
                continue

            # Parse fields inside the block
            for key, pat in AUDIT_PATTERNS.items():
                if key == 'z':
                    continue
                m = pat.search(line)
                if m and key not in current:
                    current[key] = int(m.group(1))

            # Optional creation channel breakdown
            for key, pat in CREATION_PATTERNS.items():
                m = pat.search(line)
                if m and key not in current:
                    current[key] = int(m.group(1))

    # Sort high-z → low-z (descending redshift = chronological order)
    records.sort(key=lambda r: r['z'], reverse=True)
    print(f"  Found {len(records)} audit blocks spanning "
          f"z={records[-1]['z']:.2f}–{records[0]['z']:.2f}"
          if records else "  No audit blocks found.")
    return records


def cumulative_to_rates(z_arr, counts_arr):
    """
    Convert cumulative counts to per-dz rates.
    Returns (z_mid, rate) arrays for plotting as a differential.
    """
    if len(z_arr) < 2:
        return z_arr, counts_arr
    # Work in ascending z order (high-z first = index 0)
    dz    = np.abs(np.diff(z_arr))
    dcnt  = np.abs(np.diff(counts_arr))
    rate  = np.where(dz > 0, dcnt / dz, 0.0)
    z_mid = 0.5 * (z_arr[:-1] + z_arr[1:])
    return z_mid, rate


# ============================================================
# Snapshot-based grain type census
# ============================================================

def snapshot_grain_type_census(output_dir, verbose=False):
    """
    Walk all snapdir_NNN directories and count live PartType6 grains
    by GrainType (0=SNII, 1=AGB, 2=generic/unset) at each snapshot.

    Returns list of dicts: {z, n_sn, n_agb, n_generic, n_total}
    """
    snap_dirs = sorted(glob.glob(os.path.join(output_dir, 'snapdir_*')))
    if not snap_dirs:
        return []

    print(f"\nReading PartType6 GrainType from {len(snap_dirs)} snapshots...")
    records = []

    for snap_dir in snap_dirs:
        snap_num = ''.join(filter(str.isdigit,
                                  os.path.basename(os.path.normpath(snap_dir))))
        snap_files = sorted(glob.glob(
            os.path.join(snap_dir, f'snapshot_{snap_num}.*.hdf5')))
        if not snap_files:
            snap_files = sorted(glob.glob(
                os.path.join(snap_dir, f'snapshot_{snap_num}.hdf5')))
        if not snap_files:
            continue

        try:
            with h5py.File(snap_files[0], 'r') as f:
                z = float(f['Header'].attrs['Redshift'])

                if 'PartType6' not in f:
                    records.append({'z': z, 'n_sn': 0, 'n_agb': 0,
                                    'n_generic': 0, 'n_total': 0})
                    continue

                masses = f['PartType6/Masses'][:] if 'Masses' in f['PartType6'] else np.array([])
                live   = masses > 1e-20

                if 'GrainType' in f['PartType6']:
                    gt = f['PartType6/GrainType'][:][live]
                else:
                    # GrainType not saved — count all live as generic
                    gt = np.full(int(np.sum(live)), 2, dtype=int)

                n_sn      = int(np.sum(gt == 0))
                n_agb     = int(np.sum(gt == 1))
                n_generic = int(np.sum(gt == 2))
                n_total   = len(gt)

                records.append({'z': z, 'n_sn': n_sn, 'n_agb': n_agb,
                                'n_generic': n_generic, 'n_total': n_total})

                if verbose:
                    print(f"  z={z:.3f}: total={n_total}  SN={n_sn}  "
                          f"AGB={n_agb}  generic={n_generic}")

        except Exception as e:
            if verbose:
                print(f"  [WARN] {snap_dir}: {e}")
            continue

    records.sort(key=lambda r: r['z'], reverse=True)
    return records


# ============================================================
# Plotting
# ============================================================

COLORS = {
    'thermal':    '#e74c3c',   # red
    'shock':      '#e67e22',   # orange
    'astration':  '#9b59b6',   # purple
    'internal':   '#95a5a6',   # gray
    'snii':       '#2980b9',   # blue
    'agb':        '#27ae60',   # green
    'total_dest': '#2c3e50',   # dark navy
    'total_cre':  '#16a085',   # teal
}


def plot_mechanisms(log_records, snap_records,
                    output_file='dust_mechanisms.png',
                    run_name='', mode='cumulative', show_plot=0):
    """
    Two-panel figure.
      Top    : Dust creation by channel vs redshift
      Bottom : Dust destruction by mechanism vs redshift

    mode : 'cumulative' | 'rates'
    """
    has_log   = len(log_records) > 0
    has_snaps = len(snap_records) > 0

    fig = plt.figure(figsize=(10, 10))
    gs  = GridSpec(2, 1, hspace=0.08,
                   top=0.91, bottom=0.09,
                   left=0.12, right=0.96,
                   figure=fig)

    if run_name:
        fig.suptitle(run_name, fontsize=13, fontweight='bold',
                     color='green', y=0.975)

    # Determine x-range from available data
    all_z = []
    if has_log:
        all_z += [r['z'] for r in log_records]
    if has_snaps:
        all_z += [r['z'] for r in snap_records]
    z_lo = 0.0
    z_hi = 20.0

    ylabel = ('Cumulative particle count' if mode == 'cumulative'
              else r'Rate  [particles per $\Delta z$]')

    # ----------------------------------------------------------------
    # TOP PANEL — Creation mechanisms
    # ----------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0])

    if has_log:
        z_arr = np.array([r['z'] for r in log_records])

        # Total created (always available)
        total_cre = np.array([r.get('created', 0) for r in log_records],
                             dtype=float)

        # SN/AGB split only if logged separately
        has_split = all('snii' in r or 'agb' in r for r in log_records)

        if mode == 'rates':
            z_plot, total_cre = cumulative_to_rates(z_arr, total_cre)
            if has_split:
                snii_arr = np.array([r.get('snii', 0) for r in log_records], float)
                agb_arr  = np.array([r.get('agb',  0) for r in log_records], float)
                _, snii_arr = cumulative_to_rates(z_arr, snii_arr)
                _, agb_arr  = cumulative_to_rates(z_arr, agb_arr)
        else:
            z_plot = z_arr
            if has_split:
                snii_arr = np.array([r.get('snii', 0) for r in log_records], float)
                agb_arr  = np.array([r.get('agb',  0) for r in log_records], float)

        ax1.semilogy(z_plot, np.maximum(total_cre, 0.5),
                     color=COLORS['total_cre'], lw=2.5, ls='-',
                     label='Total created (log)', zorder=5)

        if has_split:
            ax1.semilogy(z_plot, np.maximum(snii_arr, 0.5),
                         color=COLORS['snii'], lw=1.8, ls='--',
                         label='Type II SN', zorder=4)
            ax1.semilogy(z_plot, np.maximum(agb_arr, 0.5),
                         color=COLORS['agb'], lw=1.8, ls='--',
                         label='AGB', zorder=4)
        else:
            ax1.text(0.5, 0.55,
                     'SN/AGB split not logged.\nAdd NDustCreatedBySNII/AGB counters\n'
                     'to print_dust_statistics() in dust.cc',
                     transform=ax1.transAxes, fontsize=9, color='gray',
                     ha='center', va='center',
                     bbox=dict(boxstyle='round', facecolor='lightyellow',
                               alpha=0.8, edgecolor='orange'))

    if has_snaps:
        z_s   = np.array([r['z'] for r in snap_records])
        n_sn  = np.array([r['n_sn']  for r in snap_records], float)
        n_agb = np.array([r['n_agb'] for r in snap_records], float)
        n_gen = np.array([r['n_generic'] for r in snap_records], float)
        n_tot = np.array([r['n_total']   for r in snap_records], float)

        label_suffix = ' (live, snaps)'
        ax1.semilogy(z_s, np.maximum(n_tot, 0.5),
                     color=COLORS['total_cre'], lw=2.2, ls=':', alpha=0.7,
                     label=f'Total live{label_suffix}')
        if np.any(n_sn > 0):
            ax1.semilogy(z_s, np.maximum(n_sn, 0.5),
                         color=COLORS['snii'], lw=2.2, ls=':', alpha=0.7,
                         label=f'SN grains{label_suffix}')
        if np.any(n_agb > 0):
            ax1.semilogy(z_s, np.maximum(n_agb, 0.5),
                         color=COLORS['agb'], lw=2.2, ls=':', alpha=0.7,
                         label=f'AGB grains{label_suffix}')
        if np.any(n_gen > 0):
            ax1.semilogy(z_s, np.maximum(n_gen, 0.5),
                         color='#bdc3c7', lw=2.0, ls=':', alpha=0.6,
                         label=f'Generic grains{label_suffix}')

    ax1.set_ylabel(ylabel, fontsize=11)
    ax1.set_xlim(z_hi, z_lo)
    ax1.legend(fontsize=11, loc='upper left', framealpha=0.85)
    ax1.grid(True, which='both', alpha=0.25)
    ax1.tick_params(axis='x', labelbottom=False)
    ax1.text(0.02, 0.05, 'Dust Creation Channels',
             transform=ax1.transAxes, fontsize=11, fontweight='bold',
             ha='left', va='bottom',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                       alpha=0.8, edgecolor='none'))

    # ----------------------------------------------------------------
    # BOTTOM PANEL — Destruction mechanisms
    # ----------------------------------------------------------------
    ax2 = fig.add_subplot(gs[1])

    if has_log:
        z_arr    = np.array([r['z'] for r in log_records])
        thermal  = np.array([r.get('thermal',    0) for r in log_records], float)
        shock    = np.array([r.get('shock',      0) for r in log_records], float)
        astrat   = np.array([r.get('astration',  0) for r in log_records], float)
        internal = np.array([r.get('cleanup',    0) +
                             r.get('corruption', 0) +
                             r.get('bad_gas',    0)
                             for r in log_records], float)
        total_dest = thermal + shock + astrat + internal

        if mode == 'rates':
            z_plot = cumulative_to_rates(z_arr, thermal)[0]   # shared x
            _, thermal   = cumulative_to_rates(z_arr, thermal)
            _, shock      = cumulative_to_rates(z_arr, shock)
            _, astrat     = cumulative_to_rates(z_arr, astrat)
            _, internal   = cumulative_to_rates(z_arr, internal)
            _, total_dest = cumulative_to_rates(z_arr, total_dest)
        else:
            z_plot = z_arr

        ax2.semilogy(z_plot, np.maximum(total_dest, 0.5),
                     color=COLORS['total_dest'], lw=2.5, ls='-',
                     label='Total destroyed', zorder=6)
        ax2.semilogy(z_plot, np.maximum(thermal, 0.5),
                     color=COLORS['thermal'], lw=1.8, ls='--',
                     label='Thermal sputtering', zorder=5)
        ax2.semilogy(z_plot, np.maximum(astrat, 0.5),
                     color=COLORS['astration'], lw=1.8, ls='--',
                     label='Astration', zorder=5)
        ax2.semilogy(z_plot, np.maximum(shock, 0.5),
                     color=COLORS['shock'], lw=1.8, ls='--',
                     label='Shock destruction', zorder=5)
        ax2.semilogy(z_plot, np.maximum(internal, 0.5),
                     color=COLORS['internal'], lw=1.2, ls=':',
                     label='Internal (cleanup + corruption)', zorder=3)

        # Annotate fraction at final (lowest z) snapshot
        final = log_records[-1]
        tot = max(1, (final.get('thermal',    0) +
                      final.get('shock',      0) +
                      final.get('astration',  0) +
                      final.get('cleanup',    0) +
                      final.get('corruption', 0) +
                      final.get('bad_gas',    0)))
        frac_thermal  = final.get('thermal',   0) / tot * 100
        frac_shock    = final.get('shock',     0) / tot * 100
        frac_astrat   = final.get('astration', 0) / tot * 100
        frac_internal = (final.get('cleanup', 0) +
                         final.get('corruption', 0) +
                         final.get('bad_gas', 0)) / tot * 100

        summary = (f"At z={final['z']:.2f}:\n"
                   f"  Thermal:   {frac_thermal:.1f}%\n"
                   f"  Astration: {frac_astrat:.1f}%\n"
                   f"  Shock:     {frac_shock:.1f}%\n"
                   f"  Internal:  {frac_internal:.1f}%")
        ax2.text(0.5, 0.97, summary,
                 transform=ax2.transAxes, fontsize=9,
                 ha='center', va='top', family='monospace',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='#f8f9fa',
                           alpha=0.9, edgecolor='#adb5bd'))

    ax2.set_xlabel('Redshift', fontsize=12)
    ax2.set_ylabel(ylabel, fontsize=11)
    ax2.set_xlim(z_hi, z_lo)
    ax2.legend(fontsize=11, loc='upper left', framealpha=0.85)
    ax2.grid(True, which='both', alpha=0.25)
    ax2.text(0.02, 0.05, 'Dust Destruction Mechanisms',
             transform=ax2.transAxes, fontsize=11, fontweight='bold',
             ha='left', va='bottom',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                       alpha=0.8, edgecolor='none'))

    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✅  Plot saved to: {output_file}")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


# ============================================================
# Convenience: print a quick text summary table
# ============================================================

def print_summary_table(log_records):
    if not log_records:
        return
    print("\n" + "=" * 85)
    print(f"{'z':>6}  {'Created':>9}  {'Thermal':>9}  {'Shock':>7}  "
          f"{'Astrated':>9}  {'Internal':>9}  {'Net':>7}  {'Live':>6}")
    print("-" * 85)
    prev = None
    for r in reversed(log_records):   # low-z first for readability
        cre  = r.get('created',    0)
        th   = r.get('thermal',    0)
        sh   = r.get('shock',      0)
        ast  = r.get('astration',  0)
        inn  = (r.get('cleanup',   0) + r.get('corruption', 0)
                + r.get('bad_gas', 0))
        dest = th + sh + ast + inn
        net  = cre - dest
        live = r.get('live', 0)
        print(f"{r['z']:6.3f}  {cre:9d}  {th:9d}  {sh:7d}  "
              f"{ast:9d}  {inn:9d}  {net:7d}  {live:6d}")
    print("=" * 85 + "\n")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Plot dust creation/destruction mechanisms vs redshift.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('output_dir',
                        help='Simulation output directory (contains snapdir_* etc.)')
    parser.add_argument('--log', default=None,
                        help='Path to simulation log/stdout file for audit parsing. '
                             'If not given, tries <output_dir>/stdout.txt and '
                             '<output_dir>/../stdout.txt automatically.')
    parser.add_argument('--cumulative', dest='mode', action='store_const',
                        const='cumulative', default='cumulative',
                        help='Plot cumulative particle counts (default)')
    parser.add_argument('--rates', dest='mode', action='store_const',
                        const='rates',
                        help='Plot differential rates per dz instead of cumulative counts')
    parser.add_argument('--no-snaps', action='store_true',
                        help='Skip snapshot GrainType census (faster)')
    parser.add_argument('--out', default=None,
                        help='Output figure filename')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--show_plot', type=int, default=0,
                        help='Show plot interactively (1=yes, 0=no). Default: 0')

    args = parser.parse_args()

    output_dir = os.path.normpath(args.output_dir)
    if not os.path.isdir(output_dir):
        print(f"ERROR: Directory not found: {output_dir}")
        sys.exit(1)

    run_name   = os.path.basename(output_dir)
    output_fig = args.out or os.path.join(output_dir, 'dust_mechanisms.png')

    print("\n" + "=" * 70)
    print("🌌  Dust Mechanism Analysis  🌌")
    print("=" * 70)
    print(f"  Output dir : {output_dir}")
    print(f"  Mode       : {args.mode}")
    print(f"  Figure out : {output_fig}")
    print("=" * 70)

    # --- Locate log file ---
    log_path = args.log
    if log_path is None:
        # Also glob for any *.log inside the output directory
        glob_logs = sorted(glob.glob(os.path.join(output_dir, '*.log')))
        candidates = [
            os.path.join(output_dir, 'stdout.txt'),
            os.path.join(output_dir, 'stdout'),
            os.path.join(output_dir, '..', 'stdout.txt'),
            os.path.join(output_dir, '..', 'stdout'),
            os.path.join(output_dir, 'output.log'),
        ] + glob_logs
        for c in candidates:
            if os.path.exists(c):
                log_path = c
                print(f"  Auto-found log: {log_path}")
                break

    # --- Parse ---
    log_records  = parse_log_file(log_path, verbose=args.verbose) if log_path else []
    snap_records = ([] if args.no_snaps
                    else snapshot_grain_type_census(output_dir, verbose=args.verbose))

    if not log_records and not snap_records:
        print("\nERROR: No data found. Provide a log file with --log, or ensure "
              "snapdir_* directories contain PartType6 particles.")
        sys.exit(1)

    print_summary_table(log_records)

    plot_mechanisms(log_records, snap_records,
                    output_file=output_fig,
                    run_name=run_name,
                    mode=args.mode,
                    show_plot=args.show_plot)

    print("=" * 70)
    print("✅  Done!")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
