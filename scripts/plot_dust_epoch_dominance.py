#!/usr/bin/env python3
"""
plot_dust_epoch_dominance.py

For every snapshot × every S-rung, compute D/G within R200, then show:
  Left panel  — D/G vs redshift, one line per rung (S0–S10)
  Right panel — ΔD/G per physics channel vs redshift (signed)
                positive = net dust gain, negative = net dust loss

This directly answers: which physics mechanism dominates at each epoch?

First run (reads all snapshots — can take 20–40 min on Stampede):
  python plot_dust_epoch_dominance.py \
      --base_dir   .. \
      --dir_suffix _output_1024 \
      --cache      epoch_cache.npz \
      --out        epoch_dominance.png

Subsequent runs (loads cache, just re-plots — seconds):
  python plot_dust_epoch_dominance.py \
      --cache epoch_cache.npz \
      --out   epoch_dominance.png

Cache stores one D/G value per (rung, snapshot); add --no_cache to force
a fresh computation.
"""

import argparse
import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec
import h5py

try:
    from halo_utils import load_target_halo, extract_dust_spatially
except ImportError:
    print("ERROR: halo_utils.py required in the same directory.")
    sys.exit(1)

# ── Physics channel metadata ──────────────────────────────────────────────────
CHANNELS = [
    ('S1',  'Grain Temperature',   '#aaaaaa'),
    ('S2',  'Gas-Dust Drag',       '#4361ee'),
    ('S3',  'Astration',           '#7209b7'),
    ('S4',  'Thermal Sputtering',  '#c1121f'),
    ('S5',  'ISM Grain Growth',    '#2d6a4f'),
    ('S6',  'Subgrid Clumping',    '#2a9d8f'),
    ('S7',  'SN Shock Destruction','#e63946'),
    ('S8',  'Coagulation',         '#f4a261'),
    ('S9',  'Shattering',          '#e9c46a'),
    ('S10', 'Radiation Pressure',  '#a8dadc'),
]

ALL_RUNGS   = [f'S{i}' for i in range(11)]
RUNG_CMAP   = cm.get_cmap('plasma', 11)


# ── Snapshot discovery ────────────────────────────────────────────────────────

def find_snapshots(rung_dir):
    """
    Return sorted list of (snap_num, snap_base_path, cat_path) for all
    complete snapshot/catalog pairs found under rung_dir.
    """
    snap_dirs = sorted(glob.glob(os.path.join(rung_dir, 'snapdir_[0-9][0-9][0-9]')))
    found = []
    for sd in snap_dirs:
        n       = int(os.path.basename(sd).split('_')[1])
        snap    = os.path.join(sd, f'snapshot_{n:03d}')
        cat     = os.path.join(rung_dir, f'groups_{n:03d}',
                               f'fof_subhalo_tab_{n:03d}.0.hdf5')
        has_snap = bool(glob.glob(f'{snap}.*.hdf5') or
                        os.path.exists(f'{snap}.hdf5'))
        if has_snap and os.path.exists(cat):
            found.append((n, snap, cat))
    return found


# ── Per-snapshot D/G computation ──────────────────────────────────────────────

def snap_header(snap_base):
    """Read Time (scale factor) and Redshift from snapshot header."""
    files = sorted(glob.glob(f'{snap_base}.*.hdf5'))
    if not files:
        f0 = f'{snap_base}.hdf5'
        files = [f0] if os.path.exists(f0) else []
    if not files:
        return None
    with h5py.File(files[0], 'r') as f:
        return dict(
            a        = float(f['Header'].attrs['Time']),
            z        = float(f['Header'].attrs['Redshift']),
            h        = float(f['Parameters'].attrs['HubbleParam']),
        )


def compute_dg(snap_base, cat_path, verbose=False):
    """
    Compute D/G = M_dust / M_gas within R200 for one snapshot.

    Gas mass  : Group/GroupMassType[:, 0] from SubFind catalog (fast).
    Dust mass : sum of PartType6 masses within R200 from snapshot.
    R200      : Group_R_Crit200 from catalog, converted to physical kpc.

    Returns dict with z, dg, m_dust, m_gas, r200_kpc — or None on failure.
    """
    info = snap_header(snap_base)
    if info is None:
        return None
    a, z, h = info['a'], info['z'], info['h']

    # ── Halo info from catalog ────────────────────────────────────────────
    try:
        halo = load_target_halo(cat_path, snap_base,
                                particle_types=[], verbose=verbose)
    except Exception as e:
        if verbose:
            print(f"    halo load failed: {e}")
        return None

    halo_info = halo['halo_info']
    halo_pos  = halo_info['position']   # physical kpc (halo_utils convention)
    halo_idx  = halo_info.get('index', 0)

    # R200 and gas mass from catalog
    try:
        with h5py.File(cat_path, 'r') as f:
            # R200: stored in comoving h^-1 kpc → physical kpc
            r200_kpc = float(f['Group/Group_R_Crit200'][halo_idx]) * a / h
            # Gas mass: 10^10 h^-1 M_sun → M_sun
            m_gas    = float(f['Group/GroupMassType'][halo_idx, 0]) * 1e10 / h
    except Exception as e:
        if verbose:
            print(f"    catalog read failed: {e}")
        return None

    if r200_kpc <= 0 or m_gas <= 0:
        return None

    # ── Dust mass within R200 ─────────────────────────────────────────────
    try:
        dust = extract_dust_spatially(snap_base, halo_pos,
                                      radius_kpc=r200_kpc, verbose=False)
        m_dust = (np.sum(dust['Masses']) * 1e10
                  if (dust is not None and len(dust['Coordinates']) > 0)
                  else 0.0)
    except Exception as e:
        if verbose:
            print(f"    dust load failed: {e}")
        m_dust = 0.0

    return dict(z=z, dg=m_dust / m_gas, m_dust=m_dust,
                m_gas=m_gas, r200_kpc=r200_kpc)


# ── Cache helpers ─────────────────────────────────────────────────────────────

def save_cache(path, rungs, snap_nums, redshifts, dg_table):
    """
    dg_table : 2-D array shape (n_rungs, n_snaps), NaN where not computed.
    redshifts: 1-D array shape (n_snaps,).
    """
    np.savez(path,
             rungs     = np.array(rungs),
             snap_nums = np.array(snap_nums),
             redshifts = np.array(redshifts),
             dg_table  = np.array(dg_table))
    print(f"\nCache saved → {path}")


def load_cache(path):
    data = np.load(path, allow_pickle=True)
    return (list(data['rungs']),
            list(data['snap_nums']),
            data['redshifts'],
            data['dg_table'])


# ── Figure ────────────────────────────────────────────────────────────────────

def make_figure(rungs, redshifts, dg_table, out, dpi):
    """
    Left panel  — D/G vs z for each rung.
    Right panel — ΔD/G per channel (S_k − S_{k−1}) vs z, signed.
    """
    # Sort by descending redshift (high-z left), dropping any NaN-z snapshots
    valid_snaps = ~np.isnan(redshifts)
    z_all   = redshifts[valid_snaps]
    dg_all  = dg_table[:, valid_snaps]
    order   = np.argsort(z_all)[::-1]
    z_plot  = z_all[order]
    dg_plot = dg_all[:, order]

    z_min, z_max = float(np.nanmin(z_plot)), float(np.nanmax(z_plot))

    rung_idx = {r: i for i, r in enumerate(rungs)}

    fig = plt.figure(figsize=(16, 6))
    gs  = GridSpec(1, 2, figure=fig, wspace=0.38)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # ── Left: D/G per rung ───────────────────────────────────────────────
    for i, rung in enumerate(rungs):
        color = RUNG_CMAP(i / 10)
        lw    = 2.0 if rung in {'S0', 'S2', 'S5', 'S7', 'S8', 'S10'} else 0.9
        alpha = 1.0 if lw > 1 else 0.55
        valid = ~np.isnan(dg_plot[i])
        ax1.plot(z_plot[valid], dg_plot[i, valid],
                 color=color, linewidth=lw, alpha=alpha, label=rung)

    ax1.set_xlabel('Redshift', fontsize=11)
    ax1.set_ylabel('D/G within $R_{200}$', fontsize=11)
    ax1.set_title('Dust-to-Gas Ratio Evolution', fontsize=12, fontweight='bold')
    ax1.set_xlim(z_max, z_min)
    ax1.set_ylim(bottom=0)
    ax1.legend(fontsize=8, ncol=2, loc='upper left')
    ax1.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
    ax1.set_axisbelow(True)

    # Secondary x-axis: lookback time
    try:
        from scipy.integrate import quad
        def z_to_gyr(zz, h=0.7, Om=0.3, OL=0.7):
            if zz >= 1000:
                return 13.8
            t_H = 9.778 / h
            age, _ = quad(lambda a: 1.0 / (a * np.sqrt(Om / a**3 + OL)),
                          1e-4, 1.0 / (1.0 + zz))
            return 13.8 - age * t_H   # lookback time in Gyr

        ax1b = ax1.twiny()
        ax1b.set_xlim(z_max, z_min)   # same sense as ax1 — high-z left
        # Tick positions in redshift, filtered to plotted range
        zt_all = np.array([0, 0.5, 1, 2, 3, 5, 8])
        zt = zt_all[(zt_all >= z_min - 0.1) & (zt_all <= z_max + 0.1)]
        lt = np.array([z_to_gyr(zi) for zi in zt])
        ax1b.set_xticks(zt)
        ax1b.set_xticklabels([f'{t:.1f}' for t in lt], fontsize=8)
        ax1b.set_xlabel('Lookback time (Gyr)', fontsize=9)
        # No invert_xaxis — xlim already sets the correct (inverted) sense
    except ImportError:
        pass

    # ── Right: ΔD/G per channel ──────────────────────────────────────────
    ax2.axhline(0, color='#333', linewidth=1.0, zorder=2)

    all_deltas = []
    for ch_rung, ch_name, ch_color in CHANNELS:
        if ch_rung not in rung_idx:
            continue
        k = rung_idx[ch_rung]
        if k - 1 < 0:
            continue
        delta = dg_plot[k] - dg_plot[k - 1]
        valid = ~np.isnan(delta)
        if valid.sum() == 0:
            continue
        ax2.plot(z_plot[valid], delta[valid],
                 color=ch_color, linewidth=1.8, label=ch_name)
        ax2.fill_between(z_plot[valid], delta[valid], 0,
                         where=delta[valid] > 0,
                         alpha=0.10, color=ch_color)
        ax2.fill_between(z_plot[valid], delta[valid], 0,
                         where=delta[valid] < 0,
                         alpha=0.10, color=ch_color)
        all_deltas.extend(delta[valid].tolist())

    # Clip y-axis to 95th percentile so bursty z~1-2 spikes
    # don't compress the rest of the signal
    if all_deltas:
        p95 = np.percentile(np.abs(all_deltas), 95)
        ax2.set_ylim(-p95 * 1.3, p95 * 1.3)
        ax2.text(0.02, 0.97,
                 f'y clipped at ±{p95*1.3:.2e}\n(95th pctile × 1.3)',
                 transform=ax2.transAxes, fontsize=7,
                 va='top', ha='left', color='#888',
                 style='italic')

    ax2.set_xlabel('Redshift', fontsize=11)
    ax2.set_ylabel(r'$\Delta$(D/G)  =  D/G$(S_k)$ $-$ D/G$(S_{k-1})$', fontsize=11)
    ax2.set_title('Per-Channel D/G Contribution', fontsize=12, fontweight='bold')
    ax2.set_xlim(z_max, z_min)
    ax2.legend(fontsize=8.5, loc='upper left', ncol=1)
    ax2.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
    ax2.set_axisbelow(True)

    # Annotate positive = growth, negative = destruction
    ax2.text(0.98, 0.97, 'net growth ↑', transform=ax2.transAxes,
             fontsize=8, ha='right', va='top', color='#2d6a4f', style='italic')
    ax2.text(0.98, 0.03, 'net destruction ↓', transform=ax2.transAxes,
             fontsize=8, ha='right', va='bottom', color='#c1121f', style='italic')

    fig.suptitle('Halo 569  $\\cdot$  $1024^3$  $\\cdot$  '
                 'Which dust physics dominates at each epoch?',
                 fontsize=11, y=1.02)

    plt.savefig(out, dpi=dpi, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.show()


# ── Dominance summary table ───────────────────────────────────────────────────

def print_dominance_table(rungs, redshifts, dg_table):
    """
    At each epoch, report the channel with the largest |ΔD/G|.
    """
    order   = np.argsort(redshifts)[::-1]
    z_plot  = redshifts[order]
    dg_plot = dg_table[:, order]
    rung_idx = {r: i for i, r in enumerate(rungs)}

    print(f"\n{'═'*60}")
    print(f"  Dominant channel by epoch (largest |ΔD/G|)")
    print(f"{'═'*60}")
    print(f"  {'z range':>12}  {'dominant channel':<28}  {'ΔD/G':>10}")
    print(f"  {'─'*58}")

    z_bins = [(6, 4), (4, 2), (2, 1), (1, 0.5), (0.5, 0.2), (0.2, 0)]
    for z_hi, z_lo in z_bins:
        mask = (z_plot >= z_lo) & (z_plot < z_hi)
        if mask.sum() == 0:
            continue
        best_name, best_delta = '—', 0.0
        for ch_rung, ch_name, _ in CHANNELS:
            if ch_rung not in rung_idx:
                continue
            k = rung_idx[ch_rung]
            delta = np.nanmean(dg_plot[k, mask] - dg_plot[k-1, mask])
            if abs(delta) > abs(best_delta):
                best_delta, best_name = delta, ch_name
        sign = '+' if best_delta >= 0 else ''
        print(f"  {z_lo:.1f} < z < {z_hi:.1f}   {best_name:<28}  "
              f"{sign}{best_delta:.2e}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='D/G evolution and per-channel dominance across all epochs')
    parser.add_argument('--base_dir',   default=None,
                        help='Parent directory of rung subdirs')
    parser.add_argument('--dir_suffix', default='_output_1024',
                        help='Suffix for rung dirs (default: _output_1024)')
    parser.add_argument('--rungs',      nargs='+', default=ALL_RUNGS)
    parser.add_argument('--cache',      default='epoch_cache.npz',
                        help='Path to save/load cached D/G table')
    parser.add_argument('--no_cache',   action='store_true',
                        help='Force recompute even if cache exists')
    parser.add_argument('--out',        default='epoch_dominance.png')
    parser.add_argument('--dpi',        type=int, default=150)
    parser.add_argument('--verbose',    action='store_true')
    args = parser.parse_args()

    # ── Load or compute ───────────────────────────────────────────────────
    if os.path.exists(args.cache) and not args.no_cache:
        print(f"Loading cache: {args.cache}")
        rungs, snap_nums, redshifts, dg_table = load_cache(args.cache)
        print(f"  {len(rungs)} rungs × {len(snap_nums)} snapshots")

    else:
        if args.base_dir is None:
            parser.error("--base_dir required when no cache exists")

        # Discover snapshots from first available rung
        snap_list = None
        for rung in args.rungs:
            rdir = os.path.join(args.base_dir, f'{rung}{args.dir_suffix}')
            if os.path.isdir(rdir):
                snap_list = find_snapshots(rdir)
                if snap_list:
                    print(f"Discovered {len(snap_list)} snapshots from {rung}")
                    break
        if not snap_list:
            print("ERROR: no snapshots found. Check --base_dir and --dir_suffix")
            sys.exit(1)

        snap_nums = [s[0] for s in snap_list]
        n_snaps   = len(snap_nums)
        n_rungs   = len(args.rungs)
        dg_table  = np.full((n_rungs, n_snaps), np.nan)
        redshifts = np.full(n_snaps, np.nan)

        total = n_rungs * n_snaps
        done  = 0

        for ri, rung in enumerate(args.rungs):
            rdir = os.path.join(args.base_dir, f'{rung}{args.dir_suffix}')
            for si, (sn, snap_template, _) in enumerate(snap_list):
                snap_path = os.path.join(rdir,
                                         f'snapdir_{sn:03d}',
                                         f'snapshot_{sn:03d}')
                cat_path  = os.path.join(rdir,
                                         f'groups_{sn:03d}',
                                         f'fof_subhalo_tab_{sn:03d}.0.hdf5')
                done += 1
                print(f"  [{done:4d}/{total}]  {rung}  snap {sn:03d}  ...",
                      end=' ', flush=True)

                if not os.path.exists(cat_path):
                    print("no catalog — skip")
                    continue

                result = compute_dg(snap_path, cat_path, verbose=args.verbose)
                if result is None:
                    print("failed — skip")
                    continue

                dg_table[ri, si] = result['dg']
                if np.isnan(redshifts[si]):
                    redshifts[si] = result['z']
                print(f"z={result['z']:.3f}  D/G={result['dg']:.4e}")

        save_cache(args.cache, args.rungs, snap_nums, redshifts, dg_table)
        rungs = args.rungs

    # ── Print dominance summary ───────────────────────────────────────────
    print_dominance_table(rungs, redshifts, dg_table)

    # ── Plot ─────────────────────────────────────────────────────────────
    make_figure(rungs, redshifts, dg_table, args.out, args.dpi)


if __name__ == '__main__':
    main()
