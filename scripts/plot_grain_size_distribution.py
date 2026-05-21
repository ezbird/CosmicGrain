#!/usr/bin/env python3
"""
plot_grain_size_distribution.py

Quantify and compare grain size distributions in the ISM and CGM.
Reports power-law slope (vs MRN benchmark), mass-weighted mean, ceiling
fraction, and cumulative mass distribution.

Single-rung mode (detailed diagnostic):
  python plot_grain_size_distribution.py \
      --catalog  ../groups_049/fof_subhalo_tab_049.0.hdf5 \
      --snapshot ../snapdir_049/snapshot_049 \
      --out      gsd_single.png

Multi-rung mode (evolution of key stats across S0–S10):
  python plot_grain_size_distribution.py \
      --base_dir   .. \
      --dir_suffix _output_1024 \
      --rungs      S0 S1 S2 S3 S4 S5 S6 S7 S8 S9 S10 \
      --out        gsd_rungs.png

Key outputs
-----------
  Single-rung figure
    Left  — dM/d(log a) for ISM and CGM with MRN slope reference
    Right — Cumulative mass fraction F(a) for ISM and CGM

  Multi-rung figure
    Left   — Mass-weighted mean grain radius vs rung (ISM and CGM)
    Middle — Power-law slope β [dN/d(log a) ∝ a^β] vs rung; MRN = −2.5
    Right  — Ceiling fraction f_ceil and small-grain fraction f_small vs rung

  Printed table — per region: N, ⟨a⟩_mass, ⟨a⟩_num, β, f_ceil, f_small
"""

import argparse
import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
import h5py

try:
    from halo_utils import load_target_halo, extract_dust_spatially
except ImportError:
    print("ERROR: halo_utils.py required in the same directory.")
    sys.exit(1)

try:
    from scipy.optimize import curve_fit
    SCIPY = True
except ImportError:
    SCIPY = False
    print("WARNING: scipy not found — power-law fits disabled.")

# ── Constants ─────────────────────────────────────────────────────────────────
A_MIN_FIT   =  10.0   # nm — lower bound for power-law fit
A_MAX_FIT   = 150.0   # nm — upper bound (avoids 200 nm ceiling artefact)
A_CEIL      = 190.0   # nm — grains above this counted as "at ceiling"
A_SMALL     =  30.0   # nm — grains below this counted as "small"
MRN_SLOPE   =  -2.5   # dN/d(log a) ∝ a^β, β = −2.5 for MRN dn/da ∝ a^{−3.5}
N_BINS      =  30     # log-spaced radius bins
A_LO, A_HI = 5.0, 220.0   # nm — full plot range

ISM_COLOR = '#2a9d8f'
CGM_COLOR = '#e76f51'
MRN_COLOR = '#aaaaaa'
ALL_RUNGS  = [f'S{i}' for i in range(11)]


# ── Utilities ─────────────────────────────────────────────────────────────────

def get_snapshot_info(snap_base):
    files = sorted(glob.glob(f'{snap_base}.*.hdf5'))
    if not files:
        files = [f'{snap_base}.hdf5']
    if not files or not os.path.exists(files[0]):
        return None
    with h5py.File(files[0], 'r') as f:
        return dict(
            z = float(f['Header'].attrs['Redshift']),
            a = float(f['Header'].attrs['Time']),
            h = float(f['Parameters'].attrs['HubbleParam']),
        )


def compute_distances(coords, center):
    return np.sqrt(np.sum((coords - center) ** 2, axis=1))


def log_bins():
    """Return log-spaced bin edges and centres in nm."""
    edges   = np.logspace(np.log10(A_LO), np.log10(A_HI), N_BINS + 1)
    centres = np.sqrt(edges[:-1] * edges[1:])   # geometric mean
    widths  = np.diff(np.log10(edges))           # Δ(log₁₀ a)
    return edges, centres, widths


def power_law(log_a, beta, log_norm):
    return beta * log_a + log_norm


def fit_slope(a_nm, masses, edges, centres):
    """
    Fit dN/d(log a) ∝ a^β in the range A_MIN_FIT–A_MAX_FIT.
    Returns β, β_err.  Uses number counts (not mass) for MRN comparison.
    """
    if not SCIPY or len(a_nm) == 0:
        return np.nan, np.nan

    fit_mask = (a_nm >= A_MIN_FIT) & (a_nm <= A_MAX_FIT)
    if fit_mask.sum() < 5:
        return np.nan, np.nan

    # Number histogram in fit range
    counts, _ = np.histogram(a_nm[fit_mask],
                             bins=edges[(edges >= A_MIN_FIT) &
                                        (edges <= A_MAX_FIT + 1)])
    c_cents = np.sqrt(edges[:-1][:-1] * edges[1:][:-1])
    # Re-histogram with all edges, filter to fit range
    counts_all, _ = np.histogram(a_nm, bins=edges)
    fit_bin = (centres >= A_MIN_FIT) & (centres <= A_MAX_FIT)
    cnt = counts_all[fit_bin]
    cen = centres[fit_bin]

    good = cnt > 0
    if good.sum() < 3:
        return np.nan, np.nan

    try:
        popt, pcov = curve_fit(power_law,
                               np.log10(cen[good]),
                               np.log10(cnt[good]),
                               p0=[-2.5, 3.0])
        beta_err = np.sqrt(np.diag(pcov))[0]
        return popt[0], beta_err
    except Exception:
        return np.nan, np.nan


def region_gsd(a_nm, masses, label):
    """
    Compute size-distribution statistics for one spatial region.
    Returns a dict of summary stats plus histogram arrays.
    """
    if len(a_nm) == 0:
        return None

    edges, centres, widths = log_bins()

    # Mass per log-a bin
    m_per_bin = np.zeros(N_BINS)
    n_per_bin = np.zeros(N_BINS)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (a_nm >= lo) & (a_nm < hi)
        m_per_bin[i] = masses[mask].sum()
        n_per_bin[i] = mask.sum()

    m_total = masses.sum()
    dM_dlog = m_per_bin / widths          # mass per unit log10(a)
    dN_dlog = n_per_bin / widths          # number per unit log10(a)
    cumulative = np.cumsum(m_per_bin) / m_total

    beta, beta_err = fit_slope(a_nm, masses, edges, centres)

    mean_a_mass   = np.average(a_nm, weights=masses) if m_total > 0 else np.nan
    mean_a_number = np.mean(a_nm) if len(a_nm) > 0 else np.nan
    f_ceil  = masses[a_nm >= A_CEIL].sum()  / m_total
    f_small = masses[a_nm <= A_SMALL].sum() / m_total

    print(f"\n  {'─'*60}")
    print(f"  {label}  (N = {len(a_nm):,}  |  M_dust = {m_total:.3e} M☉)")
    print(f"  {'─'*60}")
    print(f"  ⟨a⟩_mass          : {mean_a_mass:.1f} nm")
    print(f"  ⟨a⟩_number        : {mean_a_number:.1f} nm")
    print(f"  Power-law slope β : {beta:.2f} ± {beta_err:.2f}  "
          f"(MRN = {MRN_SLOPE:.1f},  fit range {A_MIN_FIT:.0f}–{A_MAX_FIT:.0f} nm)")
    print(f"  f_ceiling (≥{A_CEIL:.0f} nm): {100*f_ceil:.1f}%")
    print(f"  f_small   (≤{A_SMALL:.0f} nm) : {100*f_small:.1f}%")

    return dict(
        label       = label,
        N           = len(a_nm),
        m_total     = m_total,
        centres     = centres,
        widths      = widths,
        dM_dlog     = dM_dlog,
        dN_dlog     = dN_dlog,
        cumulative  = cumulative,
        beta        = beta,
        beta_err    = beta_err,
        mean_a_mass = mean_a_mass,
        mean_a_num  = mean_a_number,
        f_ceil      = f_ceil,
        f_small     = f_small,
    )


# ── Data loading ──────────────────────────────────────────────────────────────

def load_dust(snap_base, cat_path, r_cgm, verbose=False):
    """Return (dust dict, halo_info dict) or (None, None)."""
    info = get_snapshot_info(snap_base)
    if info is None:
        return None, None
    try:
        halo     = load_target_halo(cat_path, snap_base,
                                    particle_types=[], verbose=verbose)
        halo_pos = halo['halo_info']['position']
        halo_mass= halo['halo_info']['mass'] * 1e10
        dust     = extract_dust_spatially(snap_base, halo_pos,
                                          radius_kpc=r_cgm, verbose=verbose)
    except Exception as e:
        if verbose:
            print(f"  load failed: {e}")
        return None, None
    if dust is None or len(dust['Coordinates']) == 0:
        return None, None

    r = compute_distances(dust['Coordinates'], halo_pos)
    return dict(
        r            = r,
        masses       = dust['Masses'] * 1e10,
        grain_radius = dust['GrainRadius'],
        halo_mass    = halo_mass,
        redshift     = info['z'],
    ), halo['halo_info']


# ── Single-rung figure ────────────────────────────────────────────────────────

def plot_single(data, r_ism, r_cgm, out, dpi):
    r  = data['r']
    a  = data['grain_radius']
    m  = data['masses']
    z  = data['redshift']
    M  = data['halo_mass']

    ism_mask = r < r_ism
    cgm_mask = (r >= r_ism) & (r < r_cgm)

    print(f"\n{'═'*62}")
    print(f"  Grain Size Distribution  z={z:.2f}")
    print(f"  ISM r<{r_ism:.0f} kpc   CGM {r_ism:.0f}–{r_cgm:.0f} kpc")
    print(f"{'═'*62}")

    s_ism = region_gsd(a[ism_mask], m[ism_mask], f'ISM  (r < {r_ism:.0f} kpc)')
    s_cgm = region_gsd(a[cgm_mask], m[cgm_mask],
                       f'CGM  ({r_ism:.0f}–{r_cgm:.0f} kpc)')

    fig = plt.figure(figsize=(13, 5))
    gs  = GridSpec(1, 2, figure=fig, wspace=0.35)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    edges, centres, widths = log_bins()

    # ── Panel 1: dM/d(log a) ─────────────────────────────────────────────
    for s, color, ls in [(s_ism, ISM_COLOR, '-'), (s_cgm, CGM_COLOR, '--')]:
        if s is None:
            continue
        valid = s['dM_dlog'] > 0
        ax1.plot(s['centres'][valid], s['dM_dlog'][valid],
                 color=color, linewidth=2.0, linestyle=ls,
                 label=f"{s['label']}  (β={s['beta']:.2f})")
        ax1.fill_between(s['centres'][valid], s['dM_dlog'][valid],
                         alpha=0.10, color=color)

    # MRN reference: dM/d(log a) ∝ a^{+0.5}, normalised to ISM total
    if s_ism is not None and s_ism['m_total'] > 0:
        mrn_norm = s_ism['dM_dlog'][s_ism['dM_dlog'] > 0].mean() / \
                   (centres[s_ism['dM_dlog'] > 0].mean() ** 0.5)
        mrn_y = mrn_norm * centres ** 0.5
        ax1.plot(centres, mrn_y, color=MRN_COLOR, linewidth=1.2,
                 linestyle=':', label=f'MRN reference  (β={MRN_SLOPE:.1f})')

    ax1.axvline(A_SMALL, color='#bbb', linewidth=0.8, linestyle='--')
    ax1.axvline(200,     color='#bbb', linewidth=0.8, linestyle='--')
    ax1.text(A_SMALL, ax1.get_ylim()[1] if ax1.get_ylim()[1] > 0 else 1,
             f'{A_SMALL:.0f} nm', fontsize=7, color='#999', ha='center',
             va='bottom')
    ax1.text(200, ax1.get_ylim()[1] if ax1.get_ylim()[1] > 0 else 1,
             '200 nm\nceiling', fontsize=7, color='#999', ha='center',
             va='bottom')

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Grain radius $a$ (nm)', fontsize=11)
    ax1.set_ylabel(r'${\rm d}M\,/\,{\rm d}\log_{10}a$  ($\rm M_\odot$)',
                   fontsize=11)
    ax1.set_title('Mass-weighted Size Distribution', fontsize=12,
                  fontweight='bold')
    ax1.set_xlim(A_LO, A_HI)
    ax1.legend(fontsize=9)
    ax1.grid(True, which='major', alpha=0.25, linestyle='--', linewidth=0.5)
    ax1.xaxis.set_major_formatter(ticker.ScalarFormatter())

    # ── Panel 2: Cumulative mass fraction ─────────────────────────────────
    for s, color, ls in [(s_ism, ISM_COLOR, '-'), (s_cgm, CGM_COLOR, '--')]:
        if s is None:
            continue
        ax2.plot(s['centres'], s['cumulative'],
                 color=color, linewidth=2.0, linestyle=ls,
                 label=s['label'])
        # Mark median grain radius (50th percentile of mass)
        idx50 = np.searchsorted(s['cumulative'], 0.5)
        if 0 < idx50 < len(s['centres']):
            a50 = s['centres'][idx50]
            ax2.axvline(a50, color=color, linewidth=0.9, linestyle=':',
                        alpha=0.7)
            ax2.text(a50, 0.52, f'{a50:.0f} nm', fontsize=8,
                     color=color, ha='left', va='bottom')

    ax2.axhline(0.5, color='#ccc', linewidth=0.8, linestyle='--')
    ax2.axvline(200, color='#bbb', linewidth=0.8, linestyle='--')
    ax2.set_xscale('log')
    ax2.set_xlabel('Grain radius $a$ (nm)', fontsize=11)
    ax2.set_ylabel('Cumulative mass fraction  $F(<a)$', fontsize=11)
    ax2.set_title('Cumulative Mass Distribution', fontsize=12,
                  fontweight='bold')
    ax2.set_xlim(A_LO, A_HI)
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=9, loc='upper left')
    ax2.grid(True, which='major', alpha=0.25, linestyle='--', linewidth=0.5)
    ax2.xaxis.set_major_formatter(ticker.ScalarFormatter())

    fig.suptitle(
        f'Halo 569  $\\cdot$  $1024^3$  $\\cdot$  '
        f'M$_{{200}}$={M:.2e} M$_\\odot$  $\\cdot$  z={z:.2f}',
        fontsize=10, y=1.01)

    plt.savefig(out, dpi=dpi, bbox_inches='tight')
    print(f"\nSaved: {out}")
    plt.show()


# ── Multi-rung figure ─────────────────────────────────────────────────────────

def plot_multi(results, r_ism, r_cgm, out, dpi):
    """
    Three-panel figure showing key GSD statistics vs rung for ISM and CGM.
    """
    rung_names = [r['rung']               for r in results]
    x          = np.arange(len(results))

    def get(key, region):
        return np.array([r[region][key] if r[region] else np.nan
                         for r in results])

    mean_a_ism = get('mean_a_mass', 'ism')
    mean_a_cgm = get('mean_a_mass', 'cgm')
    beta_ism   = get('beta',        'ism')
    beta_cgm   = get('beta',        'cgm')
    fceil_ism  = get('f_ceil',      'ism')
    fceil_cgm  = get('f_ceil',      'cgm')
    fsmall_ism = get('f_small',     'ism')
    fsmall_cgm = get('f_small',     'cgm')

    fig = plt.figure(figsize=(16, 5))
    gs  = GridSpec(1, 3, figure=fig, wspace=0.38)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    # ── Panel 1: mean grain radius ────────────────────────────────────────
    ax1.plot(x, mean_a_ism, 'o-', color=ISM_COLOR, linewidth=2,
             markersize=6, label=f'ISM  ($r<{r_ism:.0f}$ kpc)')
    ax1.plot(x, mean_a_cgm, 's--', color=CGM_COLOR, linewidth=2,
             markersize=6, label=f'CGM  ({r_ism:.0f}–{r_cgm:.0f} kpc)')
    ax1.axhline(200, color='#bbb', linestyle=':', linewidth=1,
                label='200 nm ceiling')
    ax1.set_xticks(x); ax1.set_xticklabels(rung_names, fontsize=9)
    ax1.set_ylabel(r'$\langle a \rangle_{\rm mass}$ (nm)', fontsize=11)
    ax1.set_title('Mass-weighted Mean Radius', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=8.5)
    ax1.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
    ax1.set_axisbelow(True)

    # ── Panel 2: power-law slope ──────────────────────────────────────────
    ax2.plot(x, beta_ism, 'o-', color=ISM_COLOR, linewidth=2,
             markersize=6, label='ISM')
    ax2.plot(x, beta_cgm, 's--', color=CGM_COLOR, linewidth=2,
             markersize=6, label='CGM')
    ax2.axhline(MRN_SLOPE, color=MRN_COLOR, linestyle='--', linewidth=1.5,
                label=f'MRN  β = {MRN_SLOPE}')
    ax2.fill_between(x, beta_ism, MRN_SLOPE,
                     alpha=0.08, color=ISM_COLOR,
                     label='ISM deviation from MRN')
    ax2.set_xticks(x); ax2.set_xticklabels(rung_names, fontsize=9)
    ax2.set_ylabel(r'Power-law slope $\beta$  '
                   r'[${\rm d}N/{\rm d}\log a \propto a^\beta$]', fontsize=11)
    ax2.set_title(f'Size Distribution Slope\n'
                  f'(fit range {A_MIN_FIT:.0f}–{A_MAX_FIT:.0f} nm)',
                  fontsize=12, fontweight='bold')
    ax2.legend(fontsize=8.5)
    ax2.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
    ax2.set_axisbelow(True)

    # ── Panel 3: ceiling and small-grain fractions ────────────────────────
    ax3.plot(x, 100 * fceil_ism,  'o-',  color=ISM_COLOR, linewidth=2,
             markersize=6, label=f'ISM  $f_{{\\rm ceil}}$ (≥{A_CEIL:.0f} nm)')
    ax3.plot(x, 100 * fceil_cgm,  's--', color=CGM_COLOR, linewidth=2,
             markersize=6, label=f'CGM  $f_{{\\rm ceil}}$')
    ax3.plot(x, 100 * fsmall_ism, 'o:',  color=ISM_COLOR, linewidth=1.5,
             markersize=5, alpha=0.7,
             label=f'ISM  $f_{{\\rm small}}$ (≤{A_SMALL:.0f} nm)')
    ax3.plot(x, 100 * fsmall_cgm, 's:',  color=CGM_COLOR, linewidth=1.5,
             markersize=5, alpha=0.7, label=f'CGM  $f_{{\\rm small}}$')
    ax3.set_xticks(x); ax3.set_xticklabels(rung_names, fontsize=9)
    ax3.set_ylabel('Mass fraction (%)', fontsize=11)
    ax3.set_title('Ceiling & Small-Grain Fractions', fontsize=12,
                  fontweight='bold')
    ax3.legend(fontsize=8, ncol=2)
    ax3.set_ylim(0, 105)
    ax3.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
    ax3.set_axisbelow(True)

    z   = results[0]['redshift']
    M   = results[0]['halo_mass']
    fig.suptitle(
        f'Halo 569  $\\cdot$  $1024^3$  $\\cdot$  '
        f'M$_{{200}}$={M:.2e} M$_\\odot$  $\\cdot$  z={z:.2f}  $\\cdot$  '
        f'ISM $r<{r_ism:.0f}$ kpc  $\\cdot$  CGM ${r_ism:.0f}$–${r_cgm:.0f}$ kpc',
        fontsize=10, y=1.01)

    plt.savefig(out, dpi=dpi, bbox_inches='tight')
    print(f"\nSaved: {out}")
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Grain size distribution: ISM vs CGM, single or multi-rung')

    # Single-rung
    parser.add_argument('--catalog',  default=None)
    parser.add_argument('--snapshot', default=None)

    # Multi-rung
    parser.add_argument('--base_dir',   default=None)
    parser.add_argument('--dir_suffix', default='_output_1024')
    parser.add_argument('--rungs',      nargs='+', default=ALL_RUNGS)

    # Shared
    parser.add_argument('--r_ism',  type=float, default=20.0)
    parser.add_argument('--r_cgm',  type=float, default=294.0)
    parser.add_argument('--out',    default='grain_size_distribution.png')
    parser.add_argument('--dpi',    type=int,   default=150)
    parser.add_argument('--verbose',action='store_true')
    args = parser.parse_args()

    # ── Single-rung mode ──────────────────────────────────────────────────
    if args.base_dir is None:
        if not (args.catalog and args.snapshot):
            parser.error("Provide --base_dir OR both --catalog and --snapshot")
        data, _ = load_dust(args.snapshot, args.catalog,
                            args.r_cgm, verbose=args.verbose)
        if data is None:
            print("ERROR: no dust data found."); return
        plot_single(data, args.r_ism, args.r_cgm, args.out, args.dpi)
        return

    # ── Multi-rung mode ───────────────────────────────────────────────────
    print(f"\n{'═'*62}")
    print(f"  Multi-rung GSD  z=0  "
          f"ISM r<{args.r_ism:.0f}kpc  CGM {args.r_ism:.0f}–{args.r_cgm:.0f}kpc")
    print(f"{'═'*62}")

    results = []
    for rung in args.rungs:
        rdir   = os.path.join(args.base_dir, f'{rung}{args.dir_suffix}')
        # Use last snapshot (z=0)
        snaps  = sorted(glob.glob(
            os.path.join(rdir, 'snapdir_[0-9][0-9][0-9]')))
        if not snaps:
            print(f"  {rung}: no snapshots found — skip")
            continue
        n      = int(os.path.basename(snaps[-1]).split('_')[1])
        snap   = os.path.join(rdir, f'snapdir_{n:03d}', f'snapshot_{n:03d}')
        cat    = os.path.join(rdir, f'groups_{n:03d}',
                              f'fof_subhalo_tab_{n:03d}.0.hdf5')

        print(f"\n  Loading {rung} (snap {n:03d})...")
        data, _ = load_dust(snap, cat, args.r_cgm, verbose=args.verbose)
        if data is None:
            print(f"  {rung}: load failed — skip"); continue

        r, a, m = data['r'], data['grain_radius'], data['masses']
        ism_mask = r < args.r_ism
        cgm_mask = (r >= args.r_ism) & (r < args.r_cgm)

        s_ism = region_gsd(a[ism_mask], m[ism_mask],
                           f'{rung} ISM')
        s_cgm = region_gsd(a[cgm_mask], m[cgm_mask],
                           f'{rung} CGM')

        results.append(dict(
            rung      = rung,
            ism       = s_ism,
            cgm       = s_cgm,
            redshift  = data['redshift'],
            halo_mass = data['halo_mass'],
        ))

    if not results:
        print("ERROR: no rungs loaded."); return

    if len(results) == 1:
        d   = results[0]
        rdir = os.path.join(args.base_dir,
                            f"{d['rung']}{args.dir_suffix}")
        snaps = sorted(glob.glob(
            os.path.join(rdir, 'snapdir_[0-9][0-9][0-9]')))
        n    = int(os.path.basename(snaps[-1]).split('_')[1])
        snap = os.path.join(rdir, f'snapdir_{n:03d}', f'snapshot_{n:03d}')
        cat  = os.path.join(rdir, f'groups_{n:03d}',
                            f'fof_subhalo_tab_{n:03d}.0.hdf5')
        data, _ = load_dust(snap, cat, args.r_cgm)
        plot_single(data, args.r_ism, args.r_cgm, args.out, args.dpi)
    else:
        plot_multi(results, args.r_ism, args.r_cgm, args.out, args.dpi)


if __name__ == '__main__':
    main()
