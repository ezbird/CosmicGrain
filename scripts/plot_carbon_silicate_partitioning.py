#!/usr/bin/env python3
"""
plot_carbon_silicate_partitioning.py

Quantify C vs Si grain partitioning between the ISM and CGM across all
S-rungs (S0–S10), isolating which physics channels drive compositional
segregation.

Single-rung mode (original):
  python plot_carbon_silicate_partitioning.py \
      --catalog  ../groups_049/fof_subhalo_tab_049.0.hdf5 \
      --snapshot ../snapdir_049/snapshot_049 \
      --out      cs_single.png

Multi-rung mode:
  python plot_carbon_silicate_partitioning.py \
      --base_dir  /scratch/halo569_1024/ \
      --rungs     S0 S1 S2 S3 S4 S5 S6 S7 S8 S9 S10 \
      --snap_pat  snapdir_049/snapshot_049 \
      --cat_pat   groups_049/fof_subhalo_tab_049.0.hdf5 \
      --out       cs_rungs.png

Directory layout assumed for multi-rung:
  {base_dir}/{rung}/{snap_pat}
  {base_dir}/{rung}/{cat_pat}
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
import h5py
import glob

try:
    from halo_utils import load_target_halo, extract_dust_spatially
except ImportError:
    print("ERROR: halo_utils.py required in the same directory.")
    exit(1)

# ── Palette ───────────────────────────────────────────────────────────────────
C_COLOR  = '#E07B39'   # carbon  — orange
SI_COLOR = '#2a9d8f'   # silicate — teal
ISM_COLOR = '#2a9d8f'
CGM_COLOR = '#e76f51'
PROFILE_RUNGS = {'S0', 'S2', 'S5', 'S8', 'S10'}   # shown in radial profile panel
PROFILE_CMAP  = cm.get_cmap('plasma', len(PROFILE_RUNGS))

ALL_RUNGS = [f'S{i}' for i in range(11)]


# ── Utilities ─────────────────────────────────────────────────────────────────

def get_snapshot_info(snapshot_base):
    files = sorted(glob.glob(f'{snapshot_base}.*.hdf5'))
    if not files:
        files = [f'{snapshot_base}.hdf5']
    if not files or not os.path.exists(files[0]):
        return None
    with h5py.File(files[0], 'r') as f:
        h = float(f['Parameters'].attrs['HubbleParam'])
        return dict(
            Time      = float(f['Header'].attrs['Time']),
            Redshift  = float(f['Header'].attrs['Redshift']),
            HubbleParam = h,
        )


def compute_distances(coords, center):
    return np.sqrt(np.sum((coords - center)**2, axis=1))


def region_stats(mask, masses, carbon_frac, grain_radius):
    """Return dict of statistics for one spatial region."""
    if mask.sum() == 0:
        return dict(m_C=0, m_Si=0, m_tot=0, cs_ratio=np.nan,
                    mean_fc=np.nan, mean_radius=np.nan, N=0)
    m   = masses[mask]
    fc  = carbon_frac[mask]
    a   = grain_radius[mask]
    m_C  = np.sum(m * fc)
    m_Si = np.sum(m * (1.0 - fc))
    return dict(
        m_C        = m_C,
        m_Si       = m_Si,
        m_tot      = m_C + m_Si,
        cs_ratio   = m_C / m_Si if m_Si > 0 else np.nan,
        mean_fc    = np.average(fc, weights=m),
        mean_radius= np.average(a,  weights=m),
        N          = mask.sum(),
    )


def radial_fc_profile(r, masses, carbon_frac, r_max, n_bins=20):
    """Mass-weighted mean f_C in radial bins out to r_max."""
    edges   = np.linspace(0, r_max, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    profile = np.full(n_bins, np.nan)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (r >= lo) & (r < hi)
        if mask.sum() == 0:
            continue
        profile[i] = np.average(carbon_frac[mask], weights=masses[mask])
    return centers, profile


def load_rung(snapshot_path, catalog_path, r_cgm, verbose=False):
    """Load dust data for one rung; return None if files missing."""
    info = get_snapshot_info(snapshot_path)
    if info is None:
        return None
    try:
        halo = load_target_halo(catalog_path, snapshot_path,
                                particle_types=[], verbose=verbose)
    except Exception as e:
        print(f"  [WARN] halo load failed: {e}")
        return None

    halo_pos  = halo['halo_info']['position']
    halo_mass = halo['halo_info']['mass'] * 1e10

    try:
        dust = extract_dust_spatially(snapshot_path, halo_pos,
                                      radius_kpc=r_cgm, verbose=verbose)
    except Exception as e:
        print(f"  [WARN] dust load failed: {e}")
        return None

    if dust is None or len(dust['Coordinates']) == 0:
        return None

    masses      = dust['Masses'] * 1e10
    carbon_frac = dust['CarbonFraction']
    grain_radius= dust['GrainRadius']
    r           = compute_distances(dust['Coordinates'], halo_pos)

    return dict(
        r=r, masses=masses, carbon_frac=carbon_frac, grain_radius=grain_radius,
        halo_mass=halo_mass, redshift=info['Redshift'],
    )


def print_rung_summary(rung, s_ism, s_cgm):
    enh = ((s_cgm['cs_ratio'] / s_ism['cs_ratio'])
           if (s_ism['cs_ratio'] and not np.isnan(s_ism['cs_ratio'])
               and s_cgm['cs_ratio'] and not np.isnan(s_cgm['cs_ratio']))
           else np.nan)
    print(f"  {rung:4s} | ISM C/Si={s_ism['cs_ratio']:.3f}  "
          f"⟨fC⟩={s_ism['mean_fc']:.3f}  ⟨a⟩={s_ism['mean_radius']:.1f}nm  N={s_ism['N']:,} | "
          f"CGM C/Si={s_cgm['cs_ratio']:.3f}  "
          f"⟨fC⟩={s_cgm['mean_fc']:.3f}  N={s_cgm['N']:,} | "
          f"enh={enh:.3f}")


# ── Single-rung figure ────────────────────────────────────────────────────────

def plot_single(data, r_ism, r_cgm, out, dpi):
    r           = data['r']
    masses      = data['masses']
    carbon_frac = data['carbon_frac']
    grain_radius= data['grain_radius']
    halo_mass   = data['halo_mass']
    redshift    = data['redshift']

    ism_mask = r < r_ism
    cgm_mask = (r >= r_ism) & (r < r_cgm)

    s_ism = region_stats(ism_mask, masses, carbon_frac, grain_radius)
    s_cgm = region_stats(cgm_mask, masses, carbon_frac, grain_radius)

    print(f"\n{'═'*60}")
    print(f"  Single rung  z={redshift:.2f}  "
          f"ISM r<{r_ism:.0f}kpc  CGM {r_ism:.0f}–{r_cgm:.0f}kpc")
    print(f"{'═'*60}")
    print_rung_summary('—', s_ism, s_cgm)
    total_C  = s_ism['m_C']  + s_cgm['m_C']
    total_Si = s_ism['m_Si'] + s_cgm['m_Si']
    print(f"\n  C in ISM: {100*s_ism['m_C']/total_C:.1f}%   "
          f"C in CGM: {100*s_cgm['m_C']/total_C:.1f}%")
    print(f"  Si in ISM: {100*s_ism['m_Si']/total_Si:.1f}%  "
          f"Si in CGM: {100*s_cgm['m_Si']/total_Si:.1f}%")

    rc, fp = radial_fc_profile(r, masses, carbon_frac, r_cgm)

    fig = plt.figure(figsize=(12, 5))
    gs  = GridSpec(1, 2, figure=fig, wspace=0.35)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # Bar chart
    regions   = [f'ISM\n$r<{r_ism:.0f}$ kpc',
                 f'CGM\n${r_ism:.0f}<r<{r_cgm:.0f}$ kpc']
    m_C_vals  = [s_ism['m_C'],  s_cgm['m_C']]
    m_Si_vals = [s_ism['m_Si'], s_cgm['m_Si']]
    x = np.arange(2)
    ax1.bar(x, m_Si_vals, 0.45, label='Silicate $(1-f_C)$',
            color=SI_COLOR, alpha=0.9, edgecolor='white', linewidth=0.8)
    ax1.bar(x, m_C_vals,  0.45, bottom=m_Si_vals, label='Carbon $(f_C)$',
            color=C_COLOR,  alpha=0.9, edgecolor='white', linewidth=0.8)
    ymax = max(a + b for a, b in zip(m_C_vals, m_Si_vals))
    for i, (mc, msi) in enumerate(zip(m_C_vals, m_Si_vals)):
        ax1.text(i, mc + msi + 0.03*ymax, f'C/Si = {mc/msi:.3f}',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax1.set_xticks(x); ax1.set_xticklabels(regions, fontsize=10)
    ax1.set_ylabel(r'Dust Mass ($\rm M_\odot$)', fontsize=11)
    ax1.set_title('C vs Si Mass Budget', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, axis='y', alpha=0.25, linestyle='--', linewidth=0.5)
    ax1.set_axisbelow(True)

    # Radial profile
    valid = ~np.isnan(fp)
    ax2.plot(rc[valid], fp[valid], color='#333', linewidth=2, zorder=3)
    ax2.fill_between(rc[valid], fp[valid], alpha=0.12, color='#333')
    ax2.axvline(r_ism, color='#888', linestyle='--', linewidth=1.2,
                label=f'ISM/CGM ({r_ism:.0f} kpc)')
    ax2.axhline(0.1, color=SI_COLOR, linestyle=':', linewidth=1.2,
                label=r'SNII birth $f_C=0.1$')
    ax2.axhline(0.6, color=C_COLOR,  linestyle=':', linewidth=1.2,
                label=r'AGB birth $f_C=0.6$')
    ax2.axvspan(0, r_ism,  alpha=0.06, color=SI_COLOR)
    ax2.axvspan(r_ism, r_cgm, alpha=0.06, color=C_COLOR)
    ax2.set_xlabel('Radius (pkpc)', fontsize=11)
    ax2.set_ylabel(r'Mass-weighted $\langle f_C \rangle$', fontsize=11)
    ax2.set_title('Carbon Fraction Radial Profile', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, r_cgm); ax2.set_ylim(0, 0.75)
    ax2.legend(fontsize=8.5, loc='upper left')
    ax2.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)

    fig.suptitle(f'Halo 569  $\\cdot$  $1024^3$  $\\cdot$  '
                 f'M$_{{200}}$={halo_mass:.2e} M$_\\odot$  $\\cdot$  z={redshift:.2f}',
                 fontsize=10, y=1.01)
    plt.savefig(out, dpi=dpi, bbox_inches='tight')
    print(f"\nSaved: {out}")
    plt.show()


# ── Multi-rung figure ─────────────────────────────────────────────────────────

def plot_multi(results, r_ism, r_cgm, out, dpi):
    """
    Three-panel figure:
      Left:   C/Si ratio in ISM and CGM across rungs
      Middle: CGM/ISM C/Si enhancement factor (>1 = carbon expelled to CGM)
      Right:  f_C radial profiles for selected rungs
    """
    rung_names  = [r['rung']      for r in results]
    ism_cs      = [r['s_ism']['cs_ratio']  for r in results]
    cgm_cs      = [r['s_cgm']['cs_ratio']  for r in results]
    enhancement = [cgm / ism if (not np.isnan(cgm) and not np.isnan(ism) and ism > 0)
                   else np.nan
                   for cgm, ism in zip(cgm_cs, ism_cs)]
    x = np.arange(len(results))

    fig = plt.figure(figsize=(16, 5))
    gs  = GridSpec(1, 3, figure=fig, wspace=0.38)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    # ── Panel 1: C/Si ratio per rung ─────────────────────────────────────
    ax1.plot(x, ism_cs, 'o-', color=ISM_COLOR, linewidth=2,
             markersize=6, label=f'ISM  ($r<{r_ism:.0f}$ kpc)')
    ax1.plot(x, cgm_cs, 's--', color=CGM_COLOR, linewidth=2,
             markersize=6, label=f'CGM  ({r_ism:.0f}–{r_cgm:.0f} kpc)')

    # Reference: expected C/Si from birth properties alone
    # SNII: fC=0.1 → C/Si = 0.1/0.9 = 0.111
    # AGB:  fC=0.6 → C/Si = 0.6/0.4 = 1.5
    # Combined depends on yield × rate, but mark SNII-only for reference
    ax1.axhline(0.1/0.9, color='#aaa', linestyle=':', linewidth=1,
                label=r'SNII-only $f_C=0.1$')

    ax1.set_xticks(x); ax1.set_xticklabels(rung_names, fontsize=9)
    ax1.set_ylabel('C/Si mass ratio', fontsize=11)
    ax1.set_title('C/Si by Region', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=8.5)
    ax1.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
    ax1.set_axisbelow(True)

    # ── Panel 2: CGM/ISM C/Si enhancement ────────────────────────────────
    ax2.plot(x, enhancement, 'D-', color='#444', linewidth=2, markersize=6)
    ax2.axhline(1.0, color='#e63946', linestyle='--', linewidth=1.5,
                label='No segregation (= 1)')
    ax2.fill_between(x, enhancement, 1.0,
                     where=[e > 1 if not np.isnan(e) else False
                            for e in enhancement],
                     alpha=0.15, color=CGM_COLOR,
                     label='C enriched in CGM')
    ax2.fill_between(x, enhancement, 1.0,
                     where=[e < 1 if not np.isnan(e) else False
                            for e in enhancement],
                     alpha=0.15, color=ISM_COLOR,
                     label='C enriched in ISM')
    ax2.set_xticks(x); ax2.set_xticklabels(rung_names, fontsize=9)
    ax2.set_ylabel('CGM C/Si  /  ISM C/Si', fontsize=11)
    ax2.set_title('CGM/ISM C/Si Enhancement', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=8.5)
    ax2.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
    ax2.set_axisbelow(True)

    # ── Panel 3: f_C radial profiles for selected rungs ───────────────────
    profile_results = [r for r in results if r['rung'] in PROFILE_RUNGS]
    colors = cm.plasma(np.linspace(0.1, 0.9, len(profile_results)))

    for res, col in zip(profile_results, colors):
        rc  = res['rc']
        fp  = res['fc_profile']
        valid = ~np.isnan(fp)
        if valid.sum() == 0:
            continue
        ax3.plot(rc[valid], fp[valid], linewidth=1.8,
                 color=col, label=res['rung'])

    ax3.axvline(r_ism, color='#888', linestyle='--', linewidth=1.0,
                label=f'{r_ism:.0f} kpc')
    ax3.axhline(0.1, color=SI_COLOR, linestyle=':', linewidth=1.0, alpha=0.7,
                label=r'SNII $f_C=0.1$')
    ax3.axhline(0.6, color=C_COLOR,  linestyle=':', linewidth=1.0, alpha=0.7,
                label=r'AGB $f_C=0.6$')
    ax3.set_xlabel('Radius (pkpc)', fontsize=11)
    ax3.set_ylabel(r'Mass-weighted $\langle f_C \rangle$', fontsize=11)
    ax3.set_title(r'$f_C$ Radial Profile (selected rungs)',
                  fontsize=12, fontweight='bold')
    ax3.set_xlim(0, r_cgm); ax3.set_ylim(0, 0.75)
    ax3.legend(fontsize=8.5, loc='upper left', ncol=2)
    ax3.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)

    # Supertitle from first loaded rung
    z   = results[0]['redshift']
    M   = results[0]['halo_mass']
    fig.suptitle(f'Halo 569  $\\cdot$  $1024^3$  $\\cdot$  '
                 f'M$_{{200}}$={M:.2e} M$_\\odot$  $\\cdot$  z={z:.2f}  '
                 f'$\\cdot$  ISM $r<{r_ism:.0f}$ kpc  $\\cdot$  '
                 f'CGM ${r_ism:.0f}$–${r_cgm:.0f}$ kpc',
                 fontsize=10, y=1.01)

    plt.savefig(out, dpi=dpi, bbox_inches='tight')
    print(f"\nSaved: {out}")
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='C vs Si partitioning: single rung or S0–S10 comparison')

    # Single-rung args
    parser.add_argument('--catalog',  default=None,
                        help='Subfind catalog path (single-rung mode)')
    parser.add_argument('--snapshot', default=None,
                        help='Snapshot base path (single-rung mode)')

    # Multi-rung args
    parser.add_argument('--base_dir', default=None,
                        help='Base directory containing rung subdirs')
    parser.add_argument('--rungs',    nargs='+', default=ALL_RUNGS,
                        help='Rung names to process (default: S0–S10)')
    parser.add_argument('--dir_suffix', default='_output_1024',
                        help='Suffix appended to rung name to form the rung directory '
                             '(default: _output_1024 → ../S0_output_1024/)')
    parser.add_argument('--snap_pat', default='snapdir_049/snapshot_049',
                        help='Snapshot path relative to each rung dir')
    parser.add_argument('--cat_pat',
                        default='groups_049/fof_subhalo_tab_049.0.hdf5',
                        help='Catalog path relative to each rung dir')

    # Shared args
    parser.add_argument('--r_ism',   type=float, default=20.0)
    parser.add_argument('--r_cgm',   type=float, default=294.0)
    parser.add_argument('--n_rbins', type=int,   default=20)
    parser.add_argument('--out',     default='carbon_silicate_partitioning.png')
    parser.add_argument('--dpi',     type=int,   default=150)
    args = parser.parse_args()

    # ── Single-rung mode ──────────────────────────────────────────────────
    if args.base_dir is None:
        if args.catalog is None or args.snapshot is None:
            parser.error("Provide either --base_dir OR both --catalog and --snapshot")
        data = load_rung(args.snapshot, args.catalog, args.r_cgm, verbose=True)
        if data is None:
            print("ERROR: could not load data."); return
        plot_single(data, args.r_ism, args.r_cgm, args.out, args.dpi)
        return

    # ── Multi-rung mode ───────────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print(f"  Multi-rung C/Si partitioning  "
          f"ISM r<{args.r_ism:.0f}kpc  CGM {args.r_ism:.0f}–{args.r_cgm:.0f}kpc")
    print(f"{'═'*70}")
    print(f"  {'Rung':4s} | {'ISM C/Si':>9} {'⟨fC⟩':>6} {'⟨a⟩nm':>7} {'N':>7} "
          f"| {'CGM C/Si':>9} {'⟨fC⟩':>6} {'N':>7} | {'enh':>6}")
    print(f"  {'─'*68}")

    results = []
    for rung in args.rungs:
        snap_path = os.path.join(args.base_dir, f'{rung}{args.dir_suffix}', args.snap_pat)
        cat_path  = os.path.join(args.base_dir, f'{rung}{args.dir_suffix}', args.cat_pat)

        print(f"  Loading {rung}...", end=' ', flush=True)
        data = load_rung(snap_path, cat_path, args.r_cgm, verbose=False)
        if data is None:
            print("SKIPPED (files not found)")
            continue
        print(f"z={data['redshift']:.2f}  N_dust={len(data['r']):,}")

        r = data['r']
        ism_mask = r < args.r_ism
        cgm_mask = (r >= args.r_ism) & (r < args.r_cgm)

        s_ism = region_stats(ism_mask, data['masses'],
                             data['carbon_frac'], data['grain_radius'])
        s_cgm = region_stats(cgm_mask, data['masses'],
                             data['carbon_frac'], data['grain_radius'])

        print_rung_summary(rung, s_ism, s_cgm)

        rc, fp = radial_fc_profile(r, data['masses'], data['carbon_frac'],
                                   args.r_cgm, args.n_rbins)

        results.append(dict(
            rung=rung, s_ism=s_ism, s_cgm=s_cgm,
            rc=rc, fc_profile=fp,
            redshift=data['redshift'], halo_mass=data['halo_mass'],
        ))

    if len(results) == 0:
        print("ERROR: no rungs loaded."); return

    if len(results) == 1:
        # Fall back to single-rung figure
        rung = results[0]
        data_single = load_rung(
            os.path.join(args.base_dir, rung['rung'], args.snap_pat),
            os.path.join(args.base_dir, rung['rung'], args.cat_pat),
            args.r_cgm)
        plot_single(data_single, args.r_ism, args.r_cgm, args.out, args.dpi)
    else:
        plot_multi(results, args.r_ism, args.r_cgm, args.out, args.dpi)


if __name__ == '__main__':
    main()
