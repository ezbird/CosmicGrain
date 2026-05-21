#!/usr/bin/env python3
"""
plot_dust_sfr_delay.py
----------------------
Overlays dust mass and SFR vs cosmic time for S10 to verify (or revise)
the claimed ~1 Gyr delay between peak star formation and peak dust abundance
(Parente+2025).

Dust mass comes from the run log file (fast, no snapshot I/O).
SFR comes from the gas StarFormationRate field summed within R200
at each snapshot (Method B from plot_sfr_halo.py — instantaneous,
no smoothing artifacts from mergers).

Output:
    dust_figures/dust_sfr_delay_{run}_{res}.png

Usage:
    python plot_dust_sfr_delay.py --run S10 --res 1024
    python plot_dust_sfr_delay.py --run S10 --res 1024 --smooth-sfr 0.15
"""

import os, re, glob, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
from scipy.ndimage import gaussian_filter1d

# ─────────────────────────────────────────────────────────────────────────────
# Constants / config
# ─────────────────────────────────────────────────────────────────────────────

SEC_PER_GYR  = 3.15576e16
MPC_IN_CM    = 3.085678e24
MSUN_IN_G    = 1.989e33
FIGDIR       = 'dust_figures'
os.makedirs(FIGDIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Cosmology helpers
# ─────────────────────────────────────────────────────────────────────────────

def cosmic_time_Gyr(a, h=0.6774, Omega0=0.3089, OmegaL=0.6911):
    """Age of Universe at scale factor a (flat ΛCDM)."""
    H0_inv_Gyr = MPC_IN_CM / (100.0 * h * 1e5) / SEC_PER_GYR
    a_arr = np.linspace(1e-4, float(a), 2000)
    Ez    = np.sqrt(Omega0 / a_arr**3 + OmegaL)
    return H0_inv_Gyr * np.trapz(1.0 / (a_arr * Ez), a_arr)


def z_to_t(z_arr, h=0.6774, Omega0=0.3089, OmegaL=0.6911):
    return np.array([cosmic_time_Gyr(1.0/(1.0+z), h, Omega0, OmegaL)
                     for z in z_arr])


def build_redshift_axis(ax, t_min, t_max, h=0.6774, Omega0=0.3089, OmegaL=0.6911):
    """Add a twin x-axis on top showing redshift."""
    # Dense t→z lookup
    a_fine = np.linspace(1e-4, 1.0, 5000)
    t_fine = np.array([cosmic_time_Gyr(a, h, Omega0, OmegaL) for a in a_fine])
    z_fine = 1.0 / a_fine - 1.0   # decreasing

    def t_of_z(z):
        return float(np.interp(1.0/(1.0+z), a_fine, t_fine))

    z_ticks = [0, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 8]
    valid   = [(z, t_of_z(z)) for z in z_ticks
               if t_min <= t_of_z(z) <= t_max]
    if not valid:
        return

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks([tv for _, tv in valid])
    ax2.set_xticklabels([str(zv) for zv, _ in valid], fontsize=9)
    ax2.set_xlabel('Redshift $z$', fontsize=11, labelpad=6)
    ax2.tick_params(axis='x', direction='in', top=True)
    return ax2


# ─────────────────────────────────────────────────────────────────────────────
# Log parser — dust mass vs redshift
# ─────────────────────────────────────────────────────────────────────────────

def parse_dust_mass_from_log(run, resolution):
    """
    Pull (z, dust_mass_Msun) from the Gadget-4 run log.
    Returns (z_arr, t_arr, mass_arr) sorted by ascending time.
    """
    patterns = [
        f'{run}_output_{resolution}/output_{run}_{resolution}.log',
        f'{run}_output/output_{run}.log',
    ]
    logfile = None
    for p in patterns:
        if os.path.exists(p):
            logfile = p
            break
    if logfile is None:
        cands = glob.glob(f'**/*{run}*{resolution}*.log', recursive=True)
        if cands:
            logfile = cands[0]
    if logfile is None:
        raise FileNotFoundError(f'No log file found for {run} at {resolution}^3')
    print(f'  Log: {logfile}')

    z_list, m_list = [], []
    re_block = re.compile(r'=== STATISTICS \(global\) ===')
    re_az    = re.compile(r'\|a=([\d.]+) z=([\d.]+)\]')
    re_mass  = re.compile(r'STATISTICS Particles:\s+\d+\s+Mass:\s+([\d.e+\-]+)')

    with open(logfile) as f:
        content = f.read()

    blocks = re_block.split(content)
    for block in blocks[1:]:
        m_az   = re_az.search(block)
        m_mass = re_mass.search(block)
        if not m_az or not m_mass:
            continue
        a    = float(m_az.group(1))
        z    = float(m_az.group(2))
        mass = float(m_mass.group(1)) * 1e10   # code (1e10 Msun) → Msun
        z_list.append(z)
        m_list.append(mass)

    if not z_list:
        raise RuntimeError('No STATISTICS blocks found in log.')

    z_arr = np.array(z_list)
    m_arr = np.array(m_list)

    # Read cosmology from any snapshot header for accurate time conversion
    h, Omega0, OmegaL = _read_cosmo(run, resolution)
    t_arr = z_to_t(z_arr, h, Omega0, OmegaL)

    # Sort by ascending time (z descending → t ascending)
    idx   = np.argsort(t_arr)
    return z_arr[idx], t_arr[idx], m_arr[idx]


def _read_cosmo(run, resolution):
    """Read h, Omega0, OmegaL from the first available snapshot header."""
    import h5py
    out_dir = f'{run}_output_{resolution}'
    for snapdir in sorted(glob.glob(os.path.join(out_dir, 'snapdir_*'))):
        for f in sorted(glob.glob(os.path.join(snapdir, 'snapshot_*.0.hdf5')) +
                        glob.glob(os.path.join(snapdir, 'snapshot_*.hdf5'))):
            try:
                with h5py.File(f, 'r') as hf:
                    a = hf['Header'].attrs
                    return (float(a.get('HubbleParam', 0.6774)),
                            float(a.get('Omega0',       0.3089)),
                            float(a.get('OmegaLambda',  0.6911)))
            except Exception:
                pass
    return 0.6774, 0.3089, 0.6911   # Planck 2015 fallback


# ─────────────────────────────────────────────────────────────────────────────
# Snapshot SFR (Method B from plot_sfr_halo.py)
# ─────────────────────────────────────────────────────────────────────────────

def _subfiles(snap_base):
    fs = sorted(glob.glob(snap_base + '.*.hdf5'))
    if not fs:
        s = snap_base + '.hdf5'
        fs = [s] if os.path.exists(s) else []
    return fs


def _find_catalog(run, snap_base, resolution):
    m = re.search(r'snapshot_(\d+)$', snap_base)
    if not m: return None
    sn  = m.group(1)
    gd  = os.path.join(f'{run}_output_{resolution}', f'groups_{sn}')
    cs  = sorted(glob.glob(os.path.join(gd, f'fof_subhalo_tab_{sn}.*.hdf5')))
    return cs[0] if cs else None


def _get_center_r200(run, snap_base, resolution):
    import h5py
    cat = _find_catalog(run, snap_base, resolution)
    if cat is None: return None, None
    try:
        with h5py.File(cat, 'r') as hf:
            if 'Group' not in hf: return None, None
            grp = hf['Group']
            if 'GroupPos' not in grp or grp['GroupPos'].shape[0] == 0:
                return None, None
            ctr  = grp['GroupPos'][0].astype(float)
            r200 = (float(grp['Group_R_Mean200'][0])
                    if 'Group_R_Mean200' in grp else None)
            return ctr, r200
    except Exception:
        return None, None


def measure_sfr_gas(run, resolution, smooth_Gyr=0.1):
    """
    Instantaneous SFR from PartType0 StarFormationRate field, summed
    within R200 at each snapshot that has a valid SubFind catalog.
    Returns (z_arr, t_arr, sfr_Msun_yr) sorted by ascending time.
    """
    import h5py
    out_dir = f'{run}_output_{resolution}'
    snaps = []
    for sd in sorted(glob.glob(os.path.join(out_dir, 'snapdir_*'))):
        for f in sorted(glob.glob(os.path.join(sd, 'snapshot_*.0.hdf5'))):
            snaps.append(re.sub(r'\.0\.hdf5$', '', f))
        for f in sorted(glob.glob(os.path.join(sd, 'snapshot_*.hdf5'))):
            if '.0.hdf5' not in f:
                snaps.append(re.sub(r'\.hdf5$', '', f))
    snaps = sorted(set(snaps))
    if not snaps:
        return None, None, None

    h, Omega0, OmegaL = _read_cosmo(run, resolution)
    sfr_field = None
    unit_factor = 1.0
    records = []

    for snap_base in snaps:
        try:
            with h5py.File(snap_base + '.0.hdf5', 'r') as hf:
                a = float(hf['Header'].attrs['Time'])
                z = float(hf['Header'].attrs['Redshift'])
        except Exception:
            continue
        if z > 8.0: continue

        ctr, r200 = _get_center_r200(run, snap_base, resolution)
        if ctr is None or r200 is None: continue

        t_Gyr    = cosmic_time_Gyr(a, h, Omega0, OmegaL)
        sfr_total = 0.0

        for fname in _subfiles(snap_base):
            try:
                with h5py.File(fname, 'r') as hf:
                    if 'PartType0' not in hf: continue
                    pt = hf['PartType0']

                    if sfr_field is None:
                        for cand in ['StarFormationRate', 'Sfr', 'SFR']:
                            if cand in pt:
                                sfr_field = cand
                                sample = pt[cand][:]
                                if sample.max() > 1e4:
                                    # code units — convert
                                    try:
                                        pa = hf['Parameters'].attrs
                                        um = float(pa.get('UnitMass_in_g',     1.989e43))
                                        ul = float(pa.get('UnitLength_in_cm',  3.085678e21))
                                        uv = float(pa.get('UnitVelocity_in_cm_per_s', 1e5))
                                        ut = ul / uv
                                        unit_factor = (um / MSUN_IN_G) / (ut / 3.15576e7)
                                    except Exception:
                                        unit_factor = 1.0
                                print(f'    SFR field="{sfr_field}"  '
                                      f'unit_factor={unit_factor:.3e}')
                                break

                    if sfr_field not in pt: continue
                    pos = pt['Coordinates'][:]
                    sfr = pt[sfr_field][:]
                    r   = np.linalg.norm(pos - ctr, axis=1)
                    mask = (r < r200) & (sfr > 0)
                    sfr_total += sfr[mask].sum() * unit_factor
            except Exception:
                pass

        records.append((t_Gyr, z, sfr_total))

    if len(records) < 3: return None, None, None
    records.sort()
    t_arr   = np.array([r[0] for r in records])
    z_arr   = np.array([r[1] for r in records])
    sfr_arr = np.array([r[2] for r in records])

    if smooth_Gyr > 0 and len(t_arr) > 5:
        # Convert sigma from Gyr to index units
        dt_mean = np.mean(np.diff(t_arr))
        sigma_idx = smooth_Gyr / dt_mean
        sfr_arr = gaussian_filter1d(sfr_arr, sigma=sigma_idx)

    sfr_arr = np.maximum(sfr_arr, 0.0)
    return z_arr, t_arr, sfr_arr


# ─────────────────────────────────────────────────────────────────────────────
# Peak finder
# ─────────────────────────────────────────────────────────────────────────────

def find_peak(t_arr, y_arr, t_min=0.5):
    """Return (t_peak, y_peak) ignoring early noisy period before t_min Gyr."""
    mask = t_arr > t_min
    if not np.any(mask):
        idx = np.argmax(y_arr)
    else:
        sub_idx = np.argmax(y_arr[mask])
        idx = np.where(mask)[0][sub_idx]
    return t_arr[idx], y_arr[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def make_plot(run, resolution, smooth_sfr_Gyr,
              z_dust, t_dust, m_dust,
              z_sfr,  t_sfr,  sfr):

    t_peak_dust, m_peak = find_peak(t_dust, m_dust)
    t_peak_sfr,  s_peak = find_peak(t_sfr,  sfr)
    delay_Gyr = t_peak_dust - t_peak_sfr

    z_peak_dust = float(np.interp(t_peak_dust, t_dust, z_dust))
    z_peak_sfr  = float(np.interp(t_peak_sfr,  t_sfr,  z_sfr))

    print(f'\n  Peak SFR:  t = {t_peak_sfr:.2f} Gyr  (z = {z_peak_sfr:.2f})'
          f'  SFR = {s_peak:.1f} Msun/yr')
    print(f'  Peak dust: t = {t_peak_dust:.2f} Gyr  (z = {z_peak_dust:.2f})'
          f'  M_dust = {m_peak:.3e} Msun')
    print(f'  Delay:     {delay_Gyr:.2f} Gyr')

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    ax2 = ax1.twinx()

    t_min = min(t_dust.min(), t_sfr.min())
    t_max = max(t_dust.max(), t_sfr.max())

    # ── Dust mass (left axis, blue) ───────────────────────────────────────────
    ax1.plot(t_dust, m_dust, color='#1f77b4', lw=2.2,
             label=r'Dust mass ($M_\odot$)', zorder=3)
    ax1.axvline(t_peak_dust, color='#1f77b4', ls=':', lw=1.5, zorder=2)
    ax1.set_ylabel(r'Total dust mass ($M_\odot$)', color='#1f77b4', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    ax1.set_yscale('log')

    # ── SFR (right axis, red) ─────────────────────────────────────────────────
    ax2.plot(t_sfr, sfr, color='#d62728', lw=2.0, ls='--',
             label=r'SFR ($M_\odot\,{\rm yr}^{-1}$)', zorder=3)
    ax2.axvline(t_peak_sfr, color='#d62728', ls=':', lw=1.5, zorder=2)
    ax2.set_ylabel(r'SFR ($M_\odot\,{\rm yr}^{-1}$)', color='#d62728', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='#d62728')
    ax2.set_yscale('log')

    # ── Delay annotation ──────────────────────────────────────────────────────
    y_ann = ax1.get_ylim()
    y_mid = 10 ** (0.5 * (np.log10(y_ann[0] + 1e-10) + np.log10(y_ann[1])))

    ax1.annotate('',
        xy=(t_peak_dust, y_mid), xytext=(t_peak_sfr, y_mid),
        arrowprops=dict(arrowstyle='<->', color='k', lw=1.5))
    ax1.text(0.5 * (t_peak_sfr + t_peak_dust), y_mid * 1.15,
             f'$\\Delta t = {delay_Gyr:.2f}$ Gyr',
             ha='center', va='bottom', fontsize=10,
             bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.85))

    # ── Peak labels ───────────────────────────────────────────────────────────
    ax1.text(t_peak_dust + 0.08, m_peak,
             f'$z={z_peak_dust:.2f}$\n$t={t_peak_dust:.1f}$ Gyr',
             color='#1f77b4', fontsize=8.5, va='center',
             bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#1f77b4', alpha=0.8))
    ax2.text(t_peak_sfr + 0.08, s_peak,
             f'$z={z_peak_sfr:.2f}$\n$t={t_peak_sfr:.1f}$ Gyr',
             color='#d62728', fontsize=8.5, va='center',
             bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#d62728', alpha=0.8))

    # ── Axes ──────────────────────────────────────────────────────────────────
    ax1.set_xlabel('Age of Universe (Gyr)', fontsize=12)
    ax1.set_xlim(t_min, t_max)
    ax1.grid(True, alpha=0.2, which='both')

    # ── Redshift twin axis on top ─────────────────────────────────────────────
    h, Omega0, OmegaL = _read_cosmo(run, resolution)
    build_redshift_axis(ax1, t_min, t_max, h, Omega0, OmegaL)

    # ── Legend (combined) ─────────────────────────────────────────────────────
    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2,
               fontsize=9, loc='lower right', framealpha=0.9)

    ax1.set_title(
        f'{run} ({resolution}$^3$) — Dust mass vs SFR temporal evolution\n'
        f'Parente+2025 claim: $\\Delta t \\sim 1$ Gyr',
        fontsize=11)

    # ── Parente+2025 reference band ───────────────────────────────────────────
    # Shade t_peak_sfr to t_peak_sfr + 1 Gyr as the "expected" dust peak window
    ax1.axvspan(t_peak_sfr + 0.5, t_peak_sfr + 1.5,
                alpha=0.07, color='green', zorder=0,
                label='Expected dust peak (Parente+2025: $\\Delta t \\sim 1$ Gyr)')

    out = os.path.join(FIGDIR, f'dust_sfr_delay_{run}_{resolution}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\n  Saved: {out}')
    return delay_Gyr, z_peak_sfr, z_peak_dust


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description='Dust mass vs SFR temporal delay plot')
    p.add_argument('--run',        default='S10')
    p.add_argument('--res',        type=int,   default=1024)
    p.add_argument('--smooth-sfr', type=float, default=0.15,
                   help='Gaussian smoothing for SFR in Gyr (default 0.15)')
    args = p.parse_args()

    print(f'\n=== Dust–SFR delay: {args.run} at {args.res}^3 ===\n')

    print('Loading dust mass from log...')
    z_dust, t_dust, m_dust = parse_dust_mass_from_log(args.run, args.res)
    print(f'  {len(z_dust)} log entries, '
          f'z=[{z_dust.max():.2f} → {z_dust.min():.2f}]')

    print('\nLoading SFR from snapshots...')
    z_sfr, t_sfr, sfr = measure_sfr_gas(args.run, args.res, args.smooth_sfr)
    if z_sfr is None:
        print('ERROR: could not load SFR from snapshots.')
        return

    print(f'  {len(z_sfr)} snapshots, '
          f'z=[{z_sfr.max():.2f} → {z_sfr.min():.2f}]')

    delay, z_sfr_peak, z_dust_peak = make_plot(
        args.run, args.res, args.smooth_sfr,
        z_dust, t_dust, m_dust,
        z_sfr,  t_sfr,  sfr)

    print(f'\n  ┌─────────────────────────────────────────┐')
    print(f'  │  Summary for paper paragraph            │')
    print(f'  ├─────────────────────────────────────────┤')
    print(f'  │  Peak SFR:   z = {z_sfr_peak:.2f}                   │')
    print(f'  │  Peak dust:  z = {z_dust_peak:.2f}                   │')
    print(f'  │  Delay:      {delay:.2f} Gyr                    │')
    print(f'  │  Parente+25: ~1 Gyr                     │')
    print(f'  └─────────────────────────────────────────┘')


if __name__ == '__main__':
    main()
