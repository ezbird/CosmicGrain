"""
plot_sfr_halo.py — Star Formation Rate vs Redshift for the main halo
=====================================================================
Two independent SFR methods are computed and overlaid:

  Method A — dM*/dt  (stellar mass differential)
    Sums PartType4 stellar mass within R200 at each snapshot and
    differentiates. Sensitive to mergers / aperture crossings; smoothed
    with a Gaussian window (--dt-smooth).

  Method B — Gas SFR field  (instantaneous)
    Sums the Sfr field of PartType0 gas cells within R200 at each
    snapshot. This is the Kennicutt–Schmidt SFR computed on-the-fly
    by Gadget-4 and is free of aperture-crossing noise.

Both methods are plotted on the same axes so you can assess where
they agree and where mergers / smoothing artifacts diverge.

Usage:
    python plot_sfr_halo.py --res 1024 --run S10
    python plot_sfr_halo.py --res 1024 --runs S4 S8 S10
    python plot_sfr_halo.py --res 512  --run S10 --dt-smooth 0.3

Output:
    dust_figures/sfr_vs_z_{run}_{res}.png
    dust_figures/sfr_vs_z_comparison_{res}.png
"""

import os, re, glob, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

# ─────────────────────────────────────────────────────────────────────────────
# halo_utils (optional)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from halo_utils import load_target_halo
    HALO_UTILS_AVAILABLE = True
except ImportError:
    HALO_UTILS_AVAILABLE = False

RUN_CONFIGS = {
    'S0':  {'label': 'S0',  'color': '#888888'},
    'S1':  {'label': 'S1',  'color': '#1f77b4'},
    'S2':  {'label': 'S2',  'color': '#ff7f0e'},
    'S3':  {'label': 'S3',  'color': '#2ca02c'},
    'S4':  {'label': 'S4',  'color': '#d62728'},
    'S5':  {'label': 'S5',  'color': '#9467bd'},
    'S6':  {'label': 'S6',  'color': '#8c564b'},
    'S7':  {'label': 'S7',  'color': '#e377c2'},
    'S8':  {'label': 'S8',  'color': '#17becf'},
    'S9':  {'label': 'S9',  'color': '#bcbd22'},
    'S10': {'label': 'S10', 'color': '#000000'},
}

FIGDIR     = 'dust_figures'
RESOLUTION = 1024
os.makedirs(FIGDIR, exist_ok=True)

SEC_PER_GYR = 3.15576e16
MPC_IN_CM   = 3.085678e24


# ─────────────────────────────────────────────────────────────────────────────
# Cosmology
# ─────────────────────────────────────────────────────────────────────────────

def cosmic_time_Gyr(a, h=0.7, Omega0=0.3, OmegaL=0.7):
    H0_inv_Gyr = MPC_IN_CM / (100.0 * h * 1e5) / SEC_PER_GYR
    a_arr = np.linspace(1e-4, float(a), 2000)
    Ez    = np.sqrt(Omega0 / a_arr**3 + OmegaL)
    return H0_inv_Gyr * np.trapz(1.0 / (a_arr * Ez), a_arr)


def build_age_axis(ax, z_min, z_max, h=0.7, Omega0=0.3, OmegaL=0.7):
    """
    Add a twin x-axis on top showing age of Universe in Gyr.
    We use a parametric approach: place ticks at chosen age values,
    convert each to its redshift, and position the tick there.
    """
    # Build a dense z→t lookup table
    a_fine = np.linspace(1e-4, 1.0, 5000)
    t_fine = np.array([cosmic_time_Gyr(a, h, Omega0, OmegaL) for a in a_fine])
    z_fine = 1.0 / a_fine - 1.0  # decreasing

    def z_of_t(t_Gyr):
        # t_fine is increasing (a increases → t increases, z decreases)
        return float(np.interp(t_Gyr, t_fine, z_fine))

    # Candidate tick ages
    candidates = [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    valid = [(t, z_of_t(t)) for t in candidates
             if z_min <= z_of_t(t) <= z_max]
    if not valid:
        return

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())   # must match the redshift axis direction
    ax2.set_xticks([zv for _, zv in valid])
    ax2.set_xticklabels([f'{tv:g}' for tv, _ in valid], fontsize=9)
    ax2.set_xlabel('Age of Universe (Gyr)', fontsize=11, labelpad=6)
    # Ensure tick marks point inward and don't fight the main title
    ax2.tick_params(axis='x', direction='in', top=True)
    return ax2


# ─────────────────────────────────────────────────────────────────────────────
# Snapshot / catalog helpers
# ─────────────────────────────────────────────────────────────────────────────

def find_snapshots(run):
    out_dir = f'{run}_output_{RESOLUTION}'
    if not os.path.isdir(out_dir):
        return []
    seen, bases = set(), []
    for sd in sorted(glob.glob(os.path.join(out_dir, 'snapdir_*'))):
        for f in sorted(glob.glob(os.path.join(sd, 'snapshot_*.0.hdf5'))):
            b = re.sub(r'\.0\.hdf5$', '', f)
            if b not in seen: seen.add(b); bases.append(b)
        for f in sorted(glob.glob(os.path.join(sd, 'snapshot_*.hdf5'))):
            if '.0.hdf5' in f: continue
            b = re.sub(r'\.hdf5$', '', f)
            if b not in seen: seen.add(b); bases.append(b)
    return sorted(bases)


def subfiles(snap_base):
    fs = sorted(glob.glob(snap_base + '.*.hdf5'))
    if not fs:
        s = snap_base + '.hdf5'
        fs = [s] if os.path.exists(s) else []
    return fs


def read_header(snap_base):
    import h5py
    for suf in ['.0.hdf5', '.hdf5']:
        f = snap_base + suf
        if not os.path.exists(f): continue
        try:
            with h5py.File(f, 'r') as hf:
                a = hf['Header'].attrs
                ul = float(a.get('UnitLength_in_cm',       3.085678e21))
                uv = float(a.get('UnitVelocity_in_cm_per_s', 1e5))
                return dict(
                    h      = float(a.get('HubbleParam',   0.7)),
                    Omega0 = float(a.get('Omega0',         0.3)),
                    OmegaL = float(a.get('OmegaLambda',   0.7)),
                    a      = float(a.get('Time',           1.0)),
                    um_cgs = float(a.get('UnitMass_in_g', 1.989e43)),
                    ul_cm  = ul,
                    ut_s   = ul / uv,          # code time unit in seconds
                )
        except Exception:
            pass
    return dict(h=0.7, Omega0=0.3, OmegaL=0.7, a=1.0,
                um_cgs=1.989e43, ul_cm=3.085678e21, ut_s=3.085678e21/1e5)


def find_catalog(run, snap_base):
    m = re.search(r'snapshot_(\d+)$', snap_base)
    if not m: return None
    sn  = m.group(1)
    gd  = os.path.join(f'{run}_output_{RESOLUTION}', f'groups_{sn}')
    cats = sorted(glob.glob(os.path.join(gd, f'fof_subhalo_tab_{sn}.*.hdf5')))
    return cats[0] if cats else None


def get_halo_center_r200(run, snap_base):
    """Return (ctr, r200) both in comoving kpc/h, from the SubFind catalog."""
    import h5py
    cat = find_catalog(run, snap_base)
    if cat is None:
        return None, None
    try:
        with h5py.File(cat, 'r') as hf:
            if 'Group' not in hf: return None, None
            grp = hf['Group']
            if 'GroupPos' not in grp or grp['GroupPos'].shape[0] == 0:
                return None, None
            ctr  = grp['GroupPos'][0].astype(float)         # ckpc/h
            r200 = float(grp['Group_R_Mean200'][0]) \
                   if 'Group_R_Mean200' in grp else None     # ckpc/h
            return ctr, r200
    except Exception as e:
        return None, None


# ─────────────────────────────────────────────────────────────────────────────
# Method A — dM*/dt
# ─────────────────────────────────────────────────────────────────────────────

def measure_sfr_stellar(run, dt_smooth_Gyr=0.3):
    """
    Sum PartType4 stellar mass within R200 at each snapshot;
    differentiate to get SFR.  Gaussian-smoothed.
    Returns (z_mid, t_mid, sfr_Msun_yr).
    """
    import h5py
    snaps = find_snapshots(run)
    if not snaps: return None, None, None

    records = []
    for snap_base in snaps:
        hdr = read_header(snap_base)
        a   = hdr['a']
        z   = 1.0 / a - 1.0
        if z > 10.0: continue

        ctr, r200 = get_halo_center_r200(run, snap_base)
        if ctr is None or r200 is None: continue

        t_Gyr = cosmic_time_Gyr(a, hdr['h'], hdr['Omega0'], hdr['OmegaL'])
        Msun  = hdr['um_cgs'] / 1.989e33
        M_star = 0.0

        for fname in subfiles(snap_base):
            try:
                with h5py.File(fname, 'r') as hf:
                    if 'PartType4' not in hf: continue
                    pt   = hf['PartType4']
                    pos  = pt['Coordinates'][:]
                    mass = pt['Masses'][:]
                    r    = np.linalg.norm(pos - ctr, axis=1)
                    mask = r < r200
                    if mask.sum() == 0: continue
                    M_star += mass[mask].sum() * Msun
            except Exception:
                pass

        records.append((t_Gyr, z, M_star))

    if len(records) < 2: return None, None, None
    records.sort()
    t_arr = np.array([r[0] for r in records])
    z_arr = np.array([r[1] for r in records])
    M_arr = np.array([r[2] for r in records])

    dt_yr = np.diff(t_arr) * 1e9
    dM    = np.diff(M_arr)
    sfr   = np.where(dt_yr > 0, dM / dt_yr, 0.0)
    t_mid = 0.5 * (t_arr[:-1] + t_arr[1:])
    z_mid = 0.5 * (z_arr[:-1] + z_arr[1:])

    if dt_smooth_Gyr > 0 and len(t_mid) > 3:
        sfr = gaussian_smooth(t_mid, sfr, dt_smooth_Gyr)
    sfr = np.maximum(sfr, 0.0)
    return z_mid, t_mid, sfr


# ─────────────────────────────────────────────────────────────────────────────
# Method B — instantaneous gas SFR field
# ─────────────────────────────────────────────────────────────────────────────

def measure_sfr_gas(run, dt_smooth_Gyr=0.1):
    """
    Sum PartType0 Sfr field (M_sun/yr in Gadget-4 internal units converted
    below) within R200 at each snapshot.
    Returns (z_arr, t_arr, sfr_Msun_yr).
    """
    import h5py
    snaps = find_snapshots(run)
    if not snaps: return None, None, None

    # On first valid snapshot, detect the SFR field name and its unit
    sfr_field = None  # will be set on first encounter

    records = []
    for snap_base in snaps:
        hdr = read_header(snap_base)
        a   = hdr['a']
        z   = 1.0 / a - 1.0
        if z > 10.0: continue

        ctr, r200 = get_halo_center_r200(run, snap_base)
        if ctr is None or r200 is None: continue

        t_Gyr = cosmic_time_Gyr(a, hdr['h'], hdr['Omega0'], hdr['OmegaL'])

        # Gadget-4 stores SFR in M_sun/yr natively when STARFORMATION is on.
        # If the unit is code_mass/code_time we convert:
        #   sfr_Msun_yr = sfr_code * um_cgs/1.989e33 / (ut_s/3.15576e7)
        # We try to detect which case we're in from the field magnitude.
        Msun_per_yr_factor = 1.0   # assume native M_sun/yr until proven otherwise
        sfr_total = 0.0

        for fname in subfiles(snap_base):
            try:
                with h5py.File(fname, 'r') as hf:
                    if 'PartType0' not in hf: continue
                    pt = hf['PartType0']

                    # Discover SFR field name on first encounter
                    if sfr_field is None:
                        candidates = ['StarFormationRate', 'Sfr',
                                      'SFR', 'star_formation_rate']
                        for c in candidates:
                            if c in pt:
                                sfr_field = c
                                # Heuristic: if max value is huge (> 1e4)
                                # it's likely in code units
                                sample = pt[c][:]
                                if sample.max() > 1e4:
                                    Msun_per_yr_factor = (
                                        hdr['um_cgs'] / 1.989e33
                                        / (hdr['ut_s'] / 3.15576e7))
                                print(f'    [gas SFR] field="{sfr_field}"  '
                                      f'unit_factor={Msun_per_yr_factor:.3e}')
                                break
                        if sfr_field is None:
                            print(f'    [gas SFR] no SFR field found in '
                                  f'PartType0. Fields: {list(pt.keys())}')
                            break

                    if sfr_field not in pt: continue
                    pos = pt['Coordinates'][:]
                    sfr = pt[sfr_field][:]
                    r   = np.linalg.norm(pos - ctr, axis=1)
                    mask = (r < r200) & (sfr > 0)
                    if mask.sum() == 0: continue
                    sfr_total += sfr[mask].sum() * Msun_per_yr_factor
            except Exception as e:
                pass

        if sfr_field is not None:
            records.append((t_Gyr, z, sfr_total))

    if len(records) < 2: return None, None, None
    records.sort()
    t_arr = np.array([r[0] for r in records])
    z_arr = np.array([r[1] for r in records])
    sfr_arr = np.array([r[2] for r in records])

    if dt_smooth_Gyr > 0 and len(t_arr) > 3:
        sfr_arr = gaussian_smooth(t_arr, sfr_arr, dt_smooth_Gyr)
    sfr_arr = np.maximum(sfr_arr, 0.0)
    return z_arr, t_arr, sfr_arr


# ─────────────────────────────────────────────────────────────────────────────
# Smoothing helper
# ─────────────────────────────────────────────────────────────────────────────

def gaussian_smooth(t, y, sigma_Gyr):
    y_s = np.zeros_like(y, dtype=float)
    for i in range(len(t)):
        w = np.exp(-0.5 * ((t - t[i]) / sigma_Gyr)**2)
        ws = w.sum()
        y_s[i] = np.dot(w, y) / ws if ws > 0 else y[i]
    return y_s


# ─────────────────────────────────────────────────────────────────────────────
# Main plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_sfr(runs, dt_smooth_Gyr=0.3, res=1024):
    global RESOLUTION
    RESOLUTION = res

    # If single run: one panel showing both methods
    # If multiple runs: one panel per method side by side
    single = (len(runs) == 1)

    if single:
        fig, ax = plt.subplots(1, 1, figsize=(9, 6))
        axes_stellar = [ax]
        axes_gas     = [ax]
    else:
        fig, (ax_s, ax_g) = plt.subplots(1, 2, figsize=(14, 6),
                                          sharey=True)
        axes_stellar = [ax_s]
        axes_gas     = [ax_g]
        ax_s.set_title('Method A — $dM_\star/dt$', fontsize=11)
        ax_g.set_title('Method B — Gas SFR field', fontsize=11)

    fig.subplots_adjust(top=0.82, hspace=0.35)

    all_z   = []
    hdr_ref = None

    for run in runs:
        cfg   = RUN_CONFIGS.get(run, {})
        color = cfg.get('color', 'black')
        lbl   = cfg.get('label', run)

        # ── Method A ──────────────────────────────────────────────────────────
        print(f'\n[Method A — dM*/dt]  {run}')
        z_s, t_s, sfr_s = measure_sfr_stellar(run, dt_smooth_Gyr)

        # ── Method B ──────────────────────────────────────────────────────────
        print(f'\n[Method B — gas SFR] {run}')
        z_g, t_g, sfr_g = measure_sfr_gas(run, dt_smooth_Gyr=0.05)

        # Grab cosmology once
        if hdr_ref is None:
            snaps = find_snapshots(run)
            if snaps: hdr_ref = read_header(snaps[-1])

        ax = axes_stellar[0]
        if z_s is not None:
            good = sfr_s > 0
            if single:
                ax.plot(z_s[good], sfr_s[good], color=color, lw=2.0,
                        ls='-', label=f'{lbl} — $dM_\star/dt$', alpha=0.85)
            else:
                ax.plot(z_s[good], sfr_s[good], color=color, lw=2.0,
                        ls='-', label=lbl, alpha=0.85)
            all_z.extend(z_s[good].tolist())

        ax = axes_gas[0]
        if z_g is not None:
            good = sfr_g > 0
            if single:
                ax.plot(z_g[good], sfr_g[good], color=color, lw=2.0,
                        ls='--', label=f'{lbl} — gas SFR', alpha=0.85)
            else:
                ax.plot(z_g[good], sfr_g[good], color=color, lw=2.0,
                        ls='-', label=lbl, alpha=0.85)
            all_z.extend(z_g[good].tolist())

    if not all_z:
        print('No SFR data to plot.')
        plt.close(fig)
        return

    z_min = max(0.0,  min(all_z) - 0.1)
    z_max = min(10.0, max(all_z) + 0.3)

    h      = hdr_ref['h']      if hdr_ref else 0.7
    Omega0 = hdr_ref['Omega0'] if hdr_ref else 0.3
    OmegaL = hdr_ref['OmegaL'] if hdr_ref else 0.7

    # Apply axis formatting to every unique axes object
    seen_axes = []
    for ax in (axes_stellar + axes_gas):
        if ax not in seen_axes:
            seen_axes.append(ax)

    for ax in seen_axes:
        ax.set_xlim(z_max, z_min)   # right→left: high-z on left
        ax.set_yscale('log')
        ax.set_xlabel('Redshift $z$', fontsize=12)
        ax.grid(True, alpha=0.25, which='both')
        ax.legend(fontsize=8, loc='upper right')
        build_age_axis(ax, z_min, z_max, h, Omega0, OmegaL)

    seen_axes[0].set_ylabel(r'SFR ($M_\odot\,{\rm yr}^{-1}$)', fontsize=12)

    suptitle = (f'Star Formation Rate within $R_{{200}}$  '
                f'— {RESOLUTION}$^3$')
    fig.suptitle(suptitle, fontsize=12, y=0.97)

    if single:
        fname = f'sfr_vs_z_{runs[0]}_{RESOLUTION}.png'
    else:
        fname = f'sfr_vs_z_comparison_{RESOLUTION}.png'
    out = os.path.join(FIGDIR, fname)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved: {out}')


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description='SFR vs redshift within R200 — two methods')
    p.add_argument('--run',  default='S10')
    p.add_argument('--runs', nargs='+', default=None)
    p.add_argument('--res',  type=int, default=1024)
    p.add_argument('--dt-smooth', type=float, default=0.3,
                   help='Gaussian smoothing for Method A in Gyr (default 0.3)')
    p.add_argument('--no-halo-utils', action='store_true')
    args = p.parse_args()

    global HALO_UTILS_AVAILABLE
    if args.no_halo_utils:
        HALO_UTILS_AVAILABLE = False

    runs = args.runs or [args.run]
    print(f'Runs: {runs}  res={args.res}  smooth={args.dt_smooth} Gyr')
    plot_sfr(runs, dt_smooth_Gyr=args.dt_smooth, res=args.res)

if __name__ == '__main__':
    main()
