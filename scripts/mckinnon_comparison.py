"""
mckinnon_comparison.py
======================
Reproduces McKinnon+2016 Figures 17 and 18 for the CosmicGrain simulation
ladder, overlaying observed reference data.

Figure 17 — Dust surface density Σ_dust vs galactocentric radius (face-on
             projection along disc angular momentum axis), out to R200.
             Compared with:
               • Draine+2014: M31 azimuthally-averaged Σ_dust
               • Scaled M31 (×2): proxy for a higher MW-like dust mass
               • Ménard+2010: Σ_dust ∝ r^{-0.8} (SDSS statistical detection)

Figure 18 — Enclosed dust mass M(<r)/M(<25 kpc) vs r out to 25 kpc.
             Compared with:
               • Draine+2014: M31 enclosed mass profile

Usage:
    python mckinnon_comparison.py --res 1024
    python mckinnon_comparison.py --res 1024 --runs S4 S8 S10
    python mckinnon_comparison.py --res 1024 --no-rotate  # skip disc alignment

Assumptions:
    Same directory layout as compare_grid_dust.py.
    Requires h5py and numpy.
"""

import os
import re
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

# ─────────────────────────────────────────────────────────────────────────────
# Run styling (matches compare_grid_dust.py exactly)
# ─────────────────────────────────────────────────────────────────────────────
RUN_CONFIGS = {
    'S0':  {'label': 'S0: Creation only',           'color': '#888888'},
    'S1':  {'label': 'S1: + Cooling',               'color': '#1f77b4'},
    'S2':  {'label': 'S2: + Drag',                  'color': '#ff7f0e'},
    'S3':  {'label': 'S3: + Astration',             'color': '#2ca02c'},
    'S4':  {'label': 'S4: + Thermal sputtering',    'color': '#d62728'},
    'S5':  {'label': 'S5: + Grain growth',          'color': '#9467bd'},
    'S6':  {'label': 'S6: + Clumping factor',       'color': '#8c564b'},
    'S7':  {'label': 'S7: + SN shock destruction',  'color': '#e377c2'},
    'S8':  {'label': 'S8: + Coagulation',           'color': '#17becf'},
    'S9':  {'label': 'S9: + Shattering',            'color': '#bcbd22'},
    'S10': {'label': 'S10: + Rad. pressure (full)', 'color': '#000000'},
}

FIGDIR     = 'dust_figures'
RESOLUTION = 512
os.makedirs(FIGDIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Reference data
# ─────────────────────────────────────────────────────────────────────────────

def draine2014_m31_surface_density():
    """
    Approximate azimuthally-averaged Σ_dust (M_sun/pc^2) vs galactocentric
    radius (kpc) for M31, digitised from Draine+2014 (ApJ 780, 172).

    Key features from the paper:
      - Inner ring at R=5.6 kpc
      - Maximum at R=11.2 kpc (~0.1 M_sun/pc^2)
      - Outer ring at R≈15.1 kpc
      - Profile extends to ~25 kpc
    """
    r = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 5.6, 6.5, 7.5, 8.5,
                  9.5, 10.5, 11.2, 12.0, 13.0, 14.0, 15.1, 16.5, 18.0,
                  20.0, 22.0, 24.0, 25.0])
    sigma = np.array([0.035, 0.030, 0.025, 0.022, 0.028, 0.045, 0.060,
                      0.055, 0.065, 0.080, 0.090, 0.095, 0.100, 0.085,
                      0.075, 0.080, 0.072, 0.040, 0.022, 0.010, 0.005,
                      0.002, 0.001])
    return r, sigma


def menard2010_power_law(r_kpc, norm_r=25.0, norm_sigma=2e-3):
    """
    Ménard+2010 (MNRAS 405, 1025): Σ_dust ∝ r^{-0.8} detected statistically
    in SDSS quasar reddening out to ~1 Mpc.  McKinnon normalises the amplitude
    to align with simulated data from 25 kpc outward.
    """
    return norm_sigma * (r_kpc / norm_r) ** (-0.8)


def draine2014_enclosed_fraction():
    """
    Enclosed dust mass fraction M(<r)/M(<25 kpc) for M31 from Draine+2014.
    Total dust mass within 25 kpc ≈ 5.4×10^7 M_sun.
    Approximated by integrating the surface density profile above.
    """
    r_pts, sig_pts = draine2014_m31_surface_density()
    r_fine = np.linspace(0, 25, 1000)
    sig_fine = np.interp(r_fine, r_pts, sig_pts, left=0.0, right=0.0)
    # dM = 2π r Σ dr  (in M_sun, with r in kpc, Σ in M_sun/pc^2)
    # 1 kpc = 1e3 pc → factor 1e6
    dr = r_fine[1] - r_fine[0]
    dM = 2.0 * np.pi * r_fine * sig_fine * dr * 1e6
    M_enc = np.cumsum(dM)
    M_enc /= M_enc[-1]  # normalise to total within 25 kpc
    return r_fine, M_enc


# ─────────────────────────────────────────────────────────────────────────────
# Snapshot utilities
# ─────────────────────────────────────────────────────────────────────────────

def find_snapshots(run):
    output_dir = f'{run}_output_{RESOLUTION}'
    if not os.path.isdir(output_dir):
        return []
    seen, bases = set(), []
    for snapdir in sorted(glob.glob(os.path.join(output_dir, 'snapdir_*'))):
        for f in sorted(glob.glob(os.path.join(snapdir, 'snapshot_*.0.hdf5'))):
            base = re.sub(r'\.0\.hdf5$', '', f)
            if base not in seen:
                seen.add(base); bases.append(base)
        for f in sorted(glob.glob(os.path.join(snapdir, 'snapshot_*.hdf5'))):
            if '.0.hdf5' in f: continue
            base = re.sub(r'\.hdf5$', '', f)
            if base not in seen:
                seen.add(base); bases.append(base)
    return sorted(bases)


def snap_redshift(snap_base):
    import h5py
    for suffix in ['.hdf5', '.0.hdf5']:
        f = snap_base + suffix
        if os.path.exists(f):
            try:
                with h5py.File(f, 'r') as hf:
                    z = hf['Header'].attrs.get('Redshift', None)
                    if z is not None: return float(z)
            except Exception:
                pass
    return None


def find_snap_near_z(snap_bases, target_z):
    best, best_dz = None, 1e30
    for sb in snap_bases:
        z = snap_redshift(sb)
        if z is not None and abs(z - target_z) < best_dz:
            best_dz = abs(z - target_z)
            best = sb
    return best, best_dz


def read_header(snap_base):
    import h5py
    for suffix in ['.0.hdf5', '.hdf5']:
        f = snap_base + suffix
        if os.path.exists(f):
            try:
                with h5py.File(f, 'r') as hf:
                    attrs = hf['Header'].attrs
                    return dict(
                        h      = float(attrs.get('HubbleParam', 0.7)),
                        a      = float(attrs.get('Time', 1.0)),
                        um_cgs = float(attrs.get('UnitMass_in_g',    1.989e43)),
                        ul_cm  = float(attrs.get('UnitLength_in_cm', 3.085678e21)),
                    )
            except Exception:
                pass
    return dict(h=0.7, a=1.0, um_cgs=1.989e43, ul_cm=3.085678e21)


def subfiles(snap_base):
    files = sorted(glob.glob(snap_base + '.*.hdf5'))
    if not files:
        single = snap_base + '.hdf5'
        files = [single] if os.path.exists(single) else []
    return files


def get_halo_center_r200(run, snap_base):
    """Read GroupPos[0] and Group_R_Mean200[0] from SubFind catalog."""
    import h5py
    m = re.search(r'snapshot_(\d+)$', snap_base)
    if not m: return None, None
    snap_num   = m.group(1)
    groups_dir = os.path.join(f'{run}_output_{RESOLUTION}', f'groups_{snap_num}')
    cats = sorted(glob.glob(os.path.join(groups_dir, f'fof_subhalo_tab_{snap_num}.*.hdf5')))
    if not cats: return None, None
    try:
        with h5py.File(cats[0], 'r') as hf:
            if 'Group' not in hf: return None, None
            grp = hf['Group']
            if 'GroupPos' not in grp or grp['GroupPos'].shape[0] == 0:
                return None, None
            ctr  = grp['GroupPos'][0].astype(float)
            r200 = float(grp['Group_R_Mean200'][0]) if 'Group_R_Mean200' in grp else None
    except Exception as e:
        print(f'  [{run}] catalog error: {e}')
        return None, None
    if r200 is None:
        return None, None
    return ctr, r200


# ─────────────────────────────────────────────────────────────────────────────
# Disc orientation — angular momentum of stars within inner_r_kpc
# ─────────────────────────────────────────────────────────────────────────────

def compute_disc_rotation_matrix(snap_base, ctr, hdr, inner_r_kpc=20.0):
    """
    Compute a rotation matrix that maps z → disc angular momentum axis,
    using PartType4 (stars) within inner_r_kpc (comoving kpc/h) of ctr.

    Returns 3×3 rotation matrix R such that  pos_rot = R @ pos.
    Returns identity if insufficient stars are found.
    """
    import h5py
    pos_list, vel_list, mass_list = [], [], []
    for fname in subfiles(snap_base):
        try:
            with h5py.File(fname, 'r') as hf:
                if 'PartType4' not in hf: continue
                pt   = hf['PartType4']
                pos  = pt['Coordinates'][:]
                vel  = pt['Velocities'][:]
                mass = pt['Masses'][:]
                r    = np.linalg.norm(pos - ctr, axis=1)
                mask = r < inner_r_kpc
                if mask.sum() == 0: continue
                pos_list.append(pos[mask])
                vel_list.append(vel[mask])
                mass_list.append(mass[mask])
        except Exception:
            pass

    if not pos_list:
        print('  [disc orientation] no stars found — using no rotation')
        return np.eye(3)

    pos_all  = np.concatenate(pos_list)  - ctr
    vel_all  = np.concatenate(vel_list)
    mass_all = np.concatenate(mass_list)

    # Mass-weighted mean velocity (bulk motion of disc)
    v_bulk = np.average(vel_all, weights=mass_all, axis=0)
    vel_all -= v_bulk

    # Angular momentum L = sum m (r × v)
    L = np.sum(mass_all[:, None] * np.cross(pos_all, vel_all), axis=0)
    L_norm = L / np.linalg.norm(L)

    # Build rotation matrix: z_hat → L_norm
    z_hat = np.array([0.0, 0.0, 1.0])
    v_rot  = np.cross(z_hat, L_norm)
    s      = np.linalg.norm(v_rot)
    c      = np.dot(z_hat, L_norm)

    if s < 1e-10:   # already aligned or anti-aligned
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])

    Kmat = np.array([[ 0,       -v_rot[2],  v_rot[1]],
                     [ v_rot[2],  0,        -v_rot[0]],
                     [-v_rot[1],  v_rot[0],  0       ]])
    R = np.eye(3) + Kmat + Kmat @ Kmat * (1.0 - c) / (s * s)
    print(f'  [disc orientation] L_hat = ({L_norm[0]:.3f},{L_norm[1]:.3f},{L_norm[2]:.3f})')
    return R


# ─────────────────────────────────────────────────────────────────────────────
# Particle loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_dust(snap_base, ctr, rmax):
    """Load PartType6 within rmax (comoving kpc/h) of ctr."""
    import h5py
    pos_list, mass_list = [], []
    for fname in subfiles(snap_base):
        try:
            with h5py.File(fname, 'r') as hf:
                if 'PartType6' not in hf: continue
                pt   = hf['PartType6']
                pos  = pt['Coordinates'][:]
                mass = pt['Masses'][:]
                r    = np.linalg.norm(pos - ctr, axis=1)
                mask = r < rmax
                if mask.sum() == 0: continue
                pos_list.append(pos[mask])
                mass_list.append(mass[mask])
        except Exception as e:
            print(f'  load_dust: {e}')
    if not pos_list: return None, None
    return np.concatenate(pos_list), np.concatenate(mass_list)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 17 — dust surface density
# ─────────────────────────────────────────────────────────────────────────────

def compute_sigma_dust(pos, mass, ctr, R_matrix, r_bins_kpc, to_pkpc, disc_half_height_kpc=5.0):
    """
    Project dust onto the face-on disc plane and compute Σ_dust in radial annuli.

    Parameters
    ----------
    pos, mass : particle arrays (comoving kpc/h)
    ctr       : halo centre (comoving kpc/h)
    R_matrix  : 3×3 rotation to disc frame
    r_bins_kpc: bin edges in physical kpc
    to_pkpc   : conversion factor comoving kpc/h → physical kpc
    disc_half_height_kpc : half-thickness cut above/below disc plane (physical kpc)

    Returns
    -------
    r_centres : physical kpc
    sigma     : M_sun / pc^2
    """
    # Shift to halo centre, convert to physical kpc
    dp = (pos - ctr) * to_pkpc      # physical kpc

    # Rotate to disc frame
    dp_rot = (R_matrix @ dp.T).T    # still physical kpc

    # Disc plane cut: |z| < disc_half_height_kpc
    mask = np.abs(dp_rot[:, 2]) < disc_half_height_kpc
    dp_disc = dp_rot[mask]
    m_disc  = mass[mask]

    # Projected radius in disc plane
    r_proj = np.sqrt(dp_disc[:, 0]**2 + dp_disc[:, 1]**2)

    # Bin masses
    mass_in_bin, _ = np.histogram(r_proj, bins=r_bins_kpc, weights=m_disc)

    # Annulus area in pc^2 (r_bins_kpc in kpc, 1 kpc = 1e3 pc)
    r_lo = r_bins_kpc[:-1] * 1e3   # pc
    r_hi = r_bins_kpc[1:]  * 1e3
    area = np.pi * (r_hi**2 - r_lo**2)  # pc^2

    # Surface density in M_sun/pc^2 (mass in code units = 1e10 M_sun)
    sigma = mass_in_bin * 1e10 / area
    r_cen = 0.5 * (r_bins_kpc[:-1] + r_bins_kpc[1:])
    return r_cen, sigma


def plot_sigma_dust(runs, use_rotation=True):
    """McKinnon+2016 Figure 17 analogue."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Radial bins: 0.5 kpc width from 0 to 250 kpc (out to ~R200)
    r_bins = np.concatenate([
        np.arange(0, 10,  0.5),
        np.arange(10, 50, 1.0),
        np.arange(50, 260, 5.0),
    ])

    for run in runs:
        cfg   = RUN_CONFIGS.get(run, {})
        color = cfg.get('color', 'black')
        label = cfg.get('label', run)

        snaps = find_snapshots(run)
        if not snaps:
            print(f'  [{run}] no snapshots')
            continue
        snap_base, dz = find_snap_near_z(snaps, 0.0)
        if dz > 0.2:
            print(f'  [{run}] no z~0 snapshot (dz={dz:.2f})')
            continue

        hdr       = read_header(snap_base)
        to_pkpc   = hdr['a'] / hdr['h']
        ctr, r200 = get_halo_center_r200(run, snap_base)
        if ctr is None:
            print(f'  [{run}] no halo center')
            continue

        print(f'  [{run}] R200={r200*to_pkpc:.1f} pkpc, loading dust...')
        pos, mass = load_dust(snap_base, ctr, r200)
        if pos is None:
            print(f'  [{run}] no dust within R200')
            continue

        if use_rotation:
            R_mat = compute_disc_rotation_matrix(snap_base, ctr, hdr)
        else:
            R_mat = np.eye(3)

        r_cen, sigma = compute_sigma_dust(pos, mass, ctr, R_mat, r_bins, to_pkpc)

        # Plot only where sigma > 0
        good = sigma > 0
        ax.plot(r_cen[good], sigma[good], color=color, lw=1.8, label=label, alpha=0.9)

    # ── Reference data ────────────────────────────────────────────────────────
    # Draine+2014 M31
    r_m31, sig_m31 = draine2014_m31_surface_density()
    ax.scatter(r_m31, sig_m31, color='gray', marker='s', s=25, zorder=5,
               label='M31 (Draine+2014)', edgecolors='none')
    ax.scatter(r_m31, sig_m31 * 2, color='gray', marker='^', s=25, zorder=5,
               label='M31 ×2 (Draine+2014)', edgecolors='none', alpha=0.6)

    # Ménard+2010: normalise amplitude to match simulations at ~25-100 kpc
    # Following McKinnon: align at r~50 kpc
    r_men = np.logspace(np.log10(20), np.log10(300), 200)
    # Typical CosmicGrain sigma at 50 kpc ~ few×10^{-4}; McKinnon aligns at ~2e-4
    sigma_men = menard2010_power_law(r_men, norm_r=50.0, norm_sigma=5e-4)
    ax.plot(r_men, sigma_men, 'k--', lw=1.5, label=r'$\Sigma\propto r^{-0.8}$ (Ménard+2010)')
    # ─────────────────────────────────────────────────────────────────────────

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('$r$ (physical kpc)', fontsize=12)
    ax.set_ylabel(r'$\Sigma_{\rm dust}$ ($M_\odot\,{\rm pc}^{-2}$)', fontsize=12)
    title = 'Dust Surface Density at $z=0$'
    if use_rotation:
        title += ' (face-on)'
    ax.set_title(title, fontsize=12)
    ax.set_xlim(0.5, 300)
    ax.set_ylim(1e-6, 1.0)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=7, loc='upper right', ncol=2)
    plt.tight_layout()
    out = os.path.join(FIGDIR, 'mckinnon_fig17_sigma_dust.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out}')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 18 — enclosed dust mass fraction
# ─────────────────────────────────────────────────────────────────────────────

def compute_enclosed_fraction(pos, mass, ctr, R_matrix, to_pkpc, r_max_kpc=25.0, nr=200):
    """
    Compute M(<r)/M(<r_max_kpc) vs r in physical kpc.

    Uses 3D radial distance (not projected), matching McKinnon who uses
    spherical apertures for the enclosed mass plot.
    """
    dp = (pos - ctr) * to_pkpc          # physical kpc
    r3d = np.sqrt(np.sum(dp**2, axis=1))

    r_arr = np.linspace(0, r_max_kpc, nr)
    M_enc = np.array([mass[r3d < r].sum() for r in r_arr])

    M_norm = M_enc[-1]
    if M_norm <= 0:
        return r_arr, np.zeros_like(r_arr)
    return r_arr, M_enc / M_norm


def plot_enclosed_mass(runs, use_rotation=True):
    """McKinnon+2016 Figure 18 analogue."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for run in runs:
        cfg   = RUN_CONFIGS.get(run, {})
        color = cfg.get('color', 'black')
        label = cfg.get('label', run)

        snaps = find_snapshots(run)
        if not snaps: continue
        snap_base, dz = find_snap_near_z(snaps, 0.0)
        if dz > 0.2: continue

        hdr       = read_header(snap_base)
        to_pkpc   = hdr['a'] / hdr['h']
        ctr, r200 = get_halo_center_r200(run, snap_base)
        if ctr is None: continue

        pos, mass = load_dust(snap_base, ctr, r200)
        if pos is None: continue

        R_mat = compute_disc_rotation_matrix(snap_base, ctr, hdr) \
                if use_rotation else np.eye(3)

        r_arr, frac = compute_enclosed_fraction(pos, mass, ctr, R_mat, to_pkpc)
        ax.plot(r_arr, frac, color=color, lw=1.8, label=label, alpha=0.9)

    # ── Draine+2014 M31 reference ─────────────────────────────────────────────
    r_d14, frac_d14 = draine2014_enclosed_fraction()
    ax.plot(r_d14, frac_d14, color='gray', ls=':', lw=2.5, zorder=5,
            label='M31 (Draine+2014)')
    # ─────────────────────────────────────────────────────────────────────────

    ax.set_xlabel('$r$ (physical kpc)', fontsize=12)
    ax.set_ylabel(r"$M_{\rm dust}(r' < r)\,/\,M_{\rm dust}(r' < 25\,{\rm kpc})$",
                  fontsize=12)
    ax.set_title('Enclosed Dust Mass at $z=0$', fontsize=12)
    ax.set_xlim(0, 25)
    ax.set_yscale('log')
    ax.set_ylim(1e-2, 1.5)
    ax.yaxis.set_major_formatter(matplotlib.ticker.LogFormatter(labelOnlyBase=False))
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=7, loc='lower right', ncol=2)
    plt.tight_layout()
    out = os.path.join(FIGDIR, 'mckinnon_fig18_enclosed_mass.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out}')


# ─────────────────────────────────────────────────────────────────────────────
# Bonus: both figures side by side on one canvas (for a paper figure)
# ─────────────────────────────────────────────────────────────────────────────

def plot_combined(runs, use_rotation=True):
    """Single figure with both panels, formatted for a paper."""
    fig, (ax17, ax18) = plt.subplots(1, 2, figsize=(14, 6))
    r_bins = np.concatenate([
        np.arange(0, 10,  0.5),
        np.arange(10, 50, 1.0),
        np.arange(50, 260, 5.0),
    ])

    for run in runs:
        cfg       = RUN_CONFIGS.get(run, {})
        color     = cfg.get('color', 'black')
        label     = cfg.get('label', run)
        snaps     = find_snapshots(run)
        if not snaps: continue
        snap_base, dz = find_snap_near_z(snaps, 0.0)
        if dz > 0.2: continue
        hdr       = read_header(snap_base)
        to_pkpc   = hdr['a'] / hdr['h']
        ctr, r200 = get_halo_center_r200(run, snap_base)
        if ctr is None: continue
        print(f'  [{run}] loading...')
        pos, mass = load_dust(snap_base, ctr, r200)
        if pos is None: continue
        R_mat = compute_disc_rotation_matrix(snap_base, ctr, hdr) \
                if use_rotation else np.eye(3)

        r_cen, sigma = compute_sigma_dust(pos, mass, ctr, R_mat, r_bins, to_pkpc)
        good = sigma > 0
        ax17.plot(r_cen[good], sigma[good], color=color, lw=1.8, label=label, alpha=0.9)

        r_arr, frac = compute_enclosed_fraction(pos, mass, ctr, R_mat, to_pkpc)
        ax18.plot(r_arr, frac, color=color, lw=1.8, label=label, alpha=0.9)

    # Fig 17 references
    r_m31, sig_m31 = draine2014_m31_surface_density()
    ax17.scatter(r_m31, sig_m31,   color='gray', marker='s', s=30, zorder=5,
                 label='M31 (Draine+2014)', edgecolors='none')
    ax17.scatter(r_m31, sig_m31*2, color='gray', marker='^', s=30, zorder=5,
                 label='M31 ×2', edgecolors='none', alpha=0.6)
    r_men = np.logspace(np.log10(20), np.log10(300), 200)
    # Ménard: black dashed in left panel only — no equivalent in right panel
    ax17.plot(r_men, menard2010_power_law(r_men, 50.0, 5e-4),
              color='black', ls='--', lw=1.5,
              label=r'$\Sigma\propto r^{-0.8}$ (Ménard+2010)')
    ax17.set_xscale('log'); ax17.set_yscale('log')
    ax17.set_xlabel('$r$ (physical kpc)', fontsize=12)
    ax17.set_ylabel(r'$\Sigma_{\rm dust}$ ($M_\odot\,{\rm pc}^{-2}$)', fontsize=12)
    title17 = 'Dust Surface Density at $z=0$'
    if use_rotation: title17 += ' (face-on)'
    ax17.set_title(title17, fontsize=11)
    ax17.set_xlim(0.5, 300); ax17.set_ylim(1e-6, 1.0)
    ax17.grid(True, alpha=0.3, which='both')
    # Full legend lives only in the left panel
    ax17.legend(fontsize=7, loc='upper right', ncol=1)

    # Fig 18 references
    # Use gray dotted so it is visually distinct from the black dashed Ménard line
    # in the left panel — readers flipping between panels won't confuse them.
    r_d14, frac_d14 = draine2014_enclosed_fraction()
    ax18.plot(r_d14, frac_d14, color='gray', ls=':', lw=2.5, zorder=5,
              label='M31 (Draine+2014)')
    ax18.set_xlabel('$r$ (physical kpc)', fontsize=12)
    ax18.set_ylabel(r"$M_{\rm dust}(r' < r)\,/\,M_{\rm dust}(r' < 25\,{\rm kpc})$",
                    fontsize=12)
    ax18.set_title('Enclosed Dust Mass at $z=0$', fontsize=11)
    ax18.set_xlim(0, 25)
    ax18.set_yscale('log')
    ax18.set_ylim(1e-2, 1.5)
    ax18.yaxis.set_major_formatter(matplotlib.ticker.LogFormatter(labelOnlyBase=False))
    ax18.grid(True, alpha=0.3, which='both')
    # Right panel: only the Draine reference label, no run legend
    ax18.legend(fontsize=8, loc='lower right')

    plt.tight_layout()
    out = os.path.join(FIGDIR, 'mckinnon_combined.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out}')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='McKinnon+2016 Fig 17/18 comparison')
    parser.add_argument('--runs', nargs='+',
                        default=['S0','S1','S2','S3','S4','S5',
                                 'S6','S7','S8','S9','S10'])
    parser.add_argument('--res', type=int, default=512)
    parser.add_argument('--no-rotate', action='store_true',
                        help='Skip disc alignment (use raw simulation frame)')
    parser.add_argument('--combined-only', action='store_true',
                        help='Only produce the combined 2-panel figure')
    args = parser.parse_args()

    global RESOLUTION
    RESOLUTION = args.res
    use_rotation = not args.no_rotate

    print(f'\nRuns: {args.runs}')
    print(f'Resolution: {RESOLUTION}^3')
    print(f'Disc rotation: {"yes" if use_rotation else "no"}')
    print(f'Output: {FIGDIR}/\n')

    if args.combined_only:
        plot_combined(args.runs, use_rotation)
    else:
        print('=== Figure 17: dust surface density ===')
        plot_sigma_dust(args.runs, use_rotation)
        print('\n=== Figure 18: enclosed mass fraction ===')
        plot_enclosed_mass(args.runs, use_rotation)
        print('\n=== Combined figure ===')
        plot_combined(args.runs, use_rotation)

    print('\nDone.')


if __name__ == '__main__':
    main()
