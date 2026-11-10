#!/usr/bin/env python3
"""Where do CosmicGrain grains in each size population live?

For every PartType6 particle, find its nearest gas particle (as
update_dust_dynamics() does) and evaluate the same environment gates that
dust.cc applies, so each size population can be attributed to the processes
it is, or is not, eligible for:

  associated   dist <= clamp(Hsml, 2, 10) code units (growth/coag/shatter only run then)
  coag-eligible  associated, n_eff >= DustCollisionDensityThresh,
                 SF gas or T <= 3000 K, a < DustMaxGrainSize
  shat-eligible  associated, n_eff <= DustCollisionDensityThresh,
                 0.8 a > DustMinGrainSize, c_s >= (1 + CF) km/s
  sputtering     T > DustThermalSputteringTemp

n_eff = n_H x dust_clumping_factor(n_H, SF), mirroring dust.cc.  Thresholds
are read from the snapshot's own Parameters block.

The snapshot is one instant: it shows where grains are now, not their
history.  Run from ~/gadget4.
"""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path('scripts').resolve()))
from cosmicgrain_hd23_gsd_overlay import read_group_center, snapshot_files

PROTONMASS = 1.67262178e-24
BOLTZMANN = 1.38065e-16
X_H = 0.76
GAMMA = 5.0 / 3.0
SFR_EPS = 1e-14


def read_particles(run, snap):
    files = snapshot_files(run, snap)
    gas = {k: [] for k in ('pos', 'rho', 'u', 'ne', 'sfr', 'hsml')}
    dust = {k: [] for k in ('pos', 'a', 'm', 'cf', 'src', 'tform')}
    with h5py.File(files[0], 'r') as f:
        header = dict(f['Header'].attrs)
        params = dict(f['Parameters'].attrs)
    for fn in files:
        with h5py.File(fn, 'r') as f:
            if 'PartType0' in f:
                g = f['PartType0']
                gas['pos'].append(g['Coordinates'][:])
                gas['rho'].append(g['Density'][:])
                gas['u'].append(g['InternalEnergy'][:])
                gas['ne'].append(g['ElectronAbundance'][:])
                gas['sfr'].append(g['StarFormationRate'][:])
                gas['hsml'].append(g['SmoothingLength'][:])
            if 'PartType6' in f:
                d = f['PartType6']
                dust['pos'].append(d['Coordinates'][:])
                dust['a'].append(d['GrainRadius'][:])
                dust['m'].append(d['Masses'][:])
                dust['cf'].append(d['CarbonMassFraction'][:])
                dust['src'].append(d['DustSource'][:])
                dust['tform'].append(d['DustFormationTime'][:])
    gas = {k: np.concatenate(v).astype(float) for k, v in gas.items()}
    dust = {k: np.concatenate(v) for k, v in dust.items()}
    for k in ('a', 'm', 'cf', 'tform'):
        dust[k] = dust[k].astype(float)
    return header, params, gas, dust


def clumping_factor(n_h, sf, params):
    if not int(params.get('DustEnableClumping', 1)):
        return np.ones_like(n_h)
    n_sf = float(params['CritPhysDensity'])
    c = np.ones_like(n_h)
    c[n_h > 0.05 * n_sf] = 1.5
    c[n_h > 0.2 * n_sf] = 3.0
    c[n_h > 0.5 * n_sf] = 10.0
    c[sf] = 30.0
    return c


def environment(header, params, gas, dust):
    a_scale = float(header['Time'])
    h = float(params['HubbleParam'])
    box = float(header['BoxSize'])
    unit_rho = float(params['UnitMass_in_g']) / float(params['UnitLength_in_cm']) ** 3
    unit_v2 = float(params['UnitVelocity_in_cm_per_s']) ** 2

    tree = cKDTree(np.mod(gas['pos'], box), boxsize=box)
    dist, idx = tree.query(np.mod(dust['pos'], box), k=1)

    rho_cgs = gas['rho'][idx] * a_scale ** -3 * unit_rho * h * h
    n_h = X_H * rho_cgs / PROTONMASS
    y = (1.0 - X_H) / (4.0 * X_H)
    mu = (1.0 + 4.0 * y) / (1.0 + y + gas['ne'][idx])
    temp = (GAMMA - 1.0) * gas['u'][idx] * unit_v2 / BOLTZMANN * PROTONMASS * mu
    sf = gas['sfr'][idx] > SFR_EPS
    n_eff = n_h * clumping_factor(n_h, sf, params)

    assoc_r = np.clip(gas['hsml'][idx], 2.0, 10.0)
    associated = (dist <= 10.0) & (dist <= assoc_r)

    thresh = float(params['DustCollisionDensityThresh'])
    a_min = float(params['DustMinGrainSize'])
    a_max = float(params['DustMaxGrainSize'])
    cs_kms = np.sqrt(BOLTZMANN * temp / (0.6 * PROTONMASS)) / 1e5

    coag = associated & (n_eff >= thresh) & (sf | (temp <= 3000.0)) & \
        (dust['a'] < a_max)
    shat = associated & (n_eff <= thresh) & (0.8 * dust['a'] > a_min) & \
        (cs_kms >= 1.0 + dust['cf'])
    sputter = temp > float(params['DustThermalSputteringTemp'])

    return dict(n_h=n_h, n_eff=n_eff, temp=temp, sf=sf, associated=associated,
                coag=coag, shat=shat, sputter=sputter)


def populations(a, params):
    a_min = float(params['DustMinGrainSize'])
    a_max = float(params['DustMaxGrainSize'])
    tol = 1e-3
    floor = a <= a_min * (1 + tol)
    spike10 = np.abs(a - 10.0) < 10.0 * tol
    spike100 = np.abs(a - 100.0) < 100.0 * tol
    ceiling = a >= a_max * (1 - tol)
    return [
        (f'floor (a = {a_min:g} nm)', floor),
        ('small (floor..10 nm)', ~floor & (a < 10.0) & ~spike10),
        ('spike at 10 nm (SNII spawn)', spike10),
        ('10-100 nm', (a > 10.0) & (a < 100.0) & ~spike10 & ~spike100),
        ('spike at 100 nm (AGB spawn)', spike100),
        ('100 nm .. ceiling', (a > 100.0) & ~spike100 & ~ceiling),
        (f'ceiling (a = {a_max:g} nm)', ceiling),
    ]


def summarize(title, sel_all, dust, env, params):
    m = dust['m']
    total = m[sel_all].sum()
    print(f'\n=== {title}: {sel_all.sum():,} particles ===')
    print(f'{"population":30s} {"N":>9s} {"mass%":>6s} | {"med n_H":>8s} '
          f'{"med T":>8s} {"SF%":>5s} | {"assoc%":>6s} {"coag%":>6s} '
          f'{"shat%":>6s} {"neither%":>8s} {"sput%":>6s} | {"SNII%":>5s} '
          f'{"AGB%":>5s} {"med a_form":>10s}')
    for name, pop in populations(dust['a'], params):
        s = pop & sel_all
        n = s.sum()
        if n == 0:
            print(f'{name:30s} {0:9d}')
            continue
        w = m[s]
        frac = lambda x: 100.0 * w[x[s]].sum() / w.sum()
        neither = env['associated'] & ~env['coag'] & ~env['shat']
        src = dust['src'][s]
        print(f'{name:30s} {n:9,d} {100 * w.sum() / total:6.2f} | '
              f'{np.median(env["n_h"][s]):8.3g} {np.median(env["temp"][s]):8.3g} '
              f'{frac(env["sf"]):5.1f} | {frac(env["associated"]):6.1f} '
              f'{frac(env["coag"]):6.1f} {frac(env["shat"]):6.1f} '
              f'{frac(neither):8.1f} {frac(env["sputter"]):6.1f} | '
              f'{100 * np.mean(src == 0):5.1f} {100 * np.mean(src == 1):5.1f} '
              f'{np.median(dust["tform"][s]):10.3f}')
    print('Gate columns are mass-weighted; SNII%/AGB% are by particle count; '
          'a_form is the scale factor at dust formation.')


def plot(path, title, sel_all, dust, env, params):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    style = Path('scripts/cosmicgrain.mplstyle')
    if style.exists():
        plt.style.use(str(style))

    pops = populations(dust['a'], params)
    fig, axs = plt.subplots(2, 4, figsize=(17, 8), sharex=True, sharey=True,
                            constrained_layout=True)
    xe = np.linspace(-6, 4, 81)
    ye = np.linspace(1, 8, 71)
    thresh = float(params['DustCollisionDensityThresh'])
    all_dust = np.ones(len(dust['a']), dtype=bool)
    for ax, (name, pop) in zip(axs.flat, pops + [('all dust', all_dust)]):
        s = pop & sel_all
        if s.sum():
            hist, _, _ = np.histogram2d(np.log10(env['n_eff'][s]),
                                        np.log10(env['temp'][s]), [xe, ye],
                                        weights=dust['m'][s])
            ax.pcolormesh(xe, ye, np.ma.masked_equal(hist.T, 0),
                          norm=matplotlib.colors.LogNorm(), cmap='viridis')
        ax.axvline(np.log10(thresh), color='tab:red', lw=1.2, ls='--')
        ax.axhline(np.log10(3000.0), color='tab:orange', lw=1.0, ls=':')
        ax.set_title(f'{name}  (N={s.sum():,})', fontsize=10)
    for ax in axs[-1]:
        ax.set_xlabel(r'log$_{10}$ $n_{\rm eff}$ [cm$^{-3}$] (nearest gas)')
    for ax in axs[:, 0]:
        ax.set_ylabel(r'log$_{10}$ T [K]')
    fig.suptitle(f'{title}: dust mass in nearest-gas n_eff-T plane '
                 f'(red: coag/shatter split at {thresh:g} cm$^{{-3}}$; '
                 'orange: 3000 K coag limit)')
    fig.savefig(path, dpi=130)
    print(f'Wrote {path}')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', default='halo441_output_1024_coag6_shat2')
    p.add_argument('--snap', type=int, default=29)
    p.add_argument('--group-index', type=int, default=1)
    p.add_argument('--aperture', type=float, default=30.0, help='pkpc')
    p.add_argument('--out', default='analysis_products/size_pileup_diagnostics')
    args = p.parse_args()

    header, params, gas, dust = read_particles(args.run, args.snap)
    env = environment(header, params, gas, dust)
    print(f'Run {args.run} snap {args.snap:03d}  z={float(header["Redshift"]):.3f}  '
          f'thresh={float(params["DustCollisionDensityThresh"]):g} cm^-3  '
          f'coag cal={float(params["DustCoagulationCalibration"]):g}  '
          f'shat cal={float(params["DustShatteringCalibration"]):g}')

    a_scale = float(header['Time'])
    h = float(params['HubbleParam'])
    box = float(header['BoxSize'])
    center = read_group_center(args.run, args.snap, args.group_index)
    delta = dust['pos'] - center
    delta -= box * np.rint(delta / box)
    central = np.linalg.norm(delta, axis=1) * a_scale / h < args.aperture
    everything = np.ones(len(dust['a']), dtype=bool)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for title, sel, tag in [('Entire simulation', everything, 'all'),
                            (f'Central galaxy (r < {args.aperture:g} pkpc)',
                             central, 'central')]:
        summarize(title, sel, dust, env, params)
        plot(f'{out}_{Path(args.run).name}_{args.snap:03d}_{tag}.png',
             title, sel, dust, env, params)


if __name__ == '__main__':
    main()
