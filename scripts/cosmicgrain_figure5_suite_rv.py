#!/usr/bin/env python3
"""Figure 5 pilot from Figure 4 per-pixel provenance CSVs.

Manifest CSV columns: halo,m200_msun,r200_pkpc,pixel_csv
Use the same --bins, --plot-radius, --zmax, --min-count and optical tables
when producing each halo's Figure 4 CSV. This script does not run SKIRT.
"""
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REQUIRED = ('x_pkpc', 'y_pkpc', 'projected_radius_pkpc', 'Rv',
            'dust_count', 'dust_mass_code', 'SNII_born_mass_fraction',
            'AGB_born_mass_fraction', 'LRN_born_mass_fraction')


def read_pixels(path):
    with open(path, newline='') as stream:
        reader = csv.DictReader(stream)
        missing = set(REQUIRED) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f'{path}: missing columns {sorted(missing)}')
        rows = list(reader)
    if not rows:
        raise ValueError(f'{path}: empty pixel CSV')
    return {key: np.asarray([float(row[key]) for row in rows]) for key in REQUIRED}


def median_interval(values):
    return np.percentile(values, [16, 50, 84]) if len(values) else [np.nan]*3


def source_delta(data, lo, hi, minimum):
    # Define low/high R(V) separately within each radial annulus and halo.
    band = (data['projected_radius_pkpc'] >= lo) & (data['projected_radius_pkpc'] < hi)
    values = data['Rv'][band]
    agb = data['AGB_born_mass_fraction'][band]
    if len(values) < 3 * minimum:
        return np.nan, int(len(values)), 0, 0
    order = np.argsort(values)
    third = len(order) // 3
    low, high = order[:third], order[-third:]
    return (float(np.median(agb[high]) - np.median(agb[low])),
            len(values), len(low), len(high))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', help='CSV: halo,m200_msun,r200_pkpc,pixel_csv')
    parser.add_argument('--prefix', default='cosmicgrain_figure5_pilot')
    parser.add_argument('--minimum-per-bin', type=int, default=10)
    parser.add_argument('--maximum-radius', type=float, default=20.0,
                        help='Common projected aperture (pkpc), default 20')
    args = parser.parse_args()
    if args.minimum_per_bin < 1 or args.maximum_radius <= 0:
        parser.error('minimum-per-bin and maximum-radius must be positive')
    manifest_path = Path(args.manifest).resolve()
    with manifest_path.open(newline='') as stream:
        entries = list(csv.DictReader(stream))
    needed = {'halo', 'm200_msun', 'r200_pkpc', 'pixel_csv'}
    if not entries or not needed.issubset(entries[0]):
        parser.error('manifest needs halo,m200_msun,r200_pkpc,pixel_csv and at least one row')
    halos = []
    for row in entries:
        path = Path(row['pixel_csv'])
        if not path.is_absolute():
            path = manifest_path.parent / path
        m200, r200 = float(row['m200_msun']), float(row['r200_pkpc'])
        if m200 <= 0 or r200 <= 0:
            raise ValueError(f"{row['halo']}: M200 and R200 must be positive")
        if not path.is_file():
            raise FileNotFoundError(
                f"{row['halo']}: Figure 4 pixel CSV not found: {path}\n"
                'Run cosmicgrain_figure4_rv_grains.py for this halo using '
                '--prefix matching the manifest, or edit pixel_csv. '
                'Relative paths are resolved from the manifest directory.'
            )
        data = read_pixels(path)
        good = np.logical_and.reduce([
            np.isfinite(data[key]) for key in REQUIRED
        ]) & (data['Rv'] > 0) & (data['dust_mass_code'] > 0) & (
            data['projected_radius_pkpc'] < args.maximum_radius)
        data = {key: value[good] for key, value in data.items()}
        if not len(data['Rv']):
            raise ValueError(f"{row['halo']}: no valid pixels within aperture")
        halos.append((row['halo'], m200, r200, data))
    if len({item[0] for item in halos}) != len(halos):
        raise ValueError('Halo labels must be unique')
    halos.sort(key=lambda item: item[1])
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5), constrained_layout=True)
    palette = plt.get_cmap('viridis')(np.linspace(.12, .9, len(halos)))
    summary = []
    radial_rows = []
    radius_edges = np.linspace(0, args.maximum_radius, 5)
    for index, ((name, m200, r200, data), color) in enumerate(zip(halos, palette)):
        rv = data['Rv']
        p16, p50, p84 = median_interval(rv)
        axes[0].errorbar(m200, p50, yerr=[[p50-p16], [p84-p50]],
                         fmt='o', color=color, ms=7, capsize=3)
        axes[0].annotate(name, (m200, p50), xytext=(5, 0),
                         textcoords='offset points', va='center', fontsize=8)
        medians = []
        for rlo, rhi in zip(radius_edges[:-1], radius_edges[1:]):
            selected = (data['projected_radius_pkpc'] >= rlo) & (
                data['projected_radius_pkpc'] < rhi)
            n = int(selected.sum())
            p = median_interval(rv[selected]) if n >= args.minimum_per_bin else [np.nan]*3
            radial_rows.append([name, m200, rlo, rhi, n, *p])
            medians.append(p)
        medians = np.asarray(medians)
        centers = (radius_edges[:-1] + radius_edges[1:]) / 2
        axes[1].plot(centers, medians[:, 1], 'o-', color=color, lw=1.7,
                     ms=4, label=name)
        axes[1].fill_between(centers, medians[:, 0], medians[:, 2],
                             color=color, alpha=.08)
        # Two fixed physical radial bands; missing or sparse bins are explicit.
        deltas = []
        for j, (rlo, rhi) in enumerate(((10, 15), (15, 20))):
            if rhi > args.maximum_radius:
                result = (np.nan, 0, 0, 0)
            else:
                result = source_delta(data, rlo, rhi, args.minimum_per_bin)
            deltas.append(result)
            delta, nband, nlow, nhigh = result
            if np.isfinite(delta):
                marker = 'o' if j == 0 else 's'
                axes[2].scatter(m200, delta, color=color, marker=marker, s=48)
        summary.append([name, m200, r200, len(rv), p16, p50, p84,
                        *(value for result in deltas for value in result)])
        print(f'{name}: N={len(rv)}, median R(V)={p50:.3f}, '
              f'10-15 kpc ΔAGB={deltas[0][0]:.3f} (N={deltas[0][1]}), '
              f'15-20 kpc ΔAGB={deltas[1][0]:.3f} (N={deltas[1][1]})')
    axes[0].set(xscale='log', xlabel=r'$M_{200c}\;(M_\odot)$',
                ylabel=r'Median $R(V)$; bars: 16th–84th percentile',
                title='(a) Extinction across halos')
    axes[1].set(xlabel='Projected radius (pkpc)', ylabel=r'Median $R(V)$',
                xlim=(0, args.maximum_radius), title='(b) Radial profiles')
    axes[1].legend(frameon=False, fontsize=8, ncol=2)
    axes[2].axhline(0, lw=1, color='.5')
    axes[2].set(xscale='log', xlabel=r'$M_{200c}\;(M_\odot)$',
                ylabel=r'$\Delta f_{\rm AGB}$ (high minus low $R(V)$)',
                title='(c) Source mix at fixed radius')
    from matplotlib.lines import Line2D
    axes[2].legend(handles=[
        Line2D([], [], color='.35', marker='o', ls='', label='10–15 pkpc'),
        Line2D([], [], color='.35', marker='s', ls='', label='15–20 pkpc')],
        frameon=False, fontsize=8)
    fig.suptitle('CosmicGrain suite: projected extinction and grain birth source')
    fig.savefig(args.prefix + '.png', dpi=220)
    fig.savefig(args.prefix + '.pdf')
    plt.close(fig)
    with open(args.prefix + '_summary.csv', 'w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['halo', 'm200_msun', 'r200_pkpc', 'N_valid_within_aperture',
                         'Rv_p16', 'Rv_median', 'Rv_p84',
                         'delta_agb_10_15', 'N_10_15', 'N_low_10_15', 'N_high_10_15',
                         'delta_agb_15_20', 'N_15_20', 'N_low_15_20', 'N_high_15_20'])
        writer.writerows(summary)
    with open(args.prefix + '_radial.csv', 'w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['halo', 'm200_msun', 'r_min_pkpc', 'r_max_pkpc',
                         'N_pixels', 'Rv_p16', 'Rv_median', 'Rv_p84'])
        writer.writerows(radial_rows)
    print(f'Saved {args.prefix}.png/.pdf and summary/radial CSVs')


if __name__ == '__main__':
    main()
