#!/usr/bin/env python3
"""CG10 paper Figure 4. Place beside halo441_rv_grain_physics.py.

Requires numpy, scipy, h5py, matplotlib and the original optical tables.
Each projected pixel is one sightline; no SKIRT radiative transfer is used.
"""
import argparse
import csv
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import halo441_rv_grain_physics as core


def apply_cosmicgrain_style(style_file=None):
    """Load the project style from an explicit path or beside this script."""
    import os
    candidates = []
    if style_file:
        candidates.append(os.path.abspath(style_file))
    candidates.extend([
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     'cosmicgrain.mplstyle'),
        os.path.join(os.getcwd(), 'cosmicgrain.mplstyle'),
    ])
    for path in candidates:
        if os.path.isfile(path):
            plt.style.use(path)
            print(f'Using Matplotlib style: {path}')
            return path
    raise FileNotFoundError(
        'cosmicgrain.mplstyle was not found beside this script or in the '
        'current directory; pass --style-file PATH.'
    )


def scatter_with_tertiles(ax, maps, props, valid, groups, colors, key):
    """Draw one equally weighted pixel relation and its tertile summaries."""
    rv = maps['rv'][valid]
    values = props[key][valid]
    ax.scatter(rv, values, s=7, alpha=.16, c='#444956', rasterized=True)
    for group, color, label in zip(groups, colors, ['Low', 'Middle', 'High']):
        x = maps['rv'][group]
        y = props[key][group]
        x16, x50, x84 = np.percentile(x, [16, 50, 84])
        y16, y50, y84 = np.percentile(y, [16, 50, 84])
        ax.errorbar(
            x50, y50,
            xerr=np.array([[x50-x16], [x84-x50]]),
            yerr=np.array([[y50-y16], [y84-y50]]),
            fmt='o', color=color, ms=7, capsize=3, label=label,
        )



def plot_combined_figure(args, maps, props, valid, groups, colors,
                         x, y, mass, radius, carbon_fraction, source,
                         pixel_index, radial_distance, source_fraction_maps):
    """One six-axis, four-part Figure 4 using identical pixel selection."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    fig.suptitle('CG10: grain populations and projected extinction-curve variation',
                 fontsize=16)
    ax = axes[0, 0]
    scatter_with_tertiles(ax, maps, props, valid, groups, colors,
                          'small_fraction')
    ax.set(title='(a) Small-grain fraction', xlabel=r'$R(V)$',
           ylabel=rf'$M(a<{args.small_grain_nm:g}\,\mathrm{{nm}})/M_\mathrm{{dust}}$',
           ylim=(-.04, 1.04))
    ax.legend(title=r'$R(V)$ tertile', frameon=False, fontsize=8)

    ax = axes[0, 1]
    scatter_with_tertiles(ax, maps, props, valid, groups, colors,
                          'carbon_fraction')
    ax.set(title='(b) Carbonaceous fraction', xlabel=r'$R(V)$',
           ylabel=r'$M_{\rm carbon}/M_{\rm dust}$', ylim=(-.04, 1.04))
    ax.legend(title=r'$R(V)$ tertile', frameon=False, fontsize=8)

    # Combine the two well-sampled outer radial bands.  The dedicated radial
    # diagnostic products retain their separation; the paper panel isolates
    # the principal provenance trend without four overlapping ribbons.
    ax = axes[0, 2]
    rmin, rmax = 10, 20
    band = valid & (radial_distance >= rmin) & (radial_distance < rmax)
    rv_band = maps['rv'][band]
    if len(rv_band) < 15:
        ax.text(.5, .5, f'Only {len(rv_band)} eligible pixels',
                transform=ax.transAxes, ha='center', va='center')
    else:
        chunks = np.array_split(np.argsort(rv_band), 5)
        for code, source_label, color in [(1, 'AGB-born', '#d59835'),
                                          (0, 'SNII-born', '#3274a1')]:
            fraction = source_fraction_maps[code][band]
            xb, mid, lo, hi = [], [], [], []
            for inds in chunks:
                if len(inds) == 0:
                    continue
                xb.append(np.median(rv_band[inds]))
                p16, p50, p84 = np.percentile(fraction[inds], [16, 50, 84])
                lo.append(p16)
                mid.append(p50)
                hi.append(p84)
            ax.fill_between(xb, lo, hi, color=color, alpha=.11)
            ax.plot(xb, mid, marker='o', lw=2, ms=5,
                    color=color, label=source_label)
    ax.set(title=f'(c) Stellar birth source, {rmin}–{rmax} pkpc',
           xlabel=r'$R(V)$',
           ylabel='Birth-source dust mass fraction', ylim=(-.04, 1.04))
    ax.legend(frameon=False, fontsize=8)

    edges = np.logspace(np.log10(radius.min()), np.log10(radius.max()), 31)
    size_centers = np.sqrt(edges[:-1] * edges[1:])
    dlog = np.diff(np.log10(edges))
    labels = ['Low', 'Middle', 'High']
    for ax, component, title, letter in [
        (axes[1, 0], carbon_fraction, 'Carbonaceous', 'd'),
        (axes[1, 1], 1-carbon_fraction, 'Silicate', 'e')
    ]:
        for group, color, label in zip(groups, colors, labels):
            chosen = group.ravel()[pixel_index]
            pix, inverse = np.unique(pixel_index[chosen], return_inverse=True)
            denominator = np.bincount(inverse, weights=mass[chosen])
            binid = np.clip(np.searchsorted(edges, radius[chosen], side='right')-1,
                            0, len(size_centers)-1)
            weighted = mass[chosen] * component[chosen] / denominator[inverse]
            curve = np.bincount(binid, weights=weighted,
                                minlength=len(size_centers)) / len(pix) / dlog
            ax.plot(size_centers, curve, color=color, lw=2, label=label)
        ax.set(title=f'({letter}) {title} GSD', xscale='log',
               xlabel='Grain radius (nm)')
        ax.legend(title=r'$R(V)$ tertile', fontsize=8, frameon=False)
    axes[1, 0].set_ylabel('Mean dust mass fraction per dex')
    axes[1, 1].set_ylabel('Mean dust mass fraction per dex')

    ax = axes[1, 2]
    for group, linestyle, rv_label in [(groups[0], '--', 'low'),
                                       (groups[2], '-', 'high')]:
        chosen = group.ravel()[pixel_index]
        pix, inverse = np.unique(pixel_index[chosen], return_inverse=True)
        denominator = np.bincount(inverse, weights=mass[chosen])
        binid = np.clip(np.searchsorted(edges, radius[chosen], side='right')-1,
                        0, len(size_centers)-1)
        for code, color, label in [(0, '#3274a1', 'SNII'),
                                   (1, '#d59835', 'AGB')]:
            weighted = mass[chosen] * (source[chosen] == code) / denominator[inverse]
            curve = np.bincount(binid, weights=weighted,
                                minlength=len(size_centers)) / len(pix) / dlog
            ax.plot(size_centers, curve, color=color, ls=linestyle, lw=2,
                    label=f'{label}; {rv_label} $R(V)$')
    ax.axvline(100, lw=1, ls=':', color='.5')
    ax.set(title='(f) Grain birth source', xscale='log',
           xlabel='Grain radius (nm)',
           ylabel='Mean birth-source mass fraction per dex')
    ax.legend(fontsize=8, frameon=False)
    output = args.prefix + '_combined.png'
    fig.savefig(output, dpi=220)
    fig.savefig(args.prefix + '_combined.pdf')
    plt.close(fig)
    print(f'Saved {output} and {args.prefix}_combined.pdf')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('snapshot', help='Quoted glob for split snapshot files')
    p.add_argument('--center-file', required=True)
    p.add_argument('--rotation-file', required=True)
    p.add_argument('--carbon-table', default='dust_optics/callqcomp.out_D16MGemt')
    p.add_argument('--silicate-table', default='dust_optics/callqcomp.out_DH16asil')
    p.add_argument('--style-file', default=None,
                   help='Path to cosmicgrain.mplstyle (default: beside script or current directory)')
    p.add_argument('--plot-radius', type=float, default=20.)
    p.add_argument('--zmax', type=float, default=20.)
    p.add_argument('--bins', type=int, default=80)
    p.add_argument('--min-count', type=int, default=5)
    p.add_argument('--min-signal-fraction', type=float, default=1e-3)
    p.add_argument('--min-ebv-fraction', type=float, default=1e-3,
                   help='Minimum positive E(B-V) as a fraction of its aperture maximum')
    p.add_argument('--rv-min', type=float, default=1.5)
    p.add_argument('--rv-max', type=float, default=10.0)
    p.add_argument('--small-grain-nm', type=float, default=50.)
    p.add_argument('--prefix', default='CG10_figure4')
    args = p.parse_args()
    if args.bins < 2 or args.min_count < 1 or args.small_grain_nm <= 0:
        p.error('bins >= 2, min-count >= 1, and small-grain-nm > 0 are required')
    if not 0 <= args.min_ebv_fraction < 1:
        p.error('min-ebv-fraction must lie in [0,1)')
    if not 0 < args.rv_min < args.rv_max:
        p.error('require 0 < rv-min < rv-max')
    apply_cosmicgrain_style(args.style_file)
    carbon = core.read_draine_qext_table(args.carbon_table)
    silicate = core.read_draine_qext_table(args.silicate_table)
    dust = core.read_dust_snapshot(args.snapshot)
    # Keep file order identical to read_dust_snapshot; a missing field must
    # fail loudly rather than silently misalign sources and dust particles.
    source_chunks = []
    for filename in core.expand_files(args.snapshot):
        with h5py.File(filename, 'r') as f:
            if 'PartType6' not in f:
                continue
            if 'DustSource' not in f['PartType6']:
                raise KeyError(f'{filename}: PartType6/DustSource missing; cannot classify provenance')
            source_chunks.append(np.asarray(f['PartType6/DustSource'][:], dtype=np.int64))
    source_all = np.concatenate(source_chunks)
    if len(source_all) != len(dust['mass']):
        raise RuntimeError('DustSource length differs from dust arrays')
    center = np.loadtxt(args.center_file).reshape(3)
    rotation = np.loadtxt(args.rotation_file).reshape(3, 3)
    xyz = core.rotate_positions(dust['coords'], center, dust['box'], dust['a'], dust['h'], rotation)
    select = ((np.abs(xyz[:, 0]) < args.plot_radius) &
              (np.abs(xyz[:, 1]) < args.plot_radius) &
              (np.abs(xyz[:, 2]) < args.zmax) &
              np.isfinite(dust['mass']) & (dust['mass'] > 0) &
              np.isfinite(dust['radius_nm']) & (dust['radius_nm'] > 0) &
              np.isfinite(dust['carbon']))
    if not np.any(select):
        raise RuntimeError('No valid dust particles in projection')
    x, y = xyz[select, 0], xyz[select, 1]
    mass, radius = dust['mass'][select], dust['radius_nm'][select]
    fc = np.clip(dust['carbon'][select], 0, 1)
    source = source_all[select]
    wb, wv = core.extinction_weights(mass, radius, fc, carbon, silicate)
    maps = core.build_maps(x, y, mass, wb, wv, args.plot_radius,
                           args.bins, 0., args.min_signal_fraction)
    props = core.build_grain_property_maps(x, y, mass, radius, fc,
                                           args.plot_radius, args.bins, args.small_grain_nm)
    dx = 2.0 * args.plot_radius / args.bins
    axis = -args.plot_radius + (np.arange(args.bins) + 0.5) * dx
    gridx, gridy = np.meshgrid(axis, axis)
    in_aperture = np.hypot(gridx, gridy) < args.plot_radius
    ebv = maps['tauB'] - maps['tauV']
    positive_ebv = in_aperture & np.isfinite(ebv) & (ebv > 0)
    if not np.any(positive_ebv):
        raise RuntimeError('No positive E(B-V) pixels inside the aperture')
    ebv_reference = float(np.nanmax(ebv[positive_ebv]))
    ebv_floor = args.min_ebv_fraction * ebv_reference
    base_quality = (
        in_aperture & np.isfinite(maps['rv'])
        & (maps['count_raw'] >= args.min_count)
        & (ebv >= ebv_floor)
        & np.isfinite(props['small_fraction'])
        & np.isfinite(props['carbon_fraction'])
    )
    below_rv = base_quality & (maps['rv'] < args.rv_min)
    above_rv = base_quality & (maps['rv'] > args.rv_max)
    valid = base_quality & (maps['rv'] >= args.rv_min) & (maps['rv'] <= args.rv_max)
    n = int(valid.sum())
    if n < 9:
        raise RuntimeError(f'Only {n} pixels pass cuts; reduce bins or min-count')
    rv = maps['rv'][valid]
    q1, q2 = np.quantile(rv, [1/3, 2/3])
    if q1 >= q2:
        raise RuntimeError('R(V) tertile boundaries coincide')
    groups = [valid & (maps['rv'] <= q1),
              valid & (maps['rv'] > q1) & (maps['rv'] <= q2),
              valid & (maps['rv'] > q2)]
    names = ['Low', 'Middle', 'High']
    colors = ['#3274a1', '#d59835', '#a03c50']
    print(f'Eligible pixels: {n}; R(V) tertile edges: {q1:.4g}, {q2:.4g}')
    print(f'Physical pixel size: {2.0 * args.plot_radius / args.bins:.4g} pkpc')
    print(f'E(B-V) floor: {ebv_floor:.6e} '
          f'({args.min_ebv_fraction:g} of aperture maximum)')
    print(f'Outside {args.rv_min:g} <= R(V) <= {args.rv_max:g}: '
          f'{np.count_nonzero(below_rv) + np.count_nonzero(above_rv)}/'
          f'{np.count_nonzero(base_quality)} quality pixels')
    print('Each pixel counts once in the distribution comparison.')
    ix, iy, inside = core.assign_pixels(x, y, args.plot_radius, args.bins)
    if not np.all(inside):
        raise RuntimeError('Selected dust particles fell outside the pixel grid')
    pixel_index = iy * args.bins + ix
    edges = np.logspace(np.log10(radius.min()), np.log10(radius.max()), 31)
    centers = np.sqrt(edges[:-1] * edges[1:])
    dlog = np.diff(np.log10(edges))
    fig, axs = plt.subplots(2, 2, figsize=(11.4, 8.5), constrained_layout=True)
    for ax, key, label in [(axs[0, 0], 'small_fraction',
                             rf'Small-grain mass fraction ($a<{args.small_grain_nm:g}$ nm)'),
                            (axs[0, 1], 'carbon_fraction', 'Carbonaceous mass fraction')]:
        ax.scatter(rv, props[key][valid], s=9, alpha=.22, color='#343b49', rasterized=True)
        for group, color in zip(groups, colors):
            ax.errorbar(np.median(maps['rv'][group]), np.median(props[key][group]),
                        xerr=np.array([[np.median(maps['rv'][group])-np.quantile(maps['rv'][group], .16)],
                                       [np.quantile(maps['rv'][group], .84)-np.median(maps['rv'][group])]]),
                        yerr=np.array([[np.median(props[key][group])-np.quantile(props[key][group], .16)],
                                       [np.quantile(props[key][group], .84)-np.median(props[key][group])]]),
                        fmt='o', ms=7, color=color, capsize=2)
        ax.set(xlabel=r'$R(V)$', ylabel=label, ylim=(-.03, 1.03))
    # Average each pixel's composition-specific GSD; this prevents dense pixels
    # from dominating solely because they contain more dust mass.
    for gi, (group, name, color) in enumerate(zip(groups, names, colors)):
        chosen = group.ravel()[pixel_index]
        pix = pixel_index[chosen]
        unique, inverse = np.unique(pix, return_inverse=True)
        total_by_pixel = np.bincount(inverse, weights=mass[chosen])
        for ax, component, title in [(axs[1, 0], fc[chosen], 'Carbonaceous'),
                                     (axs[1, 1], 1-fc[chosen], 'Silicate')]:
            binid = np.searchsorted(edges, radius[chosen], side='right') - 1
            binid = np.clip(binid, 0, len(centers)-1)
            contribution = mass[chosen] * component / total_by_pixel[inverse]
            curve = np.bincount(binid, weights=contribution,
                                minlength=len(centers)) / len(unique) / dlog
            ax.plot(centers, curve, color=color, lw=2,
                    label=f'{name} (N={len(unique)})')
        print(f'{name}: {len(unique)} pixels; median R(V)={np.median(maps["rv"][group]):.3f}; '
              f'median small fraction={np.median(props["small_fraction"][group]):.3f}; '
              f'median carbon fraction={np.median(props["carbon_fraction"][group]):.3f}')
    for ax, title in zip(axs[1], ['Carbonaceous', 'Silicate']):
        ax.set(xscale='log', xlabel='Grain radius (nm)',
               ylabel=r'Mean per-pixel $d(M_{\rm component}/M_{\rm dust})/d\log_{10}a$',
               title=title)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle('CG10: grain populations grouped by projected $R(V)$')
    fig.savefig(args.prefix + '.png', dpi=220)
    plt.close(fig)
    with open(args.prefix + '_pixels.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x_pkpc', 'y_pkpc', 'Rv', 'dust_count', 'dust_mass_code',
                         'tau_B_proxy', 'tau_V_proxy', 'small_grain_mass_fraction',
                         'carbon_mass_fraction', 'Rv_tertile'])
        yy, xx = np.where(valid)
        for py, px in zip(yy, xx):
            writer.writerow([-args.plot_radius+(px+.5)*dx,
                             -args.plot_radius+(py+.5)*dx, maps['rv'][py, px],
                             maps['count_raw'][py, px], maps['mass_raw'][py, px],
                             maps['tauB_raw'][py, px], maps['tauV_raw'][py, px],
                             props['small_fraction'][py, px], props['carbon_fraction'][py, px],
                             names[0] if maps['rv'][py, px] <= q1 else
                             names[1] if maps['rv'][py, px] <= q2 else names[2]])
    # Source records the stellar origin of the superparticle (0/1/2),
    # not the source of mass subsequently acquired from the ISM.
    source_names = {0: 'SNII-born', 1: 'AGB-born', 2: 'LRN-born'}
    source_colors = {0: '#3274a1', 1: '#d59835', 2: '#a03c50'}
    fig2, source_axes = plt.subplots(1, 3, figsize=(14, 4.5),
                                     sharex=True, sharey=True, constrained_layout=True)
    summary_path = args.prefix + '_source_summary.csv'
    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Rv_group', 'Rv_min', 'Rv_max', 'source', 'pixels',
                         'particles', 'source_mass_fraction', 'source_mass_95_105nm_fraction'])
        for ax, group, group_name in zip(source_axes, groups, names):
            chosen = group.ravel()[pixel_index]
            pix = pixel_index[chosen]
            unique, inverse = np.unique(pix, return_inverse=True)
            total_by_pixel = np.bincount(inverse, weights=mass[chosen])
            total_mass = mass[chosen].sum()
            for code, label in source_names.items():
                src = source[chosen] == code
                nsrc = int(src.sum())
                if nsrc == 0:
                    writer.writerow([group_name, np.min(maps['rv'][group]),
                                     np.max(maps['rv'][group]), label,
                                     len(unique), 0, 0, 0])
                    continue
                binid = np.searchsorted(edges, radius[chosen][src], side='right') - 1
                binid = np.clip(binid, 0, len(centers)-1)
                contribution = mass[chosen][src] / total_by_pixel[inverse[src]]
                curve = np.bincount(binid, weights=contribution,
                                    minlength=len(centers)) / len(unique) / dlog
                ax.plot(centers, curve, lw=2, color=source_colors[code], label=label)
                frac = mass[chosen][src].sum() / total_mass
                near = src & (radius[chosen] >= 95) & (radius[chosen] < 105)
                near_frac = mass[chosen][near].sum() / mass[chosen][src].sum()
                writer.writerow([group_name, np.min(maps['rv'][group]),
                                 np.max(maps['rv'][group]), label,
                                 len(unique), nsrc, frac, near_frac])
                print(f'{group_name} {label}: {nsrc} particles, '
                      f'{frac:.3%} of group dust mass; '
                      f'{near_frac:.3%} of source mass at 95-105 nm')
            ax.set(xscale='log', xlabel='Grain radius (nm)', title=f'{group_name} $R(V)$')
            ax.axvline(100, color='0.45', ls=':', lw=1)
            ax.legend(frameon=False, fontsize=8)
        source_axes[0].set_ylabel(
            r'Mean per-pixel $d(M_{\rm source}/M_{\rm dust})/d\log_{10}a$')
        fig2.suptitle('CG10: present grain sizes by stellar birth source and projected $R(V)$')
        fig2.savefig(args.prefix + '_sources.png', dpi=220)
        plt.close(fig2)
    # Compare source fractions within projected radial bands. The same valid
    # pixels, dust masses, and raw map grid are used as above.
    source_mass_maps = {
        code: core.histogram_map(
            x, y, mass * (source == code), args.plot_radius, args.bins
        )[0]
        for code in (0, 1, 2)
    }
    source_fraction_maps = {
        code: np.divide(
            source_mass_maps[code], maps['mass_raw'],
            out=np.full_like(maps['mass_raw'], np.nan, dtype=float),
            where=maps['mass_raw'] > 0
        )
        for code in (0, 1, 2)
    }
    pixel_size = 2.0 * args.plot_radius / args.bins
    pixel_centers = -args.plot_radius + (np.arange(args.bins) + 0.5) * pixel_size
    xx, yy = np.meshgrid(pixel_centers, pixel_centers)
    radial_distance = np.hypot(xx, yy)
    radial_edges = np.linspace(0, args.plot_radius, 5)

    fig3, radial_axes = plt.subplots(
        2, 4, figsize=(15, 7), sharex=True, sharey='row',
        constrained_layout=True
    )
    for column, (rmin, rmax) in enumerate(zip(radial_edges[:-1], radial_edges[1:])):
        band = valid & (radial_distance >= rmin) & (radial_distance < rmax)
        print(f'Projected radius {rmin:.1f}-{rmax:.1f} pkpc: '
              f'{np.count_nonzero(band)} valid pixels')
        for row, (code, title) in enumerate(
            [(1, 'AGB-born dust mass fraction'),
             (0, 'SNII-born dust mass fraction')]
        ):
            ax = radial_axes[row, column]
            ax.scatter(
                maps['rv'][band], source_fraction_maps[code][band],
                s=9, alpha=0.30, color='#343b49', rasterized=True
            )
            ax.set_title(f'{rmin:g}–{rmax:g} pkpc; N={band.sum()}')
            ax.set_ylim(-0.03, 1.03)
            if column == 0:
                ax.set_ylabel(title)
            if row == 1:
                ax.set_xlabel(r'$R(V)$')
    fig3.suptitle('CG10: birth-source fractions versus $R(V)$ at fixed radius')
    fig3.savefig(args.prefix + '_sources_by_radius.png', dpi=220)
    plt.close(fig3)

    radial_csv = args.prefix + '_source_fractions_by_radius.csv'
    with open(radial_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x_pkpc', 'y_pkpc', 'projected_radius_pkpc', 'Rv',
                         'dust_count', 'dust_mass_code', 'SNII_born_mass_fraction',
                         'AGB_born_mass_fraction', 'LRN_born_mass_fraction'])
        py_all, px_all = np.where(valid)
        for py, px in zip(py_all, px_all):
            writer.writerow([
                xx[py, px], yy[py, px], radial_distance[py, px],
                maps['rv'][py, px], maps['count_raw'][py, px],
                maps['mass_raw'][py, px],
                *(source_fraction_maps[code][py, px] for code in (0, 1, 2))
            ])
    # Equal-count R(V) bins within the two radial bands with useful sampling.
    # Percentiles describe pixel-to-pixel scatter; they are not uncertainties
    # on the median and spatially adjacent pixels are not independent.
    n_rv_bins = 5
    fig4, trend_axes = plt.subplots(1, 2, figsize=(10.5, 4.5),
                                    sharey=True, constrained_layout=True)
    trend_csv = args.prefix + '_binned_source_trends.csv'
    with open(trend_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['radius_min_pkpc', 'radius_max_pkpc', 'source',
                         'Rv_bin', 'N_pixels', 'Rv_min', 'Rv_max', 'Rv_median',
                         'fraction_p16', 'fraction_median', 'fraction_p84'])
        for ax, (rmin, rmax) in zip(trend_axes, [(10, 15), (15, 20)]):
            band = valid & (radial_distance >= rmin) & (radial_distance < rmax)
            rv_band = maps['rv'][band]
            order = np.argsort(rv_band)
            # np.array_split makes nonoverlapping, nearly equal-count groups,
            # including when many pixels have an identical R(V).
            chunks = np.array_split(order, min(n_rv_bins, len(order)))
            for code, label in [(1, 'AGB-born'), (0, 'SNII-born')]:
                fraction = source_fraction_maps[code][band]
                centers, medians, lower, upper = [], [], [], []
                for bin_number, indices in enumerate(chunks, start=1):
                    if len(indices) == 0:
                        continue
                    rv_values = rv_band[indices]
                    values = fraction[indices]
                    p16, median, p84 = np.percentile(values, [16, 50, 84])
                    center = np.median(rv_values)
                    centers.append(center)
                    medians.append(median)
                    lower.append(p16)
                    upper.append(p84)
                    writer.writerow([rmin, rmax, label, bin_number,
                                     len(indices), np.min(rv_values),
                                     np.max(rv_values), center, p16, median, p84])
                    ax.annotate(str(len(indices)), (center, median),
                                xytext=(0, 8 if code == 1 else -13),
                                textcoords='offset points', ha='center',
                                fontsize=8, color=source_colors[code])
                if centers:
                    ax.plot(centers, medians, marker='o', lw=2,
                            color=source_colors[code], label=label)
                    ax.fill_between(centers, lower, upper,
                                    color=source_colors[code], alpha=0.15)
            ax.set(xlabel=r'$R(V)$', title=f'{rmin}–{rmax} pkpc; N={band.sum()}',
                   ylim=(-0.03, 1.03))
            ax.legend(frameon=False)
    trend_axes[0].set_ylabel('Birth-source dust mass fraction')
    fig4.suptitle('CG10: source fractions in equal-count $R(V)$ bins\n'
                  'Shading: 16th–84th percentile; numbers: pixels per bin')
    fig4.savefig(args.prefix + '_binned_source_trends.png', dpi=220)
    plt.close(fig4)
    plot_combined_figure(args, maps, props, valid, groups, colors,
                         x, y, mass, radius, fc, source, pixel_index,
                         radial_distance, source_fraction_maps)
    print(f'Saved {args.prefix}_binned_source_trends.png and {trend_csv}')
    print(f'Saved {args.prefix}_sources_by_radius.png and {radial_csv}')
    print(f'Saved {args.prefix}.png, {args.prefix}_pixels.csv, '
          f'{args.prefix}_sources.png, and {summary_path}')


if __name__ == '__main__':
    main()
