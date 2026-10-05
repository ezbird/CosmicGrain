#!/usr/bin/env python3
"""Figure 6 pilot: CG10 projected R(V) beside a supplied Milky Way R(V) map.

From gadget4/scripts:
  python3 cosmicgrain_figure6_mw_comparison.py
  python3 cosmicgrain_figure6_mw_comparison.py --observed-npz zhang_green_xy.npz

Observed NPZ must contain a 2-D array `rv`, and `x_edges` and `y_edges`
in physical kpc (length nx+1 and ny+1). x increases across columns and y
across rows; x=y=0 should be the Galactic centre. Export this Cartesian
slice with the existing Zhang--Green analysis, using its own selection.
The script does not reinterpret a HEALPix sky map as a face-on projection.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from matplotlib.patches import Patch
import h5py
import numpy as np

import halo441_rv_grain_physics as core


def load_observed(path):
    with np.load(path) as data:
        for key in ('rv', 'x_edges', 'y_edges'):
            if key not in data:
                raise ValueError(f'{path}: missing {key!r} array')
        rv = np.asarray(data['rv'], dtype=float)
        xe = np.asarray(data['x_edges'], dtype=float)
        ye = np.asarray(data['y_edges'], dtype=float)
    if rv.ndim != 2 or rv.shape != (len(ye)-1, len(xe)-1):
        raise ValueError('Observed rv shape must be (len(y_edges)-1, len(x_edges)-1)')
    if np.any(np.diff(xe) <= 0) or np.any(np.diff(ye) <= 0):
        raise ValueError('Observed map edges must increase monotonically')
    return rv, xe, ye


def plot_sampling_diagnostic(maps, cg, args, prefix):
    """Show particle sampling and why each pixel is absent from the R(V) map."""
    count = maps['count_raw']
    count_class = np.zeros(count.shape, dtype=int)
    count_class[(count >= 1) & (count <= 2)] = 1
    count_class[(count >= 3) & (count <= 9)] = 2
    count_class[count >= 10] = 3
    count_labels = ['0', '1–2', '3–9', '≥10']
    count_colors = ['#f2f2f2', '#c7d7e9', '#6c9ac6', '#245483']

    # Order is exclusive. A finite R(V) outside the displayed range is
    # distinct from a failed extinction calculation / signal threshold.
    status = np.zeros(count.shape, dtype=int)
    status[(count > 0) & (count < args.min_count)] = 1
    enough = count >= args.min_count
    status[enough & ~np.isfinite(maps['rv'])] = 2
    status[enough & np.isfinite(maps['rv']) & ~np.isfinite(cg)] = 3
    status[np.isfinite(cg)] = 4
    status_labels = ['No dust', f'<{args.min_count} dust particles',
                     'Extinction cut / undefined', 'Outside plotted R(V) range',
                     'Shown in R(V) map']
    status_colors = ['#f2f2f2', '#99b9d8', '#e9a169', '#ba7fbe', '#397d61']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for ax, array, colors, labels, title in [
        (axes[0], count_class, count_colors, count_labels, '(a) Dust particles per pixel'),
        (axes[2], status, status_colors, status_labels, '(c) Why pixels are missing'),
    ]:
        cmap = ListedColormap(colors)
        ax.imshow(array, origin='lower', extent=maps['extent'], cmap=cmap,
                  norm=BoundaryNorm(np.arange(len(colors) + 1) - .5, cmap.N),
                  interpolation='nearest')
        ax.set_title(title)
        ax.legend(handles=[Patch(facecolor=c, label=label) for c, label in zip(colors, labels)],
                  loc='upper center', bbox_to_anchor=(.5, -.13), ncol=2, fontsize=8,
                  frameon=False)
    im = axes[1].imshow(cg, origin='lower', extent=maps['extent'], cmap='viridis',
                        norm=Normalize(args.rv_min, args.rv_max), interpolation='nearest')
    axes[1].set_title(f'(b) $R(V)$; $N_{{\\rm dust}}\\geq{args.min_count}$')
    fig.colorbar(im, ax=axes[1], shrink=.75, label=r'$R(V)$')
    for ax in axes:
        ax.set(xlabel='x (kpc)', ylabel='y (kpc)')
        ax.set_aspect('equal')
    target = prefix.parent / (prefix.name + '_sampling')
    for extension in ('png', 'pdf'):
        fig.savefig(target.with_suffix('.' + extension), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('Dust counts (0, 1–2, 3–9, ≥10):',
          *(int((count_class == i).sum()) for i in range(4)))
    print('Pixel status:', '; '.join(f'{label}: {(status == i).sum()}'
                                    for i, label in enumerate(status_labels)))
    print(f'Saved {target}.png and {target}.pdf')


def projected_mass_radii(pattern, center, rotation, dust, depth):
    """Collect gas and stellar mass at projected radii in the same slab."""
    result = {}
    for ptype in (0, 4):
        radii, masses = [], []
        for filename in core.expand_files(pattern):
            with h5py.File(filename) as handle:
                key = f'PartType{ptype}'
                if key not in handle:
                    continue
                particles = handle[key]
                if 'Masses' not in particles or 'Coordinates' not in particles:
                    raise KeyError(f'{filename}: {key} lacks Coordinates or Masses')
                xyz = core.rotate_positions(particles['Coordinates'][:], center,
                                            dust['box'], dust['a'], dust['h'], rotation)
                mass = np.asarray(particles['Masses'][:], dtype=float)
                ok = (np.abs(xyz[:, 2]) < depth) & np.isfinite(mass) & (mass > 0)
                radii.append(np.hypot(xyz[ok, 0], xyz[ok, 1]))
                masses.append(mass[ok])
        result[ptype] = (np.concatenate(radii) if radii else np.empty(0),
                         np.concatenate(masses) if masses else np.empty(0))
    return result


def plot_radial_diagnostic(args, maps, xyz_dust, selected_dust_mass,
                           center, rotation, dust, prefix):
    profiles = projected_mass_radii(args.snapshot, center, rotation, dust, args.zmax)
    radial_edges = np.arange(0, args.plot_radius, args.radial_step)
    radial_edges = np.append(radial_edges, args.plot_radius)
    mid = (radial_edges[1:] + radial_edges[:-1]) / 2
    area = np.pi * np.diff(radial_edges**2)
    conversion = 1e10 / dust['h']
    dust_r = np.hypot(xyz_dust[:, 0], xyz_dust[:, 1])
    dust_mass = np.histogram(dust_r, radial_edges, weights=selected_dust_mass)[0]
    gas_mass = np.histogram(profiles[0][0], radial_edges, weights=profiles[0][1])[0]
    star_mass = np.histogram(profiles[4][0], radial_edges, weights=profiles[4][1])[0]
    rv = maps['rv']
    count = maps['count_raw']
    positions = -args.plot_radius + (np.arange(args.bins) + .5) * maps['pixel_size']
    xx, yy = np.meshgrid(positions, positions)
    rp = np.hypot(xx, yy)
    finite_rv = np.isfinite(rv) & (rv >= args.rv_min) & (rv <= args.rv_max)

    fig, axs = plt.subplots(2, 2, figsize=(10.5, 8), constrained_layout=True)
    ax = axs[0, 0]
    for masses, name, color in [(dust_mass, 'Dust', '#b26741'),
                                (gas_mass, 'Gas', '#39799a'),
                                (star_mass, 'Stars', '#424a6e')]:
        sigma = masses * conversion / area
        ax.plot(mid, np.where(sigma > 0, sigma, np.nan), 'o-', label=name, color=color)
    ax.set_yscale('log')
    ax.set(ylabel=r'Surface density ($M_\odot\,\mathrm{kpc}^{-2}$)',
           title='(a) Projected mass profiles')
    ax.legend(frameon=False)

    ratio = np.divide(dust_mass, gas_mass, out=np.full_like(dust_mass, np.nan),
                      where=gas_mass > 0)
    axs[0, 1].plot(mid, ratio, 'o-', color='#78579c')
    axs[0, 1].set(ylabel='Dust-to-gas mass ratio', title='(b) Projected dust / gas')
    for threshold, color in [(3, '#cc8b27'), (10, '#376799')]:
        coverage, medians = [], []
        for lo, hi in zip(radial_edges[:-1], radial_edges[1:]):
            annulus = (rp >= lo) & (rp < hi)
            valid = annulus & finite_rv & (count >= threshold)
            coverage.append(valid.sum() / annulus.sum() if annulus.any() else np.nan)
            medians.append(np.median(rv[valid]) if valid.any() else np.nan)
        axs[1, 0].plot(mid, coverage, 'o-', label=rf'$N_{{\rm dust}}\geq {threshold}$',
                       color=color)
        axs[1, 1].plot(mid, medians, 'o-', label=rf'$N_{{\rm dust}}\geq {threshold}$',
                       color=color)
    axs[1, 0].set(ylabel='Fraction of annulus pixels with valid $R(V)$',
                  title='(c) 500 pc pixel coverage' if args.bins == 80 and args.plot_radius == 20
                  else f'(c) {maps["pixel_size"]:g} kpc pixel coverage', ylim=(0, 1))
    axs[1, 1].set(ylabel='Median $R(V)$ of valid pixels', title='(d) Measured $R(V)$')
    axs[1, 0].legend(frameon=False, fontsize=9)
    for ax in axs.flat:
        ax.set_xlabel('Projected radius (kpc)')
        ax.set_xlim(0, args.plot_radius)
    target = prefix.parent / (prefix.name + '_radial')
    for ext in ('png', 'pdf'):
        fig.savefig(target.with_suffix('.' + ext), dpi=200)
    plt.close(fig)
    print(f'Saved {target}.png and {target}.pdf')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--snapshot', default='../halo441_output_1024/snapdir_029/snapshot_029.*.hdf5')
    p.add_argument('--center-file', default='halo441_final_snap029_audited_center.txt')
    p.add_argument('--rotation-file', default='halo441_final_snap029_audited_rotation_matrix.txt')
    p.add_argument('--carbon-table', default='dust_optics/callqcomp.out_D16MGemt')
    p.add_argument('--silicate-table', default='dust_optics/callqcomp.out_DH16asil')
    p.add_argument('--observed-npz', type=Path, help='Cartesian Zhang--Green map, schema in docstring')
    p.add_argument('--plot-radius', type=float, default=20.)
    p.add_argument('--zmax', type=float, default=20.)
    p.add_argument('--bins', type=int, default=80)
    p.add_argument('--min-count', type=int, default=10)
    p.add_argument('--min-signal-fraction', type=float, default=1e-3)
    p.add_argument('--rv-min', type=float, default=1.5)
    p.add_argument('--rv-max', type=float, default=10.)
    p.add_argument('--prefix', default='../CG_figure6_pilot')
    p.add_argument('--radial-step', type=float, default=2., help='Radial profile width in kpc')
    args = p.parse_args()
    if args.plot_radius <= 0 or args.zmax <= 0 or args.bins < 2 or args.min_count < 1:
        p.error('invalid radius, depth, bins, or minimum dust count')
    if args.rv_max <= args.rv_min:
        p.error('rv-max must exceed rv-min')
    if args.radial_step <= 0 or args.radial_step >= args.plot_radius:
        p.error('radial-step must be positive and smaller than plot-radius')

    carbon = core.read_draine_qext_table(args.carbon_table)
    silicate = core.read_draine_qext_table(args.silicate_table)
    dust = core.read_dust_snapshot(args.snapshot)
    center = np.loadtxt(args.center_file).reshape(3)
    rotation = np.loadtxt(args.rotation_file).reshape(3, 3)
    xyz = core.rotate_positions(dust['coords'], center, dust['box'], dust['a'], dust['h'], rotation)
    chosen = ((np.abs(xyz[:, 0]) < args.plot_radius) &
              (np.abs(xyz[:, 1]) < args.plot_radius) &
              (np.abs(xyz[:, 2]) < args.zmax) &
              np.isfinite(dust['mass']) & (dust['mass'] > 0) &
              np.isfinite(dust['radius_nm']) & (dust['radius_nm'] > 0) &
              np.isfinite(dust['carbon']))
    if not np.any(chosen):
        raise RuntimeError('No usable CG10 dust particles in map volume')
    wb, wv = core.extinction_weights(dust['mass'][chosen], dust['radius_nm'][chosen],
                                     np.clip(dust['carbon'][chosen], 0, 1), carbon, silicate)
    maps = core.build_maps(xyz[chosen, 0], xyz[chosen, 1], dust['mass'][chosen],
                           wb, wv, args.plot_radius, args.bins, 0., args.min_signal_fraction)
    cg = np.where(maps['count_raw'] >= args.min_count, maps['rv'], np.nan)
    observed = None
    if args.observed_npz:
        observed, xe, ye = load_observed(args.observed_npz)
        # Restrict both data sets to the same square and R(V) range. The maps
        # remain on their native grids: no interpolation creates new data.
        ox = (xe[:-1] + xe[1:]) / 2
        oy = (ye[:-1] + ye[1:]) / 2
        square = ((np.abs(ox)[None, :] < args.plot_radius) &
                  (np.abs(oy)[:, None] < args.plot_radius))
        observed = np.where(square, observed, np.nan)
        observed = np.where((observed >= args.rv_min) & (observed <= args.rv_max),
                            observed, np.nan)
    cg = np.where((cg >= args.rv_min) & (cg <= args.rv_max), cg, np.nan)
    if not np.isfinite(cg).any():
        raise RuntimeError('No CG10 pixels survive count and R(V) cuts')

    fig, axes = plt.subplots(1, 3 if observed is not None else 2,
                             figsize=(14 if observed is not None else 10, 4.5),
                             constrained_layout=True)
    norm = Normalize(args.rv_min, args.rv_max)
    im = axes[0].imshow(cg, origin='lower', extent=maps['extent'], cmap='viridis', norm=norm)
    axes[0].set_title(f'(a) CG10, face-on; {maps["pixel_size"]:.2f} kpc pixels')
    if observed is not None:
        axes[1].imshow(observed, origin='lower', extent=[xe[0], xe[-1], ye[0], ye[-1]],
                       cmap='viridis', norm=norm, interpolation='nearest')
        axes[1].set_xlim(-args.plot_radius, args.plot_radius)
        axes[1].set_ylim(-args.plot_radius, args.plot_radius)
        axes[1].set_title('(b) Zhang--Green Cartesian slice')
    for ax in axes[:2] if observed is not None else axes[:1]:
        ax.set(xlabel='x (kpc)', ylabel='y (kpc)')
        ax.set_aspect('equal')
    fig.colorbar(im, ax=axes[:2] if observed is not None else axes[:1],
                 label=r'$R(V)$', shrink=.8)
    histax = axes[-1]
    edges = np.linspace(args.rv_min, args.rv_max, 30)
    histax.hist(cg[np.isfinite(cg)], bins=edges, density=True, histtype='step',
                linewidth=2, color='#385c9b', label=f'CG10 (N={np.isfinite(cg).sum()})')
    if observed is not None and np.isfinite(observed).any():
        histax.hist(observed[np.isfinite(observed)], bins=edges, density=True,
                    histtype='step', linewidth=2, color='#c28b26',
                    label=f'Zhang--Green (N={np.isfinite(observed).sum()})')
    histax.set(xlabel=r'$R(V)$', ylabel='Fraction per unit $R(V)$',
               title='(c) Pixel distributions' if observed is not None else '(b) Pixel distribution')
    histax.legend(fontsize=8)
    prefix = Path(args.prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'pdf'):
        fig.savefig(prefix.with_suffix('.' + extension), dpi=200)
    plt.close(fig)
    plot_sampling_diagnostic(maps, cg, args, prefix)
    plot_radial_diagnostic(args, maps, xyz[chosen], dust['mass'][chosen],
                           center, rotation, dust, prefix)
    print(f'CG10: {np.isfinite(cg).sum()} usable pixels; median R(V)={np.nanmedian(cg):.3f}')
    if observed is not None:
        print(f'Zhang--Green: {np.isfinite(observed).sum()} usable pixels; '
              f'median R(V)={np.nanmedian(observed):.3f}')
    print(f'Saved {prefix}.png and {prefix}.pdf')


if __name__ == '__main__':
    main()
