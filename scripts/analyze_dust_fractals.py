#!/usr/bin/env python3
"""
analyze_dust_comprehensive.py

Comprehensive dust analysis for Gadget-4 simulations focusing on:
- Fractal structure (correlation function)
- Spatial distribution (2D/3D maps)
- Dust-gas coupling
- Radial profiles
- Clumping statistics
- Time evolution (if multiple snapshots)

Usage:
    python analyze_dust_comprehensive.py /path/to/output/folder
    python analyze_dust_comprehensive.py /path/to/output/folder --snapshots 10,15,20
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.spatial import cKDTree
from scipy.stats import binned_statistic, binned_statistic_2d
import glob
import os
import sys
import argparse
from pathlib import Path
import time

class GadgetSnapshot:
    """Handle multi-file Gadget-4 snapshots"""
    
    def __init__(self, snapshot_base):
        self.snapshot_base = snapshot_base
        self.files = self._find_files()
        
    def _find_files(self):
        """Find all files belonging to this snapshot"""
        single_file = f"{self.snapshot_base}.hdf5"
        if os.path.exists(single_file):
            return [single_file]
        
        pattern = f"{self.snapshot_base}.*.hdf5"
        files = sorted(glob.glob(pattern))
        
        if len(files) == 0:
            raise FileNotFoundError(f"No snapshot files found for {self.snapshot_base}")
        
        return files
    
    def load_particles(self, parttype):
        """Load particle data from all files"""
        positions = []
        masses = []
        velocities = []
        metallicities = []
        boxsize = None
        sim_time = None
        redshift = None
        
        parttype_name = f'PartType{parttype}'
        
        print(f"    Reading {len(self.files)} file(s)...", end='', flush=True)
        
        for fname in self.files:
            try:
                with h5py.File(fname, 'r') as f:
                    if boxsize is None:
                        header = dict(f['Header'].attrs)
                        boxsize = header['BoxSize']
                        sim_time = header.get('Time', 1.0)
                        redshift = header.get('Redshift', 0.0)
                    
                    if parttype_name not in f.keys():
                        continue
                    
                    if 'Coordinates' in f[parttype_name].keys():
                        pos = f[parttype_name]['Coordinates'][:]
                        mass = f[parttype_name]['Masses'][:]
                        vel = f[parttype_name]['Velocities'][:]
                        
                        positions.append(pos)
                        masses.append(mass)
                        velocities.append(vel)
                        
                        # Try to get metallicity
                        if 'Metallicity' in f[parttype_name].keys():
                            met = f[parttype_name]['Metallicity'][:]
                            metallicities.append(met)
            
            except Exception as e:
                print(f"\n    Warning: Error reading {fname}: {e}")
                continue
        
        print(" Done!", flush=True)
        
        if len(positions) == 0:
            return None
        
        print(f"    Concatenating arrays...", end='', flush=True)
        result = {
            'pos': np.vstack(positions),
            'mass': np.concatenate(masses),
            'vel': np.vstack(velocities),
            'boxsize': boxsize,
            'sim_time': sim_time,
            'redshift': redshift
        }
        
        if len(metallicities) > 0:
            result['metallicity'] = np.concatenate(metallicities)
        
        print(" Done!", flush=True)
        
        return result


def compute_correlation_function(positions, boxsize, r_bins, n_sample=5000):
    """Compute two-point correlation function"""
    N = len(positions)
    if N < 10:
        return None, None
    
    print(f"    Building KD-tree...", end='', flush=True)
    t0 = time.time()
    tree = cKDTree(positions, boxsize=boxsize)
    print(f" Done! ({time.time()-t0:.1f}s)", flush=True)
    
    DD = np.zeros(len(r_bins) - 1)
    
    sample_size = min(N, n_sample)
    indices = np.random.choice(N, sample_size, replace=False)
    
    print(f"    Computing correlations for {sample_size} sampled particles:")
    
    t0 = time.time()
    for count, i in enumerate(indices):
        if count % (sample_size // 10) == 0:
            elapsed = time.time() - t0
            percent = 100 * count / sample_size
            if count > 0:
                eta = elapsed * (sample_size - count) / count
                print(f"      {percent:5.1f}% complete - Elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s", flush=True)
            else:
                print(f"      {percent:5.1f}% complete", flush=True)
        
        neighbors = tree.query_ball_point(positions[i], r_bins[-1])
        
        if len(neighbors) > 1:
            dxyz = positions[neighbors] - positions[i]
            dxyz = dxyz - boxsize * np.round(dxyz / boxsize)
            distances = np.sqrt(np.sum(dxyz**2, axis=1))
            distances = distances[distances > 0]
            hist, _ = np.histogram(distances, bins=r_bins)
            DD += hist
    
    elapsed = time.time() - t0
    print(f"      100.0% complete - Total time: {elapsed:.1f}s", flush=True)
    
    DD = DD * (N / sample_size)
    
    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
    shell_volumes = (4/3) * np.pi * (r_bins[1:]**3 - r_bins[:-1]**3)
    number_density = N / boxsize**3
    RR = number_density * shell_volumes * N
    
    mask = RR > 0
    xi = np.zeros_like(DD)
    xi[mask] = DD[mask] / RR[mask] - 1.0
    
    return r_centers, xi


def fit_fractal_dimension(r, xi, r_min=0.5, r_max=1000.0):
    """Fit power law to get fractal dimension"""
    mask = (r > r_min) & (r < r_max) & (xi > 0) & np.isfinite(xi)
    
    if np.sum(mask) < 3:
        return np.nan, np.nan
    
    log_r = np.log10(r[mask])
    log_xi = np.log10(xi[mask])
    
    coeffs = np.polyfit(log_r, log_xi, 1)
    slope = coeffs[0]
    fractal_dim = 3.0 + slope
    
    return fractal_dim, slope


def compute_clumping_factor(masses, positions, boxsize, scales):
    """
    Compute density variance (clumping factor) at different scales
    
    Clumping factor: <ρ²>/<ρ>² 
    Measures how much density fluctuates
    """
    print(f"    Computing clumping factor at {len(scales)} scales...")
    
    clumping = []
    mean_density = np.sum(masses) / boxsize**3
    
    for R in scales:
        # Sample random points
        n_samples = 200
        local_densities = []
        
        for _ in range(n_samples):
            center = np.random.uniform(0, boxsize, 3)
            
            # Find particles within R
            dxyz = positions - center
            dxyz = dxyz - boxsize * np.round(dxyz / boxsize)
            distances = np.sqrt(np.sum(dxyz**2, axis=1))
            
            mask = distances < R
            local_mass = np.sum(masses[mask])
            volume = (4/3) * np.pi * R**3
            local_density = local_mass / volume
            local_densities.append(local_density)
        
        local_densities = np.array(local_densities)
        
        # Clumping factor
        if np.mean(local_densities) > 0:
            C = np.mean(local_densities**2) / np.mean(local_densities)**2
            clumping.append(C)
        else:
            clumping.append(np.nan)
    
    return np.array(clumping)


def compute_radial_profile(positions, masses, center, r_bins):
    """Compute mass-weighted radial profile from center"""
    distances = np.sqrt(np.sum((positions - center)**2, axis=1))
    
    profile, _, _ = binned_statistic(distances, masses, statistic='sum', bins=r_bins)
    counts, _, _ = binned_statistic(distances, masses, statistic='count', bins=r_bins)
    
    # Convert to surface density
    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
    annulus_area = np.pi * (r_bins[1:]**2 - r_bins[:-1]**2)
    surface_density = profile / annulus_area
    
    return r_centers, surface_density, counts


def dust_gas_correlation_analysis(dust_pos, dust_mass, gas_pos, gas_mass, boxsize):
    """
    Analyze how well dust traces gas density
    For each dust particle, find local gas density
    """
    print("    Computing dust-gas correlation...")
    
    # Build gas density tree
    gas_tree = cKDTree(gas_pos, boxsize=boxsize)
    
    # For each dust particle, find nearby gas
    search_radius = 10.0  # kpc
    
    dust_local_gas_density = []
    
    for dust_p in dust_pos:
        neighbors = gas_tree.query_ball_point(dust_p, search_radius)
        
        if len(neighbors) > 0:
            local_gas_mass = np.sum(gas_mass[neighbors])
            volume = (4/3) * np.pi * search_radius**3
            density = local_gas_mass / volume
            dust_local_gas_density.append(density)
        else:
            dust_local_gas_density.append(0.0)
    
    return np.array(dust_local_gas_density)


def make_projection_map(positions, masses, boxsize, axis='z', npix=512):
    """Create 2D projection map"""
    
    # Select axes for projection
    if axis == 'z':
        x_idx, y_idx = 0, 1
    elif axis == 'y':
        x_idx, y_idx = 0, 2
    else:  # x
        x_idx, y_idx = 1, 2
    
    # Create 2D histogram
    H, xedges, yedges = np.histogram2d(
        positions[:, x_idx], 
        positions[:, y_idx],
        bins=npix,
        range=[[0, boxsize], [0, boxsize]],
        weights=masses
    )
    
    return H.T, xedges, yedges


def detect_galaxy_extent(positions, masses, boxsize, percentile=90):
    """
    Detect actual galaxy extent for zoom simulations
    Returns radius containing percentile% of mass
    """
    # Find center of mass
    center = np.average(positions, weights=masses, axis=0)
    
    # Compute distances from center
    dxyz = positions - center
    # Don't apply periodic boundaries - zoom region should be localized
    distances = np.sqrt(np.sum(dxyz**2, axis=1))
    
    # Sort by distance
    sorted_idx = np.argsort(distances)
    sorted_masses = masses[sorted_idx]
    sorted_distances = distances[sorted_idx]
    
    # Find radius containing percentile% of mass
    cumulative_mass = np.cumsum(sorted_masses)
    total_mass = cumulative_mass[-1]
    target_mass = (percentile / 100.0) * total_mass
    
    idx = np.searchsorted(cumulative_mass, target_mass)
    galaxy_radius = sorted_distances[idx]
    
    return center, galaxy_radius


def analyze_snapshot(snapshot_base, output_dir):
    """Complete dust analysis of one snapshot"""
    
    snap = GadgetSnapshot(snapshot_base)
    snap_name = os.path.basename(snapshot_base)
    
    print(f"\n{'='*60}")
    print(f"Analyzing {snap_name}")
    print(f"{'='*60}")
    
    # Load gas
    print("  Loading gas particles...")
    gas_data = snap.load_particles(0)
    
    # Load dust
    print("  Loading dust particles...")
    dust_data = snap.load_particles(6)
    
    if dust_data is None or gas_data is None:
        print("  ERROR: Missing gas or dust data!")
        return None
    
    boxsize = dust_data['boxsize']
    sim_time = dust_data['sim_time']
    redshift = dust_data['redshift']
    
    print(f"\n  Simulation info:")
    print(f"    Box size: {boxsize:.2f} kpc/h")
    print(f"    Time: {sim_time:.4f}")
    print(f"    Redshift: {redshift:.3f}")
    print(f"    Gas particles: {len(gas_data['pos'])}")
    print(f"    Dust particles: {len(dust_data['pos'])}")
    
    dust_pos = dust_data['pos']
    dust_mass = dust_data['mass']
    gas_pos = gas_data['pos']
    gas_mass = gas_data['mass']
    
    # Detect galaxy extent (for zoom simulations)
    print("\n  Detecting galaxy extent...")
    gas_center, gas_radius_90 = detect_galaxy_extent(gas_pos, gas_mass, boxsize, percentile=90)
    dust_center, dust_radius_90 = detect_galaxy_extent(dust_pos, dust_mass, boxsize, percentile=90)
    
    # Use gas as primary (more particles, better statistics)
    center = gas_center
    galaxy_radius = gas_radius_90
    
    print(f"    Galaxy center: ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}) kpc/h")
    print(f"    Galaxy radius (90% mass): {galaxy_radius:.1f} kpc/h")
    
    # Determine if this is a zoom simulation
    is_zoom = (galaxy_radius < boxsize / 4)
    if is_zoom:
        print(f"    Detected ZOOM simulation (galaxy << box)")
    else:
        print(f"    Detected FULL-BOX simulation")
    
    # Total masses
    total_gas_mass = np.sum(gas_mass)
    total_dust_mass = np.sum(dust_mass)
    
    print(f"    Total gas mass: {total_gas_mass:.3e}")
    print(f"    Total dust mass: {total_dust_mass:.3e}")
    print(f"    Dust/Gas ratio: {total_dust_mass/total_gas_mass:.3e}")
    
    if 'metallicity' in gas_data:
        total_metal_mass = np.sum(gas_mass * gas_data['metallicity'])
        print(f"    Total metal mass: {total_metal_mass:.3e}")
        print(f"    Dust/Metal ratio: {total_dust_mass/total_metal_mass:.3f}")
    
    results = {
        'snapshot': snap_name,
        'boxsize': boxsize,
        'sim_time': sim_time,
        'redshift': redshift,
        'n_gas': len(gas_pos),
        'n_dust': len(dust_pos),
        'total_gas_mass': total_gas_mass,
        'total_dust_mass': total_dust_mass,
        'is_zoom': is_zoom,
        'galaxy_center': center,
        'galaxy_radius': galaxy_radius,
    }
    
    # ========================================
    # 1. CORRELATION FUNCTION
    # ========================================
    print("\n  [1/5] Computing correlation function...")
    t_start = time.time()
    
    # Set bin range based on galaxy size
    if is_zoom:
        # For zoom: from resolution (~0.1 kpc) to galaxy size
        r_min_bin = 0.1
        r_max_bin = min(galaxy_radius * 1.5, boxsize / 4)
        # Fit range: avoid resolution effects and edges
        fit_r_min = 1.0  # Above resolution
        fit_r_max = galaxy_radius * 0.8  # Within main galaxy body
    else:
        # For full box: standard ranges
        r_min_bin = 0.1
        r_max_bin = boxsize / 4
        fit_r_min = 0.5
        fit_r_max = 1000.0
    
    print(f"    Correlation bins: {r_min_bin:.1f} - {r_max_bin:.1f} kpc/h")
    print(f"    Fit range: {fit_r_min:.1f} - {fit_r_max:.1f} kpc/h")
    
    r_bins = np.logspace(np.log10(r_min_bin), np.log10(r_max_bin), 20)
    r_dust, xi_dust = compute_correlation_function(dust_pos, boxsize, r_bins, n_sample=5000)
    
    if r_dust is not None:
        D_dust, slope_dust = fit_fractal_dimension(r_dust, xi_dust, 
                                                    r_min=fit_r_min, 
                                                    r_max=fit_r_max)
        
        # Diagnostic output
        mask_fit = (r_dust > fit_r_min) & (r_dust < fit_r_max) & (xi_dust > 0) & np.isfinite(xi_dust)
        print(f"    Fit diagnostics:")
        print(f"      Points in fit range: {np.sum(mask_fit)}")
        if np.sum(mask_fit) > 0:
            print(f"      r range: {r_dust[mask_fit].min():.1f} - {r_dust[mask_fit].max():.1f} kpc")
            print(f"      xi range: {xi_dust[mask_fit].min():.2e} - {xi_dust[mask_fit].max():.2e}")
            print(f"      Slope: {slope_dust:.3f}")
        print(f"    Fractal dimension: D = {D_dust:.2f}")
        
        results['correlation'] = {
            'r': r_dust,
            'xi': xi_dust,
            'D': D_dust,
            'slope': slope_dust,
            'fit_r_min': fit_r_min,
            'fit_r_max': fit_r_max
        }
    else:
        results['correlation'] = None
    
    print(f"    Completed in {time.time()-t_start:.1f}s")
    
    # ========================================
    # 2. CLUMPING FACTOR
    # ========================================
    print("\n  [2/5] Computing clumping factor...")
    t_start = time.time()
    
    # Set scales based on galaxy size
    if is_zoom:
        scale_min = 1.0  # kpc, ~resolution
        scale_max = galaxy_radius
        n_scales = 10
    else:
        scale_min = 1.0
        scale_max = 1000.0
        n_scales = 10
    
    scales = np.logspace(np.log10(scale_min), np.log10(scale_max), n_scales)
    clumping = compute_clumping_factor(dust_mass, dust_pos, boxsize, scales)
    
    results['clumping'] = {
        'scales': scales,
        'values': clumping
    }
    
    print(f"    Completed in {time.time()-t_start:.1f}s")
    
    # ========================================
    # 3. RADIAL PROFILES
    # ========================================
    print("\n  [3/5] Computing radial profiles...")
    t_start = time.time()
    
    # Use detected center (not box center!)
    print(f"    Centering on detected galaxy center")
    
    # Set radial bins based on galaxy size
    if is_zoom:
        r_max_profile = galaxy_radius * 2.0
    else:
        r_max_profile = boxsize / 2
    
    r_bins_profile = np.logspace(np.log10(10.0), np.log10(r_max_profile), 30)
    
    r_dust_prof, dust_profile, dust_counts = compute_radial_profile(
        dust_pos, dust_mass, center, r_bins_profile
    )
    
    r_gas_prof, gas_profile, gas_counts = compute_radial_profile(
        gas_pos, gas_mass, center, r_bins_profile
    )
    
    results['radial_profile'] = {
        'r': r_dust_prof,
        'dust': dust_profile,
        'gas': gas_profile,
        'dust_counts': dust_counts,
        'gas_counts': gas_counts,
        'center': center,
        'galaxy_radius': galaxy_radius
    }
    
    print(f"    Completed in {time.time()-t_start:.1f}s")
    
    # ========================================
    # 4. DUST-GAS CORRELATION
    # ========================================
    print("\n  [4/5] Analyzing dust-gas correlation...")
    t_start = time.time()
    
    dust_local_gas_dens = dust_gas_correlation_analysis(
        dust_pos, dust_mass, gas_pos, gas_mass, boxsize
    )
    
    results['dust_gas_corr'] = {
        'local_gas_density': dust_local_gas_dens
    }
    
    print(f"    Completed in {time.time()-t_start:.1f}s")
    
    # ========================================
    # 5. PROJECTION MAPS
    # ========================================
    print("\n  [5/5] Creating projection maps...")
    t_start = time.time()
    
    dust_map_xy, x_edges, y_edges = make_projection_map(dust_pos, dust_mass, boxsize, axis='z', npix=512)
    gas_map_xy, _, _ = make_projection_map(gas_pos, gas_mass, boxsize, axis='z', npix=512)
    
    results['maps'] = {
        'dust_xy': dust_map_xy,
        'gas_xy': gas_map_xy,
        'extent': [0, boxsize, 0, boxsize]
    }
    
    print(f"    Completed in {time.time()-t_start:.1f}s")
    
    # ========================================
    # CREATE PLOTS
    # ========================================
    print("\n  Creating diagnostic plots...")
    make_comprehensive_plots(results, output_dir)
    
    return results


def make_comprehensive_plots(results, output_dir):
    """Create comprehensive multi-panel diagnostic plots"""
    
    os.makedirs(output_dir, exist_ok=True)
    snap_name = results['snapshot']
    
    # ========================================
    # FIGURE 1: MAIN DIAGNOSTICS (2x2)
    # ========================================
    
    fig = plt.figure(figsize=(14, 12))
    
    # Panel 1: Correlation function
    ax1 = plt.subplot(2, 2, 1)
    if results['correlation'] is not None:
        r = results['correlation']['r']
        xi = results['correlation']['xi']
        D = results['correlation']['D']
        fit_r_min = results['correlation']['fit_r_min']
        fit_r_max = results['correlation']['fit_r_max']
        slope = results['correlation']['slope']
        
        mask = (xi > 0) & np.isfinite(xi)
        ax1.loglog(r[mask], xi[mask], 'o-', color='#d62728', 
                   label=f'Dust (D={D:.2f})', markersize=6)
        
        # Plot fit line using actual fit coefficients
        if np.isfinite(D) and np.isfinite(slope):
            # Get fit region data to determine intercept
            mask_fit = (r > fit_r_min) & (r < fit_r_max) & (xi > 0) & np.isfinite(xi)
            if np.sum(mask_fit) >= 3:
                # Recompute fit to get intercept
                log_r_fit = np.log10(r[mask_fit])
                log_xi_fit = np.log10(xi[mask_fit])
                coeffs = np.polyfit(log_r_fit, log_xi_fit, 1)
                fit_slope = coeffs[0]
                fit_intercept = coeffs[1]
                
                # Generate fit line
                r_fit = np.logspace(np.log10(fit_r_min), np.log10(fit_r_max), 100)
                log_xi_fit_line = fit_slope * np.log10(r_fit) + fit_intercept
                xi_fit = 10**log_xi_fit_line
                
                ax1.loglog(r_fit, xi_fit, '--', color='#d62728', alpha=0.7, linewidth=2.5,
                          label=f'Fit: slope={fit_slope:.2f}')
        
        # Mark fit region
        ax1.axvspan(fit_r_min, fit_r_max, alpha=0.1, color='gray', label='Fit region')
    
    ax1.set_xlabel('Separation r [kpc/h]', fontsize=11)
    ax1.set_ylabel('ξ(r)', fontsize=11)
    ax1.set_title('Two-Point Correlation Function', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Clumping factor
    ax2 = plt.subplot(2, 2, 2)
    if 'clumping' in results:
        scales = results['clumping']['scales']
        clump = results['clumping']['values']
        
        mask = np.isfinite(clump) & (clump > 0)
        ax2.loglog(scales[mask], clump[mask], 'o-', color='#ff7f0e', 
                   markersize=6, linewidth=2)
        ax2.axhline(1, color='k', linestyle=':', alpha=0.5, label='No clumping')
    
    ax2.set_xlabel('Scale R [kpc]', fontsize=11)
    ax2.set_ylabel('Clumping Factor <ρ²>/<ρ>²', fontsize=11)
    ax2.set_title('Density Variance vs Scale', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Radial profiles
    ax3 = plt.subplot(2, 2, 3)
    if 'radial_profile' in results:
        r = results['radial_profile']['r']
        dust_prof = results['radial_profile']['dust']
        gas_prof = results['radial_profile']['gas']
        galaxy_radius = results['radial_profile']['galaxy_radius']
        
        mask_dust = (dust_prof > 0) & np.isfinite(dust_prof)
        mask_gas = (gas_prof > 0) & np.isfinite(gas_prof)
        
        ax3.loglog(r[mask_dust], dust_prof[mask_dust], 'o-', 
                   color='#d62728', label='Dust', markersize=5)
        ax3.loglog(r[mask_gas], gas_prof[mask_gas], 's-', 
                   color='#1f77b4', label='Gas', markersize=5, alpha=0.7)
        
        # Mark galaxy radius (90% mass)
        ax3.axvline(galaxy_radius, color='k', linestyle=':', alpha=0.5,
                   label=f'R90 = {galaxy_radius:.0f} kpc')
    
    ax3.set_xlabel('Radius [kpc]', fontsize=11)
    ax3.set_ylabel('Surface Density [M☉/kpc²]', fontsize=11)
    ax3.set_title('Radial Mass Profiles', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Dust vs local gas density
    ax4 = plt.subplot(2, 2, 4)
    if 'dust_gas_corr' in results:
        gas_dens = results['dust_gas_corr']['local_gas_density']
        
        # 2D histogram
        mask = gas_dens > 0
        if np.sum(mask) > 10:
            bins = np.logspace(np.log10(gas_dens[mask].min()), 
                              np.log10(gas_dens[mask].max()), 50)
            hist, bin_edges = np.histogram(gas_dens[mask], bins=bins)
            
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            ax4.loglog(bin_centers, hist, 'o-', color='#2ca02c', markersize=5)
    
    ax4.set_xlabel('Local Gas Density [M☉/kpc³]', fontsize=11)
    ax4.set_ylabel('Number of Dust Particles', fontsize=11)
    ax4.set_title('Dust Distribution vs Gas Density', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'{snap_name} - Dust Diagnostics', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    outfile = os.path.join(output_dir, f'{snap_name}_diagnostics.png')
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f"    Saved: {outfile}")
    plt.close()
    
    # ========================================
    # FIGURE 2: PROJECTION MAPS (1x2)
    # ========================================
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    if 'maps' in results:
        dust_map = results['maps']['dust_xy']
        gas_map = results['maps']['gas_xy']
        extent = results['maps']['extent']
        
        # Dust map
        ax = axes[0]
        im1 = ax.imshow(dust_map, origin='lower', extent=extent,
                       cmap='inferno', norm=LogNorm(vmin=dust_map[dust_map>0].min(), 
                                                     vmax=dust_map.max()),
                       interpolation='nearest')
        ax.set_xlabel('X [kpc/h]', fontsize=11)
        ax.set_ylabel('Y [kpc/h]', fontsize=11)
        ax.set_title('Dust Surface Density', fontsize=12, fontweight='bold')
        plt.colorbar(im1, ax=ax, label='Surface Density [M☉/kpc²]')
        
        # Gas map
        ax = axes[1]
        im2 = ax.imshow(gas_map, origin='lower', extent=extent,
                       cmap='viridis', norm=LogNorm(vmin=gas_map[gas_map>0].min(), 
                                                     vmax=gas_map.max()),
                       interpolation='nearest')
        ax.set_xlabel('X [kpc/h]', fontsize=11)
        ax.set_ylabel('Y [kpc/h]', fontsize=11)
        ax.set_title('Gas Surface Density', fontsize=12, fontweight='bold')
        plt.colorbar(im2, ax=ax, label='Surface Density [M☉/kpc²]')
    
    plt.suptitle(f'{snap_name} - Face-on Projections', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    outfile = os.path.join(output_dir, f'{snap_name}_maps.png')
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f"    Saved: {outfile}")
    plt.close()


def find_snapshots(output_folder):
    """Find all snapshots in output folder"""
    output_path = Path(output_folder)
    
    snapshots = []
    
    # Look for snapdir_XXX subdirectories
    snapdirs = sorted(output_path.glob('snapdir_*'))
    
    if len(snapdirs) > 0:
        for snapdir in snapdirs:
            snap_num = snapdir.name.split('_')[-1]
            snap_base = snapdir / f'snapshot_{snap_num}'
            
            if len(glob.glob(f"{snap_base}*.hdf5")) > 0:
                snapshots.append(str(snap_base))
    else:
        # Look for snapshots directly in output folder
        snap_files = sorted(output_path.glob('snapshot_*.hdf5'))
        
        snap_nums = set()
        for f in snap_files:
            parts = f.stem.split('_')
            if len(parts) >= 2:
                num = parts[1].split('.')[0]
                snap_nums.add(num)
        
        for num in sorted(snap_nums):
            snap_base = output_path / f'snapshot_{num}'
            snapshots.append(str(snap_base))
    
    return snapshots


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive dust analysis for Gadget-4 simulations'
    )
    parser.add_argument('output_folder', help='Path to simulation output folder')
    parser.add_argument('--snapshots', 
                       help='Comma-separated list of snapshot numbers (e.g., "10,20,30")')
    parser.add_argument('--output-dir', default='dust_analysis', 
                       help='Directory for output plots (default: dust_analysis)')
    
    args = parser.parse_args()
    
    # Find snapshots
    print(f"Searching for snapshots in: {args.output_folder}")
    all_snapshots = find_snapshots(args.output_folder)
    
    if len(all_snapshots) == 0:
        print("ERROR: No snapshots found!")
        return 1
    
    print(f"Found {len(all_snapshots)} snapshots")
    
    # Filter by user selection
    if args.snapshots:
        selected_nums = [n.strip() for n in args.snapshots.split(',')]
        selected_snapshots = []
        for snap in all_snapshots:
            snap_num = os.path.basename(snap).split('_')[-1]
            
            try:
                snap_num_int = int(snap_num)
                for sel in selected_nums:
                    try:
                        if int(sel) == snap_num_int:
                            selected_snapshots.append(snap)
                            break
                    except ValueError:
                        if sel == snap_num:
                            selected_snapshots.append(snap)
                            break
            except ValueError:
                if snap_num in selected_nums:
                    selected_snapshots.append(snap)
        
        snapshots = selected_snapshots
        
        if len(snapshots) == 0:
            print(f"WARNING: No snapshots matched selection: {selected_nums}")
            return 1
    else:
        snapshots = all_snapshots
    
    print(f"Analyzing {len(snapshots)} snapshot(s)\n")
    
    # Analyze each snapshot
    all_results = []
    for snap_idx, snap in enumerate(snapshots):
        print(f"\n[Snapshot {snap_idx+1}/{len(snapshots)}]")
        try:
            results = analyze_snapshot(snap, args.output_dir)
            if results is not None:
                all_results.append(results)
        except Exception as e:
            print(f"ERROR analyzing {snap}: {e}")
            import traceback
            traceback.print_exc()
    
    # ========================================
    # TIME EVOLUTION PLOT (if multiple snapshots)
    # ========================================
    
    if len(all_results) > 1:
        print("\n" + "="*60)
        print("Creating time evolution plot...")
        print("="*60)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        times = [r['sim_time'] for r in all_results]
        redshifts = [r['redshift'] for r in all_results]
        
        # Fractal dimension evolution
        ax = axes[0, 0]
        D_values = [r['correlation']['D'] if r['correlation'] else np.nan 
                   for r in all_results]
        ax.plot(times, D_values, 'o-', markersize=8, linewidth=2, color='#d62728')
        ax.axhspan(2.3, 2.7, alpha=0.2, color='green', label='ISM-like range')
        ax.set_xlabel('Scale Factor a', fontsize=11)
        ax.set_ylabel('Fractal Dimension D', fontsize=11)
        ax.set_title('Dust Fractal Dimension Evolution', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Total dust mass evolution
        ax = axes[0, 1]
        dust_masses = [r['total_dust_mass'] for r in all_results]
        ax.semilogy(times, dust_masses, 'o-', markersize=8, linewidth=2, color='#ff7f0e')
        ax.set_xlabel('Scale Factor a', fontsize=11)
        ax.set_ylabel('Total Dust Mass [M☉]', fontsize=11)
        ax.set_title('Dust Mass Evolution', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Dust/Gas ratio evolution
        ax = axes[1, 0]
        dust_gas_ratios = [r['total_dust_mass']/r['total_gas_mass'] for r in all_results]
        ax.semilogy(times, dust_gas_ratios, 'o-', markersize=8, linewidth=2, color='#2ca02c')
        ax.set_xlabel('Scale Factor a', fontsize=11)
        ax.set_ylabel('Dust/Gas Mass Ratio', fontsize=11)
        ax.set_title('Dust/Gas Ratio Evolution', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Number of dust particles
        ax = axes[1, 1]
        n_dusts = [r['n_dust'] for r in all_results]
        ax.semilogy(times, n_dusts, 'o-', markersize=8, linewidth=2, color='#9467bd')
        ax.set_xlabel('Scale Factor a', fontsize=11)
        ax.set_ylabel('Number of Dust Particles', fontsize=11)
        ax.set_title('Dust Particle Count Evolution', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        outfile = os.path.join(args.output_dir, 'dust_evolution.png')
        plt.savefig(outfile, dpi=150, bbox_inches='tight')
        print(f"Saved: {outfile}")
        plt.close()
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Snapshot':<15} {'Time':>8} {'z':>6} {'N_dust':>8} {'D':>6} {'M_dust':>12}")
    print("-"*60)
    
    for res in all_results:
        snap_name = res['snapshot'].replace('snapshot_', '')
        time_val = res['sim_time']
        z_val = res['redshift']
        n_dust = res['n_dust']
        D = res['correlation']['D'] if res['correlation'] else np.nan
        M_dust = res['total_dust_mass']
        
        print(f"{snap_name:<15} {time_val:>8.4f} {z_val:>6.2f} {n_dust:>8} "
              f"{D:>6.2f} {M_dust:>12.3e}")
    
    print(f"\nAll plots saved to: {args.output_dir}/")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
