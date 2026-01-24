#!/usr/bin/env python3
import os, glob, h5py, numpy as np
import argparse, math, sys
import matplotlib.pyplot as plt
import json
from pathlib import Path
import traceback

MSUN_IN_G  = 1.98847e33
CM_PER_MPC = 3.085678e24
CM_PER_KPC = 3.085678e21

# ----------------------------- Observational Models ----------------------------- #
def stellar_mass_halo_mass_relation(Mhalo, z=0):
    """
    Stellar-to-halo mass relation from Behroozi et al. (2013)
    ApJ 770, 57 - "The Average Star Formation Histories of Galaxies in Dark Matter Halos"
    """
    log_Mhalo = np.log10(Mhalo)
    
    if z < 6:
        a = 1.0 / (1.0 + z)
        epsilon_0 = 0.02
        M1 = 10**(11.5 + 0.3*(a-1))
        alpha = -1.5
        ratio = epsilon_0 * (Mhalo/M1)**alpha / (1 + (Mhalo/M1)**alpha)
        z_factor = (1 + z)**(-0.5)
        return ratio * z_factor
    else:
        return 0.001 * (1 + z)**(-1.0)

def metallicity_evolution(Mhalo, z):
    """Gas-phase metallicity from Ma et al. (2016) MNRAS 456, 2140"""
    log_Mhalo = np.log10(Mhalo)
    Z_sun = 0.0142
    
    if log_Mhalo > 11.5:
        log_Z_ratio = -1.0 + 0.4 * (log_Mhalo - 12.0)
    else:
        log_Z_ratio = -2.0 + 0.2 * (log_Mhalo - 11.0)
    
    if z > 3:
        z_evolution = -0.5 * np.log10(1 + z/3.0)
    else:
        z_evolution = 0.0
    
    Z_over_Zsun = 10**(log_Z_ratio + z_evolution)
    return Z_over_Zsun * Z_sun

def dust_to_metals_ratio(z):
    """Dust-to-metals from Rémy-Ruyer et al. (2014) A&A 563, A31"""
    if z > 10:
        return 0.01
    elif z > 5:
        return 0.05 * (6/(1+z))
    else:
        DTM_z0 = 0.3
        evolution = (1 + z)**(-0.6)
        return max(0.1, DTM_z0 * evolution)

def star_formation_timescale(Mhalo, z):
    """SFR timescale from Tacconi et al. (2018) ApJ 853, 179"""
    log_Mhalo = np.log10(Mhalo)
    
    if log_Mhalo > 12:
        tau_base = 3.0
    elif log_Mhalo > 11.5:
        tau_base = 2.0
    else:
        tau_base = 1.0
    
    if z > 3:
        tau_z = tau_base * (1 + z/5.0)**(-0.8)
    else:
        tau_z = tau_base * (1 + z)**(-0.3)
    
    return max(0.5, min(10.0, tau_z))

# ----------------------------- Utility functions ----------------------------- #
def _get_scale_length(header, target="mpc"):
    ul = float(header.get("UnitLength_in_cm", CM_PER_MPC))
    if target == "mpc":  return ul / CM_PER_MPC
    if target == "kpc":  return ul / CM_PER_KPC
    return 1.0

def _mass_unit_to_Msun(header):
    um = float(header.get("UnitMass_in_g", 1.989e43))
    return um / MSUN_IN_G

def _mass_array(f, pgroup, mass_to_Msun):
    if f"{pgroup}/Coordinates" not in f:
        return None
    N = f[f"{pgroup}/Coordinates"].shape[0]
    if f"{pgroup}/Masses" in f:
        return np.array(f[f"{pgroup}/Masses"]) * mass_to_Msun
    mt = f["Header"].attrs.get("MassTable", None)
    if mt is not None and len(mt) >= (int(pgroup[-1])+1) and mt[int(pgroup[-1])] > 0:
        return np.full(N, float(mt[int(pgroup[-1])]) * mass_to_Msun, dtype=np.float64)
    return None

def _coords_array(f, pgroup, len_scale):
    if f"{pgroup}/Coordinates" not in f:
        return None
    return np.array(f[f"{pgroup}/Coordinates"]) * len_scale

def _roi_mask(coords, center, radius, box):
    if coords is None:
        return None
    d = coords - center[None,:]
    d -= np.round(d/box) * box
    r = np.sqrt((d*d).sum(axis=1))
    return r <= radius

def _gas_metallicity_array(f, pgroup="PartType0"):
    for name in (f"{pgroup}/Metallicity", f"{pgroup}/GFM_Metallicity"):
        if name in f:
            return np.array(f[name])
    if f"{pgroup}/GFM_Metals" in f:
        arr = np.array(f[f"{pgroup}/GFM_Metals"])
        if arr.ndim == 2:
            return np.sum(arr[:, 1:], axis=1)
    return None

def _gas_sfr_array(f, pgroup="PartType0"):
    for name in (f"{pgroup}/Sfr", f"{pgroup}/SFR", f"{pgroup}/StarFormationRate"):
        if name in f:
            return np.array(f[name])
    return None

# ----------------------------- Core analysis ----------------------------- #
def analyze_snapshot(snapshot_path, center=None, radius=None, debug=False):
    files = [snapshot_path] if os.path.isfile(snapshot_path) else sorted(glob.glob(os.path.join(snapshot_path, "*.hdf5")))
    
    if not files:
        return None
    
    with h5py.File(files[0], "r") as f0:
        hdr = f0["Header"].attrs
        len_scale = _get_scale_length(hdr, "mpc")
        mass_scale = _mass_unit_to_Msun(hdr)
        box = float(hdr["BoxSize"]) * len_scale
        time = float(hdr.get("Time", 1.0))
        try: 
            redshift = 1.0/time - 1.0
        except: 
            redshift = float(hdr.get("Redshift", 0.0))
    
    # Accumulators
    Mstar = Mgas = Mdust = Mdm = SFR_tot = metal_mass_gas = 0.0
    total_particles = {'gas': 0, 'star': 0, 'dust': 0, 'dm': 0}
    roi_particles = {'gas': 0, 'star': 0, 'dust': 0, 'dm': 0}
    
    for fp in files:
        with h5py.File(fp, "r") as f:
            type_mapping = {
                "PartType0": "gas", "PartType4": "star", "PartType6": "dust",
                "PartType1": "dm", "PartType2": "dm", "PartType3": "dm"
            }
            
            for pt, ptype_name in type_mapping.items():
                coords = _coords_array(f, pt, len_scale)
                masses = _mass_array(f, pt, mass_scale)
                if coords is None or masses is None:
                    continue
                
                total_particles[ptype_name] += len(masses)
                
                # Apply ROI mask if specified
                use_roi = center is not None and radius is not None
                if use_roi:
                    mask = _roi_mask(coords, np.asarray(center), radius, box)
                    if mask is None or not np.any(mask):
                        continue
                    masses_roi = masses[mask]
                    coords_roi = coords[mask]
                    roi_particles[ptype_name] += len(masses_roi)
                else:
                    masses_roi = masses
                    coords_roi = coords
                    roi_particles[ptype_name] += len(masses_roi)
                
                mass_sum = float(masses_roi.sum())
                
                # Replace the entire gas processing section (PartType0) with this heavily debugged version:

                if pt == "PartType0":  # Gas
                    print(f"    DEBUG: Starting gas processing, mass_sum={mass_sum}")
                    Mgas += mass_sum
                    
                    # SFR - need to recompute mask for full arrays
                    print(f"    DEBUG: About to get SFR array")
                    sfr = _gas_sfr_array(f, pt)
                    print(f"    DEBUG: SFR array type: {type(sfr)}, shape: {sfr.shape if sfr is not None else 'None'}")
                    
                    if sfr is not None:
                        #print(f"    DEBUG: SFR is not None, use_roi={use_roi}")
                        if use_roi:
                            #print(f"    DEBUG: Using ROI for SFR")
                            coords_full = _coords_array(f, pt, len_scale)
                            if coords_full is not None:
                                #print(f"    DEBUG: About to compute SFR mask")
                                sfr_mask = _roi_mask(coords_full, np.asarray(center), radius, box)
                                #print(f"    DEBUG: SFR mask computed, type: {type(sfr_mask)}")
                                if sfr_mask is not None and np.any(sfr_mask):
                                    #print(f"    DEBUG: Applying SFR mask")
                                    sfr_roi = sfr[sfr_mask]
                                    #SFR_tot += float(np.sum(sfr_roi))
                                    print(f"    DEBUG: SFR processing complete")
                        else:
                            #print(f"    DEBUG: Not using ROI for SFR")
                            SFR_tot += float(np.sum(sfr))
                    
                    # Metallicity - need to recompute mask for full arrays
                    #print(f"    DEBUG: About to get metallicity array")
                    Z = _gas_metallicity_array(f, pt)
                    #print(f"    DEBUG: Z array type: {type(Z)}, shape: {Z.shape if Z is not None else 'None'}")
                    
                    if Z is not None:
                        #print(f"    DEBUG: Z is not None, use_roi={use_roi}")
                        if use_roi:
                            #print(f"    DEBUG: Using ROI for metallicity")
                            coords_full = _coords_array(f, pt, len_scale)
                            masses_full = _mass_array(f, pt, mass_scale)
                            if coords_full is not None and masses_full is not None:
                                #print(f"    DEBUG: About to compute Z mask")
                                Z_mask = _roi_mask(coords_full, np.asarray(center), radius, box)
                                #print(f"    DEBUG: Z mask computed")
                                if Z_mask is not None and np.any(Z_mask):
                                    #print(f"    DEBUG: Applying Z mask")
                                    Z_roi = Z[Z_mask]
                                    masses_Z_roi = masses_full[Z_mask]
                                    metal_mass_gas += float(np.sum(masses_Z_roi * Z_roi))
                                    print(f"    DEBUG: Z processing complete with ROI")
                        else:
                            print(f"    DEBUG: Not using ROI for metallicity")
                            print(f"    DEBUG: masses shape: {masses.shape}, Z shape: {Z.shape}")
                            metal_mass_gas += float(np.sum(masses * Z))
                            print(f"    DEBUG: Z processing complete without ROI")
                    
                    print(f"    DEBUG: Gas processing complete")


                elif pt == "PartType4":  # Stars
                    Mstar += mass_sum
                elif pt == "PartType6":  # Dust
                    Mdust += mass_sum
                elif pt in ["PartType1", "PartType2", "PartType3"]:  # Dark matter
                    Mdm += mass_sum


            for pt, ptype_name in type_mapping.items():
                try:
                    coords = _coords_array(f, pt, len_scale)
                    masses = _mass_array(f, pt, mass_scale)
                    print(f"DEBUG: Processing {pt}, coords type: {type(coords)}, masses type: {type(masses)}")
                    
                    if coords is None or masses is None:
                        continue
                    
                    # Add explicit checks here
                    print(f"DEBUG: About to check masses array, shape: {masses.shape if hasattr(masses, 'shape') else 'no shape'}")
                    
                except Exception as e:
                    print(f"ERROR in {pt}: {e}")
                    print(traceback.format_exc())
                    raise e


    if debug:
        print(f"  Snapshot debug info:")
        print(f"    Total particles: {total_particles}")
        if center is not None:
            print(f"    ROI particles: {roi_particles}")
            print(f"    ROI center: {center}, radius: {radius} Mpc")
        print(f"    Masses: M_star={Mstar:.2e}, M_gas={Mgas:.2e}, M_dm={Mdm:.2e}")
    
    if Mdm == 0:
        if debug:
            print(f"    WARNING: No dark matter found!")
        return None
    
    return {
        'redshift': redshift,
        'time': time,
        'Mstar': Mstar,
        'Mgas': Mgas,
        'Mdust': Mdust,
        'Mdm': Mdm,
        'SFR': SFR_tot,
        'metal_mass_gas': metal_mass_gas,
        'box': box,
        'particles': roi_particles if center is not None else total_particles
    }

def track_evolution(snapdir, center=None, radius=None, output_dir="evolution_analysis", debug=False):
    # Find all snapshots
    snapshot_pattern = os.path.join(snapdir, "snapshot_*")
    snapshot_dirs = sorted(glob.glob(snapshot_pattern))
    
    if not snapshot_dirs:
        snapshot_pattern = os.path.join(snapdir, "snapshot_*.hdf5")
        snapshot_dirs = sorted(glob.glob(snapshot_pattern))
    
    if not snapshot_dirs:
        raise FileNotFoundError(f"No snapshots found in {snapdir}")
    
    print(f"Found {len(snapshot_dirs)} snapshots")
    if center is not None:
        print(f"Using ROI: center={center}, radius={radius} Mpc")
        print("This will select only high-resolution particles in your zoom region.")
    
    results = []
    
    for i, snap_path in enumerate(snapshot_dirs):
        print(f"Analyzing {os.path.basename(snap_path)}...")
        try:
            result = analyze_snapshot(snap_path, center, radius, debug=(debug and i < 3))
            if result is not None:
                results.append(result)
                if debug and i < 3:
                    print(f"    Analysis successful")
        except Exception as e:
            print(f"  WARNING: Failed to analyze {snap_path}: {e}")
            continue
    
    if not results:
        raise RuntimeError("No snapshots could be analyzed successfully")
    
    # Sort by redshift (highest to lowest)
    results.sort(key=lambda x: x['redshift'], reverse=True)
    
    # Compute ratios and observational expectations
    evolution_data = []
    
    for res in results:
        z = res['redshift']
        Mhalo = res['Mdm']
        
        # Compute simulation ratios
        sim_ratios = {}
        sim_ratios['Mstar_Mhalo'] = res['Mstar'] / Mhalo if Mhalo > 0 else 0
        sim_ratios['SFR_timescale_Gyr'] = (res['Mstar'] / res['SFR']) / 1e9 if res['SFR'] > 0 else np.inf
        sim_ratios['Z_gas'] = res['metal_mass_gas'] / res['Mgas'] if res['Mgas'] > 0 else 0
        sim_ratios['DTM'] = res['Mdust'] / res['metal_mass_gas'] if res['metal_mass_gas'] > 0 else 0
        sim_ratios['DGR'] = res['Mdust'] / res['Mgas'] if res['Mgas'] > 0 else 0
        
        # Compute observational expectations
        obs_expectations = {}
        obs_expectations['Mstar_Mhalo'] = stellar_mass_halo_mass_relation(Mhalo, z)
        obs_expectations['SFR_timescale_Gyr'] = star_formation_timescale(Mhalo, z)
        obs_expectations['Z_gas'] = metallicity_evolution(Mhalo, z)
        obs_expectations['DTM'] = dust_to_metals_ratio(z)
        
        # Compute ratios to expectations
        ratios_to_obs = {}
        for key in obs_expectations:
            if key in sim_ratios and obs_expectations[key] > 0:
                ratios_to_obs[key] = sim_ratios[key] / obs_expectations[key]
            else:
                ratios_to_obs[key] = np.nan
        
        evolution_data.append({
            'redshift': z,
            'time_Gyr': res['time'] * 13.8,
            'masses': {k: res[k] for k in ['Mstar', 'Mgas', 'Mdust', 'Mdm']},
            'SFR': res['SFR'],
            'simulation': sim_ratios,
            'observations': obs_expectations,
            'sim_over_obs': ratios_to_obs
        })
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save data
    with open(os.path.join(output_dir, "evolution_data.json"), "w") as f:
        json.dump(evolution_data, f, indent=2, default=str)
    
    # Create plots
    create_evolution_plots(evolution_data, output_dir)
    
    # Print summary
    print_evolution_summary(evolution_data)
    
    return evolution_data

def create_evolution_plots(data, output_dir):
    z_vals = [d['redshift'] for d in data]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Galaxy Evolution: Simulation vs Observations', fontsize=16)
    
    # Plot 1: Stellar-to-halo mass ratio
    ax = axes[0,0]
    sim_vals = [d['simulation']['Mstar_Mhalo'] for d in data]
    obs_vals = [d['observations']['Mstar_Mhalo'] for d in data]
    ax.semilogy(z_vals, sim_vals, 'o-', label='Simulation', color='red', markersize=4)
    ax.semilogy(z_vals, obs_vals, 's--', label='Behroozi+13', color='blue', markersize=4)
    ax.set_xlabel('Redshift')
    ax.set_ylabel('M*/M_halo')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Stellar-to-Halo Mass Ratio')
    ax.invert_xaxis()

    # Plot 2: SFR timescale
    ax = axes[0,1]
    sim_vals = [d['simulation']['SFR_timescale_Gyr'] for d in data]
    obs_vals = [d['observations']['SFR_timescale_Gyr'] for d in data]
    ax.semilogy(z_vals, np.clip(sim_vals, 0.1, 100), 'o-', label='Simulation', color='red', markersize=4)
    ax.semilogy(z_vals, obs_vals, 's--', label='Tacconi+18', color='blue', markersize=4)
    ax.set_xlabel('Redshift')
    ax.set_ylabel('SFR Timescale (Gyr)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Star Formation Timescale')
    ax.invert_xaxis()

    # Plot 3: Gas metallicity
    ax = axes[0,2]
    sim_vals = [d['simulation']['Z_gas'] for d in data]
    obs_vals = [d['observations']['Z_gas'] for d in data]
    ax.semilogy(z_vals, np.maximum(sim_vals, 1e-6), 'o-', label='Simulation', color='red', markersize=4)
    ax.semilogy(z_vals, obs_vals, 's--', label='Ma+16 (EAGLE)', color='blue', markersize=4)
    ax.set_xlabel('Redshift')
    ax.set_ylabel('Gas Metallicity (absolute)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Gas-Phase Metallicity')
    ax.invert_xaxis()

    # Plot 4: Dust-to-metals ratio
    ax = axes[1,0]
    sim_vals = [d['simulation']['DTM'] for d in data]
    obs_vals = [d['observations']['DTM'] for d in data]
    ax.plot(z_vals, sim_vals, 'o-', label='Simulation', color='red', markersize=4)
    ax.plot(z_vals, obs_vals, 's--', label='Rémy-Ruyer+14', color='blue', markersize=4)
    ax.set_xlabel('Redshift')
    ax.set_ylabel('Dust-to-Metals Ratio')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Dust-to-Metals Ratio')
    ax.set_ylim(0, 1)
    ax.invert_xaxis()

    # Plot 5: Ratios to observations
    ax = axes[1,1]
    colors = ['purple', 'orange', 'green', 'brown']
    for i, (key, label) in enumerate([('Mstar_Mhalo', 'M*/M_halo'), ('SFR_timescale_Gyr', 'SFR timescale'), 
                       ('Z_gas', 'Metallicity'), ('DTM', 'DTM')]):
        ratio_vals = [d['sim_over_obs'][key] for d in data if np.isfinite(d['sim_over_obs'][key])]
        z_vals_finite = [d['redshift'] for d in data if np.isfinite(d['sim_over_obs'][key])]
        if ratio_vals:
            ax.plot(z_vals_finite, ratio_vals, 'o-', label=label, color=colors[i], markersize=4)
    
    ax.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Perfect agreement')
    ax.fill_between([min(z_vals), max(z_vals)], [0.3, 0.3], [3.0, 3.0], alpha=0.2, color='gray', label='Factor of 3')
    ax.set_xlabel('Redshift')
    ax.set_ylabel('Simulation / Observations')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Simulation vs Observations')
    ax.set_yscale('log')
    ax.set_ylim(0.01, 100)
    ax.invert_xaxis()

    # Plot 6: Mass evolution
    ax = axes[1,2]
    colors = ['red', 'blue', 'green', 'black']
    for i, (key, label) in enumerate([('Mstar', 'Stars'), ('Mgas', 'Gas'), ('Mdust', 'Dust'), ('Mdm', 'Dark Matter')]):
        mass_vals = [d['masses'][key] for d in data]
        ax.semilogy(z_vals, mass_vals, 'o-', label=label, color=colors[i], markersize=4)
    ax.set_xlabel('Redshift')
    ax.set_ylabel('Mass (M_sun)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Mass Evolution')
    ax.invert_xaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'evolution_plots.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved to {output_dir}/evolution_plots.png")

def print_evolution_summary(data):
    print("\n" + "="*80)
    print("GALAXY EVOLUTION SUMMARY")
    print("="*80)
    
    z_range = f"z = {data[0]['redshift']:.2f} to {data[-1]['redshift']:.2f}"
    print(f"Redshift range: {z_range}")
    print(f"Number of snapshots: {len(data)}")
    
    # Current epoch (lowest redshift)
    current = data[-1]
    print(f"\nCurrent state (z = {current['redshift']:.2f}):")
    print(f"  M_halo = {current['masses']['Mdm']:.2e} M_sun")
    print(f"  M_star = {current['masses']['Mstar']:.2e} M_sun")
    print(f"  M_gas  = {current['masses']['Mgas']:.2e} M_sun")
    print(f"  M_dust = {current['masses']['Mdust']:.2e} M_sun")
    print(f"  SFR    = {current['SFR']:.2e} M_sun/yr")
    
    print(f"\nComparison to observations (Simulation / Expected):")
    for key, name in [('Mstar_Mhalo', 'M*/M_halo ratio'), 
                      ('SFR_timescale_Gyr', 'SFR timescale'),
                      ('Z_gas', 'Gas metallicity'),
                      ('DTM', 'Dust-to-metals ratio')]:
        ratio = current['sim_over_obs'][key]
        if np.isfinite(ratio):
            status = "✓" if 0.3 < ratio < 3.0 else "⚠"
            print(f"  {status} {name:20s}: {ratio:.4f}x")
        else:
            print(f"  ? {name:20s}: undefined")

def main():
    parser = argparse.ArgumentParser(description='Track galaxy evolution across snapshots')
    parser.add_argument('snapdir', help='Directory containing snapshots')
    parser.add_argument('--center', type=float, nargs=3, metavar=('X','Y','Z'),
                       help='ROI center coordinates in Mpc (for zoom region)')
    parser.add_argument('--radius', type=float, help='ROI radius in Mpc')
    parser.add_argument('--output', default='evolution_analysis',
                       help='Output directory for results')
    parser.add_argument('--debug', action='store_true',
                       help='Print debugging info for first few snapshots')
    
    args = parser.parse_args()
    
    center = np.array(args.center) if args.center else None
    
    try:
        evolution_data = track_evolution(args.snapdir, center, args.radius, args.output, args.debug)
        print(f"\nAnalysis complete! Results saved to {args.output}/")
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()