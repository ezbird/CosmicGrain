# halo_utils.py
import h5py
import numpy as np
import glob

# Define common fields for each particle type
PARTICLE_TYPE_FIELDS = {
    0: ['Coordinates', 'Masses', 'Velocities', 'Density', 'Metallicity', 
        'InternalEnergy', 'StarFormationRate'],  # Gas
    1: ['Coordinates', 'Velocities', 'ParticleIDs'],  # DM (high-res) - no Masses field!
    2: ['Coordinates', 'Velocities', 'ParticleIDs'],  # DM (low-res)
    4: ['Coordinates', 'Masses', 'Velocities', 'Metallicity', 
        'StellarFormationTime', 'ParticleIDs'],  # Stars
    5: ['Coordinates', 'Masses', 'Velocities', 'ParticleIDs'],  # Black holes
    6: ['Coordinates', 'Masses', 'GrainRadius', 'GrainType', 
        'DustTemperature', 'CarbonFraction', 'Velocities']  # Dust
}

def load_target_halo(catalog_file, snapshot_base, particle_types='all', 
                     output_file=None, verbose=True):
    """
    Extract target halo particles.
    
    Parameters:
    -----------
    catalog_file : str
        Path to fof_subhalo_tab file
    snapshot_base : str
        Base path to snapshot (e.g., 'snapshot_049')
    particle_types : list or 'all'
        Which particle types to extract. Default 'all' extracts all present types.
        Can specify list like [0, 1, 4, 6] for gas, DM, stars, dust
    output_file : str, optional
        If provided, save results to this HDF5 file
    verbose : bool
        Print extraction info
    
    Returns:
    --------
    dict with keys: 
        'halo_info': dict of halo properties
        'gas', 'dm', 'dm2', 'stars', 'bh', 'dust': particle data (if present)
    """
    
    # Load catalog
    cat = h5py.File(catalog_file, 'r')
    subhalo_mass = cat['Subhalo']['SubhaloMass'][:]
    target_id = np.argmax(subhalo_mass)
    
    halo_info = {
        'id': target_id,
        'mass': subhalo_mass[target_id],
        'position': cat['Subhalo']['SubhaloPos'][target_id],
        'velocity': cat['Subhalo']['SubhaloVel'][target_id],
        'halfmass_rad': cat['Subhalo']['SubhaloHalfmassRad'][target_id],
        'vmax': cat['Subhalo']['SubhaloVmax'][target_id],
        'spin': cat['Subhalo']['SubhaloSpin'][target_id]
    }
    
    if verbose:
        print(f"Target subhalo {target_id}")
        print(f"  Mass: {halo_info['mass']:.2e}")
        print(f"  Position: {halo_info['position']}")
        print(f"  Halfmass radius: {halo_info['halfmass_rad']:.2f} kpc")
        print()
    
    result = {'halo_info': halo_info}
    
    # Get DM particle masses from header (same for all DM particles)
    dm_masses = _get_dm_particle_masses(snapshot_base)
    
    # Determine which particle types to extract
    if particle_types == 'all':
        particle_types = range(7)  # Check all types 0-6
    
    # Map particle type numbers to names
    ptype_names = {0: 'gas', 1: 'dm', 2: 'dm2', 4: 'stars', 5: 'bh', 6: 'dust'}
    
    for ptype in particle_types:
        offset = int(cat['Subhalo']['SubhaloOffsetType'][target_id, ptype])
        length = int(cat['Subhalo']['SubhaloLenType'][target_id, ptype])
        
        if length > 0:
            pname = ptype_names.get(ptype, f'parttype{ptype}')
            
            if verbose:
                print(f"Extracting {length} {pname} particles (PartType{ptype})...")
            
            # Get fields for this particle type
            fields = PARTICLE_TYPE_FIELDS.get(ptype, ['Coordinates', 'Velocities'])
            
            # Check which fields actually exist
            fields = _check_available_fields(snapshot_base, ptype, fields)
            
            result[pname] = _extract_particles(snapshot_base, ptype, offset, length, fields)
            
            # For DM particles, add mass array manually from header
            if ptype in [1, 2] and dm_masses[ptype] > 0:
                result[pname]['Masses'] = np.full(length, dm_masses[ptype])
            
            if verbose:
                coords = result[pname]['Coordinates']
                r = np.sqrt(np.sum((coords - halo_info['position'])**2, axis=1))
                total_mass = result[pname]['Masses'].sum() if 'Masses' in result[pname] else 0
                print(f"  Radial range: {r.min():.2f} to {r.max():.2f} kpc")
                print(f"  Total mass: {total_mass:.2e}")
                print()
    
    cat.close()
    
    # Optionally save
    if output_file:
        if verbose:
            print(f"Saving to {output_file}...")
        _save_to_hdf5(result, output_file)
        if verbose:
            print("Done!")
    
    return result

# Unit conversions (Gadget code units)
UNIT_MASS = 1e10  # M_sun
UNIT_LENGTH = 1.0  # kpc (already in kpc)
UNIT_VELOCITY = 1.0  # km/s (already in km/s)

def convert_to_physical_units(data, mass_in_msun=True):
    """
    Convert from Gadget code units to physical units.
    
    Parameters:
    -----------
    data : dict
        Particle data dictionary
    mass_in_msun : bool
        If True, convert masses to M_sun. If False, leave in 1e10 M_sun
    
    Returns:
    --------
    dict with converted data (modifies in place and returns)
    """
    if 'Masses' in data and mass_in_msun:
        data['Masses'] = data['Masses'] * UNIT_MASS
    
    # Coordinates and velocities are already in kpc and km/s
    
    return data

def extract_dust_spatially(snapshot_base, halo_center, radius_kpc=None, verbose=True):
    """
    Extract dust particles within a radius of the halo center.
    Use this when Subfind doesn't assign dust to subhalos.
    
    Parameters:
    -----------
    snapshot_base : str
        Base path to snapshot
    halo_center : array (3,)
        Halo center position
    radius_kpc : float, optional
        Extraction radius in kpc. If None, extracts all dust and computes radii.
    verbose : bool
        Print info
    
    Returns:
    --------
    dict with dust particle data
    """
    files = sorted(glob.glob(f'{snapshot_base}.*.hdf5'))
    
    if verbose:
        print(f"Spatially extracting dust particles...")
    
    all_coords = []
    all_masses = []
    all_dust_radius = []
    all_grain_type = []
    all_dust_temp = []
    all_carbon_frac = []
    all_velocities = []
    all_formation_time = []  # ADD THIS
    
    total_dust = 0
    
    for fname in files:
        with h5py.File(fname, 'r') as f:
            if 'PartType6' not in f:
                continue
            
            dust = f['PartType6']
            npart = len(dust['Coordinates'])
            total_dust += npart
            
            coords = dust['Coordinates'][:]
            
            # Compute distance from halo center
            r = np.sqrt(np.sum((coords - halo_center)**2, axis=1))
            
            # Apply radius cut if specified
            if radius_kpc is not None:
                mask = r < radius_kpc
            else:
                mask = np.ones(len(r), dtype=bool)
            
            if mask.sum() > 0:
                all_coords.append(coords[mask])
                all_masses.append(dust['Masses'][:][mask])
                all_dust_radius.append(dust['GrainRadius'][:][mask])
                all_grain_type.append(dust['GrainType'][:][mask])
                all_dust_temp.append(dust['DustTemperature'][:][mask])
                all_carbon_frac.append(dust['CarbonFraction'][:][mask])
                all_velocities.append(dust['Velocities'][:][mask])
                
                # ADD THIS - extract DustFormationTime if it exists
                if 'DustFormationTime' in dust:
                    all_formation_time.append(dust['DustFormationTime'][:][mask])
                elif 'StellarFormationTime' in dust:
                    all_formation_time.append(dust['StellarFormationTime'][:][mask])
    
    if verbose:
        print(f"  Total dust particles in snapshot: {total_dust}")
    
    if len(all_coords) == 0:
        if verbose:
            print("  No dust particles found!")
        return None
    
    result = {
        'Coordinates': np.vstack(all_coords),
        'Masses': np.concatenate(all_masses),
        'GrainRadius': np.concatenate(all_dust_radius),
        'GrainType': np.concatenate(all_grain_type),
        'DustTemperature': np.concatenate(all_dust_temp),
        'CarbonFraction': np.concatenate(all_carbon_frac),
        'Velocities': np.vstack(all_velocities)
    }
    
    # ADD THIS - include formation time if we extracted it
    if len(all_formation_time) > 0:
        result['DustFormationTime'] = np.concatenate(all_formation_time)
    
    r = np.sqrt(np.sum((result['Coordinates'] - halo_center)**2, axis=1))
    
    if verbose:
        if radius_kpc is not None:
            print(f"  Extracted {len(r)} dust particles within {radius_kpc} kpc")
        else:
            print(f"  Extracted {len(r)} dust particles")
        print(f"  Radial range: {r.min():.2f} to {r.max():.2f} kpc")
        print(f"  Total dust mass: {result['Masses'].sum():.2e}")
    
    return result

def _get_dm_particle_masses(snapshot_base):
    """Get DM particle masses from snapshot header."""
    files = sorted(glob.glob(f'{snapshot_base}.*.hdf5'))
    with h5py.File(files[0], 'r') as f:
        mass_table = f['Header'].attrs['MassTable']
    return mass_table

def _check_available_fields(snapshot_base, parttype, requested_fields):
    """Check which requested fields actually exist in the snapshot."""
    files = sorted(glob.glob(f'{snapshot_base}.*.hdf5'))
    if not files:
        return requested_fields
    
    with h5py.File(files[0], 'r') as f:
        ptype_key = f'PartType{parttype}'
        if ptype_key not in f:
            return []
        available = list(f[ptype_key].keys())
    
    return [field for field in requested_fields if field in available]

def _extract_particles(snapshot_base, parttype, global_offset, length, fields):
    """Helper to extract particles across multiple files."""
    files = sorted(glob.glob(f'{snapshot_base}.*.hdf5'))
    
    cumulative_counts = [0]
    for fname in files:
        with h5py.File(fname, 'r') as f:
            npart = int(f['Header'].attrs['NumPart_ThisFile'][parttype])
            cumulative_counts.append(cumulative_counts[-1] + npart)
    
    global_end = global_offset + length
    all_data = {field: [] for field in fields}
    
    for i, fname in enumerate(files):
        file_start = cumulative_counts[i]
        file_end = cumulative_counts[i+1]
        
        if file_end <= global_offset or file_start >= global_end:
            continue
        
        local_start = max(0, global_offset - file_start)
        local_end = min(file_end - file_start, global_end - file_start)
        
        with h5py.File(fname, 'r') as f:
            ptype = f[f'PartType{parttype}']
            for field in fields:
                all_data[field].append(ptype[field][local_start:local_end])
    
    result = {}
    for field in fields:
        if len(all_data[field]) == 0:
            continue
        if len(all_data[field][0].shape) == 1:
            result[field] = np.concatenate(all_data[field])
        else:
            result[field] = np.vstack(all_data[field])
    
    return result

def _save_to_hdf5(data, filename):
    """Save extracted data to HDF5."""
    with h5py.File(filename, 'w') as f:
        # Save halo info as attributes
        for key, val in data['halo_info'].items():
            f.attrs[key] = val
        
        # Map back to PartType naming
        ptype_map = {'gas': 0, 'dm': 1, 'dm2': 2, 'stars': 4, 'bh': 5, 'dust': 6}
        
        for name, ptype_num in ptype_map.items():
            if name in data:
                grp = f.create_group(f'PartType{ptype_num}')
                for key, val in data[name].items():
                    grp.create_dataset(key, data=val)

def compute_radial_distance(coords, center):
    """Compute radial distance from center."""
    return np.sqrt(np.sum((coords - center)**2, axis=1))

def compute_radial_profile(coords, masses, center, rbins):
    """
    Compute radial mass profile.
    
    Parameters:
    -----------
    coords : array (N, 3)
        Particle coordinates
    masses : array (N,)
        Particle masses
    center : array (3,)
        Halo center
    rbins : array
        Radial bin edges
    
    Returns:
    --------
    r_centers : array
        Bin centers
    mass_profile : array
        Mass in each bin
    """
    r = compute_radial_distance(coords, center)
    mass_profile, _ = np.histogram(r, bins=rbins, weights=masses)
    r_centers = 0.5 * (rbins[1:] + rbins[:-1])
    return r_centers, mass_profile
