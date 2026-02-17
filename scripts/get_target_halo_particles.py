# halo_utils.py
import h5py
import numpy as np
import glob

# Define common fields for each particle type
PARTICLE_TYPE_FIELDS = {
    0: ['Coordinates', 'Masses', 'Velocities', 'Density', 'Metallicity', 
        'InternalEnergy', 'StarFormationRate'],  # Gas
    1: ['Coordinates', 'Masses', 'Velocities', 'ParticleIDs'],  # DM (high-res)
    2: ['Coordinates', 'Masses', 'Velocities', 'ParticleIDs'],  # DM (low-res)
    4: ['Coordinates', 'Masses', 'Velocities', 'Metallicity', 
        'StellarFormationTime', 'ParticleIDs'],  # Stars
    5: ['Coordinates', 'Masses', 'Velocities', 'ParticleIDs'],  # Black holes
    6: ['Coordinates', 'Masses', 'DustRadius', 'GrainType', 
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
            fields = PARTICLE_TYPE_FIELDS.get(ptype, ['Coordinates', 'Masses', 'Velocities'])
            
            # Check which fields actually exist
            fields = _check_available_fields(snapshot_base, ptype, fields)
            
            result[pname] = _extract_particles(snapshot_base, ptype, offset, length, fields)
            
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
            npart = int(f['Header'].attrs['NumPart_ThisFile'][parttype])  # Cast to int!
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
