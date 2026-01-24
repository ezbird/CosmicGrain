#!/usr/bin/env python3
"""
Comprehensive diagnostic check for completed parent simulation
Verifies outputs, checks for issues, and prepares for halo selection
"""

import glob
import h5py
import numpy as np
import os
import sys

def check_output_directory(output_dir):
    """Check if output directory exists and has expected structure"""
    print("\n" + "="*70)
    print("STEP 1: OUTPUT DIRECTORY CHECK")
    print("="*70)
    
    if not os.path.exists(output_dir):
        print(f"❌ ERROR: Output directory not found: {output_dir}")
        return False
    
    print(f"✅ Output directory exists: {output_dir}")
    
    # Check for key files
    info_file = os.path.join(output_dir, "info.txt")
    if os.path.exists(info_file):
        print(f"✅ Found info.txt")
        # Check for completion marker
        with open(info_file, 'r') as f:
            lines = f.readlines()
            if any('endrun' in line.lower() or 'simulation completed' in line.lower() for line in lines[-50:]):
                print(f"✅ Simulation completed normally (found endrun marker)")
            else:
                print(f"⚠️  WARNING: No completion marker found - check if sim finished")
    else:
        print(f"⚠️  WARNING: info.txt not found")
    
    return True


def find_snapshots(output_dir):
    """Find all snapshot files"""
    print("\n" + "="*70)
    print("STEP 2: SNAPSHOT INVENTORY")
    print("="*70)
    
    # Look for snapshots (various naming conventions)
    patterns = [
        os.path.join(output_dir, "snapshot_*.hdf5"),
        os.path.join(output_dir, "snap_*.hdf5"),
        os.path.join(output_dir, "snapdir_*/snapshot_*.hdf5"),
    ]
    
    snapshots = []
    for pattern in patterns:
        snapshots.extend(glob.glob(pattern))
    
    if not snapshots:
        print(f"❌ ERROR: No snapshots found!")
        print(f"   Searched patterns: {patterns}")
        return []
    
    snapshots = sorted(set(snapshots))
    print(f"✅ Found {len(snapshots)} snapshots")
    print(f"   First: {os.path.basename(snapshots[0])}")
    print(f"   Last:  {os.path.basename(snapshots[-1])}")
    
    return snapshots


def check_snapshot_health(snapshot_file):
    """Check individual snapshot for issues"""
    try:
        with h5py.File(snapshot_file, 'r') as f:
            # Get header info
            header = f['Header'].attrs
            time = header['Time']
            redshift = header['Redshift']
            npart = header['NumPart_ThisFile']
            
            # Check particle counts
            total_particles = sum(npart)
            if total_particles == 0:
                return False, "No particles!", time, redshift
            
            # Check for NaN/inf in positions
            if 'PartType1' in f and 'Coordinates' in f['PartType1']:
                pos = f['PartType1/Coordinates'][:]
                if not np.all(np.isfinite(pos)):
                    return False, "NaN/inf in positions!", time, redshift
            
            if 'PartType0' in f and 'Coordinates' in f['PartType0']:
                pos = f['PartType0/Coordinates'][:]
                if not np.all(np.isfinite(pos)):
                    return False, "NaN/inf in gas positions!", time, redshift
            
            return True, "OK", time, redshift
            
    except Exception as e:
        return False, f"Error reading: {e}", 0, 0


def check_all_snapshots(snapshots):
    """Check all snapshots for corruption"""
    print("\n" + "="*70)
    print("STEP 3: SNAPSHOT HEALTH CHECK")
    print("="*70)
    
    print(f"Checking {len(snapshots)} snapshots for corruption...")
    
    issues = []
    redshifts = []
    
    for i, snap in enumerate(snapshots):
        ok, msg, time, z = check_snapshot_health(snap)
        if not ok:
            print(f"❌ {os.path.basename(snap)}: {msg}")
            issues.append((snap, msg))
        redshifts.append(z)
    
    if not issues:
        print(f"✅ All {len(snapshots)} snapshots are healthy!")
    else:
        print(f"⚠️  WARNING: {len(issues)} snapshots have issues:")
        for snap, msg in issues:
            print(f"   - {os.path.basename(snap)}: {msg}")
    
    # Check redshift coverage
    print(f"\n📊 Redshift coverage:")
    print(f"   z_max = {max(redshifts):.4f}")
    print(f"   z_min = {min(redshifts):.4f}")
    print(f"   Δz_mean = {np.mean(np.diff(sorted(redshifts))):.4f}")
    
    if min(redshifts) > 0.7:
        print(f"⚠️  WARNING: Stopped at z={min(redshifts):.2f} (wanted z~0.5)")
    elif min(redshifts) < 0.3:
        print(f"⚠️  NOTE: Ran to z={min(redshifts):.2f} (lower than planned z~0.5)")
    else:
        print(f"✅ Good stopping point: z={min(redshifts):.2f}")
    
    return len(issues) == 0, redshifts


def find_halo_catalogs(output_dir):
    """Find FOF/SUBFIND catalogs"""
    print("\n" + "="*70)
    print("STEP 4: HALO CATALOG CHECK")
    print("="*70)
    
    patterns = [
        os.path.join(output_dir, "fof_subhalo_tab_*.hdf5"),
        os.path.join(output_dir, "groups_*.hdf5"),
        os.path.join(output_dir, "fof_tab_*.hdf5"),
    ]
    
    catalogs = []
    for pattern in patterns:
        catalogs.extend(glob.glob(pattern))
    
    catalogs = sorted(set(catalogs))
    
    if not catalogs:
        print(f"❌ ERROR: No halo catalogs found!")
        print(f"   Did you enable FOF+SUBFIND in Config.sh?")
        return []
    
    print(f"✅ Found {len(catalogs)} halo catalogs")
    print(f"   First: {os.path.basename(catalogs[0])}")
    print(f"   Last:  {os.path.basename(catalogs[-1])}")
    
    return catalogs


def check_halo_catalog(catalog_file):
    """Check halo catalog quality"""
    try:
        with h5py.File(catalog_file, 'r') as f:
            if 'Group' not in f:
                return 0, 0, "No Group dataset"
            
            ngroups = len(f['Group/GroupMass'][:])
            masses = f['Group/GroupMass'][:] * 1e10  # Convert to Msun
            
            # Count halos in target mass range (2-5 × 10^12 Msun)
            target_mass_min = 2e12
            target_mass_max = 5e12
            n_target = np.sum((masses >= target_mass_min) & (masses <= target_mass_max))
            
            z = f['Header'].attrs['Redshift']
            
            return ngroups, n_target, f"z={z:.4f}"
            
    except Exception as e:
        return 0, 0, f"Error: {e}"


def check_all_catalogs(catalogs):
    """Check all halo catalogs"""
    print("\n" + "="*70)
    print("STEP 5: HALO CATALOG QUALITY")
    print("="*70)
    
    print(f"Analyzing {len(catalogs)} halo catalogs...")
    print(f"\nTarget mass range: 2-5 × 10^12 Msun (MW-mass)")
    print(f"{'Catalog':<30s} {'Total Halos':>12s} {'Target Mass':>12s} {'Info':>15s}")
    print("-" * 72)
    
    good_catalogs = []
    
    for cat in catalogs[-10:]:  # Check last 10
        ngroups, n_target, info = check_halo_catalog(cat)
        status = "✅" if n_target > 5 else "⚠️ "
        print(f"{status} {os.path.basename(cat):<28s} {ngroups:>12d} {n_target:>12d} {info:>15s}")
        
        if n_target >= 5:
            good_catalogs.append(cat)
    
    if good_catalogs:
        print(f"\n✅ {len(good_catalogs)} catalogs have 5+ target-mass halos")
        print(f"   Best for zoom selection: {os.path.basename(good_catalogs[-1])}")
    else:
        print(f"\n⚠️  WARNING: Few halos in target mass range")
        print(f"   May need to adjust mass selection criteria")
    
    return good_catalogs


def check_cosmology_consistency(snapshot_file, expected):
    """Verify cosmology matches expected values"""
    print("\n" + "="*70)
    print("STEP 6: COSMOLOGY VERIFICATION")
    print("="*70)
    
    try:
        with h5py.File(snapshot_file, 'r') as f:
            header = f['Header'].attrs
            
            params_to_check = {
                'Omega0': expected.get('Omega0', 0.3158),
                'OmegaLambda': expected.get('OmegaLambda', 0.6842),
                'OmegaBaryon': expected.get('OmegaBaryon', 0.04936),
                'HubbleParam': expected.get('HubbleParam', 0.6732),
                'BoxSize': expected.get('BoxSize', 50.0),
            }
            
            print(f"Checking snapshot: {os.path.basename(snapshot_file)}")
            print(f"{'Parameter':<20s} {'Expected':>12s} {'Actual':>12s} {'Match':>8s}")
            print("-" * 52)
            
            all_good = True
            for param, expected_val in params_to_check.items():
                actual_val = header.get(param, -999)
                match = abs(actual_val - expected_val) < 1e-4
                status = "✅" if match else "❌"
                print(f"{param:<20s} {expected_val:>12.6f} {actual_val:>12.6f} {status:>8s}")
                if not match:
                    all_good = False
            
            if all_good:
                print(f"\n✅ All cosmology parameters match expected values!")
            else:
                print(f"\n❌ WARNING: Cosmology mismatch detected!")
                print(f"   This could cause issues with zoom IC generation!")
            
            return all_good
            
    except Exception as e:
        print(f"❌ ERROR checking cosmology: {e}")
        return False


def summary_report(output_dir, snapshots, catalogs):
    """Print final summary"""
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    # Estimate runtime
    if os.path.exists(os.path.join(output_dir, "info.txt")):
        with open(os.path.join(output_dir, "info.txt"), 'r') as f:
            lines = f.readlines()
            # Look for timing info
            for line in lines:
                if 'total time' in line.lower() or 'elapsed time' in line.lower():
                    print(f"⏱️  {line.strip()}")
    
    # Size check
    total_size = 0
    for snap in snapshots:
        total_size += os.path.getsize(snap)
    for cat in catalogs:
        total_size += os.path.getsize(cat)
    
    print(f"\n💾 Storage:")
    print(f"   Snapshots: {len(snapshots)} files")
    print(f"   Catalogs:  {len(catalogs)} files")
    print(f"   Total:     {total_size / 1e9:.1f} GB")
    
    print(f"\n📋 Next Steps:")
    if catalogs:
        best_catalog = sorted(catalogs)[-1]
        print(f"   1. Select zoom halo:")
        print(f"      python get_best_halo_newer.py {best_catalog} \\")
        print(f"        --mmin 50 --mmax 300 --nmax 20 --csv best_halos.csv")
        print(f"")
        print(f"   2. Pick isolated halo (isolation >15 Mpc/h)")
        print(f"")
        print(f"   3. Update zoom MUSIC2 config with halo position")
        print(f"")
        print(f"   4. Generate zoom IC:")
        print(f"      ./MUSIC zoom_from_parent.conf")
    else:
        print(f"   ⚠️  No halo catalogs found - check FOF/SUBFIND settings")


def main():
    if len(sys.argv) < 2:
        print("Usage: python check_parent_run.py <output_directory>")
        print("\nExample:")
        print("  python check_parent_run.py ./output_parent_music_128")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    
    print("\n" + "="*70)
    print("PARENT SIMULATION DIAGNOSTIC CHECK")
    print("="*70)
    print(f"Output directory: {output_dir}")
    
    # Expected cosmology (Planck 2018)
    expected_cosmo = {
        'Omega0': 0.3158,
        'OmegaLambda': 0.6842,
        'OmegaBaryon': 0.04936,
        'HubbleParam': 0.6732,
        'BoxSize': 50.0,
    }
    
    # Step-by-step checks
    if not check_output_directory(output_dir):
        sys.exit(1)
    
    snapshots = find_snapshots(output_dir)
    if not snapshots:
        print("\n❌ CRITICAL: No snapshots found! Simulation may have failed.")
        sys.exit(1)
    
    snap_ok, redshifts = check_all_snapshots(snapshots)
    
    catalogs = find_halo_catalogs(output_dir)
    
    if catalogs:
        good_catalogs = check_all_catalogs(catalogs)
    else:
        good_catalogs = []
    
    # Check cosmology on last snapshot
    if snapshots:
        check_cosmology_consistency(snapshots[-1], expected_cosmo)
    
    summary_report(output_dir, snapshots, catalogs)
    
    # Final verdict
    print("\n" + "="*70)
    if snap_ok and catalogs and good_catalogs:
        print("🎉 SUCCESS! Parent simulation completed successfully!")
        print("   Ready to select zoom target halo.")
    elif snap_ok and not catalogs:
        print("⚠️  PARTIAL SUCCESS: Snapshots OK, but no halo catalogs.")
        print("   You can still use snapshots, but need to run FOF separately.")
    else:
        print("❌ ISSUES DETECTED: Check warnings above.")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
