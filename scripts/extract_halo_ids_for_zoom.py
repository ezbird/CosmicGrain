#!/usr/bin/env python3
'''
By group (FoF) ID:

python3 extract_zoom_ids.py \
  --snapshot snapshots/snapshot_099*.hdf5 \
  --groupcat groupcats/groups_099.hdf5 \
  --group-id 123 \
  --kR 2.0 \
  --output ids_2r200_group123.txt

  OR

By subhalo (central/satellite) ID:

  python3 extract_zoom_ids.py \
  --snapshot snapshots/snapshot_099*.hdf5 \
  --groupcat groupcats/groups_099.hdf5 \
  --subhalo-id 456 \
  --kR 2.5 \
  --output ids_2p5r200_sub456.txt
'''
import argparse, os, sys, glob
import h5py
import numpy as np

def open_many(pattern_or_file):
    """Return a sorted list of files. Accepts a literal path or a glob pattern (e.g. snapshot_099*.hdf5)."""
    if os.path.isfile(pattern_or_file):
        return [pattern_or_file]
    files = sorted(glob.glob(pattern_or_file))
    if not files:
        raise FileNotFoundError(f"No files match: {pattern_or_file}")
    return files

def read_header_any(h5):
    hdr = h5['Header'].attrs
    return {
        'BoxSize': float(hdr['BoxSize']),
        'NumFilesPerSnapshot': int(hdr.get('NumFilesPerSnapshot', 1)),
        'Time': float(hdr['Time'])
    }

def read_groupcat(groupcat_path):
    """
    Read the SUBFIND catalog (single HDF5 file). Returns dicts for Group and Subhalo
    with the fields we need if present.
    """
    with h5py.File(groupcat_path, 'r') as f:
        out = {'Group': {}, 'Subhalo': {}}
        if 'Group' in f:
            G = f['Group']
            for fld in ['GroupPos','Group_R_Crit200','Group_M_Crit200']:
                if fld in G:
                    out['Group'][fld] = G[fld][:]
        if 'Subhalo' in f:
            S = f['Subhalo']
            for fld in ['SubhaloPos','SubhaloGrNr','SubhaloHalfmassRadType','SubhaloLenType']:
                if fld in S:
                    out['Subhalo'][fld] = S[fld][:]
    return out

def nearest_image_delta(dx, L):
    """Map dx to [-L/2, L/2) periodically."""
    return dx - L*np.round(dx/L)

def select_dm_within_sphere(snap_files, center, rmax, ptype=1, chunk=2_000_000):
    """
    Read PartType{ptype} positions+IDs and return IDs within sphere of radius rmax
    around 'center' using nearest-image distances. Works with multiple snapshot files.
    """
    ids_in = []
    for fn in snap_files:
        with h5py.File(fn, 'r') as f:
            if f'PartType{ptype}' not in f:  # may be empty in some files
                continue
            grp = f[f'PartType{ptype}']
            coords = grp['Coordinates']
            pid = grp['ParticleIDs']
            L = read_header_any(f)['BoxSize']

            N = coords.shape[0]
            for i0 in range(0, N, chunk):
                i1 = min(i0+chunk, N)
                X = coords[i0:i1, :]
                d = X - center[None,:]
                d = nearest_image_delta(d, L)
                r2 = np.sum(d*d, axis=1)
                m = r2 <= (rmax*rmax)
                if np.any(m):
                    ids_in.append(pid[i0:i1][m])

    if ids_in:
        return np.concatenate(ids_in)
    return np.array([], dtype=np.int64)

def main():
    ap = argparse.ArgumentParser(description="Extract DM IDs for a SUBFIND halo for zoom ICs")
    ap.add_argument("--snapshot", required=True,
                    help="Snapshot HDF5 path or glob (e.g. snapshot_099*.hdf5)")
    ap.add_argument("--groupcat", required=True,
                    help="SUBFIND group catalog HDF5 file (e.g. groups_099.hdf5 or groupcat_099.hdf5)")
    gid = ap.add_mutually_exclusive_group(required=True)
    gid.add_argument("--group-id", type=int, help="Group index (0-based) in the catalog")
    gid.add_argument("--subhalo-id", type=int, help="Subhalo index (0-based) in the catalog")
    ap.add_argument("--kR", type=float, default=2.0, help="Radius multiplier: select DM within k * R200 (default 2.0)")
    ap.add_argument("--output", required=True, help="Output text file with one ID per line")
    ap.add_argument("--ptype", type=int, default=1, help="Particle type to extract (default 1 = DM)")
    args = ap.parse_args()

    # Open snapshot files list
    snap_files = open_many(args.snapshot)
    # Read header (take from first file)
    with h5py.File(snap_files[0], 'r') as f0:
        hdr = read_header_any(f0)
        L = hdr['BoxSize']

    # Read group catalog
    cat = read_groupcat(args.groupcat)

    # Decide center and R200
    if args.group_id is not None:
        G = cat['Group']
        if 'GroupPos' not in G:
            sys.exit("ERROR: GroupPos not found in catalog.")
        if 'Group_R_Crit200' not in G:
            sys.exit("ERROR: Group_R_Crit200 not found in catalog.")
        if args.group_id < 0 or args.group_id >= len(G['GroupPos']):
            sys.exit(f"ERROR: --group-id {args.group_id} out of range.")
        centre = np.asarray(G['GroupPos'][args.group_id], dtype=float)
        R200 = float(G['Group_R_Crit200'][args.group_id])
    else:
        S = cat['Subhalo']
        if 'SubhaloPos' not in S or 'SubhaloGrNr' not in S:
            sys.exit("ERROR: SubhaloPos/SubhaloGrNr not found in catalog.")
        sid = args.subhalo_id
        if sid < 0 or sid >= len(S['SubhaloPos']):
            sys.exit(f"ERROR: --subhalo-id {sid} out of range.")
        centre = np.asarray(S['SubhaloPos'][sid], dtype=float)
        g = int(S['SubhaloGrNr'][sid])
        # Prefer Group R200 if available
        if 'Group_R_Crit200' in cat['Group'] and g >= 0 and g < len(cat['Group']['Group_R_Crit200']):
            R200 = float(cat['Group']['Group_R_Crit200'][g])
        else:
            # Fallback: use DM half-mass radius * 3 as a crude proxy
            if 'SubhaloHalfmassRadType' in S:
                Rhalf_dm = float(S['SubhaloHalfmassRadType'][sid, 1])  # PartType1
                R200 = 3.0 * Rhalf_dm
            else:
                sys.exit("ERROR: No R200 in Group and no SubhaloHalfmassRadType to estimate a radius.")

    rsel = args.kR * R200

    # Make sure the sphere won’t be mangled by a periodic seam:
    # If the center lies too close to an edge relative to rsel, you can shift
    # the center by ±L on that axis. Here we trust nearest-image to do the right thing.

    print(f"BoxSize={L:.6g}  centre={centre}  R200={R200:.6g}  selecting radius={rsel:.6g}")

    ids = select_dm_within_sphere(snap_files, centre, rsel, ptype=args.ptype)
    print(f"Selected {ids.size} DM IDs")

    # Write out one per line (as integers)
    with open(args.output, "w") as fo:
        for pid in ids:
            fo.write(f"{int(pid)}\n")

    print(f"Wrote IDs to {args.output}")

if __name__ == "__main__":
    main()
