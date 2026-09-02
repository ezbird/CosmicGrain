#!/usr/bin/env python3

"""
parent_halo_census.py

Census halos in a DM-only parent simulation and identify candidates
for a matched CosmicGrain zoom suite.

For each z=0 / latest FOF halo:
    - FOF group ID
    - GroupPos
    - GroupMass
    - R200c and M200c from snapshot particles
    - particle count inside R200c
    - nearest halo distance
    - nearest more-massive halo distance
    - nearest significant-neighbor distance
    - simple isolation flag
    - logarithmic mass-bin label
    - candidate shortlist

Supports BOTH:

    snapshot_049.hdf5

and split files such as:

    snapdir_049/snapshot_049.0.hdf5
    snapdir_049/snapshot_049.1.hdf5

Likewise supports both single-file and split FOF catalogs.

Example:

python parent_halo_census.py \
    ~/gadget4/output_parent_50Mpc_128_music_DM_only \
    --h 0.6732 \
    --omega-m 0.3158 \
    --omega-lambda 0.6842 \
    --minimum-mass 1e10 \
    --minimum-particles 100 \
    --output parent_50Mpc_halo_census.csv
"""

import argparse
import glob
import os
import re

import h5py
import numpy as np


# ============================================================================
# Constants
# ============================================================================

G_CGS = 6.67430e-8
MPC_CGS = 3.0856775814913673e24
KPC_CGS = MPC_CGS / 1000.0
MSUN_CGS = 1.98847e33

# Your GADGET/CosmicGrain convention:
# UnitMass = 1e10 Msun/h
MASS_UNIT_MSUN_OVER_H = 1.0e10


# ============================================================================
# File discovery
# ============================================================================

def find_latest_snapshot(output_dir):
    """
    Find the latest available snapshot.

    Supports:
        output_dir/snapshot_049.hdf5
        output_dir/snapshot_049.0.hdf5
        output_dir/snapdir_049/snapshot_049.0.hdf5
    """

    candidates = []

    patterns = [
        os.path.join(
            output_dir,
            "snapshot_*.hdf5"
        ),
        os.path.join(
            output_dir,
            "snapdir_*",
            "snapshot_*.hdf5"
        ),
    ]

    for pattern in patterns:

        for filename in glob.glob(pattern):

            basename = os.path.basename(filename)

            match = re.match(
                r"snapshot_(\d+)(?:\.\d+)?\.hdf5$",
                basename
            )

            if match is None:
                continue

            snapnum = int(
                match.group(1)
            )

            candidates.append(
                (
                    snapnum,
                    filename
                )
            )

    if not candidates:

        raise RuntimeError(
            f"No snapshots found under:\n  {output_dir}"
        )

    latest_snap = max(
        snapnum
        for snapnum, filename in candidates
    )

    # ------------------------------------------------------------------------
    # First preference: split snapshot inside snapdir_NNN
    # ------------------------------------------------------------------------

    split_dir_pattern = os.path.join(
        output_dir,
        f"snapdir_{latest_snap:03d}",
        f"snapshot_{latest_snap:03d}.*.hdf5"
    )

    files = sorted(
        glob.glob(
            split_dir_pattern
        )
    )

    if files:
        return (
            latest_snap,
            files
        )

    # ------------------------------------------------------------------------
    # Second possibility: split files directly in output directory
    # ------------------------------------------------------------------------

    split_flat_pattern = os.path.join(
        output_dir,
        f"snapshot_{latest_snap:03d}.*.hdf5"
    )

    files = sorted(
        glob.glob(
            split_flat_pattern
        )
    )

    if files:
        return (
            latest_snap,
            files
        )

    # ------------------------------------------------------------------------
    # Third possibility: one single snapshot file
    # ------------------------------------------------------------------------

    single_file = os.path.join(
        output_dir,
        f"snapshot_{latest_snap:03d}.hdf5"
    )

    if os.path.exists(
        single_file
    ):

        return (
            latest_snap,
            [single_file]
        )

    raise RuntimeError(
        f"Identified latest snapshot number {latest_snap}, "
        "but could not resolve its file(s)."
    )


def find_group_catalog(
    output_dir,
    snapnum
):
    """
    Locate FOF catalog.

    Supports:
        fof_subhalo_tab_049.hdf5
        fof_subhalo_tab_049.0.hdf5
        groups_049/fof_subhalo_tab_049.0.hdf5
    """

    # ------------------------------------------------------------------------
    # Split catalog inside groups_NNN
    # ------------------------------------------------------------------------

    pattern = os.path.join(
        output_dir,
        f"groups_{snapnum:03d}",
        f"fof_subhalo_tab_{snapnum:03d}.*.hdf5"
    )

    files = sorted(
        glob.glob(
            pattern
        )
    )

    if files:
        return files

    # ------------------------------------------------------------------------
    # Split catalog directly in output directory
    # ------------------------------------------------------------------------

    pattern = os.path.join(
        output_dir,
        f"fof_subhalo_tab_{snapnum:03d}.*.hdf5"
    )

    files = sorted(
        glob.glob(
            pattern
        )
    )

    if files:
        return files

    # ------------------------------------------------------------------------
    # Single-file catalog
    # ------------------------------------------------------------------------

    single_file = os.path.join(
        output_dir,
        f"fof_subhalo_tab_{snapnum:03d}.hdf5"
    )

    if os.path.exists(
        single_file
    ):

        return [
            single_file
        ]

    raise RuntimeError(
        f"No FOF catalog found for snapshot {snapnum}"
    )


# ============================================================================
# Header and units
# ============================================================================

def read_header(files):

    with h5py.File(
        files[0],
        "r"
    ) as f:

        attrs = f[
            "Header"
        ].attrs

        scale_factor = float(
            attrs["Time"]
        )

        redshift = (
            1.0
            /
            scale_factor
            -
            1.0
        )

        boxsize = float(
            attrs["BoxSize"]
        )

        mass_table = np.asarray(
            attrs.get(
                "MassTable",
                np.zeros(7)
            ),
            dtype=np.float64
        )

    return (
        scale_factor,
        redshift,
        boxsize,
        mass_table
    )


def mass_to_msun(
    mass_code,
    h
):

    return (
        mass_code
        *
        MASS_UNIT_MSUN_OVER_H
        /
        h
    )


# ============================================================================
# Particle reading
# ============================================================================

def read_dm_particles(
    files,
    mass_table
):
    """
    Read all collisionless DM particle types that may occur.

    For a uniform parent DM-only simulation this will normally just be
    PartType1, but supporting 2/3/5 makes the script more general.
    """

    coords_all = []
    masses_all = []
    ptype_counts = {}

    possible_dm_types = [
        1,
        2,
        3,
        5
    ]

    for ptype in possible_dm_types:

        group_name = (
            f"PartType{ptype}"
        )

        type_coords = []
        type_masses = []

        for filename in files:

            with h5py.File(
                filename,
                "r"
            ) as f:

                if group_name not in f:
                    continue

                group = f[
                    group_name
                ]

                if (
                    "Coordinates"
                    not in group
                ):
                    continue

                coords = np.asarray(
                    group[
                        "Coordinates"
                    ][:],
                    dtype=np.float64
                )

                if "Masses" in group:

                    masses = np.asarray(
                        group[
                            "Masses"
                        ][:],
                        dtype=np.float64
                    )

                elif (
                    len(mass_table)
                    >
                    ptype
                    and
                    mass_table[
                        ptype
                    ]
                    >
                    0.0
                ):

                    masses = np.full(
                        len(coords),
                        mass_table[
                            ptype
                        ],
                        dtype=np.float64
                    )

                else:

                    raise RuntimeError(
                        f"{group_name} has Coordinates but "
                        "no Masses field and no usable MassTable entry."
                    )

                type_coords.append(
                    coords
                )

                type_masses.append(
                    masses
                )

        if type_coords:

            coords_type = np.concatenate(
                type_coords,
                axis=0
            )

            masses_type = np.concatenate(
                type_masses
            )

            coords_all.append(
                coords_type
            )

            masses_all.append(
                masses_type
            )

            ptype_counts[
                ptype
            ] = len(
                coords_type
            )

    if not coords_all:

        raise RuntimeError(
            "No dark-matter particles were found."
        )

    return (
        np.concatenate(
            coords_all,
            axis=0
        ),
        np.concatenate(
            masses_all
        ),
        ptype_counts
    )


# ============================================================================
# FOF reading
# ============================================================================

def read_groups(
    files
):
    """
    Read the FOF halo positions, masses, and lengths.
    """

    positions_all = []
    masses_all = []
    lengths_all = []

    for filename in files:

        with h5py.File(
            filename,
            "r"
        ) as f:

            if "Group" not in f:
                continue

            group = f[
                "Group"
            ]

            if (
                "GroupPos"
                not in group
            ):

                continue

            positions = np.asarray(
                group[
                    "GroupPos"
                ][:],
                dtype=np.float64
            )

            positions_all.append(
                positions
            )

            # --------------------------------------------------------------
            # Group mass
            # --------------------------------------------------------------

            if (
                "GroupMass"
                in group
            ):

                masses = np.asarray(
                    group[
                        "GroupMass"
                    ][:],
                    dtype=np.float64
                )

            else:

                masses = np.full(
                    len(positions),
                    np.nan,
                    dtype=np.float64
                )

            masses_all.append(
                masses
            )

            # --------------------------------------------------------------
            # Group length
            # --------------------------------------------------------------

            if (
                "GroupLen"
                in group
            ):

                lengths = np.asarray(
                    group[
                        "GroupLen"
                    ][:]
                )

                if (
                    lengths.ndim
                    >
                    1
                ):

                    lengths = np.sum(
                        lengths,
                        axis=1
                    )

                lengths = lengths.astype(
                    np.int64
                )

            else:

                lengths = np.full(
                    len(positions),
                    -1,
                    dtype=np.int64
                )

            lengths_all.append(
                lengths
            )

    if not positions_all:

        raise RuntimeError(
            "No FOF Group data were found."
        )

    return (
        np.concatenate(
            positions_all,
            axis=0
        ),
        np.concatenate(
            masses_all
        ),
        np.concatenate(
            lengths_all
        )
    )


# ============================================================================
# Geometry
# ============================================================================

def periodic_delta(
    coords,
    center,
    boxsize
):

    center = np.asarray(
        center,
        dtype=np.float64
    )

    delta = (
        coords
        -
        center[
            None,
            :
        ]
    )

    if boxsize > 0.0:

        delta -= (
            boxsize
            *
            np.round(
                delta
                /
                boxsize
            )
        )

    return delta


def physical_distance_pkpc(
    coords,
    center,
    boxsize,
    scale_factor,
    h
):

    delta = periodic_delta(
        coords,
        center,
        boxsize
    )

    radius_code = np.sqrt(
        np.sum(
            delta
            *
            delta,
            axis=1
        )
    )

    return (
        radius_code
        *
        scale_factor
        /
        h
    )


# ============================================================================
# Cosmology
# ============================================================================

def critical_density_msun_pkpc3(
    redshift,
    h,
    omega_m,
    omega_lambda
):

    H0 = (
        100.0
        *
        h
        *
        1.0e5
        /
        MPC_CGS
    )

    Ez2 = (
        omega_m
        *
        (
            1.0
            +
            redshift
        )**3
        +
        omega_lambda
    )

    Hz = (
        H0
        *
        np.sqrt(
            Ez2
        )
    )

    rho_c_cgs = (
        3.0
        *
        Hz
        *
        Hz
        /
        (
            8.0
            *
            np.pi
            *
            G_CGS
        )
    )

    return (
        rho_c_cgs
        *
        KPC_CGS**3
        /
        MSUN_CGS
    )


# ============================================================================
# Spherical-overdensity calculation
# ============================================================================

def calculate_r200c(
    center,
    coords,
    masses_msun,
    boxsize,
    scale_factor,
    h,
    rho_crit,
    search_radius_pkpc
):
    """
    Calculate R200c and M200c from snapshot particles.

    Starting from the FOF center, sort particles by radius and find the
    outermost radius whose enclosed mean density is >= 200 rho_crit.
    """

    radius = physical_distance_pkpc(
        coords,
        center,
        boxsize,
        scale_factor,
        h
    )

    selection = (
        (radius > 0.0)
        &
        (
            radius
            <=
            search_radius_pkpc
        )
    )

    count = np.count_nonzero(
        selection
    )

    if count < 20:

        return (
            np.nan,
            np.nan,
            0
        )

    r = radius[
        selection
    ]

    mass = masses_msun[
        selection
    ]

    order = np.argsort(
        r
    )

    r = r[
        order
    ]

    mass = mass[
        order
    ]

    enclosed_mass = np.cumsum(
        mass,
        dtype=np.float64
    )

    volume = (
        4.0
        /
        3.0
        *
        np.pi
        *
        r**3
    )

    mean_density = (
        enclosed_mass
        /
        volume
    )

    density_threshold = (
        200.0
        *
        rho_crit
    )

    valid = np.where(
        mean_density
        >=
        density_threshold
    )[0]

    if len(valid) == 0:

        return (
            np.nan,
            np.nan,
            0
        )

    i = valid[
        -1
    ]

    return (
        float(
            r[
                i
            ]
        ),
        float(
            enclosed_mass[
                i
            ]
        ),
        int(
            i
            +
            1
        )
    )


# ============================================================================
# Environment
# ============================================================================

def pairwise_periodic_distances(
    positions,
    boxsize,
    scale_factor,
    h
):
    """
    Full NxN halo distance matrix in physical kpc.

    This is perfectly adequate for the number of halos expected in a
    50 Mpc/h, 128^3 parent box.
    """

    n_halos = len(
        positions
    )

    distances = np.empty(
        (
            n_halos,
            n_halos
        ),
        dtype=np.float64
    )

    for i in range(
        n_halos
    ):

        delta = (
            positions
            -
            positions[
                i
            ]
        )

        if boxsize > 0.0:

            delta -= (
                boxsize
                *
                np.round(
                    delta
                    /
                    boxsize
                )
            )

        distances[
            i
        ] = (
            np.sqrt(
                np.sum(
                    delta
                    *
                    delta,
                    axis=1
                )
            )
            *
            scale_factor
            /
            h
        )

    return distances


# ============================================================================
# Mass bins
# ============================================================================

def mass_bin_label(
    M200
):

    if (
        not np.isfinite(
            M200
        )
        or
        M200 <= 0
    ):

        return "invalid"

    log_mass = np.log10(
        M200
    )

    if log_mass < 10.0:
        return "<10.0"

    if log_mass < 10.5:
        return "10.0-10.5"

    if log_mass < 11.0:
        return "10.5-11.0"

    if log_mass < 11.5:
        return "11.0-11.5"

    if log_mass < 12.0:
        return "11.5-12.0"

    if log_mass < 12.5:
        return "12.0-12.5"

    return ">12.5"


# ============================================================================
# Main
# ============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Census a DM-only parent simulation and identify "
            "candidate halos for a CosmicGrain zoom suite."
        )
    )

    parser.add_argument(
        "output_dir",
        help="Parent simulation output directory"
    )

    parser.add_argument(
        "--h",
        type=float,
        default=0.6732
    )

    parser.add_argument(
        "--omega-m",
        type=float,
        default=0.3158
    )

    parser.add_argument(
        "--omega-lambda",
        type=float,
        default=0.6842
    )

    parser.add_argument(
        "--minimum-mass",
        type=float,
        default=1.0e10,
        help=(
            "Minimum M200c retained, in Msun. "
            "Default = 1e10."
        )
    )

    parser.add_argument(
        "--minimum-particles",
        type=int,
        default=100,
        help=(
            "Minimum number of parent DM particles inside R200c."
        )
    )

    parser.add_argument(
        "--so-search-pkpc",
        type=float,
        default=1000.0,
        help=(
            "Maximum physical radius searched when calculating R200c."
        )
    )

    parser.add_argument(
        "--isolation-mass-ratio",
        type=float,
        default=0.5,
        help=(
            "A significant neighbor has M >= this fraction "
            "of the target halo mass."
        )
    )

    parser.add_argument(
        "--isolation-radius-r200",
        type=float,
        default=3.0,
        help=(
            "Halo is labeled isolated when its nearest significant "
            "neighbor is at least this many target R200c away."
        )
    )

    parser.add_argument(
        "--output",
        default="parent_halo_census.csv"
    )

    args = parser.parse_args()

    # ========================================================================
    # Locate latest snapshot and FOF catalog
    # ========================================================================

    (
        snapshot_number,
        snapshot_files
    ) = find_latest_snapshot(
        args.output_dir
    )

    group_files = find_group_catalog(
        args.output_dir,
        snapshot_number
    )

    print()
    print(
        f"Using snapshot {snapshot_number:03d}"
    )

    print()
    print("Snapshot files:")

    for filename in snapshot_files:
        print(
            f"  {filename}"
        )

    print()
    print("FOF catalog files:")

    for filename in group_files:
        print(
            f"  {filename}"
        )

    # ========================================================================
    # Header
    # ========================================================================

    (
        scale_factor,
        redshift,
        boxsize,
        mass_table
    ) = read_header(
        snapshot_files
    )

    print()
    print("=" * 90)
    print("PARENT DM-ONLY HALO CENSUS")
    print("=" * 90)

    print(
        f"Snapshot           = "
        f"{snapshot_number:03d}"
    )

    print(
        f"Scale factor       = "
        f"{scale_factor:.8f}"
    )

    print(
        f"Redshift           = "
        f"{redshift:.8f}"
    )

    print(
        f"BoxSize            = "
        f"{boxsize:.6f}"
    )

    print(
        f"Analysis h         = "
        f"{args.h:.6f}"
    )

    print(
        f"Minimum M200c      = "
        f"{args.minimum_mass:.3e} Msun"
    )

    print(
        f"Minimum N200       = "
        f"{args.minimum_particles}"
    )

    print("=" * 90)

    # ========================================================================
    # Read parent DM particles
    # ========================================================================

    (
        dm_coords,
        dm_mass_code,
        dm_type_counts
    ) = read_dm_particles(
        snapshot_files,
        mass_table
    )

    dm_mass_msun = mass_to_msun(
        dm_mass_code,
        args.h
    )

    print()
    print("Dark-matter particle inventory:")

    for ptype in sorted(
        dm_type_counts
    ):

        print(
            f"  PartType{ptype}: "
            f"{dm_type_counts[ptype]:,}"
        )

    print()

    print(
        f"Total DM particles = "
        f"{len(dm_coords):,}"
    )

    print(
        f"DM particle mass minimum = "
        f"{np.min(dm_mass_msun):.6e} Msun"
    )

    print(
        f"DM particle mass maximum = "
        f"{np.max(dm_mass_msun):.6e} Msun"
    )

    print(
        f"DM particle mass median  = "
        f"{np.median(dm_mass_msun):.6e} Msun"
    )

    # ========================================================================
    # FOF groups
    # ========================================================================

    (
        group_positions,
        group_mass_code,
        group_lengths
    ) = read_groups(
        group_files
    )

    group_mass_msun = mass_to_msun(
        group_mass_code,
        args.h
    )

    print()

    print(
        f"FOF groups found = "
        f"{len(group_positions):,}"
    )

    # ========================================================================
    # Critical density
    # ========================================================================

    rho_crit = (
        critical_density_msun_pkpc3(
            redshift,
            args.h,
            args.omega_m,
            args.omega_lambda
        )
    )

    print(
        f"rho_crit(z) = "
        f"{rho_crit:.6e} Msun/pkpc^3"
    )

    # ========================================================================
    # Calculate R200c / M200c
    # ========================================================================

    halos = []

    print()
    print(
        "Calculating spherical-overdensity halo properties..."
    )

    for halo_id, center in enumerate(
        group_positions
    ):

        fof_mass = group_mass_msun[
            halo_id
        ]

        # Cheap early rejection based on FOF mass.
        if (
            np.isfinite(
                fof_mass
            )
            and
            fof_mass
            <
            0.20
            *
            args.minimum_mass
        ):

            continue

        (
            R200c,
            M200c,
            N200c
        ) = calculate_r200c(
            center=center,
            coords=dm_coords,
            masses_msun=dm_mass_msun,
            boxsize=boxsize,
            scale_factor=scale_factor,
            h=args.h,
            rho_crit=rho_crit,
            search_radius_pkpc=args.so_search_pkpc
        )

        if (
            not np.isfinite(
                M200c
            )
            or
            not np.isfinite(
                R200c
            )
        ):

            continue

        if (
            M200c
            <
            args.minimum_mass
        ):

            continue

        if (
            N200c
            <
            args.minimum_particles
        ):

            continue

        halos.append(
            {
                "halo_id": halo_id,

                "center": np.asarray(
                    center,
                    dtype=np.float64
                ).copy(),

                "fof_mass_msun": fof_mass,

                "fof_particle_count": int(
                    group_lengths[
                        halo_id
                    ]
                ),

                "M200c_msun": M200c,

                "R200c_pkpc": R200c,

                "N200c": N200c,
            }
        )

    if not halos:

        raise RuntimeError(
            "No halos passed the M200/N200 selection."
        )

    print(
        f"Resolved halos retained = "
        f"{len(halos)}"
    )

    # ========================================================================
    # Environmental statistics
    # ========================================================================

    halo_positions = np.asarray(
        [
            halo[
                "center"
            ]
            for halo in halos
        ]
    )

    halo_masses = np.asarray(
        [
            halo[
                "M200c_msun"
            ]
            for halo in halos
        ]
    )

    distance_matrix = (
        pairwise_periodic_distances(
            halo_positions,
            boxsize,
            scale_factor,
            args.h
        )
    )

    np.fill_diagonal(
        distance_matrix,
        np.inf
    )

    for i, halo in enumerate(
        halos
    ):

        # --------------------------------------------------------------------
        # Nearest resolved halo of any mass
        # --------------------------------------------------------------------

        nearest_index = int(
            np.argmin(
                distance_matrix[
                    i
                ]
            )
        )

        nearest_distance = (
            distance_matrix[
                i,
                nearest_index
            ]
        )

        nearest_mass = (
            halo_masses[
                nearest_index
            ]
        )

        # --------------------------------------------------------------------
        # Nearest halo at least as massive as target
        # --------------------------------------------------------------------

        more_massive_mask = (
            halo_masses
            >=
            halo[
                "M200c_msun"
            ]
        )

        more_massive_mask[
            i
        ] = False

        if np.any(
            more_massive_mask
        ):

            candidate_distances = (
                distance_matrix[
                    i
                ].copy()
            )

            candidate_distances[
                ~more_massive_mask
            ] = np.inf

            j = int(
                np.argmin(
                    candidate_distances
                )
            )

            nearest_more_massive_distance = (
                candidate_distances[
                    j
                ]
            )

            nearest_more_massive_mass = (
                halo_masses[
                    j
                ]
            )

        else:

            nearest_more_massive_distance = np.nan
            nearest_more_massive_mass = np.nan

        # --------------------------------------------------------------------
        # Nearest significant neighbor
        #
        # significant means:
        #
        #       M_neighbor >= ratio * M_target
        # --------------------------------------------------------------------

        significant_mask = (
            halo_masses
            >=
            args.isolation_mass_ratio
            *
            halo[
                "M200c_msun"
            ]
        )

        significant_mask[
            i
        ] = False

        if np.any(
            significant_mask
        ):

            candidate_distances = (
                distance_matrix[
                    i
                ].copy()
            )

            candidate_distances[
                ~significant_mask
            ] = np.inf

            j = int(
                np.argmin(
                    candidate_distances
                )
            )

            significant_distance = (
                candidate_distances[
                    j
                ]
            )

            significant_mass = (
                halo_masses[
                    j
                ]
            )

            significant_distance_r200 = (
                significant_distance
                /
                halo[
                    "R200c_pkpc"
                ]
            )

        else:

            significant_distance = np.nan
            significant_mass = np.nan
            significant_distance_r200 = np.inf

        isolated = (
            significant_distance_r200
            >=
            args.isolation_radius_r200
        )

        # --------------------------------------------------------------------
        # Local counts within fixed physical radii
        # --------------------------------------------------------------------

        distances = (
            distance_matrix[
                i
            ]
        )

        n_neighbor_500pkpc = int(
            np.count_nonzero(
                distances
                <
                500.0
            )
        )

        n_neighbor_1000pkpc = int(
            np.count_nonzero(
                distances
                <
                1000.0
            )
        )

        n_neighbor_2000pkpc = int(
            np.count_nonzero(
                distances
                <
                2000.0
            )
        )

        halo[
            "mass_bin"
        ] = mass_bin_label(
            halo[
                "M200c_msun"
            ]
        )

        halo[
            "nearest_halo_distance_pkpc"
        ] = nearest_distance

        halo[
            "nearest_halo_mass_msun"
        ] = nearest_mass

        halo[
            "nearest_more_massive_distance_pkpc"
        ] = nearest_more_massive_distance

        halo[
            "nearest_more_massive_mass_msun"
        ] = nearest_more_massive_mass

        halo[
            "significant_neighbor_distance_pkpc"
        ] = significant_distance

        halo[
            "significant_neighbor_mass_msun"
        ] = significant_mass

        halo[
            "significant_neighbor_distance_R200"
        ] = significant_distance_r200

        halo[
            "isolated"
        ] = bool(
            isolated
        )

        halo[
            "neighbor_count_500pkpc"
        ] = n_neighbor_500pkpc

        halo[
            "neighbor_count_1000pkpc"
        ] = n_neighbor_1000pkpc

        halo[
            "neighbor_count_2000pkpc"
        ] = n_neighbor_2000pkpc

    # ========================================================================
    # Sort by M200 descending
    # ========================================================================

    halos.sort(
        key=lambda halo:
        halo[
            "M200c_msun"
        ],
        reverse=True
    )

    # ========================================================================
    # Console table
    # ========================================================================

    print()
    print("=" * 118)
    print("PARENT HALO SAMPLE")
    print("=" * 118)

    print(
        f"{'ID':>5} "
        f"{'M200c':>12} "
        f"{'R200c':>8} "
        f"{'N200':>7} "
        f"{'mass bin':>11} "
        f"{'sig neighbor':>13} "
        f"{'d/R200':>8} "
        f"{'N<1Mpc':>7} "
        f"{'isolated':>9}"
    )

    print(
        "-" * 118
    )

    for halo in halos:

        significant_distance = (
            halo[
                "significant_neighbor_distance_pkpc"
            ]
        )

        significant_r200 = (
            halo[
                "significant_neighbor_distance_R200"
            ]
        )

        if np.isfinite(
            significant_distance
        ):

            significant_distance_text = (
                f"{significant_distance:13.1f}"
            )

        else:

            significant_distance_text = (
                f"{'none':>13}"
            )

        if np.isfinite(
            significant_r200
        ):

            significant_r200_text = (
                f"{significant_r200:8.2f}"
            )

        else:

            significant_r200_text = (
                f"{'inf':>8}"
            )

        print(
            f"{halo['halo_id']:5d} "
            f"{halo['M200c_msun']:12.3e} "
            f"{halo['R200c_pkpc']:8.2f} "
            f"{halo['N200c']:7d} "
            f"{halo['mass_bin']:>11s} "
            f"{significant_distance_text} "
            f"{significant_r200_text} "
            f"{halo['neighbor_count_1000pkpc']:7d} "
            f"{str(halo['isolated']):>9s}"
        )

    print(
        "=" * 118
    )

    # ========================================================================
    # Save CSV
    # ========================================================================

    columns = [
        "halo_id",

        "center_x",
        "center_y",
        "center_z",

        "fof_mass_msun",
        "fof_particle_count",

        "M200c_msun",
        "R200c_pkpc",
        "N200c",

        "mass_bin",

        "nearest_halo_distance_pkpc",
        "nearest_halo_mass_msun",

        "nearest_more_massive_distance_pkpc",
        "nearest_more_massive_mass_msun",

        "significant_neighbor_distance_pkpc",
        "significant_neighbor_mass_msun",
        "significant_neighbor_distance_R200",

        "neighbor_count_500pkpc",
        "neighbor_count_1000pkpc",
        "neighbor_count_2000pkpc",

        "isolated",
    ]

    with open(
        args.output,
        "w"
    ) as f:

        f.write(
            ",".join(
                columns
            )
            +
            "\n"
        )

        for halo in halos:

            row = [
                halo[
                    "halo_id"
                ],

                halo[
                    "center"
                ][0],

                halo[
                    "center"
                ][1],

                halo[
                    "center"
                ][2],

                halo[
                    "fof_mass_msun"
                ],

                halo[
                    "fof_particle_count"
                ],

                halo[
                    "M200c_msun"
                ],

                halo[
                    "R200c_pkpc"
                ],

                halo[
                    "N200c"
                ],

                halo[
                    "mass_bin"
                ],

                halo[
                    "nearest_halo_distance_pkpc"
                ],

                halo[
                    "nearest_halo_mass_msun"
                ],

                halo[
                    "nearest_more_massive_distance_pkpc"
                ],

                halo[
                    "nearest_more_massive_mass_msun"
                ],

                halo[
                    "significant_neighbor_distance_pkpc"
                ],

                halo[
                    "significant_neighbor_mass_msun"
                ],

                halo[
                    "significant_neighbor_distance_R200"
                ],

                halo[
                    "neighbor_count_500pkpc"
                ],

                halo[
                    "neighbor_count_1000pkpc"
                ],

                halo[
                    "neighbor_count_2000pkpc"
                ],

                int(
                    halo[
                        "isolated"
                    ]
                ),
            ]

            values = []

            for value in row:

                if isinstance(
                    value,
                    (
                        float,
                        np.floating
                    )
                ):

                    values.append(
                        f"{value:.10e}"
                    )

                else:

                    values.append(
                        str(
                            value
                        )
                    )

            f.write(
                ",".join(
                    values
                )
                +
                "\n"
            )

    # ========================================================================
    # Mass-bin statistics
    # ========================================================================

    print()
    print("=" * 80)
    print("MASS-BIN COUNTS")
    print("=" * 80)

    bins = [
        "<10.0",
        "10.0-10.5",
        "10.5-11.0",
        "11.0-11.5",
        "11.5-12.0",
        "12.0-12.5",
        ">12.5",
    ]

    for mass_bin in bins:

        subset = [
            halo
            for halo in halos
            if (
                halo[
                    "mass_bin"
                ]
                ==
                mass_bin
            )
        ]

        isolated_subset = [
            halo
            for halo in subset
            if halo[
                "isolated"
            ]
        ]

        print(
            f"{mass_bin:12s} "
            f"N={len(subset):4d}   "
            f"isolated={len(isolated_subset):4d}"
        )

    # ========================================================================
    # Suggested targets
    # ========================================================================

    print()
    print("=" * 100)
    print("SUGGESTED INITIAL TARGET CANDIDATES")
    print("=" * 100)

    desired_bins = [
        "10.0-10.5",
        "10.5-11.0",
        "11.0-11.5",
        "11.5-12.0",
        "12.0-12.5",
    ]

    for mass_bin in desired_bins:

        candidates = [
            halo
            for halo in halos
            if (
                halo[
                    "mass_bin"
                ]
                ==
                mass_bin
                and
                halo[
                    "isolated"
                ]
            )
        ]

        # Prefer large separation from significant neighbors.
        candidates.sort(
            key=lambda halo:
            halo[
                "significant_neighbor_distance_R200"
            ],
            reverse=True
        )

        print()
        print(
            f"{mass_bin}:"
        )

        if not candidates:

            print(
                "  no isolated candidates"
            )

            continue

        for halo in candidates[
            :5
        ]:

            print(
                f"  halo {halo['halo_id']:4d} "
                f"M200={halo['M200c_msun']:.3e} Msun  "
                f"R200={halo['R200c_pkpc']:.1f} pkpc  "
                f"N200={halo['N200c']:5d}  "
                f"nearest significant="
                f"{halo['significant_neighbor_distance_R200']:.1f} R200  "
                f"N(<1Mpc)="
                f"{halo['neighbor_count_1000pkpc']}"
            )

    # ========================================================================
    # Overall summary
    # ========================================================================

    isolated_halos = [
        halo
        for halo in halos
        if halo[
            "isolated"
        ]
    ]

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(
        f"Resolved halos retained = "
        f"{len(halos)}"
    )

    print(
        f"Isolated halos          = "
        f"{len(isolated_halos)}"
    )

    print(
        f"M200 range              = "
        f"{min(h['M200c_msun'] for h in halos):.3e} "
        f"- "
        f"{max(h['M200c_msun'] for h in halos):.3e} Msun"
    )

    print()

    print(
        f"Saved: {args.output}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()
