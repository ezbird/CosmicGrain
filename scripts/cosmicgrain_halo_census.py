#!/usr/bin/env python3

"""
cosmicgrain_halo_census.py

Census the z=0 halo population in a CosmicGrain zoom simulation.

Goals
-----
For every FOF halo:

  * read the catalog position/mass
  * estimate R200c and M200c directly from snapshot particles
  * count HR and low-resolution dark matter inside R200c
  * quantify low-resolution contamination
  * measure gas, stellar, and dust content inside R200c
  * measure baryonic/dust content inside a central physical aperture
  * classify halos as EXCELLENT / GOOD / MARGINAL / CONTAMINATED

The script is intended primarily to answer:

    "How many galaxies in this zoom are actually useful for a
     multi-halo CosmicGrain population study?"

Particle convention
-------------------
PartType0 = gas
PartType1 = high-resolution dark matter
PartType2 = intermediate-resolution dark matter
PartType3 = coarse dark matter
PartType4 = stars
PartType5 = outermost dark matter
PartType6 = dust

Example
-------
python cosmicgrain_halo_census.py \
    '../S10_output_1024/snapdir_047/snapshot_047.*.hdf5' \
    '../S10_output_1024/groups_047/fof_subhalo_tab_047.*.hdf5' \
    --h 0.6732 \
    --central-aperture-pkpc 20 \
    --output halo569_zoom_halo_census.csv
"""

import argparse
import glob
import os
import sys

import h5py
import numpy as np


# ======================================================================
# Constants
# ======================================================================

G_CGS = 6.67430e-8
MPC_CGS = 3.0856775814913673e24
MSUN_CGS = 1.98847e33

# CosmicGrain/GADGET particle types
GAS = 0
HRDM = 1
LRDM_TYPES = (2, 3, 5)
STARS = 4
DUST = 6


# ======================================================================
# Small utilities
# ======================================================================

def expand_files(pattern):
    files = sorted(glob.glob(pattern))

    if not files:
        raise RuntimeError(
            f"No files matched:\n  {pattern}"
        )

    return files


def periodic_delta(coords, center, boxsize):
    """
    Minimum-image displacement in snapshot coordinate units.
    """

    d = coords - center[None, :]

    if boxsize > 0:
        d -= boxsize * np.round(d / boxsize)

    return d


def physical_radius_pkpc(coords, center, boxsize, a, h):
    """
    Convert comoving snapshot coordinate separation to physical kpc.

    Coordinates are assumed to use the usual kpc/h convention.
    """

    d = periodic_delta(
        coords,
        center,
        boxsize
    )

    r_code = np.sqrt(
        np.sum(d * d, axis=1)
    )

    return r_code * a / h


def mass_to_msun(mass_code, h):
    """
    CosmicGrain UnitMass = 1e10 Msun/h.
    """

    return mass_code * 1.0e10 / h


# ======================================================================
# Snapshot I/O
# ======================================================================

def read_snapshot_header(files):

    with h5py.File(files[0], "r") as f:

        attrs = f["Header"].attrs

        a = float(attrs["Time"])
        z = 1.0 / a - 1.0

        boxsize = float(attrs["BoxSize"])

        mass_table = np.asarray(
            attrs.get(
                "MassTable",
                np.zeros(7)
            ),
            dtype=np.float64
        )

    return a, z, boxsize, mass_table


def read_particle_type(files, ptype, mass_table):
    """
    Read Coordinates and Masses for one particle type.

    If individual Masses are absent, use Header/MassTable.
    """

    coords_all = []
    masses_all = []

    group_name = f"PartType{ptype}"

    for filename in files:

        with h5py.File(filename, "r") as f:

            if group_name not in f:
                continue

            g = f[group_name]

            if "Coordinates" not in g:
                continue

            coords = np.asarray(
                g["Coordinates"][:],
                dtype=np.float64
            )

            if "Masses" in g:

                masses = np.asarray(
                    g["Masses"][:],
                    dtype=np.float64
                )

            elif mass_table[ptype] > 0:

                masses = np.full(
                    len(coords),
                    mass_table[ptype],
                    dtype=np.float64
                )

            else:

                raise RuntimeError(
                    f"{group_name} has no Masses dataset "
                    "and MassTable is zero."
                )

            coords_all.append(coords)
            masses_all.append(masses)

    if not coords_all:

        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty(0, dtype=np.float64)
        )

    return (
        np.concatenate(coords_all, axis=0),
        np.concatenate(masses_all, axis=0)
    )


# ======================================================================
# FOF catalog
# ======================================================================

def read_fof_catalog(files):
    """
    Read GroupPos and GroupMass.

    We deliberately keep this simple: the catalog supplies candidate
    centers, while M200c/R200c are recalculated directly from particles.
    """

    positions = []
    masses = []

    for filename in files:

        with h5py.File(filename, "r") as f:

            if "Group" not in f:
                continue

            g = f["Group"]

            if "GroupPos" not in g:
                raise RuntimeError(
                    f"GroupPos not found in {filename}"
                )

            positions.append(
                np.asarray(
                    g["GroupPos"][:],
                    dtype=np.float64
                )
            )

            if "GroupMass" in g:

                masses.append(
                    np.asarray(
                        g["GroupMass"][:],
                        dtype=np.float64
                    )
                )

            else:

                masses.append(
                    np.full(
                        len(g["GroupPos"]),
                        np.nan
                    )
                )

    if not positions:

        raise RuntimeError(
            "No FOF groups were found."
        )

    return (
        np.concatenate(positions),
        np.concatenate(masses)
    )


# ======================================================================
# Cosmology
# ======================================================================

def critical_density_msun_pkpc3(z, h, omega_m, omega_lambda):
    """
    Critical density at redshift z in Msun / physical kpc^3.
    """

    H0 = (
        100.0
        * h
        * 1.0e5
        / MPC_CGS
    )

    Ez2 = (
        omega_m * (1.0 + z)**3
        +
        omega_lambda
    )

    Hz = H0 * np.sqrt(Ez2)

    rho_c_cgs = (
        3.0
        * Hz**2
        /
        (8.0 * np.pi * G_CGS)
    )

    # g/cm^3 -> Msun/kpc^3
    kpc_cgs = MPC_CGS / 1000.0

    rho_c = (
        rho_c_cgs
        *
        kpc_cgs**3
        /
        MSUN_CGS
    )

    return rho_c


# ======================================================================
# SO halo calculation
# ======================================================================

def calculate_r200c(
    center,
    all_coords,
    all_masses_msun,
    boxsize,
    a,
    h,
    rho_crit,
    search_radius_pkpc=1000.0
):
    """
    Calculate R200c and M200c directly from all simulation particles.

    We sort particles by physical radius and find the outermost radius
    where the enclosed mean density is >= 200 rho_crit.
    """

    r = physical_radius_pkpc(
        all_coords,
        center,
        boxsize,
        a,
        h
    )

    mask = (
        (r > 0)
        &
        (r <= search_radius_pkpc)
    )

    if np.count_nonzero(mask) < 10:
        return np.nan, np.nan

    rr = r[mask]
    mm = all_masses_msun[mask]

    order = np.argsort(rr)

    rr = rr[order]
    mm = mm[order]

    cumulative_mass = np.cumsum(mm)

    volume = (
        (4.0 / 3.0)
        *
        np.pi
        *
        rr**3
    )

    mean_density = (
        cumulative_mass
        /
        volume
    )

    target = 200.0 * rho_crit

    valid = np.where(
        mean_density >= target
    )[0]

    if len(valid) == 0:
        return np.nan, np.nan

    i = valid[-1]

    return (
        rr[i],
        cumulative_mass[i]
    )


# ======================================================================
# Measurements
# ======================================================================

def measure_inside(
    coords,
    masses_msun,
    center,
    radius_pkpc,
    boxsize,
    a,
    h
):

    if len(coords) == 0:
        return 0, 0.0

    r = physical_radius_pkpc(
        coords,
        center,
        boxsize,
        a,
        h
    )

    sel = (
        r <= radius_pkpc
    )

    return (
        int(np.count_nonzero(sel)),
        float(np.sum(masses_msun[sel]))
    )


# ======================================================================
# Quality classification
# ======================================================================

def classify_halo(
    contamination_mass_fraction,
    contamination_particle_fraction,
    nstar,
    ngas,
    ndust
):
    """
    These are deliberately analysis-oriented initial thresholds.

    They can easily be changed after seeing the census.
    """

    if (
        contamination_mass_fraction > 0.01
        or
        contamination_particle_fraction > 0.01
    ):
        return "CONTAMINATED"

    if (
        nstar >= 500
        and
        ngas >= 500
        and
        ndust >= 1000
    ):
        return "EXCELLENT"

    if (
        nstar >= 200
        and
        ngas >= 200
        and
        ndust >= 300
    ):
        return "GOOD"

    if (
        nstar >= 50
        or
        ngas >= 50
        or
        ndust >= 100
    ):
        return "MARGINAL"

    return "POOR"


# ======================================================================
# Main
# ======================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Census halos in the high-resolution region "
            "of a CosmicGrain zoom simulation."
        )
    )

    parser.add_argument(
        "snapshot",
        help="Snapshot glob"
    )

    parser.add_argument(
        "group_catalog",
        help="FOF group catalog glob"
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
        "--central-aperture-pkpc",
        type=float,
        default=20.0
    )

    parser.add_argument(
        "--so-search-pkpc",
        type=float,
        default=1000.0
    )

    parser.add_argument(
        "--minimum-fof-mass-msun",
        type=float,
        default=1.0e9,
        help=(
            "Skip catalog groups below this FOF mass. "
            "Default: 1e9 Msun"
        )
    )

    parser.add_argument(
        "--output",
        default="cosmicgrain_halo_census.csv"
    )

    args = parser.parse_args()

    snapshot_files = expand_files(
        args.snapshot
    )

    group_files = expand_files(
        args.group_catalog
    )

    print()
    print("Reading snapshot:")
    for f in snapshot_files:
        print(" ", f)

    a, z, boxsize, mass_table = (
        read_snapshot_header(
            snapshot_files
        )
    )

    print()
    print("=" * 90)
    print("COSMICGRAIN HIGH-RESOLUTION HALO CENSUS")
    print("=" * 90)
    print(f"Scale factor       = {a:.8f}")
    print(f"Redshift           = {z:.8f}")
    print(f"Analysis h         = {args.h:.6f}")
    print(f"BoxSize            = {boxsize:.6f}")
    print(
        f"Central aperture   = "
        f"{args.central_aperture_pkpc:.2f} pkpc"
    )
    print("=" * 90)

    # ------------------------------------------------------------------
    # Load particles
    # ------------------------------------------------------------------

    particle_data = {}

    for ptype in range(7):

        coords, masses = read_particle_type(
            snapshot_files,
            ptype,
            mass_table
        )

        particle_data[ptype] = {
            "coords": coords,
            "mass_code": masses,
            "mass_msun": mass_to_msun(
                masses,
                args.h
            )
        }

        print(
            f"PartType{ptype}: "
            f"{len(coords):,} particles"
        )

    # ------------------------------------------------------------------
    # All particles for SO calculation
    # ------------------------------------------------------------------

    all_coords_list = []
    all_mass_list = []

    for ptype in range(7):

        if len(
            particle_data[ptype]["coords"]
        ) == 0:
            continue

        all_coords_list.append(
            particle_data[ptype]["coords"]
        )

        all_mass_list.append(
            particle_data[ptype]["mass_msun"]
        )

    all_coords = np.concatenate(
        all_coords_list,
        axis=0
    )

    all_masses = np.concatenate(
        all_mass_list
    )

    # ------------------------------------------------------------------
    # FOF catalog
    # ------------------------------------------------------------------

    print()
    print("Reading FOF catalog:")
    for f in group_files:
        print(" ", f)

    group_pos, group_mass_code = (
        read_fof_catalog(
            group_files
        )
    )

    group_mass_msun = (
        mass_to_msun(
            group_mass_code,
            args.h
        )
    )

    print()
    print(
        f"FOF groups found = "
        f"{len(group_pos):,}"
    )

    # ------------------------------------------------------------------
    # Critical density
    # ------------------------------------------------------------------

    rho_crit = (
        critical_density_msun_pkpc3(
            z,
            args.h,
            args.omega_m,
            args.omega_lambda
        )
    )

    print(
        f"rho_crit(z) = "
        f"{rho_crit:.6e} Msun/pkpc^3"
    )

    # ------------------------------------------------------------------
    # Analyze groups
    # ------------------------------------------------------------------

    rows = []

    print()
    print("=" * 125)
    print(
        f"{'ID':>5} "
        f"{'M200c':>12} "
        f"{'R200c':>8} "
        f"{'Mstar':>12} "
        f"{'Mgas':>12} "
        f"{'Mdust':>12} "
        f"{'Nstar':>7} "
        f"{'Ngas':>7} "
        f"{'Ndust':>7} "
        f"{'LRmass%':>9} "
        f"{'quality':>13}"
    )
    print("-" * 125)

    for gid, center in enumerate(group_pos):

        fof_mass = group_mass_msun[gid]

        if (
            np.isfinite(fof_mass)
            and
            fof_mass
            <
            args.minimum_fof_mass_msun
        ):
            continue

        r200, m200 = calculate_r200c(
            center,
            all_coords,
            all_masses,
            boxsize,
            a,
            args.h,
            rho_crit,
            search_radius_pkpc=args.so_search_pkpc
        )

        if (
            not np.isfinite(r200)
            or
            not np.isfinite(m200)
            or
            r200 <= 0
        ):
            continue

        # --------------------------------------------------------------
        # Particle populations inside R200
        # --------------------------------------------------------------

        counts = {}
        masses = {}

        for ptype in range(7):

            n, m = measure_inside(
                particle_data[ptype]["coords"],
                particle_data[ptype]["mass_msun"],
                center,
                r200,
                boxsize,
                a,
                args.h
            )

            counts[ptype] = n
            masses[ptype] = m

        # --------------------------------------------------------------
        # Low-resolution contamination
        # --------------------------------------------------------------

        lr_count = sum(
            counts[p]
            for p in LRDM_TYPES
        )

        lr_mass = sum(
            masses[p]
            for p in LRDM_TYPES
        )

        dm_count = (
            counts[HRDM]
            +
            lr_count
        )

        dm_mass = (
            masses[HRDM]
            +
            lr_mass
        )

        if dm_count > 0:
            contamination_particle_fraction = (
                lr_count
                /
                dm_count
            )
        else:
            contamination_particle_fraction = np.nan

        if dm_mass > 0:
            contamination_mass_fraction = (
                lr_mass
                /
                dm_mass
            )
        else:
            contamination_mass_fraction = np.nan

        # --------------------------------------------------------------
        # Central aperture
        # --------------------------------------------------------------

        central = {}

        for ptype in (
            GAS,
            STARS,
            DUST
        ):

            n, m = measure_inside(
                particle_data[ptype]["coords"],
                particle_data[ptype]["mass_msun"],
                center,
                args.central_aperture_pkpc,
                boxsize,
                a,
                args.h
            )

            central[ptype] = {
                "count": n,
                "mass": m
            }

        quality = classify_halo(
            contamination_mass_fraction,
            contamination_particle_fraction,
            counts[STARS],
            counts[GAS],
            counts[DUST]
        )

        row = {
            "halo_id": gid,

            "center_x": center[0],
            "center_y": center[1],
            "center_z": center[2],

            "fof_mass_msun": fof_mass,

            "R200c_pkpc": r200,
            "M200c_msun": m200,

            "N_gas_R200": counts[GAS],
            "N_hrdm_R200": counts[HRDM],
            "N_lrdm2_R200": counts[2],
            "N_lrdm3_R200": counts[3],
            "N_star_R200": counts[STARS],
            "N_lrdm5_R200": counts[5],
            "N_dust_R200": counts[DUST],

            "Mgas_R200_msun": masses[GAS],
            "Mhrdm_R200_msun": masses[HRDM],
            "Mlrdm_R200_msun": lr_mass,
            "Mstar_R200_msun": masses[STARS],
            "Mdust_R200_msun": masses[DUST],

            "lr_dm_particle_fraction": (
                contamination_particle_fraction
            ),

            "lr_dm_mass_fraction": (
                contamination_mass_fraction
            ),

            "N_gas_20pkpc": central[GAS]["count"],
            "N_star_20pkpc": central[STARS]["count"],
            "N_dust_20pkpc": central[DUST]["count"],

            "Mgas_20pkpc_msun": central[GAS]["mass"],
            "Mstar_20pkpc_msun": central[STARS]["mass"],
            "Mdust_20pkpc_msun": central[DUST]["mass"],

            "quality": quality,
        }

        rows.append(row)

        print(
            f"{gid:5d} "
            f"{m200:12.3e} "
            f"{r200:8.2f} "
            f"{masses[STARS]:12.3e} "
            f"{masses[GAS]:12.3e} "
            f"{masses[DUST]:12.3e} "
            f"{counts[STARS]:7d} "
            f"{counts[GAS]:7d} "
            f"{counts[DUST]:7d} "
            f"{100.0 * contamination_mass_fraction:9.3f} "
            f"{quality:>13}"
        )

    print("=" * 125)

    # ------------------------------------------------------------------
    # Sort by M200 descending
    # ------------------------------------------------------------------

    rows.sort(
        key=lambda x: x["M200c_msun"],
        reverse=True
    )

    # ------------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------------

    if rows:

        columns = list(
            rows[0].keys()
        )

        with open(
            args.output,
            "w"
        ) as f:

            f.write(
                ",".join(columns)
                +
                "\n"
            )

            for row in rows:

                values = []

                for c in columns:

                    value = row[c]

                    if isinstance(
                        value,
                        (float, np.floating)
                    ):
                        values.append(
                            f"{value:.10e}"
                        )
                    else:
                        values.append(
                            str(value)
                        )

                f.write(
                    ",".join(values)
                    +
                    "\n"
                )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    quality_counts = {}

    for row in rows:

        q = row["quality"]

        quality_counts[q] = (
            quality_counts.get(q, 0)
            +
            1
        )

    print()
    print("=" * 90)
    print("CENSUS SUMMARY")
    print("=" * 90)

    print(
        f"Halos analyzed     = {len(rows)}"
    )

    for q in [
        "EXCELLENT",
        "GOOD",
        "MARGINAL",
        "POOR",
        "CONTAMINATED"
    ]:

        print(
            f"{q:16s} = "
            f"{quality_counts.get(q, 0)}"
        )

    # Clean population
    clean = [
        r
        for r in rows
        if r["quality"]
        in ("EXCELLENT", "GOOD")
    ]

    if clean:

        masses_clean = np.array(
            [
                r["M200c_msun"]
                for r in clean
            ]
        )

        print()
        print(
            "Potential science-sample halos:"
        )

        print(
            f"  N              = "
            f"{len(clean)}"
        )

        print(
            f"  M200c min      = "
            f"{np.min(masses_clean):.3e} Msun"
        )

        print(
            f"  M200c max      = "
            f"{np.max(masses_clean):.3e} Msun"
        )

        print(
            f"  dynamic range  = "
            f"{np.max(masses_clean) / np.min(masses_clean):.2f}"
        )

    else:

        print()
        print(
            "No GOOD/EXCELLENT halos found "
            "with the current initial thresholds."
        )

    print()
    print(
        f"Saved: {args.output}"
    )

    print("=" * 90)


if __name__ == "__main__":
    main()
