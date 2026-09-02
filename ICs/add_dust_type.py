#!/usr/bin/env python3

"""
add_dust_type.py

Prepare a MUSIC2/Arepo-format HDF5 initial-condition file for CosmicGrain
(GADGET-4 with NTYPES=7) by adding an empty PartType6 group and extending
the standard 6-entry Header particle-type arrays to length 7.

The script is safe to run in-place and is idempotent:
- If Header arrays already have length 7, they are validated.
- If PartType6 already exists, its required empty datasets are validated.
- Existing non-dust particle data are left untouched.

Usage:
    python3 add_dust_type.py IC_file.hdf5
    python3 add_dust_type.py input.hdf5 output.hdf5
"""

import argparse
import os
import shutil
import sys

import h5py
import numpy as np


DUST_TYPE = 6


def extend_to_seven(arr, fill_value):
    """Return a 7-element copy of a particle-type header array."""
    a = np.asarray(arr)

    if a.ndim != 1:
        raise RuntimeError(
            f"Expected a 1-D Header particle-type array, got shape {a.shape}"
        )

    if len(a) == 7:
        return a.copy()

    if len(a) != 6:
        raise RuntimeError(
            f"Expected Header particle-type array of length 6 or 7, got {len(a)}"
        )

    out = np.empty(7, dtype=a.dtype)
    out[:6] = a
    out[6] = np.asarray(fill_value, dtype=a.dtype)
    return out


def replace_attr(attrs, name, value):
    """
    Replace an HDF5 attribute even when its shape changes.

    h5py AttributeManager.modify() preserves the existing attribute's
    shape/type and therefore cannot reliably change a 6-element Header
    array into a 7-element one. Delete and recreate instead.
    """
    if name in attrs:
        del attrs[name]
    attrs.create(name, value)


def ensure_header_ntypes7(header):
    attrs = header.attrs

    required = ("NumPart_ThisFile", "NumPart_Total", "MassTable")
    for name in required:
        if name not in attrs:
            raise RuntimeError(f"Header is missing required attribute '{name}'")

    npart_this = np.asarray(attrs["NumPart_ThisFile"])
    npart_tot = np.asarray(attrs["NumPart_Total"])
    mass_table = np.asarray(attrs["MassTable"])

    lengths = {len(npart_this), len(npart_tot), len(mass_table)}

    if lengths == {6}:
        print("Extending Header arrays from 6 to 7 particle types...")

        replace_attr(
            attrs,
            "NumPart_ThisFile",
            extend_to_seven(npart_this, 0),
        )
        replace_attr(
            attrs,
            "NumPart_Total",
            extend_to_seven(npart_tot, 0),
        )
        replace_attr(
            attrs,
            "MassTable",
            extend_to_seven(mass_table, 0.0),
        )

        if "NumPart_Total_HighWord" in attrs:
            replace_attr(
                attrs,
                "NumPart_Total_HighWord",
                extend_to_seven(
                    np.asarray(attrs["NumPart_Total_HighWord"]), 0
                ),
            )

    elif lengths == {7}:
        print("Header arrays already contain 7 particle types.")
    else:
        raise RuntimeError(
            "Header particle-type arrays have inconsistent lengths: "
            f"NumPart_ThisFile={len(npart_this)}, "
            f"NumPart_Total={len(npart_tot)}, "
            f"MassTable={len(mass_table)}"
        )

    # Re-read after replacement.
    npart_this = np.asarray(attrs["NumPart_ThisFile"])
    npart_tot = np.asarray(attrs["NumPart_Total"])
    mass_table = np.asarray(attrs["MassTable"])

    if len(npart_this) != 7 or len(npart_tot) != 7 or len(mass_table) != 7:
        raise RuntimeError("Failed to create 7-entry Header arrays")

    if int(npart_this[DUST_TYPE]) != 0:
        raise RuntimeError(
            f"Header NumPart_ThisFile[6] must be 0, got {npart_this[DUST_TYPE]}"
        )

    if int(npart_tot[DUST_TYPE]) != 0:
        raise RuntimeError(
            f"Header NumPart_Total[6] must be 0, got {npart_tot[DUST_TYPE]}"
        )

    if float(mass_table[DUST_TYPE]) != 0.0:
        raise RuntimeError(
            f"Header MassTable[6] must be 0, got {mass_table[DUST_TYPE]}"
        )

    if "NumPart_Total_HighWord" in attrs:
        high = np.asarray(attrs["NumPart_Total_HighWord"])

        if len(high) == 6:
            replace_attr(
                attrs,
                "NumPart_Total_HighWord",
                extend_to_seven(high, 0),
            )
            high = np.asarray(attrs["NumPart_Total_HighWord"])

        if len(high) != 7:
            raise RuntimeError(
                "Header NumPart_Total_HighWord is not length 7"
            )

        if int(high[DUST_TYPE]) != 0:
            raise RuntimeError(
                "Header NumPart_Total_HighWord[6] must be 0"
            )


def create_empty_dataset(group, name, shape, dtype):
    """Create or validate an empty dataset."""
    if name in group:
        ds = group[name]
        if ds.shape != shape:
            raise RuntimeError(
                f"{group.name}/{name} exists with shape {ds.shape}, "
                f"expected {shape}"
            )
        return

    group.create_dataset(name, shape=shape, dtype=dtype)


def ensure_parttype6(h5):
    """
    Add an empty PartType6 group with the minimal standard particle datasets
    expected by the CosmicGrain IC reader.

    The file contains zero initial dust particles; CosmicGrain creates them
    during the simulation.
    """
    if "PartType6" in h5:
        print("PartType6 already exists; validating it...")
        dust = h5["PartType6"]
    else:
        print("Creating empty PartType6 group...")
        dust = h5.create_group("PartType6")

    # Match common MUSIC2/Arepo/GADGET datatypes where possible.
    coord_dtype = np.float32
    vel_dtype = np.float32
    mass_dtype = np.float32
    id_dtype = np.uint64

    # Infer datatypes from existing high-resolution particles when available.
    template_group = None
    for candidate in ("PartType1", "PartType0", "PartType2"):
        if candidate in h5:
            template_group = h5[candidate]
            break

    if template_group is not None:
        if "Coordinates" in template_group:
            coord_dtype = template_group["Coordinates"].dtype
        if "Velocities" in template_group:
            vel_dtype = template_group["Velocities"].dtype
        if "Masses" in template_group:
            mass_dtype = template_group["Masses"].dtype
        if "ParticleIDs" in template_group:
            id_dtype = template_group["ParticleIDs"].dtype

    create_empty_dataset(
        dust, "Coordinates", (0, 3), coord_dtype
    )
    create_empty_dataset(
        dust, "Velocities", (0, 3), vel_dtype
    )
    create_empty_dataset(
        dust, "ParticleIDs", (0,), id_dtype
    )
    create_empty_dataset(
        dust, "Masses", (0,), mass_dtype
    )


def validate_file(path):
    with h5py.File(path, "r") as h5:
        if "Header" not in h5:
            raise RuntimeError("Missing Header group")

        attrs = h5["Header"].attrs

        for name in ("NumPart_ThisFile", "NumPart_Total", "MassTable"):
            if name not in attrs:
                raise RuntimeError(f"Missing Header/{name}")
            if len(np.asarray(attrs[name])) != 7:
                raise RuntimeError(
                    f"Header/{name} is not length 7"
                )

        if int(np.asarray(attrs["NumPart_ThisFile"])[6]) != 0:
            raise RuntimeError("NumPart_ThisFile[6] is not zero")

        if int(np.asarray(attrs["NumPart_Total"])[6]) != 0:
            raise RuntimeError("NumPart_Total[6] is not zero")

        if float(np.asarray(attrs["MassTable"])[6]) != 0.0:
            raise RuntimeError("MassTable[6] is not zero")

        if "NumPart_Total_HighWord" in attrs:
            high = np.asarray(attrs["NumPart_Total_HighWord"])
            if len(high) != 7 or int(high[6]) != 0:
                raise RuntimeError(
                    "NumPart_Total_HighWord is not valid for Type6"
                )

        if "PartType6" not in h5:
            raise RuntimeError("Missing PartType6 group")

        dust = h5["PartType6"]

        expected = {
            "Coordinates": (0, 3),
            "Velocities": (0, 3),
            "ParticleIDs": (0,),
            "Masses": (0,),
        }

        for name, shape in expected.items():
            if name not in dust:
                raise RuntimeError(f"Missing PartType6/{name}")

            if dust[name].shape != shape:
                raise RuntimeError(
                    f"PartType6/{name} has shape {dust[name].shape}, "
                    f"expected {shape}"
                )


def process_file(input_path, output_path):
    input_path = os.path.abspath(input_path)
    output_path = os.path.abspath(output_path)

    if not os.path.isfile(input_path):
        raise RuntimeError(f"Input file does not exist: {input_path}")

    in_place = input_path == output_path

    print()
    print(f"Adding PartType6 support to: {input_path}")
    print("Mode: in-place" if in_place else f"Output: {output_path}")

    if not in_place:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        shutil.copy2(input_path, output_path)

    with h5py.File(output_path, "r+") as h5:
        if "Header" not in h5:
            raise RuntimeError("HDF5 file does not contain a Header group")

        ensure_header_ntypes7(h5["Header"])
        ensure_parttype6(h5)

        h5.flush()

    validate_file(output_path)

    print("CosmicGrain Type6 preparation complete.")
    print("Validation passed.")
    print()


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extend a 6-type MUSIC2/GADGET HDF5 IC to CosmicGrain "
            "NTYPES=7 and add an empty PartType6 group."
        )
    )

    parser.add_argument(
        "input",
        help="Input HDF5 initial-condition file",
    )

    parser.add_argument(
        "output",
        nargs="?",
        default=None,
        help=(
            "Optional output file. If omitted, the input file is modified "
            "in place."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    output = args.output if args.output is not None else args.input

    try:
        process_file(args.input, output)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
