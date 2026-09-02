#!/usr/bin/env python3
"""
export_skirt_inputs.py

Export CosmicGrain (Gadget-4) snapshot data to SKIRT 9 particle-import format.

Produces text files per snapshot in <output_dir>/<label>/:
  stars.txt          — PartType4 star particles   (position, smoothing, mass, Z, age)
  gas.txt             — PartType0 gas particles    (position, smoothing, mass, Z, SFR)
  dust.txt            — PartType6 dust superparticles, ALL channels combined
                         (position, smoothing, mass, f_carbon, grain_radius_nm)
  dust_silicate.txt   — dust.txt split by (1 - f_carbon), SNII + AGB + LRN combined
  dust_carbon.txt     — dust.txt split by f_carbon, SNII + AGB + LRN combined
  dust_lrn.txt         — LRN-channel dust ONLY (isolated for paper 2 with/without
                          comparison; not summed into the files above)

All positions/smoothing lengths are output in PHYSICAL PARSECS, masses in
Msun, ages in YEARS -- matching SKIRT's actual default per-column units
(confirmed via TextInFile.cpp and this build's own column log). Internally,
particle selection/aperture math works in physical kpc; conversion to pc
happens only at file-write time. An earlier draft wrote kpc/Gyr directly
with a single freeform header comment SKIRT doesn't parse (its regex
requires "# column N: description (unit)", one per line) -- this meant
positions were silently read as pc when they were actually kpc-scale
(1000x error), compounded with a separate unit bug that produced physical
Mpc instead of kpc (another 1000x), for a net 1e6x scale error. Fixed:
files now use the correct pc/yr values AND the structured header format
SKIRT actually recognizes, so units are both correct and self-declared.
Dust smoothing lengths are estimated from the N nearest neighbours (default 32)
since PartType6 does not store an SPH h directly.

Why dust_lrn.txt is separate
-----------------------------
CosmicGrain's dust source channels are distinguished by a per-particle
`GrainType` (or `DustCreationChannel`) field written at creation:
    1 -> SNII    2 -> AGB    3 -> LRN  (see dust_particle_log.cpp DUST_EVENT_* enum)
For the LRN-vs-no-LRN comparison in paper 2, the LRN contribution needs to be
importable into SKIRT as its own ParticleMedium so its effect on the mock SED
can be toggled on/off and its individual contribution reported, rather than
being folded silently into the silicate bucket. dust_lrn.txt uses the same
column layout as dust.txt and can be added as a fourth ParticleMedium block,
or omitted entirely to produce a "no-LRN" SED for comparison.

Usage
-----
  # Single snapshot (z=0, snap 049, 1024^3 S10 run):
  python export_skirt_inputs.py ../S10_output_1024/ 049

  # Range of snapshots:
  python export_skirt_inputs.py ../S10_output_1024/ 040 049

  # Specify aperture and output dir explicitly:
  python export_skirt_inputs.py ../S10_output_1024/ 049 \
      --aperture 30.0 --outdir ./skirt_inputs/

  # Use the halo's snapshot-specific R200 aperture (recommended for
  # comparing multiple redshifts):
  python export_skirt_inputs.py ../S10_output_1024/ 040 049 \
      --aperture-r200 --outdir ./skirt_inputs/

  # Skip the LRN channel entirely (e.g. if the field isn't present in an
  # older snapshot that predates the LRN channel being added):
  python export_skirt_inputs.py ../S10_output_1024/ 049 --no-lrn

Dependencies
------------
  numpy, h5py, scipy (for KD-tree smoothing length estimate)
  halo_utils   (your existing module — must be on PYTHONPATH)

SKIRT ski-file notes
---------------------
  stars.txt columns:  x y z h  M_init  Z  age
    Units:  kpc kpc kpc kpc  Msun  1  Gyr
    SKIRT component: ParticleSource with BruzualCharlotSEDFamily (or BPASS)

  gas.txt columns:    x y z h  M_gas  Z  SFR
    Units:  kpc kpc kpc kpc  Msun  1  Msun/yr
    SKIRT component: used only for birth-cloud subgrid if desired (optional)

  dust*.txt columns:  x y z h  M_dust  f_carbon  a_nm
    Units:  kpc kpc kpc kpc  Msun  1  nm
    SKIRT component: ParticleMedium with custom ConfigurableDustMix

  f_carbon and a_nm are non-standard SKIRT columns — a spatially-varying dust
  mix (or composition binning) is needed in the ski file. The silicate/carbon
  split below is the simplest working approach: two ParticleMedium blocks fed
  from dust_silicate.txt and dust_carbon.txt, each with fixed pure-composition
  optical constants. For the LRN with/without comparison, add or remove the
  dust_lrn.txt ParticleMedium block (itself pre-split into silicate-only
  optical constants, since the LRN channel has f_carbon = 0 by construction —
  see LRN_CARBON_FRACTION in dust_source_lrn.cc).
"""

import sys
import os
import argparse
import glob
import numpy as np
import h5py
from pathlib import Path

# ── optional KD-tree for dust smoothing lengths ─────────────────────────────
try:
    from scipy.spatial import cKDTree
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("WARNING: scipy not found — dust smoothing lengths will be estimated "
          "from mean inter-particle spacing (less accurate).")

# ── halo_utils — use your existing API ──────────────────────────────────────
try:
    from halo_utils import get_halo569_reference, get_halo569
    HALO_UTILS_AVAILABLE = True
except ImportError:
    HALO_UTILS_AVAILABLE = False
    print("WARNING: halo_utils not found — will use Group[0] as fallback.")

# GrainType / DustCreationChannel encoding (see dust_particle_log.cpp)
GRAINTYPE_SNII = 1
GRAINTYPE_AGB = 2
GRAINTYPE_LRN = 3

DEFAULT_APERTURE_PKPC = 30.0
DEFAULT_N_NEIGHBORS = 32


# ─────────────────────────────────────────────────────────────────────────
# Snapshot I/O helpers
# ─────────────────────────────────────────────────────────────────────────

def find_snapshot_files(sim_dir, snap_num):
    """Locate the (possibly multi-chunk) HDF5 files for a given snapshot number."""
    snap_tag = f"{snap_num:03d}"
    candidates = sorted(glob.glob(os.path.join(sim_dir, f"snapdir_{snap_tag}", f"snapshot_{snap_tag}.*.hdf5")))
    if not candidates:
        candidates = sorted(glob.glob(os.path.join(sim_dir, f"snapshot_{snap_tag}.hdf5")))
    if not candidates:
        candidates = sorted(glob.glob(os.path.join(sim_dir, f"snapshot_{snap_tag}.*.hdf5")))
    if not candidates:
        raise FileNotFoundError(f"No snapshot files found for snap {snap_tag} under {sim_dir}")
    return candidates


def read_header(files):
    with h5py.File(files[0], "r") as f:
        header = dict(f["Header"].attrs)
        params = dict(f["Parameters"].attrs)
    h = float(params["HubbleParam"])  # NOTE: lives under Parameters, not Header
    a = float(header["Time"])
    return header, params, h, a


def load_particle_field(files, ptype, field):
    """Concatenate a field across all chunk files for a given particle type.
    Returns None if the field/particle type is absent in every chunk."""
    chunks = []
    for fname in files:
        with h5py.File(fname, "r") as f:
            key = f"PartType{ptype}"
            if key not in f or field not in f[key]:
                continue
            chunks.append(f[key][field][()])
    if not chunks:
        return None
    return np.concatenate(chunks, axis=0)


# ─────────────────────────────────────────────────────────────────────────
# Halo center / aperture selection
# ─────────────────────────────────────────────────────────────────────────

def get_halo_center(sim_dir, snap_num, files, h, a, snap_num_z0=None):
    """Return (center_pkpc, r200_pkpc) using halo_utils if available,
    else fall back to the FOF Group[0] center.

    halo_utils requires a two-step call: get_halo569_reference() computes a
    reference position once (anchored to a fixed snapshot, snap_num_z0, so
    the same halo is tracked consistently across the whole simulation), and
    get_halo569() uses that reference to locate the halo at each individual
    snapshot. groups_dir is the FOF/SUBFIND catalog directory — its parent
    is treated internally as the simulation output_dir.
    """
    if HALO_UTILS_AVAILABLE:
        try:
            output_dir = Path(sim_dir)
            groups_dir = output_dir / f"groups_{snap_num:03d}"
            ref = get_halo569_reference(output_dir, snap_num_z0=snap_num_z0, verbose=False)
            result = get_halo569(groups_dir, snap_num, ref, verbose=False)
            if result is None:
                raise RuntimeError("get_halo569 returned None (halo not found near reference)")
            # NOTE: result["center"] is in comoving ckpc/h (raw code units), matching
            # the documented Halo 569 reference centers — NOT physical kpc despite the
            # plain key name. Convert with the same a/h/1000 factor used for particles.
            center_comoving = np.asarray(result["center"])
            # NOTE: raw * a / h alone gives genuine physical kpc (confirmed
            # against BoxSize=50000 raw units matching a 50 Mpc/h comoving
            # box). An earlier draft divided by an extra 1000 here, which
            # silently converted physical kpc into physical Mpc -- making
            # the aperture/selection math compare against numbers 1000x
            # too large. Fixed.
            center_pkpc = center_comoving * a / h
            r200_pkpc = float(result.get("r200_pkpc", np.nan))
            print(f"  [halo_utils] center = {center_pkpc} pkpc, r200 = {r200_pkpc:.2f} pkpc")
            return center_pkpc, r200_pkpc
        except Exception as exc:
            print(f"  [halo_utils] failed ({exc}); falling back to Group[0]")

    # Fallback: FOF Group[0] center, comoving -> physical.
    # NOTE: catalog data lives in groups_{snap:03d}/fof_subhalo_tab_*.hdf5,
    # NOT in the snapshot chunk files — snapshot files have no "Group" key.
    groups_dir = Path(sim_dir) / f"groups_{snap_num:03d}"
    catalog_chunks = sorted(groups_dir.glob(f"fof_subhalo_tab_{snap_num:03d}*.hdf5"))
    if not catalog_chunks:
        catalog_chunks = sorted(groups_dir.glob(f"fof_tab_{snap_num:03d}*.hdf5"))
    if not catalog_chunks:
        raise RuntimeError(f"No halo_utils and no FOF catalog found in {groups_dir} — cannot determine center.")
    with h5py.File(catalog_chunks[0], "r") as f:
        if "Group" not in f or "GroupPos" not in f["Group"] or len(f["Group"]["GroupPos"]) == 0:
            raise RuntimeError(f"Catalog chunk {catalog_chunks[0]} has no groups — cannot determine center.")
        group_pos_comoving = f["Group"]["GroupPos"][0]
    center_pkpc = np.asarray(group_pos_comoving) * a / h  # ckpc/h -> physical kpc
    print(f"  [fallback] Using Group[0] center at {center_pkpc} pkpc")
    return center_pkpc, np.nan


def select_within_aperture(pos_pkpc, center_pkpc, aperture_pkpc):
    d = np.linalg.norm(pos_pkpc - center_pkpc[None, :], axis=1)
    return d <= aperture_pkpc


def estimate_smoothing_lengths(pos_pkpc, n_neighbors=DEFAULT_N_NEIGHBORS):
    """Estimate an SPH-like smoothing length for particles lacking one
    (PartType6 dust superparticles) from the distance to the Nth nearest
    neighbor. Falls back to a mean inter-particle spacing estimate if
    scipy is unavailable."""
    n = len(pos_pkpc)
    if n == 0:
        return np.zeros(0)
    if SCIPY_AVAILABLE and n > n_neighbors:
        tree = cKDTree(pos_pkpc)
        dists, _ = tree.query(pos_pkpc, k=n_neighbors + 1)  # +1: includes self at d=0
        return dists[:, -1]
    # Fallback: crude uniform estimate from mean spacing within the bounding box
    box_vol = np.prod(np.ptp(pos_pkpc, axis=0)) if n > 1 else 1.0
    mean_spacing = (box_vol / max(n, 1)) ** (1.0 / 3.0)
    print("  WARNING: using uniform smoothing-length fallback (no scipy / too few particles)")
    return np.full(n, mean_spacing)


# ─────────────────────────────────────────────────────────────────────────
# Per-species export
# ─────────────────────────────────────────────────────────────────────────

def export_stars(files, center_pkpc, aperture_pkpc, h, a, out_path):
    pos = load_particle_field(files, 4, "Coordinates")
    if pos is None:
        print("  No PartType4 (stars) found — skipping stars.txt")
        return
    mass = load_particle_field(files, 4, "Masses")
    hsml = load_particle_field(files, 4, "SmoothingLength")
    metallicity = load_particle_field(files, 4, "Metallicity")
    sft = load_particle_field(files, 4, "StellarFormationTime")  # scale factor at formation

    # Internal working unit is physical kpc throughout selection (matches
    # center_pkpc/aperture_pkpc and the project's established R_ISM_PKPC
    # convention). Converted to physical PARSECS only at write time below,
    # since that's what SKIRT's ParticleSource expects for position/size
    # columns (confirmed via its own column log and BoxSize sanity check).
    pos_pkpc = pos * a / h
    mask = select_within_aperture(pos_pkpc, center_pkpc, aperture_pkpc)

    mass_msun = mass[mask] * 1.0e10 / h
    if hsml is not None:
        hsml_pkpc = hsml[mask] * a / h
    else:
        hsml_pkpc = estimate_smoothing_lengths(pos_pkpc[mask])

    # crude age from formation scale factor -> Gyr (flat LambdaCDM approx via header a)
    age_gyr = scale_factor_to_age_gyr(sft[mask], a)
    age_yr = age_gyr * 1.0e9  # SKIRT's default expected unit for age is yr

    z = metallicity[mask] if metallicity is not None else np.zeros(mask.sum())

    pos_pc = (pos_pkpc[mask] - center_pkpc) * 1000.0   # halo-relative physical kpc -> pc
    hsml_pc = hsml_pkpc * 1000.0

    data = np.column_stack([
        pos_pc[:, 0], pos_pc[:, 1], pos_pc[:, 2],
        hsml_pc, mass_msun, z, age_yr,
    ])
    columns = [
        ("position x", "pc"), ("position y", "pc"), ("position z", "pc"),
        ("size h", "pc"), ("initial mass", "Msun"), ("metallicity", "1"), ("age", "yr"),
    ]
    write_skirt_table(out_path, data, columns)
    print(f"  stars.txt: {mask.sum()} particles, {mass_msun.sum():.3e} Msun total")


def export_gas(files, center_pkpc, aperture_pkpc, h, a, out_path):
    pos = load_particle_field(files, 0, "Coordinates")
    if pos is None:
        print("  No PartType0 (gas) found — skipping gas.txt")
        return
    mass = load_particle_field(files, 0, "Masses")
    hsml = load_particle_field(files, 0, "SmoothingLength")
    metallicity = load_particle_field(files, 0, "Metallicity")
    sfr = load_particle_field(files, 0, "StarFormationRate")

    pos_pkpc = pos * a / h  # physical kpc for selection; converted to pc at write time
    mask = select_within_aperture(pos_pkpc, center_pkpc, aperture_pkpc)
    if sfr is not None:
        mask &= sfr > 0  # star-forming gas only

    mass_msun = mass[mask] * 1.0e10 / h
    hsml_pkpc = hsml[mask] * a / h
    z = metallicity[mask] if metallicity is not None else np.zeros(mask.sum())
    sfr_vals = sfr[mask] if sfr is not None else np.zeros(mask.sum())

    pos_pc = (pos_pkpc[mask] - center_pkpc) * 1000.0
    hsml_pc = hsml_pkpc * 1000.0

    data = np.column_stack([
        pos_pc[:, 0], pos_pc[:, 1], pos_pc[:, 2],
        hsml_pc, mass_msun, z, sfr_vals,
    ])
    columns = [
        ("position x", "pc"), ("position y", "pc"), ("position z", "pc"),
        ("size h", "pc"), ("mass", "Msun"), ("metallicity", "1"), ("SFR", "Msun/yr"),
    ]
    write_skirt_table(out_path, data, columns)
    print(f"  gas.txt: {mask.sum()} star-forming particles")


def export_dust(files, center_pkpc, aperture_pkpc, h, a, out_dir, include_lrn=True):
    """Writes dust.txt (all channels) and dust_lrn.txt (LRN channel only,
    NOT included in dust.txt's silicate/carbon split targets separately —
    see split_dust_components)."""
    pos = load_particle_field(files, 6, "Coordinates")
    if pos is None:
        print("  No PartType6 (dust) found — skipping dust.txt")
        return None

    mass = load_particle_field(files, 6, "Masses")
    f_carbon = load_particle_field(files, 6, "CarbonMassFraction")
    grain_radius_nm = load_particle_field(files, 6, "GrainRadius")
    grain_type = load_particle_field(files, 6, "GrainType")  # 1=SNII 2=AGB 3=LRN

    pos_pkpc = pos * a / h  # physical kpc for selection; converted to pc at write time
    mask = select_within_aperture(pos_pkpc, center_pkpc, aperture_pkpc)

    if grain_type is None and include_lrn:
        print("  WARNING: GrainType field not found — cannot isolate LRN dust "
              "(likely an older snapshot predating the LRN channel). "
              "dust_lrn.txt will not be written.")

    mass_msun = mass[mask] * 1.0e10 / h
    hsml_pkpc = estimate_smoothing_lengths(pos_pkpc[mask])
    fc = f_carbon[mask] if f_carbon is not None else np.zeros(mask.sum())
    a_nm = grain_radius_nm[mask] if grain_radius_nm is not None else np.full(mask.sum(), 10.0)

    pos_pc = (pos_pkpc[mask] - center_pkpc) * 1000.0
    hsml_pc = hsml_pkpc * 1000.0

    # NOTE: grain size (a_nm) unit is NOT yet verified against SKIRT's actual
    # ParticleMedium/dust import convention the way position/age were for
    # stars.txt -- that requires reaching a dust import step in a real run
    # and reading SKIRT's own column log, same methodology as before. Do not
    # trust nm vs micron here until confirmed the same way.
    all_data = np.column_stack([
        pos_pc[:, 0], pos_pc[:, 1], pos_pc[:, 2],
        hsml_pc, mass_msun, fc, a_nm,
    ])
    columns = [
        ("position x", "pc"), ("position y", "pc"), ("position z", "pc"),
        ("size h", "pc"), ("dust mass", "Msun"), ("carbon fraction", "1"),
        ("grain radius", "nm"),  # UNVERIFIED -- see note above
    ]
    dust_path = out_dir / "dust.txt"
    write_skirt_table(dust_path, all_data, columns)
    print(f"  dust.txt: {mask.sum()} particles, {mass_msun.sum():.3e} Msun total "
          f"(all channels combined)")

    lrn_path = None
    if include_lrn and grain_type is not None:
        gt = grain_type[mask]
        lrn_mask = gt == GRAINTYPE_LRN
        n_lrn = int(lrn_mask.sum())
        if n_lrn == 0:
            print("  dust_lrn.txt: 0 LRN-channel particles found in aperture — not written")
        else:
            lrn_data = all_data[lrn_mask]
            lrn_path = out_dir / "dust_lrn.txt"
            write_skirt_table(lrn_path, lrn_data, columns)
            m_lrn = mass_msun[lrn_mask].sum()
            m_tot = mass_msun.sum()
            print(f"  dust_lrn.txt: {n_lrn} particles, {m_lrn:.3e} Msun "
                  f"({100 * m_lrn / m_tot:.2f}% of total dust mass in aperture)")

    return dust_path, lrn_path


def split_dust_components(dust_path, out_dir, label=""):
    """Split a dust.txt-format file into dust_silicate{label}.txt and
    dust_carbon{label}.txt by (1 - f_carbon) and f_carbon respectively,
    for feeding two separate SKIRT ParticleMedium blocks with distinct
    fixed-composition optical constants.

    label lets this be reused for the LRN-only file, e.g.
      split_dust_components(dust_lrn_path, out_dir, label="_lrn")
    -> dust_silicate_lrn.txt, dust_carbon_lrn.txt
    (LRN grains are O-rich/silicate by construction, f_carbon = 0, so the
    carbon file will be empty — this is expected, not a bug.)
    """
    data = np.loadtxt(dust_path)
    if data.size == 0:
        print(f"  Split {Path(dust_path).name}: file is empty — skipping split")
        return None, None
    if data.ndim == 1:
        data = data[None, :]
    x, y, z, hh, M, fc, a_nm = data.T

    M_sil = M * (1.0 - fc)
    M_carb = M * fc

    sil_data = np.column_stack([x, y, z, hh, M_sil, np.zeros_like(fc), a_nm])
    carb_data = np.column_stack([x, y, z, hh, M_carb, np.ones_like(fc), a_nm])

    sil_path = out_dir / f"dust_silicate{label}.txt"
    carb_path = out_dir / f"dust_carbon{label}.txt"
    columns = [
        ("position x", "pc"), ("position y", "pc"), ("position z", "pc"),
        ("size h", "pc"), ("dust mass", "Msun"), ("carbon fraction", "1"),
        ("grain radius", "nm"),  # UNVERIFIED, see export_dust note
    ]
    write_skirt_table(sil_path, sil_data, columns)
    write_skirt_table(carb_path, carb_data, columns)

    M_tot = M.sum()
    M_s = M_sil.sum()
    M_c = M_carb.sum()
    print(f"  Split {dust_path.name}:")
    print(f"    Total:    {M_tot:.3e} Msun")
    print(f"    Silicate: {M_s:.3e} Msun ({100 * M_s / max(M_tot, 1e-30):.1f}%)  -> {sil_path.name}")
    print(f"    Carbon:   {M_c:.3e} Msun ({100 * M_c / max(M_tot, 1e-30):.1f}%)  -> {carb_path.name}")
    return sil_path, carb_path


# ─────────────────────────────────────────────────────────────────────────
# Small utilities
# ─────────────────────────────────────────────────────────────────────────

def scale_factor_to_age_gyr(sft, a_now, H0=67.7, Om0=0.31):
    """Rough flat-LambdaCDM lookback-time age estimate from formation scale
    factor. Adequate for SKIRT SED assignment; not for precision cosmology."""
    from numpy import sqrt, arcsinh

    def t_of_a(a):
        Ol0 = 1.0 - Om0
        H0_gyr = H0 * 1.0227e-3  # km/s/Mpc -> 1/Gyr
        return (2.0 / (3.0 * H0_gyr * sqrt(Ol0))) * arcsinh(sqrt(Ol0 / Om0) * a ** 1.5)

    return np.clip(t_of_a(a_now) - t_of_a(sft), 0.0, None)


def write_skirt_table(path, data, columns):
    """Write a SKIRT-importable text table with a properly structured header.

    `columns` is a list of (description, unit) tuples, one per column.
    Writes header lines in the exact format SKIRT's TextInFile.cpp parser
    requires (confirmed against the regex in getNextInfoLine):
        # column N: <description> (<unit>)
    A single freeform comment line (e.g. numpy.savetxt's default header
    behavior) does NOT match this regex and is silently ignored by SKIRT,
    which then falls back to positional default units -- this was the
    source of a real bug (positions read as pc when our data was actually
    kpc-scale, off by 1000x, compounded with a separate kpc/Mpc conversion
    bug to a net 1e6x error). Writing the real structured format removes
    the dependency on SKIRT's undocumented per-column default assumptions
    entirely -- the file is now self-describing and correct regardless of
    what any given SKIRT build assumes by default.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header_lines = [f"column {i + 1}: {desc} ({unit})" for i, (desc, unit) in enumerate(columns)]
    header = "\n".join(header_lines)
    np.savetxt(path, data, fmt="%.6e", header=header)


# ─────────────────────────────────────────────────────────────────────────
# Main driver
# ─────────────────────────────────────────────────────────────────────────

def process_snapshot(sim_dir, snap_num, aperture_pkpc, out_root, include_lrn=True, snap_num_z0=None, aperture_r200=False):
    print(f"\n=== Snapshot {snap_num:03d} ===")
    files = find_snapshot_files(sim_dir, snap_num)
    _, _, h, a = read_header(files)

    center_pkpc, r200_pkpc = get_halo_center(sim_dir, snap_num, files, h, a, snap_num_z0=snap_num_z0)

    # Optionally use the halo's snapshot-specific virial radius as the
    # selection aperture. This is resolved independently at each snapshot,
    # so a redshift sequence naturally follows the evolving R200.
    if aperture_r200:
        if not np.isfinite(r200_pkpc) or r200_pkpc <= 0:
            raise ValueError(
                f"Snapshot {snap_num:03d}: invalid R200={r200_pkpc}; "
                "--aperture-r200 requires a valid halo R200."
            )
        aperture_pkpc = float(r200_pkpc)
        print(f"  Using snapshot-specific R200 aperture: {aperture_pkpc:.2f} pkpc")
    else:
        aperture_pkpc = float(aperture_pkpc)
        print(f"  Using fixed aperture: {aperture_pkpc:.2f} pkpc")

    label = f"snap{snap_num:03d}"
    out_dir = Path(out_root) / label
    out_dir.mkdir(parents=True, exist_ok=True)

    export_stars(files, center_pkpc, aperture_pkpc, h, a, out_dir / "stars.txt")
    export_gas(files, center_pkpc, aperture_pkpc, h, a, out_dir / "gas.txt")
    result = export_dust(files, center_pkpc, aperture_pkpc, h, a, out_dir, include_lrn=include_lrn)

    if result is not None:
        dust_path, lrn_path = result
        split_dust_components(dust_path, out_dir)
        if lrn_path is not None:
            split_dust_components(lrn_path, out_dir, label="_lrn")

    write_summary(out_dir, snap_num, center_pkpc, r200_pkpc, aperture_pkpc)


def write_summary(out_dir, snap_num, center_pkpc, r200_pkpc, aperture_pkpc):
    with open(out_dir / "export_summary.txt", "w") as f:
        f.write(f"Snapshot: {snap_num:03d}\n")
        f.write(f"Center (physical kpc):\n")
        f.write(f"  x={center_pkpc[0]:.2f}  y={center_pkpc[1]:.2f}  "
                f"z={center_pkpc[2]:.2f}\n")
        f.write(f"R200: {r200_pkpc:.2f} pkpc\n")
        f.write(f"Aperture: {aperture_pkpc:.2f} pkpc\n")


def main():
    parser = argparse.ArgumentParser(
        description="Export CosmicGrain snapshots to SKIRT 9 particle-import format.")
    parser.add_argument("sim_dir", help="Path to simulation output directory")
    parser.add_argument("snap_start", type=int, help="Snapshot number (or start of range)")
    parser.add_argument("snap_end", type=int, nargs="?", default=None,
                         help="End of snapshot range (inclusive), optional")
    parser.add_argument("--aperture", type=float, default=DEFAULT_APERTURE_PKPC,
                         help=f"Selection aperture in physical kpc (default {DEFAULT_APERTURE_PKPC})")
    parser.add_argument("--aperture-r200", action="store_true",
                         help="Use the snapshot-specific R200 returned by halo_utils as the "
                              "selection aperture. Overrides --aperture and is recommended "
                              "for multi-redshift exports.")
    parser.add_argument("--outdir", type=str, default="./skirt_inputs",
                         help="Output root directory (default ./skirt_inputs)")
    parser.add_argument("--no-lrn", action="store_true",
                         help="Skip writing dust_lrn.txt even if the GrainType field is present")
    parser.add_argument("--snap-z0", type=int, default=None,
                         help="Snapshot number to anchor the Halo 569 reference position to "
                              "(passed as snap_num_z0 to get_halo569_reference). Defaults to "
                              "halo_utils' own default (typically the final/z=0 snapshot) if omitted.")
    args = parser.parse_args()

    snap_end = args.snap_end if args.snap_end is not None else args.snap_start
    for snap_num in range(args.snap_start, snap_end + 1):
        process_snapshot(
            args.sim_dir, snap_num, args.aperture, args.outdir,
            include_lrn=not args.no_lrn,
            snap_num_z0=args.snap_z0,
            aperture_r200=args.aperture_r200,
        )


if __name__ == "__main__":
    main()
