#!/usr/bin/env python3
"""
generate_params.py
-------------------
Generate Gadget4 parameter files by merging a shared base parameter file
with resolution-dependent overrides and per-run overrides.

This supports:
  - the S0-S10 dust-physics ladder,
  - resolution-convergence runs, and
  - the 12-halo CosmicGrain science suite at 512/1024/2048/4096 resolution.

Workflow:
  1. Edit param_base.txt for any shared change (yields, timescales, etc.)
  2. Edit RES_OVERRIDES below for anything that scales with resolution
  3. Edit GRID below to adjust the physics ladder / convergence runs
  4. Edit HALO_IDS below to adjust the multi-halo science suite
  5. Run:  python generate_params.py
  6. All generated param files are regenerated automatically

Usage:
    python generate_params.py               # generate all
    python generate_params.py S3_512        # regenerate one physics-ladder label
    python generate_params.py halo295_512   # regenerate one halo/resolution file
    python generate_params.py --dry-run     # print diffs, write nothing

Resolution-dependent parameters (v1.2.0)
-----------------------------------------
Dust process calibrations are timescale multipliers (larger = slower) and use
the Paper II recipe at every resolution: Coag 22.3, Shat 1.485, Growth 1.12,
with DustEnableClumping 2 (Blitz & Rosolowsky dense phase).  This recipe was
calibrated on halo441 at 1024^3 against the Zhang & Green R_V structure
function and the Hensley & Draine (2023) grain-size distribution; it has not
been separately calibrated at 512^3, 2048^3 or 4096^3.

What does change with resolution: softenings (eps ∝ N^-1/3), the timestep
cap, DustParticlesPerSNII/AGB (10 at 512^3, 25 above), and DustUpdateCadence
(5 at 512^3, 7 above).  CritPhysDensity 0.3, DustShockAmbientDensity 0.5 and
DustCollisionDensityThresh 15 (legacy clumping modes only) are the values used
by the halo441 runs at 512^3, 1024^3 and 2048^3.  4096^3 has never been run.
"""

import sys
import re
import argparse
import datetime
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
# Resolve paths from this script rather than from the caller's working
# directory, so both of these work identically:
#   python scripts/generate_params.py ...
#   cd scripts && python generate_params.py ...
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent
BASE_FILE  = REPO_ROOT / "params" / "param_base.txt"
OUTPUT_DIR = REPO_ROOT / "params"

# ── Dust switch names (must match order in param_base.txt) ────────────────────
ALL_SWITCHES = [
    "DustEnableCreation",
    "DustEnableCooling",
    "DustEnableDrag",
    "DustEnableAstration",
    "DustEnableSputtering",
    "DustEnableGrowth",
    "DustEnableClumping",
    "DustEnableShockDestruction",
    "DustEnableCoagulation",
    "DustEnableShattering",
    "DustEnableRadiationPressure",
]

# ── Per-resolution shared overrides ───────────────────────────────────────────
# These are merged BEFORE per-run dust switches, so dust switches always win.
#
# Softening scaling rule: ε ∝ N^{-1/3}, i.e. halve every time linear
# resolution doubles.  512→1024: ×0.5.  512→2048: ×0.25.  512→4096: ×0.125.
#
# MUSIC2 PartType2 may contain several coarse mass levels.  For the multi-halo
# zoom suite compile Config_zoom.sh with INDIVIDUAL_GRAVITY_SOFTENING=4 so
# PartType2 selects from the available collisionless softening classes by mass.
#
# Dust calibrations, sampling and thresholds: see the module docstring.
#
# MinGasHsml is fixed globally at 0 in param_base.txt.
# Controlled CG10 runs showed that an imposed nonzero floor can alter the
# hydrodynamic evolution, whereas MinGasHsml=0 is stable at 512³, 1024³,
# and 2048³. It is therefore not treated as resolution dependent.
RES_OVERRIDES = {
    # ── 512³ ──────────────────────────────────────────────────────────────────
    # m_gas ~ 1e7 Msun  |  softenings are the baseline
    "512": {
        "InitCondFile":               "ICs/IC_zoom_512_halo569_50Mpc_music_ellipsoid_with_dust",

        "CritPhysDensity":            "0.3",
        "DustCollisionDensityThresh": "15.0",
        "DustShockAmbientDensity":    "0.5",

        # Calibration
        "DustGrowthCalibration":      "1.12",
        "DustShatteringCalibration":  "1.485",
        "DustCoagulationCalibration": "22.3",  # Paper II recipe; not separately calibrated at 512

        # Dust sampling
        "DustParticlesPerSNII":       "10",
        "DustParticlesPerAGB":        "10",
        "DustParticlesPerLRN":        "1",

        # Gravitational softenings (comoving kpc) — baseline
        "SofteningComovingClass0":    "6.0",    # Gas (adaptive backup floor)
        "SofteningMaxPhysClass0":     "3.0",
        "SofteningComovingClass1":    "12.0",   # HR DM
        "SofteningMaxPhysClass1":     "6.0",
        "SofteningComovingClass2":    "16.0",   # Intermediate DM
        "SofteningMaxPhysClass2":     "8.0",
        "SofteningComovingClass3":    "32.0",   # Coarse DM
        "SofteningMaxPhysClass3":     "16.0",
        "SofteningComovingClass4":    "6.0",    # Stars
        "SofteningMaxPhysClass4":     "3.0",
        "SofteningComovingClass5":    "64.0",   # Outermost DM
        "SofteningMaxPhysClass5":     "32.0",
        "SofteningComovingClass6":    "6.0",    # Dust (match stars)
        "SofteningMaxPhysClass6":     "3.0",
        "MinimumComovingHydroSoftening": "4.0",
        "MaxSizeTimestep":            "0.01",   # this was calibrated by trial and error (higher timestep produces fewer stars and dust)
        "DustUpdateCadence":          "5",
    },

    # ── 1024³ ─────────────────────────────────────────────────────────────────
    # New MUSIC2 halo-suite high-resolution masses:
    #   m_gas = 1.59470e6 Msun/h
    #   m_DM  = 8.60802e6 Msun/h
    #
    # In the new MUSIC2 zoom ICs, PartType2 can contain more than one coarse
    # particle mass.  The zoom executable should therefore be compiled with
    # INDIVIDUAL_GRAVITY_SOFTENING=4 (PartType2 only).  Classes 2/3/5 below
    # provide a discrete ladder of softenings for those mass levels rather than
    # representing one fixed Type2 mass.
    #
    # HR gas/stars/dust/DM softenings retain the established 1024 values.
    "1024": {
        "InitCondFile":               "ICs/IC_zoom_1024_halo569_50Mpc_music_ellipsoid_with_dust",

        # Star formation
        "CritPhysDensity": "0.3",

        "DustCollisionDensityThresh": "15.0",
        "DustShockAmbientDensity":    "0.5",

        # Calibration
        # v1.2.0 Paper II calibration (halo441 1024, DustEnableClumping 2)
        "DustGrowthCalibration":      "1.12",
        "DustShatteringCalibration":  "1.485",
        "DustCoagulationCalibration": "22.3",

        # Dust sampling
        "DustParticlesPerSNII":       "25",
        "DustParticlesPerAGB":        "25",
        "DustParticlesPerLRN":        "1",

        # Softenings halved
        "SofteningComovingClass0":    "3.0",
        "SofteningMaxPhysClass0":     "1.5",
        "SofteningComovingClass1":    "6.0",
        "SofteningMaxPhysClass1":     "3.0",
        "SofteningComovingClass2":    "12.0",   # level-9 coarse/background DM
        "SofteningMaxPhysClass2":     "6.0",
        "SofteningComovingClass3":    "16.0",
        "SofteningMaxPhysClass3":     "8.0",
        "SofteningComovingClass4":    "3.0",
        "SofteningMaxPhysClass4":     "1.5",
        "SofteningComovingClass5":    "32.0",
        "SofteningMaxPhysClass5":     "16.0",
        "SofteningComovingClass6":    "3.0",
        "SofteningMaxPhysClass6":     "1.5",
        "MinimumComovingHydroSoftening": "2.0",
        "MaxSizeTimestep":            "0.00425",
    },

    # ── 2048³ ─────────────────────────────────────────────────────────────────
    # m_gas ~ 1.5e5 Msun  |  softenings × 0.25 vs 512³
    #
    "2048": {
        "InitCondFile":               "ICs/IC_zoom_2048_halo569_50Mpc_music_ellipsoid_with_dust",

        # Star formation
        "CritPhysDensity":            "0.3",

        # Dust collision / shock thresholds
        "DustCollisionDensityThresh": "15",
        "DustShockAmbientDensity":    "0.5",

        # Calibration
        "DustGrowthCalibration":      "1.12",
        "DustShatteringCalibration":  "1.485",
        "DustCoagulationCalibration": "22.3",  # Paper II recipe; not separately calibrated at 2048

        # Dust sampling: fixed production multiplicities
        "DustParticlesPerSNII":       "25",
        "DustParticlesPerAGB":        "25",
        "DustParticlesPerLRN":        "1",

        # Softenings quartered
        "SofteningComovingClass0":    "1.5",
        "SofteningMaxPhysClass0":     "0.75",
        "SofteningComovingClass1":    "3.0",
        "SofteningMaxPhysClass1":     "1.5",
        "SofteningComovingClass2":    "6.0",
        "SofteningMaxPhysClass2":     "3.0",
        "SofteningComovingClass3":    "8.0",
        "SofteningMaxPhysClass3":     "4.0",
        "SofteningComovingClass4":    "1.5",
        "SofteningMaxPhysClass4":     "0.75",
        "SofteningComovingClass5":    "16.0",
        "SofteningMaxPhysClass5":     "8.0",
        "SofteningComovingClass6":    "1.5",
        "SofteningMaxPhysClass6":     "0.75",
        "MinimumComovingHydroSoftening": "1.0",
        "MaxSizeTimestep":            "0.00425",
    },

    # ── 4096³ ─────────────────────────────────────────────────────────────────
    # m_gas ~ 1.9e4 Msun  |  softenings × 0.125 vs 512³
    #
    # MinimumComovingHydroSoftening = 0.5 ckpc
    #   Halved from 2048³ consistent with the ε ∝ N^{-1/3} rule. At 4096³
    #   the SPH kernel will be adaptive down to this floor; ensure it does not
    #   fall below ~2–3× the interparticle spacing in the densest ISM cells.
    "4096": {
        "InitCondFile":               "ICs/IC_zoom_4096_halo569_50Mpc_music_ellipsoid_with_dust",

        # Star formation
        "CritPhysDensity":            "0.3",

        # Dust collision / shock thresholds
        "DustCollisionDensityThresh": "15",
        "DustShockAmbientDensity":    "0.5",

        # Calibration
        "DustGrowthCalibration":      "1.12",
        "DustShatteringCalibration":  "1.485",
        "DustCoagulationCalibration": "22.3",  # Paper II recipe; 4096 never run

        # Dust sampling
        "DustParticlesPerSNII":       "25",
        "DustParticlesPerAGB":        "25",
        "DustParticlesPerLRN":        "1",

        # Gravitational softenings (comoving kpc) — ×0.125 vs 512³ baseline
        "SofteningComovingClass0":    "0.75",   # Gas
        "SofteningMaxPhysClass0":     "0.375",
        "SofteningComovingClass1":    "1.5",    # HR DM
        "SofteningMaxPhysClass1":     "0.75",
        # PartType2 classes chosen by mass (INDIVIDUAL_GRAVITY_SOFTENING=4): the four coarse
        # levels shared with 2048 get the same softenings as there (6, 16); the new finest level 3.
        "SofteningComovingClass2":    "3.0",
        "SofteningMaxPhysClass2":     "1.5",
        "SofteningComovingClass3":    "6.0",
        "SofteningMaxPhysClass3":     "3.0",
        "SofteningComovingClass4":    "0.75",   # Stars
        "SofteningMaxPhysClass4":     "0.375",
        "SofteningComovingClass5":    "16.0",
        "SofteningMaxPhysClass5":     "8.0",
        "SofteningComovingClass6":    "0.75",   # Dust (match stars)
        "SofteningMaxPhysClass6":     "0.375",
        "MinimumComovingHydroSoftening": "0.5",
        "MaxSizeTimestep":            "0.00425",
    },
}

# ── Simulation grid ───────────────────────────────────────────────────────────
# Each entry: label → dict of parameter overrides.
#
# Physics-ladder / convergence entries typically override OutputDir and
# DustEnable* switches. Halo-suite entries additionally override InitCondFile.
# Everything else comes from param_base.txt or RES_OVERRIDES above.

def make_switches(n_active):
    """Return a dict with the first n_active switches ON, rest OFF."""
    on = {"DustEnableClumping": "2"}   # ON = Blitz & Rosolowsky dense phase (v1.2.0 default)
    return {sw: (on.get(sw, "1") if i < n_active else "0")
            for i, sw in enumerate(ALL_SWITCHES)}

GRID = {
    # ── 512³ physics ladder ──────────────────────────────────────────────────
    "S0_512":  {**make_switches(1),  **{"OutputDir": "S0_output_512"}},
    "S1_512":  {**make_switches(2),  **{"OutputDir": "S1_output_512"}},
    "S2_512":  {**make_switches(3),  **{"OutputDir": "S2_output_512"}},
    "S3_512":  {**make_switches(4),  **{"OutputDir": "S3_output_512"}},
    "S4_512":  {**make_switches(5),  **{"OutputDir": "S4_output_512"}},
    "S5_512":  {**make_switches(6),  **{"OutputDir": "S5_output_512"}},
    "S6_512":  {**make_switches(7),  **{"OutputDir": "S6_output_512"}},
    "S7_512":  {**make_switches(8),  **{"OutputDir": "S7_output_512"}},
    "S8_512":  {**make_switches(9),  **{"OutputDir": "S8_output_512"}},
    "S9_512":  {**make_switches(10), **{"OutputDir": "S9_output_512"}},
    "S10_512": {**make_switches(11), **{"OutputDir": "S10_output_512"}},

    # ── 1024³ — promoted runs ────────────────────────────────────────────────
    "S0_1024":  {**make_switches(1),  **{"OutputDir": "S0_output_1024"}},
    "S1_1024":  {**make_switches(2),  **{"OutputDir": "S1_output_1024"}},
    "S2_1024":  {**make_switches(3),  **{"OutputDir": "S2_output_1024"}},
    "S3_1024":  {**make_switches(4),  **{"OutputDir": "S3_output_1024"}},
    "S4_1024":  {**make_switches(5),  **{"OutputDir": "S4_output_1024"}},
    "S5_1024":  {**make_switches(6),  **{"OutputDir": "S5_output_1024"}},
    "S6_1024":  {**make_switches(7),  **{"OutputDir": "S6_output_1024"}},
    "S7_1024":  {**make_switches(8),  **{"OutputDir": "S7_output_1024"}},
    "S8_1024":  {**make_switches(9),  **{"OutputDir": "S8_output_1024"}},
    "S9_1024":  {**make_switches(10), **{"OutputDir": "S9_output_1024"}},
    "S10_1024": {**make_switches(11), **{"OutputDir": "S10_output_1024"}},

    # ── 2048³ — resolution convergence ──────────────────────────────────────
    "S0_2048":  {**make_switches(1),  **{"OutputDir": "S0_output_2048"}},
    "S10_2048": {**make_switches(11), **{"OutputDir": "S10_output_2048"}},

    # ── 4096³ — extreme resolution convergence ───────────────────────────────
    # Only S0 (gravity+creation baseline) and S10 (full physics) are run at
    # this tier. Add intermediate rungs here if convergence tests warrant it.
    # IMPORTANT: generate ICs with MUSIC2 before submitting these jobs.
    #   Stampede allocation: expect ~4–8× the 2048³ node-hours per snapshot.
    #   Suggested node count: 80–160 nodes, 24–36 tasks/node, MaxMemSize 4800.
    "S0_4096":  {**make_switches(1),  **{"OutputDir": "S0_output_4096"}},
    "S10_4096": {**make_switches(11), **{"OutputDir": "S10_output_4096"}},
}


# ── 12-halo CosmicGrain science suite ───────────────────────────────────────
#
# Every halo/resolution combination gets its own ready-to-run generated parameter
# file, but these files are NOT maintained independently.  They all inherit:
#
#   - the full S10 dust-physics model (all DustEnable* switches ON),
#   - the same RES_OVERRIDES for a given resolution, and
#   - the same shared physics/cosmology/output settings from param_base.txt.
#
# Thus the only halo-specific values are InitCondFile and OutputDir.  This gives
# convenient one-command launching while preserving identical numerical/physical
# settings between halos at fixed resolution.
#
# Generated naming convention:
#   params/param_halo295_512.txt
#   params/param_halo1653_1024.txt   (CG07 replacement)
#   ...
#   params/param_halo9235_4096.txt
#
# MUSIC2 IC naming convention:
#   ICs/halo<ID>/IC_halo<ID>_zoom_<RES>.hdf5
#
# Gadget InitCondFile intentionally omits the .hdf5 suffix.

HALO_IDS = [
    295, 308, 441,
    859, 1481, 1653,
    3352, 3879, 3886,
    5834, 7723, 9235,
]

HALO_RESOLUTIONS = ("512", "1024", "2048", "4096")

for halo_id in HALO_IDS:
    for res in HALO_RESOLUTIONS:
        label = f"halo{halo_id}_{res}"
        output_dir = f"halo{halo_id}_output_{res}"
        if halo_id == 441 and res == "1024":
            output_dir = "halo441_output_1024_paperII"
        GRID[label] = {
            **make_switches(11),
            "InitCondFile": f"ICs/halo{halo_id}/IC_halo{halo_id}_zoom_{res}",
            "OutputDir": output_dir,
        }


# ── Core logic ────────────────────────────────────────────────────────────────

def parse_base(path: Path) -> list:
    """
    Parse param file into an ordered list of (raw_line, key_or_None) tuples.
    Preserves comments, blank lines, and section headers exactly.
    """
    lines = []
    for raw in path.read_text().splitlines(keepends=True):
        stripped = raw.strip()
        if not stripped or stripped.startswith("%"):
            lines.append((raw, None))
            continue
        m = re.match(r'^(\w+)\s+\S+', stripped)
        if m:
            lines.append((raw, m.group(1)))
        else:
            lines.append((raw, None))
    return lines


def apply_overrides(parsed_lines: list, overrides: dict) -> tuple:
    applied = set()
    new_lines = []
    for raw, key in parsed_lines:
        if key and key in overrides:
            # Preserve the original spacing between key and value
            spacing_match = re.match(r'^(\w+)(\s+)\S+', raw)
            spacing = spacing_match.group(2) if spacing_match else "  "
            comment_match = re.search(r'(%.*)', raw)
            comment = "  " + comment_match.group(1) if comment_match else ""
            new_raw = f"{key}{spacing}{overrides[key]}{comment}\n"
            new_lines.append((new_raw, key))
            applied.add(key)
        else:
            new_lines.append((raw, key))

    missing = set(overrides.keys()) - applied
    return new_lines, missing


def get_res_overrides(label: str) -> dict:
    """Extract resolution from label suffix and return matching RES_OVERRIDES."""
    for res, overrides in RES_OVERRIDES.items():
        if label.endswith(f"_{res}"):
            return dict(overrides)
    return {}


def generate_one(label: str, overrides: dict, parsed_base: list) -> str:
    """Generate param file for one label. Returns the file content."""
    # Merge: res_overrides < per-run overrides (per-run wins)
    merged = {**get_res_overrides(label), **overrides}

    lines, missing = apply_overrides(parsed_base, merged)

    extra = []
    if missing:
        extra.append(f"\n%---- Auto-appended by generate_params.py for {label} ----\n")
        for k in sorted(missing):
            extra.append(f"{k:<40} {merged[k]}\n")

    content = "".join(l for l, _ in lines) + "".join(extra)

    header = (
        f"% Auto-generated by generate_params.py — DO NOT EDIT DIRECTLY\n"
        f"% Label   : {label}\n"
        f"% Edit param_base.txt or generate_params.py, then re-run generator\n"
        f"% Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"%\n"
    )
    return header + content


def diff_summary(label: str, merged: dict) -> str:
    """Human-readable summary of what differs from base for this label."""
    lines = [f"\n  {label}:"]
    for k, v in sorted(merged.items()):
        lines.append(f"    {k:<40} = {v}")
    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("labels", nargs="*",
                        help="Labels to generate (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would change, write nothing")
    parser.add_argument("--base", default=str(BASE_FILE),
                        help=f"Base param file (default: {BASE_FILE})")
    args = parser.parse_args()

    base_path = Path(args.base)
    if not base_path.exists():
        sys.exit(f"ERROR: base param file not found: {base_path}")

    parsed_base = parse_base(base_path)
    print(f"Base file: {base_path}  ({len(parsed_base)} lines)")

    labels = args.labels if args.labels else list(GRID.keys())
    unknown = [l for l in labels if l not in GRID]
    if unknown:
        sys.exit(f"ERROR: unknown label(s): {', '.join(unknown)}\n"
                 f"Known: {', '.join(GRID.keys())}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for label in labels:
        overrides = GRID[label]
        merged    = {**get_res_overrides(label), **overrides}
        content   = generate_one(label, overrides, parsed_base)
        out_path  = OUTPUT_DIR / f"param_{label}.txt"

        if args.dry_run:
            print(diff_summary(label, merged))
        else:
            out_path.write_text(content)
            print(f"  Written: {out_path}")

    if args.dry_run:
        print(f"\n(dry run — nothing written)")
    else:
        print(f"\nGenerated {len(labels)} param file(s) in {OUTPUT_DIR}/")
        print("Re-run any time you change param_base.txt, RES_OVERRIDES, or GRID.")


if __name__ == "__main__":
    main()
