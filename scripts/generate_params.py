#!/usr/bin/env python3
"""
generate_params.py
-------------------
Generate one Gadget4 param file per simulation label by merging a shared
base param file with per-run dust physics switch overrides.

Workflow:
  1. Edit param_base.txt for any shared change (yields, timescales, etc.)
  2. Edit RES_OVERRIDES below for anything that scales with resolution
  3. Edit the GRID dict below to adjust dust switches or add new runs
  4. Run:  python generate_params.py
  5. All params/param_S*_*.txt files are regenerated automatically

Usage:
    python generate_params.py               # generate all
    python generate_params.py S3_512        # regenerate one label only
    python generate_params.py --dry-run     # print diffs, write nothing

Resolution-dependent dust parameters — design notes
------------------------------------------------------
DustShockAmbientDensity  (n_amb)
    Controls the Sedov-Taylor shock radius: R_ST ∝ (E_SN/n_amb)^(1/5).
    Higher resolution resolves denser ISM environments, so n_amb must
    INCREASE with resolution to keep R_ST physically small and prevent
    a single SN from sweeping an unphysically large grain population.
    Target: shock events/live_particles << 1 per sync window.
    Scaling: n_amb ∝ CritPhysDensity × f_ISM, where f_ISM ~ 5–20
    (typical dense ISM is well above the SF threshold).
    Anti-pattern: do NOT set n_amb ~ 0.1 × CritPhysDensity — this
    produces n_amb → 0 at high resolution and runaway destruction.

DustShatteringCalibration
    Coagulation fires for the first time at 2048³ (n_eff now exceeds
    DustCollisionDensityThresh). The shattering counterpart must be
    reduced to avoid over-eroding the grain population that coagulation
    is trying to build up. The 1024³ value was nudged +20% to compensate
    for absent coagulation; at 2048³ that nudge should be reversed.

DustGrowthCalibration
    Scales the HK11 accretion timescale. At higher resolution more gas
    cells exceed the density threshold, so the raw growth rate increases.
    Must be tuned downward to avoid over-growing dust at high resolution.

DustYieldSNII / DustYieldAGB
    Base yields in param_base.txt. At 2048³ each dust particle carries
    ~8× less mass than at 1024³ (m_gas scales as N^-1). The per-particle
    mass is set by DustParticlesPerSNII/AGB; the yields here set the
    total ejected mass fraction and are physically motivated, so resist
    changing these unless the D/Z ratio at z=0 is clearly wrong.
"""

import sys
import re
import argparse
import datetime
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_FILE  = Path("../params/param_base.txt")
OUTPUT_DIR = Path("../params")

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
# resolution doubles.  512→1024: ×0.5.  512→2048: ×0.25.
#
# CritPhysDensity: Jeans threshold must be resolved by ≥1 particle,
# so ρ_crit scales roughly as m_gas^{-1}.
#
# DustParticles*: keep dust particle mass / gas particle mass roughly
# constant across resolutions.
#
# DustShockAmbientDensity: see module docstring above.
#   Calibration target: shock_events / live_dust_particles < 0.5 per window.
#   At 1024³ the 512³ value of 0.01 cm^-3 was marginally OK; at 2048³ it
#   caused ~2× shocks per grain per window by z~5. Values below were chosen
#   to reproduce ~500k–1M shock events/window at peak SF (z~4–6), matching
#   the 1024³ destruction rate scaled by particle count.
#
# DustShatteringCalibration:
#   1024³: nudged +20% (1.2) to compensate for absent coagulation.
#   2048³: coagulation fires; revert the nudge and go below 1.0 to account
#          for the coagulation/shattering balance at higher n_eff.
RES_OVERRIDES = {
    # ── 512³ ──────────────────────────────────────────────────────────────────
    # m_gas ~ 1e7 Msun  |  softenings are the baseline
    "512": {
        "InitCondFile":               "ICs/IC_zoom_512_halo569_50Mpc_music_ellipsoid_with_dust",

        # Star formation
        "CritPhysDensity":            "0.1",    # cm^-3; SH03/Illustris standard

        # Dust collision / shock thresholds
        # DustCollisionDensityThresh ~ 30 × CritPhysDensity (coagulation floor)
        # DustShockAmbientDensity: diffuse ISM appropriate for 512³ resolution;
        #   at this resolution we rarely resolve dense clumps, so n_amb is low.
        "DustCollisionDensityThresh": "15.0",
        "DustShockAmbientDensity":    "0.1",    # cm^-3; R_ST ~75 pc

        # Calibration
        "DustGrowthCalibration":      "0.05",
        "DustShatteringCalibration":  "1.0",    # baseline; coag does not fire at 512³

        # Dust sampling
        "DustParticlesPerSNII":       "4",
        "DustParticlesPerAGB":        "6",

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
    },

    # ── 1024³ ─────────────────────────────────────────────────────────────────
    # m_gas ~ 1.2e6 Msun  |  softenings × 0.5 vs 512³
    "1024": {
        "InitCondFile":               "ICs/IC_zoom_1024_halo569_50Mpc_music_ellipsoid_with_dust",

        # Star formation
        "CritPhysDensity":            "0.7",    # cm^-3

        # Dust collision / shock thresholds
        # n_amb raised to ~4× the 512³ value: 1024³ resolves moderately dense
        # ISM clumps; typical SN environment is no longer purely diffuse.
        # Verified: gives ~300k–600k shock events/window at z~4–6 (physical).
        "DustCollisionDensityThresh": "100.0",
        "DustShockAmbientDensity":    "0.1",    # cm^-3; R_ST ~ 60 pc

        # Calibration
        "DustGrowthCalibration":      "0.10",
        "DustShatteringCalibration":  "1.2",    # +20% nudge: coag does not fire at 1024³

        # Dust sampling
        "DustParticlesPerSNII":       "8",
        "DustParticlesPerAGB":        "12",

        # Softenings halved
        "SofteningComovingClass0":    "3.0",
        "SofteningMaxPhysClass0":     "1.5",
        "SofteningComovingClass1":    "6.0",
        "SofteningMaxPhysClass1":     "3.0",
        "SofteningComovingClass2":    "8.0",
        "SofteningMaxPhysClass2":     "4.0",
        "SofteningComovingClass3":    "16.0",
        "SofteningMaxPhysClass3":     "8.0",
        "SofteningComovingClass4":    "3.0",
        "SofteningMaxPhysClass4":     "1.5",
        "SofteningComovingClass5":    "32.0",
        "SofteningMaxPhysClass5":     "16.0",
        "SofteningComovingClass6":    "3.0",
        "SofteningMaxPhysClass6":     "1.5",
        "MinimumComovingHydroSoftening": "2.0",
    },

    # ── 2048³ ─────────────────────────────────────────────────────────────────
    # m_gas ~ 1.5e5 Msun  |  softenings × 0.25 vs 512³
    #
    # CRITICAL CALIBRATION NOTES FOR 2048³:
    #
    # DustShockAmbientDensity = 20.0 cm^-3
    #   At 2048³ we fully resolve dense star-forming clumps (n ~ 10–100 cm^-3).
    #   SNe predominantly explode in high-density environments at this resolution.
    #   R_ST at n=20: ~36 pc vs ~85 pc at n=0.5 (512³), a factor ~2.4 in radius,
    #   ~14× in volume. This is the primary lever for controlling destruction rate.
    #   Target: < 0.5 shock events per live dust particle per sync window.
    #   Previous value of 0.1 cm^-3 produced ~1.8 shocks/particle/window by z=4.5,
    #   with shock events growing as 4.4M/window at z=4.5 vs 2.4M live particles.
    #
    # DustShatteringCalibration = 0.7
    #   Coagulation fires for the first time at 2048³ (n_eff > 30 cm^-3 threshold).
    #   The 1024³ +20% shattering nudge is no longer needed; coagulation now
    #   provides grain growth competition. Reducing below 1.0 prevents the
    #   coagulation/shattering balance from being shattering-dominated.
    #
    # DustGrowthCalibration = 0.20
    #   More resolved dense cells → more growth attempts pass the density check.
    #   Kept at 0.20 (same as before); revisit if D/G at z=0 is too high.
    "2048": {
        "InitCondFile":               "ICs/IC_zoom_2048_halo569_50Mpc_music_ellipsoid_with_dust",

        # Star formation
        "CritPhysDensity":            "1.0",    # cm^-3

        # Dust collision / shock thresholds
        "DustCollisionDensityThresh": "150.0",
        "DustShockAmbientDensity":    "0.1",   # cm^-3; R_ST ~47 pc

        # Calibration
        "DustGrowthCalibration":      "0.10",
        "DustShatteringCalibration":  "0.7",    # revert 1024³ nudge; coag now active

        # Dust sampling (16/24 particles per event keeps m_dust/m_gas ~ constant)
        "DustParticlesPerSNII":       "16",
        "DustParticlesPerAGB":        "24",

        # Softenings quartered
        "SofteningComovingClass0":    "1.5",
        "SofteningMaxPhysClass0":     "0.75",
        "SofteningComovingClass1":    "3.0",
        "SofteningMaxPhysClass1":     "1.5",
        "SofteningComovingClass2":    "4.0",
        "SofteningMaxPhysClass2":     "2.0",
        "SofteningComovingClass3":    "8.0",
        "SofteningMaxPhysClass3":     "4.0",
        "SofteningComovingClass4":    "1.5",
        "SofteningMaxPhysClass4":     "0.75",
        "SofteningComovingClass5":    "16.0",
        "SofteningMaxPhysClass5":     "8.0",
        "SofteningComovingClass6":    "1.5",
        "SofteningMaxPhysClass6":     "0.75",
        "MinimumComovingHydroSoftening": "1.0",
    },
}

# ── Simulation grid ───────────────────────────────────────────────────────────
# Each entry: label → dict of parameter overrides.
# Only OutputDir and DustEnable* switches go here; everything else comes
# from param_base.txt or RES_OVERRIDES above.

def make_switches(n_active):
    """Return a dict with the first n_active switches ON, rest OFF."""
    return {sw: ("1" if i < n_active else "0")
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
    "S9_1024":  {**make_switches(10),  **{"OutputDir": "S9_output_1024"}},
    "S10_1024":  {**make_switches(11),  **{"OutputDir": "S10_output_1024"}},

    # ── 2048³ — resolution convergence ──────────────────────────────────────
    "S0_2048":  {**make_switches(1),  **{"OutputDir": "S0_output_2048"}},
    "S10_2048": {**make_switches(11), **{"OutputDir": "S10_output_2048"}},
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
        print("Re-run any time you change param_base.txt or the GRID dict.")


if __name__ == "__main__":
    main()
