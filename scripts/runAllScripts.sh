#!/usr/bin/env bash
set -Eeuo pipefail

# Run the single-halo, final-snapshot CosmicGrain analysis suite.
#
# First use:
#   bash run_halo_z0_analysis.sh ../halo441_output_1024 1
#
# Later uses can omit GROUP_INDEX after creating OUTPUT_DIR/analysis_target.env:
#   GROUP_INDEX=1
#   EXPECTED_M200_MSUN=1.0606e12        # optional but strongly recommended
#   GALAXY_APERTURE_R200=0.1             # optional; default 0.1
#   GALAXY_APERTURE_PKPC=30              # optional fixed-pkpc override
#
# Then:
#   bash run_halo_z0_analysis.sh ../halo441_output_1024

usage() {
    cat <<'EOF'
Usage: run_halo_z0_analysis.sh OUTPUT_DIR [GROUP_INDEX]

Runs only single-halo analyses on the latest snapshot/catalog pair.
The target group is read from GROUP_INDEX or OUTPUT_DIR/analysis_target.env.

Environment overrides:
  GROUP_INDEX              Target FOF group (required unless positional arg given)
  EXPECTED_M200_MSUN       Expected z=0 M200c; enables a validation guard
  GALAXY_APERTURE_R200     Galaxy/ISM aperture as a fraction of R200
                           (default: 0.1; ignored when PKPC override is set)
  GALAXY_APERTURE_PKPC     Optional fixed-pkpc override
  DUST_AGE_BREAKS_GYR      Two age boundaries (default: "3 7")
  PLOTS_DIR                Output directory (default: OUTPUT_DIR/all_plots_z0)
  PYTHON                   Python executable (default: python3)
EOF
}

[[ $# -ge 1 && $# -le 2 ]] || { usage >&2; exit 2; }

OUTPUT_DIR=${1%/}
CLI_GROUP_INDEX=${2:-}
[[ -d "$OUTPUT_DIR" ]] || { echo "ERROR: output directory not found: $OUTPUT_DIR" >&2; exit 2; }

TARGET_FILE="$OUTPUT_DIR/analysis_target.env"
if [[ -f "$TARGET_FILE" ]]; then
    # This file must contain simple trusted shell assignments belonging to this run.
    # shellcheck disable=SC1090
    source "$TARGET_FILE"
fi

GROUP_INDEX=${CLI_GROUP_INDEX:-${GROUP_INDEX:-}}
[[ "$GROUP_INDEX" =~ ^[0-9]+$ ]] || {
    echo "ERROR: supply the target group index, e.g.:" >&2
    echo "  bash $0 $OUTPUT_DIR 1" >&2
    echo "Or set GROUP_INDEX in $TARGET_FILE" >&2
    exit 2
}

PYTHON=${PYTHON:-python3}
GALAXY_APERTURE_R200=${GALAXY_APERTURE_R200:-0.1}
GALAXY_APERTURE_PKPC=${GALAXY_APERTURE_PKPC:-}
DUST_AGE_BREAKS_GYR=${DUST_AGE_BREAKS_GYR:-"3 7"}
PLOTS_DIR=${PLOTS_DIR:-$OUTPUT_DIR/all_plots_z0}
DATE=$(date +%Y-%m-%d)
RUN_NAME=$(basename "$OUTPUT_DIR")
LOG_DIR="$PLOTS_DIR/logs"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$PLOTS_DIR" "$LOG_DIR"

read -r AGE_YOUNG_MAX AGE_OLD_MIN AGE_EXTRA <<< "$DUST_AGE_BREAKS_GYR"
[[ -n "${AGE_YOUNG_MAX:-}" && -n "${AGE_OLD_MIN:-}" && -z "${AGE_EXTRA:-}" ]] || {
    echo 'ERROR: DUST_AGE_BREAKS_GYR must contain exactly two values, e.g. "3 7"' >&2
    exit 2
}

# Select the numerically latest snapshot for which both a snapshot and catalog exist.
mapfile -t SNAP_CANDIDATES < <(
    find "$OUTPUT_DIR" -maxdepth 2 -type f -path '*/snapdir_*/snapshot_*.0.hdf5' -printf '%f\n' \
        | sed -nE 's/^snapshot_([0-9]+)\.0\.hdf5$/\1/p' \
        | sort -n -u -r
)

SNAP_PAD=""
for candidate in "${SNAP_CANDIDATES[@]}"; do
    candidate=$(printf '%03d' "$((10#$candidate))")
    if [[ -f "$OUTPUT_DIR/groups_${candidate}/fof_subhalo_tab_${candidate}.0.hdf5" ]]; then
        SNAP_PAD=$candidate
        break
    fi
done
[[ -n "$SNAP_PAD" ]] || {
    echo "ERROR: no paired snapshot/catalog was found under $OUTPUT_DIR" >&2
    exit 2
}
SNAP=$((10#$SNAP_PAD))

SNAPSHOT="$OUTPUT_DIR/snapdir_${SNAP_PAD}/snapshot_${SNAP_PAD}"
CATALOG="$OUTPUT_DIR/groups_${SNAP_PAD}/fof_subhalo_tab_${SNAP_PAD}.0.hdf5"

# Use the paper-standard aperture, 0.1 R200 by default.  The catalog radius is
# stored in comoving kpc/h, so convert it to physical kpc using a/h.  A fixed
# GALAXY_APERTURE_PKPC value remains available for explicit diagnostic runs.
if [[ -n "$GALAXY_APERTURE_PKPC" ]]; then
    APERTURE_MODE="fixed-pkpc"
    APERTURE_TAG="${GALAXY_APERTURE_PKPC//./p}pkpc"
else
    GALAXY_APERTURE_PKPC=$(
        "$PYTHON" - "$CATALOG" "$GROUP_INDEX" "$GALAXY_APERTURE_R200" <<'PY'
import h5py
import numpy as np
import sys

catalog, group_text, fraction_text = sys.argv[1:]
group_index = int(group_text)
fraction = float(fraction_text)
if not (0.0 < fraction <= 1.0):
    raise SystemExit("GALAXY_APERTURE_R200 must be in (0, 1]")

with h5py.File(catalog, "r") as f:
    radii = f["Group/Group_R_Crit200"]
    if group_index >= len(radii):
        raise SystemExit(
            f"group {group_index} is not present in first catalog chunk "
            f"({len(radii)} groups)"
        )
    r_ckpc_h = float(radii[group_index])
    attrs = f["Header"].attrs if "Header" in f else f.attrs
    a = float(np.asarray(attrs.get("Time", 1.0)).reshape(-1)[0])
    h = float(np.asarray(attrs.get("HubbleParam", 0.6732)).reshape(-1)[0])

print(f"{fraction * r_ckpc_h * a / h:.8f}")
PY
    ) || {
        echo "ERROR: could not derive the R200-scaled galaxy aperture." >&2
        exit 2
    }
    APERTURE_MODE="r200-scaled"
    APERTURE_TAG="r${GALAXY_APERTURE_R200//./p}R200"
fi

if [[ "$RUN_NAME" =~ _output_([0-9]+)$ ]]; then
    RES=${BASH_REMATCH[1]}
else
    RES="unknown"
fi

declare -a COMPLETED=()
declare -a SKIPPED=()

has_option() {
    local script=$1 option=$2
    "$PYTHON" "$script" --help 2>&1 | grep -q -- "$option"
}

require_script_options() {
    local script=$1
    shift
    [[ -f "$script" ]] || {
        SKIPPED+=("$script (script not found)")
        return 1
    }
    local option
    for option in "$@"; do
        if ! has_option "$script" "$option"; then
            SKIPPED+=("$script (missing required safe option: $option)")
            return 1
        fi
    done
}

run_logged() {
    local label=$1
    shift
    echo
    echo "--- $label ---"
    "$@" 2>&1 | tee "$LOG_DIR/${label//[^A-Za-z0-9_.-]/_}.log"
    COMPLETED+=("$label")
}

echo "============================================================"
echo " CosmicGrain single-halo z=0/final-snapshot analysis"
echo " Run          : $RUN_NAME"
echo " Output dir   : $OUTPUT_DIR"
echo " Snapshot     : $SNAP_PAD (latest paired snapshot/catalog)"
echo " Group index  : $GROUP_INDEX"
echo " Resolution   : $RES"
echo " Aperture     : $GALAXY_APERTURE_PKPC pkpc"
echo " Aperture mode: $APERTURE_MODE ($APERTURE_TAG)"
echo " Dust ages    : <${AGE_YOUNG_MAX}, ${AGE_YOUNG_MAX}-${AGE_OLD_MIN}, >${AGE_OLD_MIN} Gyr"
echo " Products     : $PLOTS_DIR"
echo "============================================================"

# Acceptance audit comes first and is a hard gate. This catches wrong-group
# selection before time is spent making figures.
AUDIT="$SCRIPT_DIR/cosmicgrain_zoom_postrun_audit_v3.py"
if require_script_options "$AUDIT" --group-index; then
    audit_args=(
        "$PYTHON" "$AUDIT" "$OUTPUT_DIR"
        --group-index "$GROUP_INDEX"
        --galaxy-aperture-pkpc "$GALAXY_APERTURE_PKPC"
    )
    if [[ -n "${EXPECTED_M200_MSUN:-}" ]]; then
        audit_args+=(--parent-m200-msun "$EXPECTED_M200_MSUN")
    fi

    # The canonical audit JSON/CSV is reserved for the paper-standard
    # R200-scaled aperture. Fixed-pkpc diagnostics get aperture-specific names
    # under the plots directory and cannot overwrite the paper row.
    if [[ "$APERTURE_MODE" == "fixed-pkpc" ]]; then
        audit_args+=(
            --output-prefix
            "$PLOTS_DIR/${RUN_NAME}_snap${SNAP_PAD}_audit_${APERTURE_TAG}"
        )
    fi
    run_logged acceptance_audit "${audit_args[@]}"
else
    echo "ERROR: $AUDIT with --group-index is required as the safety gate." >&2
    exit 2
fi

# Complete dust health report.
SUMMARY="$SCRIPT_DIR/dust_snapshot_summary.py"
if require_script_options "$SUMMARY" --group-index --snap; then
    run_logged dust_snapshot_summary \
        "$PYTHON" "$SUMMARY" "$OUTPUT_DIR" --snap "$SNAP" \
        --group-index "$GROUP_INDEX" \
        --galaxy-aperture-pkpc "$GALAXY_APERTURE_PKPC"
fi

# Age-coded grain radius, composition, mass, velocity, temperature, and age.
AGE="$SCRIPT_DIR/plot_dust_histograms_agecoded.py"
if require_script_options "$AGE" --group-index; then
    run_logged agecoded_histograms \
        "$PYTHON" "$AGE" --catalog "$CATALOG" --snapshot "$SNAPSHOT" \
            --group-index "$GROUP_INDEX" \
            --age-breaks "$AGE_YOUNG_MAX" "$AGE_OLD_MIN" \
            --out "$PLOTS_DIR/agecoded_${RUN_NAME}_snap${SNAP_PAD}_${DATE}.png"
fi

# Grain-size distribution in the galaxy and CGM.
GSD="$SCRIPT_DIR/plot_gsd_comparison.py"
if require_script_options "$GSD" --group-index --snap; then
    run_logged grain_size_distribution \
        "$PYTHON" "$GSD" "$OUTPUT_DIR" --snap "$SNAP_PAD" \
        --group-index "$GROUP_INDEX" --ism-aperture "$GALAXY_APERTURE_PKPC" \
        --output "$PLOTS_DIR/gsd_${RUN_NAME}_snap${SNAP_PAD}_${APERTURE_TAG}_${DATE}.png"
fi

# Radial D/G and D/Z for this one run (not the historical S0--S10 ladder).
RADIAL="$SCRIPT_DIR/plot_radial_dgr.py"
if require_script_options "$RADIAL" --group-index --snap --output; then
    if has_option "$RADIAL" --run-dir; then
        radial_run_args=(--run-dir "$OUTPUT_DIR")
    else
        radial_run_args=("$OUTPUT_DIR")
    fi
    run_logged radial_dg_dz \
        "$PYTHON" "$RADIAL" "${radial_run_args[@]}" --snap "$SNAP" \
        --group-index "$GROUP_INDEX" \
        --output "$PLOTS_DIR/radial_dg_dz_${RUN_NAME}_snap${SNAP_PAD}_${APERTURE_TAG}_${DATE}.png"
fi

# Single-run D/Z versus gas metallicity diagnostic.
DZMETAL="$SCRIPT_DIR/plot_dz_vs_metallicity.py"
if require_script_options "$DZMETAL" --group-index --snap --output; then
    if has_option "$DZMETAL" --run-dir; then
        dz_run_args=(--run-dir "$OUTPUT_DIR")
    else
        dz_run_args=("$OUTPUT_DIR")
    fi
    run_logged dz_vs_metallicity \
        "$PYTHON" "$DZMETAL" "${dz_run_args[@]}" --snap "$SNAP" \
        --group-index "$GROUP_INDEX" --aperture "$GALAXY_APERTURE_PKPC" \
        --output "$PLOTS_DIR/dz_vs_z_${RUN_NAME}_snap${SNAP_PAD}_${APERTURE_TAG}_${DATE}.png"
fi

# Composition/source survival plots, if their current versions support safe
# explicit target selection.
for spec in \
    "$SCRIPT_DIR/plot_composition_histogram.py:composition_histogram" \
    "$SCRIPT_DIR/plot_composition_cumulative.py:composition_cumulative"
do
    script=${spec%%:*}
    label=${spec##*:}
    if require_script_options "$script" --group-index --snap; then
        run_logged "$label" \
            "$PYTHON" "$script" "$OUTPUT_DIR" --snap "$SNAP" \
            --group-index "$GROUP_INDEX" \
            --output "$PLOTS_DIR/${label}_${RUN_NAME}_snap${SNAP_PAD}_${DATE}.png"
    fi
done

echo
echo "============================================================"
echo "Completed: ${#COMPLETED[@]}"
printf '  %s\n' "${COMPLETED[@]}"
if ((${#SKIPPED[@]})); then
    echo "Skipped safely: ${#SKIPPED[@]}"
    printf '  %s\n' "${SKIPPED[@]}"
    echo
    echo "A skipped plotting script must be given the listed explicit CLI option"
    echo "before it can safely be used for the new 12-halo suite."
fi
echo "Products: $PLOTS_DIR"
echo "============================================================"
