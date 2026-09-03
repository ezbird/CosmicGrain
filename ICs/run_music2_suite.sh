#!/bin/bash

set -u
set -o pipefail

# ================================================================
# CosmicGrain MUSIC2 suite runner
#
# Runs the 12-halo suite in resolution order:
#   512 -> 1024 -> 2048 -> 4096
#
# Key behavior:
#   - MUSIC2 is launched from each halo directory.
#   - Every MUSIC2 config is normalized so that
#
#         transfer_file = ../camb_transfer_50Mpc_zoom_z99.dat
#
#     which resolves to the shared transfer file in ~/gadget4/ICs.
#   - Existing CosmicGrain-ready ICs are skipped.
#   - Existing raw MUSIC2 ICs are converted in place without
#     rerunning MUSIC2.
#   - After a newly generated IC passes CosmicGrain validation,
#     disposable MUSIC2 working files are removed:
#         dump_transfer.txt
#         input_powerspec.txt
#         white_noise_*.bin
#   - Any MUSIC2, conversion, or validation failure stops the suite.
# ================================================================


MUSIC_DIR="$HOME/MUSIC2/build"
MUSIC_BIN="$MUSIC_DIR/MUSIC"

CONFIG_DIR="$HOME/gadget4/ICs/MUSIC2_params"
IC_ROOT="$HOME/gadget4/ICs"
LOG_DIR="$IC_ROOT/MUSIC2_logs"

# MUSIC2 runs from $IC_ROOT/haloXXXX, so this shared file is one
# directory above the run directory.
TRANSFER_FILE="$IC_ROOT/camb_transfer_50Mpc_zoom_z99.dat"
TRANSFER_CONFIG_PATH="../camb_transfer_50Mpc_zoom_z99.dat"

# Prefer the copy in ~/gadget4/ICs if present; retain compatibility
# with the earlier ~/gadget4/scripts location.
if [ -f "$IC_ROOT/add_dust_type.py" ]; then
    ADD_DUST_SCRIPT="$IC_ROOT/add_dust_type.py"
else
    ADD_DUST_SCRIPT="$HOME/gadget4/scripts/add_dust_type.py"
fi


HALOS=(
    295
    308
    441
    859
    1481
    1534
    3352
    3879
    3886
    5834
    7723
    9235
)

RESOLUTIONS=(
    512
    1024
    2048
    4096
)


# ----------------------------------------------------------------
# Normalize one MUSIC2 config's transfer_file entry.
#
# This edits the config in place only when needed.  The line must
# already exist; if it does not, stop rather than silently appending
# a potentially misplaced option.
# ----------------------------------------------------------------
normalize_transfer_path() {
    local config="$1"

    if ! grep -Eq '^[[:space:]]*transfer_file[[:space:]]*=' "$config"; then
        echo "ERROR: transfer_file entry not found in config:"
        echo "  $config"
        return 1
    fi

    python3 - "$config" "$TRANSFER_CONFIG_PATH" <<'PY'
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
wanted = sys.argv[2]

text = path.read_text()

pattern = re.compile(
    r'^([ \t]*transfer_file[ \t]*=[ \t]*)([^#\n]*?)([ \t]*(?:#.*)?)$',
    re.MULTILINE,
)

matches = list(pattern.finditer(text))
if len(matches) != 1:
    raise SystemExit(
        f"Expected exactly one transfer_file entry in {path}, "
        f"found {len(matches)}"
    )

def repl(match):
    prefix = match.group(1)
    suffix = match.group(3)
    return f"{prefix}{wanted}{suffix}"

new_text = pattern.sub(repl, text, count=1)

if new_text != text:
    path.write_text(new_text)
    print(f"Updated transfer_file in: {path}")
else:
    print(f"transfer_file already correct: {path}")
PY
}


# ----------------------------------------------------------------
# Remove MUSIC2 scratch/intermediate products only after the final
# HDF5 IC has been generated, converted, and validated successfully.
#
# These are not needed by Gadget-4/CosmicGrain and are reproducible
# from the retained MUSIC2 config + seed hierarchy.
# ----------------------------------------------------------------
cleanup_music2_workfiles() {
    local halo_dir="$1"

    local removed=0
    local file

    for file in \
        "$halo_dir/dump_transfer.txt" \
        "$halo_dir/input_powerspec.txt"
    do
        if [ -f "$file" ]; then
            rm -f -- "$file"
            removed=1
        fi
    done

    # Use nullglob so an unmatched pattern does not become a literal filename.
    shopt -s nullglob
    local white_noise_files=("$halo_dir"/wnoise_*.bin)
    shopt -u nullglob

    if [ "${#white_noise_files[@]}" -gt 0 ]; then
        rm -f -- "${white_noise_files[@]}"
        removed=1
    fi

    if [ "$removed" -eq 1 ]; then
        echo "Cleaned MUSIC2 working files from:"
        echo "  $halo_dir"
    fi
}


# ----------------------------------------------------------------
# Validate that an HDF5 IC is already ready for CosmicGrain NTYPES=7.
# ----------------------------------------------------------------
is_cosmicgrain_ready() {
    python3 - "$1" <<'PY'
import sys
import h5py

filename = sys.argv[1]

try:
    with h5py.File(filename, "r") as f:
        if "Header" not in f or "PartType6" not in f:
            sys.exit(1)

        h = f["Header"].attrs

        for key in ("NumPart_ThisFile", "NumPart_Total", "MassTable"):
            if key not in h or len(h[key]) != 7:
                sys.exit(1)

        if h["NumPart_ThisFile"][6] != 0:
            sys.exit(1)

        if h["NumPart_Total"][6] != 0:
            sys.exit(1)

        if h["MassTable"][6] != 0:
            sys.exit(1)

        if "NumPart_Total_HighWord" in h:
            if len(h["NumPart_Total_HighWord"]) != 7:
                sys.exit(1)
            if h["NumPart_Total_HighWord"][6] != 0:
                sys.exit(1)

        pt6 = f["PartType6"]

        expected = {
            "Coordinates": (0, 3),
            "Velocities": (0, 3),
            "ParticleIDs": (0,),
            "Masses": (0,),
        }

        for key, shape in expected.items():
            if key not in pt6:
                sys.exit(1)
            if pt6[key].shape != shape:
                sys.exit(1)

    sys.exit(0)

except Exception:
    sys.exit(1)
PY
}


echo
echo "============================================================"
echo " CosmicGrain MUSIC2 suite"
echo "============================================================"
echo

if [ ! -x "$MUSIC_BIN" ]; then
    echo "ERROR: MUSIC2 executable not found:"
    echo "  $MUSIC_BIN"
    exit 1
fi

if [ ! -d "$CONFIG_DIR" ]; then
    echo "ERROR: MUSIC2 config directory not found:"
    echo "  $CONFIG_DIR"
    exit 1
fi

if [ ! -f "$TRANSFER_FILE" ]; then
    echo "ERROR: shared CAMB transfer file not found:"
    echo "  $TRANSFER_FILE"
    echo
    echo "Each config will use:"
    echo "  transfer_file = $TRANSFER_CONFIG_PATH"
    exit 1
fi

if [ ! -f "$ADD_DUST_SCRIPT" ]; then
    echo "ERROR: add_dust_type.py not found."
    echo "Checked preferred/fallback locations:"
    echo "  $IC_ROOT/add_dust_type.py"
    echo "  $HOME/gadget4/scripts/add_dust_type.py"
    exit 1
fi

if ! python3 -c "import h5py, numpy" >/dev/null 2>&1; then
    echo "ERROR: Python h5py/numpy environment is unavailable."
    exit 1
fi

mkdir -p "$LOG_DIR"


# ================================================================
# Normalize ALL suite configs before launching anything.
#
# This prevents a later resolution/halo from reaching MUSIC2 with
# the obsolete transfer path.
# ================================================================
echo "Checking MUSIC2 config transfer paths..."
echo

for RES in "${RESOLUTIONS[@]}"; do
    for HALO in "${HALOS[@]}"; do
        CONFIG="$CONFIG_DIR/halo${HALO}_music2_${RES}.conf"

        if [ ! -f "$CONFIG" ]; then
            echo "ERROR: config file not found:"
            echo "  $CONFIG"
            exit 1
        fi

        if ! normalize_transfer_path "$CONFIG"; then
            exit 1
        fi
    done
done

echo
echo "All suite configs now use:"
echo "  transfer_file = $TRANSFER_CONFIG_PATH"
echo


# ================================================================
# Main suite
# ================================================================
for RES in "${RESOLUTIONS[@]}"; do

    echo
    echo "============================================================"
    echo " STARTING ${RES} SUITE"
    echo "============================================================"
    echo

    for HALO in "${HALOS[@]}"; do

        CONFIG="$CONFIG_DIR/halo${HALO}_music2_${RES}.conf"
        HALO_DIR="$IC_ROOT/halo${HALO}"

        # Canonical suite filename.
        IC_FILE="$HALO_DIR/IC_halo${HALO}_zoom_${RES}.hdf5"
        LOG_FILE="$LOG_DIR/halo${HALO}_music2_${RES}.log"

        echo "------------------------------------------------------------"
        echo "Halo:       $HALO"
        echo "Resolution: $RES"
        echo "Config:     $CONFIG"
        echo "Output:     $IC_FILE"
        echo "Log:        $LOG_FILE"
        echo "------------------------------------------------------------"

        mkdir -p "$HALO_DIR"

        # --------------------------------------------------------
        # Existing IC
        # --------------------------------------------------------
        if [ -f "$IC_FILE" ]; then

            if is_cosmicgrain_ready "$IC_FILE"; then
                echo "SKIP: CosmicGrain-ready IC already exists."

                # Clean leftover scratch files from an earlier successful
                # generation if they are still present.
                cleanup_music2_workfiles "$HALO_DIR"

                echo
                continue
            fi

            echo "Existing raw MUSIC2 IC found."
            echo "Running add_dust_type.py in place..."
            echo

            python3 "$ADD_DUST_SCRIPT" "$IC_FILE"
            STATUS=$?

            if [ "$STATUS" -ne 0 ]; then
                echo
                echo "ERROR: add_dust_type.py failed:"
                echo "  $IC_FILE"
                exit "$STATUS"
            fi

            if ! is_cosmicgrain_ready "$IC_FILE"; then
                echo
                echo "ERROR: IC still fails CosmicGrain validation:"
                echo "  $IC_FILE"
                exit 1
            fi

            cleanup_music2_workfiles "$HALO_DIR"

            echo
            echo "DONE: existing IC is now CosmicGrain-ready."
            echo
            continue
        fi


        # --------------------------------------------------------
        # Generate new MUSIC2 IC
        # --------------------------------------------------------
        echo "Running MUSIC2..."
        echo

        cd "$HALO_DIR" || exit 1

        "$MUSIC_BIN" "$CONFIG" \
            2>&1 | tee "$LOG_FILE"

        STATUS=${PIPESTATUS[0]}

        if [ "$STATUS" -ne 0 ]; then
            echo
            echo "============================================================"
            echo " ERROR"
            echo "============================================================"
            echo
            echo "MUSIC2 failed for:"
            echo "  halo       = $HALO"
            echo "  resolution = $RES"
            echo "  exit code  = $STATUS"
            echo
            echo "Log:"
            echo "  $LOG_FILE"
            exit "$STATUS"
        fi

        if [ ! -f "$IC_FILE" ]; then
            echo
            echo "ERROR: MUSIC2 exited successfully, but the expected IC"
            echo "was not found:"
            echo "  $IC_FILE"
            echo
            echo "Check that the config filename entry is:"
            echo "  IC_halo${HALO}_zoom_${RES}"
            echo
            echo "Log:"
            echo "  $LOG_FILE"
            exit 1
        fi


        # --------------------------------------------------------
        # Add CosmicGrain PartType6 support in place
        # --------------------------------------------------------
        echo
        echo "MUSIC2 complete."
        echo "Adding CosmicGrain PartType6 support in place..."
        echo

        python3 "$ADD_DUST_SCRIPT" "$IC_FILE"
        STATUS=$?

        if [ "$STATUS" -ne 0 ]; then
            echo
            echo "============================================================"
            echo " ERROR"
            echo "============================================================"
            echo
            echo "add_dust_type.py failed for:"
            echo "  $IC_FILE"
            exit "$STATUS"
        fi

        if ! is_cosmicgrain_ready "$IC_FILE"; then
            echo
            echo "ERROR: generated IC fails CosmicGrain validation:"
            echo "  $IC_FILE"
            exit 1
        fi

        # Only delete MUSIC2 scratch products after all validation succeeds.
        cleanup_music2_workfiles "$HALO_DIR"

        echo
        echo "DONE:"
        echo "  $IC_FILE"
        echo

    done

    echo
    echo "============================================================"
    echo " ${RES} SUITE COMPLETE"
    echo "============================================================"

done


echo
echo "============================================================"
echo " ALL MUSIC2 RUNS COMPLETE"
echo "============================================================"
echo
echo "IC root:"
echo "  $IC_ROOT"
echo
echo "Logs:"
echo "  $LOG_DIR"
echo
