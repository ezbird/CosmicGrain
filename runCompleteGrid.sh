#!/usr/bin/env bash
# ==============================================================================
# run_grid.sh
#
# CosmicGrain simulation runner
#
# Supports:
#   1. S0--S10 controlled physics-ladder runs
#   2. CG01--CG12 production halo suite
#
# Each run is defined by:
#
#     LABEL | PARAM_FILE | CONFIG_FILE
#
# The script:
#   - validates the requested parameter/config files
#   - rebuilds CosmicGrain when the Config.sh changes
#   - reads OutputDir directly from the parameter file
#   - archives an existing output directory rather than overwriting it
#   - saves the exact parameter/config/output-list files used
#   - records git/build/run metadata
#   - runs with MPI
#   - records wall time and exit status
#   - proceeds to the next run only after success
#
# Examples:
#
#   ./run_grid.sh halo441_1024
#       Run CG10 / Halo 441 at 1024 resolution
#
#   ./run_grid.sh S10_512
#       Run the full-physics 512 controlled run
#
#   ./run_grid.sh 3
#       Run entry 3 (1-indexed)
#
#   ./run_grid.sh 2 4
#       Run entries 2 through 4 inclusive
#
#   ./run_grid.sh
#       Run EVERY entry in GRID.
#       Use this intentionally!
#
# IMPORTANT:
#   OutputDir is taken from the parameter file. This keeps generate_params.py
#   as the single source of truth for simulation output-directory names.
# ==============================================================================

set -euo pipefail


# ==============================================================================
# Global settings
# ==============================================================================

NP=24
EXE="./CosmicGrain"


# ==============================================================================
# Grid definition
#
# Format:
#
#   LABEL | PARAM_FILE | CONFIG_FILE
#
# Labels are only identifiers for selecting/reporting runs.
# Output directories come from OutputDir in the parameter file.
# ==============================================================================

GRID=(

    # ==========================================================================
    # CONTROLLED PHYSICS LADDER — 512^3
    # ==========================================================================

    # S0: creation only — inert dust baseline
    "S0_512   | params/param_S0_512.txt   | configs/Config_zoom.sh"

    # S1: + cooling
    "S1_512   | params/param_S1_512.txt   | configs/Config_zoom.sh"

    # S2: + Epstein drag
    "S2_512   | params/param_S2_512.txt   | configs/Config_zoom.sh"

    # S3: + astration
    "S3_512   | params/param_S3_512.txt   | configs/Config_zoom.sh"

    # S4: + thermal sputtering
    "S4_512   | params/param_S4_512.txt   | configs/Config_zoom.sh"

    # S5: + grain growth
    "S5_512   | params/param_S5_512.txt   | configs/Config_zoom.sh"

    # S6: + subgrid clumping
    "S6_512   | params/param_S6_512.txt   | configs/Config_zoom.sh"

    # S7: + SN shock destruction
    "S7_512   | params/param_S7_512.txt   | configs/Config_zoom.sh"

    # S8: + coagulation
    "S8_512   | params/param_S8_512.txt   | configs/Config_zoom.sh"

    # S9: + shattering
    "S9_512   | params/param_S9_512.txt   | configs/Config_zoom.sh"

    # S10: + radiation pressure — full physics
    "S10_512  | params/param_S10_512.txt  | configs/Config_zoom.sh"


    # ==========================================================================
    # CONTROLLED PHYSICS LADDER — 1024^3
    # ==========================================================================

    "S0_1024   | params/param_S0_1024.txt   | configs/Config_zoom.sh"
    "S1_1024   | params/param_S1_1024.txt   | configs/Config_zoom.sh"
    "S2_1024   | params/param_S2_1024.txt   | configs/Config_zoom.sh"
    "S3_1024   | params/param_S3_1024.txt   | configs/Config_zoom.sh"
    "S4_1024   | params/param_S4_1024.txt   | configs/Config_zoom.sh"
    "S5_1024   | params/param_S5_1024.txt   | configs/Config_zoom.sh"
    "S6_1024   | params/param_S6_1024.txt   | configs/Config_zoom.sh"
    "S7_1024   | params/param_S7_1024.txt   | configs/Config_zoom.sh"
    "S8_1024   | params/param_S8_1024.txt   | configs/Config_zoom.sh"
    "S9_1024   | params/param_S9_1024.txt   | configs/Config_zoom.sh"
    "S10_1024  | params/param_S10_1024.txt  | configs/Config_zoom.sh"


    # ==========================================================================
    # CONTROLLED RESOLUTION-CONVERGENCE RUNS — 2048^3
    # ==========================================================================

    "S0_2048   | params/param_S0_2048.txt   | configs/Config_zoom.sh"
    "S10_2048  | params/param_S10_2048.txt  | configs/Config_zoom.sh"


    # ==========================================================================
    # CG01--CG12 PRODUCTION HALO SUITE — 1024^3
    #
    # Ordered by increasing M200c.
    # ==========================================================================

    # CG01 — Parent halo 9235
    "halo9235_1024 | params/param_halo9235_1024.txt | configs/Config_zoom.sh"

    # CG02 — Parent halo 7723
    "halo7723_1024 | params/param_halo7723_1024.txt | configs/Config_zoom.sh"

    # CG03 — Parent halo 5834
    "halo5834_1024 | params/param_halo5834_1024.txt | configs/Config_zoom.sh"

    # CG04 — Parent halo 3886
    "halo3886_1024 | params/param_halo3886_1024.txt | configs/Config_zoom.sh"

    # CG05 — Parent halo 3879
    "halo3879_1024 | params/param_halo3879_1024.txt | configs/Config_zoom.sh"

    # CG06 — Parent halo 3352
    "halo3352_1024 | params/param_halo3352_1024.txt | configs/Config_zoom.sh"

    # CG07 — Parent halo 1534
    "halo1534_1024 | params/param_halo1534_1024.txt | configs/Config_zoom.sh"

    # CG08 — Parent halo 1481
    "halo1481_1024 | params/param_halo1481_1024.txt | configs/Config_zoom.sh"

    # CG09 — Parent halo 859
    "halo859_1024  | params/param_halo859_1024.txt  | configs/Config_zoom.sh"

    # CG10 — Parent halo 441 — Milky-Way-mass halo
    "halo441_1024  | params/param_halo441_1024.txt  | configs/Config_zoom.sh"

    # CG11 — Parent halo 308
    "halo308_1024  | params/param_halo308_1024.txt  | configs/Config_zoom.sh"

    # CG12 — Parent halo 295
    "halo295_1024  | params/param_halo295_1024.txt  | configs/Config_zoom.sh"
)


# ==============================================================================
# Helpers
# ==============================================================================

log()
{
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

die()
{
    echo "ERROR: $*" >&2
    exit 1
}

sep()
{
    echo "══════════════════════════════════════════════════════════════════════"
}


# Read OutputDir from a parameter file.
#
# Example:
#
#   OutputDir   halo441_output_1024
#
# becomes:
#
#   halo441_output_1024
#
get_output_dir()
{
    local param_file="$1"
    local dir

    dir=$(
        awk '
            /^[[:space:]]*OutputDir[[:space:]]+/ {
                print $2
                exit
            }
        ' "$param_file"
    )

    [[ -n "$dir" ]] || die "Could not determine OutputDir from $param_file"

    # Strip a trailing slash if present.
    dir="${dir%/}"

    echo "$dir"
}


# ==============================================================================
# Argument parsing
# ==============================================================================

N_RUNS=${#GRID[@]}
RUN_INDICES=()

if [[ $# -eq 0 ]]; then

    # No arguments -> run everything.
    for i in "${!GRID[@]}"; do
        RUN_INDICES+=("$i")
    done

elif [[ $# -eq 1 ]]; then

    if [[ "$1" =~ ^[0-9]+$ ]]; then

        # Single integer -> run that entry (1-indexed).
        idx=$(( $1 - 1 ))

        [[ $idx -ge 0 && $idx -lt $N_RUNS ]] \
            || die "Index $1 out of range (1-$N_RUNS)"

        RUN_INDICES=("$idx")

    else

        # Label -> find exact matching entry.
        found=0

        for i in "${!GRID[@]}"; do

            label=$(
                echo "${GRID[$i]}" |
                awk -F'|' '{print $1}' |
                xargs
            )

            if [[ "$label" == "$1" ]]; then
                RUN_INDICES=("$i")
                found=1
                break
            fi
        done

        [[ $found -eq 1 ]] \
            || die "No run labelled '$1' found in grid"

    fi

elif [[ $# -eq 2 &&
        "$1" =~ ^[0-9]+$ &&
        "$2" =~ ^[0-9]+$ ]]; then

    # Two integers -> inclusive range.
    start=$(( $1 - 1 ))
    end=$(( $2 - 1 ))

    [[ $start -ge 0 &&
       $end -lt $N_RUNS &&
       $start -le $end ]] \
        || die "Range $1-$2 invalid (grid has $N_RUNS entries)"

    for i in $(seq "$start" "$end"); do
        RUN_INDICES+=("$i")
    done

else

    die "Usage: $0 [label | index | start_idx end_idx]"

fi


# ==============================================================================
# Pre-flight summary
# ==============================================================================

sep
log "CosmicGrain simulation runner"
log "Grid contains $N_RUNS total entries"
log "Will execute ${#RUN_INDICES[@]} run(s):"

for i in "${RUN_INDICES[@]}"; do

    entry="${GRID[$i]}"

    label=$(echo "$entry" | awk -F'|' '{print $1}' | xargs)
    param=$(echo "$entry" | awk -F'|' '{print $2}' | xargs)
    cfg=$(echo "$entry"   | awk -F'|' '{print $3}' | xargs)

    [[ -f "$param" ]] || die "Param file not found: $param"
    [[ -f "$cfg"   ]] || die "Config file not found: $cfg"

    dir=$(get_output_dir "$param")

    log "  [$((i+1))/$N_RUNS]"
    log "      label  : $label"
    log "      param  : $param"
    log "      config : $cfg"
    log "      output : $dir"

done

sep


# ==============================================================================
# Main loop
# ==============================================================================

PREV_CONFIG=""
COMPLETED=()
FAILED=()

for i in "${RUN_INDICES[@]}"; do

    entry="${GRID[$i]}"

    LABEL=$(echo "$entry" | awk -F'|' '{print $1}' | xargs)
    PARAM=$(echo "$entry" | awk -F'|' '{print $2}' | xargs)
    CFG=$(echo "$entry"   | awk -F'|' '{print $3}' | xargs)

    sep
    log "Starting run $((i+1))/$N_RUNS: $LABEL"


    # --------------------------------------------------------------------------
    # Validate inputs
    # --------------------------------------------------------------------------

    [[ -f "$PARAM" ]] || die "Param file not found: $PARAM"
    [[ -f "$CFG"   ]] || die "Config file not found: $CFG"

    DIR=$(get_output_dir "$PARAM")
    LOG="${DIR}/output_${LABEL}.log"


    # --------------------------------------------------------------------------
    # Build
    # --------------------------------------------------------------------------

    if [[ "$CFG" != "$PREV_CONFIG" ]]; then

        log "Config changed -> rebuilding CosmicGrain with $CFG"

        # Gadget4 build system expects Config.sh in the repository root.
        cp "$CFG" Config.sh

        make clean
        make -j"${NP}"

        [[ -x "$EXE" ]] \
            || die "Executable not found after build: $EXE"

        log "Build complete: $EXE"

        PREV_CONFIG="$CFG"

    else

        log "Config unchanged -> skipping rebuild"

        [[ -x "$EXE" ]] \
            || die "Executable not found: $EXE"

    fi


    # --------------------------------------------------------------------------
    # Output directory
    # --------------------------------------------------------------------------

    if [[ -d "$DIR" ]]; then

        BACKUP="${DIR}_backup_$(date +%Y%m%d_%H%M%S)"

        log "WARNING: $DIR already exists"
        log "Archiving existing directory -> $BACKUP"

        mv "$DIR" "$BACKUP"

    fi

    mkdir -p "$DIR"


    # --------------------------------------------------------------------------
    # Preserve exact run inputs
    # --------------------------------------------------------------------------

    cp "$PARAM" "${DIR}/parameter_file_used.txt"
    cp "$CFG"   "${DIR}/config_file_used.sh"

    # Copy the output list specified by the parameter file.
    OUTPUT_LIST=$(
        awk '
            /^[[:space:]]*OutputListFilename[[:space:]]+/ {
                print $2
                exit
            }
        ' "$PARAM"
    )

    if [[ -n "$OUTPUT_LIST" && -f "$OUTPUT_LIST" ]]; then
        cp "$OUTPUT_LIST" "${DIR}/output_snapshots_used.txt"
    fi


    # --------------------------------------------------------------------------
    # Git/build metadata
    # --------------------------------------------------------------------------

    GIT_HASH=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
    GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
    GIT_DIRTY=$(git status --porcelain 2>/dev/null | wc -l || true)

    cat > "${DIR}/run_metadata.txt" <<META
label       : $LABEL
param_file  : $PARAM
config_file : $CFG
output_dir  : $DIR
executable  : $EXE

start_time  : $(date '+%Y-%m-%d %H:%M:%S')
hostname    : $(hostname)
np          : $NP

git_hash    : $GIT_HASH
git_branch  : $GIT_BRANCH
git_dirty   : $GIT_DIRTY modified/untracked entries
META


    # --------------------------------------------------------------------------
    # Run
    # --------------------------------------------------------------------------

    log "Launching:"
    log "  mpirun -np $NP --bind-to core --map-by core $EXE $PARAM"
    log "Log:"
    log "  $LOG"

    RUN_START=$(date +%s)

    # Do not allow set -e to kill the runner before we record the failure.
    set +e

    mpirun -np "$NP" \
        --bind-to core \
        --map-by core \
        "$EXE" "$PARAM" 2>&1 | tee "$LOG"

    EXIT_CODE=${PIPESTATUS[0]}

    set -e


    # --------------------------------------------------------------------------
    # Runtime
    # --------------------------------------------------------------------------

    RUN_END=$(date +%s)
    ELAPSED=$(( RUN_END - RUN_START ))

    ELAPSED_H=$(( ELAPSED / 3600 ))
    ELAPSED_M=$(( (ELAPSED % 3600) / 60 ))
    ELAPSED_S=$(( ELAPSED % 60 ))

    {
        echo
        echo "end_time    : $(date '+%Y-%m-%d %H:%M:%S')"
        echo "wall_time   : ${ELAPSED_H}h ${ELAPSED_M}m ${ELAPSED_S}s"
        echo "exit_code   : $EXIT_CODE"
    } >> "${DIR}/run_metadata.txt"


    # --------------------------------------------------------------------------
    # Success / failure handling
    # --------------------------------------------------------------------------

    if [[ $EXIT_CODE -ne 0 ]]; then

        log "ERROR: Run '$LABEL' exited with code $EXIT_CODE"
        log "Runtime: ${ELAPSED_H}h ${ELAPSED_M}m ${ELAPSED_S}s"
        log "Inspect:"
        log "  $LOG"

        FAILED+=("$LABEL")

        # Only prompt when attached to an interactive terminal.
        if [[ -t 0 ]]; then

            read -rp "Continue with remaining runs? [y/N] " REPLY

            if [[ "${REPLY,,}" != "y" ]]; then
                log "Aborting remaining runs."
                break
            fi

        else

            log "Non-interactive mode -> aborting remaining runs"
            break

        fi

    else

        log "Run '$LABEL' completed successfully"
        log "Runtime: ${ELAPSED_H}h ${ELAPSED_M}m ${ELAPSED_S}s"

        COMPLETED+=("$LABEL")

    fi

done


# ==============================================================================
# Final summary
# ==============================================================================

sep

log "CosmicGrain run sequence complete"
log "Completed (${#COMPLETED[@]}): ${COMPLETED[*]:-none}"
log "Failed    (${#FAILED[@]}): ${FAILED[*]:-none}"

sep
