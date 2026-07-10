#!/usr/bin/env bash
# ==============================================================================
# run_grid.sh
# Daisy-chain a grid of Gadget4 dust physics runs back to back.
#
# Each "run" is defined by a (label, param_file, config_file) triple.
# The script rebuilds Gadget4 for each config, creates the output dir,
# runs the simulation, and moves on to the next only on success.
#
# Usage:
#   ./run_grid.sh               → runs the full grid defined below
#   ./run_grid.sh S0_512        → runs only the entry labelled "S0_512"
#   ./run_grid.sh 3             → runs only the 3rd entry (1-indexed)
#   ./run_grid.sh 2 4           → runs entries 2 through 4 (inclusive)
#
# Output layout:
#   7_output_<label>/
#       output_<label>.log      full mpirun + Gadget4 stdout/stderr
#       run_metadata.txt        timestamp, git hash, config used
# ==============================================================================
set -euo pipefail

# ── MPI settings ──────────────────────────────────────────────────────────────
NP=24                          # number of MPI tasks

# ── Grid definition ───────────────────────────────────────────────────────────
# Format:  LABEL | PARAM_FILE | CONFIG_FILE
# Add/remove rows to define your physics grid.
# Labels become the output directory suffix: 7_output_<LABEL>
# If CONFIG_FILE is the same as the previous run, Gadget4 is NOT rebuilt
# (saves ~2 min per run).  Set to "SAME" to skip rebuild explicitly.
GRID=(
    # ── 512³ physics ladder ──────────────────────────────────────────────────
    # S0: creation only — inert dust baseline
    "S0_512  | params/param_S0_512.txt  | configs/Config_zoom.sh"
    # S1: + cooling (dust temperature equilibrium)
    "S1_512  | params/param_S1_512.txt  | configs/Config_zoom.sh"
    # S2: + Epstein drag
    "S2_512  | params/param_S2_512.txt  | configs/Config_zoom.sh"
    # S3: + astration
    "S3_512  | params/param_S3_512.txt  | configs/Config_zoom.sh"
    # S4: + thermal sputtering
    "S4_512  | params/param_S4_512.txt  | configs/Config_zoom.sh"
    # S5: + grain growth
    "S5_512  | params/param_S5_512.txt  | configs/Config_zoom.sh"
    # S6: + subgrid clumping factor
    "S6_512  | params/param_S6_512.txt  | configs/Config_zoom.sh"
    # S7: + SN shock destruction
    "S7_512  | params/param_S7_512.txt  | configs/Config_zoom.sh"
    # S8: + coagulation
    "S8_512  | params/param_S8_512.txt  | configs/Config_zoom.sh"
    # S9: + shattering (first fully self-regulated grain size distribution)
    "S9_512  | params/param_S9_512.txt  | configs/Config_zoom.sh"
    # S10: + radiation pressure — full physics
    "S10_512 | params/param_S10_512.txt | configs/Config_zoom.sh"

    # ── 1024³ — promoted runs only (calibrate at 512³ first) ─────────────────
    "S0_1024  | params/param_S0_1024.txt  | configs/Config_zoom.sh"
    "S1_1024  | params/param_S1_1024.txt | configs/Config_zoom.sh"
    "S2_1024  | params/param_S2_1024.txt  | configs/Config_zoom.sh"
    "S3_1024  | params/param_S3_1024.txt | configs/Config_zoom.sh"
    "S4_1024  | params/param_S4_1024.txt  | configs/Config_zoom.sh"
    "S5_1024  | params/param_S5_1024.txt | configs/Config_zoom.sh"
    "S6_1024  | params/param_S6_1024.txt  | configs/Config_zoom.sh"
    "S7_1024  | params/param_S7_1024.txt | configs/Config_zoom.sh"
    "S8_1024  | params/param_S8_1024.txt  | configs/Config_zoom.sh"
    "S9_1024  | params/param_S9_1024.txt | configs/Config_zoom.sh"
    "S10_1024 | params/param_S10_1024.txt | configs/Config_zoom.sh"

    # ── 2048³ — resolution convergence only ───────────────────────────────────
    "S0_2048  | params/param_S0_2048.txt  | configs/Config_zoom.sh"
    "S10_2048 | params/param_S10_2048.txt | configs/Config_zoom.sh"
)

# ── Helpers ───────────────────────────────────────────────────────────────────
log()  { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
die()  { echo "ERROR: $*" >&2; exit 1; }
sep()  { echo "══════════════════════════════════════════════════════════════"; }

# ── Argument parsing ──────────────────────────────────────────────────────────
# Build index list of which runs to execute
N_RUNS=${#GRID[@]}
RUN_INDICES=()

if [[ $# -eq 0 ]]; then
    # No args → run everything
    for i in "${!GRID[@]}"; do RUN_INDICES+=($i); done
elif [[ $# -eq 1 ]]; then
    if [[ "$1" =~ ^[0-9]+$ ]]; then
        # Single integer → run that entry (1-indexed)
        idx=$(( $1 - 1 ))
        [[ $idx -ge 0 && $idx -lt $N_RUNS ]] || die "Index $1 out of range (1–$N_RUNS)"
        RUN_INDICES=($idx)
    else
        # Label string → find matching entry
        found=0
        for i in "${!GRID[@]}"; do
            label=$(echo "${GRID[$i]}" | awk -F'|' '{print $1}' | xargs)
            if [[ "$label" == "$1" ]]; then
                RUN_INDICES=($i); found=1; break
            fi
        done
        [[ $found -eq 1 ]] || die "No run labelled '$1' found in grid"
    fi
elif [[ $# -eq 2 && "$1" =~ ^[0-9]+$ && "$2" =~ ^[0-9]+$ ]]; then
    # Two integers → range (1-indexed, inclusive)
    start=$(( $1 - 1 ))
    end=$(( $2 - 1 ))
    [[ $start -ge 0 && $end -lt $N_RUNS && $start -le $end ]] \
        || die "Range $1–$2 invalid (grid has $N_RUNS entries)"
    for i in $(seq $start $end); do RUN_INDICES+=($i); done
else
    die "Usage: $0 [label | index | start_idx end_idx]"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
sep
log "Gadget4 dust physics grid runner"
log "Will execute ${#RUN_INDICES[@]} run(s):"
for i in "${RUN_INDICES[@]}"; do
    entry="${GRID[$i]}"
    label=$(echo "$entry" | awk -F'|' '{print $1}' | xargs)
    param=$(echo "$entry" | awk -F'|' '{print $2}' | xargs)
    cfg=$(echo "$entry"   | awk -F'|' '{print $3}' | xargs)
    log "  [$((i+1))/$N_RUNS]  $label  (param: $param  config: $cfg)"
done
sep

# ── Main loop ─────────────────────────────────────────────────────────────────
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

    # Validate inputs exist before doing any work
    [[ -f "$PARAM" ]] || die "Param file not found: $PARAM"
    [[ -f "$CFG"   ]] || die "Config file not found: $CFG"

    DIR="S${LABEL#S}_output_${LABEL##*_}"   # e.g. S0_512 → S0_output_512
    # Simpler: just parse label directly
    # LABEL format is always S{N}_{RES}, e.g. S0_512, S10_1024
    RES="${LABEL##*_}"          # everything after last underscore → 512
    SN="${LABEL%_*}"            # everything before last underscore → S0
    DIR="${SN}_output_${RES}"   # → S0_output_512
    LOG="${DIR}/output_${LABEL}.log"

    # ── Build ────────────────────────────────────────────────────────────────
    if [[ "$CFG" != "$PREV_CONFIG" ]]; then
        log "Config changed → rebuilding Gadget4 with $CFG"

        # Swap in the new Config.sh (Gadget4 looks for Config.sh in cwd)
        cp "$CFG" Config.sh

        make clean
        make -j${NP}
        log "Build complete"
        PREV_CONFIG="$CFG"
    else
        log "Config unchanged — skipping rebuild"
    fi

    # ── Output directory ─────────────────────────────────────────────────────
    if [[ -d "$DIR" ]]; then
        log "WARNING: $DIR already exists — archiving to ${DIR}_backup_$(date +%Y%m%d_%H%M%S)"
        mv "$DIR" "${DIR}_backup_$(date +%Y%m%d_%H%M%S)"
    fi
    mkdir -p "$DIR"

    # ── Metadata ─────────────────────────────────────────────────────────────
    cat > "${DIR}/run_metadata.txt" <<META
label       : $LABEL
param_file  : $PARAM
config_file : $CFG
start_time  : $(date '+%Y-%m-%d %H:%M:%S')
git_hash    : $(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
hostname    : $(hostname)
np          : $NP
META

    # ── Run ──────────────────────────────────────────────────────────────────
    log "Launching: mpirun -np $NP ./Gadget4 $PARAM → $LOG"
    RUN_START=$(date +%s)

    set +e   # don't abort on non-zero exit so we can record failure
    mpirun -np $NP --bind-to core --map-by core \
        ./CosmicGrain "$PARAM" 2>&1 | tee "$LOG"
    EXIT_CODE=${PIPESTATUS[0]}
    set -e

    RUN_END=$(date +%s)
    ELAPSED=$(( RUN_END - RUN_START ))
    ELAPSED_H=$(( ELAPSED / 3600 ))
    ELAPSED_M=$(( (ELAPSED % 3600) / 60 ))
    ELAPSED_S=$(( ELAPSED % 60 ))

    echo "end_time    : $(date '+%Y-%m-%d %H:%M:%S')" >> "${DIR}/run_metadata.txt"
    echo "wall_time   : ${ELAPSED_H}h ${ELAPSED_M}m ${ELAPSED_S}s" >> "${DIR}/run_metadata.txt"
    echo "exit_code   : $EXIT_CODE" >> "${DIR}/run_metadata.txt"

    if [[ $EXIT_CODE -ne 0 ]]; then
        log "ERROR: Run '$LABEL' exited with code $EXIT_CODE after ${ELAPSED_H}h ${ELAPSED_M}m"
        log "Log saved to $LOG — inspect before continuing"
        FAILED+=("$LABEL")
        # Ask whether to continue or abort
        if [[ -t 0 ]]; then   # only prompt if stdin is a terminal
            read -rp "Continue with remaining runs? [y/N] " REPLY
            [[ "${REPLY,,}" == "y" ]] || { log "Aborting grid run."; break; }
        else
            log "Non-interactive mode — aborting remaining runs"
            break
        fi
    else
        log "Run '$LABEL' completed in ${ELAPSED_H}h ${ELAPSED_M}m ${ELAPSED_S}s"
        COMPLETED+=("$LABEL")
    fi
done

# ── Final summary ─────────────────────────────────────────────────────────────
sep
log "Grid run complete"
log "  Completed (${#COMPLETED[@]}): ${COMPLETED[*]:-none}"
log "  Failed    (${#FAILED[@]}):    ${FAILED[*]:-none}"
sep
