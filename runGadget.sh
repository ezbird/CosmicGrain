#!/usr/bin/env bash
set -euo pipefail

# ==========================
# Resolution parameter
# ==========================
# Usage:
#   ./runCG.sh        → defaults to 1024
#   ./runCG.sh 512    → uses 512
#   ./runCG.sh 2048   → uses 2048

RES="${1:-1024}"

# ==========================
# Derived names
# ==========================
DIR="S10_output_${RES}"
PARAMS="params/param_S10_${RES}.txt"
LOG="${DIR}/output_${RES}.log"

# ==========================
# Build
# ==========================
make clean
make -j23

# ==========================
# Run
# ==========================
rm -rf "$DIR"
mkdir "$DIR"
mpirun -np 24 --bind-to core --map-by core ./CosmicGrain "$PARAMS" 2>&1 | tee "$LOG"
