#!/bin/bash
# Submit the 16-run 1024^3 C-S grid as a sequential Slurm chain (afterany, so
# one failed run does not stop the rest).  Baseline c100_r100 goes first.
# Usage (on Colibri):  bash submit_cs_grid.sh [first_dependency_jobid]
set -e
cd ~/CosmicGrain-br06
EXE=~/jobexe/CosmicGrain.csgrid.$(git rev-parse --short HEAD)
cp ./CosmicGrain "$EXE"
RUNS="c100_r100 c050_r100 c200_r100 c025_r100
      c100_r050 c100_r071 c100_r141
      c050_r050 c050_r071 c050_r141
      c200_r050 c200_r071 c200_r141
      c025_r050 c025_r071 c025_r141"
dep=${1:+--dependency=afterany:$1}
for n in $RUNS; do
  p=params/param_halo441_1024_cs_$n.txt
  [ -f "$p" ] || { echo "missing $p"; exit 1; }
  j=$(sbatch --parsable $dep -J cs_$n --export=ALL,CSGRID_EXE=$EXE ~/runCSgrid.slurm "$p")
  echo "$n $j" | tee -a ~/cs_grid_jobs.txt
  dep="--dependency=afterany:$j"
done
