#!/bin/bash
# Column D/G estimator tests (same setup as run_dtg_tests.sh).
cd ~/gadget4
run() { echo "$(date '+%F %T') start $1" >> dtg_tests.status
  mpirun -np 24 $2 ./CosmicGrain_dtg params/param_halo441_1024_dtgtest_$1.txt 2 25 > halo441_output_1024_dtgtest_$1/run.log 2>&1
  echo "$(date '+%F %T') end $1 rc=$?" >> dtg_tests.status; }
run colq8 "-x CG_DTG_Q=8"
run colq16 "-x CG_DTG_Q=16"
echo "$(date '+%F %T') ALL DONE 2" >> dtg_tests.status
