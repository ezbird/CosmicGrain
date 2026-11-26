#!/bin/bash
# D/G estimator timing/accuracy tests: 1024 Paper II run restarted from snapshot 025 (a=0.80), ~15 min each, 24 ranks.
cd ~/gadget4
run() { # name env
  echo "$(date '+%F %T') start $1" >> dtg_tests.status
  mpirun -np 24 $2 ./CosmicGrain_dtg params/param_halo441_1024_dtgtest_$1.txt 2 25 > halo441_output_1024_dtgtest_$1/run.log 2>&1
  echo "$(date '+%F %T') end $1 rc=$?" >> dtg_tests.status
}
run m0 ""
run m1q8 "-x CG_DTG_Q=8"
run m1q4 "-x CG_DTG_Q=4"
echo "$(date '+%F %T') ALL DONE" >> dtg_tests.status
