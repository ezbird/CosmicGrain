#!/bin/bash
# 1024 final-recipe check: SN heating 10^7.5 K + C-S calibration c100_r050 (ShatCal 1.485), z=99 -> 0.
cd ~/gadget4
out=halo441_output_1024_dT3e7_c100_r050
mkdir -p $out
echo "$(date) start $out" >> dT_tests.status
mpirun --bind-to none -np 24 -x CG_TIMESTEP_LIMITER=1 -x CG_FEEDBACK_DELTA_T=3.16e7 ./CosmicGrain_dT params/param_halo441_1024_dT3e7_c100_r050.txt > $out/output.log 2>&1
echo "$(date) end $out rc=$?" >> dT_tests.status
