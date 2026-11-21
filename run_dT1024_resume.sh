#!/bin/bash
# Resume halo441_output_1024_dT3e7 (SN heating 10^7.5 K) from its z=1.8 restart files to z=0.
cd ~/gadget4
out=halo441_output_1024_dT3e7
rm -f $out/stop
echo "$(date) resume start" >> dT_tests.status
mpirun --bind-to none -np 24 -x CG_TIMESTEP_LIMITER=1 -x CG_FEEDBACK_DELTA_T=3.16e7 ./CosmicGrain_dT params/param_halo441_1024_dT3e7.txt 1 > $out/output_resume.log 2>&1
echo "$(date) resume end rc=$?" >> dT_tests.status
