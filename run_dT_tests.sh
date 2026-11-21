#!/usr/bin/env bash
# SN heating DeltaT convergence tests (CG_FEEDBACK_DELTA_T), each stopped at z=1.8
# (after snapdir_014/groups_014) with Gadget's stop file.  One MPI job at a time.
# Controls (DeltaT=1e7, same code): 512 halo441_output_512_lim_vsig;
# 1024 = Colibri grid baseline halo441_output_1024_cs_c100_r100.
cd /home/cygnus/gadget4
STATUS=dT_tests.status
run() {  # deltaT param outdir
  mkdir -p "$3"
  echo "$(date) start dT=$1 $2" >> $STATUS
  mpirun --bind-to none -np 24 -x CG_TIMESTEP_LIMITER=1 -x CG_FEEDBACK_DELTA_T=$1 ./CosmicGrain_dT "$2" > "$3/output.log" 2>&1 &
  local pid=$!
  while kill -0 $pid 2>/dev/null; do
    if [ -d "$3/groups_014" ] && [ ! -f "$3/stop" ]; then sleep 60; touch "$3/stop"; echo "$(date) stop file $3" >> $STATUS; fi
    sleep 30
  done
  wait $pid; echo "$(date) end $3 rc=$?" >> $STATUS
}
run 3.16e7 params/param_halo441_512_dT3e7.txt  halo441_output_512_dT3e7
run 1e8    params/param_halo441_512_dT1e8.txt  halo441_output_512_dT1e8
run 3.16e7 params/param_halo441_1024_dT3e7.txt halo441_output_1024_dT3e7
run 1e8    params/param_halo441_1024_dT1e8.txt halo441_output_1024_dT1e8
echo "$(date) all done" >> $STATUS
