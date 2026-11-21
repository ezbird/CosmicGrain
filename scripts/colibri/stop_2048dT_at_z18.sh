#!/bin/bash
# Stop Colibri job cg2048dT (halo441_output_2048_dT3e7) once snapshot 14 (z=1.8) and its groups are written.
# Gadget writes restart files on the stop file, so the run can be resumed to z=0 later.
D=CosmicGrain-br06/halo441_output_2048_dT3e7
until ssh -n colibri "test -d $D/groups_014" 2>/dev/null; do sleep 600; done
sleep 120; ssh -n colibri "touch $D/stop"; echo "$(date '+%F %T') stop file written for 2048_dT3e7" >> ~/gadget4/dT_tests.status
