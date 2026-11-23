#!/bin/bash
# Job 786 hits its 3-day wall time at 2026-11-24 09:40 before reaching z=1.8.
# Write the stop file at 08:30 so Gadget writes clean restart files and exits.
D=CosmicGrain-br06/halo441_output_2048_dT3e7
while [ $(date +%s) -lt $(date -d "2026-11-24 08:30" +%s) ]; do sleep 300; done
ssh -n colibri "touch $D/stop" && echo "$(date '+%F %T') stop file written for 2048_dT3e7 (before wall time)" >> ~/gadget4/dT_tests.status
