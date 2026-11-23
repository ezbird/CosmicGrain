#!/bin/bash
# Stop job 786 (2048 dT3e7) cleanly once snapshot 012 (z = 2.5) and its group catalogue are written.
# Falls back to 2026-11-24 08:30, an hour before the wall time.
D=CosmicGrain-br06/halo441_output_2048_dT3e7
until ssh -n -o ConnectTimeout=20 colibri "test -d $D/groups_012 && test -d $D/snapdir_013 -o -n \"\$(find $D/groups_012 -mmin +5 -print -quit)\"" \
      || [ $(date +%s) -ge $(date -d "2026-11-24 08:30" +%s) ]; do sleep 300; done
ssh -n colibri "touch $D/stop" && echo "$(date '+%F %T') stop file written for 2048_dT3e7 (after snap 012 / fallback)" >> ~/gadget4/dT_tests.status
