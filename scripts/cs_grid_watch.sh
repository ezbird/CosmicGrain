#!/bin/bash
# Poll Colibri for finished C-S grid runs; copy snaps 027-029, all group
# catalogues and logs to cygnus, then score them (scripts/cs_grid_score.py).
# Light work only (rsync + ~1 min of Python per run). Log: analysis/cs_grid/watch.log
cd ~/gadget4
mkdir -p analysis/cs_grid
LOG=analysis/cs_grid/watch.log
while true; do
  jobs=$(ssh colibri 'cat ~/cs_grid_jobs.txt' 2>/dev/null)
  pending=0
  while read -r name jid; do
    [ -z "$jid" ] && continue
    out=halo441_output_1024_cs_$name
    [ -f "$out/.scored" ] && continue
    state=$(ssh colibri "sacct -j $jid -X -n -o State" 2>/dev/null | awk '{print $1}')
    case "$state" in
      COMPLETED)
        echo "$(date '+%F %T') $name ($jid) completed; fetching" >> $LOG
        mkdir -p $out
        rsync -a colibri:CosmicGrain-br06/$out/{snapdir_027,snapdir_028,snapdir_029,cpu.txt,sfr.txt,energy.txt,timebins.txt,dust_evolution.txt,parameters-usedvalues,stellar_evolution.txt} $out/ >> $LOG 2>&1
        rsync -a --include='groups_*/***' --exclude='*' colibri:CosmicGrain-br06/$out/ $out/ >> $LOG 2>&1
        rsync -a colibri:CosmicGrain-br06/slurm-cs_$name-$jid.out $out/ >> $LOG 2>&1
        if python3 scripts/cs_grid_score.py $out >> $LOG 2>&1; then
          touch $out/.scored; echo "$(date '+%F %T') $name scored" >> $LOG
        else
          echo "$(date '+%F %T') $name scoring FAILED" >> $LOG; touch $out/.scored
        fi ;;
      FAILED|CANCELLED*|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL)
        echo "$(date '+%F %T') $name ($jid) ended $state - not scored" >> $LOG
        mkdir -p $out; touch $out/.scored ;;
      *) pending=$((pending+1)) ;;
    esac
  done <<< "$jobs"
  [ -n "$jobs" ] && [ $pending -eq 0 ] && { echo "$(date '+%F %T') all runs handled" >> $LOG; break; }
  sleep 900
done
