#!/bin/bash
# Poll Colibri for the c100_r050 repeats (rep_jobs.txt); copy snaps 027-029, all group
# catalogues and logs to cygnus, then score them (scripts/cs_grid_score.py).
# ssh runs with -n and rsync with </dev/null: inside the while-read loop they would otherwise eat the job list.
# Light work only (rsync + ~1 min of Python per run). Log: analysis/cs_grid/watch.log
cd ~/gadget4
mkdir -p analysis/cs_grid
LOG=analysis/cs_grid/watch_rep.log
while true; do
  jobs=$(ssh colibri 'cat ~/rep_jobs.txt' 2>/dev/null)
  pending=0
  while read -r name jid; do
    [ -z "$jid" ] && continue
    out=halo441_output_1024_cs_c100_r050_$name
    [ -f "$out/.scored" ] && continue
    state=$(ssh -n colibri "sacct -j $jid -X -n -o State" 2>/dev/null | awk '{print $1}')
    case "$state" in
      COMPLETED)
        echo "$(date '+%F %T') $name ($jid) completed; fetching" >> $LOG
        mkdir -p $out
        rsync -a colibri:CosmicGrain-br06/$out/{snapdir_027,snapdir_028,snapdir_029,cpu.txt,sfr.txt,energy.txt,timebins.txt,dust_evolution.txt,parameters-usedvalues,stellar_evolution.txt} $out/ < /dev/null >> $LOG 2>&1
        rsync -a --include='groups_*/***' --exclude='*' colibri:CosmicGrain-br06/$out/ $out/ < /dev/null >> $LOG 2>&1
        rsync -a colibri:CosmicGrain-br06/slurm-$name-$jid.out $out/ < /dev/null >> $LOG 2>&1
        python3 scripts/cs_grid_score.py --observer-radius 3.5 $out < /dev/null >> $LOG 2>&1   ; python3 scripts/cs_grid_score.py --observer-radius 3.0 $out < /dev/null >> $LOG 2>&1
        if python3 scripts/cs_grid_score.py $out < /dev/null >> $LOG 2>&1; then
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
  sleep 600
done
