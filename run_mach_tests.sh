#!/bin/bash
# DustShatteringMach 512 tests (branch feature/shattering-mach, binary CosmicGrain_mach).
# 1) identity: M default (param absent) vs reltest reference, to sfr row 2310, then stop.
# 2) baselines at 10^7 and 10^7.5 K (Paper II calibration, M=1); 3) M = 1.5, 2, 3 at 10^7.5 K.
cd ~/gadget4
st=mach_tests.status
run(){ # name dT
  out=halo441_output_512_$1; mkdir -p $out
  echo "$(date '+%F %T') start $1" >> $st
  mpirun --bind-to none -np 24 -x CG_FEEDBACK_DELTA_T=$2 ./CosmicGrain_mach params/param_halo441_512_$1.txt > $out/output.log 2>&1
  echo "$(date '+%F %T') end $1 rc=$?" >> $st
}
# 1) identity test
out=halo441_output_512_machtest; mkdir -p $out
echo "$(date '+%F %T') start machtest" >> $st
mpirun --bind-to none -np 24 -x CG_FEEDBACK_DELTA_T=3.16e7 ./CosmicGrain_mach params/param_halo441_512_machtest.txt > $out/output.log 2>&1 &
pid=$!
while kill -0 $pid 2>/dev/null; do
  n=$(wc -l < $out/sfr.txt 2>/dev/null || echo 0)
  [ "$n" -ge 2310 ] && touch $out/stop
  sleep 30
done
if cmp -s <(head -2309 $out/sfr.txt) <(head -2309 halo441_output_512_reltest/sfr.txt) && \
   cmp -s <(grep Sync-Point $out/output.log | head -2300) <(grep Sync-Point halo441_output_512_reltest/output.log | head -2300); then
  echo "$(date '+%F %T') machtest IDENTICAL (sfr 2309 rows, Sync-Points)" >> $st
else
  echo "$(date '+%F %T') machtest DIFFERS - queue stopped" >> $st; exit 1
fi
grep -m1 "DustShatteringMach" $out/output.log >> $st
run mach_dT1e7_M1  1e7
run mach_dT3e7_M1  3.16e7
run mach_dT3e7_M2  3.16e7
run mach_dT3e7_M3  3.16e7
run mach_dT3e7_M1p5 3.16e7
echo "$(date '+%F %T') ALL DONE" >> $st
