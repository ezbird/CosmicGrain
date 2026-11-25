# CosmicGrain run tracker

Last updated: 2026-11-22 09:40 MST. Claude updates this file whenever a job starts, finishes or a decision is made.
Quick status from a terminal: `tmux ls`, `cat dT_tests.status`, `cat analysis/cs_grid/summary.txt`,
`ssh colibri squeue -u ez`.

---

## 1. 1024³ C–S grid: bring the R_V structure function down to Zhang & Green, keep the HD23-like GSD

**Goal.** S2_RV is +0.83 dex above ZG (R_V scatter 2.5× too large; the normalised shape already matches).
Find coagulation (C) and shattering (S) rates that lower the scatter while keeping the 5–20 kpc GSD.

**Design.** 4×4 grid: c = coag rate / baseline ∈ {0.25, 0.5, 1, 2}; r = (C/S) / baseline ∈ {0.5, 0.71, 1, 1.41}.
DustCoagulationCalibration = 22.3/c, DustShatteringCalibration = 2.97·r/c (larger = slower). Growth 1.12, BR06 on.
Baseline = paper run `halo441_output_1024_br06_velfix` (22.3 / 2.97). Plus a duplicate baseline on 2 nodes (noise floor).

**Where.** Colibri, 1 node per run, sequential chain (jobs 610–625), duplicate = job 626 (waits for 2 free nodes).
Code 11d2b48 (timestep limiter + vsig on), binary `~/jobexe/CosmicGrain.csgrid.11d2b48`. ~3 h per run.

**Results appear automatically.** cygnus tmux `csgrid` fetches each finished run and scores it:
- `analysis/cs_grid/summary.txt`: one line per run
- `analysis/cs_grid/summary.png`: heatmaps on the c–r plane
- `analysis/cs_grid/scores.csv`: per snapshot (27–29)
- `analysis/cs_grid/watch.log`: fetch/score log

**Score targets** (baseline velfix in brackets): S2 log-ratio → 0 [+0.83], σ_RV ratio → 1 [2.5],
5–20 kpc GSD distance to HD23 ≤ 0.37 [0.37], distance to baseline GSD small [0], f(<20 nm) → 0.23 [0.12], median R_V ~3.1 [3.65].
Snapshot-to-snapshot spread of S2 ≈ ±0.07 dex.

| run | job | status | S2 dex | σ_RV ratio | GSD→HD23 | GSD→base | R_V |
|---|---|---|---|---|---|---|---|
| velfix (reference) | – | done | +0.83 | 2.53 | 0.37 | 0 | 3.65 |
| c100_r100 (baseline, new code) | 610 | done | +0.83 | 2.83 | 0.35 | 0.09 | 4.09 |
| c050_r100 | 611 | done | +0.34 | 1.45 | 0.48 | 0.16 | 3.10 |
| c200_r100 | 612 | done | +0.68 | 2.48 | 0.35 | 0.10 | 3.95 |
| c025_r100 | 613 | done | **+0.10** | **0.96** | 0.50 | 0.25 | 3.02 |
| c100_r050 | 614 | done | **+0.06** | **1.05** | 0.41 | 0.13 | **3.11** |
| c025_r141 | 787 | done | +0.28 | 1.28 | 0.47 | 0.18 | 3.00 |
| lane A (n9): c100_r071 → c200_r141 | 788–789 | 788 running (done ~19:10), 789 ~22:50 | | | | | |
| lane B (full node, starts when a node frees): c050_r141 → c100_r141 → c200_r050 | 798–800 | waiting for a node | | | | | |
| lane C (full node, starts when a node frees): c025_r071 → c200_r071 → c050_r050 | 801–803 | waiting for a node | | | | | |
| lane D (40 cores on n8, shared): c050_r071 → c025_r050 | 806–807 | 806 running, 4× slower than a full node (~14.5 h/run): done ~00:30, 807 ~Sun 15:00 | | | | | |
| c100_r100_dup (2 nodes) | 626 | done | +0.72 | 2.07 | 0.40 | 0.11 | 4.04 |

Expected completion: ~Monday 23 Nov midday (runs take ~3.5 h, not 3), sooner if nodes free up.
**11-21 09:50, chain split into lanes.** The other user's CFD jobs reserve 8 GB per core, so n5–n7 have no
memory left despite their idle cores. Only n8 had room (40 cores). Lanes B and C start on any node that
frees (when the 2048 test stops, or the CFD jobs end). Same binary (11d2b48) and ΔT 1e7 everywhere.
**Noise floor (dup vs c100_r100; differs in core count and chaos): S2 ±0.1 dex, σ_RV ratio ±0.4, GSD→HD23 ±0.05,
R_V ±0.05.** Core count matters no more than run-to-run chaos does.

**11-22 09:40: 14 of 16 grid runs scored, plus the dup** (full table: `analysis/cs_grid/summary.txt`, heatmaps: `summary.png`).
Remaining: c200_r050, c050_r050 (full nodes), c025_r050 (n8).
S2 ≤ 0.15 dex: c100_r050 (+0.06, GSD 0.41), c025_r100 (+0.10, GSD 0.50), c050_r141 (+0.10, GSD 0.48).
On the c = 1 column, r alone takes S2 from +1.08 (r141) to +0.06 (r050), monotonic. At c ≤ 0.5, S2 is ~0.1–0.35 everywhere.
The c = 2 column keeps the GSD closest to HD23 (0.31–0.35) but S2 stays at ~0.7. c200_r050 decides whether that column can reach both.
**Best so far (11-21 17:00): c100_r050**, i.e. baseline coagulation with 2× faster shattering: S2 +0.06, σ_RV ratio 1.05,
R_V 3.11, f(<20 nm) 0.21 (HD23 0.23), and a 5–20 kpc GSD distance of 0.41 (dup baseline 0.40, so within noise).
Lowering C/S through faster shattering fixes S2 without drifting the GSD; slowing coagulation drifts it.
The r050 row (c050/c200/c025) is next to confirm.
Full Paper II figure set for c100_r050 (snap 29, own dust half-mass radius 1.90 kpc): `paper_II_figures_c100_r050/`.
Caveat: the A_V S2 sits above ZG in the new-code runs (+0.69 c100_r100, +1.36 dup, +0.78 c100_r050 vs +0.24 velfix,
at 2.57 kpc). This comes from the limiter code, not the calibration. Watch it as a second score.
**First axis (11-21):** slowing coagulation lowers S2 steeply. At c = 0.25, S2 reaches +0.10 dex and the σ_RV ratio
0.96, but the GSD drifts from HD23 (0.50 vs 0.35; f(<20 nm) 0.33 vs HD23 0.23; a_med 56 vs 104 nm). The r rows
(less shattering at low c) are what may recover the GSD; watch c025_r141 and c050_r141.

Note: c100_r100 (new code, limiter + vsig) reproduces velfix's S2 (+0.83) but has a higher median R_V (4.09 vs 3.65).
**Compare grid runs to c100_r100, not to velfix.**

**Second score at R = 3.5 kpc (~2 disk scale lengths)** in `analysis/cs_grid_R3.5/summary.txt`. A Sun-like 5.5 kpc
(3 R_d) is too dust-thin: 4 of 8 observers have no pixels with E(B-V) ≥ 0.03. Velfix snapshot 29, S2 excess by radius:
2.57 kpc +0.81, 3.5 kpc +0.93, 4.5 kpc +0.91, 5.5 kpc +1.03 (2 observers only). **The ~1 dex gap does not come from
where the mock Sun sits.** At 3.5 kpc, snapshots 27–29: velfix +0.86, c100_r100 +0.81.

**Open:** check how much the Zhang & Green inference compresses R_V variance (could be part of the 1 dex gap).

---

## 2. Why 2048³ is dust-poor: star-formation / feedback convergence

**Finding.** The 2048 central runs out of gas (z=0 cold gas 5.7e8 vs 7.8e9 M☉ at 1024; dust 7e5 vs 1.2e7).
Dust per metal is fine (D/Z 0.094 vs 0.104), so it is a gas problem. Zoom SFR rises with resolution
(z=2–3: 512 23, 1024 28, 2048 53 M☉/yr). At z≈2, 2048 has 1.4–1.7× more star-forming gas, 30% shorter
depletion time (0.87 vs 1.22 Gyr) and 2× denser star-forming gas. The loss predates the limiter.

**Theories.**
1. SN heating ΔT too low (hardcoded 1e7 K; EAGLE uses 10^7.5): heated gas in denser high-res gas radiates
   the energy away. **Testing now.**
2. Springel–Hernquist SF law (t* ∝ ρ^-1/2) is not resolution-independent → pressure law
   (Schaye & Dalla Vecchia 2008). Bigger code change; long-term fix for 4096.
3. ~~Heated star-forming gas snapped back to the effective EOS~~ **ruled out** (sfr_eos.cc: hot SF gas
   stops forming stars and cools radiatively, only floored at the EOS energy).

**Test (cygnus, tmux `dTtest`, script `run_dT_tests.sh`, status `dT_tests.status`).**
`CG_FEEDBACK_DELTA_T` env switch (feedback.cc, default 1e7 K), binary `CosmicGrain_dT`.
Each run stops at z=1.8 (snapshot 14). Compare zoom SFR(z), star-forming gas mass, central M* at z=2.5–1.8
and the 512→1024 trend.

| run | ΔT | status | control |
|---|---|---|---|
| halo441_output_512_dT3e7 | 10^7.5 | done | 512_lim_vsig (ΔT 1e7, same code) |
| halo441_output_512_dT1e8 | 10^8 | done | 〃 |
| halo441_output_1024_dT3e7 | 10^7.5 | done to z=1.8; resumed; stopped at snap 25 on TimeLimitCPU (85% of 80000 s); limit raised to 300000 and **resumed again 11-22 09:30** (~4–5 h to z=0) | Colibri c100_r100 (grid baseline) |
| halo441_output_1024_dT1e8 | 10^8 | done | 〃 |

512 results, mean SFR in M☉/yr (box total):

| z | ΔT 1e7 (control) | 10^7.5 | 10^8 |
|---|---|---|---|
| 3–4 | 51 | 43 | 40 |
| 2.5–3 | 32 | 28 | 39 |
| 2–2.5 | 25 | 21 | 25 |
| 1.8–2 | 25 | 22 | 21 |

At 512 the effect is modest: 10–20% suppression, and 10^8 is noisy. What matters is the 1024 runs, i.e. whether the 512→1024 rise shrinks.

**1024 results (11-21).** Mean SFR (M☉/yr):

| run | z 3–4 | 2.5–3 | 2–2.5 | 1.8–2 |
|---|---|---|---|---|
| 512 ΔT 1e7 | 51 | 32 | 25 | 25 |
| 512 ΔT 10^7.5 | 43 | 28 | 21 | 22 |
| 1024 ΔT 1e7 (c100_r100, same code) | 67 | 38 | 47 | 44 |
| 1024 ΔT 10^7.5 | 66 | 30 | 26 | 25 |
| 1024 ΔT 10^8 | 58 | 28 | 23 | 25 |
| 1024 velfix (old code, no limiter) | 54 | 28 | 27 | 32 |

512→1024 rise at z = 1.8–2.5: ×1.8 at 1e7 vs ×1.2 at 10^7.5. **10^7.5 (the EAGLE value) mostly removes the
resolution trend**, and 10^8 adds little beyond it. Candidate: ΔT = 10^7.5 → 2048 test to z=2.

**2048 test RUNNING (11-21 09:40):** Colibri job 786 `cg2048dT`, 2 nodes (n10–11), ΔT 10^7.5 (env verified on
the ranks), params/param_halo441_2048_dT3e7.txt (= 2048 vsig physics, fresh start), binary 2429f28 code.
Output ~/CosmicGrain-br06/halo441_output_2048_dT3e7. cygnus tmux `stop2048` writes the stop file after
groups_014 (z=1.8); restart files allow continuing to z=0. Pace: a=0.198 (z=4.1) after 7.3 h, ~0.008/h → z=1.8 around Sun 22 Nov afternoon or evening. Judge: zoom SFR at z=1.8–3 vs 1024 dT3e7 (26/25).

Next if a ΔT converges 512→1024: 2048 to z=2 on Colibri (~12–17 h, 3 nodes), then a full 2048.
Note: a feedback change will shift the 1024 GSD/R_V too → re-check the grid winner afterwards.

---

## 3. 2048³ timestep limiter (crash prevention for 2048 / 4096)

- **Job 480** (plain limiter): reached z=0 in 16.9 h, no crash; 2709 density jumps (max ×24 000).
  z=0 copied to `halo441_output_2048_br06_lim`. Central is gas/dust-poor (see §2).
- **Job 552 `cg2048vsig`** (Colibri, 2 nodes): same restart (snap 24) with the stronger signal-velocity
  wake rule. Early: fewer, smaller jumps (102 vs 150; max ×737 vs ×1505), almost none on bins 17–18, same cost.
  **Result (stopped 11-20 23:20 at a = 0.850).** a = 0.769–0.850, same restart and same 374 cores:

  | | 480 (limiter) | 552 (limiter + vsig) |
  |---|---|---|
  | density jumps | 1298 | 881 (−32%) |
  | jumps > ×100 / > ×1000 | 86 / 10 | 56 / 5 |
  | largest jump | ×8028 | ×4823 |
  | jumps on bins 17–19 | 709 | 90 |
  | sync points | 17 926 | 25 065 |
  | wall time | 6.6 h | 7.7 h (+17%) |

  vsig removes most jumps on the longest steps and halves the extreme ones, but it does not eliminate them.
  At 4096 the extremes matter most, so **keep vsig on** (default) for +17% cost.
  Decision 11-20: **stop at a = 0.85** (a background watcher scancels it); running to z=0 would only reproduce 480's gas-poor galaxy. Its 2 nodes then go to grid job 626.
- 512 tests: limiter on/vsig raise central M* by 20–30% vs off (noise ~15%); vsig cost +28% wall at 512.

---

## 4. Paper II manuscript and figures

- Draft: `paper_II_manuscript/paper_II.tex` (figures 1–7 + 8d in place, captions written).
  To fix: cite keys `Schlafly2016`, `Hensley2023`; stale "Discussion"/"Planned Figures" text; old PNG figure near the top.
- Figures: `python3 scripts/cosmicgrain_paper_II_figures.py --run halo441_output_1024_br06_velfix
  --zg-map data/observations/zhang_green/Rv_map_new.h5 --outdir paper_II_figures_v2 --skip-analysis`
- New figure 5f (`figure5f_mocksun_example`): ZG vs best-matching CG mock-Sun cutout + R_V PDFs + S2 (remake of the old 4-panel plot). Open: mock-Sun radius 2.57 kpc is only ~1.5 disk scale lengths (Sun ≈ 3 R_d ≈ 5.5 kpc here) — consider rescoring grid at ~5.5 kpc too.
- Figures will need regenerating from the grid winner (or a new paper run) once chosen.

---

## 5. Path to the final main-branch commit
Branch `fix/energy-ledger` carries: energy ledger + SN/AGB guards (69a798b), restart ages (f660434), hydro watchdog (72a3cba, f343548),
timestep limiter (ce434b0) + vsig rule (11d2b48), C–S grid tooling (d1a36e0), ΔT switch (2429f28).
Before merging: choose final ΔT / SF law and C–S calibration, update param defaults and docs, a clean 1024 paper run on the final commit,
512 regression vs the reference, then merge + tag.

## Decisions waiting on you
- [x] Job 552: stop at a = 0.85 (decided 11-20).
- [x] Commits: fine to keep committing on fix/energy-ledger; **end goal = one final, well-tested merge to main** (see §5).

## Log
- 11-22 09:50 Paused csgrid and stop2048 watchers during the Colibri SSH outage (possible IP block on cygnus); one probe every 30 min. Restart both when back: tmux new -d -s csgrid 'bash scripts/cs_grid_watch.sh'; tmux new -d -s stop2048 'bash scripts/colibri/stop_2048dT_at_z18.sh'.
- 11-22 09:30 Colibri SSH refuses connections since some time after 05:59 (handshake reset; TCP fine). Jobs unaffected; watchers retry. If it persists the 2048 stop at z=1.8 cannot be written; stop it manually later. cygnus dT3e7 hit TimeLimitCPU at snap 25, resumed.
- 11-21 17:00 c100_r050 is the best grid point so far (S2 +0.06, GSD within noise of baseline). c025_r141 S2 +0.28, GSD 0.47.
- 11-21 09:50 Grid re-queued as 4 lanes (A–D); old 615–625 chain cancelled (backup ~/cs_grid_jobs.chain-backup.txt on Colibri).
- 11-21 09:40 Started 2048 ΔT=10^7.5 (job 786, 2 nodes) and resumed 1024 dT3e7 to z=0 on cygnus. Grid c-axis scored: c025 S2 +0.10 but GSD drifts.
- 11-21 09:35 Watcher fix of 11-20 broke rsync (ssh -n); runs 611/612/613/626 were marked scored without data. Fixed (rsync </dev/null; scoring now fails if nothing was scored) and refetching. ΔT 1024 tests done: 10^7.5 converges SFR 512→1024.
- 11-20 23:20 Job 552 auto-cancelled at a=0.850; compared to 480 (§3). Job 626 (dup) now waits on 2 free nodes.
- 11-20 23:05 Grid watcher had stopped after run 610 (ssh inside the read loop ate the job list). Fixed with ssh -n and restarted (tmux csgrid); it also scores at 3.5 kpc now. 552 at a=0.846, auto-cancel at 0.85. ΔT 512 tests done; 1024 10^7.5 at z≈3.3.
- 11-20 19:00 Colibri cleanup (372 → 94 GB): deleted 1024 br06/shat3/paper/colibri/celldir/dist*/scale380, 2048 sf03/sf10/hwatch*/ledger, 512 br06_velfix, restart files of 2048_br06 and 2048_br06_lim. Copies of br06, shat3, paper, sf03, sf10 verified on cygnus. Kept: velfix, grid, 2048_br06 (snapdir_024 for 552/480), 2048_br06_lim, 2048_br06_vsig, ICs.
- 11-20 18:21 ΔT tests started on cygnus (512 10^7.5 first).
- 11-20 17:00 C–S grid submitted (610–626); grid A/B/C deleted from Colibri (verified copies on cygnus).
- 11-20 15:05 Job 480 finished (2048 limiter, z=0).

### 2026-11-22 16:17 — cygnus 1024 dT3e7 reached z=0 (snap 29)
Same calibrations as c100_r100; only change is ΔT = 10^7.5 K. Scores (snaps 27–29):
S2 0.95 (2.57 kpc) / 0.95 (3 kpc) / 0.89 (3.5 kpc); GSD→HD23 0.447; GSD→base 0.123; f<20 nm 0.056 (c100_r100 0.162); R_V 3.65.
Galaxy (snap 27/28/29) vs c100_r100: map dust mass +55–75%; dust half-mass radius 2.85/3.26/3.69 vs 1.90/1.77/2.76 kpc; SFR at z=0 1.0 vs 1.8 M☉/yr.
Reading: the disk is more extended and dustier, as hoped, but the small-grain peak is mostly gone, so the C–S calibration must be re-tuned for 10^7.5 K.
Plot: analysis/cs_grid/compare/dT3e7_vs_c100*.png. cygnus is now idle.

### 2026-11-22 18:16 — cygnus: 1024 dT3e7 + c100_r050 started (tmux dTr050)
Final-recipe check: ΔT = 10^7.5 K (CG_FEEDBACK_DELTA_T=3.16e7) + ShatCal 1.485 (c100_r050), z=99→0, 24 ranks, CosmicGrain_dT.
params/param_halo441_1024_dT3e7_c100_r050.txt, run_dT1024_r050.sh; log in dT_tests.status. Expected ~16 h → done ~10:00 on 23 Nov.
Colibri still refusing ssh at 18:12 (connection reset); watchers csgrid/stop2048 remain paused.
Galaxy (snap 29, <30 kpc): velfix / c100_r100 / c100_r050 / dT3e7 — M* 6.7/7.5/7.6/4.5e10; Mdust 1.19/0.93/0.92/1.61e7; D/G .0014/.0011/.0012/.0025; D/Z .10/.13/.12/.18; dust r_half 3.3/3.6/2.4/4.8 kpc; star r_half ~3 kpc all. (scratchpad galprops.py)

### 2026-11-23 09:xx — morning check
- cygnus dTr050 at z=0.63 (snap 20) after 15 h; faster than dT3e7 was. ETA z=0 ~late tonight / early 24 Nov (earlier "16 h" estimate was wrong: dT3e7 took ~33 h total).
- Colibri: still connection reset at 09:08.
- Why dT3e7 lost small grains (snap 29, dust within 30 kpc, nearest-gas): not sputtering (dust in T>5e5 K gas 1.0% vs 1.3%). Dust sits in more diffuse gas: n_H<1 holds 94% of dust (c100_r100 78%, c100_r050 54%); essentially none >15 cm^-3 in any run. f<20 is low in every density bin (0.03-0.06 vs 0.05-0.19). Shattering rate scales with gas density, so 10^7.5 K likely needs more than r=0.5 boost; r~0.25 candidate.
- Small-grain deficit is not a shell-placement artifact: f<20 ~0.06 in 1-2 r_half and 2-4 r_half alike (10^7 K runs: 0.1-0.25, highest in outskirts).

### 2026-11-23 09:57 — Colibri back
- All grid jobs COMPLETED during outage: 800 c200_r050, 803 c050_r050, 807 c025_r050 (+ earlier ones). Watcher csgrid restarted (reniced) to fetch/score them.
- 2048 job 786 (10^7.5 K): only at z=2.58 after 48 h (snap 11 = z=3 at 22 Nov 13:14; ~13–20 h per snapshot now). Wall time ends 24 Nov 09:40, far short of z=1.8. Periodic restart files every 2 h. New tmux stop2048 writes the stop file at 24 Nov 08:30 (scripts/colibri/stop_2048dT_before_walltime.sh). Old z=1.8 stop script retired.
- Mean SFR over z=3–2.6 (a 0.25–0.279), M☉/yr:
  512: 10^7 K 29 (br06) / 33 (lim_vsig); 10^7.5 K 29
  1024: 10^7 K 42 (c100_r100); 10^7.5 K 32
  2048: 10^7 K 46 (br06, older code); 10^7.5 K 48
  → 10^7.5 K suppresses SFR at 1024 but NOT at 2048; the 2048 excess (+50% over 1024) persists.

### 23 Nov 12:40 status check
- cygnus dTr050 (dT3e7 + c100_r050): z = 0.43, snapdir_022; ~10 h to z = 0 at the current rate (ETA ~23:00 tonight). Watcher bqqipzhb7 still armed.
- Colibri 786 (2048 dT3e7): z = 2.51, 21 h wall left; snapshot 012 (z = 2.5) due within ~1 h; restart files written 11:54. Pace ~0.019 in ln a per hour, so z ~ 2.0 by the 08:30 stop file (snap 013 at z = 2 borderline). Grid jobs 787-807 all COMPLETED and scored.
- 12:46 Decided with user: stop 786 early. tmux stop2048 now runs scripts/colibri/stop_2048dT_after_snap12.sh (stop file once groups_012 is written and 5 min old; fallback 24 Nov 08:30). No resubmit.
- Speed: 786 (374 cores) reached a=0.25 after 99 600 s wall vs 8 300 s for the old 2048_br06 (748 cores, older code, 10^7 K) → 12× slower for 2× fewer cores. Likely hotter 10^7.5 K gas (higher signal speeds, shorter steps) plus limiter/vsig wake-ups; not yet profiled. Matters for any production 2048 at 10^7.5 K.

### 23 Nov 13:10 — why job 786 (2048, 10^7.5 K) is 12–17× slower per core than the old 2048 run
- Not hardware: n10/n11 have no swap, 630 GB available, CPUs at 3.5 GHz; IB links Active 200 Gb/s, no errors.
- Not physics work: grav-updates per a-bin identical to old 2048_br06 (e.g. 1.54e9 vs 1.51e9 at a=0.25-0.286); 786 splits them into 2× more sync points.
- Not the 10^7.5 K physics per se: the 1024 runs at 10^7.5 K cost the same core-hours as the grid runs; feedback kernel sizes of young stars at z=3 are the same in both 2048 runs (median ~190 gas in r0; scratchpad fbkernel.py).
- Where the time goes (a = 0.25-0.286, 23.6 h): 96% in 2290 steps > 5 s (old run: 60 such steps, all full syncs). 59 AGB full-sync windows = 3.2 h (up to 500 s each, ~320k AGB events per window through cg_fb_dist::find_kernels); the other 2231 slow steps = 19.4 h.
- Live stack samples (eu-stack via srun --overlap): ranks sit for tens of seconds in MPI collectives inside CosmicGrain routines: estimate_local_dust_to_gas_ratio / CellDirectory::lookup (Alltoallv), associate_dust_with_gas / resolve_remote_gas (Alltoall), cg_fb_dist::find_kernels (Allreduce/Allgather, AGB + SN), and domain_do_local_refine (Sendrecv, ~25 s in one step). Fixed per-step costs (logs, drift) grew 20-40× over the run on Colibri but stayed flat on cygnus.
- MPI transport: ranks use Open MPI ob1/TCP over IPoIB (OMPI_MCA_btl_tcp_if_include=ib0; no libucp mapped, no /dev/infiniband fds) although the spack Open MPI has pml ucx. All multi-node CosmicGrain jobs have run this way.
- Working diagnosis: CosmicGrain's distributed dust/feedback routines do several all-to-all exchanges per step whose volume scales with total dust (0.7M -> 4.1M) and star counts, not active particles; over TCP across 374 ranks they become latency/bandwidth bound and dominate. Next: (1) short benchmark restart of 786 with UCX (OMPI_MCA_pml=ucx) vs TCP; (2) cut dust D/G refresh and AGB-window volume (only active dust; batch AGB events).
- 13:20 Submitted transport benchmarks on Colibri, both `--dependency=afterany:786`, 2 nodes, 1 h 05 min, same binary (jobexe/CosmicGrain.786) and env: **1192 bench_tcp** (pml ob1, TCP over ib0, as 786) and **1193 bench_ucx** (OMPI_MCA_pml=ucx, UCX_NET_DEVICES=mlx5_0:1). Script ~/bench2048.slurm; each copies 786's final restart files into halo441_output_2048_dT3e7_bench_{tcp,ucx} (delete after, 12 GB each). Compare sync points and a reached per wall hour, and dust/feedback ms per step.
- 14:50 **Benchmark result.** 786 completed cleanly 13:36 at a=0.28628 (snap/groups 012 written). 1193 bench_ucx (n10–11, OMPI_MCA_pml=ucx): 6091 steps, a 0.2863 → 0.3039 in 1.06 h = 629 ms/step (786 over TCP just before: 5107 ms/step, Δa 0.0077 in 3.8 h) → **~8× faster**. Per step: dust 241 vs 1181 ms, feedback 84 vs 599, drift 4 vs 273, logs 3 vs 114, domain 47 vs 306. Pace ≈ 0.06 in ln a per hour → z=0 in roughly 1–2 days on 2 nodes.
  1192 bench_tcp (n8–9, pml ob1/TCP) **segfaulted after 9 steps** (sync 79555, 51 ranks, in resolve_remote_gas ← associate_dust_with_gas, dust.cc); UCX passed the same step. No core files. Cause unknown (latent dust-association bug exposed by a different reduction order, or transport); no fresh-process TCP baseline, so the 8× includes any restart effect, but drift/logs dropping to old-run levels points at the transport.
  Restart-file copies deleted; bench dirs keep logs only. Recommendation: use UCX for all multi-node runs (runBR06.slurm: export OMPI_MCA_pml=ucx UCX_NET_DEVICES=mlx5_0:1; drop the tcp ib0 include).
- 16:05 runBR06.slurm on Colibri switched to UCX (user OK); backup ~/runBR06.slurm.bak_tcp; repo copy scripts/colibri/runBR06.slurm.
- TCP bench crash explained: the fault is in the reply loop of resolve_remote_gas (links[back[k].slot], garbage slot = corrupt/incomplete reply data) during the 2nd of three DUST_ASSOC_VERIFY self-checks that run after every (re)start. Each sends every live grain to every task: 4.18M grains × 373 tasks ≈ 1.6e9 queries per call (~29 s each even over UCX). Fix in working tree (uncommitted, not yet compiled — cygnus busy): skip that self-check when global live grains > 200 000 (DUST_SELFCHECK_MAX_GRAINS). Compile + 512 smoke test after the 1024 run ends.

### 23 Nov evening — 1024 vs 2048 at z = 3 and 2.5 (analysis/res_compare/)
Runs: 512/1024/2048 at 10^7 K (512_lim_vsig, 1024 cs_c100_r100, 2048_br06 = older code) and 10^7.5 K (512/1024/2048 dT3e7).
- **Main halo is converged:** M200 1.0e12 (z=3) / 1.45e12 (z=2.5) identical; at 10^7.5 K M*(<10 pkpc) 3.97→4.56e10 (z=3), 4.48→5.00e10 (z=2.5); SFR within R200 21→29.5 (z=3), 12.9→13.1 (z=2.5); M_gas(<10 pkpc) 0.79→0.77e10.
- 2048 central is more compact (r_half,* 0.86→0.62 pkpc at z=3), SF gas denser (mass-wt p90 nH 27→100 cm^-3 at z=3, 11→26 at z=2.5), t_dep shorter (0.30→0.23, 0.59→0.53 Gyr). Less warm-hot halo gas (3e5–3e6 K: 1.97→1.42e10 at z=3; 3.03→1.97e10 at z=2.5; M_gas(<R200) −18% at z=2.5). Dust in central lower (z=3 D/Z ≈ 0.09 vs 0.04; z=2.5 0.17 vs 0.14).
- 10^7.5 K suppresses the central's early SF at 1024 (z=8–5 SFH 11→5, 29→12 Msun/yr) but not at 2048 (11→15, 32→27).
- **Zoom SFR excess is in the other halos, mainly 1e10–1e11 Msun:** z=2.5 total SFR 24.9 (1024 3e7) vs 46.3 (2048 3e7); main 12.9 vs 13.1; other halos ≥1e10: 9.4 vs 30.3. Stacked 1e10–1e11 halos at z=2.5: 10^7.5 K cuts M* 19.2→11.0e9 and SFR 16.3→6.6 at 1024, but at 2048 M* 21.3→20.8e9, SFR 26.0→23.3 (no effect). Their SF gas is far below the Dalla Vecchia & Schaye critical density (p90 < 5 vs n_crit 65–184 cm^-3), so classic numerical overcooling does not explain it — physics predicts 10^7.5 should work at least as well at 2048. Suspect: how the stochastic heating is applied at 2048 in low-mass halos (needs feedback diagnostics on a short 2048 run).

## 23 Nov 18:10 — release plan agreed (v1.2.0)
- dTr050 (1024, 10^7.5 K + c100_r050, code 2429f28) at z=0.23; ETA z=0 ~00:00-02:00 24 Nov. Watcher bqqipzhb7.
- On finish: score (f<20 nm, GSD L1 vs HD23, S2 vs ZG, R_V, galaxy props) vs dT3e7 and c100_r050. User decides if it is THE Paper II run. If small grains short -> ShatCal ~0.74 rerun (~30 h) before tagging.
- If accepted: v1.2.0 = 2429f28 + result-neutral changes only (dust self-check cap verified bit-identical on 512, param defaults = final recipe, UCX Colibri script, RC1 guard -> warning, user's docs edits). Push fix/energy-ledger, PR to main, tag v1.2.0, release notes; delete merged side branches (all verified contained, incl. cherry-picked hotfixes).
- 2048 feedback issue in 1e10-1e11 halos -> v1.2.1 if code changes; does not block v1.2.0.

### 2026-11-24 01:30 — dTr050 (10^7.5 K + c100_r050) finished and scored
z=0 at 00:59, rc=0. Scores, snaps 27-29 (S2 dex at 2.57 / 3 / 3.5 kpc):
| run | S2 dex | L1 HD23 | f<20 | a_med | R_V | M* | Mdust | D/Z | SFR |
| c100_r050 (10^7) | 0.06/0.07/0.10 | 0.413 | 0.212 | 89 | 3.11 | 7.6e10 | 0.92e7 | .12 | 2.9 |
| dT3e7 (10^7.5, r100) | 0.95/0.95/0.89 | 0.447 | 0.056 | 104 | 3.65 | 4.5e10 | 1.61e7 | .18 | 1.0 |
| dTr050 (10^7.5, r050) | 0.27/0.30/0.35 | 0.485 | 0.063 | 94 | 3.32 | 4.1e10 | 1.20e7 | .17 | 2.1 |
Reading: halving the shattering timescale fixed most of the S2 excess (9x -> 1.9x ZG) and R_V, but the small-grain peak is still missing (f<20 0.063 vs HD23 0.23); shattering lever on f<20 is weak at 10^7.5 K (0.056 -> 0.063), so r=0.25 alone is unlikely to restore it. Plot analysis/cs_grid/compare/final_dTr050.png. Awaiting user decision on the Paper II run. cygnus idle.

### 2026-11-24 11:10 — Colibri: three 10^7.5 K runs to recover the small-grain peak
User chose a 3-run bracket (2 nodes each, UCX, ~3 h each). Binary ~/jobexe/CosmicGrain.786 (2429f28 code), CG_FEEDBACK_DELTA_T=3.16e7, script scripts/colibri/runDTcs.slurm (~/runDTcs.slurm), jobs in ~/dT_cs_jobs.txt.
| run | Coag | Shat | tau_coag | tau_shat | job |
| dT3e7_cs_c025_r071 | 5.575 | 2.1 | 7.6 Myr | 114 Myr | 1423 (n8-9) |
| dT3e7_cs_c050_r050 | 11.15 | 1.485 | 15 Myr | 80 Myr | 1424 (n10-11) |
| dT3e7_cs_c100_r025 | 22.3 | 0.7425 | 30 Myr | 40 Myr | 1425 (pending, Resources) |
Why: halving tau_shat at 10^7.5 moved f<20 only 0.056 -> 0.063, with the same number of shattering events (1.87M vs 2.00M at 10^7); 44-62% of shattering calls fail the size-dependent velocity gate (only large grains shatter), and in the 10^7 grid faster coagulation gave the most small grains (c025_r071 f<20 0.37). Watcher: tmux dTcs (scripts/dT_cs_watch.sh) fetches + scores at 2.57/3/3.5 kpc; log analysis/cs_grid/watch_dT.log. Note summary.png heatmaps place these on the same (c, r) points as the 10^7 grid; use summary.txt rows by name.

### 2026-11-24 17:15 — 10^7.5 K bracket scored (Colibri 1423-1425, 2 nodes, 380 ranks)
| run | S2 x ZG | R_V | f<20 | a_med | L1 HD | M* | Mgas | Mdust | gas rh | dust rh | SFR |
| 10^7 c100_r050 (Colibri 190) | 1.15 | 3.11 | 0.212 | 89 | 0.41 | 7.6e10 | 7.9e9 | 0.9e7 | 7.9 | 2.4 | 2.9 |
| dT c100_r050 (cygnus 24) | 1.9 | 3.32 | 0.063 | 94 | 0.49 | 4.1e10 | 4.7e9 | 1.2e7 | 6.8 | 2.5 | 2.1 |
| dT c100_r025 | 6.8 | 3.43 | 0.176 | 66 | 0.49 | 4.1e10 | 9.5e9 | 2.0e7 | 13.0 | 7.2 | 0.6 |
| dT c050_r050 | 45 | 4.48 | 0.063 | 124 | 0.30 | 4.0e10 | 9.7e9 | 2.0e7 | 11.4 | 8.2 | 1.2 |
| dT c025_r071 | 95 | 7.75 | 0.024 | 184 | 0.20 | 4.2e10 | 7.8e9 | 1.6e7 | 10.9 | 7.3 | 0.8 |
Faster coagulation backfires at 10^7.5 (grains grow, R_V and S2 blow up). Faster shattering (r025) restores most of the small grains (0.176) and R_V 3.43, but S2 is 6.8x ZG.
Caveat: all three Colibri dT runs have a very different galaxy from the cygnus dT run (gas 2x, gas/dust half-mass radius 11-13 / 7-8 vs 6.8 / 2.5 kpc, SFR 0.6-1.2 vs 2.1) at the same M*. Either run-to-run/rank-count stochasticity at 10^7.5 or a platform difference; it is consistent across all three, so the cygnus run may be the outlier. Mock Suns sit at a fixed 2.57 kpc, inside a 7 kpc dust disk on Colibri. Proposed control: dT c100_r050 on Colibri (same as cygnus run). Plot analysis/cs_grid/compare/dT_bracket.png.

### 2026-11-24 ~18:00 — Colibri: control + repeat + midpoint (user approved)
- 1489 dT3e7_cs_c100_r050 (2 nodes, 380 ranks): control = cygnus dTr050 physics on Colibri. Big disk or compact galaxy?
- 1490 dT3e7_cs_c100_r025_1n (1 node, 190 ranks, scripts/colibri/runDTcs1n.slurm): repeat of the best candidate, different rank count.
- 1491 dT3e7_cs_c100_r035_1n (1 node, ShatCal 1.05, tau_shat 57 Myr): midpoint between r050 and r025.
Watcher tmux dTcs restarted (same jobs file). Also rescoring candidates with mock Suns at 5.5 kpc (analysis/cs_grid_R5.5).

### 2026-11-24 21:10 — control / repeat / midpoint scored
| run (10^7.5 K unless noted) | ranks | S2 x ZG (2.57/3.5) | R_V | f<20 | a_med | M* | Mgas | Mdust | D/Z | dust rh | SFR |
| 10^7 c100_r050 | 190 | 1.15/1.26 | 3.11 | 0.212 | 89 | 7.6e10 | 7.9e9 | 0.9e7 | .12 | 2.4 | 2.9 |
| c100_r050 cygnus | 24 | 1.9/2.2 | 3.32 | 0.063 | 94 | 4.1e10 | 4.7e9 | 1.2e7 | .17 | 2.5 | 2.1 |
| c100_r050 Colibri (control) | 380 | 4.3/3.9 | 3.26 | 0.080 | 78 | 3.8e10 | 8.9e9 | 2.6e7 | .22 | 7.0 | 1.5 |
| c100_r035_1n | 190 | 2.6/2.8 | 3.07 | 0.113 | 75 | 4.1e10 | 8.4e9 | 1.9e7 | .20 | 5.7 | 0.9 |
| c100_r025 | 380 | 6.8/6.9 | 3.43 | 0.176 | 66 | 4.1e10 | 9.5e9 | 2.0e7 | .21 | 7.2 | 0.6 |
| c100_r025_1n | 190 | 1.2/1.26 | 3.08 | 0.101 | 75 | 3.9e10 | 7.8e9 | 1.9e7 | .21 | 5.0 | 1.1 |
- Control reproduces the big gas-rich disk on Colibri: the cygnus 24-rank run is the odd one (chance or rank count).
- Identical physics (r025, 380 vs 190 ranks) gives S2 6.8x vs 1.2x ZG and f<20 0.18 vs 0.10: run-to-run scatter at 10^7.5 is large (~0.75 dex in S2), much bigger than the 0.1 dex seen at 10^7.
- Faster shattering raises small grains on Colibri (r050 0.080 -> r035 0.113 -> r025 0.10/0.18) without a clear S2 penalty.
- Best single run: dT3e7_cs_c100_r025_1n (S2 1.2x ZG, R_V 3.08, f<20 0.10, M* 3.9e10, D/Z 0.21). Code = 2429f28 (release code). Plot analysis/cs_grid/compare/dT_r025_final.png.
- CORRECTION 21:30 (user caught it): no 10^7.5 run has a separate small-grain bump. Mass fraction <10 nm / 5-15 nm, 5-20 kpc, snaps 27-29: 10^7 c100_r050 0.095/0.147 (peak at 13 nm); 10^7.5 runs 0.022-0.058 / 0.034-0.101, no peak below 30 nm. Faster shattering raises f<20 by broadening the large-grain peak toward 15-30 nm (shoulder), not by building the 10 nm bump. "Restores about half the bump" was wrong.

### 2026-11-24 22:15 — Paper II run chosen: halo441_output_1024_cs_c100_r050 (10^7 K, 22.3 / 1.485 / 1.12)
User's choice (best GSD incl. the 13 nm bump, S2 1.15x ZG, R_V 3.11; caveat M* 7.6e10 ~2x abundance matching, SFR 2.9).
Code 11d2b48 = release 2429f28 minus the inactive CG_FEEDBACK_DELTA_T override. Log clean. Full output (30 snaps, 9.1 GB) being copied to cygnus (tmux fetchfull).
- Repeats with the release binary (~/jobexe/CosmicGrain.786, no dT env, scripts/colibri/runRel.slurm): rep380 job 1586 (2 nodes), rep570 job 1587 (3 nodes, waiting for nodes). Watcher tmux rep (scripts/rep_watch.sh), log analysis/cs_grid/watch_rep.log.
- Release prep: params/param_base.txt dust/SF defaults = this run; params/param_halo441_1024_paperII.txt = this run's params (relative IC path). run.cc: TimeMax/TimeBegin RC1 checks are now warnings (comoving + AGB mode 2 still required). dust.cc self-check cap. 512 identity test (CosmicGrain_rel vs halo441_output_512_dT3e7, to a=0.2) running in tmux reltest.

## 2026-11-25 — Paper II run repeats and release test

- **512 release identity test passed:** CosmicGrain_rel (run.cc warnings + dust self-check cap) to a=0.21 is bit-identical to the 2429f28 reference (sfr.txt 2309 rows, all Sync-Points). Committed 6c28fda.
- **Repeats of cs_c100_r050 with the release binary (Colibri jobs 1586 rep380, 1587 rep570), snapshots 27-29:**

| run | S2/ZG 2.57 kpc | 3.0 | 3.5 | R_V | <10 nm | 5-15 nm | small peak | M* | SFR | Mdust |
|---|---|---|---|---|---|---|---|---|---|---|
| original | 1.15x | 1.17x | 1.26x | 3.11 | 0.095 | 0.147 | 12-16 nm | 7.57e10 | 2.88 | 9.2e6 |
| rep380 | 3.1x | 3.2x | 3.4x | 3.32 | 0.063 | 0.097 | 11-12 nm | 7.52e10 | 2.35 | 1.1e7 |
| rep570 | 2.0x | 2.3x | 2.3x | 3.56 | 0.067 | 0.114 | 13-16 nm | 7.65e10 | 2.64 | 9.7e6 |

- Galaxy and small-grain peak reproduce. S2 scatters ~0.2 dex; the original is the low (best) end. Ensemble mean S2 ~1.9x ZG, R_V ~3.33. Still the best grid point (neighbours c100_r071 3.2x, c050_r050 2.7x are single runs, so their scatter is unknown).
- **Paper II output: three copies.** Colibri, ~/gadget4/halo441_output_1024_cs_c100_r050, and /media/cygnus/SPACETIME/cosmicgrain_paperII/ (USB backup, rsync rc=0, file lists identical, 11 GB).
- Docs reviewed against the code for v1.2.0 (408fde4); stale comments and the 1024 generator block fixed (71cbb5d).

## 2026-11-25 — Colibri cleanup and first 4096 test
- Copied halo441_output_2048_br06_vsig to cygnus (21 GB, verified). Deleted on Colibri (user OK): all 1024 C-S grid/repeat runs except cs_c100_r050, 1024_br06_velfix, 2048_br06, 2048_br06_lim, 2048_br06_vsig, 2048 bench dirs. Colibri home ~400 GB -> 60 GB. Kept: 1024_cs_c100_r050 (Paper II, 3rd copy), 2048_dT3e7 (+restart files, for v1.2.1).
- Colibri ~/CosmicGrain-br06 fast-forwarded to 46b4b00 (release code); old local edits stashed; 18 untracked param files that differed moved to ~/untracked_pre_v120. Config_zoom.sh listeners=3 (local). Binary ~/jobexe/CosmicGrain.v120.
- **Job 1823 cg4096test** (4 nodes n7-n10, 760 ranks, 24 h): params/param_halo441_4096_test.txt (Paper II recipe, 10^7 K, MaxSizeTimestep 0.00425, coarse-DM softenings matched to 2048, MaxMemSize 3600, TimeMax 0.2, restart files every 4 h). Started 11:34; ICs read, domain decomposition OK.
- **Paper II figures:** `paper_II_figures/` (only folder kept; others deleted per user) from halo441_output_1024_cs_c100_r050 snap 29, mock Suns fixed at 2.574 kpc (grid scoring radius; default would be the run's own 1.90 kpc). z=0: S2(R_V)/ZG at 1 kpc 0.82, median face-on R_V 2.85 (snaps 27/28: 1.15/1.50, 3.12/3.34). At 5.5 kpc: S2(R_V) 2.9x ZG, S2(A_V) ~0.1x ZG; at 2.574 kpc S2(A_V) 4x ZG at 1 kpc (radial A_V gradient of the compact dust disk). Script c50d754.
- **4096 test 1823 at 2.7 h (11-25 14:18):** a=0.077 (z=12), 3018 steps, snapshots 000-003 (6.4-8.1 GB each), largest allocation 2.0 GB/rank. Cost to z=12: 2026 core-h vs 232 (2048 dT3e7) and 259 (2048 br06) -> ~8-9x, as expected for 8x particles. SFR at z=12: 0.62 (1024) / 1.75 (2048 br06, 10^7 K) / 3.17 (4096). 8 HYDRO_WATCH (300-400 km/s next to 1e7 K SN-heated gas), 0 fast particles, ENERGY_JUMP only from feedback/hydro (<3%). Pace slowing (a 0.074->0.077 in 26 min); projected end ~z=4.5-5 at the 24 h limit.

## 2026-11-25 16:22 — DustShatteringMach 512 tests (cygnus, tmux `mach`)
Question: does a larger Mach number in the HA19 grain-velocity gate (a_crit ~ M^-3; we use M=1, HA19 used 3 for shattering)
recover the 5-15 nm small-grain bump in the 10^7.5 K galaxy (better M*, extended dust disk, but no bump)?
Pre-check (snap 29, dust 5-20 kpc): Paper II f(5-15 nm) 0.179, 82% of the bump in grains >200 Myr old (processed, not
fresh SNII); 1024 10^7.5 r050 0.029; 512 lim_vsig (10^7, ShatCal 2.97) 0.030, so 512 must first show the contrast.
Branch feature/shattering-mach: optional runtime param DustShatteringMach (default 1; absent = default). Binary CosmicGrain_mach.
Queue (run_mach_tests.sh, log mach_tests.status), each ~1.7-2 h on 24 ranks, Paper II calibrations 22.3/1.485/1.12:
machtest (identity vs 512_reltest to sfr row 2310) -> mach_dT1e7_M1 -> mach_dT3e7_M1 -> M2 -> M3 -> M1p5.
Score: f(5-15 nm), f(<5 nm), small-peak location, f<20, carbon fraction, galaxy props (bump2.py-style; centre by shrinking sphere).
