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
