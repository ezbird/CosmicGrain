# CosmicGrain run tracker

Last updated: 2026-11-20 18:30 MST. Claude updates this file whenever a job starts, finishes or a decision is made.
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
| c100_r100 (baseline, new code) | 610 | running | | | | | |
| c050_r100 | 611 | queued | | | | | |
| c200_r100 | 612 | queued | | | | | |
| c025_r100 | 613 | queued | | | | | |
| c100_r050 / r071 / r141 | 614–616 | queued | | | | | |
| c050_r050 / r071 / r141 | 617–619 | queued | | | | | |
| c200_r050 / r071 / r141 | 620–622 | queued | | | | | |
| c025_r050 / r071 / r141 | 623–625 | queued | | | | | |
| c100_r100_dup (2 nodes) | 626 | waiting for nodes | | | | | |

Expected completion: ~Sunday 22 Nov afternoon.

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
| halo441_output_512_dT3e7 | 10^7.5 | running (started 18:21) | 512_lim_vsig (ΔT 1e7, same code) |
| halo441_output_512_dT1e8 | 10^8 | queued | 〃 |
| halo441_output_1024_dT3e7 | 10^7.5 | queued (~10 h on cygnus) | Colibri c100_r100 (grid baseline) |
| halo441_output_1024_dT1e8 | 10^8 | queued (~10 h) | 〃 |

Next if a ΔT converges 512→1024: 2048 to z=2 on Colibri (~12–17 h, 3 nodes), then a full 2048.
Note: a feedback change will shift the 1024 GSD/R_V too → re-check the grid winner afterwards.

---

## 3. 2048³ timestep limiter (crash prevention for 2048 / 4096)

- **Job 480** (plain limiter): reached z=0 in 16.9 h, no crash; 2709 density jumps (max ×24 000).
  z=0 copied to `halo441_output_2048_br06_lim`. Central is gas/dust-poor (see §2).
- **Job 552 `cg2048vsig`** (Colibri, 2 nodes): same restart (snap 24) with the stronger signal-velocity
  wake rule. Early: fewer, smaller jumps (102 vs 150; max ×737 vs ×1505), almost none on bins 17–18, same cost.
  At a = 0.796 (z = 0.26) after 3.3 h. **Decide** after it passes a ≈ 0.83 whether to continue to z = 0.
- 512 tests: limiter on/vsig raise central M* by 20–30% vs off (noise ~15%); vsig cost +28% wall at 512.

---

## 4. Paper II manuscript and figures

- Draft: `paper_II_manuscript/paper_II.tex` (figures 1–7 + 8d in place, captions written).
  To fix: cite keys `Schlafly2016`, `Hensley2023`; stale "Discussion"/"Planned Figures" text; old PNG figure near the top.
- Figures: `python3 scripts/cosmicgrain_paper_II_figures.py --run halo441_output_1024_br06_velfix
  --zg-map data/observations/zhang_green/Rv_map_new.h5 --outdir paper_II_figures_v2 --skip-analysis`
- Figures will need regenerating from the grid winner (or a new paper run) once chosen.

---

## Decisions waiting on you
- [ ] Continue job 552 to z=0 after it passes a ≈ 0.83?
- [ ] Push commit d1a36e0 (and the ΔT switch once committed) to GitHub?

## Log
- 11-20 18:21 ΔT tests started on cygnus (512 10^7.5 first).
- 11-20 17:00 C–S grid submitted (610–626); grid A/B/C deleted from Colibri (verified copies on cygnus).
- 11-20 15:05 Job 480 finished (2048 limiter, z=0).
