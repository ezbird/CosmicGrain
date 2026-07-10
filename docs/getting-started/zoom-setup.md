# Setting Up the Halo 569 Zoom Simulation

This documents how the CosmicGrain zoom region (Halo 569, in a 50 Mpc parent
box) was originally selected and built. It's based on contemporaneous notes
from November, cross-checked line-by-line against the actual scripts to
confirm they match what was really run.

**The entire pipeline only needed three Python scripts.** A much larger
number of similarly-named scripts exist in this folder (`get_best_halo*.py`,
`find_good_zoom_halo.py`, `trace_ids_to_lagrangian*.py`,
`find_lagrangian_center.py`, `extract_halo_particles.py`, and others) —
those were exploratory alternatives written along the way and are **not**
part of the documented, actually-used pipeline below. See
[Scripts not used](#scripts-not-used-in-the-final-pipeline) at the bottom.

## 1. Run a parent simulation

- 128³ particles, 50 Mpc box, dark matter only.
- Initial conditions generated with MUSIC2: `./MUSIC make_parent.conf`
- Transfer functions (how density fluctuations evolve — encodes baryon
  physics, dark matter behavior, and cosmic evolution turning primordial
  fluctuations into galaxy seeds) generated with CAMB: `./camb mycamb.ini`
- **FOF and SUBFIND must be enabled** in the parent run — the next step reads
  the resulting halo catalog directly.

Config files: `mycamb.ini`, `make_parent.conf` (kept alongside this doc /
in the ICs directory).

## 2. Choose the target halo

```bash
python choose_halo.py
```

Reads the parent run's z=0 SUBFIND catalog
(`fof_subhalo_tab_049.hdf5`) and selects candidates with:
- Mass 2–5 × 10¹² Msun/h
- More than 500 particles (well-resolved)

For each candidate it reports mass, particle count, distance from the box
edge, and distance to the nearest neighboring halo — so you can pick one
that's both Milky Way-mass and reasonably isolated (avoiding both edge
effects and contamination from nearby massive structures). **Halo 569** in
this catalog was the one selected.

> This script has the SUBFIND file path and mass-cut thresholds hardcoded
> at the top — edit those directly for a different box/halo search rather
> than passing CLI arguments.

## 3. Extract the halo's particle IDs

```bash
python extract_zoom_ids.py
```

Reads `Subhalo/SubhaloLenType` and `Subhalo/SubhaloOffsetType` for the
chosen halo (edit `halo_id = 569` at the top of the script for a different
target) from the same SUBFIND catalog, then pulls the corresponding
contiguous block of dark matter particle IDs from the z=0 snapshot
(`PartType1/ParticleIDs`).

Output: `halo_569_particles.txt` — one particle ID per line.

## 4. Trace those particles back to the initial conditions

```bash
python get_zoom_id_positions.py
```

This is the key step. It takes the particle ID list from step 3 and looks
those same IDs up directly in the **parent IC file**
(`ICs/IC_parent_50Mpc_128_music.hdf5`) — since Gadget particle IDs are
conserved from the initial conditions through the entire simulation, this
is a direct ID-matched lookup, not a fitted or approximate trace-back. The
resulting positions are exactly where those particles started out at the
beginning of the simulation — the Lagrangian region that will collapse into
Halo 569 by z=0.

Positions are then normalized to the unit cube `[0, 1]`, which is the
coordinate convention MUSIC2 expects for its `region_point_file` /
particle-based region definitions.

Output: `halo_569_IC_positions_normalized.txt`

> Like `extract_zoom_ids.py`, the halo ID, input file, and parent IC path
> are hardcoded at the top of the script rather than passed as arguments.

## 5. Generate the zoom initial conditions

Add the normalized position file to the zoom MUSIC2 config:

```
# in make_zoom.conf
region_point_file = halo_569_IC_positions_normalized.txt
```

Then run:

```bash
./MUSIC make_zoom.conf
```

This produces the actual zoom initial conditions — high resolution in the
Lagrangian region that collapses into Halo 569, progressively lower
resolution further away — ready to hand to Gadget-4/CosmicGrain.

Config file: `make_zoom.conf` (kept alongside this doc / in the ICs
directory).

---

## Scripts not used in the final pipeline

These were written at various points while exploring halo selection and
zoom-region definition, but the actual pipeline above didn't end up using
them. Kept for reference/history rather than as a maintained interface;
safe to move to an archive subfolder if you want to tidy the scripts
directory further.

**Alternative halo-ranking approaches** (superseded by the simpler
`choose_halo.py` mass+isolation cut): `get_best_halo.py`,
`get_best_halo_newer.py`, `find_good_zoom_halo.py`, `find_zoom_candidates.py`

**Alternative particle-ID / Lagrangian-tracing approaches** (superseded by
the simpler direct-ID-lookup method in `get_zoom_id_positions.py`):
`extract_halo_particles.py`, `extract_halo_ids_for_zoom.py`,
`save_particleids_for_desired_halo.py`, `trace_ids_to_lagrangian.py`,
`trace_ids_to_lagrangian_fixed.py`, `find_lagrangian_center.py`

**IC/zoom quality-check tools** (not part of the core pipeline, but
potentially still useful if you want to sanity-check a future zoom setup —
worth deciding case by case whether to keep): `check_parent_run.py`,
`debug_ic_positions.py`, `check_ic_particles.py`, `ic_slice_viewer.py`,
`ic_visualizer.py`, `check_zoom_quality.py`, `verify_halos.py`,
`test_zoom_viability.py`, `extract_halo_manual.py` (byte-identical
duplicate of `test_zoom_viability.py`)

**Ongoing zoom-run visualization/tracking** (used throughout later
development, not part of initial setup — see the main `scripts/README.md`
for the confirmed paper-figure scripts instead): `studyzoom.py`,
`view_zoom.py`, `zoom_halo_report.py`, `zoom_halo_track.py`,
`find_halo_center.py`, `find_zoom_halos.py`
