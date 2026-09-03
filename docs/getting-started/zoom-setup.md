# Building the CosmicGrain Zoom Suite

This page records the current production workflow for selecting target halos
from the 50 \(h^{-1}\,\mathrm{Mpc}\) dark-matter-only parent run and generating
CosmicGrain-ready MUSIC2 initial conditions. The older Halo 569 workflow was
useful during development, but the current science suite consists of 12 halos
at four resolution levels.

## Current suite

The selected parent-catalog halo IDs are:

```text
295  308  441  859  1481  1534
3352  3879  3886  5834  7723  9235
```

Each halo has ICs at nominal effective resolutions \(512^3\), \(1024^3\),
\(2048^3\), and \(4096^3\), for 48 files in total. The canonical filename is:

```text
ICs/halo<HALO>/IC_halo<HALO>_zoom_<RES>.hdf5
```

The suite spans dwarf through super-Milky-Way halo masses. Candidate selection
also considers isolation, distance from the periodic box boundary, particle
count, and suitability of the traced Lagrangian region.

## 1. Run and catalog the parent volume

Run the \(512^3\) dark-matter-only parent calculation through \(z=0\), with
FOF and SUBFIND enabled at the output used for selection. The parent and zoom
calculations use:

| Quantity | Value |
| --- | ---: |
| Box size | \(50\,h^{-1}\,\mathrm{Mpc}\) comoving |
| \(\Omega_{\rm m}\) | 0.3158 |
| \(\Omega_\Lambda\) | 0.6842 |
| \(\Omega_{\rm b}\) for baryonic zoom ICs | 0.04936 |
| \(h\) | 0.6732 |
| \(n_s\) | 0.965 |
| \(\sigma_8\) | 0.811 |
| Initial scale factor | 0.01 (\(z=99\)) |

The parent run itself has `OmegaBaryon 0.0` because it contains only dark
matter. MUSIC2 must use the physical baryon density when splitting the
high-resolution matter into gas and dark matter.

## 2. Build and rank the halo census

From `scripts/`:

```bash
python3 parent_halo_census.py
python3 select_zoom_halo_candidates_v2.py
```

The census and ranking products are:

```text
parent_50Mpc_halo_census.csv
zoom_halo_candidates_all_ranked.csv
zoom_halo_candidates_selected.csv
zoom_halo_candidates_selected.txt
```

Selection should not be based on mass alone. Inspect the isolation metrics,
edge distance, particle count, and neighboring massive halos before accepting
a target.

## 3. Prepare and trace the Lagrangian regions

Prepare the selected parent-halo particle sets:

```bash
python3 prepare_lagrangian_particle_sets_v2.py
```

Then trace those particle IDs to the parent initial conditions:

```bash
python3 trace_lagrangian_regions_to_initial.py
```

The corresponding utilities and outputs live under:

```text
scripts/lagrangian_particle_sets/
scripts/lagrangian_regions_initial/
scripts/lagrangian_regions_minimal/
```

Before generating expensive ICs, inspect the traced extents:

```bash
python3 diagnose_lagrangian_region_extents.py
```

## 4. Generate the MUSIC2 configuration files

Generate all halo/resolution configurations with:

```bash
python3 make_music2_zoom_config_v2.py
```

Review at least one configuration from every resolution level. In particular,
confirm the output filename, region file, cosmology, transfer-function input,
box length, random seed/noise settings, refinement levels, and baryon split.

## 5. Generate and post-process all ICs

From the IC directory, run the canonical suite driver:

```bash
cd ~/gadget4/ICs
bash run_music2_suite.sh
```

For every configuration, the driver:

1. runs MUSIC2;
2. writes the canonical HDF5 IC;
3. extends the GADGET header arrays from six to seven particle types;
4. creates the intentionally empty `PartType6` datasets required by
   CosmicGrain; and
5. performs the per-file readiness check.

The empty dust group is deliberate: dust particles are created later by
stellar feedback, not placed in the initial conditions.

## 6. Validate the complete suite

Run the independent suite validator before deleting MUSIC2 intermediates or
starting production simulations:

```bash
python3 ~/gadget4/scripts/validate_music2_ic_suite.py \
    --ic-root ~/gadget4/ICs
```

For a long unattended run:

```bash
mkdir -p ~/gadget4/ICs/MUSIC2_logs
nohup python3 ~/gadget4/scripts/validate_music2_ic_suite.py \
    --ic-root ~/gadget4/ICs \
    > ~/gadget4/ICs/MUSIC2_logs/validate_ic_suite.log 2>&1 &
```

The detailed checklist, result interpretation, and September 2026 suite
result are recorded in [IC-suite validation](../validation/ic-suite.md).

Do not remove `wnoise_*.bin` files until the validator reports:

```text
Files: 48 PASS, 0 WARN, 0 FAIL (48 total)
FINAL STATUS: PASS
```

After a clean result, first review the exact deletion set:

```bash
find ~/gadget4/ICs -type f -name 'wnoise_*.bin' -print
```

Then remove only those MUSIC2 intermediates:

```bash
find ~/gadget4/ICs -type f -name 'wnoise_*.bin' -delete
```

Keep the MUSIC2 configurations, selected-halo tables, Lagrangian-region
inputs, logs, validation CSV, and final HDF5 ICs as the reproducibility record.

## 7. Run a GADGET startup test

IC validation establishes structural and numerical consistency; it does not
prove that a production configuration has enough memory or that the final
zoom remains contamination-free at \(z=0\). Before committing to a long run:

1. start each intended IC with the actual CosmicGrain executable and parameter
   file;
2. confirm that all seven particle types are accepted;
3. confirm domain decomposition and the first force/hydrodynamic steps;
4. inspect memory use and particle balance; and
5. after evolution, measure low-resolution contamination within the target
   halo and its analysis aperture.

The \(4096^3\) IC particle counts vary greatly with Lagrangian-region geometry.
Halo 1534 contains 517,537,024 initial particles, so validity alone does not
make every highest-resolution run equally practical.

## Historical note: Halo 569

Halo 569 was the original single-halo development target. Its earlier
three-script workflow established the direct particle-ID trace-back approach,
but its hardcoded filenames and one-halo selection cuts are not the maintained
interface for the new suite. Use the versioned scripts listed above for new
targets.
