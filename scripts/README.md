# CosmicGrain Analysis Scripts

This folder contains the full set of analysis and visualization scripts used
during CosmicGrain's development. Most of these were exploratory or one-off
tools used while debugging the code or selecting the target halo (Halo 569)
from the parent box. **The scripts below are the ones that directly produced
the figures in the CosmicGrain methods paper**, and are the best starting
point if you want to reproduce a published result or adapt the analysis
pipeline for your own runs.

## Shared utilities (required by most scripts below)

- **`halo_utils.py`** — core shared library for identifying Halo 569 and
  computing R200c/M200c via spherical overdensity. Primary API:
  ```python
  ref  = get_halo569_reference(output_dir)
  halo = get_halo569(groups_dir, snap_num, ref)
  ```
  See the module docstring for full unit conventions (comoving vs. physical
  kpc, code mass units, etc.) — read this before adapting any script below to
  a new halo or run.

- **`get_target_halo_particles.py`** — defines `load_target_halo()`, used by
  the age-coded histogram script to extract all particle types within a halo.

## Figure-producing scripts

| Script | Produces | Usage |
|---|---|---|
| **`visualize_snapshots.py`** | Dust surface density projections across the physics ladder (Fig. showing all S0–S10 runs) | `python visualize_snapshots.py <snapshot_path>` — see `--help` for panel/zoom options |
| **`run_radial_evolution.py`** | Cumulative radial dust distribution at multiple redshifts | `python run_radial_evolution.py ../S10_output_1024/ --redshifts 0 0.5 1 2 3 --rmax-factor 1.0 --outdir radial_evolution --summary dust_radial_evolution.pdf` |
| **`plot_mdust_mstar_all_halos.py`** | Dust mass vs. stellar mass evolutionary track, vs. observations/SIMBA | `python plot_mdust_mstar_all_halos.py ../S10_output_1024/ --simba-download --simba-dir ./simba/ --output mdust_mstar_S10_1024.png` |
| **`plot_dust_histograms_agecoded.py`** | Age-coded histograms of grain radius, mass, velocity, age | `python plot_dust_histograms_agecoded.py --catalog ../groups_049/fof_subhalo_tab_049.0.hdf5 --snapshot ../snapdir_049/snapshot_049 --out dust_histograms_agecoded.png --rmax 300` |
| **`plot_gsd_comparison.py`** | Grain-size distribution vs. MRN/Weingartner & Draine/THEMIS models | `python plot_gsd_comparison.py ../S10_output_1024/` (add `--snap 049`, `--r-ism 20`, `--n-bins 45` etc. as needed) |
| **`plot_dust_evolution.py`** | 5-panel evolution figure: M_dust/M_star, SFR, dM_dust/dt, D/G, D/Z vs. redshift | `python plot_dust_evolution.py ../S10_output_1024/` (or pass multiple run dirs + `--labels` for a convergence comparison) |
| **`plot_radial_dgr.py`** | Radial D/G and D/Z profiles across the physics ladder at z=0 | `python plot_radial_dgr.py --res 1024` (add `--runs S0 S4 S10` or `--r-max 50` as needed) |
| **`plot_dz_vs_metallicity.py`** | Galaxy-integrated D/Z vs. gas-phase metallicity, vs. observations/other sims | `python plot_dz_vs_metallicity.py --res 1024` (or `--res 512 1024 2048` for a convergence-mode plot) |

All scripts expect to be run from inside this `scripts/` directory (so that
`halo_utils.py` and `sleek.mplstyle` resolve via relative import), and all
support `--help` for the full list of options.

## Unit conventions (apply across all scripts above)

- Positions in snapshots/catalogs: comoving kpc/h → physical kpc via `* a / h`
- Masses: `1e10 Msun/h` (Gadget code units)
- `HubbleParam` (`h`) is read from `f["Parameters"].attrs`, **not**
  `f["Header"].attrs`
- `GrainRadius` is already in nm in the HDF5 output (unit conversion applied
  on write) — do not apply an additional conversion factor

## Everything else in this folder

The remaining scripts were used during development: halo selection from the
parent box, zoom-region setup and verification, one-off debugging/health
checks, and earlier iterations of the plots above that were superseded as the
analysis pipeline matured. They aren't required to reproduce the paper's
results and are kept here for development history rather than as a
maintained, documented interface. If you're looking for something specific
and it isn't in the table above, it's likely one of these — feel free to open
an issue if you'd like a particular one documented or cleaned up.
