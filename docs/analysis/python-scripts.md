# Python Analysis and Validation Scripts

The following scripts are the current entry points. Always consult
`python3 SCRIPT --help` because options evolve with the analysis.

## Shared infrastructure

| Script | Purpose |
| --- | --- |
| `halo_utils.py` | Canonical halo definitions, target matching, spherical-overdensity masses, and main-progenitor tracking |
| `cosmicgrain.mplstyle` | Shared figure styling |

## Halo and global diagnostics

| Script | Purpose |
| --- | --- |
| `cosmicgrain_halo_census.py` | Per-snapshot halo and baryonic census |
| `cosmicgrain_zoom_postrun_audit_v2.py` | Evolved zoom/contamination audit |
| `cosmicgrain_global_conservation_audit.py` | Whole-box mass and element closure |
| `cosmicgrain_dust_radial_diagnostic.py` | Dust radial inventory |
| `dust_snapshot_summary.py` | Snapshot-wide dust and source summary |

## Dust evolution

| Script | Purpose |
| --- | --- |
| `compute_growth_stats.py` | Grain-growth diagnostics |
| `compute_coag_stats.py` | Coagulation diagnostics |
| `compute_shatter_stats.py` | Shattering diagnostics |
| `compute_shock_stats.py` | Shock-destruction diagnostics |
| `compute_astration_mass.py` | Dust astration budget |
| `compute_sfr_dust_peak_delay.py` | SFR-to-dust timing |
| `check_mass_conservation.py` | Initial/final conservation checks |

## Figures and synthetic observations

| Script | Purpose |
| --- | --- |
| `plot_dust_evolution.py` | Dust abundance and rate evolution |
| `plot_dz_vs_metallicity.py` | D/Z versus metallicity |
| `plot_mdust_mstar_all_halos.py` | Dust mass versus stellar mass |
| `plot_radial_dgr.py` | Radial D/G and D/Z |
| `plot_radial_evolution.py` | Redshift evolution of radial structure |
| `plot_gsd_comparison.py` | Grain-size distribution comparisons |
| `plot_composition_evolution.py` | Carbon/silicate evolution |
| `plot_composition_cumulative.py` | Cumulative composition distributions |
| `plot_halo_projection_full_ladder.py` | Physics-ladder projections |
| `export_skirt_inputs_MRN.py` | Fixed-MRN SKIRT control export |
| `export_skirt_inputs_sizebins.py` | Particle-size/composition SKIRT export |
| `generate_ski.py` | SKIRT configuration generation |
| `plot_skirt_sed.py` | SKIRT SED plotting |

Scripts labeled `_old` are archival comparisons rather than maintained entry
points.
