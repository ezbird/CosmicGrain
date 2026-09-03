# Compile-Time Configuration

The following flags define the standard seven-type cosmological zoom build.
Exact production configurations should be archived with every run.

| Flag | Purpose |
| --- | --- |
| `NTYPES=7` | Enables the seventh particle type used by dust |
| `DUST` | Compiles CosmicGrain dust creation and evolution |
| `COOLING` | Enables radiative gas cooling |
| `STARFORMATION` | Enables star formation |
| `FEEDBACK` | Enables stellar feedback and delayed enrichment |
| `METALS` | Enables metallicity/element tracking |
| `IDS_64BIT` | Prevents particle-ID overflow as dust particles are spawned |
| `PERIODIC` | Periodic cosmological volume |
| `DOUBLEPRECISION=1` | Double-precision core calculations |
| `ADAPTIVE_HYDRO_SOFTENING` | Adaptive gas gravitational softening |
| `WENDLAND_C4` | Adopted SPH kernel |
| `PMGRID` / `HRPMGRID` | Particle-mesh settings for the zoom calculation |
| `PM_ZOOM_OPTIMIZED` | Optimized zoom PM treatment |
| `TREEPM_NOTIMESPLIT` | Adopted TreePM time integration option |

Debug-only feedback limiters such as `FEEDBACK_LIMIT_DULOG` and
`FEEDBACK_T_CAP` are diagnostics, not default production physics.

After compilation, inspect the startup configuration and `[DUST_FLAGS]`
summary. A successful compilation does not guarantee that runtime dust flags
or parameter names are correct.
