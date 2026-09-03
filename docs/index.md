# CosmicGrain

Welcome to the official documentation for **CosmicGrain**, a live dust-physics extension for GADGET-4.

CosmicGrain follows explicit PartType6 dust superparticles alongside gas, stars, and dark matter. The current model includes stellar dust production, drag, astration, sputtering, shock destruction, grain growth, coagulation, shattering, radiation pressure, dust cooling, evolving carbon/silicate composition, and element-resolved enrichment.

## Documentation

- **Getting started** — installation, compilation, first runs, HPC use, and zoom setup
- **Physics** — detailed descriptions of the implemented dust processes
- **Running** — parameter files, outputs, restarts, performance, and troubleshooting
- **Analysis** — snapshots, halo catalogs, radial profiles, and dust diagnostics
- **Developer** — architecture, particle types, source tree, and adding new physics
- **Reference** — compile flags, parameters, constants, units, and equations
- **Validation** — numerical validation benchmarks and convergence limitations

## Current status

A complete dust-enabled cosmological zoom validation run has evolved from z≈98 to z=0 with whole-box baryonic mass conserved to a fractional drift of 7.75×10⁻⁹ and exact carbon+silicate closure of the surviving dust mass.

The current science program expands the original single-halo development run
into a 12-halo sample spanning dwarf through super-Milky-Way systems. Four
MUSIC2 zoom levels per halo provide 48 initial conditions for validation and
convergence work. All 48 passed the September 2026 suite-level IC audit.

The 512³-equivalent zoom level is treated as a **validation resolution rather
than a physically converged galaxy calculation**. Convergence testing proceeds
through 1024³, 2048³, and 4096³-equivalent zooms where computationally
practical.

The LRN dust-production channel is implemented, but its absolute normalization remains under active calibration.
