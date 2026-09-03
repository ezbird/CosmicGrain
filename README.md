# CosmicGrain

A live dust physics extension for GADGET-4, implementing explicit on-the-fly dust in cosmological simulations. Notably, the dust superparticles (PartType6) have individual masses, grain sizes, compositions, positions, and velocities.

CosmicGrain follows the creation, evolution, transport, growth, and destruction of dust grains self-consistently alongside gas, stars, and dark matter. The project is designed to study the origin and evolution of cosmic dust from high redshift to the present day, with particular emphasis on dusty galaxies observed by JWST and ALMA.

For more information, visit: https://cosmicgrain.space

---

## Physics Modules

CosmicGrain models the full dust lifecycle through:

* **Dust creation** from SNII, AGB stars, and luminous red novae (LRNe)
* **Astration** during star formation
* **Gas-dust drag**
* **Thermal sputtering** in hot gas
* **SN shock destruction**
* **Grain growth**
* **Subgrid clumping enhancement**
* **Coagulation**
* **Shattering**
* **Radiation pressure**
* **Evolving carbon/silicate grain composition**
* **Element-resolved gas enrichment** (C, N, O, Ne, Mg, Si, Fe)

---

## Physics Ladder

Runs S0–S10 progressively activate additional dust processes:

| Run | Physics Added                   |
| --- | ------------------------------- |
| S0  | Dust creation                   |
| S1  | Dust cooling                    |
| S2  | Gas-dust drag                   |
| S3  | Astration                       |
| S4  | Thermal sputtering              |
| S5  | ISM grain growth                |
| S6  | Subgrid clumping                |
| S7  | SN shock destruction            |
| S8  | Coagulation                     |
| S9  | Shattering                      |
| S10 | Radiation pressure (full model) |

---

## Current Zoom Program

The current CosmicGrain program follows 12 isolated targets selected from a
50 \(h^{-1}\) Mpc dark-matter-only parent volume. The sample spans dwarf
through super-Milky-Way halo masses. Halo 569 remains the historical
single-halo development target.

| Property   | Value                  |
| ---------- | ---------------------- |
| Parent Box | 50 Mpc (comoving)      |
| Halos      | 12 selected targets    |
| Resolution | 512³–4096³ equivalent  |
| Cosmology  | Planck 2015            |
| h          | 0.6732                 |
| Ωm         | 0.3158                 |
| Ωb         | 0.0494                 |

---

## Validation and Numerical Status

A complete dust-enabled **Halo 295 / 512** zoom validation run has evolved from **z = 98.10 to z = 0**, exercising star formation, enrichment, live PartType6 dust, domain decomposition, and FOF/SUBFIND.

Whole-box bookkeeping gives:

| Test | Result |
| --- | ---: |
| Fractional baryonic-mass drift | 7.747311 × 10⁻⁹ |
| Fractional dark-matter mass drift | 7.375289 × 10⁻¹⁶ |
| Dust carbon + silicate mass closure | 1.000000000000 |
| Non-finite particle masses | 0 |
| Negative particle masses | 0 |
| Invalid dust carbon fractions | 0 |

This is a **code-validation benchmark, not a physical-convergence claim**. The 512³-equivalent zoom level is used primarily for inexpensive validation; quantitative internal galaxy structure, dust morphology, D/G, D/Z, and dust transport are being convergence-tested at higher resolution.

The LRN dust-production channel is implemented and operational, but its absolute yield normalization remains under active calibration.

See `docs/validation/` for the detailed validation record.

The separate MUSIC2 initial-condition audit also reports **48 PASS, 0 WARN,
0 FAIL** for the 12-halo × four-resolution production suite.

---

## Compilation

CosmicGrain inherits the build system of GADGET-4.

Select a build target in:

```text
Makefile.systype
```

For example:

```text
SYSTYPE="stampede"
```

Then compile:

```bash
make -j
```

Required libraries typically include:

* MPI
* HDF5
* FFTW3
* GSL

---

## Included Initial Conditions

Example initial conditions are provided in:

```text
ICs/
```

Included datasets:

* 50 Mpc parent volume
* Halo 569 zoom-in initial conditions (512^3-resolution)

The 48 production zoom initial conditions are omitted from the repository
because of file size. Their MUSIC2 configurations, Lagrangian-region inputs,
and validation records should be retained with the project data.

---

## Repository Structure

```text
buildsystem/      Compiler and machine configurations
configs/          Example compile-time configuration files
ICs/              Example initial conditions and IC generation tools
scripts/          Analysis and utility scripts
src/              CosmicGrain source code
```

---

## Running

Example:

```bash
mpirun -np 16 ./CosmicGrain parameterfile.txt
```

Example compile-time configurations are available in:

```text
configs/
```

---

## Scientific Goals

Current applications include:

* Evolution of the cosmic dust budget
* Dust-to-gas and dust-to-metal ratios
* Dust in galactic halos and the CGM
* Dust production from SNII, AGB stars, and LRNe
* Synthetic observations for JWST and ALMA comparisons
* Dusty high-redshift galaxies (z ≳ 6)

---

## Citation

If you use CosmicGrain in scientific work, please cite:

* The CosmicGrain code paper (in preparation)
* Springel et al. (2021), GADGET-4
