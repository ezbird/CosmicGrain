# CosmicGrain

A live dust physics extension for GADGET-4, implementing explicit N-body dust superparticles (PartType6) in cosmological simulations.

CosmicGrain follows the creation, evolution, transport, growth, and destruction of dust grains self-consistently alongside gas, stars, and dark matter. The project is designed to study the origin and evolution of cosmic dust from high redshift to the present day, with particular emphasis on dusty galaxies observed by JWST and ALMA.

For more information, visit: https://cosmicgrain.space

---

## Physics Modules

CosmicGrain models the full dust lifecycle through:

* **Dust creation** from SNII and AGB stars
* **Gas-dust drag** (Epstein regime)
* **Thermal sputtering** in hot gas
* **SN shock destruction**
* **ISM grain growth**
* **Subgrid clumping enhancement**
* **Coagulation**
* **Shattering**
* **Radiation pressure**
* **Astration** during star formation

Dust is represented by live superparticles with individual masses, grain sizes, compositions, positions, and velocities.

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

## Example Simulation

The primary CosmicGrain science run follows a Milky Way-mass zoom galaxy ("Halo 569") selected from a 50 Mpc cosmological volume.

| Property   | Value                  |
| ---------- | ---------------------- |
| Parent Box | 50 Mpc (comoving)      |
| Halo       | Halo 569               |
| Resolution | Up to 4096³ equivalent |
| Cosmology  | Planck 2015            |
| h          | 0.6732                 |
| Ωm         | 0.3158                 |
| Ωb         | 0.0494                 |

---

## Compilation

CosmicGrain inherits the build system of GADGET-4.

Select a build target in:

```text
Makefile.systype
```

For example:

```text
SYSTYPE="Generic-gcc"
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
* Halo 569 zoom-in initial conditions (512-resolution)

Higher-resolution zoom initial conditions are omitted from the repository because of file size limitations.

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
* Dust production from stellar sources
* Synthetic observations for JWST and ALMA comparisons
* Dusty high-redshift galaxies (z ≳ 6)

---

## Citation

If you use CosmicGrain in scientific work, please cite:

* The CosmicGrain code paper (in preparation)
* Springel et al. (2021), GADGET-4

