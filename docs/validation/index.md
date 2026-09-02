# Validation and Numerical Status

## End-to-end validation run

A dust-enabled **Halo 295 / 512** zoom completed from \(z=98.1046\) to \(z=0\). The run exercises star formation, stellar enrichment, explicit PartType6 creation and evolution, domain decomposition, FOF/SUBFIND, and particle exchange.

## Whole-box conservation benchmark

| Quantity | Result |
| --- | ---: |
| Initial baryonic mass | 2.611556729 × 10¹² M☉ |
| Final baryonic mass | 2.611556749 × 10¹² M☉ |
| Fractional baryonic-mass drift | **7.747311 × 10⁻⁹** |
| Fractional dark-matter mass drift | **7.375289 × 10⁻¹⁶** |
| Dust C + silicate / total dust | **1.000000000000** |
| Non-finite masses | **0** |
| Negative masses | **0** |
| Invalid dust carbon fractions | **0** |

At \(z=0\), the box contains 136,764 gas particles, 1,044 star particles, and 8,365 surviving dust particles. The initial gas count was 137,808, so final gas + star particle counts close exactly at 137,808.

The current mass and composition-integrity checks therefore **PASS**.

## Metal bookkeeping

Gas metallicity, stellar metallicity, dust mass, and the tracked gas-element reservoirs evolve smoothly from zero initial enrichment. C, N, O, Ne, Mg, Si, and Fe are audited separately.

`GasZ + StarZ + Dust` is treated as a **bookkeeping trend rather than an automatically conserved scalar**, because formal conservation depends on the precise yield, stellar-remnant, and metallicity conventions used by the enrichment implementation.

## Code validation versus physical convergence

The successful 512 run demonstrates end-to-end numerical integrity; it does **not** establish convergence of galaxy-scale dust predictions.

For Halo 295 at \(z=0\), the central 30 pkpc contains only:

| Component | Particle count |
| --- | ---: |
| Gas | 63 |
| Stars | 152 |
| Dust | 9 |

The surviving dust mass inside 30 pkpc is \(1.743\times10^3\,M_\odot\), about 0.283% of the surviving dust mass inside \(R_{200}\). At this resolution, the central ISM, stellar population, enrichment history, and dust distribution are too sparsely sampled for this result alone to be interpreted as physical dust evacuation.

The convergence program therefore repeats the same diagnostics at 1024³, 2048³, and 4096³-equivalent zoom resolution.

## Current caveats

1. **512³-equivalent zooms are validation runs.** Internal galaxy structure, D/G, D/Z, dust morphology, and dust transport require resolution testing.
2. **LRN dust production is not yet finally calibrated.** The source channel is implemented and produces tagged PartType6 particles, but its absolute yield normalization remains under investigation.
3. **Composition is presently represented by an evolving carbon fraction.** More detailed hybrid/Astrodust-inspired composition models are a future direction.
4. **Transport histories require evolving halo centers.** Raw displacement between `BirthPos` and present-day position contains halo bulk motion and must not be interpreted directly as a dust-outflow distance.

## Regression benchmark

Future changes to particle exchange, feedback, enrichment, or dust mass-transfer routines should rerun the whole-box conservation audit. A material degradation relative to this Halo 295 / 512 benchmark should be investigated before accepting the change.
