# Validation and Numerical Status

Validation is divided into three distinct layers:

1. **Initial-condition integrity** — the complete 12-halo, 48-file MUSIC2
   suite is tested before simulation; see [MUSIC2 IC-suite
   validation](ic-suite.md).
2. **End-to-end code integrity** — a dust-enabled zoom is evolved from the
   initial conditions to \(z=0\), with whole-box conservation and particle
   fields audited.
3. **Physical convergence** — scientific observables are compared across
   resolution levels. A successful code or IC test is not a convergence claim.

## End-to-end validation run

The Paper II run (**Halo 441 / 1024**, v1.2.0 code, `DustEnableClumping 2`) was evolved from
\(z=97.88\) to \(z=0\). It exercises star formation, stellar enrichment, PartType6
creation and evolution, domain decomposition, FOF/SUBFIND and particle exchange.

## Whole-box conservation benchmark

From `scripts/check_mass_conservation.py` and a direct read of the first and last snapshots:

| Quantity | Result |
| --- | ---: |
| Baryonic mass (gas + stars + dust), first → last | 2.5263 × 10¹² M☉ → 2.5263 × 10¹² M☉ |
| Fractional baryonic-mass drift | **1.73 × 10⁻⁸** |
| Fractional total-mass drift | **2.7 × 10⁻¹²** |
| Dark-matter mass drift | **0** |
| Largest snapshot-to-snapshot baryonic change | 2.6 × 10⁻⁷ M☉ (z ≈ 2) |
| Non-finite or negative masses | **0** |
| Dust `CarbonMassFraction` outside [0, 1] or non-finite | **0** |

Particle counts close exactly: 1,066,464 initial gas particles = 909,172 gas + 157,292 stars at
\(z=0\). The box also holds 1,019,472 dust particles at \(z=0\). The record is in
[`paperII_1024_global_conservation.txt`](paperII_1024_global_conservation.txt).

The mass and composition-integrity checks **PASS**.

## Metal bookkeeping

Gas metallicity, stellar metallicity, dust mass, and the tracked gas-element reservoirs evolve smoothly from zero initial enrichment. C, N, O, Ne, Mg, Si, and Fe are audited separately.

`GasZ + StarZ + Dust` is treated as a **bookkeeping trend rather than an automatically conserved scalar**, because formal conservation depends on the precise yield, stellar-remnant, and metallicity conventions used by the enrichment implementation.

## Code validation versus physical convergence

Passing this audit shows numerical integrity, not convergence of galaxy-scale dust predictions.
Convergence is tested separately by repeating the science diagnostics at 512³, 1024³ and 2048³
([Resolution](resolution.md)).

## Recent regression fixes

The current validation baseline includes several fixes that materially changed
what should be considered a trustworthy production run:

- the legacy instantaneous enrichment contribution was removed from the active
  enrichment path in favor of the delayed stellar-feedback treatment;
- stellar metallicity bookkeeping was corrected to use the intended hydrogen
  mass fraction convention;
- dust state and cadence bookkeeping across restarts was repaired and is now
  explicitly serialized/tested; and
- snapshot continuation includes dust-ID and particle-state safeguards before
  the initial domain decomposition, and stellar ages are now correct after a
  snapshot restart.

v1.2.0 adds further fixes that change results:

- the Hubble parameter \(h\) is included in dust time, mass, and energy unit
  conversions, and dust velocities are converted between physical and
  comoving units;
- feedback and dust neighbour searches are independent of the domain
  decomposition, so neighbour sets no longer depend on the number of MPI
  ranks; and
- an SPH timestep limiter with wake-up, an SN reheating guard, an AGB
  fast-donor guard, and dust/star velocity-runaway guards remove the
  high-resolution blow-ups seen at 2048³.

Because enrichment changes can strongly alter the dust budget even when gas and
stellar masses change modestly, results produced before these fixes should not
be mixed quantitatively with the current production baseline without a targeted
comparison.

See [Resolution and gas smoothing length](resolution.md) for the separate
`MinGasHsml` convergence tests.

## Current caveats

1. **512³-equivalent zooms are validation runs.** Internal galaxy structure, D/G, D/Z, dust morphology, and dust transport require resolution testing.
2. **LRN dust production is not yet finally calibrated.** The source channel is implemented and produces tagged PartType6 particles, but its absolute yield normalization remains under investigation.
3. **Composition is presently represented by an evolving carbon fraction.** More detailed hybrid/Astrodust-inspired composition models are a future direction.
4. **Transport histories require evolving halo centers.** Raw displacement between `BirthPos` and present-day position contains halo bulk motion and must not be interpreted directly as a dust-outflow distance.

## Regression benchmark

Future changes to particle exchange, feedback, enrichment, or dust mass-transfer routines should rerun the whole-box conservation audit. A material degradation relative to this Halo 441 / 1024 benchmark should be investigated before accepting the change.
