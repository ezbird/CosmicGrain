# CG10 dust removal source audit — 2026-10-13

Source inspected: the supplied 3,886-line dust.cc paste. It has not been
established that this exact source was compiled into the completed run.
This is an audit and a read-only log scanner, not a simulation patch.

## Findings relevant to the missing IDs

1. **Physics-driven removal explicitly erases IDs.**
   destroy_dust_particle_to_gas() returns carbon and silicate mass to an
   associated gas particle, increments gas mass and MassMetallicity, then
   converts dust to type 3, sets its ID to zero and mass to 1e-30 code units.
   Subsequent compaction removes the marked particle. This is consistent
   with missing IDs and increased gas-phase metals, but does not identify
   which removal mechanism actually ran.

2. **The main physical cutoff is grain radius, not superparticle mass.**
   Thermal erosion removes the entire current ensemble when its predicted
   radius is <= DustMinGrainSize. Shock erosion does so when radius is below
   that floor. Both return the full pre-update dust mass once that condition
   triggers. Thus a particle carrying substantial represented mass can
   disappear when its representative grains become too small. The 1e-30
   mass sentinel is extremely small and is not a plausible ordinary sampling
   mass cutoff. Shattering reduces representative radius without reducing
   ensemble mass, so it can indirectly make particles susceptible to the
   subsequent size cutoff. This is a model choice requiring evaluation,
   not automatically a programming error.

3. **Corruption paths can discard real mass without returning it.**
   cleanup_invalid_dust_particles() and the growth corruption guard set
   particle mass to the sentinel and clear the ID/type without gas return.
   Invalid radius/composition/temperature/position/velocity can trigger
   cleanup even when the particle mass is finite and substantial. The source
   explicitly labels zeroed DustP metadata with surviving P mass as a possible
   domain-exchange problem. That label is a diagnostic hypothesis; it does
   not prove the exchange code failed. These printf warnings are unconditional
   and can appear with DustDebugLevel=0. The growth corruption guard also
   prints unconditionally. Runtime incidence has not yet been checked.

4. **The sputtering density fallback is malformed.**
   The source calculates dimensionless rho_minus27=rho_cgs/1e-27, but its
   fallback is 1e-4 / 1e-27 * PROTONMASS / 1e-27, approximately 1.67e26.
   If the intended fallback is n~1e-4 cm^-3 times PROTONMASS, a single
   normalization gives ~0.167, not 1.67e26. The existing expression can force
   the sputtering timescale down to its 1 Myr minimum when density is zero,
   negative or nonfinite. This is a conditional source defect; positive
   finite densities do not use this fallback. No evidence yet establishes
   that it triggered in the completed run.

5. **Coagulation does not merge particle IDs in this file.**
   The implemented coagulation path changes radius and composition while
   retaining total particle mass and ID. Shattering also retains mass and
   ID and creates no new particles. Therefore this module's coagulation is
   not an explanation for missing IDs via particle merging. Other modules
   could still alter particles and have not been audited here.

6. **Runtime parameter changes affect more than direct timescales.**
   dust_clumping_factor() uses fractions of CritPhysDensity for non-star-
   forming gas. With nH=0.25 cm^-3, clumping is 10 for CritPhysDensity=0.3
   and 3 for CritPhysDensity=1.0. At otherwise equal states in this example,
   the HK11 density scaling and growth calibrations 0.3 versus 0.4 imply
   a nominal 2048 growth timescale ~4.44 times the 1024 value, before growth
   caps and other gates. This is an illustrative code-level consequence,
   not a measurement of actual gas conditions. The shorter shattering
   calibration in 2048 can also affect the size distribution and subsequent
   sputtering vulnerability. This comparison cannot isolate resolution.

## Important diagnostic limitations

- DustDebugLevel=0 suppresses most physics statistics and destruction
  audit text. Absence of those records does not mean no destruction.
- Cleanup detailed ID reports are capped at 50 per cleanup call; matches
  against missing cohorts are lower bounds.
- NDustDestroyedByBadGasIndex is incremented for a rejected association
  followed by return 0; despite its name and inclusion in printed destruction
  totals, that branch does not destroy the particle. Do not treat it as a
  removal count.
- Source-level operation order permits later functions to be called after
  thermal destruction; the call sites and helper guards need broader review
  before altering this. No causal conclusion is drawn here.
- No hydrodynamics, domain exchange, feedback call sites, event logger,
  executable provenance or runtime logs have yet been inspected.

## Next read-only check

Extract the archive from the gadget4 root, then scan the existing stdout log:

```bash
tar -xzf cg10_removal_log_audit_20261013_v1.tar.gz

python3 scripts/cosmicgrain_cg10_removal_log_audit_v1.py \
  output_CG10_2048_mingashsml0.0.log \
  --cache-dir analysis_products/cg10_dust_transition_v1/cache \
  --outdir analysis_products/cg10_removal_log_audit_v1
```

Replace the log filename if needed. Multiple nonoverlapping restart logs
can be supplied as additional positional paths. Gzip logs are supported.
Duplicate/overlapping logs can duplicate warning totals. The scanner reads
no simulation snapshots and modifies no simulation files. It uses existing
transition caches to match cleanup/corruption IDs directly to the particles
that disappeared from the central cohorts. It handles signed printf ID
representations without losing uint64 precision.

Send removal_log_summary.json and warning_examples.txt. If cache files are
unavailable, the warning scan still works; direct cohort matches are omitted.
If there are no corruption warnings in complete logs from the matching source
version, the physical size-cutoff route becomes a stronger working explanation;
mechanism-specific event logs or source-consistent gas/size diagnostics would
still be needed to distinguish shocks, thermal erosion and astration.

Validation: synthetic cache/log test passed exact IDs above 2^63, signed ID
representation, missing-cohort selection, distinct-ID deduplication across
cleanup/growth reports and cleanup summary totals. The real run log is not
available locally and has not been scanned.
