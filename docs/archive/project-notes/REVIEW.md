# CosmicGrain stellar mass-return audit

Reviewed 2026-10-15. Scope: the nine files in the supplied stellar_mass_return_review.tar.gz.
This review does not modify simulation source, snapshots, parameters, or figures.

## Finding

The supplied delayed-feedback implementation transfers metal ejecta and associated
stellar-condensed dust, but does not propagate the table's total returned-mass
budget through the SSP integration and into its star/gas mass updates.

The finding is established in these supplied modules. A whole-repository search
and production-executable provenance check would be needed to exclude another
channel elsewhere and establish which historical runs used precisely this source.
The observed CG10 stellar-current/birth ratio independently supports this finding.

## Source trace

1. SNIIYieldEntry/HypernovaYieldEntry retain M_ejecta; the loader reads and validates it,
   and interpolation carries it. The raw tables define M_ejecta = M_final - M_cut.
2. SNIIIMFYield in snii_yields.h carries elemental yields, Z_per_Mstar and energy,
   but no total-ejecta fraction. build_imf_yield_table integrates metals and energy,
   not M_ejecta. The feedback caller therefore has no SSP total-return budget.
3. AGBYieldEntry retains M_lost, but AGBIMFYield similarly carries only element/metal
   fractions. build_imf_yield_table does not integrate M_lost. Although an
   individual-model get_mass_lost helper exists, feedback.cc does not call it.
4. feedback.cc deposit_energy_stochastic adds dMZ, not total gas ejecta, to gas mass.
   SNII and AGB each debit MZ_total_code from the star after deposition succeeds.
   The metal budget is partitioned into gas-phase metals and dust; the bulk
   nonmetal mass is not passed through that transaction.
5. starformation.cc records birth mass at star creation and does not contain a
   separate bulk-return step. There is no FactorSN-based particle-mass subtraction
   in the supplied star-formation file. The EOS implementation is outside this archive.

Useful source locations in the uploaded version:
- snii_yields.h: SNIIIMFYield; snii_yields.cc: build_imf_yield_table (starts at 720)
- agb_yields.h: AGBIMFYield; agb_yields.cc: build_imf_yield_table (starts at 155)
- feedback.cc: gas mass update 1430-1433; SN stellar debit 2432-2444;
  AGB stellar debit 2839-2851; AGB event scheduling 2616 onward.

## Quantitative budget

Using the supplied two-branch Kroupa-like IMF normalized over 0.08-100 Msun,
HypernovaFraction=0.5, the exact existing trapezoidal integration convention,
SN progenitors 13-40 Msun and the actual AGB grid 0.8-7 Msun:

| Initial Z | SN metals | SN total ejecta | AGB metals | AGB total loss | Combined metals | Combined total |
|---|---:|---:|---:|---:|---:|---:|
| 0.001 | 1.570% | 8.717% | 0.177% | 23.390% | 1.747% | 32.108% |
| 0.004 | 1.267% | 8.293% | 0.246% | 23.828% | 1.513% | 32.121% |
| 0.020 | 1.227% | 7.667% | 0.580% | 23.364% | 1.807% | 31.032% |

All percentages are fractions of initial SSP mass. The last column sums the
available total-loss columns after ALL tabulated phases, regardless of age.
It is not the physically returned fraction of CG10 at z=0. In particular some
low-mass AGB lifetimes exceed the age of the Universe; the solar-Z 0.8 Msun
row ends at about 27.3 Gyr. The tables also do not cover SN progenitors 8-13 or
40-100 Msun in the existing integration. This is not a complete universal return model.

User's CG10 snapshot test: M_current/M_birth = 0.9828536243960632, so the net
reduction is 1.71463756%. This agrees in scale with the implemented metal-only
budget, though it is not an exact closure test without stellar metallicities,
ages, event flags, dust astration and all transfer histories.

For illustration only, keeping birth populations fixed and replacing completed
metal-only loss by completed tabulated total loss gives
log10[(1-R_total)/(1-R_metals)] = about -0.15 to -0.16 dex for these examples.
This is smaller than CG10's ~0.39 dex UM excess. It is NOT a valid post-processing
correction or a prediction of a self-consistent rerun: recycling changes gas supply,
metal dilution, cooling, later star formation, feedback and dust processing.

## Timing issue: do not implement a one-line replacement

The existing AGB feedback uses one IMF-integrated event once age >= MIN_AGB_TIME
and then sets FeedbackFlag bit 1. The threshold value is defined outside the
supplied files. The integration includes every AGB model, whose tabulated
lifetimes vary strongly with mass and metallicity. It does not use those lifetimes
to gate the per-mass contributions when the event fires. Adding the whole
integrated M_lost budget to this one event would return long-lived stars' envelopes
too early. Timing of the existing AGB metal/dust release is also a model approximation.

## Requirements for a future implementation

- Carry distinct total-mass, total-metal, elemental and dust budgets per initial SSP mass.
- Define cumulative age- and metallicity-dependent returns, especially for AGB stars,
  with only increments released on each update. Verify net versus total elemental
  yield conventions rather than adding initial metals twice.
- Specify progenitor coverage/remnants for gaps instead of extrapolating implicitly.
- Debit total ejecta once; deposit total-minus-condensed-dust into gas and the dust
  component into PartType6. Normalize SSP yields to immutable birth mass.
- Evolve gas composition and mass consistently; check momentum and thermal-energy
  bookkeeping when gas mass changes, including the code's thermodynamic variables.
- Retry unsuccessful events without double deposition; test complete transfers,
  inactive stars, MPI boundaries, and restart/snapshot state.
- Test isolated SSPs and mass/metal/element closure before a controlled pilot zoom.
  Do not assume the present FeedbackFlag scheme can represent continuous AGB return.

No feedback patch or production-run change is included in this audit.

## Reproduction and validation

From the gadget4 root, using the diagnostic script in this package:

    python3 audit_stellar_return_budget.py --data-dir data

This writes analysis_products/stellar_return_audit/ssp_return_budgets.csv and
budget_metadata.json. It uses numpy only and reads the three yield tables.
The package contains the complete results obtained from the supplied tables.

An independent C++ harness compiled the UNMODIFIED supplied snii_yields.cc and
agb_yields.cc with an empty local gadgetconfig.h. The Python metal integrals
matched the actual C++ get_imf_yields outputs at Z=0, .001, .004, .01, .02, .03
with relative tolerance 1e-12. This checks the reproduced metal integration,
not a hydrodynamical simulation or a future total-return implementation.
