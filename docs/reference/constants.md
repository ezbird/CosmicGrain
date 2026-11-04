# Source-Level Constants

Most user-facing dust controls are runtime parameters, but several numerical or physical choices remain hardcoded in the current implementation. This page records the most important ones found in the audited source tree; it is not a replacement for reading the implementation when changing the model.

## Dust bookkeeping

| Source constant / value | Current value | Role |
| --- | ---: | --- |
| `MIN_DUST_PARTICLE_MASS` | `1e-10` | General minimum-mass scale used by dust bookkeeping paths |
| `DUST_MASS_TO_DESTROY` | `1e-30` | Sentinel mass used when marking a dust particle for removal |
| `DUST_SFR_EPS` | `1e-14` | Numerical SFR threshold used to identify star-forming gas in dust processing |
| Silicate O/Mg/Si/Fe mass fractions | 0.69876708 / 0.07680773 / 0.08097301 / 0.14345218 | Adopted silicate stoichiometric partition |

The stellar-creation routine also uses a local `1e-15` minimum representative particle-mass scale when deciding how many requested superparticles can represent a newly created dust mass.

## Grain processing

The current source includes several hardcoded model choices, including:

- thermal-sputtering characteristic temperature `2e6 K` and exponent `2.5`;
- shock Sedov coefficient `ξ = 1.033`;
- radiation-pressure transition grain size `a0 = 35 nm`;
- grain material density `2.4 g cm⁻³` in drag/radiation calculations;
- grain-growth effective temperature `20 K`, sticking coefficient `0.3`, and reference metallicity mass fraction `0.02`;
- dust-cooling emissivity index `β = 2`, reference temperature `100 K`, reference radius `0.1 µm`, and reference absorption efficiency `1.3e-4`.

These values are implementation choices rather than runtime parameters. If one becomes a quantity that users are expected to vary systematically, it should preferably be promoted to a named runtime parameter and documented there.

The shattering fragment slope and maximum fragment fraction are runtime
parameters, not hardcoded constants. Their defaults are \(q=3.3\) and 0.80;
see [Runtime Parameters](parameters.md).

## Stellar feedback

The audited feedback implementation uses four delayed SNII time bins (`SNII_TIME_BINS = 4`) and a CCSN number normalization of `0.011` per solar mass formed for the LRN-rate calculation. Feedback kernel searches and their safety limits are also controlled by source-level constants.

## Documentation rule

Hardcoded values are particularly easy for documentation to become stale against. When modifying one of these constants, update this page and the corresponding Physics page in the same change.
