# Stellar Enrichment and Feedback

CosmicGrain couples dust creation to a time-resolved stellar enrichment and
feedback model. Dust yields cannot be interpreted independently of the gas
ejecta, element reservoirs, and thermal-energy deposition that create the
local conditions in which grains form and subsequently evolve.

## Delayed enrichment

Feedback is delayed rather than deposited when the star particle forms, and
every yield is normalized to the stored `StellarBirthMass`.

- **SNII and hypernovae** (`HypernovaFraction`) release their IMF-integrated
  budget in four equal tranches spanning 9–40 Myr; at most one overdue tranche
  fires per feedback call.
- **AGB** mass return uses IMF-integrated MESA yield tables (`AGBYieldFile`).
  CosmicGrain requires `AGBFeedbackMode 2` (age-dependent return of the total
  ejecta mass, metals plus hydrogen/helium, including AGB dust) and stops
  otherwise. Returns are made every `AGBFeedbackIntervalMyr` of stellar age.
- **LRN** dust is injected once, with the first SNII tranche.
- **SNIa** are not modeled.

See [Dust Creation](dust-creation.md) for how each channel is sampled into
dust superparticles.

## Element-resolved enrichment

The gas tracks C, N, O, Ne, Mg, Si, and Fe separately. SNII/hypernova yields
depend on progenitor mass and metallicity, while AGB yields are integrated
from the adopted stellar-yield grid. Total metallicity and the individual
element reservoirs are updated together.

Dust formation is element limited. The code cannot condense more carbon or
silicate material than is available in the ejecta. Gas-to-dust and
dust-to-gas transfers operate on absolute element masses (silicate as O, Mg,
Si, and Fe in fixed ratios) and then resynchronize gas mass and metallicity.

## Stochastic thermal feedback

SNII/hypernova energy is deposited stochastically rather than divided as a
small temperature increment among every neighbor:

1. the kernel starts from the nearest gas particle's smoothing length and is
   enlarged until it holds at least 16 gas particles (counted over all
   ranks), up to 4 times its initial size or 20 physical kpc;
2. the event owner keeps the 512 nearest gas particles (ties broken by ID);
3. the heating energy of each receiver is computed from its own mass and mean
   molecular weight;
4. receivers are drawn with a counter-based RNG keyed on star ID, step, and
   draw, and heated by \(\Delta T=10^7\) K; energy that cannot be placed is
   banked in the star's `EnergyReservoir` and retried later.

The environment variable `CG_FEEDBACK_DELTA_T` (in K, e.g. `3.16e7`) overrides
\(\Delta T\) for convergence tests. Gas already hotter than \(10^8\) K is never
picked, so repeated hits cannot stack one particle far above \(\Delta T\).

Every task sees every event, and deposits are applied in (star ID, event
kind) order, so neighbour sets and random choices do not depend on the domain
decomposition.

AGB ejecta leave the star at its velocity and mix inelastically into the
receiving gas, with momentum conserved through the star's mass debit. If the
star moves more than 3000 km/s relative to the gas, its velocity is treated as
corrupt: the ejecta are taken at the gas velocity (no dissipative heating) and
the star is reported.

## Relationship to dust creation

The enrichment calculation first determines the ejecta and element budget.
The dust module then condenses the allowed fraction into explicitly tagged
PartType6 particles:

| Source | Current role |
| --- | --- |
| SNII/hypernova | Tranche-resolved dust and metal production |
| AGB | Delayed enrichment and carbon-rich dust packets |
| LRN | One-time early dust channel; normalization remains under calibration |

The dust source tag is retained after injection, allowing analysis of how much
information about stellar origin survives later grain growth, erosion,
coagulation, shattering, transport, and astration.

## Validation expectations

After changing yields, feedback timing, dust sampling, or mass-transfer code,
rerun:

- a short feedback timeline diagnostic;
- the global mass and element-conservation audit;
- dust-source count and mass summaries;
- the carbon-plus-silicate closure check; and
- a restart-continuity test spanning at least one feedback event.
