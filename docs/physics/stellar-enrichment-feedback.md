# Stellar Enrichment and Feedback

CosmicGrain couples dust creation to a time-resolved stellar enrichment and
feedback model. Dust yields cannot be interpreted independently of the gas
ejecta, element reservoirs, and thermal-energy deposition that create the
local conditions in which grains form and subsequently evolve.

## Delayed enrichment

Stellar particles retain their birth mass and age in an expanding universe.
Feedback is therefore delayed rather than deposited entirely when the stellar
particle forms.

- SNII and hypernova ejecta are released in eight time tranches spanning
  approximately 3–40 Myr.
- AGB enrichment follows the delayed mass return from the stellar population
  and uses IMF-integrated MESA yield tables.
- The LRN dust channel is injected once with the first SNII tranche rather
  than repeated in all eight tranches.
- SNII dust sampling is distributed across the full tranche sequence so that
  delayed feedback does not accidentally multiply the requested number of
  dust superparticles by eight.

`StellarBirthMass` is stored for star particles so the original SSP mass is
available for diagnostics and future normalization. Do not assume from the
field's presence alone that every yield path uses birth mass; the active
feedback implementation must be checked when changing normalization.

## Element-resolved enrichment

The gas tracks C, N, O, Ne, Mg, Si, and Fe separately. SNII/hypernova yields
depend on progenitor mass and metallicity, while AGB yields are integrated
from the adopted stellar-yield grid. Total metallicity and the individual
element reservoirs are updated together.

Dust formation is element limited. The code cannot condense more carbon or
silicate material than is physically available in the relevant ejecta and
gas reservoir. Gas-to-dust and dust-to-gas transfers operate on absolute
tracked element masses and then resynchronize gas mass and metallicity fields.

## Stochastic thermal feedback

SNII/hypernova energy is deposited stochastically rather than divided as a
small temperature increment among every neighbor. The implementation:

1. obtains an adaptive gas-neighbor kernel, requiring at least 16 receivers;
2. computes the available event energy;
3. evaluates the heating energy of each receiver using its own mass and mean
   molecular weight;
4. stochastically selects receivers; and
5. heats selected gas by a fixed \(\Delta T=3.0\times10^6\,\mathrm{K}\).

`FeedbackFlag` and `EnergyReservoir` support feedback-state tracking and
diagnostics. Debug-only temperature and internal-energy limiters should not be
silently enabled in production configurations.

## Relationship to dust creation

The enrichment calculation first determines the ejecta and element budget.
The dust module then condenses the allowed fraction into explicitly tagged
PartType6 particles:

| Source | Current role |
| --- | --- |
| SNII/hypernova | Prompt, tranche-resolved dust and metal production |
| AGB | Delayed enrichment and carbon/silicate dust production |
| LRN | One-time early dust channel; normalization remains under calibration |
| SNIa | Delayed enrichment channel; not a current stellar dust source |

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
