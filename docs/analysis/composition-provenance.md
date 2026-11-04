# Composition and Provenance

CosmicGrain separates **current composition** from **stellar origin**.

The current model tracks an evolving carbon fraction, allowing the surviving dust mass to be separated into carbonaceous and silicate components. The stellar source tag records whether a grain population was born through the SNII, AGB, or LRN channel.

These quantities should not be treated as synonyms. A source tag is a birth-history label; later sputtering, shocks, accretion, coagulation, and shattering can alter the grain population.

## Recommended diagnostics

Report both

1. current dust mass grouped by stellar birth source; and
2. current grain size/composition conditioned on that source.

This makes it possible to ask how strongly the present-day dust population remembers its stellar origin after ISM processing.

## Integrity check

For the implemented two-component composition bookkeeping, carbonaceous plus silicate dust mass should close to total surviving dust mass to numerical precision. A failure of that closure is a code/data-integrity problem rather than an astrophysical result.
