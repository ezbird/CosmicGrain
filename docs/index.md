# CosmicGrain

**Dust evolution for GADGET-4 cosmological simulations.**

CosmicGrain is an open-source dust simulation code built on GADGET-4. Individual grain populations are represented by explicit `PartType6` superparticles that carry their own dynamics, size, composition, and stellar origin while evolving through creation, transport, growth, collisions, and destruction.

CosmicGrain is distributed as a complete GADGET-4-based simulation code rather than a standalone plug-in. It can be used to run the included dust model, explore alternative dust prescriptions and parameters, or serve as a foundation for new dust-physics implementations.

## Why evolve the grains explicitly?

Treating dust as a dynamical particle population allows the grain distribution to respond to both its **local physical environment** and its **history**. Stellar sources seed grains with different characteristic sizes and compositions; drag transports them relative to the gas; growth, coagulation, and shattering reshape the size distribution; and sputtering, shocks, and astration remove material.

The resulting particle population can be analyzed directly in terms of grain size, composition, and provenance, or connected to observables such as dust-to-gas ratios, extinction curves, and spatially varying \(R(V)\).

## Using CosmicGrain

<div class="cg-grid-3" markdown="1">

<div class="cg-card" markdown="1">
### Run
Use the complete CosmicGrain model for cosmological or galaxy simulations with explicit dust evolution.
</div>

<div class="cg-card" markdown="1">
### Experiment
Change source yields, grain sizes, processing efficiencies, destruction prescriptions, and other dust parameters.
</div>

<div class="cg-card" markdown="1">
### Extend
Use the existing dust-particle and GADGET-4 coupling infrastructure as a foundation for implementing new dust physics.
</div>

</div>

## Documentation map

- **Getting Started** — obtain the complete source, compile it, and run a first simulation
- **Running CosmicGrain** — parameters, initial conditions, outputs, restarts, and performance
- **Dust Physics** — the physical prescriptions implemented in the current model
- **Analysis** — particle fields, grain populations, composition, extinction, maps, and synthetic observations
- **Validation** — numerical integrity, resolution sensitivity, and convergence limitations
- **Developer Guide** — current architecture and guidance for modifying or adding dust physics
- **Reference** — compile flags, runtime parameters, particle types, and units

## Current science program

The current zoom suite contains 12 halos spanning dwarf through super-Milky-Way masses, with multiple resolution levels used for validation and convergence work. Lower-resolution calculations are treated as numerical validation where appropriate rather than automatically as converged galaxy predictions.

The public documentation distinguishes **implemented physics**, **current calibration choices**, and **numerical/scientific validation**. Parameters used for a particular production suite should therefore be archived with that run rather than assumed to be universal CosmicGrain defaults.
