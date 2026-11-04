# Coding Conventions

CosmicGrain inherits GADGET-4's C++ style and MPI-oriented execution model. New
dust code should favor clarity of ownership over broad changes to host-code
infrastructure.

## Placement

- Put the physical prescription under `src/dust/` when practical.
- Keep host-code hooks in feedback, star formation, integration, or I/O small.
- Do not move generic GADGET-4 behavior into the dust module merely to reduce
  the number of touched files.
- Treat the modular `src/dust/{creation,dynamics,processing,...}` tree in the
  architecture page as a **planned refactor**, not the present source layout.

## State and units

- State units explicitly at interface boundaries.
- `GrainRadius` is stored numerically in **nm**; convert explicitly when a CGS
  radius is required.
- Keep `CarbonMassFraction` finite and within [0,1].
- Preserve `DustSource` and `BirthPos` when particles move between tasks or are
  written/read through snapshots.
- Avoid deriving physical time from a gravity time bin when the accumulated
  dust interval is already supplied by the cadence layer.

## Conservation and ownership

For any process that transfers mass or elements, document which routine owns
the debit and which owns the credit. In particular, stellar feedback partitions
condensed versus gas-phase ejecta before the dust creation helper spawns
PartType6 representatives; the spawning routine must not subtract the same mass
again.

## MPI and particle lifecycle

Particle creation/destruction changes global bookkeeping. New code must remain
safe across domain decomposition, particle exchange, synchronization, binary
restart, and snapshot continuation. Never assume a local particle index remains
stable after an exchange or compaction.

## Diagnostics

Development-only diagnostics should be cheap when disabled and should identify
the physical process in their prefix. Promote a diagnostic into permanent
output only when it supports reproducibility, validation, or analysis.

## Minimum tests

A material dust-physics change should include finite/range checks, a mass or
element bookkeeping check where applicable, a restart-continuity test, and an
end-to-end dust-bearing run. Numerical changes affecting gas resolution should
also be checked against the `MinGasHsml` guidance in Validation.
