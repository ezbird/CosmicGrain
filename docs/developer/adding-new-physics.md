# Adding New Dust Physics

CosmicGrain is intended not only to run the included dust model but also to provide infrastructure for experimenting with new dust prescriptions.

## Before adding a process

1. reproduce a known reference run or focused regression test;
2. identify which particle fields and local gas/stellar quantities the process requires;
3. decide whether the process transfers mass, momentum, composition, or only internal grain state;
4. determine how it behaves across restarts and particle exchange; and
5. define a diagnostic that can demonstrate that the process is operating as intended.

## Keep the interface narrow

A new process should ideally receive the state it needs, update the dust/gas quantities it owns, and return without embedding its physical prescription in generic GADGET-4 infrastructure. If a host-code hook is required, keep that hook small and put the physical implementation in the dust module.

## Example: a new destruction mechanism

A destruction model may need grain radius/composition, dust mass, local gas density/temperature, relative velocity, and timestep. Its implementation should clearly define

- the destruction rate or timescale;
- how mass and elements are returned to the gas, if applicable;
- how a fully destroyed superparticle is handled;
- timestep limiting or stability requirements; and
- conservation and restart tests.

## Runtime control

New physical processes should have an explicit compile/runtime control consistent with the existing model, documented parameters with units, and a startup diagnostic showing whether the process is active.

## Validation checklist

At minimum, test the process in isolation where possible, verify finite/non-negative state variables, check relevant mass/element closure, test restart continuity, and then repeat an end-to-end dust conservation audit.
