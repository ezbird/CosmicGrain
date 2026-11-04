# Dust Time Integration

CosmicGrain does not simply call every dust process on every gravity step. The
current integration layer in `src/main/run.cc` accumulates the **physical time**
covered by synchronization points and performs a dust update when the configured
cadence is reached.

## Cadence state

`DustUpdateCadence` is a runtime integer parameter. The current source accepts
values from 1 through 1,000,000. A value of 1 updates the dust physics at every
eligible synchronization point; larger values accumulate multiple intervals
before calling the dust update.

For cosmological integration, the code converts the normal GADGET integration
interval to physical time using the current cosmological factor before adding it
to `DustAccumulatedPhysicalTime`.

When the cadence is reached, `advance_dust_physics()` calls:

```text
update_dust_dynamics(..., DustAccumulatedPhysicalTime, ...)
dust_global_synchronization(...)
```

and then clears the accumulated interval and per-window creation/destruction
bookkeeping.

## Output-safe flushing

The integration interface can force an update before the nominal cadence is
reached. This is important at output or other synchronization boundaries: a
snapshot should not represent particle state at one time while silently leaving
a physically elapsed dust interval unapplied.

## Writing a new process

A new dust process should use the physical interval passed through the dust
update rather than infer a timestep from an unrelated particle time bin. Where
a process represents a continuous rate, prefer a time integration that remains
well behaved when the cadence is changed. Changes to cadence-sensitive physics
must be tested with more than one `DustUpdateCadence` value and across a restart
boundary.

## Restart interaction

The accumulated synchronization-point count and physical interval are persisted
in the CosmicGrain restart extension. See [Restart Files](../running/restart-files.md).
The active `DustUpdateCadence` value itself is read from the parameter file on
startup.
