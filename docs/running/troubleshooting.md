# Troubleshooting

## IC rejected at startup

Confirm that all header arrays have length seven and that the empty PartType6
group contains zero-length Coordinates, Velocities, ParticleIDs, and Masses.
Run `validate_music2_ic_suite.py` before changing the simulation code.

## Particle-ID errors

Use `IDS_64BIT` and rerun the exact global uniqueness check. Never renumber an
accepted IC casually: IDs are also used to trace Lagrangian regions and match
particles across time.

## Collective gravity or domain-decomposition hang

Check the last successful sync point, newly spawned dust counts, timebin
assignment, rank imbalance, memory pressure, and whether all ranks reached the
same collective. Preserve the full log rather than only the final line.

## Unexpected dust mass jump

Break the result down by SNII, AGB, and LRN source; inspect feedback tranche
counts; verify that `DustParticlesPerSNII` is distributed across the complete
event rather than applied independently eight times; then run the whole-box
mass and element audit.

## D/G or D/Z changes unexpectedly

Verify center, aperture, phase selection, and denominator before changing
physics. D/Z using gas-phase metals differs from D/Z using gas+dust metals.

## Slow execution

Count domain decompositions, FOF/SUBFIND calls, active feedback/dust
diagnostics, and snapshot writes. Compare the actual particle population and
timebin distribution rather than inferring cost only from nominal resolution.
