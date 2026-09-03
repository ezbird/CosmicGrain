# Particle Types and HDF5 Layout

CosmicGrain is compiled with `NTYPES=7`. Six-element GADGET headers are not
valid for a dust-enabled build even when no dust exists initially.

| Type | Role in the current zoom workflow |
| ---: | --- |
| 0 | Gas |
| 1 | High-resolution dark matter |
| 2 | Coarser dark matter in the zoom hierarchy |
| 3 | Reserved/unused in the accepted IC suite |
| 4 | Stars formed during the simulation |
| 5 | Reserved for black holes or other configured GADGET use; unused here |
| 6 | Live CosmicGrain dust superparticles |

## Initial conditions

MUSIC2 initially writes the conventional six-type layout. The suite runner
extends `NumPart_ThisFile`, `NumPart_Total`,
`NumPart_Total_HighWord`, and `MassTable` to seven elements and creates an
empty `PartType6` group with:

```text
Coordinates   (0, 3)
Velocities    (0, 3)
ParticleIDs   (0,)
Masses        (0,)
```

The empty group and zero header count are intentional. Stellar feedback later
creates PartType6 particles.

## Evolved dust particles

Every live dust superparticle carries dynamical fields plus `Masses`,
`GrainRadius`, `CarbonMassFraction`, `DustTemperature`, `DustSource`,
formation-time information, and birth position. `DustSource` uses 0=SNII,
1=AGB, and 2=LRN. Carbon and silicate masses must sum to total dust mass
within floating-point tolerance.

Particle IDs are globally unique across all types and must use 64-bit storage
for dust-enabled production runs.
