# Initial Conditions

CosmicGrain production ICs use HDF5 format and a seven-type GADGET header.
MUSIC2 output must be post-processed before it is passed to a build compiled
with `NTYPES=7`.

## Required initial state

- PartType0: gas in the high-resolution region
- PartType1: high-resolution dark matter
- PartType2: coarser dark matter surrounding the zoom
- PartTypes3–5: empty in the accepted suite
- PartType6: present but empty

The header arrays `NumPart_ThisFile`, `NumPart_Total`,
`NumPart_Total_HighWord`, and `MassTable` must each contain seven entries.
See [Particle Types](../developer/particle-types.md).

Never begin a production run from a MUSIC2 file based only on a successful
generator exit code. Run the complete [IC-suite
validator](../validation/ic-suite.md), then perform a GADGET startup test.

The accepted filename convention is:

```text
ICs/halo<HALO>/IC_halo<HALO>_zoom_<RES>.hdf5
```

Do not edit accepted HDF5 ICs in place after validation. If post-processing
changes, regenerate or create a newly named, newly validated version.
