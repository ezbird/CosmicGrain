# Units and Apertures

## Snapshot units

The analysis scripts use the GADGET conventions stored in each snapshot:

| Quantity | Convention |
| --- | --- |
| Position | comoving \(\mathrm{kpc}/h\) |
| Physical position | coordinate \(\times a/h\) |
| Mass | \(10^{10}\,M_\odot/h\) |
| Grain radius | written to HDF5 in nm; do not convert it a second time |
| Scale factor | `Header/Time` |
| Redshift | `Header/Redshift` |
| Hubble parameter | read from the snapshot parameters when required |

Always use periodic coordinate differences before centering or measuring
distances.

## Halo and galaxy apertures

`M200c` and `R200c` are defined relative to 200 times the critical density.
They describe the halo and should be measured consistently with the adopted
spherical-overdensity routine.

Galaxy-integrated quantities such as stellar mass, gas mass, SFR, dust mass,
D/G, and D/Z are aperture dependent. Every table and figure must state the
aperture explicitly. Do not silently mix quantities measured within
`R200c`, a fixed physical radius, and an ISM selection. The established
single-halo diagnostics commonly use a 20 pkpc ISM aperture, while halo-wide
budgets use `R200c`; the 12-halo paper must record its adopted galaxy aperture
alongside every catalog value.

`D/G` and `D/Z` must also state whether gas and metals are total, gas phase,
or restricted to an ISM phase selection.
