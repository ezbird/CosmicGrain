# Dust Physics

CosmicGrain represents dust with explicit **PartType6 superparticles** carrying individual mass, grain radius, carbon fraction, position, velocity, formation time, and stellar-source information.

The current dust lifecycle includes:

- stellar dust creation from **SNII, AGB stars, and luminous red novae (LRNe)**
- dust temperature (`DustEnableCooling`; evolves the grain temperature only, with no back-reaction on gas cooling)
- gas-dust drag
- astration
- thermal sputtering
- ISM grain growth
- subgrid clumping (Blitz & Rosolowsky 2006 dense phase by default)
- SN shock destruction
- coagulation
- shattering
- radiation pressure
- evolving carbon/silicate composition

## How a dust update runs

Dust microphysics runs every `DustUpdateCadence` global synchronization points
(v1.2.0 default 7) and before every snapshot, with the exact physical time
accumulated since the last update. Each update:

1. associates every grain with its nearest eligible gas particle within 10
   code length units, searched across all MPI ranks;
2. refreshes the local D/G used by coagulation and shattering (cached per
   grain, refreshed every 10 updates);
3. runs dust temperature, drag (with sputtering), and radiation pressure, then
   growth, coagulation, and shattering for grains within
   \(\max(2,\min(H_{\rm sml},10))\) code length units of their gas.

A grain whose gas lives on another rank is sent there for its update and
returned, and astration and shock destruction gather their events from all
ranks, so results do not depend on the domain decomposition. Stellar dust
creation, astration, and shock destruction run inside the star-formation and
feedback steps, not on the dust cadence.

The enrichment model additionally tracks C, N, O, Ne, Mg, Si, and Fe in the gas.
See [Stellar Enrichment and Feedback](stellar-enrichment-feedback.md) for the
delayed SNII/hypernova tranches, MESA AGB yields, stochastic heating, and the
connection between element availability and dust creation.

## Primary physical foundations

- **Dust creation:** Todini & Ferrara (2001); Nozawa et al. (2003); Ferrarotti & Gail (2006)
- **Grain growth:** Hirashita & Kuo (2011); Asano et al. (2013)
- **Coagulation and shattering timescales:** Trayford et al. (2026), Eqs. 9–10; adapted to continuous representative radii in CosmicGrain
- **Turbulent shattering velocities:** Hirashita & Aoyama (2019), following Ormel et al. (2009) and Yan et al. (2004)
- **Thermal sputtering:** Draine & Salpeter (1979); Tsai & Mathews (1995); McKinnon et al. (2017)
- **Shock destruction:** Jones et al. (1994, 1996); Bocchio et al. (2014)
- **Drag coupling:** McKinnon et al. (2018)
- **Grain temperature/opacity:** Draine & Lee (1984); Mathis et al. (1983); Draine & Li (2007)
- **General framework:** Dwek (1998); McKinnon et al. (2016)

## Model-status note

Implementation does not imply final calibration. In particular, the **LRN source channel is operational but its absolute dust-yield normalization remains under active investigation**. Quantitative galaxy-scale dust distributions are also being tested for numerical convergence across the zoom-resolution ladder.
