# Dust Physics

CosmicGrain represents dust with explicit **PartType6 superparticles** carrying individual mass, grain radius, carbon fraction, position, velocity, formation time, and stellar-source information.

The current dust lifecycle includes:

- stellar dust creation from **SNII, AGB stars, and luminous red novae (LRNe)**
- dust cooling
- gas-dust drag
- astration
- thermal sputtering and sublimation
- ISM grain growth
- subgrid clumping
- SN shock destruction
- coagulation
- shattering
- radiation pressure
- evolving carbon/silicate composition

The enrichment model additionally tracks C, N, O, Ne, Mg, Si, and Fe in the gas.
See [Stellar Enrichment and Feedback](stellar-enrichment-feedback.md) for the
delayed SNII/hypernova tranches, MESA AGB yields, stochastic heating, and the
connection between element availability and dust creation.

## Primary physical foundations

- **Dust creation:** Todini & Ferrara (2001); Nozawa et al. (2003); Ferrarotti & Gail (2006)
- **Grain growth:** Hirashita & Kuo (2011); Asano et al. (2013)
- **Thermal sputtering:** Draine & Salpeter (1979); Tsai & Mathews (1995); McKinnon et al. (2017)
- **Shock destruction:** Jones et al. (1994, 1996); Bocchio et al. (2014)
- **Drag coupling:** McKinnon et al. (2018)
- **Grain temperature/opacity:** Draine & Lee (1984); Mathis et al. (1983); Draine & Li (2007)
- **General framework:** Dwek (1998); McKinnon et al. (2016)

## Model-status note

Implementation does not imply final calibration. In particular, the **LRN source channel is operational but its absolute dust-yield normalization remains under active investigation**. Quantitative galaxy-scale dust distributions are also being tested for numerical convergence across the zoom-resolution ladder.
