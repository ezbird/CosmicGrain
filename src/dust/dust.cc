/*! \file dust.cc
 *  \brief Implementation of on-the-fly dust evolution model
 *
 *  This module creates dust particles from stellar feedback and evolves
 *  them through gas drag, grain growth, thermal erosion, and shock destruction.
 * 
 *  Physics references:
 *  DUST CREATION:
 *  - Todini & Ferrara 2001: SN dust yields
 *  - Ferrarotti & Gail 2006: AGB dust yields
 *  - Nozawa et al. 2003: Dust condensation in SN ejecta
 * 
 *  SHOCK DESTRUCTION:
 *  - McKee & Ostriker 1977: SN energetics (10^51 erg standard)
 *  - Sedov-Taylor solution: Self-similar blast wave expansion
 *  - Jones et al. 1994, 1996: Grain shattering in shocks
 *  - Bocchio et al. 2014: Modern destruction efficiencies
 * 
 *  THERMAL SPUTTERING:
 *  - Draine & Salpeter 1979: Thermal sputtering physics
 *  - Tsai & Mathews 1995: Sputtering in hot gas
 * 
 *  GRAIN GROWTH:
 *  - Asano et al. 2013: Accretion timescales in molecular clouds
 *  - Hirashita & Kuo 2011: Subgrid dust models
 *    - ACCRETION EFFICIENCY:
 *     - Grain size evolution combines accretion + coagulation
 *     - Only accretion depletes gas-phase metals (affects cooling)
 *     - accretion_efficiency parameter in dust_grain_growth_subgrid()
 *       controls the balance (see function for detailed documentation)
 * 
 *  GENERAL FRAMEWORK:
 *  - Dwek 1998: Dust evolution in the ISM
 *  - McKinnon et al. 2016: Dust in simulations
 */

#include "gadgetconfig.h"

#ifdef DUST

#include <gsl/gsl_math.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../dust/dust.h"
#include "../cooling_sfr/feedback.h"
#include "../cooling_sfr/spatial_hash_zoom.h"
#include "../data/allvars.h"
#include "../data/dtypes.h"
#include "../data/intposconvert.h"
#include "../data/mymalloc.h"
#include "../logs/logs.h"
#include "../logs/timer.h"
#include "../system/system.h"
#include "../cooling_sfr/cooling.h"

// Dust particle parameters
#define DUST_PARTICLE_TYPE 6
#define MIN_DUST_PARTICLE_MASS  1e-10  // Minimum mass to create dust particle; this should be plenty small to allow for very small grains, but prevents creating huge numbers of dust particles from tiny amounts of dust mass
#define DUST_MIN_GRAIN_SIZE  1.0       // nm
#define DUST_MAX_GRAIN_SIZE  200.0     // nm
#define DUST_MIN_TIMEBIN 15          // Minimum gravity timebin for dust particles; without this, dust can get to very low timebins and slow down the sim
#define DUST_SFR_EPS 1e-14             // Minimum SFR to consider a gas cell star-forming

#define DUST_PRINT(...) do{ if(All.DustDebugLevel){ \
  printf("[DUST|T=%d|a=%.6g z=%.3f] ", All.ThisTask, (double)All.Time, 1.0/All.Time-1.0); \
  printf(__VA_ARGS__); } }while(0)

extern double get_random_number(void);

// Function declarations
// NOTE: estimate_molecular_fraction and scale_factor_to_physical_time are declared
// here but appear to have no definition in this file and no call sites — likely dead.
// Keeping declarations to avoid breaking any external linkage that may exist.
double estimate_molecular_fraction(double n_H, double Z, double T);
void dust_grain_growth_subgrid(simparticles *Sp, int dust_idx, int gas_idx, double dt);
double scale_factor_to_physical_time(double delta_a);

// Access feedback's global spatial hash
extern spatial_hash_zoom gas_hash;
extern spatial_hash_zoom star_hash;
extern spatial_hash_zoom dust_hash;
extern void rebuild_feedback_spatial_hash(simparticles *Sp, double max_feedback_radius, MPI_Comm comm);

// Module-level statistics
long long NDustCreated              = 0;
long long NDustDestroyed            = 0;
double    TotalDustMass             = 0.0;
long long LocalDustCreatedThisStep  = 0;
long long LocalDustDestroyedThisStep= 0;
double    LocalDustMassChange       = 0.0;
int       DustNeedsSynchronization  = 0;
long long GlobalDustCount           = 0;

// Hash usage tracking
static long long HashSearches       = 0;
static long long HashSearchesFailed = 0;  // That is, no neighbor found

// Destruction mechanism tracking
long long NDustDestroyedByThermal      = 0;
long long NDustDestroyedByShock        = 0;
long long NDustDestroyedByAstration    = 0;
double    TotalDustMassAstrated        = 0.0;

// -----------------------------------------------------------------------
// NEW: Per-pathway destruction counters for diagnosing mystery losses.
// These track deletions that happen OUTSIDE the physics enable flags.
// -----------------------------------------------------------------------
long long NDustDestroyedByCleanup      = 0;  // cleanup_invalid_dust_particles()
long long NDustDestroyedByCorruption   = 0;  // dust_grain_growth_subgrid() corruption removal
long long NDustDestroyedByBadGasIndex  = 0;  // stale hash returned non-gas particle

// Growth/erosion tracking
long long NGrainGrowthEvents        = 0;
long long NGrainErosionEvents       = 0;
double    TotalMassGrown            = 0.0;
double    TotalMassEroded           = 0.0;

// Coagulation
long long NCoagulationEvents        = 0;
double    TotalMassCoagulated       = 0.0;

/**
 * Clean up invalid dust particles, with detailed corruption diagnostics.
 *
 * Prints per-task reason tallies so we can identify whether corruption
 * is coming from domain exchange (GrainRadius zeroed), creation bugs
 * (bad position/velocity), or something else.
 */
void cleanup_invalid_dust_particles(simparticles *Sp)
{
  // Reason counters (local to this task)
  int n_bad_radius   = 0;
  int n_bad_mass     = 0;
  int n_bad_pos      = 0;
  int n_bad_vel      = 0;
  int n_bad_carbon   = 0;
  int n_bad_temp     = 0;
  int cleaned        = 0;

  for(int i = 0; i < Sp->NumPart; i++) {
    if(Sp->P[i].getType() != DUST_PARTICLE_TYPE) continue;

    double a    = Sp->DustP[i].GrainRadius;
    double mass = Sp->P[i].getMass();
    double cf   = Sp->DustP[i].CarbonFraction;
    double temp = Sp->DustP[i].DustTemperature;
    double pos[3];
    Sp->intpos_to_pos(Sp->P[i].IntPos, pos);

    bool is_corrupt = false;

    if(a <= 0.0 || !isfinite(a))                                          { n_bad_radius++; is_corrupt = true; }
    if(mass < 1e-20 || !isfinite(mass))                                   { n_bad_mass++;   is_corrupt = true; }
    if(!isfinite(pos[0]) || !isfinite(pos[1]) || !isfinite(pos[2]))       { n_bad_pos++;    is_corrupt = true; }
    if(!isfinite(Sp->P[i].Vel[0]) || !isfinite(Sp->P[i].Vel[1]) ||
       !isfinite(Sp->P[i].Vel[2]))                                        { n_bad_vel++;    is_corrupt = true; }
    if(!isfinite(cf) || cf < 0.0 || cf > 1.0)                             { n_bad_carbon++; is_corrupt = true; }
    if(!isfinite(temp) || temp < 0.0)                                     { n_bad_temp++;   is_corrupt = true; }

    if(is_corrupt) {
      // Distinguish domain-exchange victims from genuine corruption.
      // When DomainDecomp migrates a dust particle, the P[] struct arrives
      // correctly (type, mass, ID) but DustP[] is zeroed by memset — so
      // GrainRadius == 0.0 exactly while CarbonFraction, DustTemperature,
      // etc. are also all exactly 0.0.  That is a different failure mode
      // from, say, a NaN from a numerical explosion.
      bool is_domain_exchange_victim = (a == 0.0 && cf == 0.0 && temp == 0.0);

      // Print first few individual cases for detailed inspection
      static int detail_count = 0;
      if(detail_count < 50) {
        printf("[CORRUPT_DETAIL|T=%d|Step=%d] idx=%d ID=%lld: "
               "a=%.3e mass=%.3e cf=%.3f T=%.1f "
               "pos=(%.2f,%.2f,%.2f) vel=(%.2f,%.2f,%.2f) %s\n",
               All.ThisTask, All.NumCurrentTiStep, i,
               (long long)Sp->P[i].ID.get(),
               a, mass, cf, temp,
               pos[0], pos[1], pos[2],
               Sp->P[i].Vel[0], Sp->P[i].Vel[1], Sp->P[i].Vel[2],
               is_domain_exchange_victim ? "[LIKELY_DOMAIN_EXCHANGE_VICTIM]" : "[GENUINE_CORRUPTION]");
        detail_count++;
      }

      // Mark for removal
      Sp->P[i].setMass(1e-30);
      Sp->P[i].ID.set(0);
      Sp->P[i].setType(3);
      memset(&Sp->DustP[i], 0, sizeof(dust_data));
      Sp->DustP[i].GrainRadius = DUST_MIN_GRAIN_SIZE;

      // NEW: Track every removal through this path
      NDustDestroyedByCleanup++;
      cleaned++;
    }
  }

  // Each task reports its own tally (not just task 0) so we can see
  // whether corruption is concentrated on specific tasks after domain exchange
  if(cleaned > 0) {
    // Count how many were domain exchange victims vs genuine corruption
    // (domain exchange victims have all DustP fields zeroed by memset)
    int n_domain_exchange = 0;
    // We already tallied per-reason; if bad_radius is the dominant reason
    // and bad_mass/bad_pos/bad_vel/bad_carbon are zero, likely domain exchange.
    // The CORRUPT_DETAIL prints above show the [LIKELY_DOMAIN_EXCHANGE_VICTIM] tag.
    printf("[CLEANUP|T=%d|Step=%d|a=%.4f] Removed %d corrupted dust particles: "
           "bad_radius=%d bad_mass=%d bad_pos=%d bad_vel=%d bad_carbon=%d bad_temp=%d "
           "(if bad_radius dominates with cf=0,T=0 → domain exchange DustP not sent) "
           "| RunningTotal=%lld\n",
           All.ThisTask, All.NumCurrentTiStep, All.Time,
           cleaned,
           n_bad_radius, n_bad_mass, n_bad_pos, n_bad_vel, n_bad_carbon, n_bad_temp,
           NDustDestroyedByCleanup);
  }
}

/**
 * Convert entropy to temperature for a gas particle
 * Accounts for ionization state via electron fraction
 */
double get_temperature_from_entropy(simparticles *Sp, int idx)
{
  double utherm = Sp->get_utherm_from_entropy(idx);
  double ne = Sp->SphP[idx].Ne;
  
  double XH = HYDROGEN_MASSFRAC;
  double Y = (1.0 - XH) / (4.0 * XH);
  double mu = (1.0 + 4.0 * Y) / (1.0 + Y + ne);
  
  double temp = (GAMMA - 1.0) * utherm * (All.UnitEnergy_in_cgs / All.UnitMass_in_g) 
                / BOLTZMANN * PROTONMASS * mu;
  
  return temp;
}

/**
 * Erode dust grain through thermal sputtering
 * Returns 1 if particle was destroyed (too small), 0 otherwise
 */
int erode_dust_grain_thermal(simparticles *Sp, int dust_idx, int nearest_gas_input, double T_gas, double dt)
{
  if(!All.DustEnableSputtering) return 0;
    
    double a = Sp->DustP[dust_idx].GrainRadius;
    
    // INSTANT DESTRUCTION in extremely hot gas (> 1e6 K)
    // Physics: Dust sublimates in < 1 kyr at these temperatures
    if(T_gas > 1e6) {
      double dust_mass = Sp->P[dust_idx].getMass();
      
    // Use passed gas instead of searching
    if(nearest_gas_input >= 0) {
      double gas_mass = Sp->P[nearest_gas_input].getMass();
        double current_Z = Sp->SphP[nearest_gas_input].Metallicity;
        double new_Z = current_Z + (dust_mass / gas_mass);
        Sp->SphP[nearest_gas_input].Metallicity = new_Z;
        
        #ifdef STARFORMATION
        Sp->SphP[nearest_gas_input].MassMetallicity = gas_mass * new_Z;
        #endif
      }
      
      //DUST_PRINT("[INSTANT_SUBLIMATION] Dust destroyed at T=%.2e K (too hot!)\n", T_gas);
      
      Sp->DustP[dust_idx].GrainRadius = DUST_MIN_GRAIN_SIZE;
      Sp->P[dust_idx].setMass(1e-30);
      Sp->P[dust_idx].setType(3);
      Sp->P[dust_idx].ID.set(0);
      memset(&Sp->DustP[dust_idx], 0, sizeof(dust_data));

      LocalDustMassChange -= dust_mass;
      LocalDustDestroyedThisStep++;
      DustNeedsSynchronization = 1;
      NDustDestroyedByThermal++;
      TotalMassEroded += dust_mass;
      
      return 1;  // DESTROYED
    }
  
  // Cool enough that no sputtering occurs, let's exit early
  if(T_gas < All.DustThermalSputteringTemp) {
    return 0;
  }
  
  // ===============================================================================
  // Sputtering timescale: McKinnon et al. (2017), Eq. 2
  // tau_sp = (0.17 Gyr) * (a_-1 / rho_-27) * [(T0/T)^omega + 1]
  // where a_-1 = grain radius in units of 0.1 um
  //       rho_-27 = gas density in units of 1e-27 g/cm^3
  // Saturates at high T (physically correct), and scales linearly with grain size

  // T0 and omega from Tielens et al. (1994), as used by McKinnon+2017
  double T0    = 2e6;    // K, characteristic sputtering temperature
  double omega = 2.5;    // power law index

  // Convert grain radius to units of 0.1 um (= 1e-5 cm)
  // GrainRadius is stored in [YOUR UNITS - adjust conversion accordingly]
  double a_cgs  = a * 1e-7;   // convert to cm
  double a_minus1 = a_cgs / 1e-5;              // units of 0.1 um

  // Convert gas density to units of 1e-27 g/cm^3
  double rho_cgs  = Sp->SphP[nearest_gas_input].Density
                    * All.UnitDensity_in_cgs
                    * All.cf_a3inv;
  double rho_minus27 = rho_cgs / 1e-27;

  // Guard against zero/unphysical density
  if(rho_minus27 <= 0.0 || !isfinite(rho_minus27)) rho_minus27 = 1e-4 / 1e-27 * PROTONMASS / 1e-27; // fallback

  double tau_sputter_yr = 0.17e9 * (a_minus1 / rho_minus27)
                          * (pow(T0 / T_gas, omega) + 1.0);
  
  // ===============================================================================

  // Apply reasonable bounds
  if(tau_sputter_yr < 1e6)  tau_sputter_yr = 1e6;   // 1 Myr floor (very hot gas)
  if(tau_sputter_yr > 1e9) tau_sputter_yr = 1e9;    // 1 Gyr ceiling (cool gas)
  
  double tau_sputter = tau_sputter_yr * SEC_PER_YEAR / All.UnitTime_in_s;
  double da_dt = -a / tau_sputter;
  double da = da_dt * dt;

  // If timestep exceeds sputtering timescale, grain is fully destroyed this step
  if(da <= -a) {
      // destroy particle, return metals to gas...
      return 1;
  }

  double a_new = a + da;
  
  if(a_new <= 0.0 || !isfinite(a_new))
    {
      DUST_PRINT("[BUG] Thermal erosion created invalid a_new=%.3e (a=%.3e, da=%.3e)\n",
                a_new, a, da);
        Sp->P[dust_idx].setMass(1e-30);
        Sp->P[dust_idx].ID.set(0);
        Sp->P[dust_idx].setType(3); // hack! Changing this to an unused parttype so find_nearest_dust_particle() skips it
      return 1; // Destroy it
    }

  if(a_new < DUST_MIN_GRAIN_SIZE) {
    double dust_mass = Sp->P[dust_idx].getMass();
    
    if(nearest_gas_input >= 0) {
      double gas_mass = Sp->P[nearest_gas_input].getMass();
      double current_Z = Sp->SphP[nearest_gas_input].Metallicity;
      double new_Z = current_Z + (dust_mass / gas_mass);
      Sp->SphP[nearest_gas_input].Metallicity = new_Z;
      
      #ifdef STARFORMATION
      Sp->SphP[nearest_gas_input].MassMetallicity = gas_mass * new_Z;
      #endif
      
      DUST_PRINT("[SPUTTERING] Grain eroded at T=%.2e K: a=%.3f nm, dust %.2e Msun\n",
           T_gas, DUST_MIN_GRAIN_SIZE, dust_mass);
    }
    
    Sp->DustP[dust_idx].GrainRadius = DUST_MIN_GRAIN_SIZE;
    Sp->P[dust_idx].setMass(1e-30);
    Sp->P[dust_idx].setType(3); // hack! Changing this to an unused parttype so find_nearest_dust_particle() skips it
    Sp->P[dust_idx].ID.set(0);
    memset(&Sp->DustP[dust_idx], 0, sizeof(dust_data));

    LocalDustMassChange -= dust_mass;
    LocalDustDestroyedThisStep++;
    DustNeedsSynchronization = 1;
    NDustDestroyedByThermal++;
    TotalMassEroded += dust_mass;
    
    return 1;
  }
  
  Sp->DustP[dust_idx].GrainRadius = a_new;
  
  double mass_ratio = pow(a_new / a, 3.0);

  // Catch any NaN early in erosion before it corrupts the particle array
  if(!isfinite(mass_ratio) || mass_ratio < 0.0 || mass_ratio > 1.5) {
    if(All.ThisTask == 0) {
      DUST_PRINT("[ERROR] Invalid mass_ratio=%.3e in erosion (a=%.3e→%.3e)\n",
                 mass_ratio, a, a_new);
    }
    return 0;  // Abort this erosion event
  }

  double old_mass = Sp->P[dust_idx].getMass();
  double new_mass = old_mass * mass_ratio;
  double mass_lost = old_mass - new_mass;
  
  Sp->P[dust_idx].setMass(new_mass);
  
  // Use passed gas index instead of searching  
  if(nearest_gas_input >= 0) {
    double gas_mass = Sp->P[nearest_gas_input].getMass();
    double current_Z = Sp->SphP[nearest_gas_input].Metallicity;
    double new_Z = current_Z + (mass_lost / gas_mass);
    Sp->SphP[nearest_gas_input].Metallicity = new_Z;
    
    #ifdef STARFORMATION
    Sp->SphP[nearest_gas_input].MassMetallicity = gas_mass * new_Z;
    #endif
  }
  
  LocalDustMassChange -= mass_lost;
  NGrainErosionEvents++;
  TotalMassEroded     += mass_lost;
  
  static int erosion_count = 0;
  erosion_count++;
  if(erosion_count % 10000 == 0 && All.ThisTask == 0) {  // Print every 10,000th event
      DUST_PRINT("[EROSION] Grain shrunk: %.2f → %.2f nm (dm=%.2e, T=%.0f K)\n",
                a, a_new, mass_lost, T_gas);
  }
  
  return 0;
}

static int destroy_dust_particle_to_gas(simparticles *Sp, int dust_idx, 
                                         int nearest_gas, long long *counter)
{
  double dust_mass = Sp->P[dust_idx].getMass();
  
  if(nearest_gas >= 0) {
    double gas_mass = Sp->P[nearest_gas].getMass();
    double new_Z = Sp->SphP[nearest_gas].Metallicity + (dust_mass / gas_mass);
    Sp->SphP[nearest_gas].Metallicity = new_Z;
    #ifdef STARFORMATION
    Sp->SphP[nearest_gas].MassMetallicity = gas_mass * new_Z;
    #endif
  }
  
  Sp->DustP[dust_idx].GrainRadius = DUST_MIN_GRAIN_SIZE;
  Sp->P[dust_idx].setMass(1e-30);
  Sp->P[dust_idx].setType(3);
  Sp->P[dust_idx].ID.set(0);
  memset(&Sp->DustP[dust_idx], 0, sizeof(dust_data));
  
  LocalDustMassChange -= dust_mass;
  LocalDustDestroyedThisStep++;
  DustNeedsSynchronization = 1;
  if(counter) (*counter)++;
  TotalMassEroded += dust_mass;
  
  return 1;
}

/**
500 km/s shock + 10 nm grain: ~90% destruction chance (outright shattering)
500 km/s shock + 100 nm grain: ~30% destruction chance
200 km/s shock + 10 nm grain: ~10% destruction chance
200 km/s shock + 100 nm grain: 0% destruction, but erosion
 */
int erode_dust_grain_shock(simparticles *Sp, int dust_idx, double shock_velocity_km_s, 
                           double distance_to_sn, double shock_radius)
{
  double a = Sp->DustP[dust_idx].GrainRadius;
  
  // ========================================================================
  // Attenuate local shock velocity by distance (Sedov-Taylor: v ∝ r^{-3/2})
  // Grain at shock edge sees ~30% of peak velocity; grain at center sees full.
  // ========================================================================
  double r_frac = (shock_radius > 0) ? (distance_to_sn / shock_radius) : 0.0;
  if(r_frac < 0.0) r_frac = 0.0;
  if(r_frac > 1.0) r_frac = 1.0;
  
  double velocity_attenuation = pow(1.0 - 0.7 * r_frac, 1.5);
  if(velocity_attenuation < 0.3) velocity_attenuation = 0.3;
  
  double local_velocity = shock_velocity_km_s * velocity_attenuation;
  
  // ========================================================================
  // Single grain-size factor used consistently for both shattering and erosion
  // ========================================================================
  double size_factor;
  if     (a < 20.0)  size_factor = 1.5;
  else if(a < 50.0)  size_factor = 1.2;
  else if(a > 150.0) size_factor = 0.7;
  else               size_factor = 1.0;
  
  // ========================================================================
  // Find nearest gas once — reused for all three destruction paths
  // ========================================================================
  int nearest_gas = find_nearest_gas_particle(Sp, dust_idx, 5.0, NULL);
  
  // ========================================================================
  // STEP 1: Outright shattering (stochastic, velocity-gated)
  // ========================================================================
  if(local_velocity > 150.0) {
    double velocity_factor = (local_velocity - 150.0) / 350.0;
    if(velocity_factor > 1.0) velocity_factor = 1.0;
    
    // Unified size factor: small grains shatter more easily
    double destruction_size_factor = 50.0 / a;
    if(destruction_size_factor > 3.0) destruction_size_factor = 3.0;
    if(destruction_size_factor < 0.3) destruction_size_factor = 0.3;
    
    double destruction_prob = velocity_factor * destruction_size_factor * size_factor;
    if(destruction_prob > 0.9) destruction_prob = 0.9;
    
    if(get_random_number() < destruction_prob) {
      return destroy_dust_particle_to_gas(Sp, dust_idx, nearest_gas,
                                          &NDustDestroyedByShock);
    }
  }
  
  // ========================================================================
  // STEP 2: Erosion using local (attenuated) velocity
  // ========================================================================
  double base_efficiency = get_shock_destruction_efficiency(local_velocity);
  double erosion_fraction = base_efficiency * size_factor;
  if(erosion_fraction > 0.95) erosion_fraction = 0.95;
  
  double a_new = a * (1.0 - erosion_fraction * 0.8);
  if(a_new <= 0.0 || !isfinite(a_new)) a_new = 0.0;
  
  if(a_new < DUST_MIN_GRAIN_SIZE) {
    return destroy_dust_particle_to_gas(Sp, dust_idx, nearest_gas,
                                        &NDustDestroyedByShock);
  }
  
  // ========================================================================
  // STEP 3: Grain survived — update size and return eroded mass to gas
  // ========================================================================
  Sp->DustP[dust_idx].GrainRadius = a_new;
  
  double mass_ratio = pow(a_new / a, 3.0);
  if(!isfinite(mass_ratio) || mass_ratio < 0.0 || mass_ratio > 1.5) {
    DUST_PRINT("[ERROR] Invalid mass_ratio=%.3e in shock erosion (a=%.3e→%.3e)\n",
               mass_ratio, a, a_new);
    return 0;
  }
  
  double old_mass = Sp->P[dust_idx].getMass();
  double mass_lost = old_mass * (1.0 - mass_ratio);
  Sp->P[dust_idx].setMass(old_mass - mass_lost);
  
  if(nearest_gas >= 0) {
    double gas_mass = Sp->P[nearest_gas].getMass();
    double new_Z = Sp->SphP[nearest_gas].Metallicity + (mass_lost / gas_mass);
    Sp->SphP[nearest_gas].Metallicity = new_Z;
    #ifdef STARFORMATION
    Sp->SphP[nearest_gas].MassMetallicity = gas_mass * new_Z;
    #endif
  }
  
  LocalDustMassChange -= mass_lost;
  NGrainErosionEvents++;
  TotalMassEroded += mass_lost;
  
  return 0;
}

/*!
 * \brief Calculate Epstein drag stopping timescale for a dust grain
 * 
 * Implements the standard Epstein drag formula with supersonic correction
 * following McKinnon+2018 MNRAS 478, 2851 (equations 8-9).
 */
static double calculate_drag_timescale(
    double grain_radius_nm,
    double grain_density,
    double gas_density_cgs,
    double gas_temperature,
    double relative_velocity_cgs,
    double mu_gas,
    double gamma_gas)
{
    const double k_boltzmann = 1.38064852e-16;
    const double m_proton = 1.6726219e-24;
    const double sec_to_myr = 3.15576e13;
    
    double grain_radius_cm = grain_radius_nm * 1e-7;
    
    double cs_cgs = sqrt(gamma_gas * k_boltzmann * gas_temperature / (mu_gas * m_proton));
    
    if(cs_cgs < 1e3 || gas_density_cgs < 1e-30)
    {
        return 50.0;
    }
    
    double t_stop_subsonic = (sqrt(M_PI * gamma_gas) * grain_radius_cm * grain_density) 
                             / (2.0 * sqrt(2.0) * gas_density_cgs * cs_cgs);
    
    double mach = relative_velocity_cgs / cs_cgs;
    double supersonic_factor = 1.0;
    
    if(mach > 0.1)
    {
        supersonic_factor = 1.0 / sqrt(1.0 + (9.0 * M_PI / 128.0) * mach * mach);
    }
    
    double t_stop_sec = t_stop_subsonic * supersonic_factor;
    double drag_timescale_myr = t_stop_sec / sec_to_myr;
    
    if(drag_timescale_myr < 0.001) drag_timescale_myr = 0.001;
    if(drag_timescale_myr > 50.0) drag_timescale_myr = 50.0;
    
    return drag_timescale_myr;
}

/**
 * Dust-gas interaction with drag and thermal erosion
 */
int dust_gas_interaction(simparticles *Sp, int dust_idx, int nearest_gas, double dt)
{
  if(!All.DustEnableDrag) return 0;
  
  if(nearest_gas < 0) return 0;

  // -----------------------------------------------------------------------
  // Guard against stale hash returning a converted star or other
  // non-gas particle. Gas particles are type 0; anything else means the
  // hash was built when that slot was gas but it has since been converted.
  // Reading SphP[] for a non-gas particle is undefined behaviour and can
  // produce garbage T_gas values that trigger erode_dust_grain_thermal.
  // -----------------------------------------------------------------------
  if(Sp->P[nearest_gas].getType() != 0) {
    static long long bad_gas_warns = 0;
    bad_gas_warns++;
    NDustDestroyedByBadGasIndex++;   // Count every occurrence, not just warned ones
    if(bad_gas_warns <= 50) {
      printf("[BAD_GAS_IDX|T=%d|Step=%d] dust_idx=%d nearest_gas=%d "
             "has type=%d (expected 0!), mass=%.3e, ID=%lld "
             "| RunningBadCount=%lld\n",
             All.ThisTask, All.NumCurrentTiStep,
             dust_idx, nearest_gas,
             Sp->P[nearest_gas].getType(),
             Sp->P[nearest_gas].getMass(),
             (long long)Sp->P[nearest_gas].ID.get(),
             bad_gas_warns);
    }
    return 0;  // Skip — do not process drag against a non-gas particle
  }

  // ==================================================================
  // STEP 1: Extract gas properties
  // ==================================================================
  
  double gas_vel[3] = {Sp->P[nearest_gas].Vel[0],
                       Sp->P[nearest_gas].Vel[1],
                       Sp->P[nearest_gas].Vel[2]};
  
  double gas_density = Sp->SphP[nearest_gas].Density * All.cf_a3inv;
  double gas_density_cgs = gas_density * All.UnitDensity_in_cgs;
  double n_H = gas_density_cgs / PROTONMASS;
  
  double utherm = Sp->get_utherm_from_entropy(nearest_gas);
  double T_gas = utherm * (All.UnitEnergy_in_cgs / All.UnitMass_in_g) 
                 / BOLTZMANN * PROTONMASS * 0.6;
  
  // ==================================================================
  // STEP 2: Calculate relative velocity
  // ==================================================================
  
  double vrel_x = Sp->P[dust_idx].Vel[0] - gas_vel[0];
  double vrel_y = Sp->P[dust_idx].Vel[1] - gas_vel[1];
  double vrel_z = Sp->P[dust_idx].Vel[2] - gas_vel[2];
  double vrel = sqrt(vrel_x*vrel_x + vrel_y*vrel_y + vrel_z*vrel_z);
  double vrel_cgs = vrel * All.UnitVelocity_in_cm_per_s;
  
  // ==================================================================
  // STEP 3: Calculate Epstein drag timescale
  // ==================================================================
  
  double mu_gas = 0.6;
  double gamma_gas = 5.0/3.0;

  double drag_timescale_myr = calculate_drag_timescale(
      Sp->DustP[dust_idx].GrainRadius,
      2.4,
      gas_density_cgs,
      T_gas,
      vrel_cgs,
      mu_gas,
      gamma_gas
  );
  
  double drag_timescale = drag_timescale_myr * 1e6 * SEC_PER_YEAR / All.UnitTime_in_s;
  
  // ==================================================================
  // STEP 4: Apply drag (exact analytical solution)
  // ==================================================================
  
  double drag_factor = 1.0 - exp(-dt / drag_timescale);

  for(int k = 0; k < 3; k++) {
    Sp->P[dust_idx].Vel[k] += drag_factor * (gas_vel[k] - Sp->P[dust_idx].Vel[k]);
  }
  
  // ========================================================================
  // STEP 5: Update dust temperature
  // ========================================================================

  double T_CMB = 2.7 * (1.0 / All.Time);

  double T_dust_eq = sqrt(T_gas * T_CMB);

  if(T_dust_eq > 2000.0) T_dust_eq = 2000.0;
  if(T_dust_eq < T_CMB) T_dust_eq = T_CMB;

  double T_dust = Sp->DustP[dust_idx].DustTemperature;
  double tau_thermal = 10e6 * SEC_PER_YEAR / All.UnitTime_in_s;
  double alpha = 1.0 - exp(-dt / tau_thermal);

  T_dust = T_dust * (1.0 - alpha) + T_dust_eq * alpha;

  if(T_dust < T_CMB) T_dust = T_CMB;
  if(T_dust > 2000.0) T_dust = 2000.0;
  if(!isfinite(T_dust) || T_dust <= 0.0) T_dust = T_CMB;

  Sp->DustP[dust_idx].DustTemperature = T_dust;
  
  // ==================================================================
  // STEP 6: Thermal erosion check
  // ==================================================================
  
    if(T_gas > All.DustThermalSputteringTemp) {
      int destroyed = erode_dust_grain_thermal(Sp, dust_idx, nearest_gas, T_gas, dt);
      return destroyed;
    }
  
  // ==================================================================
  // STEP 7: Diagnostic output (sample 1% of particles)
  // ==================================================================

  if(All.ThisTask == 0 && Sp->P[dust_idx].ID.get() % 100 == 0) {
    static int drag_samples = 0;
    if(drag_samples < 500) {
    
      double vel_kms = sqrt(Sp->P[dust_idx].Vel[0]*Sp->P[dust_idx].Vel[0] + 
                          Sp->P[dust_idx].Vel[1]*Sp->P[dust_idx].Vel[1] + 
                          Sp->P[dust_idx].Vel[2]*Sp->P[dust_idx].Vel[2]);
      vel_kms *= All.UnitVelocity_in_cm_per_s / 1e5;
      
      double cs_cgs = sqrt(gamma_gas * BOLTZMANN * T_gas / (mu_gas * PROTONMASS));
      double mach_number = vrel_cgs / cs_cgs;
      double vrel_kms = vrel_cgs / 1e5;

      double dt_myr = dt * All.UnitTime_in_s / (1e6 * SEC_PER_YEAR);
      double drag_factor_raw = 1.0 - exp(-dt / drag_timescale);
      double drag_factor_used = drag_factor;

      DUST_PRINT(
        "[DUST_DRAG] vel=%.1f km/s Δv=%.1f km/s "
        "nH=%.3e cm^-3 T=%.1e K Mach=%.2f "
        "t_drag=%.2f Myr dt=%.2f Myr f_drag=%.3f→%.3f "
        "a=%.2f nm\n",
        vel_kms, vrel_kms,
        n_H, T_gas, mach_number,
        drag_timescale_myr, dt_myr,
        drag_factor_raw, drag_factor_used,
        Sp->DustP[dust_idx].GrainRadius
      );
            
      drag_samples++;
    }
  }

  return 0;
}

/**
 * Create dust particles from stellar feedback
 */
void create_dust_particles_from_feedback(simparticles *Sp, int star_idx, 
                                         double metals_produced, int feedback_type)
{ 
  if(!All.DustEnableCreation) return;

  double dust_yield_fraction;
  double velocity_scale;
  
  if(feedback_type == 1) {
    dust_yield_fraction = All.DustYieldSNII;
    velocity_scale = All.DustVelocitySNII;
  } else if(feedback_type == 2) {
    dust_yield_fraction = All.DustYieldAGB;
    velocity_scale = All.DustVelocityAGB;
  } else {
    return;
  }
  
  double total_dust_mass = metals_produced * dust_yield_fraction;
  
  if(total_dust_mass < MIN_DUST_PARTICLE_MASS) {
    return;
  }
  
  // Gas metallicity should be reduced so some of those metals are now locked in grains
  // Find the nearest gas particle to the star and reduce its metallicity accordingly
  int nearest_gas = find_nearest_gas_particle(Sp, star_idx, 2.0, NULL);
  if(nearest_gas >= 0) {
      double gas_mass = Sp->P[nearest_gas].getMass();
      double dZ = total_dust_mass / gas_mass;
      Sp->SphP[nearest_gas].Metallicity -= dZ;
      if(Sp->SphP[nearest_gas].Metallicity < 0) Sp->SphP[nearest_gas].Metallicity = 0;
      #ifdef STARFORMATION
      Sp->SphP[nearest_gas].MassMetallicity = gas_mass * Sp->SphP[nearest_gas].Metallicity;
      #endif
  }

  int n_dust_particles = (feedback_type == 1) ? All.DustParticlesPerSNII : All.DustParticlesPerAGB;
  double dust_mass_per_particle = total_dust_mass / n_dust_particles;

  for(int n = 0; n < n_dust_particles; n++) {
    if(Sp->NumPart >= Sp->MaxPart) {
      if(All.ThisTask == 0) {
        DUST_PRINT("[WARNING] Cannot create dust particle - particle array full\n");
      }
      break;
    }
    
    double theta = acos(2.0 * get_random_number() - 1.0);
    double phi   = 2.0 * M_PI * get_random_number();
    
    double offset_min, offset_max;
    if(feedback_type == 1) {
      offset_min = All.DustOffsetMinSNII;
      offset_max = All.DustOffsetMaxSNII;
    } else {
      offset_min = All.DustOffsetMinAGB;
      offset_max = All.DustOffsetMaxAGB;
    }

    double r = offset_min + (offset_max - offset_min) * get_random_number();

    double offset_kpc[3];
    offset_kpc[0] = r * sin(theta) * cos(phi);
    offset_kpc[1] = r * sin(theta) * sin(phi);
    offset_kpc[2] = r * cos(theta);
    
  double star_vel[3];
  star_vel[0] = Sp->P[star_idx].Vel[0];
  star_vel[1] = Sp->P[star_idx].Vel[1];
  star_vel[2] = Sp->P[star_idx].Vel[2];

  double initial_velocity[3];
  initial_velocity[0] = velocity_scale * sin(theta) * cos(phi) / All.UnitVelocity_in_cm_per_s * 1e5;
  initial_velocity[1] = velocity_scale * sin(theta) * sin(phi) / All.UnitVelocity_in_cm_per_s * 1e5;
  initial_velocity[2] = velocity_scale * cos(theta) / All.UnitVelocity_in_cm_per_s * 1e5;

  initial_velocity[0] += star_vel[0];
  initial_velocity[1] += star_vel[1];
  initial_velocity[2] += star_vel[2];
    
    spawn_dust_particle(Sp, offset_kpc, dust_mass_per_particle, initial_velocity, star_idx, feedback_type);
    
    int new_idx = Sp->NumPart - 1;

    if(feedback_type == 1) {
      Sp->DustP[new_idx].GrainRadius = 10.0;
      Sp->DustP[new_idx].CarbonFraction = 0.1;
      Sp->DustP[new_idx].GrainType = 0;
    }
    else if(feedback_type == 2) {
      Sp->DustP[new_idx].GrainRadius = 100.0;
      Sp->DustP[new_idx].CarbonFraction = 0.6;
      Sp->DustP[new_idx].GrainType = 1;
    }
  }

  LocalDustCreatedThisStep += n_dust_particles;
  LocalDustMassChange      += total_dust_mass;
  DustNeedsSynchronization  = 1;

if(All.ThisTask == 0) {
  static int velocity_samples = 0;
  if(velocity_samples < 50) {
    int new_idx = Sp->NumPart - 1;
    if(new_idx >= 0 && Sp->P[new_idx].getType() == DUST_PARTICLE_TYPE) {
      double vel_mag = sqrt(Sp->P[new_idx].Vel[0]*Sp->P[new_idx].Vel[0] + 
                           Sp->P[new_idx].Vel[1]*Sp->P[new_idx].Vel[1] + 
                           Sp->P[new_idx].Vel[2]*Sp->P[new_idx].Vel[2]);
      vel_mag *= All.UnitVelocity_in_cm_per_s / 1e5;
      
      double star_vel_mag = sqrt(Sp->P[star_idx].Vel[0]*Sp->P[star_idx].Vel[0] + 
                                Sp->P[star_idx].Vel[1]*Sp->P[star_idx].Vel[1] + 
                                Sp->P[star_idx].Vel[2]*Sp->P[star_idx].Vel[2]);
      star_vel_mag *= All.UnitVelocity_in_cm_per_s / 1e5;
      
      int nearest_gas = find_nearest_gas_particle(Sp, star_idx, 2.0, NULL);
      double rho = 1.0;
      double gas_vel_mag_diag = 0.0;
      
      if(nearest_gas >= 0) {
        double gas_density = Sp->SphP[nearest_gas].Density * All.cf_a3inv;
        rho = gas_density * All.UnitDensity_in_cgs / PROTONMASS;
        
        gas_vel_mag_diag = sqrt(Sp->P[nearest_gas].Vel[0]*Sp->P[nearest_gas].Vel[0] + 
                          Sp->P[nearest_gas].Vel[1]*Sp->P[nearest_gas].Vel[1] + 
                          Sp->P[nearest_gas].Vel[2]*Sp->P[nearest_gas].Vel[2]);
        gas_vel_mag_diag *= All.UnitVelocity_in_cm_per_s / 1e5;
      }
      
      DUST_PRINT("[DUST_CREATE] vel_dust=%.1f km/s vel_star=%.1f km/s vel_gas=%.1f km/s "
                 "rho=%.3e cm^-3 grain_r=%.2f nm feedback_type=%d\n",
                 vel_mag, star_vel_mag, gas_vel_mag_diag, 
                 rho, Sp->DustP[new_idx].GrainRadius, feedback_type);
      
      velocity_samples++;
    }
  }
}
}

/**
 * Global synchronization of dust statistics
 */
void dust_global_synchronization(simparticles *Sp, MPI_Comm Communicator,
                                 long long dust_created,
                                 long long dust_destroyed,
                                 double   dust_mass_change)
{
  NDustCreated   += dust_created;
  NDustDestroyed += dust_destroyed;
  TotalDustMass  += dust_mass_change;

  LocalDustCreatedThisStep    = 0;
  LocalDustDestroyedThisStep  = 0;
  LocalDustMassChange         = 0.0;
  DustNeedsSynchronization    = 0;

  long long local_max_id = All.MaxID;
  MPI_Allreduce(&local_max_id, &All.MaxID, 1, MPI_LONG_LONG, MPI_MAX, Communicator);
}

/**
 * Spawn a single dust particle
 */
void spawn_dust_particle(simparticles *Sp, double offset_kpc[3], double dust_mass, 
                         double initial_velocity[3], int star_idx, int feedback_type)
{
  if(Sp->NumPart >= Sp->MaxPart) {
    static int warning_count = 0;
    if(warning_count < 10 && All.ThisTask == 0) {
      printf("[DUST_ERROR] T=%d: Cannot create dust - array full (NumPart=%d, MaxPart=%d)\n",
             All.ThisTask, Sp->NumPart, Sp->MaxPart);
      warning_count++;
    }
    return;
  }

  int new_idx = Sp->NumPart;
  
  double star_pos[3];
  Sp->intpos_to_pos(Sp->P[star_idx].IntPos, star_pos);
  
  double dust_pos[3];
  dust_pos[0] = star_pos[0] + offset_kpc[0];
  dust_pos[1] = star_pos[1] + offset_kpc[1];
  dust_pos[2] = star_pos[2] + offset_kpc[2];
  
  for(int i = 0; i < 3; i++) {
    while(dust_pos[i] < 0)
      dust_pos[i] += All.BoxSize;
    while(dust_pos[i] >= All.BoxSize)
      dust_pos[i] -= All.BoxSize;
  }
  
  Sp->pos_to_intpos(dust_pos, Sp->P[new_idx].IntPos);
  
  Sp->P[new_idx].setType(DUST_PARTICLE_TYPE);
  Sp->P[new_idx].setMass(dust_mass);
  Sp->P[new_idx].Metallicity = 1.0;

  Sp->P[new_idx].Vel[0] = initial_velocity[0];
  Sp->P[new_idx].Vel[1] = initial_velocity[1];
  Sp->P[new_idx].Vel[2] = initial_velocity[2];
  Sp->P[new_idx].ID.set(All.MaxID + 1);
  All.MaxID++;

  if(DUST_PARTICLE_TYPE < NTYPES && All.SofteningClassOfPartType[DUST_PARTICLE_TYPE] >= 0) {
    Sp->P[new_idx].setSofteningClass(All.SofteningClassOfPartType[DUST_PARTICLE_TYPE]);
  } else {
    Sp->P[new_idx].setSofteningClass(All.SofteningClassOfPartType[4]);
  }
  
  Sp->DustP[new_idx].GrainRadius = 10.0;
  Sp->DustP[new_idx].CarbonFraction = 0.3;
  Sp->DustP[new_idx].GrainType = 2;
  
  if(feedback_type == 1) {
    Sp->DustP[new_idx].DustTemperature = 1000.0;
  } else {
    Sp->DustP[new_idx].DustTemperature = 600.0;
  }

  double T_CMB = 2.7 * (1.0 / All.Time);
  if(Sp->DustP[new_idx].DustTemperature < T_CMB) {
    Sp->DustP[new_idx].DustTemperature = T_CMB;
  }

  Sp->P[new_idx].StellarAge = All.Time;
  Sp->P[new_idx].Ti_Current = All.Ti_Current;
  
  Sp->P[new_idx].TimeBinGrav = All.HighestActiveTimeBin;
  Sp->P[new_idx].TimeBinHydro = 0;
  Sp->TimeBinsGravity.timebin_add_particle(new_idx, star_idx, 
    All.HighestActiveTimeBin,
    Sp->TimeBinSynchronized[All.HighestActiveTimeBin]);

    // The dust can really slow down the sim if it ends up in very small timebins, so we set a floor here
    if(Sp->P[new_idx].getType() == DUST_PARTICLE_TYPE) {
        if(Sp->P[new_idx].TimeBinGrav < DUST_MIN_TIMEBIN)
            Sp->P[new_idx].TimeBinGrav = DUST_MIN_TIMEBIN;
    }


  if(Sp->DustP[new_idx].GrainRadius <= 0.0 || !isfinite(Sp->DustP[new_idx].GrainRadius)) {
    DUST_PRINT("[SPAWN_BUG] Just set GrainRadius but it's %.3e for new particle at idx=%d!\n",
               Sp->DustP[new_idx].GrainRadius, new_idx);
    Sp->DustP[new_idx].GrainRadius = 10.0;
  }

  Sp->NumPart++;
  GlobalDustCount++;
}

/**
 * Analyze grain size distribution
 */
void analyze_grain_size_distribution(simparticles *Sp)
{
  if(All.ThisTask != 0) return;
  
  const int NBINS = 6;
  double bin_edges[NBINS+1] = {0.0, 10.0, 50.0, 100.0, 150.0, 200.0, 500.0};
  int bin_counts[NBINS] = {0};
  double bin_masses[NBINS] = {0.0};
  
  int total_grains = 0;
  double total_mass = 0.0;
  
  for(int i = 0; i < Sp->NumPart; i++) {
    if(Sp->P[i].getType() == DUST_PARTICLE_TYPE && Sp->P[i].getMass() > 1e-20) {
      double mass = Sp->P[i].getMass();
      
      total_grains++;
      total_mass += mass;
      
      for(int b = 0; b < NBINS; b++) {
        if(Sp->DustP[i].GrainRadius >= bin_edges[b] && Sp->DustP[i].GrainRadius < bin_edges[b+1]) {
          bin_counts[b]++;
          bin_masses[b] += mass;
          break;
        }
      }
    }
  }
  
  if(total_grains == 0) return;
  
  DUST_PRINT("=== GRAIN SIZE DISTRIBUTION ===\n");
  DUST_PRINT("  Total: %d grains, %.3e Msun\n", total_grains, total_mass);
  
  for(int b = 0; b < NBINS; b++) {
    if(bin_counts[b] > 0) {
      double frac_num = (double)bin_counts[b] / total_grains;
      double frac_mass = bin_masses[b] / total_mass;
      DUST_PRINT("  [%.0f-%.0f nm]: %d grains (%.1f%%), %.2e Msun (%.1f%%)\n",
                 bin_edges[b], bin_edges[b+1], bin_counts[b], 
                 frac_num*100, bin_masses[b], frac_mass*100);
    }
  }
  DUST_PRINT("================================\n");
}

/**
 * Consume dust when gas forms stars (astration)
 */
void consume_dust_by_astration(simparticles *Sp, int gas_idx, double stellar_mass_formed, int star_idx, double hsml)
{
  if(!All.DustEnableAstration) return;
  
  double gas_mass = Sp->P[gas_idx].getMass();
  double sf_fraction;
  if(stellar_mass_formed >= gas_mass) {
      sf_fraction = 1.0;  // full conversion — all gas becomes star
  } else {
      sf_fraction = stellar_mass_formed / (gas_mass + stellar_mass_formed);
  }
  
  double search_radius = 3.0 * hsml;
  if(search_radius > 5.0) search_radius = 5.0;
  
  const int MAX_NEIGHBORS = 100;
  int neighbor_indices[MAX_NEIGHBORS];
  double neighbor_distances[MAX_NEIGHBORS];
  int n_neighbors = 0;
  
  dust_hash.find_neighbors(Sp, gas_idx, search_radius,
                          neighbor_indices, neighbor_distances,
                          &n_neighbors, MAX_NEIGHBORS);
  
  if(n_neighbors == 0) return;
  
  double total_dust_mass = 0.0;
  for(int i = 0; i < n_neighbors; i++) {
    int dust_idx = neighbor_indices[i];
    if(Sp->P[dust_idx].getType() == DUST_PARTICLE_TYPE) {
      total_dust_mass += Sp->P[dust_idx].getMass();
    }
  }
  
  if(total_dust_mass < 1e-20) return;
  
  double dust_to_consume = total_dust_mass * sf_fraction;
  
  double weight_sum = 0.0;
  for(int i = 0; i < n_neighbors; i++) {
    if(neighbor_distances[i] > 0) {
      weight_sum += 1.0 / neighbor_distances[i];
    }
  }
  
  int dust_consumed_count = 0;
  double dust_consumed_mass = 0.0;
  
  for(int i = 0; i < n_neighbors; i++) {
    int dust_idx = neighbor_indices[i];
    if(Sp->P[dust_idx].getType() != DUST_PARTICLE_TYPE) continue;
    
    double weight = (neighbor_distances[i] > 0) ? (1.0 / neighbor_distances[i]) : 1.0;
    double this_dust_fraction = weight / weight_sum;
    double mass_loss = dust_to_consume * this_dust_fraction;
    
    double current_mass = Sp->P[dust_idx].getMass();
    double new_mass = current_mass - mass_loss;
    
    if(new_mass < 1e-20) {
      Sp->P[dust_idx].setMass(1e-30);
      Sp->P[dust_idx].setType(3);
      Sp->P[dust_idx].ID.set(0);
      memset(&Sp->DustP[dust_idx], 0, sizeof(dust_data));
      dust_consumed_count++;
      dust_consumed_mass += current_mass;
    } else {
      Sp->P[dust_idx].setMass(new_mass);
      dust_consumed_mass += mass_loss;
    }
  }
  
  NDustDestroyedByAstration += dust_consumed_count;
  TotalDustMassAstrated += dust_consumed_mass;
  
  static int astration_count = 0;
  astration_count++;
  if(astration_count % 100 == 0 && All.ThisTask == 0) {
    DUST_PRINT("[ASTRATION] Event #%d: SF=%.2e Msun, consumed %d dust (%.2e Msun)\n",
               astration_count, stellar_mass_formed, dust_consumed_count, dust_consumed_mass);
  }

  // At the end, after dust_consumed_mass is tallied:
  if(star_idx >= 0 && dust_consumed_mass > 0) {
    double star_mass = Sp->P[star_idx].getMass();
    if(star_mass > 0)
      Sp->P[star_idx].Metallicity += dust_consumed_mass / star_mass;
  }

}


// *********************************************************************
// RADIATION PRESSURE
double radiation_pressure_efficiency(double a_nm, double carbon_fraction)
{
    const double a0_nm = 16.0;
    double Q_pr;

    if(a_nm < a0_nm) {
        Q_pr = a_nm / a0_nm;
    } else {
        Q_pr = 1.0;
    }

    double species_factor = 0.5 + carbon_fraction;
    Q_pr *= species_factor;

    if(Q_pr > 2.0) Q_pr = 2.0;
    if(Q_pr < 0.0) Q_pr = 0.0;

    return Q_pr;
}

/**
 * Estimate stellar luminosity from mass and age for radiation pressure calculation.
 * NOTE: Simplified placeholder — see comments for proper treatment using MESA/Starburst99 tables.
 */
double stellar_luminosity(simparticles *Sp, int star_idx)
{
  const double L_sun_cgs  = 3.828e33;
  const double M_sun_cgs  = 1.989e33;

  double M_star_cgs = Sp->P[star_idx].getMass() * All.UnitMass_in_g;
  double M_star_sun = M_star_cgs / M_sun_cgs;

  double stellar_age_yr = (All.Time - Sp->P[star_idx].StellarAge)
                          * All.UnitTime_in_s / SEC_PER_YEAR;

  double L_over_M;

  if(stellar_age_yr < 3e6) {
    L_over_M = 1000.0;
  } else if(stellar_age_yr < 10e6) {
    double frac = (stellar_age_yr - 3e6) / 7e6;
    L_over_M = 1000.0 - frac * 500.0;
  } else if(stellar_age_yr < 40e6) {
    double frac = (stellar_age_yr - 10e6) / 30e6;
    L_over_M = 500.0 - frac * 400.0;
  } else {
    double frac = (stellar_age_yr - 40e6) / 60e6;
    L_over_M = 100.0 - frac * 90.0;
  }

  if(L_over_M < 0.0) L_over_M = 0.0;

  return L_over_M * M_star_sun * L_sun_cgs;
}

void dust_radiation_pressure(simparticles *Sp, int dust_idx, int nearest_gas, double dt)
{
  if(!All.DustEnableRadiationPressure) return;

  double search_radius = (nearest_gas >= 0) ? 2.0 * Sp->SphP[nearest_gas].Hsml : 2.0;
  if(search_radius > 5.0) search_radius = 5.0;

  const int MAX_STAR_NEIGHBORS = 50;
  int   neighbor_indices[MAX_STAR_NEIGHBORS];
  double neighbor_distances[MAX_STAR_NEIGHBORS];
  int n_neighbors = 0;

  star_hash.find_neighbors(Sp, dust_idx, search_radius,
                           neighbor_indices, neighbor_distances,
                           &n_neighbors, MAX_STAR_NEIGHBORS);

  if(n_neighbors == 0) return;

  double a_nm  = Sp->DustP[dust_idx].GrainRadius;
  double a_cm  = a_nm * 1e-7;
  double CF    = Sp->DustP[dust_idx].CarbonFraction;
  double Q_pr  = radiation_pressure_efficiency(a_nm, CF);

  double grain_mass_cgs = Sp->P[dust_idx].getMass() * All.UnitMass_in_g;

  const double c_cgs = 2.998e10;

  double a_rad[3] = {0.0, 0.0, 0.0};

  for(int i = 0; i < n_neighbors; i++) {
    int star_idx = neighbor_indices[i];

    if(Sp->P[star_idx].getType() != 4) continue;

    double stellar_age_yr = (All.Time - Sp->P[star_idx].StellarAge)
                            * All.UnitTime_in_s / SEC_PER_YEAR;
    if(stellar_age_yr > 100e6) continue;

    double L_cgs = stellar_luminosity(Sp, star_idx);
    if(L_cgs <= 0.0) continue;

    double r_kpc = neighbor_distances[i];
    double r_cgs = r_kpc * 3.086e21;
    if(r_cgs <= 0.0) continue;

    double dxyz[3];
    Sp->nearest_image_intpos_to_pos(Sp->P[dust_idx].IntPos,
                                    Sp->P[star_idx].IntPos, dxyz);
    double r_code = neighbor_distances[i];
    double unit_vec[3];
    for(int k = 0; k < 3; k++)
      unit_vec[k] = dxyz[k] / r_code;

    double flux_cgs = L_cgs / (4.0 * M_PI * r_cgs * r_cgs * c_cgs);
    double accel    = flux_cgs * Q_pr * M_PI * a_cm * a_cm / grain_mass_cgs;

    for(int k = 0; k < 3; k++)
      a_rad[k] += accel * unit_vec[k];
  }

  double accel_code = All.UnitVelocity_in_cm_per_s / All.UnitTime_in_s;
  for(int k = 0; k < 3; k++)
    Sp->P[dust_idx].Vel[k] += (a_rad[k] / accel_code) * dt;
}


/**
 * Update dust particle dynamics (drag, grain growth, coagulation).
 *
 * Called every timestep but only does real work every 10 steps.
 */
void update_dust_dynamics(simparticles *Sp, double dt, MPI_Comm Communicator)
{

// TEMPORARY!
// I have a situation where I suspect some dust particles are getting stuck on very small timebins (<=13)
// and not evolving properly. This debug block will print out the ID, timebin, velocity, and position of 
// any such particles every 100 steps to help diagnose the issue. Once we confirm whether this is happening 
// or not, we can remove this block.
if(All.NumCurrentTiStep % 100 == 0 && All.ThisTask == 0) {
    for(int i = 0; i < Sp->NumPart; i++) {
        if(Sp->P[i].getType() == DUST_PARTICLE_TYPE && 
           Sp->P[i].TimeBinGrav <= 13) {
            double pos[3];
            Sp->intpos_to_pos(Sp->P[i].IntPos, pos);
            double vel = sqrt(Sp->P[i].Vel[0]*Sp->P[i].Vel[0] + 
                             Sp->P[i].Vel[1]*Sp->P[i].Vel[1] + 
                             Sp->P[i].Vel[2]*Sp->P[i].Vel[2]);
            printf("[SMALL_TIMEBIN_DUST] ID=%lld bin=%d vel=%.1f km/s pos=(%.1f,%.1f,%.1f)\n",
                   (long long)Sp->P[i].ID.get(), Sp->P[i].TimeBinGrav,
                   vel * All.UnitVelocity_in_cm_per_s/1e5,
                   pos[0], pos[1], pos[2]);
        }
    }
}


  // ============================================================
  // NEW: One-time flag verification at simulation start.
  // Confirms the parameter file is being read correctly.
  // ============================================================
  static bool flags_printed = false;
  if(!flags_printed && All.ThisTask == 0) {
    printf("[DUST_FLAGS|Step=%d] Creation=%d Drag=%d Growth=%d Coagulation=%d "
           "Sputtering=%d ShockDestruction=%d Astration=%d RadPressure=%d Clumping=%d\n",
           All.NumCurrentTiStep,
           All.DustEnableCreation, All.DustEnableDrag,
           All.DustEnableGrowth,   All.DustEnableCoagulation,
           All.DustEnableSputtering, All.DustEnableShockDestruction,
           All.DustEnableAstration,  All.DustEnableRadiationPressure,
           All.DustEnableClumping);
    flags_printed = true;
  }

  // ============================================================
  // Cadence guard — skip 9 out of 10 steps entirely, no MPI
  // ============================================================
  if(All.NumCurrentTiStep % 10 != 0)
    return;

  // ============================================================
  // Collective check: any dust exist across all tasks?
  // ============================================================
  long long local_count  = GlobalDustCount;
  long long global_count = 0;
  MPI_Allreduce(&local_count, &global_count, 1, MPI_LONG_LONG, MPI_SUM, Communicator);

  if(global_count == 0)
    return;

  // ============================================================
  // Timing and setup
  // ============================================================
  static double total_time_in_dust = 0.0;
  static int    dust_call_count    = 0;
  double t_start = MPI_Wtime();

  // ============================================================
  // Ensure spatial hash is built (needed for neighbor finding)
  // ============================================================
  int need_hash_rebuild = 0;
  if(All.ThisTask == 0)
    need_hash_rebuild = !gas_hash.is_built;
  MPI_Bcast(&need_hash_rebuild, 1, MPI_INT, 0, Communicator);

  if(need_hash_rebuild) {
    if(All.ThisTask == 0)
      DUST_PRINT("WARNING: Hash not built, building now for dust operations\n");
    rebuild_feedback_spatial_hash(Sp, 10.0, Communicator);
  }

  // ============================================================
  // One-time hash verification (task 0 only)
  // ============================================================
  static bool verified = false;
  if(!verified && All.ThisTask == 0) {
    DUST_PRINT("=== HASH VERIFICATION ===\n");
    DUST_PRINT("  Hash built:    %s\n",   gas_hash.is_built ? "YES" : "NO");
    DUST_PRINT("  Cells per dim: %d\n",   gas_hash.n_cells_per_dim);
    DUST_PRINT("  Cell size:     %.3f kpc\n", gas_hash.cell_size);
    DUST_PRINT("  Gas particles: %d\n",   gas_hash.total_particles);
    DUST_PRINT("=========================\n");
    verified = true;
  }

  // ============================================================
  // Main dust dynamics loop
  // ============================================================
  for(int i = 0; i < Sp->NumPart; i++) {
    if(Sp->P[i].getType() != DUST_PARTICLE_TYPE) continue;
    if(Sp->P[i].getMass() <= 1e-20) continue;

    // Enforce minimum gravity timebin floor
    // without this, the dust can get to really low timebins and slow down the entire sim
    if(Sp->P[i].TimeBinGrav < DUST_MIN_TIMEBIN) {
        Sp->TimeBinsGravity.timebin_move_particle(i, 
            Sp->P[i].TimeBinGrav, DUST_MIN_TIMEBIN);
        Sp->P[i].TimeBinGrav = DUST_MIN_TIMEBIN;
    }

    double dist_kpc  = -1.0;
    int nearest_gas  = find_nearest_gas_particle(Sp, i, 5.0, &dist_kpc);
    if(nearest_gas < 0) continue;

    dust_gas_interaction(Sp, i, nearest_gas, dt * 10);
    dust_radiation_pressure(Sp, i, nearest_gas, dt * 10);
    
    if(dist_kpc <= 2.0) {
      if(All.DustEnableGrowth)
        dust_grain_growth_subgrid(Sp, i, nearest_gas, dt * 10);

      if(All.DustEnableCoagulation)
        dust_grain_coagulation(Sp, i, nearest_gas, dt * 10);
    }
  }

  // ============================================================
  // Periodic diagnostics (every 500 steps, task 0 only)
  // ============================================================
  if(All.NumCurrentTiStep % 500 == 0) {
    print_dust_statistics(Sp);
    analyze_dust_gas_coupling_global(Sp);
    analyze_grain_size_distribution(Sp);

    if(All.ThisTask == 0) {
      DUST_PRINT("[GROWTH_SUMMARY] Total growth events so far: %lld\n", NGrainGrowthEvents);

      double M_cold = 0, M_warm = 0, M_hot = 0, M_vhot = 0;
      for(int i = 0; i < Sp->NumGas; i++) {
        double T = get_temperature_from_entropy(Sp, i);
        double M = Sp->P[i].getMass();
        if     (T < 1e4) M_cold += M;
        else if(T < 1e5) M_warm += M;
        else if(T < 1e6) M_hot  += M;
        else             M_vhot += M;
      }
      double M_tot = M_cold + M_warm + M_hot + M_vhot;
      DUST_PRINT("[GAS_BUDGET] cold <10^4=%.1f%%  warm <10^5=%.1f%%  hot <10^6=%.1f%%  vhot=%.1f%%\n",
                 100*M_cold/M_tot, 100*M_warm/M_tot,
                 100*M_hot/M_tot,  100*M_vhot/M_tot);

      int printed = 0;
      for(int i = 0; i < Sp->NumPart && printed < 3; i++) {
        if(Sp->P[i].getType() != DUST_PARTICLE_TYPE || Sp->P[i].getMass() <= 1e-20) continue;
        DUST_PRINT("[DUST_SAMPLE] i=%d ID=%lld M=%.2e a=%.2f nm CF=%.2f GT=%d\n",
                   i, (long long)Sp->P[i].ID.get(), Sp->P[i].getMass(),
                   Sp->DustP[i].GrainRadius, Sp->DustP[i].CarbonFraction,
                   Sp->DustP[i].GrainType);
        printed++;
      }
    }
  }

  // ============================================================
  // Timing summary (every 100 calls)
  // ============================================================
  double dt_dust = MPI_Wtime() - t_start;
  total_time_in_dust += dt_dust;
  dust_call_count++;

  if(dust_call_count % 100 == 0 && All.ThisTask == 0) {
    printf("[DUST_TIMING] Called %d times, avg %.3f sec/call, total %.1f sec\n",
           dust_call_count, total_time_in_dust / dust_call_count, total_time_in_dust);
  }
}

/**
 * Remove destroyed dust particles and compact array.
 *
 * NOTE: This function's comment says "POTENTIALLY NO LONGER USING THIS".
 * It is kept here but a call-count diagnostic has been added to confirm
 * whether it is actually being invoked. If it is never called, it can be
 * safely removed. Do NOT remove the function silently — verify first.
 */
void destroy_dust_particles(simparticles *Sp)
{
  // NEW: Confirm this function is still being called
  static long long destroy_call_count = 0;
  destroy_call_count++;
  printf("[DESTROY_DUST_CALLED|T=%d|Step=%d] Call #%lld, scanning %d particles\n",
         All.ThisTask, All.NumCurrentTiStep, destroy_call_count, Sp->NumPart);

  int dust_destroyed = 0;
  
  for(int i = 0; i < Sp->NumPart; i++) {
    if((Sp->P[i].getType() == DUST_PARTICLE_TYPE || Sp->P[i].getType() == 3) && 
      (Sp->P[i].getMass() < 1e-20 || Sp->P[i].ID.get() == 0)) {
      dust_destroyed++;
    }
  }
  
  if(dust_destroyed == 0) {
    return;
  }
  
  int new_num_gas = 0;
  for(int i = 0; i < Sp->NumGas; i++) {
    if(Sp->P[i].getType() == 0 && Sp->P[i].getMass() > 0.0 && Sp->P[i].ID.get() != 0) {
      if(new_num_gas != i) {
        Sp->P[new_num_gas] = Sp->P[i];
        Sp->SphP[new_num_gas] = Sp->SphP[i];
        #ifdef DUST
        memset(&Sp->DustP[new_num_gas], 0, sizeof(dust_data));
        #endif
      } else {
        #ifdef DUST
        memset(&Sp->DustP[new_num_gas], 0, sizeof(dust_data));
        #endif
      }
      new_num_gas++;
    }
  }
  
  int new_num_part = new_num_gas;
  for(int i = Sp->NumGas; i < Sp->NumPart; i++) {
    if((Sp->P[i].getType() == DUST_PARTICLE_TYPE || Sp->P[i].getType() == 3) && 
      (Sp->P[i].getMass() < 1e-20 || Sp->P[i].ID.get() == 0)) {
      continue;
    }
    
    if(new_num_part != i) {
      Sp->P[new_num_part] = Sp->P[i];
    }

    if(Sp->P[i].getType() == DUST_PARTICLE_TYPE) {
      if(new_num_part != i) {
        Sp->DustP[new_num_part] = Sp->DustP[i];
      }
    } else {
      memset(&Sp->DustP[new_num_part], 0, sizeof(dust_data));
    }

    new_num_part++;
  }
  
  Sp->NumPart = new_num_part;
  Sp->NumGas = new_num_gas;
}

/**
 * Find nearest gas particle to dust using spatial hash.
 */
int find_nearest_gas_particle(simparticles *Sp, int dust_idx,
                              double max_r_kpc, double *out_dist_kpc)
{
  if(out_dist_kpc) *out_dist_kpc = -1.0;

  if(Sp->NumGas == 0) return -1;
  if(max_r_kpc <= 0)  return -1;

  if(gas_hash.is_built) {
    HashSearches++;

    double nearest_dist = -1.0;
    int nearest = gas_hash.find_nearest_particle(Sp, dust_idx, max_r_kpc, &nearest_dist);

    if(nearest < 0 || !(nearest_dist >= 0) || nearest_dist > max_r_kpc) {
      HashSearchesFailed++;
      return -1;
    }

    // Type guard: stale hash may return a converted star (type 4) instead of gas
    if(Sp->P[nearest].getType() != 0) {
      HashSearchesFailed++;
      return -1;
    }

    if(out_dist_kpc) *out_dist_kpc = nearest_dist;
    return nearest;
  }

  return -1;
}

/**
 * Calculate velocity difference between dust and gas particles
 */
double calculate_velocity_difference(simparticles *Sp, int dust_idx, int gas_idx)
{
  double dv[3];
  dv[0] = Sp->P[dust_idx].Vel[0] - Sp->P[gas_idx].Vel[0];
  dv[1] = Sp->P[dust_idx].Vel[1] - Sp->P[gas_idx].Vel[1];
  dv[2] = Sp->P[dust_idx].Vel[2] - Sp->P[gas_idx].Vel[2];
  
  return sqrt(dv[0]*dv[0] + dv[1]*dv[1] + dv[2]*dv[2]);
}

/**
 * Print dust statistics, including full per-pathway destruction audit.
 */
void print_dust_statistics(simparticles *Sp)
{
  if(All.ThisTask != 0) return;
  
  int dust_count = 0;
  double total_dust_mass = 0.0;
  double avg_grain_size = 0.0;
  double avg_temperature = 0.0;
  
  for(int i = 0; i < Sp->NumPart; i++) {
    if(Sp->P[i].getType() == DUST_PARTICLE_TYPE && Sp->P[i].getMass() > 1e-20) {
      dust_count++;
      total_dust_mass += Sp->P[i].getMass();
      avg_grain_size += Sp->DustP[i].GrainRadius;
      avg_temperature += Sp->DustP[i].DustTemperature;
    }
  }
  
  if(dust_count > 0) {
    avg_grain_size /= dust_count;
    avg_temperature /= dust_count;
  }

  int cmb_floor = 0;
  int very_cold = 0;
  int cold_ism = 0;
  int warm_ism = 0;
  int hot_ism = 0;
  int pre_sublimation = 0;

  for(int i = 0; i < Sp->NumPart; i++) {
    if(Sp->P[i].getType() == DUST_PARTICLE_TYPE && Sp->P[i].getMass() > 1e-20) {
      double T = Sp->DustP[i].DustTemperature;
      
      if(T < 10.0) cmb_floor++;
      else if(T < 50.0) very_cold++;
      else if(T < 100.0) cold_ism++;
      else if(T < 500.0) warm_ism++;
      else if(T < 1000.0) hot_ism++;
      else pre_sublimation++;
    }
  }
  
  DUST_PRINT("=== STATISTICS (rank 0) ===\n");
  DUST_PRINT("STATISTICS Particles: %d  Mass: %.3e Msun\n", dust_count, total_dust_mass);
  DUST_PRINT("STATISTICS Avg grain size: %.2f nm\n", avg_grain_size);
  DUST_PRINT("STATISTICS Avg temperature: %.1f K\n", avg_temperature);
  if(dust_count > 0) {
    DUST_PRINT("STATISTICS  < 10 K (CMB floor):        %d (%.1f%%)\n", cmb_floor, 100.0*cmb_floor/dust_count);
    DUST_PRINT("STATISTICS  10-50 K (Cold clouds):     %d (%.1f%%)\n", very_cold, 100.0*very_cold/dust_count);
    DUST_PRINT("STATISTICS  50-100 K (Cool ISM):       %d (%.1f%%)\n", cold_ism, 100.0*cold_ism/dust_count);
    DUST_PRINT("STATISTICS  100-500 K (Warm ISM):      %d (%.1f%%)\n", warm_ism, 100.0*warm_ism/dust_count);
    DUST_PRINT("STATISTICS  500-1000 K (Hot ISM):      %d (%.1f%%)\n", hot_ism, 100.0*hot_ism/dust_count);
    DUST_PRINT("STATISTICS  1000-2000 K (Near sublim): %d (%.1f%%)\n", pre_sublimation, 100.0*pre_sublimation/dust_count);
  }
  DUST_PRINT("========================\n");
  DUST_PRINT("STATISTICS Hash searches:       %lld\n", HashSearches);
  if(HashSearches > 0) {
    DUST_PRINT("STATISTICS Hash success rate:    %.1f%%\n", 
               100.0 * (HashSearches - HashSearchesFailed) / HashSearches);
  }
  if(HashSearchesFailed > 0) {
    DUST_PRINT("STATISTICS [WARNING] Failed searches: %lld (%.1f%%)\n",
               HashSearchesFailed, 100.0 * HashSearchesFailed / HashSearches);
  }
  DUST_PRINT("STATISTICS Growth events: %lld (%.2e Msun grown)\n", NGrainGrowthEvents, TotalMassGrown);
  DUST_PRINT("STATISTICS Partial erosion events: %lld\n", NGrainErosionEvents);
  DUST_PRINT("========================\n");

  // -----------------------------------------------------------------------
  // NEW: Full destruction audit — every deletion pathway in one place.
  // If physics flags are all OFF, the only non-zero entries should be
  // Cleanup, Corruption, and BadGasIndex. Any non-zero physics entries
  // while flags are OFF means a flag check is missing somewhere.
  // -----------------------------------------------------------------------
  long long total_by_physics  = NDustDestroyedByThermal + NDustDestroyedByShock + NDustDestroyedByAstration;
  long long total_by_internal = NDustDestroyedByCleanup + NDustDestroyedByCorruption + NDustDestroyedByBadGasIndex;
  long long total_destroyed   = total_by_physics + total_by_internal;

  DUST_PRINT("=== DESTRUCTION AUDIT (cumulative) ===\n");
  DUST_PRINT("  --- Physics mechanisms (gated by enable flags) ---\n");
  DUST_PRINT("  Thermal sputtering:     %lld  (flag=%d)\n", NDustDestroyedByThermal,   All.DustEnableSputtering);
  DUST_PRINT("  Shock destruction:      %lld  (flag=%d)\n", NDustDestroyedByShock,     All.DustEnableShockDestruction);
  DUST_PRINT("  Astration:              %lld  (flag=%d, mass=%.2e Msun)\n",
             NDustDestroyedByAstration, All.DustEnableAstration, TotalDustMassAstrated);
  DUST_PRINT("  --- Internal / unconditional paths ---\n");
  DUST_PRINT("  cleanup_invalid():      %lld  ← domain exchange corruption?\n", NDustDestroyedByCleanup);
  DUST_PRINT("  growth corruption:      %lld  ← zeroed DustP state in growth loop\n", NDustDestroyedByCorruption);
  DUST_PRINT("  bad gas index (stale):  %lld  ← hash returned non-gas type\n", NDustDestroyedByBadGasIndex);
  DUST_PRINT("  --- Totals ---\n");
  DUST_PRINT("  By physics:             %lld\n", total_by_physics);
  DUST_PRINT("  By internal paths:      %lld  ← THIS should be ~0 if all flags OFF\n", total_by_internal);
  DUST_PRINT("  GRAND TOTAL destroyed:  %lld\n", total_destroyed);
  DUST_PRINT("  Total created:          %lld\n", NDustCreated);
  DUST_PRINT("  Net (created-destroyed):%lld\n", NDustCreated - total_destroyed);
  DUST_PRINT("  Current live particles: %d\n",   dust_count);
  DUST_PRINT("=======================================\n");

  DUST_PRINT("STATISTICS Coagulation events: %lld (%.2e Msun coagulated)\n",
           NCoagulationEvents, TotalMassCoagulated);
  DUST_PRINT("========================\n");
}

void analyze_dust_gas_coupling_global(simparticles *Sp)
{
  if(All.ThisTask != 0) return;
  
  double total_vel_diff = 0.0;
  int dust_count = 0;
  double max_vel_diff = 0.0;
    
  for(int i = 0; i < Sp->NumPart; i++) {
    if(Sp->P[i].getType() == DUST_PARTICLE_TYPE && Sp->P[i].getMass() > 1e-20) {
      int nearest_gas = find_nearest_gas_particle(Sp, i, 5.0, NULL);
      if(nearest_gas >= 0) {
        double vel_diff = calculate_velocity_difference(Sp, i, nearest_gas);
        total_vel_diff += vel_diff;
        if(vel_diff > max_vel_diff) max_vel_diff = vel_diff;
        dust_count++;
      }
    }
  }
    
  if(dust_count > 0) {
    DUST_PRINT("[COUPLING] %d particles, avg Δv=%.3f, max=%.3f (Rank 0)\n",
               dust_count, total_vel_diff/dust_count, max_vel_diff);
  }
}

/**
 * Calculate supernova shock radius using Sedov-Taylor solution
 */
double calculate_sn_shock_radius(double sn_energy_erg, double gas_density_cgs, double time_myr)
{
  const double xi = 1.033;
    
  double time_sec = time_myr * 1e6 * SEC_PER_YEAR;
  double radius_cm = xi * pow((sn_energy_erg * time_sec * time_sec / gas_density_cgs), 0.2);
  double radius_kpc = radius_cm / (1000.0 * PARSEC);
    
  return radius_kpc;
}

/**
 * Calculate current shock radius for a SN
 */
double calculate_current_sn_shock_radius(simparticles *Sp, int sn_star_idx)
{
  const double sn_energy_erg = 1e51;
  
  double nearest_dist = -1.0;
  int nearest_gas = gas_hash.find_nearest_particle(Sp, sn_star_idx, 50.0, &nearest_dist);

  // Guard: hash may return a stale index that now points to a star (type 4) after
  // gas→star conversion.  Reading SphP[] for a non-gas particle is UB.
  if(nearest_gas >= 0 && Sp->P[nearest_gas].getType() != 0)
    nearest_gas = -1;

  double gas_density_cgs = 1.0 * PROTONMASS;

  if(nearest_gas >= 0 && nearest_gas < Sp->NumGas) {
    double gas_density_code = Sp->SphP[nearest_gas].Density * All.cf_a3inv;
    double measured = gas_density_code * All.UnitDensity_in_cgs;
    
    if(measured > 0.01 * PROTONMASS) {
      gas_density_cgs = measured;
    }
  }
  
  double characteristic_time_myr = 3.0;
  
  double radius = calculate_sn_shock_radius(sn_energy_erg, gas_density_cgs, 
                                           characteristic_time_myr);
  
  if(radius < 1.0) radius = 1.0;
  if(radius > 10.0) radius = 10.0;
  
  return radius;
}

/**
 * Calculate shock velocity
 */
double calculate_shock_velocity(double sn_energy_erg, double gas_density_cgs, double time_myr)
{
  double radius_cm = calculate_sn_shock_radius(sn_energy_erg, gas_density_cgs, time_myr) * 1000.0 * PARSEC;
  double time_sec  = time_myr * 1e6 * SEC_PER_YEAR;
    
  double velocity_cm_per_s = (2.0/5.0) * radius_cm / time_sec;
    
  return velocity_cm_per_s / 1e5;
}

  double get_shock_destruction_efficiency(double shock_velocity_km_s)
  {
    if(shock_velocity_km_s < 50.0) {
      return 0.0;
    }
    else if(shock_velocity_km_s < 100.0) {
      return 0.15 * (shock_velocity_km_s - 50.0) / 50.0;
    }
    else if(shock_velocity_km_s < 200.0) {
      return 0.15 + 0.35 * (shock_velocity_km_s - 100.0) / 100.0;
    }
    else if(shock_velocity_km_s < 400.0) {
      return 0.50 + 0.30 * (shock_velocity_km_s - 200.0) / 200.0;
    }
    else if(shock_velocity_km_s < 500.0) {
      return 0.80 + 0.15 * (shock_velocity_km_s - 400.0) / 100.0;
    }
    else {
      return 0.95;
    }
  }

/**
 * Main dust destruction function from SN shocks
 */
void destroy_dust_from_sn_shocks(simparticles *Sp, int sn_star_idx, 
                                 double sn_energy, double metals_produced)
{
  if(!All.DustEnableShockDestruction) return;
  
  static int sn_call_count = 0;
  sn_call_count++;
  
  double shock_radius_kpc = calculate_current_sn_shock_radius(Sp, sn_star_idx);
  if(shock_radius_kpc > 5.0) shock_radius_kpc = 5.0;

  const double characteristic_time_myr = 3.0;
  const double time_sec = characteristic_time_myr * 1e6 * SEC_PER_YEAR;
  const double radius_cm = shock_radius_kpc * 1000.0 * PARSEC;
  
  double shock_velocity = (2.0/5.0) * radius_cm / time_sec;
  shock_velocity /= 1e5;

  double nearest_dist = -1.0;
  int nearest_gas = gas_hash.find_nearest_particle(Sp, sn_star_idx, 50.0, &nearest_dist);

  // Guard stale hash — this result is only used for the debug distance print,
  // but guard it anyway to avoid any future accidental SphP reads.
  if(nearest_gas >= 0 && Sp->P[nearest_gas].getType() != 0) {
    nearest_gas = -1;
    nearest_dist = -1.0;
  }

  if(sn_call_count <= 10 && All.ThisTask == 0) {
    DUST_PRINT("[SN_SHOCK_DEBUG] Call #%d: shock_radius=%.2f kpc, shock_velocity=%.1f km/s, nearest_gas_dist=%.2f kpc\n",
               sn_call_count, shock_radius_kpc, shock_velocity, nearest_dist);
  }

  int dust_in_shock  = 0;
  int dust_destroyed = 0;
  int dust_eroded    = 0;
  
  int neighbors[2048];
  double distances[2048];
  int n_found = 0;

  dust_hash.find_neighbors(Sp, sn_star_idx, shock_radius_kpc,
                          neighbors, distances, &n_found, 2048);

  for(int k = 0; k < n_found; k++) {
      int i = neighbors[k];
      if(Sp->P[i].getType() != DUST_PARTICLE_TYPE) continue;
      if(Sp->P[i].getMass() <= 1e-20) continue;
      
      dust_in_shock++;
      int destroyed = erode_dust_grain_shock(Sp, i, shock_velocity,
                                            distances[k], shock_radius_kpc);
      if(destroyed) dust_destroyed++;
      else dust_eroded++;
  }
  
  if(All.ThisTask == 0 && dust_in_shock > 0) {
    DUST_PRINT("[DUST_SN] SN affected %d dust: %d destroyed, %d eroded\n",
               dust_in_shock, dust_destroyed, dust_eroded);
  }

  if(sn_call_count <= 10 && All.ThisTask == 0) {
    DUST_PRINT("[SN_SHOCK_RESULT] Dust found in shock: %d, destroyed: %d, eroded: %d\n",
               dust_in_shock, dust_destroyed, dust_eroded);
  }
}

/**
 * Find nearest dust particle to a gas cell (O(N) brute force).
 *
 * NOTE: This function appears to have no call sites in the current codebase
 * and was superseded by the hash-based find_nearest_gas_particle approach.
 * Marked for removal — verify it is not called from any other translation
 * unit before deleting.
 */
int find_nearest_dust_particle(simparticles *Sp, int gas_idx)
{
  double min_dist2 = 1e30;
  int nearest_dust = -1;
  
  double search_radius_kpc = 10.0;
  
  for(int i = 0; i < Sp->NumPart; i++) {
    if(Sp->P[i].getType() == DUST_PARTICLE_TYPE && 
       Sp->P[i].getMass() > 1e-20) {
      
      double dxyz[3];
      Sp->nearest_image_intpos_to_pos(Sp->P[gas_idx].IntPos, 
                                       Sp->P[i].IntPos, dxyz);
      
      double r2 = dxyz[0]*dxyz[0] + dxyz[1]*dxyz[1] + dxyz[2]*dxyz[2];
      
      if(r2 > search_radius_kpc * search_radius_kpc) continue;
      
      if(r2 < min_dist2) {
        min_dist2 = r2;
        nearest_dust = i;
      }
    }
  }
  
  return nearest_dust;
}



/* For grain growth implementation from Hirashita & Kuo 2011 */
static inline double tau_acc_yr_HK11(double nH_cm3, double T_K,
                                     double Z_massfrac, double Zsun_massfrac,
                                     double a_cm, double S,
                                     int species /*0=sil,1=carb*/)
{
  const double pref = (species==0) ? 6.30e7 : 5.59e7;

  if(nH_cm3 <= 0 || T_K <= 0 || Z_massfrac <= 0 || Zsun_massfrac <= 0 || a_cm <= 0 || S <= 0)
    return HUGE_VAL;

  const double a_um = a_cm * 1e4;
  const double a01  = a_um / 0.1;
  const double n3   = nH_cm3 / 1e3;
  const double T50  = T_K / 50.0;
  const double Zrat = Z_massfrac / Zsun_massfrac;
  const double S03  = S / 0.3;

  return pref * a01 * (1.0/Zrat) * (1.0/n3) * pow(T50, -0.5) * (1.0/S03);
}

double dust_clumping_factor(double n_H, int is_star_forming)
{
    if(!All.DustEnableClumping) return 1.0;
    
    if(is_star_forming) return 30.0;
    
    const double n_sf = All.CritPhysDensity;
    
    if(n_H > 0.5 * n_sf)  return 10.0;
    if(n_H > 0.2 * n_sf)  return 3.0;
    if(n_H > 0.05 * n_sf) return 1.5;
    
    return 1.0;
}

/**
 * Dust grain coagulation in dense gas
 */
void dust_grain_coagulation(simparticles *Sp, int dust_idx, int gas_idx, double dt)
{
  if(!All.DustEnableCoagulation) return;
  
  double gas_density_code = Sp->SphP[gas_idx].Density * All.cf_a3inv;
  double gas_density_cgs  = gas_density_code * All.UnitDensity_in_cgs;
  double n_H              = (gas_density_cgs * HYDROGEN_MASSFRAC) / PROTONMASS;

  double DustClumpingFactor     = dust_clumping_factor(n_H, Sp->SphP[gas_idx].Sfr > DUST_SFR_EPS);
  double n_eff                  = n_H * DustClumpingFactor;

  if(n_eff < All.DustCoagulationDensityThresh) return;
  
  double T_gas = get_temperature_from_entropy(Sp, gas_idx);
  
  if(T_gas > 100.0) return; // this is a rough molecular cloud filter
  
  double a = Sp->DustP[dust_idx].GrainRadius;
  double M_dust = Sp->P[dust_idx].getMass();
  
  if(a <= 0.0 || M_dust <= 0.0 || !isfinite(a) || !isfinite(M_dust)) return;
  
  if(a >= All.DustCoagulationMaxSize) return;
  
  double a_micron = a / 1000.0;
  
  double tau_coag_yr = 1e7 * (100.0 / n_eff) * (0.1 / a_micron);
  tau_coag_yr *= All.DustCoagulationCalibration;
  
  if(tau_coag_yr < 1e6)  tau_coag_yr = 1e6;
  if(tau_coag_yr > 1e9)  tau_coag_yr = 1e9;
  
  double tau_coag = tau_coag_yr * SEC_PER_YEAR / All.UnitTime_in_s;
  
  double growth_factor = 1.0 + (dt / tau_coag);
  if(growth_factor > 1.2) growth_factor = 1.2;
  
  double M_new = M_dust * growth_factor;
  double dM = M_new - M_dust;
  
  if(!isfinite(dM) || dM <= 0.0) return;
  
  double size_ratio = pow(M_new / M_dust, 1.0/3.0);
  double a_new = a * size_ratio;
  
  if(a_new > All.DustCoagulationMaxSize) {
    a_new = All.DustCoagulationMaxSize;
    size_ratio = a_new / a;
    M_new = M_dust * pow(size_ratio, 3.0);
    dM = M_new - M_dust;
  }
  
  if(!isfinite(a_new) || a_new <= a || a_new > All.DustCoagulationMaxSize) return;
  
  Sp->P[dust_idx].setMass(M_new);
  Sp->DustP[dust_idx].GrainRadius = a_new;
  
  NCoagulationEvents++;
  TotalMassCoagulated += dM;
  
  static int coag_samples = 0;
  if(coag_samples < 100 && All.ThisTask == 0 && Sp->P[dust_idx].ID.get() % 10000 == 0) {
      DUST_PRINT("[COAGULATION] Event #%d: a=%.1f→%.1f nm, M=%.3e→%.3e Msun, "
                "n_H=%.1f n_eff=%.1f (C=%.0f) cm^-3, T=%.0f K, tau_coag=%.1f Myr\n",
                coag_samples, a, a_new, M_dust, M_new,
                n_H, n_eff, DustClumpingFactor,
                T_gas, tau_coag_yr/1e6);
    coag_samples++;
  }
  
  if(NCoagulationEvents % 10000 == 0 && All.ThisTask == 0) {
    DUST_PRINT("[COAGULATION_STATS] Events: %lld, Total mass grown: %.3e Msun\n",
               NCoagulationEvents, TotalMassCoagulated);
  }
}

/**
 * Subgrid grain growth model (HK11-based)
 */
void dust_grain_growth_subgrid(simparticles *Sp, int dust_idx, int gas_idx, double dt)
{
  if(!All.DustEnableGrowth) return;

  static int total_calls        = 0;
  static int failed_hot         = 0;
  static int failed_no_metals   = 0;
  static int failed_low_density = 0;
  static int failed_low_fmol    = 0;
  static int failed_no_dust     = 0;
  static int failed_too_far     = 0;
  static int failed_max_dz      = 0;
  static int failed_bad_tau     = 0;
  static int passed_all         = 0;

  static int used_species_sil   = 0;
  static int used_species_carb  = 0;

  static int fmol_diffuse  = 0;
  static int fmol_moderate = 0;
  static int fmol_dense    = 0;
  static int fmol_sf       = 0;

  total_calls++;

  if(total_calls % 5000 == 0 && All.ThisTask == 0) {
    DUST_PRINT("=== HK11 GROWTH DIAGNOSTICS (after %d attempts) Rank 0 only ===\n", total_calls);
    DUST_PRINT("  Failed hot (T>%.0e K):    %6d (%.1f%%)\n",
               All.DustThermalSputteringTemp, failed_hot, 100.0*failed_hot/total_calls);
    DUST_PRINT("  Failed no metals:        %6d (%.1f%%)\n",
               failed_no_metals, 100.0*failed_no_metals/total_calls);
    DUST_PRINT("  Failed low density:      %6d (%.1f%%)\n",
               failed_low_density, 100.0*failed_low_density/total_calls);
    DUST_PRINT("  Failed f_mol too low:    %6d (%.1f%%)\n",
               failed_low_fmol, 100.0*failed_low_fmol/total_calls);
    DUST_PRINT("  Failed no dust nearby:   %6d (%.1f%%)\n",
               failed_no_dust, 100.0*failed_no_dust/total_calls);
    DUST_PRINT("  Failed dust too far:     %6d (%.1f%%)\n",
               failed_too_far, 100.0*failed_too_far/total_calls);
    DUST_PRINT("  Failed max D/Z reached:  %6d (%.1f%%)\n",
               failed_max_dz, 100.0*failed_max_dz/total_calls);
    DUST_PRINT("  Failed bad tau_acc:      %6d (%.1f%%)\n",
               failed_bad_tau, 100.0*failed_bad_tau/total_calls);
    DUST_PRINT("  PASSED all checks:       %6d (%.1f%%)\n",
               passed_all, 100.0*passed_all/total_calls);
    DUST_PRINT("  Species picks: sil=%d  carb=%d\n", used_species_sil, used_species_carb);
    DUST_PRINT("  f_mol: diffuse=%d  moderate=%d  dense=%d  sf=%d\n",
               fmol_diffuse, fmol_moderate, fmol_dense, fmol_sf);
    DUST_PRINT("  Total mass grown:        %.3e Msun\n", TotalMassGrown);
    DUST_PRINT("  Growth events:           %lld\n", NGrainGrowthEvents);
    if(NGrainGrowthEvents > 0)
      DUST_PRINT("  Avg mass per event:      %.3e Msun\n", TotalMassGrown / NGrainGrowthEvents);
    DUST_PRINT("===============================================\n");
  }

  // ------------------------------------------------------------------
  // Gate 1: temperature — if T > sputtering threshold, no growth
  // ------------------------------------------------------------------
  const double T_gas = get_temperature_from_entropy(Sp, gas_idx);
  if(T_gas > All.DustThermalSputteringTemp) {
    failed_hot++;
    return;
  }

  // ------------------------------------------------------------------
  // Gate 2: metallicity — need metals to accrete
  // ------------------------------------------------------------------
  const double Z_gas = Sp->SphP[gas_idx].Metallicity;
  if(Z_gas < 1e-4) {
    failed_no_metals++;
    return;
  }

  // ------------------------------------------------------------------
  // Gate 3: density — compute n_eff now and bail early if diffuse.
  // This is cheap and must precede the D/Z check so that diffuse-gas
  // particles are not misattributed to failed_max_dz.
  // ------------------------------------------------------------------
  double gas_density_code    = Sp->SphP[gas_idx].Density * All.cf_a3inv;
  double gas_density_cgs     = gas_density_code * All.UnitDensity_in_cgs;
  double n_H                 = (gas_density_cgs * HYDROGEN_MASSFRAC) / PROTONMASS;
  double DustClumpingFactor  = dust_clumping_factor(n_H, Sp->SphP[gas_idx].Sfr > DUST_SFR_EPS);
  double n_eff_cm3           = n_H * DustClumpingFactor;

  if(n_eff_cm3 < 0.1) {
    failed_low_density++;
    return;
  }

  // ------------------------------------------------------------------
  // Gate 4: D/Z cap — now that we know this particle is in dense-enough
  // gas, a saturated D/Z is a genuine physics ceiling, not a density
  // artifact.  Early exit here is still a performance win.
  // ------------------------------------------------------------------
  double M_gas_quick  = Sp->P[gas_idx].getMass();
  double M_dust_quick = Sp->P[dust_idx].getMass();
  if(M_dust_quick >= M_gas_quick * Z_gas * 0.5) {
    failed_max_dz++;
    return;
  }

  // ------------------------------------------------------------------
  // Molecular fraction
  // ------------------------------------------------------------------
  double f_mol = 0.05;

  #ifdef STARFORMATION
    if(Sp->SphP[gas_idx].Sfr > DUST_SFR_EPS) {
      f_mol = 0.8;
      fmol_sf++;
    } else if(n_eff_cm3 > 100.0) {
      f_mol = 0.5;
      fmol_dense++;
    } else if(n_eff_cm3 > 10.0) {
      f_mol = 0.2;
      fmol_moderate++;
    } else {
      fmol_diffuse++;
    }
  #else
    if(n_H > 100.0) {
      f_mol = 0.5;
      fmol_dense++;
    } else if(n_H > 10.0) {
      f_mol = 0.2;
      fmol_moderate++;
    } else {
      fmol_diffuse++;
    }
  #endif

  if(Z_gas > 0.01) {
    f_mol *= 1.5;
    if(f_mol > 1.0) f_mol = 1.0;
  }

  if(f_mol < 0.01) {
    failed_low_fmol++;
    return;
  }

  // ------------------------------------------------------------------
  // Dust particle checks
  // ------------------------------------------------------------------
  const int nearest_dust = dust_idx;
  if(nearest_dust < 0) {
    failed_no_dust++;
    return;
  }

  double dxyz[3];
  Sp->nearest_image_intpos_to_pos(Sp->P[gas_idx].IntPos,
                                  Sp->P[nearest_dust].IntPos, dxyz);
  const double dist_kpc = sqrt(dxyz[0]*dxyz[0] + dxyz[1]*dxyz[1] + dxyz[2]*dxyz[2]);

  if(dist_kpc > 5.0) {
    failed_too_far++;
    return;
  }

  const double a      = Sp->DustP[nearest_dust].GrainRadius;
  const double M_dust = Sp->P[nearest_dust].getMass();

  // ------------------------------------------------------------------
  // Corruption check (domain exchange victim or numerical explosion)
  // Can happen in numerous ways...
  //  - 1. DOMAIN_EXCHANGE: GrainRadius is exactly 0.0 and all DustP fields are
  //    zero — this is a domain decomp problem where DustP[] was not sent with
  //    the particle (Gadget zeroed the slot with memset).  
  //  - 2. NUMERICAL: GrainRadius is NaN, Inf, or negative — a genuine numerical
  //    explosion upstream (bad dt, NaN propagation in growth/drag).
  // ------------------------------------------------------------------
  if(a <= 0.0 || !isfinite(a) || M_dust <= 0.0 || !isfinite(M_dust)) {

    bool is_domain_exchange = (a == 0.0 &&
                               Sp->DustP[nearest_dust].CarbonFraction == 0.0 &&
                               Sp->DustP[nearest_dust].DustTemperature == 0.0);

    NDustDestroyedByCorruption++;

    static int corruption_count = 0;
    corruption_count++;
    printf("[GROWTH_CORRUPTION|T=%d|Step=%d] #%d: idx=%d a=%.3e M=%g type=%d ID=%lld "
           "subtype=%s | RunningTotal=%lld\n",
           All.ThisTask, All.NumCurrentTiStep, corruption_count,
           nearest_dust, a, M_dust, Sp->P[nearest_dust].getType(),
           (long long)Sp->P[nearest_dust].ID.get(),
           is_domain_exchange ? "DOMAIN_EXCHANGE_DustP_not_sent"
                              : "NUMERICAL_NaN_or_negative",
           NDustDestroyedByCorruption);

    Sp->P[nearest_dust].setMass(1e-30);
    Sp->P[nearest_dust].ID.set(0);
    Sp->P[nearest_dust].setType(3);
    memset(&Sp->DustP[nearest_dust], 0, sizeof(dust_data));
    Sp->DustP[nearest_dust].GrainRadius = DUST_MIN_GRAIN_SIZE;
    return;
  }

  // ------------------------------------------------------------------
  // HK11 accretion timescale
  // ------------------------------------------------------------------
  const double CF  = Sp->DustP[nearest_dust].CarbonFraction;
  const int species = (CF >= 0.5) ? 1 : 0;
  if(species == 1) used_species_carb++; else used_species_sil++;

  const double T_eff_K       = 20.0;
  const double Zsun_massfrac = 0.02;
  const double S_stick       = 0.3;

  double n_eff_for_growth = n_eff_cm3;
  if(n_eff_for_growth < 100.0) n_eff_for_growth = 100.0;

  double a_cm       = a * 1e-7;
  double tau_acc_yr = tau_acc_yr_HK11(n_eff_for_growth, T_eff_K,
                                      Z_gas, Zsun_massfrac,
                                      a_cm, S_stick, species);

  tau_acc_yr *= All.DustGrowthCalibration;

  if(!isfinite(tau_acc_yr) || tau_acc_yr <= 0.0) {
    failed_bad_tau++;
    return;
  }
  if(tau_acc_yr < 1e6)  tau_acc_yr = 1e6;
  if(tau_acc_yr > 5e9)  tau_acc_yr = 5e9;

  const double tau_acc_code = tau_acc_yr * SEC_PER_YEAR / All.UnitTime_in_s;

  // ------------------------------------------------------------------
  // Grain size and mass update
  // ------------------------------------------------------------------
  double a_new = a * exp(f_mol * dt / tau_acc_code);
  double da    = a_new - a;
  if(!isfinite(da) || da <= 0.0) return;
  if(a_new > DUST_MAX_GRAIN_SIZE) {
    a_new = DUST_MAX_GRAIN_SIZE;
    da    = a_new - a;
    if(da <= 0.0) return;
  }

  if(a_new < DUST_MIN_GRAIN_SIZE || a_new > DUST_MAX_GRAIN_SIZE) return;

  // accretion_efficiency removed — dm is applied at full efficiency.
  // Coagulation effects are handled by dust_grain_coagulation().
  double dm = M_dust * (3.0 * da / a);
  if(!isfinite(dm) || dm <= 0.0) return;

  const double M_gas     = Sp->P[gas_idx].getMass();
  const double M_metals  = M_gas * Z_gas;

  // Maximum accepted dust to metallicity ratio --------------------------------------------------------------
  // -- Initially had a constant (~0.5) but this led to excessive early growth in low-metallicity gas
  // -- Observationally motivated: D/Z builds from ~0.2 at z>4 toward ~0.5 at z=0, so we want a redshift-dependent 
  // -- cap that allows higher D/Z at late times but is more restrictive at early times when the gas is metal-poor and prone to runaway growth
  // -- the concern, what if this artificially suppresses growth in metal-rich high-z environments where dust genuinely should be building up?
  double z = 1.0/All.Time - 1.0;
  double max_dust_to_metal = 0.5 / (1.0 + 0.15 * z);
  if(max_dust_to_metal < 0.05) max_dust_to_metal = 0.05;

  const double M_dust_max        = M_metals * max_dust_to_metal;

  // The early D/Z guard above catches most cases; this is a fine-grained
  // cap on the computed dm to avoid overshooting within the timestep.
  if(M_dust >= M_dust_max) {
    failed_max_dz++;
    return;
  }

  if(M_dust + dm > M_dust_max) {
    dm = M_dust_max - M_dust;
    if(dm <= 0.0) return;  // already at cap — no dead failed_max_dz++ here

    da    = (dm / M_dust) * a / 3.0;
    a_new = a + da;

    if(a_new < DUST_MIN_GRAIN_SIZE || a_new > DUST_MAX_GRAIN_SIZE) return;
  }

  if(dm > M_metals) dm = 0.99 * M_metals;

  const double max_dm_per_step = 0.2 * M_dust;
  if(dm > max_dm_per_step) dm = max_dm_per_step;
  if(dm <= 0.0) return;

  passed_all++;

  static int dt_printed = 0;
  if(dt_printed < 10 && All.ThisTask == 0) {
    double dt_myr = dt * All.UnitTime_in_s / (1e6 * SEC_PER_YEAR);
    DUST_PRINT("[GROWTH_DEBUG] dt = %.3e code units = %.3f Myr\n", dt, dt_myr);
    DUST_PRINT("[GROWTH_DEBUG] da = %.3f nm, dm = %.3e Msun (capped at %.3e)\n",
              da, dm, max_dm_per_step);
    dt_printed++;
  }

  Sp->P[nearest_dust].setMass(M_dust + dm);
  Sp->DustP[nearest_dust].GrainRadius = a_new;

  double Z_new = Z_gas - (dm / M_gas);
  if(Z_new < 1e-5) Z_new = 1e-5;
  Sp->SphP[gas_idx].Metallicity = Z_new;

  #ifdef STARFORMATION
    Sp->SphP[gas_idx].MassMetallicity = M_gas * Z_new;
  #endif

  NGrainGrowthEvents++;
  TotalMassGrown += dm;

  static int growth_count = 0;
  growth_count++;
  if(growth_count % 10000 == 0 && All.ThisTask == 0) {
    DUST_PRINT("[HK11_GROWTH] Event #%d: species=%s CF=%.2f f_mol=%.3f n_H=%.1f→%.1f cm^-3 (C=%.0f)\n",
              growth_count, (species==1 ? "carb" : "sil"), CF, f_mol,
              n_H, n_eff_cm3, DustClumpingFactor);
    DUST_PRINT("[HK11_GROWTH] tau_acc=%.2e yr (%.2f Myr) | n_eff=%.0f cm^-3 T_eff=%.0f K | Z=%.4f\n",
               tau_acc_yr, tau_acc_yr/1e6, n_eff_cm3, T_eff_K, Z_gas);
    DUST_PRINT("[HK11_GROWTH] Grain: a=%.2f→%.2f nm | dm=%.3e (M_dust: %.3e→%.3e)\n",
               a, a_new, dm, M_dust, M_dust+dm);
    DUST_PRINT("[HK11_GROWTH] Gas: M_gas=%.3e | Z=%.4f→%.4f | M_metals=%.3e | dZ=%.3e\n",
               M_gas, Z_gas, Z_new, M_metals, (Z_gas - Z_new));
  }
}

#endif /* DUST */