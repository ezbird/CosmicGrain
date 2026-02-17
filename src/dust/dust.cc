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
#define MIN_DUST_PARTICLE_MASS  1e-10   // Minimum mass to create dust particle; this should be plenty small to allow for very small grains, but prevents creating huge numbers of dust particles from tiny amounts of dust mass
#define DUST_MIN_GRAIN_SIZE  1.0        // nm
#define DUST_MAX_GRAIN_SIZE  200.0      // nm

#define DUST_PRINT(...) do{ if(All.DustDebugLevel){ \
  printf("[DUST|T=%d|a=%.6g z=%.3f] ", All.ThisTask, (double)All.Time, 1.0/All.Time-1.0); \
  printf(__VA_ARGS__); } }while(0)

extern double get_random_number(void);

// Function declarations
double estimate_molecular_fraction(double n_H, double Z, double T);
void dust_grain_growth_subgrid(simparticles *Sp, int dust_idx, int gas_idx, double dt);
double scale_factor_to_physical_time(double delta_a);

// Access feedback's global spatial hash
extern spatial_hash_zoom gas_hash;
extern void rebuild_feedback_spatial_hash(simparticles *Sp, double max_feedback_radius);

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
static double    TotalHashSearchTime = 0.0;

// Destruction mechanism tracking
long long NDustDestroyedByThermal   = 0;
long long NDustDestroyedByShock     = 0;
long long NDustDestroyedByAstration = 0;
double TotalDustMassAstrated        = 0.0;

// Growth/erosion tracking
long long NGrainGrowthEvents        = 0;
long long NGrainErosionEvents       = 0;
double    TotalMassGrown            = 0.0;
double    TotalMassEroded           = 0.0;

// Coagulation
long long NCoagulationEvents        = 0;
double    TotalMassCoagulated       = 0.0;

/**
 * Clean up invalid dust particles
 */
void cleanup_invalid_dust_particles(simparticles *Sp)
{
  int cleaned = 0;
  
  for(int i = 0; i < Sp->NumPart; i++) {
    if(Sp->P[i].getType() == DUST_PARTICLE_TYPE) {
      double a = Sp->DustP[i].GrainRadius;
      double mass = Sp->P[i].getMass();
      double pos[3];
      Sp->intpos_to_pos(Sp->P[i].IntPos, pos);
      
      // Check for ANY corruption
      bool is_corrupt = false;
      
      if(a <= 0.0 || !isfinite(a)) is_corrupt = true;
      if(mass < 1e-20 || !isfinite(mass)) is_corrupt = true;
      if(!isfinite(pos[0]) || !isfinite(pos[1]) || !isfinite(pos[2])) is_corrupt = true;
      if(!isfinite(Sp->P[i].Vel[0]) || !isfinite(Sp->P[i].Vel[1]) || !isfinite(Sp->P[i].Vel[2])) is_corrupt = true;
      
      if(is_corrupt) {
        Sp->P[i].setMass(1e-30);
        Sp->P[i].ID.set(0);
        Sp->P[i].setType(3); // hack! Changing this to an unused parttype so find_nearest_dust_particle() skips it
        memset(&Sp->DustP[i], 0, sizeof(dust_data));
        Sp->DustP[i].GrainRadius = DUST_MIN_GRAIN_SIZE;
        cleaned++;
      }
    }
  }
  
  if(cleaned > 0 && All.ThisTask == 0) {
    DUST_PRINT("[CLEANUP] Marked %d corrupted particles for removal\n", cleaned);
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
  
  // Sputtering timescale (Tsai & Mathews 1995, Draine & Salpeter 1979)
  // Calibration: ~10 Myr at T=10^7 K (hot ISM), ~100 Myr at T=10^6 K (warm ISM)
  double tau_sputter_yr = 1e8 * pow(1e6 / T_gas, 1.5);
  
  // Apply reasonable bounds
  if(tau_sputter_yr < 1e6)  tau_sputter_yr = 1e6;   // 1 Myr floor (very hot gas)
  if(tau_sputter_yr > 1e9) tau_sputter_yr = 1e9;    // 1 Gyr ceiling (cool gas)
  
  double tau_sputter = tau_sputter_yr * SEC_PER_YEAR / All.UnitTime_in_s;
  double da_dt = -a / tau_sputter;
  double da = da_dt * dt;
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

/**
500 km/s shock + 10 nm grain: ~90% destruction chance (outright shattering)
500 km/s shock + 100 nm grain: ~30% destruction chance
200 km/s shock + 10 nm grain: ~10% destruction chance
200 km/s shock + 100 nm grain: 0% destruction, but erosion
 */
int erode_dust_grain_shock(simparticles *Sp, int dust_idx, double shock_velocity_km_s, 
                           double distance_to_sn, double shock_radius)
{
  if(!All.DustEnableShockDestruction) return 0;
  
  double a = Sp->DustP[dust_idx].GrainRadius;
  
  double base_efficiency = get_shock_destruction_efficiency(shock_velocity_km_s);
  
  double distance_factor = 1.0 - 0.7 * (distance_to_sn / shock_radius);
  if(distance_factor < 0.2) distance_factor = 0.2;
  
  double size_factor = 1.0;
  if(a < 20) {
    size_factor = 1.5;   // Small grains easier to destroy
  } else if(a < 50) {
    size_factor = 1.2;
  } else if(a > 150) {
    size_factor = 0.7;   // Large grains harder to destroy
  }
  
  double erosion_fraction = base_efficiency * distance_factor * size_factor;
  if(erosion_fraction > 0.95) erosion_fraction = 0.95;
  
  // ========================================================================
  // STEP 1: Check for OUTRIGHT DESTRUCTION (shattering) before erosion
  // ========================================================================
  // Physics: Strong shocks can shatter grains completely, not just erode them
  // Destruction probability depends on:
  //   - Shock strength (velocity)
  //   - Grain size (small grains easier to destroy)
  //   - Distance from center
  
  double destruction_prob = 0.0;
  
  if(shock_velocity_km_s > 150.0) {
    // Base destruction probability from shock strength
    double velocity_factor = (shock_velocity_km_s - 150.0) / 350.0;  // 0 at 150 km/s, 1 at 500 km/s
    if(velocity_factor > 1.0) velocity_factor = 1.0;
    
    // Size factor: small grains easier to shatter
    // 10 nm grain: factor=3.0, 50 nm: factor=1.0, 150 nm: factor=0.3
    double destruction_size_factor = 50.0 / a;  // Inverse relationship
    if(destruction_size_factor > 3.0) destruction_size_factor = 3.0;
    if(destruction_size_factor < 0.3) destruction_size_factor = 0.3;
    
    // Combine factors
    destruction_prob = velocity_factor * destruction_size_factor * distance_factor;
    
    // Cap at reasonable maximum
    if(destruction_prob > 0.9) destruction_prob = 0.9;
    
    // Roll the dice
    if(get_random_number() < destruction_prob) {
      // GRAIN COMPLETELY DESTROYED BY SHATTERING
      double dust_mass = Sp->P[dust_idx].getMass();
      
      int nearest_gas = find_nearest_gas_particle(Sp, dust_idx, 2.0, NULL);
      if(nearest_gas >= 0) {
        double gas_mass = Sp->P[nearest_gas].getMass();
        double current_Z = Sp->SphP[nearest_gas].Metallicity;
        double new_Z = current_Z + (dust_mass / gas_mass);
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
      NDustDestroyedByShock++;
      TotalMassEroded += dust_mass;
      
      return 1;  // Destroyed!
    }
  }
  
  // ========================================================================
  // STEP 2: If grain survived shattering, apply EROSION (shrinking)
  // ========================================================================
  
  double a_new = a * (1.0 - erosion_fraction * 0.8);  // Erosion shrinks grain
  
  if(a_new <= 0.0 || !isfinite(a_new)) {
    a_new = 0.0;  // Numerical error → destroy
  }

  // Check if eroded below minimum size
  if(a_new < DUST_MIN_GRAIN_SIZE || a_new == 0.0) {
    // Grain eroded too small, destroy it
    double dust_mass = Sp->P[dust_idx].getMass();
    
    int nearest_gas = find_nearest_gas_particle(Sp, dust_idx, 2.0, NULL);
    if(nearest_gas >= 0) {
      double gas_mass = Sp->P[nearest_gas].getMass();
      double current_Z = Sp->SphP[nearest_gas].Metallicity;
      double new_Z = current_Z + (dust_mass / gas_mass);
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
    NDustDestroyedByShock++;
    TotalMassEroded += dust_mass;
    
    return 1;
  }
  
  // ========================================================================
  // STEP 3: Grain survived both shattering and erosion - update its size
  // ========================================================================
  
  Sp->DustP[dust_idx].GrainRadius = a_new;
  
  double mass_ratio = pow(a_new / a, 3.0);
  if(!isfinite(mass_ratio) || mass_ratio < 0.0 || mass_ratio > 1.5) {
    if(All.ThisTask == 0) {
      DUST_PRINT("[ERROR] Invalid mass_ratio=%.3e in shock erosion (a=%.3e→%.3e)\n",
                 mass_ratio, a, a_new);
    }
    return 0;
  }

  double old_mass = Sp->P[dust_idx].getMass();
  double new_mass = old_mass * mass_ratio;
  double mass_lost = old_mass - new_mass;
  
  Sp->P[dust_idx].setMass(new_mass);
  
  int nearest_gas = find_nearest_gas_particle(Sp, dust_idx, 5.0, NULL);
  if(nearest_gas >= 0) {
    double gas_mass = Sp->P[nearest_gas].getMass();
    double current_Z = Sp->SphP[nearest_gas].Metallicity;
    double new_Z = current_Z + (mass_lost / gas_mass);
    Sp->SphP[nearest_gas].Metallicity = new_Z;
    
    #ifdef STARFORMATION
    Sp->SphP[nearest_gas].MassMetallicity = gas_mass * new_Z;
    #endif
  }
  
  LocalDustMassChange -= mass_lost;
  NGrainErosionEvents++;
  TotalMassEroded     += mass_lost;
  
  return 0;  // Survived (but eroded)
}

/*!
 * \brief Calculate Epstein drag stopping timescale for a dust grain
 * 
 * Implements the standard Epstein drag formula with supersonic correction
 * following McKinnon+2018 MNRAS 478, 2851 (equations 8-9).
 * 
 * \param grain_radius_nm Grain radius in nanometers
 * \param grain_density Internal grain density in g/cm³ (typically 2.4)
 * \param gas_density Gas density in g/cm³
 * \param gas_temperature Gas temperature in Kelvin
 * \param relative_velocity_cgs Relative velocity between dust and gas in cm/s
 * \param mu_gas Mean molecular weight (typically 0.6 for ionized, 1.3 for neutral)
 * \param gamma_gas Adiabatic index (typically 5/3)
 * 
 * \return Stopping timescale in Myr (with numerical caps applied)
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
    // Physical constants (CGS)
    const double k_boltzmann = 1.38064852e-16;  // erg/K
    const double m_proton = 1.6726219e-24;      // g
    const double sec_to_myr = 3.15576e13;       // conversion factor
    
    // Convert grain radius to cm
    double grain_radius_cm = grain_radius_nm * 1e-7;
    
    // Calculate sound speed (cm/s)
    double cs_cgs = sqrt(gamma_gas * k_boltzmann * gas_temperature / (mu_gas * m_proton));
    
    // Avoid division by zero or unphysical values
    if(cs_cgs < 1e3 || gas_density_cgs < 1e-30)  // c_s < 0.01 km/s or extremely low density
    {
        return 50.0;  // Return maximum timescale for pathological cases
    }
    
    // Subsonic Epstein drag stopping time (seconds)
    // t_s = (√(πγ) * a * ρ_gr) / (2√2 * ρ_gas * c_s)
    double t_stop_subsonic = (sqrt(M_PI * gamma_gas) * grain_radius_cm * grain_density) 
                             / (2.0 * sqrt(2.0) * gas_density_cgs * cs_cgs);
    
    // Supersonic correction factor
    // Correction: [1 + (9π/128) * Mach²]^(-1/2)
    double mach = relative_velocity_cgs / cs_cgs;
    double supersonic_factor = 1.0;
    
    if(mach > 0.1)  // Only apply for non-negligible velocities
    {
        supersonic_factor = 1.0 / sqrt(1.0 + (9.0 * M_PI / 128.0) * mach * mach);
    }
    
    // Apply correction
    double t_stop_sec = t_stop_subsonic * supersonic_factor;
    
    // Convert to Myr
    double drag_timescale_myr = t_stop_sec / sec_to_myr;
    
    // Apply numerical stability caps
    // Floor: 1 kyr (prevents timestep issues in very dense regions)
    // Ceiling: 50 Myr (prevents infinite timescales in very diffuse regions)
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
  
  // Use pre-found gas instead of searching
  if(nearest_gas < 0) return 0;  // No gas nearby
  
  // ==================================================================
  // STEP 1: Extract gas properties
  // ==================================================================
  
  // Gas velocity
  double gas_vel[3] = {Sp->P[nearest_gas].Vel[0],
                       Sp->P[nearest_gas].Vel[1],
                       Sp->P[nearest_gas].Vel[2]};
  
  // Gas density (physical, not comoving)
  double gas_density = Sp->SphP[nearest_gas].Density * All.cf_a3inv;
  double gas_density_cgs = gas_density * All.UnitDensity_in_cgs;
  double n_H = gas_density_cgs / PROTONMASS;  // Number density in cm^-3
  
  // Gas temperature
  double utherm = Sp->get_utherm_from_entropy(nearest_gas);
  double T_gas = utherm * (All.UnitEnergy_in_cgs / All.UnitMass_in_g) 
                 / BOLTZMANN * PROTONMASS * 0.6;  // Assuming mu = 0.6
  
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
  
  double mu_gas = 0.6;      // Mean molecular weight (ionized gas)
  double gamma_gas = 5.0/3.0;  // Adiabatic index

  double drag_timescale_myr = calculate_drag_timescale(
      Sp->DustP[dust_idx].GrainRadius,    // in nm
      2.4,                                // grain density g/cm³
      gas_density_cgs,                    // g/cm³
      T_gas,                              // K
      vrel_cgs,                           // cm/s
      mu_gas,
      gamma_gas
  );
  
  // Convert drag timescale to code units
  double drag_timescale = drag_timescale_myr * 1e6 * SEC_PER_YEAR / All.UnitTime_in_s;
  
  // ==================================================================
  // STEP 4: Apply drag
  // If we assume v_gas and t_s are constant over a timestep dt, this 
  // differential equation has an exact analytical solution! Nice.
  // ==================================================================
  
  double drag_factor = 1.0 - exp(-dt / drag_timescale);

  // If we need to handle any instability, cap the drag factor per timestep with these 2 lines:
  // const double MAX_DRAG_FACTOR = 0.5;  // Limit per timestep for stability
  // if(drag_factor > MAX_DRAG_FACTOR) drag_factor = MAX_DRAG_FACTOR;

  for(int k = 0; k < 3; k++) {
    Sp->P[dust_idx].Vel[k] += drag_factor * (gas_vel[k] - Sp->P[dust_idx].Vel[k]);
  }
  
  // ========================================================================
  // STEP 5: Update dust temperature (PROPER PHYSICS)
  // ========================================================================
  // Dust temperature is determined by thermal equilibrium, NOT gas temperature!
  // 
  // Physical processes:
  // - Collisional heating from gas (proportional to n_gas * sqrt(T_gas))
  // - Radiative cooling (proportional to T_dust^4)
  // - Result: T_dust << T_gas in hot, diffuse gas
  //
  // Simple approximation: T_dust ~ sqrt(T_gas * T_CMB) for collisional heating
  // (Geometric mean between gas temp and CMB floor)

  double T_CMB = 2.7 * (1.0 / All.Time);  // CMB temperature at this redshift

  // Dust equilibrium temperature is MUCH cooler than gas in hot regions
  // Use simple scaling: T_dust^2 ~ T_gas * T_CMB (balance heating/cooling)
  double T_dust_eq = sqrt(T_gas * T_CMB);

  // For very hot gas (T > 1e6 K), dust should be destroyed, not heated
  // But if we're here, thermal sputtering hasn't destroyed it yet
  // Cap dust temperature at reasonable maximum before destruction
  if(T_dust_eq > 2000.0) T_dust_eq = 2000.0;  // Dust sublimates above ~2000 K

  // Enforce CMB floor
  if(T_dust_eq < T_CMB) T_dust_eq = T_CMB;

  // Thermal coupling timescale (10 Myr as before)
  double T_dust = Sp->DustP[dust_idx].DustTemperature;
  double tau_thermal = 10e6 * SEC_PER_YEAR / All.UnitTime_in_s;
  double alpha = 1.0 - exp(-dt / tau_thermal);

  // Converge toward EQUILIBRIUM temperature, not gas temperature
  T_dust = T_dust * (1.0 - alpha) + T_dust_eq * alpha;

  // Final safety checks
  if(T_dust < T_CMB) T_dust = T_CMB;  // CMB floor
  if(T_dust > 2000.0) T_dust = 2000.0;  // Sublimation ceiling
  if(!isfinite(T_dust) || T_dust <= 0.0) T_dust = T_CMB;  // Fix any NaN/zero

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
    if(drag_samples < 500) {  // Sample first 500 drag events
    
      double vel_kms = sqrt(Sp->P[dust_idx].Vel[0]*Sp->P[dust_idx].Vel[0] + 
                          Sp->P[dust_idx].Vel[1]*Sp->P[dust_idx].Vel[1] + 
                          Sp->P[dust_idx].Vel[2]*Sp->P[dust_idx].Vel[2]);
      vel_kms *= All.UnitVelocity_in_cm_per_s / 1e5;
      
      // Calculate Mach number for diagnostics
      double cs_cgs = sqrt(gamma_gas * BOLTZMANN * T_gas / (mu_gas * PROTONMASS));
      double mach_number = vrel_cgs / cs_cgs;
      double vrel_kms = vrel_cgs / 1e5;

      // Timestep in Myr
      double dt_myr = dt * All.UnitTime_in_s / (1e6 * SEC_PER_YEAR);

      // Raw (uncapped) drag factor
      double drag_factor_raw = 1.0 - exp(-dt / drag_timescale);

      // Actual drag factor used
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
  
  /* How many dust particles to create with each event?
   All are starting at 10nm in radius. (do a size distribution later?)
   SNe are more compact, energetic ejecta → fewer, more massive particles
   AGB winds are extended, gentle → more numerous, lighter particles

   For questions like: where does dust go? which grains survive shocks?  more particles = better sampling
   For questions about total dust mass or average grain size, fewer particles may suffice.
  */
  // How many dust particles to create with each event?
  // All start at initial size (10nm for SN, 100nm for AGB)
  // More particles = better spatial/size sampling, slower performance
  // Fewer particles = faster, but coarser sampling
  int n_dust_particles = (feedback_type == 1) ? All.DustParticlesPerSNII : All.DustParticlesPerAGB;
  double dust_mass_per_particle = total_dust_mass / n_dust_particles;

  // FIND GAS VELOCITY ONCE (before loop) - NEW SECTION
  int nearest_gas = find_nearest_gas_particle(Sp, star_idx, 2.0, NULL);  // 2 kpc max
  double gas_vel[3] = {0.0, 0.0, 0.0};
  if(nearest_gas >= 0) {
    gas_vel[0] = Sp->P[nearest_gas].Vel[0];
    gas_vel[1] = Sp->P[nearest_gas].Vel[1];
    gas_vel[2] = Sp->P[nearest_gas].Vel[2];
  } else {
    // Fallback: use star velocity if no gas found
    gas_vel[0] = Sp->P[star_idx].Vel[0];
    gas_vel[1] = Sp->P[star_idx].Vel[1];
    gas_vel[2] = Sp->P[star_idx].Vel[2];
  }

  for(int n = 0; n < n_dust_particles; n++) {
    if(Sp->NumPart >= Sp->MaxPart) {
      if(All.ThisTask == 0) {
        DUST_PRINT("[WARNING] Cannot create dust particle - particle array full\n");
      }
      break;
    }
    
    // Random position (unchanged)
    double theta = acos(2.0 * get_random_number() - 1.0);
    double phi   = 2.0 * M_PI * get_random_number();
    
    double offset_min, offset_max;
    if(feedback_type == 1) {  // SN
      offset_min = All.DustOffsetMinSNII;
      offset_max = All.DustOffsetMaxSNII;
    } else {  // AGB  
      offset_min = All.DustOffsetMinAGB;
      offset_max = All.DustOffsetMaxAGB;
    }

    double r = offset_min + (offset_max - offset_min) * get_random_number();

    double offset_kpc[3];
    offset_kpc[0] = r * sin(theta) * cos(phi);
    offset_kpc[1] = r * sin(theta) * sin(phi);
    offset_kpc[2] = r * cos(theta);
    
  // Velocity relative to STAR
  double star_vel[3];
  star_vel[0] = Sp->P[star_idx].Vel[0];
  star_vel[1] = Sp->P[star_idx].Vel[1];
  star_vel[2] = Sp->P[star_idx].Vel[2];

  // Random kick direction
  double initial_velocity[3];
  initial_velocity[0] = velocity_scale * sin(theta) * cos(phi) / All.UnitVelocity_in_cm_per_s * 1e5;
  initial_velocity[1] = velocity_scale * sin(theta) * sin(phi) / All.UnitVelocity_in_cm_per_s * 1e5;
  initial_velocity[2] = velocity_scale * cos(theta) / All.UnitVelocity_in_cm_per_s * 1e5;

  // ADD STAR VELOCITY (not gas!)
  initial_velocity[0] += star_vel[0];
  initial_velocity[1] += star_vel[1];
  initial_velocity[2] += star_vel[2];
    
    spawn_dust_particle(Sp, offset_kpc, dust_mass_per_particle, initial_velocity, star_idx, feedback_type);
    
    int new_idx = Sp->NumPart - 1;

    if(feedback_type == 1) { // SN
      Sp->DustP[new_idx].GrainRadius = 10.0;
      Sp->DustP[new_idx].CarbonFraction = 0.1;
      Sp->DustP[new_idx].GrainType = 0;
    }
    else if(feedback_type == 2) {  // AGB
      Sp->DustP[new_idx].GrainRadius = 100.0;
      Sp->DustP[new_idx].CarbonFraction = 0.6;
      Sp->DustP[new_idx].GrainType = 1;
    }
  }

  LocalDustCreatedThisStep += n_dust_particles;
  LocalDustMassChange      += total_dust_mass;
  DustNeedsSynchronization  = 1;

  // Log initial velocities for first few dust creation events
if(All.ThisTask == 0) {
  static int velocity_samples = 0;
  if(velocity_samples < 50) {  // Sample first 50 dust particles created
    // Get the last created dust particle
    int new_idx = Sp->NumPart - 1;
    if(new_idx >= 0 && Sp->P[new_idx].getType() == DUST_PARTICLE_TYPE) {
      double vel_mag = sqrt(Sp->P[new_idx].Vel[0]*Sp->P[new_idx].Vel[0] + 
                           Sp->P[new_idx].Vel[1]*Sp->P[new_idx].Vel[1] + 
                           Sp->P[new_idx].Vel[2]*Sp->P[new_idx].Vel[2]);
      vel_mag *= All.UnitVelocity_in_cm_per_s / 1e5;  // Convert to km/s
      
      double star_vel_mag = sqrt(Sp->P[star_idx].Vel[0]*Sp->P[star_idx].Vel[0] + 
                                Sp->P[star_idx].Vel[1]*Sp->P[star_idx].Vel[1] + 
                                Sp->P[star_idx].Vel[2]*Sp->P[star_idx].Vel[2]);
      star_vel_mag *= All.UnitVelocity_in_cm_per_s / 1e5;  // Convert to km/s
      
      int nearest_gas = find_nearest_gas_particle(Sp, star_idx, 2.0, NULL);
      double rho = 1.0;  // Default
      double gas_vel_mag = 0.0;
      
      if(nearest_gas >= 0) {
        double gas_density = Sp->SphP[nearest_gas].Density * All.cf_a3inv;
        rho = gas_density * All.UnitDensity_in_cgs / PROTONMASS;
        
        // NEW: Get gas velocity
        gas_vel_mag = sqrt(Sp->P[nearest_gas].Vel[0]*Sp->P[nearest_gas].Vel[0] + 
                          Sp->P[nearest_gas].Vel[1]*Sp->P[nearest_gas].Vel[1] + 
                          Sp->P[nearest_gas].Vel[2]*Sp->P[nearest_gas].Vel[2]);
        gas_vel_mag *= All.UnitVelocity_in_cm_per_s / 1e5;  // Convert to km/s
      }
      
      DUST_PRINT("[DUST_CREATE] vel_dust=%.1f km/s vel_star=%.1f km/s vel_gas=%.1f km/s "
                 "rho=%.3e cm^-3 grain_r=%.2f nm feedback_type=%d\n",
                 vel_mag, star_vel_mag, gas_vel_mag, 
                 rho, Sp->DustP[new_idx].GrainRadius, feedback_type);
      
      velocity_samples++;
    }
  }
}
}

/**
 * Global synchronization of dust statistics
 * Running this every timestep.
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

  // Synchronize MaxID across all tasks to prevent duplicate IDs
  // This was found to be an issue when restarting from snapshots
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
    // Print warning but DON'T create particle
    static int warning_count = 0;
    if(warning_count < 10 && All.ThisTask == 0) {
      printf("[DUST_ERROR] T=%d: Cannot create dust - array full (NumPart=%d, MaxPart=%d)\n",
             All.ThisTask, Sp->NumPart, Sp->MaxPart);
      warning_count++;
    }
    return;  // Return without creating
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
  
  // Initialize these; they will get changed based on feedback type below, but set defaults here
  Sp->DustP[new_idx].GrainRadius = 10.0;
  Sp->DustP[new_idx].CarbonFraction = 0.3;
  Sp->DustP[new_idx].GrainType = 2;
  
  // Dust condenses in cool ejecta, NOT hot shocked gas
  // Set initial temperature based on formation environment
  if(feedback_type == 1) {  // Type II SN
    // SN dust forms at T ~ 1000 K after ejecta has cooled
    // (Todini & Ferrara 2001, Nozawa et al. 2003)
    Sp->DustP[new_idx].DustTemperature = 1000.0;  // K 
  } else {  // AGB
    // AGB dust forms in cool winds at T ~ 600 K
    // (Ferrarotti & Gail 2006)
    Sp->DustP[new_idx].DustTemperature = 600.0;  // K 
  }

  // Enforce CMB floor (especially important at high z)
  double T_CMB = 2.7 * (1.0 / All.Time);  // T_CMB(z) = 2.7(1+z) K
  if(Sp->DustP[new_idx].DustTemperature < T_CMB) {
    Sp->DustP[new_idx].DustTemperature = T_CMB;
  }

  Sp->P[new_idx].StellarAge = All.Time;
  Sp->P[new_idx].Ti_Current = All.Ti_Current;
  
  //Sp->TimeBinsGravity.ActiveParticleList[Sp->TimeBinsGravity.NActiveParticles++] = new_idx;
  Sp->P[new_idx].TimeBinGrav = All.HighestActiveTimeBin;
  Sp->P[new_idx].TimeBinHydro = 0;
  Sp->TimeBinsGravity.timebin_add_particle(new_idx, star_idx, 
    All.HighestActiveTimeBin,
    Sp->TimeBinSynchronized[All.HighestActiveTimeBin]);

  // DIAGNOSTIC: Verify particle is properly initialized BEFORE incrementing NumPart
  if(Sp->DustP[new_idx].GrainRadius <= 0.0 || !isfinite(Sp->DustP[new_idx].GrainRadius)) {
    DUST_PRINT("[SPAWN_BUG] Just set GrainRadius but it's %.3e for new particle at idx=%d!\n",
               Sp->DustP[new_idx].GrainRadius, new_idx);
    Sp->DustP[new_idx].GrainRadius = 10.0;  // Force it
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
  double bin_edges[NBINS+1] = {0.0, 10.0, 50.0, 100.0, 150.0, 200.0, 500.0};  // nm
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
 * 
 * @param Sp Simulation particles
 * @param gas_idx Index of gas particle forming stars
 * @param stellar_mass_formed Mass converted to stars
 */
void consume_dust_by_astration(simparticles *Sp, int gas_idx, double stellar_mass_formed)
{
  if(!All.DustEnableAstration) return;
  
  double gas_mass = Sp->P[gas_idx].getMass();
  double sf_fraction = stellar_mass_formed / (gas_mass + stellar_mass_formed);
  
  // Find dust within ~3× gas smoothing length
  double search_radius = 3.0 * Sp->SphP[gas_idx].Hsml;
  if(search_radius > 5.0) search_radius = 5.0;  // Cap at 5 kpc
  
  // Find nearby dust particles
  const int MAX_NEIGHBORS = 100;
  int neighbor_indices[MAX_NEIGHBORS];
  double neighbor_distances[MAX_NEIGHBORS];
  int n_neighbors = 0;
  
  gas_hash.find_neighbors(Sp, gas_idx, search_radius,
                          neighbor_indices, neighbor_distances,
                          &n_neighbors, MAX_NEIGHBORS);
  
  if(n_neighbors == 0) return;
  
  // Calculate total dust mass in region
  double total_dust_mass = 0.0;
  for(int i = 0; i < n_neighbors; i++) {
    int dust_idx = neighbor_indices[i];
    if(Sp->P[dust_idx].getType() == DUST_PARTICLE_TYPE) {
      total_dust_mass += Sp->P[dust_idx].getMass();
    }
  }
  
  if(total_dust_mass < 1e-20) return;
  
  // Consume dust proportional to SF fraction
  double dust_to_consume = total_dust_mass * sf_fraction;
  
  // Distribute consumption among nearby dust particles (weighted by inverse distance)
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
    
    // Weight by inverse distance (closer dust more likely to be in SF region)
    double weight = (neighbor_distances[i] > 0) ? (1.0 / neighbor_distances[i]) : 1.0;
    double this_dust_fraction = weight / weight_sum;
    double mass_loss = dust_to_consume * this_dust_fraction;
    
    double current_mass = Sp->P[dust_idx].getMass();
    double new_mass = current_mass - mass_loss;
    
    if(new_mass < 1e-20) {
      // Destroy particle completely
      Sp->P[dust_idx].setMass(1e-30);
      Sp->P[dust_idx].setType(3);
      Sp->P[dust_idx].ID.set(0);
      memset(&Sp->DustP[dust_idx], 0, sizeof(dust_data));
      dust_consumed_count++;
      dust_consumed_mass += current_mass;
    } else {
      // Reduce mass
      Sp->P[dust_idx].setMass(new_mass);
      dust_consumed_mass += mass_loss;
    }
  }
  
  // Update global statistics
  NDustDestroyedByAstration += dust_consumed_count;
  TotalDustMassAstrated += dust_consumed_mass;
  
  static int astration_count = 0;
  astration_count++;
  if(astration_count % 100 == 0 && All.ThisTask == 0) {
    DUST_PRINT("[ASTRATION] Event #%d: SF=%.2e Msun, consumed %d dust (%.2e Msun)\n",
               astration_count, stellar_mass_formed, dust_consumed_count, dust_consumed_mass);
  }
}

/**
 * Update dust particle dynamics
 */
void update_dust_dynamics(simparticles *Sp, double dt, MPI_Comm Communicator)
{

    // TEMPORARY: Corruption detector
    int corrupted = 0;
    for(int i = 0; i < Sp->NumPart; i++) {
        if(Sp->P[i].getType() == 6) {
            if(Sp->DustP[i].DustTemperature < 10.0 || 
              Sp->DustP[i].GrainRadius <= 0.0 ||
              !isfinite(Sp->DustP[i].CarbonFraction)) {
                printf("[CORRUPTION_DETECTED] Step=%d Task=%d idx=%d ID=%lld T=%.1f a=%.3e C=%.3f\n",
                      All.NumCurrentTiStep, All.ThisTask, i,
                      (long long)Sp->P[i].ID.get(),
                      Sp->DustP[i].DustTemperature,
                      Sp->DustP[i].GrainRadius,
                      Sp->DustP[i].CarbonFraction);
                corrupted++;
            }
        }
    }
    if(corrupted > 0) {
        printf("[CORRUPTION_TOTAL] Step=%d Task=%d: %d corrupted dust particles\n",
              All.NumCurrentTiStep, All.ThisTask, corrupted);
    }

  // ============================================================
  // SYNCHRONIZE GlobalDustCount across all tasks FIRST
  // ============================================================
  long long local_count = GlobalDustCount;
  long long global_count = 0;
  MPI_Allreduce(&local_count, &global_count, 1, MPI_LONG_LONG, MPI_SUM, Communicator);
  
  // Only process dust every 10 steps
  if(All.NumCurrentTiStep % 10 != 0) {
    return;
  }

  // Quick exit if no dust particles exist yet
  if(global_count == 0) {
    return;
  }
  
  // ============================================================
  // NOW initialize timing and MPI (all tasks past early exits)
  // ============================================================
  
  static double total_time_in_dust = 0.0;
  static int dust_call_count = 0;
  double t_start = MPI_Wtime();
  
  int ThisTask;
  MPI_Comm_rank(Communicator, &ThisTask);

  int need_hash_rebuild = 0;
  if(All.ThisTask == 0) {
    if(!gas_hash.is_built) {
      need_hash_rebuild = 1;
    }
  }

  MPI_Bcast(&need_hash_rebuild, 1, MPI_INT, 0, Communicator);

  if(need_hash_rebuild) {
    if(All.ThisTask == 0) {
      DUST_PRINT("WARNING: Hash not built, building now for dust operations\n");
    }
    double typical_search_radius = 10.0;
    rebuild_feedback_spatial_hash(Sp, typical_search_radius);
  }

  // How are dust particles moving under gravity?
  /*
  if(All.NumCurrentTiStep % 100 == 0 && All.ThisTask == 0) {
    double max_accel = 0.0;
    int dust_with_accel = 0;
    
    for(int i = 0; i < Sp->NumPart; i++) {
      if(Sp->P[i].getType() == DUST_PARTICLE_TYPE && Sp->P[i].getMass() > 1e-20) {
        // Check if particle has gravitational acceleration
        double a2 = Sp->P[i].GravAccel[0]*Sp->P[i].GravAccel[0] +
                    Sp->P[i].GravAccel[1]*Sp->P[i].GravAccel[1] +
                    Sp->P[i].GravAccel[2]*Sp->P[i].GravAccel[2];
        
        if(a2 > 0) {
          dust_with_accel++;
          if(a2 > max_accel) max_accel = a2;
        }
      }
    }
    DUST_PRINT("[GRAVITY_CHECK] %d dust particles have nonzero GravAccel, max=%.3e\n",
              dust_with_accel, sqrt(max_accel));
  }
  */

  static bool verified = false;
  if(!verified && All.ThisTask == 0) {
    DUST_PRINT("=== HASH VERIFICATION ===\n");
    DUST_PRINT("  Hash built: %s\n", gas_hash.is_built ? "YES" : "NO");
    DUST_PRINT("  Cells per dim: %d\n", gas_hash.n_cells_per_dim);
    DUST_PRINT("  Cell size: %.3f kpc\n", gas_hash.cell_size);
    DUST_PRINT("  Total cells: %d^3 = %d\n", 
               gas_hash.n_cells_per_dim,
               gas_hash.n_cells_per_dim * gas_hash.n_cells_per_dim * gas_hash.n_cells_per_dim);
    DUST_PRINT("  Gas particles: %d\n", gas_hash.total_gas_particles);
    DUST_PRINT("=========================\n");
    verified = true;
  }

  // ============================================================================
  // DUST DYNAMICS LOOP (drag + grain growth)
  // ============================================================================
  // Process each dust particle: update timebins, apply drag, grow grains
  // ONE gas lookup per dust particle (used for both drag and growth)
  // ============================================================================
  int growth_attempts = 0;
  int growth_success = 0;
  double t_growth_start = MPI_Wtime();

  for(int i = 0; i < Sp->NumPart; i++) {
      if(Sp->P[i].getType() == DUST_PARTICLE_TYPE && Sp->P[i].getMass() > 1e-20) {
        double dist_kpc = -1.0;
        int nearest_gas = find_nearest_gas_particle(Sp, i, 5.0, &dist_kpc);
        
        if(nearest_gas >= 0) {
            // Drag + thermal coupling
            dust_gas_interaction(Sp, i, nearest_gas, dt * 10);
            
            if(dist_kpc <= 2.0) {
                // Accretion growth (depletes gas metals)
                if(All.DustEnableGrowth) {
                    dust_grain_growth_subgrid(Sp, i, nearest_gas, dt * 10);
                }
                
                // Coagulation growth (NO metal depletion)
                if(All.DustEnableCoagulation) {
                    dust_grain_coagulation(Sp, i, nearest_gas, dt * 10);
                }
            }
        }
      }
  }

  double t_growth_end = MPI_Wtime();
  double growth_time = t_growth_end - t_growth_start;

  // Diagnostics (every 500 steps)
  if(All.NumCurrentTiStep % 500 == 0 && All.ThisTask == 0) {
      if(All.DustEnableGrowth && growth_attempts > 0) {
          DUST_PRINT("[GROWTH_TIMING] Checked %d dust particles in %.3f sec (%.1f dust/sec)\n",
                    growth_attempts, growth_time, growth_attempts / growth_time);
          DUST_PRINT("[GROWTH_TIMING] Success rate: %d/%d = %.1f%%\n",
                    growth_success, growth_attempts, 
                    100.0 * growth_success / growth_attempts);
      }
  }

  // What temperature regimes are our dust particles in?
  if(All.NumCurrentTiStep % 500 == 0 && All.ThisTask == 0) {
    int in_hot_gas = 0;
    int in_warm_gas = 0;
    int in_cool_gas = 0;
    
    for(int i = 0; i < Sp->NumPart; i++) {
      if(Sp->P[i].getType() == DUST_PARTICLE_TYPE && Sp->P[i].getMass() > 1e-20) {
        double T = Sp->DustP[i].DustTemperature;
        if(T > 1e7) in_hot_gas++;
        else if(T > 1e6) in_warm_gas++;
        else in_cool_gas++;
      }
    }
    
    // How about the gas particles themselves?
    if(All.NumCurrentTiStep % 100 == 0 && All.ThisTask == 0) {
      
      double M_gas_cold = 0.0;   // T < 1e4 K
      double M_gas_warm = 0.0;   // 1e4-1e5 K  
      double M_gas_hot = 0.0;    // 1e5-1e6 K
      double M_gas_vhot = 0.0;   // >1e6 K
      
      for(int i = 0; i < Sp->NumGas; i++) {
        double T = get_temperature_from_entropy(Sp, i);
        double M = Sp->P[i].getMass();
        
        if(T < 1e4) M_gas_cold += M;
        else if(T < 1e5) M_gas_warm += M;
        else if(T < 1e6) M_gas_hot += M;
        else M_gas_vhot += M;
      }
      
      double M_total = M_gas_cold + M_gas_warm + M_gas_hot + M_gas_vhot;
      
      DUST_PRINT("[GAS_BUDGET] Cold (<1e4 K): %.1f%%\n", 100*M_gas_cold/M_total);
      DUST_PRINT("[GAS_BUDGET] Warm (1e4-1e5): %.1f%%\n", 100*M_gas_warm/M_total);
      DUST_PRINT("[GAS_BUDGET] Hot (1e5-1e6): %.1f%%\n", 100*M_gas_hot/M_total);
      DUST_PRINT("[GAS_BUDGET] Very hot (>1e6): %.1f%%\n", 100*M_gas_vhot/M_total);
    }

    // Let's also check density
    double n_bins[] = {0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0, 100.0};
    int n_count[8] = {0};
    double n_mass[8] = {0.0};
    
    for(int i = 0; i < Sp->NumGas; i++) {
      double rho_phys = Sp->SphP[i].Density * All.cf_a3inv;
      double rho_cgs = rho_phys * All.UnitDensity_in_cgs;
      double n_H = (rho_cgs * HYDROGEN_MASSFRAC) / PROTONMASS;
      
      for(int b = 0; b < 7; b++) {
        if(n_H >= n_bins[b] && n_H < n_bins[b+1]) {
          n_count[b]++;
          n_mass[b] += Sp->P[i].getMass();
          break;
        }
      }
      if(n_H >= n_bins[7]) {
        n_count[7]++;
        n_mass[7] += Sp->P[i].getMass();
      }
    }
    
    DUST_PRINT("[DENSITY_DISTRIBUTION] of gas at z=%.3f\n", 1.0/All.Time - 1.0);
    for(int b = 0; b < 8; b++) {
      if(b < 7) {
        DUST_PRINT("  %.3f-%.2f cm^-3: %6d particles, M=%.3e (%.1f%%)\n",
                  n_bins[b], n_bins[b+1], n_count[b], n_mass[b],
                  100.0*n_count[b]/Sp->NumGas);
      } else {
        DUST_PRINT("  >%.1f cm^-3:      %6d particles, M=%.3e (%.1f%%)\n",
                  n_bins[7], n_count[7], n_mass[7],
                  100.0*n_count[7]/Sp->NumGas);
      }
    }

  }


  if(All.NumCurrentTiStep % 500 == 0)
  {
    print_dust_statistics(Sp);
    analyze_dust_gas_coupling_global(Sp);
    analyze_grain_size_distribution(Sp);
    
    if(All.ThisTask == 0) {
      DUST_PRINT("[GROWTH_SUMMARY] Total growth events so far: %lld\n", NGrainGrowthEvents);
    }
  }

  // Dust particle check (only every 500 steps)
  if(All.ThisTask == 0 && All.NumCurrentTiStep % 500 == 0) {
    int printed = 0;
    for(int i=0; i<Sp->NumPart && printed<3; i++) {
      if(Sp->P[i].getType() == DUST_PARTICLE_TYPE && Sp->P[i].getMass() > 1e-20) {
        DUST_PRINT("[CHECK] i=%d ID=%lld M=%.2e a=%.2f nm CF=%.2f GT=%d\n",
          i, (long long)Sp->P[i].ID.get(), Sp->P[i].getMass(),
          Sp->DustP[i].GrainRadius, Sp->DustP[i].CarbonFraction, Sp->DustP[i].GrainType);
        printed++;
      }
    }
  }

  double t_end = MPI_Wtime();
  double dt_dust = t_end - t_start;
  total_time_in_dust += dt_dust;
  dust_call_count++;
  
  if(dust_call_count % 100 == 0 && All.ThisTask == 0) {
    printf("[DUST_TIMING] Called %d times, avg %.3f sec/call, total %.1f sec\n",
           dust_call_count, total_time_in_dust/dust_call_count, total_time_in_dust);
  }

}

/**
 * Remove destroyed dust particles and compact array
 * POTENTIALLY NO LONGER USING THIS!!!
 */
void destroy_dust_particles(simparticles *Sp)
{
  int dust_destroyed = 0;
  
  // Be sure not to find a dust particle that has already been destroyed (1e-30 in mass)
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
        // Gas stays in place - still zero DustP in case this slot previously had dust
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

    // Handle DustP for ALL particles, even when new_num_part == i
    if(Sp->P[i].getType() == DUST_PARTICLE_TYPE) {
      if(new_num_part != i) {
        Sp->DustP[new_num_part] = Sp->DustP[i];
      }
      // else: dust stays in place, DustP already correct
    } else {
      // Non-dust: ALWAYS zero, even if not moved!
      memset(&Sp->DustP[new_num_part], 0, sizeof(dust_data));
    }

    new_num_part++;
  }
  
  Sp->NumPart = new_num_part;
  Sp->NumGas = new_num_gas;
}

/**
 * Find nearest gas particle to dust using spatial hash
 * Much faster than O(N) brute force for large simulations!
 * @param Sp            simulation particles
 * @param dust_idx      dust particle index
 * @param max_r_kpc     max search radius (kpc)
 * @param out_dist_kpc  optional output distance (kpc). If no gas found, set to -1.
 * @return nearest gas index, or -1 if none found within max_r_kpc
 */
int find_nearest_gas_particle(simparticles *Sp, int dust_idx,
                              double max_r_kpc, double *out_dist_kpc)
{
  if(out_dist_kpc) *out_dist_kpc = -1.0;

  if(Sp->NumGas == 0) return -1;
  if(max_r_kpc <= 0)  return -1;

  // Prefer hash if built
  if(gas_hash.is_built) {
    HashSearches++;

    double nearest_dist = -1.0;
    int nearest = gas_hash.find_nearest_gas_particle(Sp, dust_idx, max_r_kpc, &nearest_dist);

    // Defensive: treat "no index" or "bad distance" or "outside search radius" as not found
    if(nearest < 0 || !(nearest_dist >= 0) || nearest_dist > max_r_kpc) {
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
 * Print dust statistics
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
      avg_grain_size += Sp->DustP[i].GrainRadius;  // nm
      avg_temperature += Sp->DustP[i].DustTemperature;
    }
  }
  
  if(dust_count > 0) {
    avg_grain_size /= dust_count;
    avg_temperature /= dust_count;
  }

  // Analyze dust temperature with realistic bins
  int cmb_floor = 0;      // T < 10 K (CMB-limited, high-z)
  int very_cold = 0;      // 10 K < T < 50 K (cold molecular clouds)
  int cold_ism = 0;       // 50 K < T < 100 K (typical ISM)
  int warm_ism = 0;       // 100 K < T < 500 K (warm neutral medium)
  int hot_ism = 0;        // 500 K < T < 1000 K (ionized regions)
  int pre_sublimation = 0; // 1000 K < T < 2000 K (near destruction)

  for(int i = 0; i < Sp->NumPart; i++) {
    if(Sp->P[i].getType() == DUST_PARTICLE_TYPE && Sp->P[i].getMass() > 1e-20) {
      double T = Sp->DustP[i].DustTemperature;
      
      if(T < 10.0) cmb_floor++;
      else if(T < 50.0) very_cold++;
      else if(T < 100.0) cold_ism++;
      else if(T < 500.0) warm_ism++;
      else if(T < 1000.0) hot_ism++;
      else pre_sublimation++;  // 1000-2000 K
    }
  }
  
  DUST_PRINT("=== STATISTICS (rank 0) ===\n");
  DUST_PRINT("STATISTICS Particles: %d  Mass: %.3e Msun\n", dust_count, total_dust_mass);
  DUST_PRINT("STATISTICS Avg grain size: %.2f nm\n", avg_grain_size);
  DUST_PRINT("STATISTICS Avg temperature: %.1f K\n", avg_temperature);
  DUST_PRINT("STATISTICS  < 10 K (CMB floor):        %d (%.1f%%)\n", cmb_floor, 100.0*cmb_floor/dust_count);
  DUST_PRINT("STATISTICS  10-50 K (Cold clouds):     %d (%.1f%%)\n", very_cold, 100.0*very_cold/dust_count);
  DUST_PRINT("STATISTICS  50-100 K (Cool ISM):       %d (%.1f%%)\n", cold_ism, 100.0*cold_ism/dust_count);
  DUST_PRINT("STATISTICS  100-500 K (Warm ISM):      %d (%.1f%%)\n", warm_ism, 100.0*warm_ism/dust_count);
  DUST_PRINT("STATISTICS  500-1000 K (Hot ISM):      %d (%.1f%%)\n", hot_ism, 100.0*hot_ism/dust_count);
  DUST_PRINT("STATISTICS  1000-2000 K (Near sublim): %d (%.1f%%)\n", pre_sublimation, 100.0*pre_sublimation/dust_count);
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
  DUST_PRINT("STATISTICS Destroyed by thermal: %lld\n", NDustDestroyedByThermal);
  DUST_PRINT("STATISTICS Destroyed by shocks: %lld\n", NDustDestroyedByShock);
  DUST_PRINT("STATISTICS Total mass eroded: %.2e Msun\n", TotalMassEroded);
  DUST_PRINT("STATISTICS Destroyed by astration: %lld (%.2e Msun)\n",
           NDustDestroyedByAstration, TotalDustMassAstrated);
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
      int nearest_gas = find_nearest_gas_particle(Sp, i, 5.0, NULL);  // 5 kpc max
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
  
  // Use spatial hash for nearby gas search
  double nearest_dist = -1.0;
  int nearest_gas = gas_hash.find_nearest_gas_particle(Sp, sn_star_idx, 50.0, &nearest_dist);
  
  double gas_density_cgs = 1.0 * PROTONMASS;  // Conservative default

  if(nearest_gas >= 0 && nearest_gas < Sp->NumGas) {
    double gas_density_code = Sp->SphP[nearest_gas].Density * All.cf_a3inv;
    double measured = gas_density_code * All.UnitDensity_in_cgs;
    
    // Only reject true vacuum (IGM)
    if(measured > 0.01 * PROTONMASS) {
      gas_density_cgs = measured;
    }
  }
  
  // Use realistic Sedov-Taylor time: 3-5 Myr (peak of shock destruction phase)
  double characteristic_time_myr = 3.0;
  
  double radius = calculate_sn_shock_radius(sn_energy_erg, gas_density_cgs, 
                                           characteristic_time_myr);
  
  // Realistic bounds for SN shocks
  if(radius < 1.0) radius = 1.0;   // 1 kpc minimum
  if(radius > 10.0) radius = 10.0; // 10 kpc maximum
  
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

  /**
   * Get dust destruction efficiency from shock velocity
   * 
   * Based on:
   * - Jones et al. (1996): Grain shattering physics
   * - Bocchio et al. (2014): Modern destruction models
   * - Slavin et al. (2015): SNR shock observations
   * - Hu et al. (2019): Cosmological implementation
   * 
   * Physics: Grain destruction dominated by:
   * - Sputtering (thermal + kinetic) at v < 200 km/s
   * - Shattering (grain-grain collisions) at v > 200 km/s
   * - Complete destruction at v > 500 km/s
   */
  double get_shock_destruction_efficiency(double shock_velocity_km_s)
  {
    // Below threshold: thermal sputtering only (minimal destruction)
    if(shock_velocity_km_s < 50.0) {
      return 0.0;
    }
    // Threshold to moderate shocks (50-100 km/s): sputtering + early shattering
    else if(shock_velocity_km_s < 100.0) {
      // 0% at 50 km/s → 15% at 100 km/s (Jones+96, Bocchio+14)
      return 0.15 * (shock_velocity_km_s - 50.0) / 50.0;
    }
    // Moderate shocks (100-200 km/s): significant shattering begins
    else if(shock_velocity_km_s < 200.0) {
      // 15% at 100 km/s → 50% at 200 km/s (Bocchio+14, Slavin+15)
      return 0.15 + 0.35 * (shock_velocity_km_s - 100.0) / 100.0;
    }
    // Strong shocks (200-400 km/s): efficient grain destruction
    else if(shock_velocity_km_s < 400.0) {
      // 50% at 200 km/s → 80% at 400 km/s (Jones+96, Hu+19)
      return 0.50 + 0.30 * (shock_velocity_km_s - 200.0) / 200.0;
    }
    // Very strong shocks (400-500 km/s): near-complete destruction
    else if(shock_velocity_km_s < 500.0) {
      // 80% at 400 km/s → 95% at 500 km/s
      return 0.80 + 0.15 * (shock_velocity_km_s - 400.0) / 100.0;
    }
    // Extremely fast shocks (>500 km/s): essentially complete destruction
    else {
      return 0.95;  // 95% cap (some refractory grains may survive)
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
  
  // STEP 1: Get shock radius (with clamping)
  double shock_radius_kpc = calculate_current_sn_shock_radius(Sp, sn_star_idx);
  //if(shock_radius_kpc < 1.0) shock_radius_kpc = 0.1; // floor; maybe don't need this?
  if(shock_radius_kpc > 5.0) shock_radius_kpc = 5.0;

  // STEP 2: Calculate shock velocity from radius
  const double characteristic_time_myr = 3.0;
  const double time_sec = characteristic_time_myr * 1e6 * SEC_PER_YEAR;
  const double radius_cm = shock_radius_kpc * 1000.0 * PARSEC;
  
  double shock_velocity = (2.0/5.0) * radius_cm / time_sec;  // cm/s
  shock_velocity /= 1e5;  // Convert to km/s

  // ========================================================================
  // OPTIMIZATION: Use spatial hash for gas search
  // ========================================================================
  double nearest_dist = -1.0;
  int nearest_gas = gas_hash.find_nearest_gas_particle(Sp, sn_star_idx, 
                                                        50.0, &nearest_dist);
  
  if(sn_call_count <= 10 && All.ThisTask == 0) {
    DUST_PRINT("[SN_SHOCK_DEBUG] Call #%d: shock_radius=%.2f kpc, shock_velocity=%.1f km/s, nearest_gas_dist=%.2f kpc\n",
               sn_call_count, shock_radius_kpc, shock_velocity, nearest_dist);
  }

  // ========================================================================
  // OPTIMIZATION: Only loop over DUST particles (not all particles)
  // ========================================================================
  int dust_in_shock  = 0;
  int dust_destroyed = 0;
  int dust_eroded    = 0;
  
  // Much faster: only check dust particles, skip gas/stars/etc
  for(int i = 0; i < Sp->NumPart; i++) {
    // Quick rejection: skip non-dust immediately
    if(Sp->P[i].getType() != DUST_PARTICLE_TYPE) continue;
    if(Sp->P[i].getMass() <= 1e-20) continue;  // Already flagged for deletion
    
    // Calculate distance to SN
    double dxyz[3];
    Sp->nearest_image_intpos_to_pos(Sp->P[i].IntPos, Sp->P[sn_star_idx].IntPos, dxyz);
    double distance = sqrt(dxyz[0]*dxyz[0] + dxyz[1]*dxyz[1] + dxyz[2]*dxyz[2]);
    
    // Early exit if outside shock radius (most particles)
    if(distance >= shock_radius_kpc) continue;
    
    // Dust is inside shock - apply destruction
    dust_in_shock++;
    
    int destroyed = erode_dust_grain_shock(Sp, i, shock_velocity, 
                                           distance, shock_radius_kpc);
    
    if(destroyed) {
      dust_destroyed++;
    } else {
      dust_eroded++;
    }
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
 * Find nearest dust particle to a gas cell
 */
int find_nearest_dust_particle(simparticles *Sp, int gas_idx)
{
  double min_dist2 = 1e30;
  int nearest_dust = -1;
  
  // LIMIT SEARCH RADIUS!
  double search_radius_kpc = 10.0;  // Only search within 10 kpc
  
  for(int i = 0; i < Sp->NumPart; i++) {
    if(Sp->P[i].getType() == DUST_PARTICLE_TYPE && 
       Sp->P[i].getMass() > 1e-20) {
      
      double dxyz[3];
      Sp->nearest_image_intpos_to_pos(Sp->P[gas_idx].IntPos, 
                                       Sp->P[i].IntPos, dxyz);
      
      double r2 = dxyz[0]*dxyz[0] + dxyz[1]*dxyz[1] + dxyz[2]*dxyz[2];
      
      // Early exit if too far
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
  // HK11 eqs. (23),(24) with individual scalings for silicate and carbonaceous grains
  const double pref = (species==0) ? 6.30e7 : 5.59e7; // yr

  if(nH_cm3 <= 0 || T_K <= 0 || Z_massfrac <= 0 || Zsun_massfrac <= 0 || a_cm <= 0 || S <= 0)
    return HUGE_VAL;

  const double a_um = a_cm * 1e4;          // cm -> micron
  const double a01  = a_um / 0.1;          // a_{0.1}
  const double n3   = nH_cm3 / 1e3;        // n_3
  const double T50  = T_K / 50.0;          // T_50
  const double Zrat = Z_massfrac / Zsun_massfrac; // Z/Zsun
  const double S03  = S / 0.3;             // S_0.3

  return pref * a01 * (1.0/Zrat) * (1.0/n3) * pow(T50, -0.5) * (1.0/S03);
}

// Simple dust clumping factor model; denser gas = higher clumping;
// See Hopkins & Lee 2016 for discussion.
// Bringing in CritPhysDensity allows for calibration to the SF threshold to ensure 
// that the intermediate levels get called. Necessary for different high-res vs low-res zoom simulations.
double dust_clumping_factor(double n_H, int is_star_forming)
{
    if(!All.DustEnableClumping) return 1.0;
    
    if(is_star_forming) return 30.0;
    
    // At 1024^3, typical SF threshold is ~0.02 cm^-3
    // Scale thresholds relative to SF threshold
    const double n_sf = All.CritPhysDensity;  // Current SF threshold
    
    if(n_H > 0.5 * n_sf)  return 10.0;   // 50% of SF threshold
    if(n_H > 0.2 * n_sf)  return 3.0;    // 20% of SF threshold  
    if(n_H > 0.05 * n_sf) return 1.5;    // 5% of SF threshold
    
    return 1.0;
}

/**
 * Dust grain coagulation in dense gas
 * 
 * PHYSICS:
 *  - Grains collide and stick in dense molecular clouds
 *  - Timescale: τ_coag ~ (n_dust * σ_coll * v_rel)^-1
 *  - Ormel et al. 2007, Okuzumi et al. 2012: τ_coag ~ 1-100 Myr in GMCs
 * 
 * KEY DIFFERENCE FROM ACCRETION:
 *  - Coagulation: Grain + Grain → Bigger Grain (NO metal depletion from gas!)
 *  - Accretion: Gas Metals → Grain Growth (DOES deplete gas metals)
 * 
 * @param Sp Simulation particles
 * @param dust_idx Index of dust particle
 * @param gas_idx Index of nearest gas particle
 * @param dt Timestep in code units
 */
void dust_grain_coagulation(simparticles *Sp, int dust_idx, int gas_idx, double dt)
{
  if(!All.DustEnableCoagulation) return;
  
  // ============================================================
  // STEP 1: Check if conditions are right for coagulation
  // ============================================================
  
  // Get gas properties
  double gas_density_code = Sp->SphP[gas_idx].Density * All.cf_a3inv;
  double gas_density_cgs = gas_density_code * All.UnitDensity_in_cgs;
  double n_H = (gas_density_cgs * HYDROGEN_MASSFRAC) / PROTONMASS;
  
  // Coagulation only significant in dense gas (GMCs: n > 100 cm^-3)
  if(n_H < All.DustCoagulationDensityThresh) return;
  
  double T_gas = get_temperature_from_entropy(Sp, gas_idx);
  
  // Coagulation requires cold gas (grains stick, not bounce)
  // Above ~100 K, grains bounce off each other instead of sticking
  if(T_gas > 100.0) return;
  
  // Get current grain properties
  double a = Sp->DustP[dust_idx].GrainRadius;  // nm
  double M_dust = Sp->P[dust_idx].getMass();   // code units
  
  // Sanity checks
  if(a <= 0.0 || M_dust <= 0.0 || !isfinite(a) || !isfinite(M_dust)) return;
  
  // Don't coagulate if already at maximum size
  if(a >= All.DustCoagulationMaxSize) return;
  
  // ============================================================
  // STEP 2: Calculate coagulation timescale
  // ============================================================
  
  // Physics: τ_coag = 1 / (n_dust * σ * v_rel)
  // where:
  //   n_dust = dust number density
  //   σ = collision cross section ~ π * (2a)^2
  //   v_rel = relative velocity between grains ~ sqrt(3 * k * T / m_grain)
  //
  // Empirical fit from Ormel et al. 2007, Okuzumi et al. 2012:
  //   τ_coag ~ 10^7 yr * (100 cm^-3 / n_H) * (0.1 μm / a)
  //
  // This captures the density and size dependencies
  
  double a_micron = a / 1000.0;  // Convert nm to microns
  
  // Fiducial timescale: 10 Myr at n=100 cm^-3, a=0.1 μm
  double tau_coag_yr = 1e7 * (100.0 / n_H) * (0.1 / a_micron);
  
  // Apply calibration factor (allows tuning)
  tau_coag_yr *= All.DustCoagulationCalibration;
  
  // Apply reasonable limits
  if(tau_coag_yr < 1e6)  tau_coag_yr = 1e6;   // 1 Myr floor
  if(tau_coag_yr > 1e9)  tau_coag_yr = 1e9;   // 1 Gyr ceiling
  
  // Convert to code units
  double tau_coag = tau_coag_yr * SEC_PER_YEAR / All.UnitTime_in_s;
  
  // ============================================================
  // STEP 3: Calculate mass growth from coagulation
  // ============================================================
  
  // Exponential growth: dM/dt = M / τ_coag
  // Solution: M(t) = M_0 * exp(t / τ_coag)
  // For small dt: M_new ≈ M_0 * (1 + dt/τ_coag)
  
  double growth_factor = 1.0 + (dt / tau_coag);
  
  // For stability, cap growth per timestep at 20%
  if(growth_factor > 1.2) growth_factor = 1.2;
  
  double M_new = M_dust * growth_factor;
  double dM = M_new - M_dust;
  
  // Sanity check
  if(!isfinite(dM) || dM <= 0.0) return;
  
  // ============================================================
  // STEP 4: Calculate new grain radius
  // ============================================================
  
  // Assume constant grain material density (ρ_grain ~ 2-3 g/cm^3)
  // Mass scales as a^3, so: M_new/M_old = (a_new/a_old)^3
  
  double size_ratio = pow(M_new / M_dust, 1.0/3.0);
  double a_new = a * size_ratio;
  
  // Cap at maximum size
  if(a_new > All.DustCoagulationMaxSize) {
    a_new = All.DustCoagulationMaxSize;
    
    // Recalculate consistent mass for capped size
    size_ratio = a_new / a;
    M_new = M_dust * pow(size_ratio, 3.0);
    dM = M_new - M_dust;
  }
  
  // Sanity checks
  if(!isfinite(a_new) || a_new <= a || a_new > All.DustCoagulationMaxSize) return;
  
  // ============================================================
  // STEP 5: Apply changes
  // ============================================================
  
  Sp->P[dust_idx].setMass(M_new);
  Sp->DustP[dust_idx].GrainRadius = a_new;
  
  // CRITICAL: Gas metallicity is UNCHANGED!
  // Coagulation is grain + grain → bigger grain
  // No metal depletion from gas phase
  
  // ============================================================
  // STEP 6: Update statistics
  // ============================================================
  
  NCoagulationEvents++;
  TotalMassCoagulated += dM;
  
  // Diagnostic output (sample 1 in 10,000)
  static int coag_samples = 0;
  if(coag_samples < 100 && All.ThisTask == 0 && Sp->P[dust_idx].ID.get() % 10000 == 0) {
    DUST_PRINT("[COAGULATION] Event #%d: a=%.1f→%.1f nm, M=%.3e→%.3e Msun, "
               "n_H=%.1f cm^-3, T=%.0f K, tau_coag=%.1f Myr\n",
               coag_samples, a, a_new, M_dust, M_new,
               n_H, T_gas, tau_coag_yr/1e6);
    coag_samples++;
  }
  
  // Print summary statistics every 10,000 events
  if(NCoagulationEvents % 10000 == 0 && All.ThisTask == 0) {
    DUST_PRINT("[COAGULATION_STATS] Events: %lld, Total mass grown: %.3e Msun\n",
               NCoagulationEvents, TotalMassCoagulated);
  }
}

/**
 * Subgrid grain growth model (HK11-based)
 *
 * Uses density, metallicity, and star formation to estimate molecular fraction,
 * then applies Hirashita & Kuo (2011) accretion timescale scalings.
 *
 * PHYSICAL MODEL:
 *  - Grain growth requires gas-phase metals to accrete onto dust
 *  - Growth is fastest in molecular clouds (H2 dominates, metals stick)
 *  - Growth rate: da/dt = f_mol * a / tau_acc (HK11 eq. 22)
 *  - f_mol = molecular fraction (where growth happens)
 *  - tau_acc = accretion timescale (depends on density, T, metallicity)
 *
 * * ACCRETION EFFICIENCY:
 *  - Grain size evolution includes accretion AND coagulation
 *  - Only accretion removes metals from gas (affects cooling/SF)
 *  - accretion_efficiency = fraction of size growth from accretion
 *  - See detailed comments below for tuning guidance
 * 
 * ASSUMPTIONS (since we don't track individual gas-phase elements):
 *  - Available metals for growth = bulk metallicity Z
 *  - Species (carbonaceous vs silicate) determined by dust CarbonFraction:
 *       CF >= 0.5 → carbonaceous (HK11 species=1)
 *       CF <  0.5 → silicate     (HK11 species=0)
 *
 * NUMERICS:
 *  - Grow grain radius: da/dt = f_mol * a / tau_acc
 *  - Convert to mass growth for superparticle: dm = M_dust * (3 da / a)
 *  - Enforce: (i) max dust-to-metal ratio, (ii) max dm per step for stability
 */
void dust_grain_growth_subgrid(simparticles *Sp, int dust_idx, int gas_idx, double dt)
{
  if(!All.DustEnableGrowth) return;
  
  // This is the fraction of growth that goes to accretion vs coagulation.
  // Lowering this number should let grains grow in size (coagulation) without removing as many metals from gas-phase cooling.
  const double accretion_efficiency = 1.0;  // PROBABLY DO NOT NEED THIS ANY MORE NOW THAT COAGULATION IS STAND-ALONE

  // --------------------------
  // Diagnostics counters
  // --------------------------
  static int total_calls        = 0;
  static int failed_hot         = 0;
  static int failed_no_metals   = 0;
  static int failed_low_fmol    = 0;
  static int failed_no_dust     = 0;
  static int failed_too_far     = 0;
  static int failed_max_dz      = 0;
  static int failed_bad_tau     = 0;
  static int passed_all         = 0;

  static int used_species_sil   = 0;
  static int used_species_carb  = 0;
  
  // Track f_mol distribution for diagnostics
  static int fmol_diffuse = 0;   // f_mol = 0.05
  static int fmol_moderate = 0;  // f_mol = 0.2
  static int fmol_dense = 0;     // f_mol = 0.5
  static int fmol_sf = 0;        // f_mol = 0.8

  total_calls++;

  // Print summary occasionally (rank 0 only)
  if(total_calls % 5000 == 0 && All.ThisTask == 0) {
    DUST_PRINT("=== HK11 GROWTH DIAGNOSTICS (after %d attempts) Rank 0 only ===\n", total_calls);
    DUST_PRINT("  Accretion efficiency: %.2f (%.0f%% accretion, %.0f%% coagulation)\n",
               accretion_efficiency, 
               accretion_efficiency*100, 
               (1-accretion_efficiency)*100);
    DUST_PRINT("  Failed hot (T>%.0e K):    %6d (%.1f%%)\n",
               All.DustThermalSputteringTemp, failed_hot, 100.0*failed_hot/total_calls);
    DUST_PRINT("  Failed no metals:        %6d (%.1f%%)\n",
               failed_no_metals, 100.0*failed_no_metals/total_calls);
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
    DUST_PRINT("  f_mol distribution:\n");
    DUST_PRINT("    Diffuse (0.05): %d  Moderate (0.2): %d  Dense (0.5): %d  SF (0.8): %d\n",
               fmol_diffuse, fmol_moderate, fmol_dense, fmol_sf);
    DUST_PRINT("  Total mass grown:        %.3e Msun\n", TotalMassGrown);
    DUST_PRINT("  Growth events:           %lld\n", NGrainGrowthEvents);
    if(NGrainGrowthEvents > 0)
      DUST_PRINT("  Avg mass per event:      %.3e Msun\n", TotalMassGrown / NGrainGrowthEvents);
    DUST_PRINT("===============================================\n");
  }

  // --------------------------
  // Basic gas filters
  // --------------------------
  const double T_gas = get_temperature_from_entropy(Sp, gas_idx);
  const double Z_gas = Sp->SphP[gas_idx].Metallicity;   // mass fraction

  // Don't grow in hot gas (sputtering dominates over growth)
  if(T_gas > All.DustThermalSputteringTemp) {
    failed_hot++;
    return;
  }

  // Need some metals for growth
  if(Z_gas < 1e-4) {
    failed_no_metals++;
    return;
  }


  // --------------------------
  // Clumping
  // --------------------------
  // Get gas number density to infer molecular fraction
  const double SFR_EPS = 1e-14;
  double gas_density_code = Sp->SphP[gas_idx].Density * All.cf_a3inv;  // Physical density
  double gas_density_cgs = gas_density_code * All.UnitDensity_in_cgs;
  double n_H = (gas_density_cgs * HYDROGEN_MASSFRAC) / PROTONMASS;  // Hydrogen number density [cm^-3]
  double DustClumpingFactor = dust_clumping_factor(n_H, Sp->SphP[gas_idx].Sfr > SFR_EPS); // "C"
  double n_eff_cm3 = n_H * DustClumpingFactor;  // see Hopkins 2016 Table 1 for C values
  if(n_eff_cm3 < 0.1) return;  // Skip growth in very diffuse gas

  // --------------------------
  // Molecular fraction estimate (f_mol)
  // --------------------------
  // Physical reasoning:
  // - Grain growth requires gas-phase metals to stick to dust
  // - This happens primarily in molecular (H2) gas, not atomic (HI) gas
  // - f_mol = fraction of gas that is molecular (where growth occurs)
  // - Higher density → more H2 formation (self-shielding)
  // - Star formation → indicates dense, molecular environment
  // - Higher metallicity → more dust shielding → easier H2 formation
  //
  // References:
  // - Krumholz et al. 2009: H2 fraction vs. density and metallicity
  // - Asano et al. 2013: Grain growth in molecular clouds
  // - Hirashita & Kuo 2011: Subgrid dust model implementation
  
  double f_mol = 0.05;  // Baseline: diffuse atomic ISM (HI-dominated)

  #ifdef STARFORMATION
    if(Sp->SphP[gas_idx].Sfr > SFR_EPS) {
      // Star-forming regions: very molecular (n >> 100 cm^-3, strong shielding)
      // Typical f_mol ~ 0.7-0.9 in star-forming cores (McKee & Ostriker 2007)
      f_mol = 0.8;
      fmol_sf++;
    } else if(n_eff_cm3 > 100.0) {
      // Dense molecular clouds: n > 100 cm^-3
      // Self-shielding allows H2 formation even without active SF
      // Typical f_mol ~ 0.3-0.7 (Glover & Mac Low 2011)
      f_mol = 0.5;
      fmol_dense++;
    } else if(n_eff_cm3 > 10.0) {
      // Moderately dense gas: 10 < n < 100 cm^-3
      // Transitional regime between atomic and molecular
      // Typical f_mol ~ 0.1-0.3 (depends on radiation field)
      f_mol = 0.2;
      fmol_moderate++;
    } else {
      // Diffuse ISM: n < 10 cm^-3, mostly atomic
      fmol_diffuse++;
    }
  #else
    // Without SF tracking, use density alone as proxy
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

  // Metallicity enhancement: more metals → more dust → better H2 shielding
  // At Z > Z_sun, expect f_mol to be ~50% higher for same density
  // (Krumholz et al. 2011 metallicity scaling)
  if(Z_gas > 0.01) {  // Above solar metallicity
    f_mol *= 1.5;
    if(f_mol > 1.0) f_mol = 1.0;  // Physical cap
  }

  // Safety: if molecular fraction is negligible, skip growth
  // (Avoids wasting time on diffuse atomic gas with f_mol < 1%)
  if(f_mol < 0.01) {
    failed_low_fmol++;
    return;
  }

  // --------------------------
  // Find a nearby dust particle to grow
  // --------------------------
  // We represent dust as superparticles: one particle = population of grains
  // Grow the nearest dust particle to this gas cell
  const int nearest_dust = dust_idx;
  if(nearest_dust < 0) {
    failed_no_dust++;
    return;
  }

  // Distance cut: dust must be close to gas for growth to occur
  // (Prevents growing dust in halo that is nowhere near this gas cell)
  double dxyz[3];
  Sp->nearest_image_intpos_to_pos(Sp->P[gas_idx].IntPos,
                                  Sp->P[nearest_dust].IntPos, dxyz);
  const double dist_kpc = sqrt(dxyz[0]*dxyz[0] + dxyz[1]*dxyz[1] + dxyz[2]*dxyz[2]);

  if(dist_kpc > 5.0) {  // 5 kpc maximum separation
    failed_too_far++;
    return;
  }

  // --------------------------
  // Read dust state
  // --------------------------
  const double a = Sp->DustP[nearest_dust].GrainRadius;   // Grain radius [nm]
  const double M_dust = Sp->P[nearest_dust].getMass();     // Total dust mass [code units]

  // Sanity checks (catch corrupted particles)
  if(a <= 0.0 || !isfinite(a) || M_dust <= 0.0 || !isfinite(M_dust)) {
    if(All.ThisTask == 0) {
      DUST_PRINT("[HK11_GROWTH] Invalid dust state: idx=%d a=%.3e M=%g type=%d ID=%lld\n",
                 nearest_dust, a, M_dust, Sp->P[nearest_dust].getType(),
                 (long long)Sp->P[nearest_dust].ID.get());
    }

    // Remove corrupted particle
    Sp->P[nearest_dust].setMass(1e-30);
    Sp->P[nearest_dust].ID.set(0);
    Sp->P[nearest_dust].setType(3);
    memset(&Sp->DustP[nearest_dust], 0, sizeof(dust_data));
    Sp->DustP[nearest_dust].GrainRadius = DUST_MIN_GRAIN_SIZE;
    return;
  }

  // --------------------------
  // Determine grain species (carbonaceous vs silicate)
  // This just changed the constant in the HK11 tau_acc formula
  // which makes carbon grains grow ~11% faster than silicates
  // --------------------------
  const double CF = Sp->DustP[nearest_dust].CarbonFraction;
  const int species = (CF >= 0.5) ? 1 : 0;   // 1=carbonaceous, 0=silicate
  if(species == 1) used_species_carb++; else used_species_sil++;

  // --------------------------
  // Calculate accretion timescale (HK11 formula)
  // --------------------------
  // Subgrid assumptions for where growth occurs (unresolved molecular clouds):
  // - n_eff = 100 cm^-3 (typical GMC density)
  // - T_eff = 20 K (cold molecular gas)


  //if(n_eff_cm3 < 100.0) n_eff_cm3 = 100.0;          // keep HK11 baseline
  //if(n_eff_cm3 > 1e4)   n_eff_cm3 = 1e4;
  const double T_eff_K   = 20.0;    // Effective temperature in cold GMCs

  const double Zsun_massfrac = 0.02;  // Solar metallicity (mass fraction)
  const double S_stick = 0.3;         // Sticking coefficient (HK11 default)


  // Minimum density for growth timescale calculation
  // At low resolution, we represent volume-averaged GMCs where growth occurs
  // Real molecular cloud cores have n ~ 100-10,000 cm^-3
  // Use a floor to prevent unrealistically slow growth timescales
  double n_eff_for_growth = n_eff_cm3;
  if(n_eff_for_growth < 100.0) n_eff_for_growth = 100.0;  // Set floor; will naturally become less important as sim resolution is increased
                                                          // at 10^6M, n_eff = 1-3; at 10^4M, n_eff = 300-3000 with clumping

  // HK11 accretion timescale: tau_acc(a, Z, n, T, species)
  double a_cm = a * 1e-7;  // Convert nm to cm for HK11 formula
  double tau_acc_yr = tau_acc_yr_HK11(n_eff_for_growth, T_eff_K,
                                      Z_gas, Zsun_massfrac,
                                      a_cm, S_stick, species);

  // Apply calibration factor (allows tuning growth efficiency)
  // DustGrowthCalibration = 1.0 uses HK11 formula as-is
  // < 1.0 makes growth faster, > 1.0 makes growth slower
  tau_acc_yr *= All.DustGrowthCalibration;

  // Sanity limits on tau_acc (prevent numerical issues)
  if(!isfinite(tau_acc_yr) || tau_acc_yr <= 0.0) {
    failed_bad_tau++;
    return;
  }
  if(tau_acc_yr < 1e6)  tau_acc_yr = 1e6;   // 1 Myr floor
  if(tau_acc_yr > 5e9)  tau_acc_yr = 5e9;   // 5 Gyr ceiling

  // Convert to code time units
  const double tau_acc_code = tau_acc_yr * SEC_PER_YEAR / All.UnitTime_in_s;

  // --------------------------
  // Calculate radius growth
  // --------------------------
  // Growth equation: da/dt = f_mol * a / tau_acc
  // Only grow in molecular fraction of gas (f_mol)
  const double da_dt = f_mol * a / tau_acc_code;   // [cm / code time]
  double da = da_dt * dt;
  if(!isfinite(da) || da <= 0.0) return;

  // Cap at maximum grain size (200 nm)
  double a_new = a + da;
  if(a_new > DUST_MAX_GRAIN_SIZE) {
    a_new = DUST_MAX_GRAIN_SIZE;
    da = a_new - a;
    if(da <= 0.0) return;
  }

  // Prevent going below minimum size
  if(a_new < DUST_MIN_GRAIN_SIZE || a_new > DUST_MAX_GRAIN_SIZE) return;

  // --------------------------
  // Convert radius growth to mass growth
  // --------------------------
  // For small da: dm/M = 3 da/a (from spherical geometry)
  // This assumes the dust superparticle represents a population
  double dm = M_dust * (3.0 * da / a) * accretion_efficiency;
  if(!isfinite(dm) || dm <= 0.0) return;

  // --------------------------
  // Enforce metal budget constraint
  // --------------------------
  // Dust cannot exceed available gas-phase metals
  const double M_gas = Sp->P[gas_idx].getMass();
  const double M_metals = M_gas * Z_gas;

  // Maximum dust-to-metal ratio (typically 0.3-0.5 in MW)
  const double max_dust_to_metal = 0.5;
  const double M_dust_max = M_metals * max_dust_to_metal;

  if(M_dust >= M_dust_max) {
    failed_max_dz++;
    return;
  }

  if(M_dust + dm > M_dust_max) {
    // Limit growth to avoid exceeding metal budget
    dm = M_dust_max - M_dust;
    if(dm <= 0.0) { failed_max_dz++; return; }

    // Recalculate consistent a_new from limited dm
    da = (dm / M_dust) * a / 3.0;
    a_new = a + da;

    if(a_new < DUST_MIN_GRAIN_SIZE || a_new > DUST_MAX_GRAIN_SIZE) return;
  }

  // Prevent gas metallicity from going negative
  if(dm > M_metals) dm = 0.99 * M_metals;

  // Per-step cap: don't grow more than 5% per timestep (numerical stability)
  const double max_dm_per_step = 0.2 * M_dust;
  if(dm > max_dm_per_step) dm = max_dm_per_step;
  if(dm <= 0.0) return;

  // --------------------------
  // Apply growth: update dust and gas
  // --------------------------
  passed_all++;

  static int dt_printed = 0;
  if(dt_printed < 10 && All.ThisTask == 0) {
    double dt_myr = dt * All.UnitTime_in_s / (1e6 * SEC_PER_YEAR);
    DUST_PRINT("[GROWTH_DEBUG] dt = %.3e code units = %.3f Myr\n", dt, dt_myr);
    DUST_PRINT("[GROWTH_DEBUG] da = %.3f nm, dm = %.3e Msun (capped at %.3e)\n",
              da, dm, max_dm_per_step);
    dt_printed++;
  }

  // Dust gains mass and radius
  Sp->P[nearest_dust].setMass(M_dust + dm);
  Sp->DustP[nearest_dust].GrainRadius = a_new;

  // Gas loses metals (converted to dust)
  double Z_new = Z_gas - (dm / M_gas);
  if(Z_new < 1e-10) Z_new = 1e-10;  // Floor to prevent negative metallicity
  Sp->SphP[gas_idx].Metallicity = Z_new;

  #ifdef STARFORMATION
    // Keep MassMetallicity consistent with Metallicity
    Sp->SphP[gas_idx].MassMetallicity = M_gas * Z_new;
  #endif

  // Update global diagnostics
  NGrainGrowthEvents++;
  TotalMassGrown += dm;

  // --------------------------
  // Occasional debug output
  // --------------------------
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