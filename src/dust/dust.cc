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
#include "../cooling_sfr/feedback_spatial_hash.h"
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
#define MIN_DUST_PARTICLE_MASS  1e-7      // Minimum mass to create dust particle
#define DUST_MIN_GRAIN_SIZE     1.0e-6   // 10 nm in cm
#define DUST_MAX_GRAIN_SIZE     2.0e-5   // 200 nm in cm (0.2 microns)

#define DUST_PRINT(...) do{ if(All.DustDebugLevel){ \
  printf("[DUST|T=%d|a=%.6g z=%.3f] ", All.ThisTask, (double)All.Time, 1.0/All.Time-1.0); \
  printf(__VA_ARGS__); } }while(0)

extern double get_random_number(void);

// Function declarations
double estimate_molecular_fraction(double n_H, double Z, double T);
void dust_grain_growth_subgrid(simparticles *Sp, int gas_idx, double dt);
double scale_factor_to_physical_time(double delta_a);

// Access feedback's global spatial hash
extern spatial_hash_improved gas_hash;
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
static long long BruteForceSearches = 0;
static long long HashSearchesFailed = 0;  // No neighbor found
static double    TotalHashSearchTime = 0.0;

// Destruction mechanism tracking
long long NDustDestroyedByThermal   = 0;
long long NDustDestroyedByShock     = 0;

// Growth/erosion tracking
long long NGrainGrowthEvents        = 0;
long long NGrainErosionEvents       = 0;
double    TotalMassGrown            = 0.0;
double    TotalMassEroded           = 0.0;



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
  
  destroy_dust_particles(Sp);
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
int erode_dust_grain_thermal(simparticles *Sp, int dust_idx, double T_gas, double dt)
{
  double a = Sp->DustP[dust_idx].GrainRadius;
  
  if(T_gas < All.DustThermalSputteringTemp) {
    return 0;
  }
  
  double tau_sputter_yr = 1e10 * pow(1e6 / T_gas, 1.5);
  if(tau_sputter_yr < 1e5)  tau_sputter_yr = 1e5;   // At 100,000 years will allow faster erosion
  if(tau_sputter_yr > 1e10) tau_sputter_yr = 1e10;  // At 10 Gyr, should allow slower sputtering at cool temps
  
  double tau_sputter = tau_sputter_yr * SEC_PER_YEAR / All.UnitTime_in_s;
  double da_dt = -a / tau_sputter;
  double da = da_dt * dt;
  double a_new = a + da;
  
  if(a_new <= 0.0 || !isfinite(a_new))
    {
      DUST_PRINT("[BUG] Thermal erosion created invalid a_new=%.3e (a=%.3e, da=%.3e)\n",
                a_new, a, da);
      return 1; // Destroy it
    }

  if(a_new < DUST_MIN_GRAIN_SIZE) {
    double dust_mass = Sp->P[dust_idx].getMass();
    
    int nearest_gas = find_nearest_gas_particle(Sp, dust_idx);
    if(nearest_gas >= 0) {
      double gas_mass = Sp->P[nearest_gas].getMass();
      double current_Z = Sp->SphP[nearest_gas].Metallicity;
      double new_Z = current_Z + (dust_mass / gas_mass);
      Sp->SphP[nearest_gas].Metallicity = new_Z;
      
      #ifdef STARFORMATION
      Sp->SphP[nearest_gas].MassMetallicity = gas_mass * new_Z;
      #endif
      
      DUST_PRINT("[SPUTTERING] Grain eroded at T=%.2e K: a=%.3f nm, dust %.2e Msun\n",
           T_gas, DUST_MIN_GRAIN_SIZE*1e7, dust_mass);
    }
    
    Sp->DustP[dust_idx].GrainRadius = DUST_MIN_GRAIN_SIZE;
    Sp->P[dust_idx].setMass(1e-30);
    Sp->P[dust_idx].setType(3); // hack! Changing this to an unused parttype so find_nearest_dust_particle() skips it
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
  
  int nearest_gas = find_nearest_gas_particle(Sp, dust_idx);
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
  
  static int erosion_count = 0;
  erosion_count++;
  if(erosion_count % 10 == 0 && All.ThisTask == 0) {
    DUST_PRINT("[EROSION] Grain shrunk: %.2f → %.2f nm (dm=%.2e, T=%.0f K)\n",
               a*1e7, a_new*1e7, mass_lost, T_gas);
  }
  
  return 0;
}

/**
 * Erode dust grain through shock shattering/sputtering
 * Returns 1 if particle was destroyed (too small), 0 otherwise
 */
int erode_dust_grain_shock(simparticles *Sp, int dust_idx, double shock_velocity_km_s, 
                           double distance_to_sn, double shock_radius)
{
  double a = Sp->DustP[dust_idx].GrainRadius;
  
  double base_efficiency = get_shock_destruction_efficiency(shock_velocity_km_s);
  
  double distance_factor = 1.0 - 0.7 * (distance_to_sn / shock_radius);
  if(distance_factor < 0.2) distance_factor = 0.2;
  
  double grain_size_nm = a * 1e7;
  double size_factor = 1.0;
  if(grain_size_nm < 20) {
    size_factor = 1.5;
  } else if(grain_size_nm < 50) {
    size_factor = 1.2;
  } else if(grain_size_nm > 150) {
    size_factor = 0.7;
  }
  
  double erosion_fraction = base_efficiency * distance_factor * size_factor;
  if(erosion_fraction > 0.95) erosion_fraction = 0.95;
  
  double a_new = a * (1.0 - erosion_fraction * 0.5);
  
  if(a_new <= 0.0 || !isfinite(a_new))
    {
      DUST_PRINT("[BUG] Shock shattering created invalid a_new=%.3e (a=%.3e, erosion=%.3f)\n",
                a_new, a, erosion_fraction);
      return 1;
    }

  if(a_new < DUST_MIN_GRAIN_SIZE) {
    double dust_mass = Sp->P[dust_idx].getMass();
    
    int nearest_gas = find_nearest_gas_particle(Sp, dust_idx);
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
    Sp->P[dust_idx].setType(3); // hack! Changing this to an unused parttype so find_nearest_dust_particle() skips it
    memset(&Sp->DustP[dust_idx], 0, sizeof(dust_data));

    LocalDustMassChange -= dust_mass;
    LocalDustDestroyedThisStep++;
    DustNeedsSynchronization = 1;
    NDustDestroyedByShock++;
    TotalMassEroded += dust_mass;
    
    return 1;
  }
  
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
  
  int nearest_gas = find_nearest_gas_particle(Sp, dust_idx);
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
  
  return 0;
}

/**
 * Dust-gas interaction with drag and thermal erosion
 */
int dust_gas_interaction(simparticles *Sp, int dust_idx, double dt)
{
  int nearest_gas = find_nearest_gas_particle(Sp, dust_idx);
  if(nearest_gas < 0) return 0;
  
  // Apply drag
  double gas_vel[3] = {Sp->P[nearest_gas].Vel[0],
                       Sp->P[nearest_gas].Vel[1],
                       Sp->P[nearest_gas].Vel[2]};
  
  double gas_density = Sp->SphP[nearest_gas].Density * All.cf_a3inv;  // Physical density
  double gas_density_cgs = gas_density * All.UnitDensity_in_cgs;
  double n_H = gas_density_cgs / PROTONMASS;  // Number density in cm^-3

  // Drag timescale from Draine & Salpeter 1979
  // τ_drag ≈ (ρ_dust × a) / (ρ_gas × v_th) ~ a few Myr in ISM
  // drag should be stronger in the ISM and weaker in the halo
  double drag_timescale_myr = 5.0 * (10.0 / std::max(n_H, 0.1));  // Shorter in dense gas
  if(drag_timescale_myr > 50.0) drag_timescale_myr = 50.0;  // Cap at 50 Myr
  if(drag_timescale_myr < 1.0)  drag_timescale_myr = 1.0;   // Floor at 1 Myr

  double drag_timescale = drag_timescale_myr * 1e6 * SEC_PER_YEAR / All.UnitTime_in_s;
  
  double drag_factor = 1.0 - exp(-dt / drag_timescale);
  const double MAX_DRAG_FACTOR = 0.05;
  if(drag_factor > MAX_DRAG_FACTOR) drag_factor = MAX_DRAG_FACTOR;
  
  for(int k = 0; k < 3; k++) {
    Sp->P[dust_idx].Vel[k] += drag_factor * (gas_vel[k] - Sp->P[dust_idx].Vel[k]);
  }
  
  // Update dust temperature
  double utherm = Sp->get_utherm_from_entropy(nearest_gas);
  double T_gas = utherm * (All.UnitEnergy_in_cgs / All.UnitMass_in_g) 
                 / BOLTZMANN * PROTONMASS * 0.6;
  
  double T_dust = Sp->DustP[dust_idx].DustTemperature;
  double tau_thermal = 1e6 * SEC_PER_YEAR / All.UnitTime_in_s;
  double alpha = 1.0 - exp(-dt / tau_thermal);
  T_dust = T_dust * (1.0 - alpha) + T_gas * alpha;
  Sp->DustP[dust_idx].DustTemperature = T_dust;

  // Thermal erosion
  if(T_gas > All.DustThermalSputteringTemp) {
    int destroyed = erode_dust_grain_thermal(Sp, dust_idx, T_gas, dt);
    return destroyed;
  }

  return 0;
}

/**
 * Create dust particles from stellar feedback
 */
void create_dust_particles_from_feedback(simparticles *Sp, int star_idx, 
                                         double metals_produced, int feedback_type)
{
  double dust_yield_fraction;
  double velocity_scale;
  
  // Temporary debugging
  static int first_dust_creations = 0;
  if(first_dust_creations < 10 && All.ThisTask == 0) {
      double star_age = get_stellar_age_Myr(Sp->P[star_idx].StellarAge, 0.0);
      DUST_PRINT("[FIRST_DUST] Creation #%d: Star %d, age=%.3f Myr, feedback_type=%d (1=SNII, 2=AGB), CF=%.2f\n",
                first_dust_creations, star_idx, star_age, feedback_type,
                feedback_type == 1 ? 0.1 : 0.6);
      first_dust_creations++;
  }



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
  */
  int n_dust_particles = (feedback_type == 1) ? 10 : 15;   // SNe: 10, AGB: 15
  double dust_mass_per_particle = total_dust_mass / n_dust_particles;

  for(int n = 0; n < n_dust_particles; n++) {
    if(Sp->NumPart >= Sp->MaxPart) {
      if(All.ThisTask == 0) {
        DUST_PRINT("[WARNING] Cannot create dust particle - particle array full\n");
      }
      break;
    }
    
    /*
    Now, let's randomly distribute the dust particles in a sphere around the star.
    Need have some offset so they don't immediately interact with the star particle 
    (this happens and things go awry!). SN should be more extended, AGB closer in.
    */
    double theta = acos(2.0 * get_random_number() - 1.0);
    double phi   = 2.0 * M_PI * get_random_number();
    
    // Different scales for different sources
    double offset_min, offset_max;
    if(feedback_type == 1) {  // SN
        // if v ~ 200 km/s → travel ~600 pc in 3 Myr
      offset_min = All.DustOffsetMinSNII;
      offset_max = All.DustOffsetMaxSNII;
    } else {  // AGB  
        // if v ~ 20 km/s → travel ~150 pc in 10 Myr
      offset_min = All.DustOffsetMinAGB;
      offset_max = All.DustOffsetMaxAGB;
    }

    // Random radius in range (shell, not sphere surface)
    double r = offset_min + (offset_max - offset_min) * get_random_number();

    double offset_kpc[3];
    offset_kpc[0] = r * sin(theta) * cos(phi);
    offset_kpc[1] = r * sin(theta) * sin(phi);
    //offset_kpc[2] = r * sin(theta) * cos(phi);
    offset_kpc[2] = r * cos(theta);
    
    double initial_velocity[3];
    initial_velocity[0] = velocity_scale * sin(theta) * cos(phi) / All.UnitVelocity_in_cm_per_s * 1e5;
    initial_velocity[1] = velocity_scale * sin(theta) * sin(phi) / All.UnitVelocity_in_cm_per_s * 1e5;
    initial_velocity[2] = velocity_scale * cos(theta) / All.UnitVelocity_in_cm_per_s * 1e5;
    
    initial_velocity[0] += Sp->P[star_idx].Vel[0];
    initial_velocity[1] += Sp->P[star_idx].Vel[1];
    initial_velocity[2] += Sp->P[star_idx].Vel[2];
    
    spawn_dust_particle(Sp, offset_kpc, dust_mass_per_particle, initial_velocity, star_idx);
    
    int new_idx = Sp->NumPart - 1;

    if(feedback_type == 1) { // SN
      Sp->DustP[new_idx].GrainRadius = 1e-6;      // 10 nm
      Sp->DustP[new_idx].CarbonFraction = 0.1;
      Sp->DustP[new_idx].GrainType = 0;
    }
    else if(feedback_type == 2) {  // AGB
      Sp->DustP[new_idx].GrainRadius = 1e-6;      // 10 nm
      Sp->DustP[new_idx].CarbonFraction = 0.6;
      Sp->DustP[new_idx].GrainType = 1;
    }
  }

  LocalDustCreatedThisStep += n_dust_particles;
  LocalDustMassChange      += total_dust_mass;
  DustNeedsSynchronization  = 1;

  if(All.ThisTask == 0) {
    static long long total_calls = 0;
    static double total_dust_created = 0.0;
    total_calls++;
    total_dust_created += total_dust_mass;
    
    if(total_calls % 100 == 0) {
      DUST_PRINT("[DUST_STATS] Feedback events: %lld, Total dust: %.3e Msun, Avg: %.3e Msun/event\n",
                 total_calls, total_dust_created, total_dust_created/total_calls);
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

  // Synchronize MaxID across all tasks to prevent duplicate IDs
  // This was found to be an issue when restarting from snapshots
  long long local_max_id = All.MaxID;
  MPI_Allreduce(&local_max_id, &All.MaxID, 1, MPI_LONG_LONG, MPI_MAX, Communicator);
}

/**
 * Spawn a single dust particle
 */
void spawn_dust_particle(simparticles *Sp, double offset_kpc[3], double dust_mass, 
                         double initial_velocity[3], int star_idx)
{
  if(Sp->NumPart >= Sp->MaxPart) {
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
  
  Sp->DustP[new_idx].GrainRadius = 1e-6;  // 10 nm
  Sp->DustP[new_idx].CarbonFraction = 0.3;
  Sp->DustP[new_idx].GrainType = 2;
  
  int nearest_gas = find_nearest_gas_particle(Sp, new_idx);
  if(nearest_gas >= 0) {
    double utherm = Sp->get_utherm_from_entropy(nearest_gas);
    double T_gas = utherm * (All.UnitEnergy_in_cgs / All.UnitMass_in_g) 
                  / BOLTZMANN * PROTONMASS * 0.6;
    Sp->DustP[new_idx].DustTemperature = T_gas;
  } else {
    Sp->DustP[new_idx].DustTemperature = 100.0;
  }

  Sp->P[new_idx].StellarAge = All.Time;
  Sp->P[new_idx].Ti_Current = All.Ti_Current;
  
  int dust_timebin = All.HighestActiveTimeBin;
  Sp->P[new_idx].TimeBinGrav  = dust_timebin;
  Sp->P[new_idx].TimeBinHydro = 0;
  
  // DIAGNOSTIC: Verify particle is properly initialized BEFORE incrementing NumPart
  if(Sp->DustP[new_idx].GrainRadius <= 0.0 || !isfinite(Sp->DustP[new_idx].GrainRadius)) {
    DUST_PRINT("[SPAWN_BUG] Just set GrainRadius but it's %.3e for new particle at idx=%d!\n",
               Sp->DustP[new_idx].GrainRadius, new_idx);
    Sp->DustP[new_idx].GrainRadius = 1e-6;  // Force it
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
      double grain_size_nm = Sp->DustP[i].GrainRadius * 1e7;
      double mass = Sp->P[i].getMass();
      
      total_grains++;
      total_mass += mass;
      
      for(int b = 0; b < NBINS; b++) {
        if(grain_size_nm >= bin_edges[b] && grain_size_nm < bin_edges[b+1]) {
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
 * Update dust particle dynamics
 */
void update_dust_dynamics(simparticles *Sp, double dt, MPI_Comm Communicator)
{
  int ThisTask;
  MPI_Comm_rank(Communicator, &ThisTask);

  // Hash should already be built by feedback routines, but do here just in case
  if(!gas_hash.is_built) {
    if(All.ThisTask == 0) {
      DUST_PRINT("WARNING: Hash not built, building now for dust operations\n");
    }
    // Rebuild with reasonable search radius for dust operations
    double typical_search_radius = 10.0;  // kpc, matches your find_nearest_dust_particle limit
    rebuild_feedback_spatial_hash(Sp, typical_search_radius);
  }

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

  // Find a reasonable timebin for dust (similar to gas)
  int dust_timebin = All.HighestActiveTimeBin - 5;  // 2^5 = 32× longer than shortest
  if(dust_timebin < All.LowestActiveTimeBin) dust_timebin = All.LowestActiveTimeBin;
  if(dust_timebin > All.HighestActiveTimeBin) dust_timebin = All.HighestActiveTimeBin;

  for(int i = 0; i < Sp->NumPart; i++) {
    if(Sp->P[i].getType() == DUST_PARTICLE_TYPE && Sp->P[i].getMass() > 1e-20) {
      // Put dust on longer timesteps (don't need tiny steps for slow-evolving dust)
      Sp->P[i].TimeBinGrav = dust_timebin;
      dust_gas_interaction(Sp, i, dt);
    }
  }

  
/*
  // Clean up orphan dust in the halo
  // This is primarily to reduce computational cost from distant uninteracting dust
  // FUTURE: make this a param.txt option
  static long long orphans_removed = 0;
  int removed_this_step = 0;

  for(int i = 0; i < Sp->NumPart; i++) {
    if(Sp->P[i].getType() == DUST_PARTICLE_TYPE && Sp->P[i].getMass() > 1e-20) {
      
      // Check if dust is in the halo (>30 kpc from center)
      double pos[3];
      Sp->intpos_to_pos(Sp->P[i].IntPos, pos);
      
      // Find center (or use known halo center)
      double dx = pos[0] - All.BoxSize/2;  // Adjust if halo isn't centered
      double dy = pos[1] - All.BoxSize/2;
      double dz = pos[2] - All.BoxSize/2;
      double r = sqrt(dx*dx + dy*dy + dz*dz);
      
      // Remove dust beyond 150 kpc (SHOULD PROBABLY BE PARAMETERIZED)
      if(r > 150.0) {  // kpc
        Sp->P[i].setMass(1e-30);
        Sp->P[i].ID.set(0);
        Sp->P[i].setType(3);
        removed_this_step++;
        orphans_removed++;
      }
    }
  }

  if(removed_this_step > 0 && All.ThisTask == 0 && All.NumCurrentTiStep % 100 == 0) {
    DUST_PRINT("[ORPHAN_CLEANUP] Removed %d halo dust particles (total: %lld)\n",
              removed_this_step, orphans_removed);
  }
*/

  if(All.NumCurrentTiStep % 100 == 0)
  {
    print_dust_statistics(Sp);
    analyze_dust_gas_coupling_global(Sp);
    analyze_grain_size_distribution(Sp);
    
    if(All.ThisTask == 0) {
      DUST_PRINT("[GROWTH_SUMMARY] Total growth events so far: %lld\n", NGrainGrowthEvents);
    }
  }

  // Temporary test
  if(All.ThisTask == 0) {
    int printed = 0;
    for(int i=0; i<Sp->NumPart && printed<5; i++) {
      if(Sp->P[i].getType() == DUST_PARTICLE_TYPE && Sp->P[i].getMass() > 1e-20) {
        DUST_PRINT("[CHECK] i=%d ID=%lld M=%g a=%g CF=%g GT=%d\n",
          i, (long long)Sp->P[i].ID.get(), Sp->P[i].getMass(),
          Sp->DustP[i].GrainRadius, Sp->DustP[i].CarbonFraction, Sp->DustP[i].GrainType);
        printed++;
      }
    }
  }


}

/**
 * Remove destroyed dust particles and compact array
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
 * Find nearest gas particle to dust
 * 
 This is nearest among the first 1000 gas particles in memory, which may now work!?
 Try slower brute force method first.
 
int find_nearest_gas_particle(simparticles *Sp, int dust_idx)
{
  if(Sp->NumGas == 0) return -1;
  
  double min_dist2 = 1e30;
  int nearest_gas = -1;
  
  int search_limit = (Sp->NumGas < 1000) ? Sp->NumGas : 1000;
  
  for(int i = 0; i < search_limit; i++) {
    double dxyz[3];
    Sp->nearest_image_intpos_to_pos(Sp->P[i].IntPos, Sp->P[dust_idx].IntPos, dxyz);
    
    double r2 = dxyz[0]*dxyz[0] + dxyz[1]*dxyz[1] + dxyz[2]*dxyz[2];
    
    if(r2 < min_dist2) {
      min_dist2 = r2;
      nearest_gas = i;
    }
  }
  
  return nearest_gas;
}


// Here is the brute force version... DON'T use this!
int find_nearest_gas_particle(simparticles *Sp, int dust_idx)
{
  if(Sp->NumGas == 0) return -1;

  double min_dist2 = 1e60;
  int nearest_gas = -1;

  for(int i = 0; i < Sp->NumGas; i++)
  {
    double dxyz[3];
    Sp->nearest_image_intpos_to_pos(Sp->P[i].IntPos, Sp->P[dust_idx].IntPos, dxyz);
    double r2 = dxyz[0]*dxyz[0] + dxyz[1]*dxyz[1] + dxyz[2]*dxyz[2];
    if(r2 < min_dist2) { min_dist2 = r2; nearest_gas = i; }
  }

  return nearest_gas;
}
*/

/**
 * Find nearest gas particle to dust using spatial hash
 * Much faster than O(N) brute force for large simulations
 */
int find_nearest_gas_particle(simparticles *Sp, int dust_idx)
{
  if(Sp->NumGas == 0) return -1;

  // Use spatial hash if available (should always be built by feedback)
  if(gas_hash.is_built) {
    HashSearches++;  // ← ADD THIS
    
    double nearest_dist;
    double max_search_radius = 50.0;  // kpc
    
    int nearest = gas_hash.find_nearest_gas_particle(Sp, dust_idx, 
                                                      max_search_radius, 
                                                      &nearest_dist);
    
    if(nearest < 0) {
      HashSearchesFailed++;  // ← ADD THIS
      
      static int not_found_count = 0;
      if(not_found_count < 10 && All.ThisTask == 0) {
        DUST_PRINT("[HASH_SEARCH] Particle %d: no gas found within %.1f kpc\n",
                   dust_idx, max_search_radius);
        not_found_count++;
      }
    }
    
    return nearest;
  }
  
  // Fallback to brute force if hash not built (should be rare)
  BruteForceSearches++;  // ← ADD THIS
  
  if(All.ThisTask == 0) {
    static int fallback_warning = 0;
    if(fallback_warning == 0) {
      DUST_PRINT("[WARNING] Using brute force search - hash not built!\n");
      fallback_warning = 1;
    }
  }
  
  double min_dist2 = 1e60;
  int nearest_gas = -1;

  for(int i = 0; i < Sp->NumGas; i++)
  {
    double dxyz[3];
    Sp->nearest_image_intpos_to_pos(Sp->P[i].IntPos, Sp->P[dust_idx].IntPos, dxyz);
    double r2 = dxyz[0]*dxyz[0] + dxyz[1]*dxyz[1] + dxyz[2]*dxyz[2];
    if(r2 < min_dist2) { min_dist2 = r2; nearest_gas = i; }
  }

  return nearest_gas;
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
      avg_grain_size += Sp->DustP[i].GrainRadius * 1e7;  // nm
      avg_temperature += Sp->DustP[i].DustTemperature;
    }
  }
  
  if(dust_count > 0) {
    avg_grain_size /= dust_count;
    avg_temperature /= dust_count;
  }

// ANALYZE THE DUST TEMPERATURE
int hot_grains = 0;  // T > 1e7 K
int warm_grains = 0; // 1e6 < T < 1e7 K  
int cool_grains = 0; // T < 1e6 K

for(int i = 0; i < Sp->NumPart; i++) {
  if(Sp->P[i].getType() == DUST_PARTICLE_TYPE && Sp->P[i].getMass() > 1e-20) {
    double T = Sp->DustP[i].DustTemperature;
    if(T > 1e7) hot_grains++;
    else if(T > 1e6) warm_grains++;
    else cool_grains++;
  }
}
  
  DUST_PRINT("=== DUST STATISTICS (rank 0) ===\n");
  DUST_PRINT("  Particles: %d  Mass: %.3e Msun\n", dust_count, total_dust_mass);
  DUST_PRINT("  Avg grain size: %.2f nm\n", avg_grain_size);
  DUST_PRINT("  Avg temperature: %.1f K\n", avg_temperature);
  DUST_PRINT("  Temperature bins: <1M K: %d, 1-10M K: %d, >10M K: %d\n",
             cool_grains, warm_grains, hot_grains);
  DUST_PRINT("--- Hash Search Statistics ---\n");
  DUST_PRINT("  Hash searches:       %lld\n", HashSearches);
  DUST_PRINT("  Brute force searches: %lld\n", BruteForceSearches);
  if(HashSearches > 0) {
    DUST_PRINT("  Hash success rate:    %.1f%%\n", 
               100.0 * (HashSearches - HashSearchesFailed) / HashSearches);
    DUST_PRINT("  Hash speedup:         %.1fx (vs brute force)\n",
               (double)(HashSearches + BruteForceSearches) / 
               std::max(1LL, BruteForceSearches));
  }
  if(HashSearchesFailed > 0) {
    DUST_PRINT("  [WARNING] Failed searches: %lld (%.1f%%)\n",
               HashSearchesFailed, 100.0 * HashSearchesFailed / HashSearches);
  }
  DUST_PRINT("  Growth events: %lld (%.2e Msun grown)\n", NGrainGrowthEvents, TotalMassGrown);
  DUST_PRINT("  Partial erosion events: %lld\n", NGrainErosionEvents);
  DUST_PRINT("  Destroyed by thermal: %lld\n", NDustDestroyedByThermal);
  DUST_PRINT("  Destroyed by shocks: %lld\n", NDustDestroyedByShock);
  DUST_PRINT("  Total mass eroded: %.2e Msun\n", TotalMassEroded);
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
      int nearest_gas = find_nearest_gas_particle(Sp, i);
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
    
  int nearest_gas = find_nearest_gas_particle(Sp, sn_star_idx);
  if(nearest_gas < 0) {
    double typical_ism_density = 1.0;
    double gas_density_cgs = typical_ism_density * PROTONMASS;
    return calculate_sn_shock_radius(sn_energy_erg, gas_density_cgs, 0.1);
  }
    
  double gas_density_code = Sp->SphP[nearest_gas].Density * All.cf_a3inv;
  double gas_density_cgs  = gas_density_code * All.UnitDensity_in_cgs;
    
  double characteristic_time_myr = 1.0;
  double radius = calculate_sn_shock_radius(sn_energy_erg, gas_density_cgs, characteristic_time_myr);
    
  if(radius < 0.5) radius = 0.5;
    
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
 */
// In get_shock_destruction_efficiency(), around line 815:
double get_shock_destruction_efficiency(double shock_velocity_km_s)
{
  if(shock_velocity_km_s < 100.0) {
    return 0.0;
  }
  else if(shock_velocity_km_s < 200.0) {
    return 0.02 * (shock_velocity_km_s - 100.0) / 100.0;  // 0-2% (was 0-10%)
  }
  else if(shock_velocity_km_s < 400.0) {
    return 0.02 + 0.05 * (shock_velocity_km_s - 200.0) / 200.0;  // 2-7% (was 10-25%)
  }
  else {
    return 0.07;  // Cap at 7% (was 25%)
  }
}

/**
 * Main dust destruction function from SN shocks
 */
void destroy_dust_from_sn_shocks(simparticles *Sp, int sn_star_idx, 
                                 double sn_energy, double metals_produced)
{
  double shock_radius_kpc = calculate_current_sn_shock_radius(Sp, sn_star_idx);
    
  if(shock_radius_kpc < 0.3) shock_radius_kpc = 0.3;  // 300 pc minimum
  if(shock_radius_kpc > 1.5) shock_radius_kpc = 1.5;  // 1.5 kpc maximum

  int    nearest_gas     = find_nearest_gas_particle(Sp, sn_star_idx);
  double gas_density_cgs = 1.0 * PROTONMASS;
    
  if(nearest_gas >= 0) {
    double gas_density_code = Sp->SphP[nearest_gas].Density * All.cf_a3inv;
    gas_density_cgs = gas_density_code * All.UnitDensity_in_cgs;
  }
    
  double shock_velocity = calculate_shock_velocity(1e51, gas_density_cgs, 1.0);
    
  if(shock_velocity < 100.0) {
    return;  // Ignore weak shocks entirely
  }

  if(All.ThisTask == 0) {
    DUST_PRINT("[DUST_SN] SN from star %d: shock_radius=%.2f kpc, shock_velocity=%.0f km/s\n",
               Sp->P[sn_star_idx].ID.get(), shock_radius_kpc, shock_velocity);
  }
    
  int dust_in_shock  = 0;
  int dust_destroyed = 0;
  int dust_eroded    = 0;
  
  for(int i = 0; i < Sp->NumPart; i++) {
    if(Sp->P[i].getType() == DUST_PARTICLE_TYPE && Sp->P[i].getMass() > 1e-20) {
      double dxyz[3];
      Sp->nearest_image_intpos_to_pos(Sp->P[i].IntPos, Sp->P[sn_star_idx].IntPos, dxyz);
      double distance = sqrt(dxyz[0]*dxyz[0] + dxyz[1]*dxyz[1] + dxyz[2]*dxyz[2]);
      
      if(distance < shock_radius_kpc) {
        dust_in_shock++;
        
        int destroyed = erode_dust_grain_shock(Sp, i, shock_velocity, 
                                               distance, shock_radius_kpc);
        
        if(destroyed) {
          dust_destroyed++;
        } else {
          dust_eroded++;
        }
      }
    }
  }
  
  if(All.ThisTask == 0 && dust_in_shock > 0) {
    DUST_PRINT("[DUST_SN] SN affected %d dust: %d destroyed, %d eroded\n",
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


/**
 * Sophisticated subgrid grain growth model (HK11-based)
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
void dust_grain_growth_subgrid(simparticles *Sp, int gas_idx, double dt)
{
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
  if(total_calls % 1000 == 0 && All.ThisTask == 0) {
    DUST_PRINT("=== HK11 GROWTH DIAGNOSTICS (after %d attempts) ===\n", total_calls);
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

  // Get gas number density to infer molecular fraction
  double gas_density_code = Sp->SphP[gas_idx].Density * All.cf_a3inv;  // Physical density
  double gas_density_cgs = gas_density_code * All.UnitDensity_in_cgs;
  double n_H = gas_density_cgs / PROTONMASS;  // Hydrogen number density [cm^-3]

  #ifdef STARFORMATION
    if(Sp->SphP[gas_idx].Sfr > 0.0) {
      // Star-forming regions: very molecular (n >> 100 cm^-3, strong shielding)
      // Typical f_mol ~ 0.7-0.9 in star-forming cores (McKee & Ostriker 2007)
      f_mol = 0.8;
      fmol_sf++;
    } else if(n_H > 100.0) {
      // Dense molecular clouds: n > 100 cm^-3
      // Self-shielding allows H2 formation even without active SF
      // Typical f_mol ~ 0.3-0.7 (Glover & Mac Low 2011)
      f_mol = 0.5;
      fmol_dense++;
    } else if(n_H > 10.0) {
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
  const int nearest_dust = find_nearest_dust_particle(Sp, gas_idx);
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
  const double a = Sp->DustP[nearest_dust].GrainRadius;   // Grain radius [cm]
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
  // These are effective values for the subgrid accretion environment
  const double n_eff_cm3 = 100.0;   // Effective density in molecular clouds
  const double T_eff_K   = 20.0;    // Effective temperature in cold GMCs

  const double Zsun_massfrac = 0.02;  // Solar metallicity (mass fraction)
  const double S_stick = 0.3;         // Sticking coefficient (HK11 default)

  // HK11 accretion timescale: tau_acc(a, Z, n, T, species)
  double tau_acc_yr = tau_acc_yr_HK11(n_eff_cm3, T_eff_K,
                                      Z_gas, Zsun_massfrac,
                                      a, S_stick, species);

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
  double dm = M_dust * (3.0 * da / a);
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
              da*1e7, dm, max_dm_per_step);
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
  if(growth_count % 100 == 0 && All.ThisTask == 0) {
    DUST_PRINT("[HK11_GROWTH] Event #%d: species=%s CF=%.2f f_mol=%.3f n_H=%.1f cm^-3\n",
               growth_count, (species==1 ? "carb" : "sil"), CF, f_mol, n_H);
    DUST_PRINT("  tau_acc=%.2e yr (%.2f Myr) | n_eff=%.0f cm^-3 T_eff=%.0f K | Z=%.4f\n",
               tau_acc_yr, tau_acc_yr/1e6, n_eff_cm3, T_eff_K, Z_gas);
    DUST_PRINT("  Grain: a=%.2f→%.2f nm | dm=%.3e (M_dust: %.3e→%.3e)\n",
               a*1e7, a_new*1e7, dm, M_dust, M_dust+dm);
    DUST_PRINT("  Gas: M_gas=%.3e | Z=%.4f→%.4f | M_metals=%.3e | dZ=%.3e\n",
               M_gas, Z_gas, Z_new, M_metals, (Z_gas - Z_new));
  }
}


#endif /* DUST */