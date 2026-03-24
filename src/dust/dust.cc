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
#define DUST_MASS_TO_DESTROY   1e-30
#define DUST_MIN_GRAIN_SIZE  1.0       // nm
#define DUST_MAX_GRAIN_SIZE  200.0     // nm
#define DUST_MIN_TIMEBIN 15            // Minimum gravity timebin for dust particles; without this, dust can get to very low timebins and slow down the sim
#define DUST_SFR_EPS 1e-14             // Minimum SFR to consider a gas cell star-forming
#define DUST_SUBLIMATION_TEMP 2000.0   // Temperature at which dust sublimates (K)
// Single grain mass at DUST_MIN_GRAIN_SIZE and silicate density
// m = (4/3)π(0.5e-7 cm)³ × 2.4 g/cm³ / 1.989e33 g/Msun
#define DUST_SINGLE_GRAIN_MASS_MSUN  6.32e-55   // M☉; purely informational

#define DUST_PRINT(...) do{ if(All.DustDebugLevel){ \
  printf("[DUST|T=%d|a=%.6g z=%.3f] ", All.ThisTask, (double)All.Time, 1.0/All.Time-1.0); \
  printf(__VA_ARGS__); } }while(0)

extern double get_random_number(void);

// Function declarations
void dust_grain_growth_subgrid(simparticles *Sp, int dust_idx, int gas_idx, double dt);
static int destroy_dust_particle_to_gas(simparticles *Sp, int dust_idx,
                                         int nearest_gas, long long *counter,
                                         double *mass_counter);
double dust_clumping_factor(double n_H, int is_star_forming);

// Access feedback's global spatial hash
extern spatial_hash_zoom gas_hash;
extern spatial_hash_zoom star_hash;
extern spatial_hash_zoom dust_hash;
extern void rebuild_feedback_spatial_hash(simparticles *Sp, double max_feedback_radius, double dust_search_radius, MPI_Comm comm);

// Module-level statistics
long long NDustCreated              = 0;
long long NDustCreatedBySNII        = 0;
long long NDustCreatedByAGB         = 0;
long long NDustDestroyed            = 0;
double    TotalDustMass             = 0.0;
long long LocalDustCreatedThisStep  = 0;
long long LocalDustDestroyedThisStep= 0;
double    LocalDustMassChange       = 0.0;
int       DustNeedsSynchronization  = 0;
long long GlobalDustCount           = 0;
long long NShatteringEvents         = 0;
double    TotalSizeReductionShattering = 0.0;  // sum of (a_old - a_new) across events

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
double TotalMassDestroyedByThermal  = 0.0;  // full thermal destructions
double TotalMassDestroyedByShock    = 0.0;  // full shock destructions
double TotalMassErodedByThermal     = 0.0;  // partial sputtering events
double TotalMassErodedByShock       = 0.0;  // partial shock erosion events

// Coagulation
long long NCoagulationEvents        = 0;

/**
 * Clean up invalid dust particles, with detailed corruption diagnostics.
 *
 * Prints per-task reason tallies so we can identify whether corruption
 * is coming from domain exchange (GrainRadius zeroed), creation bugs
 * (bad position/velocity), or something else.
 */
void cleanup_invalid_dust_particles(simparticles *Sp)
{
  int n_bad_radius   = 0;
  int n_bad_mass     = 0;
  int n_bad_pos      = 0;
  int n_bad_vel      = 0;
  int n_bad_carbon   = 0;
  int n_bad_temp     = 0;
  int n_zero_radius_nonzero_mass = 0;
  int n_fully_zeroed             = 0;
  int cleaned        = 0;
  int detail_count = 0;  // local, not static: get up to 50 per call

  for(int i = 0; i < Sp->NumPart; i++) {
    if(Sp->P[i].getType() != DUST_PARTICLE_TYPE) continue;

    double a    = Sp->DustP[i].GrainRadius;
    double mass = Sp->P[i].getMass();
    double cf   = Sp->DustP[i].CarbonFraction;
    double temp = Sp->DustP[i].DustTemperature;
    double pos[3];
    Sp->intpos_to_pos(Sp->P[i].IntPos, pos);

    bool is_corrupt = false;

    // Run through every non-sensical check with can think of, such as a negative particle radius or mass, and flag if necessary...
    if(a <= 0.0 || !isfinite(a))                                          { n_bad_radius++; is_corrupt = true; }
    if(!isfinite(mass) || mass <= 0.0)                                    { n_bad_mass++; is_corrupt = true; }
    if(!isfinite(pos[0]) || !isfinite(pos[1]) || !isfinite(pos[2]))       { n_bad_pos++;    is_corrupt = true; }
    if(!isfinite(Sp->P[i].Vel[0]) || !isfinite(Sp->P[i].Vel[1]) ||
       !isfinite(Sp->P[i].Vel[2]))                                        { n_bad_vel++;    is_corrupt = true; }
    if(!isfinite(cf) || cf < 0.0 || cf > 1.0)                             { n_bad_carbon++; is_corrupt = true; }
    if(!isfinite(temp) || temp < 0.0)                                     { n_bad_temp++;   is_corrupt = true; }

    if(is_corrupt) {
      bool is_domain_exchange_victim = (a == 0.0 && cf == 0.0 && temp == 0.0);

      if(is_domain_exchange_victim && mass > DUST_MASS_TO_DESTROY)
        n_zero_radius_nonzero_mass++;  // DustP lost but P[] intact → exchange timing issue
      else if(is_domain_exchange_victim && mass < DUST_MASS_TO_DESTROY)
        n_fully_zeroed++;              // both P[] and DustP zeroed → catastrophic
      
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

      Sp->P[i].setMass(DUST_MASS_TO_DESTROY);
      Sp->P[i].ID.set(0);
      Sp->P[i].setType(3);
      memset(&Sp->DustP[i], 0, sizeof(dust_data));
      Sp->DustP[i].GrainRadius = DUST_MIN_GRAIN_SIZE;

      NDustDestroyedByCleanup++;
      cleaned++;
    }
  }

  if(cleaned > 0) {
    printf("[CLEANUP|T=%d|Step=%d|a=%.4f z=%.3f] Removed %d corrupted dust particles:\n"
           "  bad_radius=%d bad_mass=%d bad_pos=%d bad_vel=%d bad_carbon=%d bad_temp=%d\n"
           "  zero_radius_nonzero_mass=%d (DustP lost, P[] intact → exchange timing)\n"
           "  fully_zeroed=%d            (both P[] and DustP zeroed → catastrophic)\n"
           "  RunningTotal=%lld\n",
           All.ThisTask, All.NumCurrentTiStep, All.Time, 1.0/All.Time - 1.0,
           cleaned,
           n_bad_radius, n_bad_mass, n_bad_pos, n_bad_vel, n_bad_carbon, n_bad_temp,
           n_zero_radius_nonzero_mass,
           n_fully_zeroed,
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

void dust_grain_shattering(simparticles *Sp, int dust_idx, int gas_idx, double dt)
{
  // Diagnostic counters (static: accumulated over entire run on this task)
  static int shat_calls       = 0;
  static int shat_failed_vel  = 0;
  static int shat_failed_dens = 0;
  static int shat_failed_size = 0;
  static int shat_passed      = 0;
  shat_calls++;

  if(shat_calls % 50000 == 0 && All.ThisTask == 0)
    DUST_PRINT("[SHAT_DIAG] calls=%d  failed: vel=%d(%.1f%%) dens=%d(%.1f%%) "
               "size=%d(%.1f%%)  passed=%d(%.1f%%)\n",
               shat_calls,
               shat_failed_vel,  100.0 * shat_failed_vel  / shat_calls,
               shat_failed_dens, 100.0 * shat_failed_dens / shat_calls,
               shat_failed_size, 100.0 * shat_failed_size / shat_calls,
               shat_passed,      100.0 * shat_passed      / shat_calls);

  // -----------------------------------------------------------------------
  // Gas properties
  // -----------------------------------------------------------------------
  double gas_density_code = Sp->SphP[gas_idx].Density * All.cf_a3inv;
  double gas_density_cgs  = gas_density_code * All.UnitDensity_in_cgs;
  double n_H              = (gas_density_cgs * HYDROGEN_MASSFRAC) / PROTONMASS;
  double T_gas            = get_temperature_from_entropy(Sp, gas_idx);

  // -----------------------------------------------------------------------
  // Gate 1: turbulent velocity must exceed shattering threshold.
  // We use sound speed as a conservative proxy for turbulent velocity; in
  // warm/hot diffuse ISM where shattering operates, cs ~ v_turb is reasonable.
  // Threshold from Jones et al. (1994): ~1 km/s silicate, ~2 km/s carbonaceous.
  // -----------------------------------------------------------------------
  double cs_cgs        = sqrt(BOLTZMANN * T_gas / (0.6 * PROTONMASS));
  double v_turb_kms    = cs_cgs / 1e5;
  double CF            = Sp->DustP[dust_idx].CarbonFraction;
  double v_shatter_kms = 1.0 + CF;  // 1 km/s for pure silicate, 2 km/s for pure carbon

  if(v_turb_kms < v_shatter_kms) { shat_failed_vel++;  return; }

  // -----------------------------------------------------------------------
  // Gate 2: shattering is active only in warm/diffuse gas (n_eff below
  // threshold). Dense gas is dominated by coagulation instead. We use the
  // clumping-factor-weighted effective density so the criterion is
  // resolution-independent.
  // -----------------------------------------------------------------------
  double DustClumpingFactor = dust_clumping_factor(n_H, Sp->SphP[gas_idx].Sfr > DUST_SFR_EPS);
  double n_eff              = n_H * DustClumpingFactor;

  if(n_eff > All.DustCollisionDensityThresh) { shat_failed_dens++; return; }

  // -----------------------------------------------------------------------
  // Gate 3: grain must be physical
  // -----------------------------------------------------------------------
  double a      = Sp->DustP[dust_idx].GrainRadius;
  double M_dust = Sp->P[dust_idx].getMass();

  if(a <= DUST_MIN_GRAIN_SIZE || M_dust <= 0.0 || !isfinite(a)) { shat_failed_size++; return; }

  // -----------------------------------------------------------------------
  // Shattering timescale (Hirashita & Kuo 2011, eq. 9).
  // tau_shat scales with grain size and inversely with dust-to-gas ratio and
  // relative velocity. The velocity_factor amplifies the rate for grains well
  // above the shattering threshold.
  // -----------------------------------------------------------------------
  double dust_to_gas     = M_dust / Sp->P[gas_idx].getMass();
  double v_excess        = v_turb_kms - v_shatter_kms;
  double velocity_factor = 1.0 + v_excess / v_shatter_kms;
  double a_micron        = a / 1000.0;  // nm → µm

  double tau_shat_yr = 1e8 * (1.0 / dust_to_gas) * (a_micron / 0.1) / velocity_factor;
  tau_shat_yr       *= All.DustShatteringCalibration;

  // Hard floor/ceiling to avoid unphysical extremes
  tau_shat_yr = std::max(tau_shat_yr, 1e6);
  tau_shat_yr = std::min(tau_shat_yr, 1e10);

  double tau_shat = tau_shat_yr * SEC_PER_YEAR / All.UnitTime_in_s;

  // -----------------------------------------------------------------------
  // Stochastic shattering event.
  // Shattering is catastrophic, not continuous erosion: a grain either suffers
  // a high-velocity collision and fragments this timestep, or it does not.
  // The Poisson probability of at least one event in dt is 1 - exp(-dt/tau).
  // Fragment size: MRN a^-3.5 distribution gives a mean fragment radius of
  // ~a/3 (Jones et al. 1996), so we reduce radius by factor 3, which
  // corresponds to a factor ~30 reduction in grain mass. Total superparticle
  // mass is conserved — the mass is still dust, just distributed among many
  // smaller grains represented by the new radius.
  // -----------------------------------------------------------------------
  double P_shatter = 1.0 - exp(-dt / tau_shat);

  if(get_random_number() >= P_shatter) return;  // no event this timestep

  double a_new = a * 0.33;
  if(a_new < DUST_MIN_GRAIN_SIZE) a_new = DUST_MIN_GRAIN_SIZE;

  Sp->DustP[dust_idx].GrainRadius = a_new;

  shat_passed++;
  NShatteringEvents++;
  TotalSizeReductionShattering += (a - a_new);

  if(All.ThisTask == 0 && (NShatteringEvents <= 100 || NShatteringEvents % 10000 == 0))
    DUST_PRINT("[SHATTERING] Event #%lld: a=%.1f→%.1f nm  "
               "n_H=%.2f n_eff=%.2f cm^-3  T=%.0f K  "
               "v_turb=%.2f km/s  tau=%.1f Myr  P=%.3e\n",
               NShatteringEvents, a, a_new,
               n_H, n_eff, T_gas,
               v_turb_kms, tau_shat_yr / 1e6, P_shatter);
}

/**
 * Erode dust grain through thermal sputtering
 * Returns 1 if particle was destroyed (too small), 0 otherwise
 */
int erode_dust_grain_thermal(simparticles *Sp, int dust_idx, int nearest_gas_input, double T_gas, double dt)
{
  if(!All.DustEnableSputtering) return 0;
    
    double a = Sp->DustP[dust_idx].GrainRadius;
    
    // INSTANT DESTRUCTION via sublimation — grain temperature exceeds ~1500K
    // This is genuinely T_dust driven, not T_gas
    // Silicate grains (CF~0.1) sublimate at ~1550 K and pure carbon grains at ~2000 K
    double T_sublimate = 1500.0 + 500.0 * Sp->DustP[dust_idx].CarbonFraction;
    if(Sp->DustP[dust_idx].DustTemperature > T_sublimate) {
        DUST_PRINT("[SUBLIMATION] Dust destroyed at T_dust=%.0f K\n", 
                  Sp->DustP[dust_idx].DustTemperature);
        return destroy_dust_particle_to_gas(Sp, dust_idx, nearest_gas_input,
                            &NDustDestroyedByThermal, &TotalMassDestroyedByThermal);
    }

    // INSTANT DESTRUCTION in extremely hot gas (> 1e6 K)
    // Physics: Dust sublimates in < 1 kyr at these temperatures, treat as instant
    //if(T_gas > 1e6) {
    //    DUST_PRINT("[RAPID_SPUTTERING] Dust destroyed at T_gas=%.2e K\n", T_gas);
    //    return destroy_dust_particle_to_gas(Sp, dust_idx, nearest_gas_input,
    //                        &NDustDestroyedByThermal, &TotalMassDestroyedByThermal);
    //}
  
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

  // Convert grain radius (stored in nm) to units of 0.1 µm for McKinnon 2017 formula
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
  
  // Composition correction: carbonaceous grains have lower binding energy
  // (~4 eV) than silicates (~6 eV), so they sputter faster.
  // CF=0 (pure silicate): reference rate
  // CF=1 (pure carbon):   ~1.5x faster
  double CF = Sp->DustP[dust_idx].CarbonFraction;
  double U_sil = 6.0;
  double U_car = 4.0;

  double U_eff = (1.0 - CF) * U_sil + CF * U_car;
  double composition_factor = U_sil / U_eff;   // ~1 for silicate, ~1.5 for carbon
  tau_sputter_yr /= composition_factor;

  // ===============================================================================

  // Apply reasonable bounds
  if(tau_sputter_yr < 1e6)  tau_sputter_yr = 1e6;   // 1 Myr floor (very hot gas)
  if(tau_sputter_yr > 1e9) tau_sputter_yr = 1e9;    // 1 Gyr ceiling (cool gas)
  
  double tau_sputter = tau_sputter_yr * SEC_PER_YEAR / All.UnitTime_in_s;
  double da_dt = -a / tau_sputter;   // Every grain loses the same absolute thickness per unit time regardless of size
  double da    = da_dt * dt;

  // If timestep exceeds sputtering timescale, grain is fully destroyed this step
  if(da <= -a)
      return destroy_dust_particle_to_gas(Sp, dust_idx, nearest_gas_input,
                             &NDustDestroyedByThermal, &TotalMassDestroyedByThermal);

  double a_new = a + da;
  
  if(a_new <= 0.0 || !isfinite(a_new)) {
    DUST_PRINT("[BUG] Thermal erosion created invalid a_new=%.3e\n", a_new);
    return destroy_dust_particle_to_gas(Sp, dust_idx, nearest_gas_input,
                             &NDustDestroyedByThermal, &TotalMassDestroyedByThermal);
  }

  // After applying erosion, see if the new grain radius is too small — if so, destroy the particle and return its mass to gas
  if(a_new < DUST_MIN_GRAIN_SIZE)
    return destroy_dust_particle_to_gas(Sp, dust_idx, nearest_gas_input,
                             &NDustDestroyedByThermal, &TotalMassDestroyedByThermal);
  
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
  TotalMassErodedByThermal += mass_lost;
  
  static int erosion_count = 0;
  erosion_count++;
  if(erosion_count % 10000 == 0 && All.ThisTask == 0) {  // Print every 10,000th event
      DUST_PRINT("[EROSION] Grain shrunk: %.2f → %.2f nm (dm=%.2e, T=%.0f K)\n",
                a, a_new, mass_lost, T_gas);
  }

  return 0;
}

/**
 * Destroy a dust particle and return its mass to the nearest gas cell conserving total metal mass
 *
 * This is the single destruction path for all physics-driven dust
 * removal (thermal sputtering, shock destruction) excluding astration
 */
static int destroy_dust_particle_to_gas(simparticles *Sp, int dust_idx,
                                         int nearest_gas, long long *counter,
                                         double *mass_counter)
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
  
  Sp->P[dust_idx].setMass(DUST_MASS_TO_DESTROY);
  Sp->P[dust_idx].setType(3);
  Sp->P[dust_idx].ID.set(0);
  memset(&Sp->DustP[dust_idx], 0, sizeof(dust_data));
  Sp->DustP[dust_idx].GrainRadius = DUST_MIN_GRAIN_SIZE;
  
  LocalDustMassChange -= dust_mass;
  LocalDustDestroyedThisStep++;
  DustNeedsSynchronization = 1;
  if(counter) (*counter)++;
  if(mass_counter) (*mass_counter) += dust_mass; 
  
  return 1;
}

double calculate_sedov_velocity_from_radius(double radius_kpc, double rho_cgs)
{
    const double xi = 1.033;
    const double E  = 1e51;  // erg
    
    double R_cm = radius_kpc * 1000.0 * PARSEC;
    
    // v = (2/5) * xi^(5/2) * sqrt(E/rho) * R^(-3/2)
    double xi_factor = pow(xi, 2.5);  // ≈ 1.085
    double v_cm_s = (2.0/5.0) * xi_factor * sqrt(E / rho_cgs) 
                    * pow(R_cm, -1.5);
    
    return v_cm_s / 1e5;  // km/s
}

/**
 * Grain destruction via Sedov-Taylor shock interaction.
 *
 * Step 1: Stochastic shattering for v_local > 50 km/s (Jones et al. 1994 threshold).
 *         Probability scales with velocity ramp and grain-size factor (50/a).
 * Step 2: Partial erosion for survivors using Bocchio et al. 2014 Table 6
 *         efficiency curves, blended by CarbonFraction.
 * Step 3: Mass returned to nearest gas cell as metals.
 *
 * NOTE on subgrid mode: when the dust hash cell size exceeds the physical shock
 * radius, destroy_dust_from_sn_shocks() passes an inflated effective_search_radius
 * so that find_neighbors() can actually locate dust. In that case distance_to_sn
 * may be >> shock_radius and normal attenuation would kill all local_velocity.
 * We detect this by checking distance_to_sn > shock_radius and skip attenuation,
 * treating all dust found as representative of the local shock environment.
 */
int erode_dust_grain_shock(simparticles *Sp, int dust_idx, double shock_velocity_km_s,
                           double distance_to_sn, double shock_radius,
                           int nearest_gas_hint)
{
  double a = Sp->DustP[dust_idx].GrainRadius;

  // ========================================================================
  // Attenuate local shock velocity by distance (Sedov-Taylor: v ∝ r^{-3/2})
  // Grain at shock edge sees ~30% of peak velocity; grain at center sees full.
  //
  // SUBGRID MODE: if the grain is beyond the physical shock radius (because
  // we widened the search to overcome hash resolution limits), skip attenuation
  // entirely — the grain is a subgrid representative of the shocked volume.
  // ========================================================================
  double local_velocity;
  if(distance_to_sn <= shock_radius) {
    // Normal case: grain is physically within the blast wave
    double r_frac = (shock_radius > 0) ? (distance_to_sn / shock_radius) : 0.0;
    if(r_frac < 0.0) r_frac = 0.0;
    if(r_frac > 1.0) r_frac = 1.0;

    double velocity_attenuation = pow(1.0 - 0.7 * r_frac, 1.5);
    if(velocity_attenuation < 0.3) velocity_attenuation = 0.3;

    local_velocity = shock_velocity_km_s * velocity_attenuation;
  } else {
      // This branch should be unreachable:
      // If this fires something has gone wrong upstream.
      DUST_PRINT("[BUG] erode_dust_grain_shock: distance_to_sn=%.3f > shock_radius=%.3f\n",
                distance_to_sn, shock_radius);
      return 0;
  }

  // ========================================================================
  // Single grain-size factor used consistently for both shattering and erosion
  // ========================================================================
  double size_factor;
  if     (a < 20.0)  size_factor = 1.5;
  else if(a < 50.0)  size_factor = 1.2;
  else if(a > 100.0) size_factor = 0.7;
  else               size_factor = 1.0;

  // ========================================================================
  // Find nearest gas once — reused for all three destruction paths
  // ========================================================================
  int nearest_gas = (nearest_gas_hint >= 0) ? nearest_gas_hint
                                             : find_nearest_gas_particle(Sp, dust_idx, 5.0, NULL);

  // ========================================================================
  // STEP 1: Outright shattering (stochastic, velocity-gated)
  // Threshold: Jones et al. 1994
  // ========================================================================
  if(local_velocity > 50.0) {
    double velocity_factor = (local_velocity - 50.0) / 350.0;
    if(velocity_factor > 1.0) velocity_factor = 1.0;

    double destruction_size_factor = 50.0 / a;
    if(destruction_size_factor > 3.0) destruction_size_factor = 3.0;
    if(destruction_size_factor < 0.3) destruction_size_factor = 0.3;

    double destruction_prob = velocity_factor * destruction_size_factor;
    if(destruction_prob > 0.9) destruction_prob = 0.9;

    if(get_random_number() < destruction_prob) {
      return destroy_dust_particle_to_gas(Sp, dust_idx, nearest_gas,
                                    &NDustDestroyedByShock, &TotalMassDestroyedByShock);
    }
  }

  // ========================================================================
  // STEP 2: Erosion using local (attenuated) velocity
  // Get destruction efficiency from Bocchio et al. 2014 Table 6, blended by CarbonFraction
  // ========================================================================
  double base_efficiency = get_shock_destruction_efficiency(local_velocity,
                                                            Sp->DustP[dust_idx].CarbonFraction);
  double erosion_fraction = base_efficiency * size_factor;
  if(erosion_fraction > 0.95) erosion_fraction = 0.95;

  double a_new = a * (1.0 - erosion_fraction * 0.8);
  if(a_new <= 0.0 || !isfinite(a_new)) a_new = 0.0;

  if(a_new < DUST_MIN_GRAIN_SIZE) {
    return destroy_dust_particle_to_gas(Sp, dust_idx, nearest_gas,
                                    &NDustDestroyedByShock, &TotalMassDestroyedByShock);
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
  TotalMassErodedByShock += mass_lost;

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
  
 
  // ==================================================================
  // STEP 5: Thermal erosion check
  // ==================================================================
  
    if(T_gas > All.DustThermalSputteringTemp) {
      int destroyed = erode_dust_grain_thermal(Sp, dust_idx, nearest_gas, T_gas, dt);
      return destroyed;
    }
  
  // ==================================================================
  // STEP 6: Diagnostic output (sample 1% of particles)
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

  // "metals_produced" is the total mass of metals produced by this feedback event (e.g. SN ejecta or AGB winds)
  // this varies, and therefore the initial mass of the newly formed dust will vary in mass.
  double total_dust_mass = metals_produced * dust_yield_fraction;
  
  if(total_dust_mass < MIN_DUST_PARTICLE_MASS) {
    return;
  }
  
  int n_dust_particles = (feedback_type == 1) ? All.DustParticlesPerSNII : All.DustParticlesPerAGB;
  double dust_mass_per_particle = total_dust_mass / n_dust_particles;

  if(dust_mass_per_particle < 1e-15) {
    if(All.ThisTask == 0)
        DUST_PRINT("[CREATION_SKIP] Per-particle mass %.3e below floor, skipping\n",
                   dust_mass_per_particle);
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

  // Increment counters:
  if(feedback_type == 1) NDustCreatedBySNII += n_dust_particles;
  else                   NDustCreatedByAGB  += n_dust_particles;

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
  MPI_Allreduce(&local_max_id, &All.MaxID, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, Communicator);
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
  
  // Just placeholders, these get set properly in create_dust_particles_from_feedback() after the loop
  Sp->DustP[new_idx].GrainRadius = 10.0;
  Sp->DustP[new_idx].CarbonFraction = 0.3;
  Sp->DustP[new_idx].GrainType = 2;
  
  // Set initial dust temperature to CMB floor
  double T_CMB = 2.7 / All.Time;
  Sp->DustP[new_idx].DustTemperature = T_CMB;

  Sp->P[new_idx].StellarAge = All.Time;
  Sp->P[new_idx].Ti_Current = All.Ti_Current;
  
  //Sp->P[new_idx].TimeBinGrav = All.HighestActiveTimeBin;
  Sp->P[new_idx].TimeBinHydro = 0;

    // The dust can really slow down the sim if it ends up in very small timebins, so we set a floor here
        int new_bin = All.HighestActiveTimeBin;
        if(new_bin < DUST_MIN_TIMEBIN) new_bin = DUST_MIN_TIMEBIN;

        Sp->P[new_idx].TimeBinGrav = new_bin;
        Sp->TimeBinsGravity.timebin_add_particle(new_idx, star_idx, new_bin,
            Sp->TimeBinSynchronized[new_bin]);

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
    if(Sp->P[i].getType() == DUST_PARTICLE_TYPE && Sp->P[i].getMass() > DUST_MASS_TO_DESTROY) {
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

void consume_dust_by_astration(simparticles *Sp, int gas_idx, double stellar_mass_formed, int star_idx, double hsml)
{
  if(!All.DustEnableAstration) return;
  
  double gas_mass = Sp->P[gas_idx].getMass();  // ← missing

  double rho_code = Sp->SphP[gas_idx].Density * All.cf_a3inv;
  double cell_radius = cbrt(3.0 * gas_mass / (4.0 * M_PI * rho_code));

  double search_radius = cell_radius;

  // Nver search beyond the gas smoothing length
  // This is the physically motivated upper limit — dust beyond the SPH kernel
  // has no business being astrated by this star formation event
  double max_radius = std::max(cell_radius, (double)Sp->SphP[gas_idx].Hsml);
  if(search_radius > max_radius) search_radius = max_radius;
  
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
    if(Sp->P[dust_idx].getType() == DUST_PARTICLE_TYPE)
      total_dust_mass += Sp->P[dust_idx].getMass();
  }
  
  if(total_dust_mass < DUST_MASS_TO_DESTROY) return;

  double local_DG = total_dust_mass / gas_mass;
  double dust_to_consume = stellar_mass_formed * local_DG;
  if(dust_to_consume > total_dust_mass) dust_to_consume = total_dust_mass;
  
  double weight_sum = 0.0;
  for(int i = 0; i < n_neighbors; i++)
    if(neighbor_distances[i] > 0)
      weight_sum += 1.0 / neighbor_distances[i];
  
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
    
    if(new_mass < DUST_MASS_TO_DESTROY) {
      Sp->P[dust_idx].setMass(0.0);
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

  // -----------------------------------------------------------------------
  // Diagnostics
  // -----------------------------------------------------------------------
  static int astration_count = 0;
  astration_count++;

  // First 20 events per task: full picture of radius, floor, and consumption
  if(astration_count <= 20)
    DUST_PRINT("[ASTRATION_CHECK|T=%d] #%d: "
           "hsml=%.3e search_r=%.3e cell_size=%.3e "
           "SF=%.3e gas_mass=%.3e D/G=%.3e "
           "dust_to_consume=%.3e n_neighbors=%d consumed=%d mass=%.3e\n",
           All.ThisTask, astration_count,
           hsml, search_radius, dust_hash.cell_size,
           stellar_mass_formed, gas_mass, local_DG,
           dust_to_consume, n_neighbors, dust_consumed_count, dust_consumed_mass);

  // Every 100 events: check if high-neighbor events are systematic
  if(astration_count % 100 == 0)
    DUST_PRINT("[ASTRATION] Event #%d: "
               "search_r=%.2e (hsml=%.2e cell=%.2e) "
               "D/G=%.3e consumed=%d (%.2e Msun)\n",
               astration_count,
               search_radius, hsml, dust_hash.cell_size,
               local_DG, dust_consumed_count, dust_consumed_mass);

  // Flag suspiciously large sweeps immediately
  if(dust_consumed_count > 20)
    DUST_PRINT("[ASTRATION_LARGE|T=%d|Step=%d] consumed=%d neighbors=%d "
           "search_r=%.3e hsml=%.3e cell_size=%.3e D/G=%.3e\n",
           All.ThisTask, All.NumCurrentTiStep,
           dust_consumed_count, n_neighbors,
           search_radius, hsml, dust_hash.cell_size, local_DG);

  if(star_idx >= 0 && dust_consumed_mass > 0) {
    double star_mass = Sp->P[star_idx].getMass();
    if(star_mass > 0)
      Sp->P[star_idx].Metallicity += dust_consumed_mass / star_mass;
  }
}


// *********************************************************************
// Efficiency of radiation pressure coupling to a dust grain, Q_pr
// This is a simplified function that captures the general behavior of Q_pr as a function of grain size and composition.
double radiation_pressure_efficiency(double a_nm, double carbon_fraction)
{
    // Reasonable transition wavelength for optical/near-UV field
    const double a0_nm = 80.0;

    // smooth rise to order unity
    double Qpr = a_nm / (a_nm + a0_nm);

    // give it a modest composition dependence, not huge...
    // silicates have ~0.8× efficiency, carbon ~1.2×
    double Q_sil = 0.8;
    double Q_carbon = 1.2;
    double species_factor = Q_sil + (Q_carbon - Q_sil) * carbon_fraction;

    Qpr *= species_factor;

    if(Qpr > 2.0) Qpr = 2.0;
    if(Qpr < 0.0) Qpr = 0.0;

    return Qpr;
}

/**
 * Estimate stellar luminosity from mass and age for radiation pressure calculation.
 * NOTE: Simplified placeholder, but could really use MESA/Starburst/BPASS, etc
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

/**
 * Apply radiation pressure from nearby young stars to a dust grain.
 *
 * Physics: Stellar photons impart momentum to dust grains via absorption
 * and scattering. The radiation pressure force on a single grain is:
 *
 *   F_rad = (L / 4π r² c) × Q_pr × π a²
 *
 * where Q_pr is the radiation pressure efficiency (Draine & Lee 1984),
 * a is the grain radius, and the flux L/4πr²c assumes optically thin
 * conditions between star and grain.
 *
 * Only stars younger than 100 Myr contribute — older stellar populations
 * have negligible UV/optical luminosity relative to their mass.
 *
 * Search radius: we use the gas smoothing length as the local ISM scale,
 * floored by one star hash cell (so we always search at least one cell),
 * and capped at 10 kpc beyond which the 1/r² flux is negligible.
 *
 * References:
 *   Draine & Lee 1984: Q_pr radiation pressure efficiency curves
 *   Draine & Li 2007: Grain opacity and composition treatment
 */
void dust_radiation_pressure(simparticles *Sp, int dust_idx, int nearest_gas, double dt)
{
  if(!All.DustEnableRadiationPressure) return;

  // ── Diagnostic counters (accumulated over full run on this task) ──────────
  static long long radp_calls       = 0;  // total calls
  static long long radp_with_stars  = 0;  // calls where hash found ≥1 neighbor
  static long long radp_with_young  = 0;  // calls where ≥1 young star contributed
  static double    radp_total_accel = 0.0; // sum of |a_rad| for avg diagnostic
  radp_calls++;
  // ─────────────────────────────────────────────────────────────────────────

  // ── Search radius ─────────────────────────────────────────────────────────
  // Use gas smoothing length as the local ISM scale — this scales correctly
  // with resolution (smaller Hsml at 1024³ means tighter, more physical search).
  // Floor at one star hash cell width to guarantee the hash finds neighbors.
  // Hard cap at 10 kpc: beyond this, even a 1000 L☉/M☉ stellar population
  // contributes < 1e-6 of the flux at 1 kpc, negligible for dynamics.
  // Guard against nearest_gas == -1 before accessing SphP[].
  double search_radius = (nearest_gas >= 0)
      ? std::max(2.0 * (double)Sp->SphP[nearest_gas].Hsml, 2.0 * star_hash.cell_size)
      : 2.0 * star_hash.cell_size;
  if(search_radius > 10.0) search_radius = 10.0;
  // ─────────────────────────────────────────────────────────────────────────

  const int MAX_STAR_NEIGHBORS = 50;
  int    neighbor_indices[MAX_STAR_NEIGHBORS];
  double neighbor_distances[MAX_STAR_NEIGHBORS];
  int    n_neighbors = 0;

  star_hash.find_neighbors(Sp, dust_idx, search_radius,
                           neighbor_indices, neighbor_distances,
                           &n_neighbors, MAX_STAR_NEIGHBORS);

  // ── Periodic diagnostic print (fires whether or not neighbors found) ──────
  if(radp_calls % 500000 == 0 && All.ThisTask == 0)
    DUST_PRINT("[RADIATION_PRESSURE_RATE] calls=%lld  with_any_stars=%.1f%%  "
               "with_young_stars=%.1f%%  avg_|a|=%.2e code\n",
               radp_calls,
               100.0 * radp_with_stars / radp_calls,
               100.0 * radp_with_young / radp_calls,
               radp_total_accel / radp_calls);
  // ─────────────────────────────────────────────────────────────────────────

  if(n_neighbors == 0) return;
  radp_with_stars++;

  // ── Grain properties ──────────────────────────────────────────────────────
  double a_nm  = Sp->DustP[dust_idx].GrainRadius;   // nm
  double a_cm  = a_nm * 1e-7;                        // cm
  double CF    = Sp->DustP[dust_idx].CarbonFraction;
  double Q_pr  = radiation_pressure_efficiency(a_nm, CF);
  // Grain superparticle mass in CGS — needed to convert force → acceleration
  double grain_mass_cgs = Sp->P[dust_idx].getMass() * All.UnitMass_in_g;
  // ─────────────────────────────────────────────────────────────────────────

  const double c_cgs = 2.998e10;   // speed of light, cm/s

  // Accumulate radiation acceleration vector from all young neighbors
  double a_rad[3] = {0.0, 0.0, 0.0};

  for(int i = 0; i < n_neighbors; i++) {
    int star_idx = neighbor_indices[i];

    // Only PartType4 stellar particles
    if(Sp->P[star_idx].getType() != 4) continue;

    // Age cut: stars older than 100 Myr contribute negligible UV/optical flux
    double stellar_age_yr = (All.Time - Sp->P[star_idx].StellarAge)
                            * All.UnitTime_in_s / SEC_PER_YEAR;
    if(stellar_age_yr > 100e6) continue;

    // Luminosity from simplified age-dependent L/M model (see stellar_luminosity())
    double L_cgs = stellar_luminosity(Sp, star_idx);
    if(L_cgs <= 0.0) continue;

    // Distance in CGS — used for inverse-square flux falloff
    double r_kpc = neighbor_distances[i];
    double r_cgs = r_kpc * 3.086e21;   // kpc → cm
    if(r_cgs <= 0.0) continue;

    // Unit vector from star to dust grain (radiation pushes away from star)
    double dxyz[3];
    Sp->nearest_image_intpos_to_pos(Sp->P[dust_idx].IntPos,
                                    Sp->P[star_idx].IntPos, dxyz);
    double unit_vec[3];
    for(int k = 0; k < 3; k++)
      unit_vec[k] = dxyz[k] / r_kpc;   // r_kpc used here since dxyz is in code units

    // Radiation pressure force → acceleration on grain superparticle
    // F = (L / 4π r² c) × Q_pr × π a²   [force on single grain cross-section]
    // a = F / M_grain                     [acceleration of superparticle]
    double flux_cgs = L_cgs / (4.0 * M_PI * r_cgs * r_cgs * c_cgs);
    double accel    = flux_cgs * Q_pr * M_PI * a_cm * a_cm / grain_mass_cgs;

    for(int k = 0; k < 3; k++)
      a_rad[k] += accel * unit_vec[k];
  }

  // ── Update diagnostics ────────────────────────────────────────────────────
  double accel_mag = sqrt(a_rad[0]*a_rad[0] + a_rad[1]*a_rad[1] + a_rad[2]*a_rad[2]);
  if(accel_mag > 0.0) radp_with_young++;
  radp_total_accel += accel_mag;
  // ─────────────────────────────────────────────────────────────────────────

  // ── Apply velocity kick ───────────────────────────────────────────────────
  // Convert CGS acceleration to code units and integrate over timestep dt.
  // a_rad is in cm/s², accel_code converts to code velocity / code time.
  double accel_code = All.UnitVelocity_in_cm_per_s / All.UnitTime_in_s;
  for(int k = 0; k < 3; k++)
    Sp->P[dust_idx].Vel[k] += (a_rad[k] / accel_code) * dt;
  // ─────────────────────────────────────────────────────────────────────────
}

void update_dust_temperature(simparticles *Sp, int dust_idx, int gas_idx, double dt)
{
    // ================================================================
    // Dust temperature via modified-blackbody energy balance.
    //
    // We solve for the equilibrium temperature T_eq analytically:
    //
    //   P_emit(T_eq) = P_CMB + P_ISRF + P_gas
    //
    // where each term is:
    //   P_emit  = C_emit × T_eq^(4+β)          [modified blackbody emission]
    //   P_CMB   = C_emit × T_CMB^(4+β)          [CMB photon bath]
    //   P_ISRF  = C_emit × T_ISRF^(4+β)         [interstellar radiation field]
    //   P_gas   = f_eff × 2 n_H k_B (T_gas - T_dust) α_T v_th π a²
    //                                            [gas-grain collisional coupling]
    //
    // Emission uses the Draine & Lee (1984) opacity law: Q_abs ∝ a T^β, β=2
    // → C_emit × T^(4+β) describes the grain's total emitted power
    //
    // Grain cooling times are << simulation timestep for all grain sizes,
    // so we solve for T_eq directly (equilibrium assumption) and then
    // relax T_dust → T_eq over τ_cool to avoid numerical oscillations.
    //
    // References: Hollenbach & McKee 1979, Draine & Lee 1984,
    //             Mathis et al. 1983 (ISRF), Draine & Li 2007
    // ================================================================

    if(!All.DustEnableCooling) {
        double T_CMB_floor = 2.7 / All.Time;
        if(Sp->DustP[dust_idx].DustTemperature < T_CMB_floor)
            Sp->DustP[dust_idx].DustTemperature = T_CMB_floor;
        return;
    }

    // ----------------------------------------------------------------
    // Grain and gas properties
    // ----------------------------------------------------------------
    const double beta      = 2.0;        // emissivity spectral index (silicate)
    const double rho_grain = 2.4;        // g/cm³, silicate grain density
    const double alpha_T   = 0.1;        // thermal accommodation coefficient
                                         // (Burke & Hollenbach 1983)
    const double sigma_SB  = 5.6704e-5;  // erg/cm²/s/K⁴
    const double T_ref     = 100.0;      // K, Q_abs reference temperature
    const double a_ref_cm  = 1e-5;       // cm = 0.1 μm, Q_abs reference size
    const double Q_ref     = 1.3e-4;     // Q_abs(a_ref, T_ref) [Draine & Lee 1984]

    double a_nm = Sp->DustP[dust_idx].GrainRadius;
    double a_cm = a_nm * 1e-7;
    double CF   = Sp->DustP[dust_idx].CarbonFraction;

    // Carbon grains: lower β (~1.5) and higher normalisation (Draine 2003)
    double beta_eff = beta - 0.5 * CF;
    double Q_eff    = Q_ref * (1.0 + CF) * (a_cm / a_ref_cm);

    double T_CMB  = 2.7 / All.Time;
    double T_dust = Sp->DustP[dust_idx].DustTemperature;
    if(T_dust <= 0.0 || !isfinite(T_dust)) T_dust = T_CMB;

    double T_gas       = get_temperature_from_entropy(Sp, gas_idx);
    double rho_gas_cgs = Sp->SphP[gas_idx].Density * All.cf_a3inv * All.UnitDensity_in_cgs;
    double n_H         = rho_gas_cgs * HYDROGEN_MASSFRAC / PROTONMASS;

    // ----------------------------------------------------------------
    // Emission coefficient: P_emit = C_emit × T^(4+β)
    // ----------------------------------------------------------------
    double C_emit = 4.0 * M_PI * a_cm * a_cm * Q_eff * sigma_SB
                    / pow(T_ref, beta_eff);

    // ----------------------------------------------------------------
    // Heating term 1: CMB photon bath
    // ----------------------------------------------------------------
    double P_CMB = C_emit * pow(T_CMB, 4.0 + beta_eff);

    // ----------------------------------------------------------------
    // Heating term 2: Interstellar radiation field (ISRF)
    // T_ISRF ~ 17K at z=0 from Reach et al. 1995, Mathis et al. 1983; scales weakly
    // with redshift following the increasing SFR density at high-z.
    // ----------------------------------------------------------------
    double z_now  = 1.0 / All.Time - 1.0;
    double T_ISRF = 17.0 * pow(1.0 + z_now, 0.25);
    double P_ISRF = C_emit * pow(T_ISRF, 4.0 + beta_eff);

    // ----------------------------------------------------------------
    // Heating term 3: gas-grain collisional coupling
    //
    // Base rate: P_coll = 2 n_H k_B (T_gas - T_dust) α_T v_th π a²
    //
    // Hollenbach-McKee (1979) effective factor f_eff accounts for:
    //   - Grain charge (positive grains repel ions → reduced coupling)
    //   - Electron contribution at high T_gas (T > ~10^4 K, electrons
    //     dominate and can increase coupling by ×2-3)
    //   - Ion sticking at intermediate T
    //
    // Simple piecewise approximation of HM79 Fig. 1:
    //   cold neutral gas (T < 10^3 K):  f_eff ~ 1.0  (neutrals dominate)
    //   warm gas (10^3–10^4 K):         f_eff ~ 1.5  (ions begin contributing)
    //   hot gas  (> 10^4 K):            f_eff ~ 2.5  (electrons dominate)
    // ----------------------------------------------------------------
    double f_eff;
    if     (T_gas < 1e3) f_eff = 1.0;
    else if(T_gas < 1e4) f_eff = 1.0 + 1.5 * (T_gas - 1e3) / 9e3;
    else                 f_eff = 2.5;

    double v_th  = sqrt(8.0 * BOLTZMANN * T_gas / (M_PI * PROTONMASS));
    double P_gas = f_eff * 2.0 * n_H * BOLTZMANN * (T_gas - T_dust)
                   * alpha_T * v_th * M_PI * a_cm * a_cm;
    double dt_cgs = dt * All.UnitTime_in_s;

    // ----------------------------------------------------------------
    // Back-reaction: remove energy from gas due to dust-gas coupling.
    // P_gas > 0 → heat flows gas→dust → gas cools.
    // P_gas < 0 → dust heats gas (e.g. stochastically heated grains)
    //             include this too for energy conservation.
    // Note that EVERY dust particle will potentially cool nearby gas here...
    // Watch this carefully especially at high resolution...
    // ----------------------------------------------------------------
    double m_grain_cgs = (4.0/3.0) * M_PI * a_cm*a_cm*a_cm * rho_grain;
    if(m_grain_cgs > 0.0) {
        double M_dust_cgs = Sp->P[dust_idx].getMass() * All.UnitMass_in_g;
        double N_grains   = M_dust_cgs / m_grain_cgs;

        double dE_gas_cgs = -P_gas * N_grains * dt_cgs;  // negative when cooling gas
        double M_gas_cgs  = Sp->P[gas_idx].getMass() * All.UnitMass_in_g;
        double du_code    = (dE_gas_cgs / M_gas_cgs)
                            / (All.UnitVelocity_in_cm_per_s
                              * All.UnitVelocity_in_cm_per_s);

        double u_old = Sp->get_utherm_from_entropy(gas_idx);
        double u_new = u_old + du_code;

        // Limit: don't drain more than 20% of internal energy in a single dust step.
        // This prevents multi-dust-neighbor accumulation from overcooling a gas particle
        // faster than the cooling solver can update ne. Without this, Gadget4 can fail to converge in convert_u_to_temp().
        double max_drain = 0.20 * u_old;
        if((u_old - u_new) > max_drain) u_new = u_old - max_drain;

        // Floor: don't cool gas below CMB temperature
        double u_CMB_floor = (1.5 * BOLTZMANN * T_CMB)
                            / (0.6 * PROTONMASS)
                            / (All.UnitVelocity_in_cm_per_s
                                * All.UnitVelocity_in_cm_per_s);
        if(u_new < u_CMB_floor) u_new = u_CMB_floor;

        if(u_new > 0.0 && isfinite(u_new) && u_new != u_old) {
            Sp->set_entropy_from_utherm(u_new, gas_idx);
            // If we changed u by more than ~5%, the stored ne is now stale.
            // Reset it so convert_u_to_temp re-derives ionization from scratch
            // rather than inheriting an inconsistent value.
            if(std::abs(u_new - u_old) > 0.05 * u_old) {
                Sp->SphP[gas_idx].Ne = 0.0;  // let cooling.cc re-solve for ne
            }
            set_thermodynamic_variables_safe(Sp, gas_idx);
        }
    }

    // ----------------------------------------------------------------
    // Solve for equilibrium temperature:
    //   C_emit × T_eq^(4+β) = P_CMB + P_ISRF + P_gas
    //   → T_eq = [(P_CMB + P_ISRF + P_gas) / C_emit]^(1/(4+β))
    //
    // Clamp to CMB floor if collisional cooling (cold gas) would
    // drive T_eq below the photon bath temperature.
    // ----------------------------------------------------------------
    double P_total = P_CMB + P_ISRF + P_gas;
    if(P_total < P_CMB) P_total = P_CMB;

    double T_eq = pow(P_total / C_emit, 1.0 / (4.0 + beta_eff));

    // ----------------------------------------------------------------
    // Relax toward T_eq over the grain cooling timescale τ_cool.
    // This prevents numerical oscillations when T_dust is far from T_eq,
    // and correctly handles the case where dt >> τ_cool (full relaxation)
    // or dt << τ_cool (small perturbation).
    //
    // τ_cool = C_grain × T_dust / P_emit  [thermal energy / cooling rate]
    // ----------------------------------------------------------------
    double c_v            = 7e6;   // erg/g/K (Draine & Li 2001, valid T > 20K)
    double C_grain        = m_grain_cgs * c_v;
    double P_emit_current = C_emit * pow(T_dust + 1.0, 4.0 + beta_eff);
    double tau_cool       = (P_emit_current > 0.0)
                            ? (C_grain * T_dust / P_emit_current)
                            : 1e10;

    double relax  = 1.0 - exp(-dt_cgs / tau_cool);
    double T_new  = T_dust + relax * (T_eq - T_dust);

    // ----------------------------------------------------------------
    // Safety bounds
    // ----------------------------------------------------------------
    if(T_new < T_CMB)               T_new = T_CMB;
    if(T_new > DUST_SUBLIMATION_TEMP) T_new = DUST_SUBLIMATION_TEMP;
    if(!isfinite(T_new))            T_new = T_CMB;

    Sp->DustP[dust_idx].DustTemperature = T_new;
}

/**
 * Update dust particle dynamics (drag, grain growth, coagulation).
 *
 * Called every timestep but only does real work every 10 steps.
 */
void update_dust_dynamics(simparticles *Sp, double dt, MPI_Comm Communicator)
{

  // ============================================================
  // NEW: One-time flag verification at simulation start.
  // Confirms the parameter file is being read correctly.
  // ============================================================
  static bool flags_printed = false;
  if(!flags_printed && All.ThisTask == 0) {
    printf("[DUST_FLAGS|Step=%d] Creation=%d Drag=%d Growth=%d Coagulation=%d "
       "Sputtering=%d ShockDestruction=%d Astration=%d RadPressure=%d "
       "Clumping=%d Cooling=%d\n",
           All.NumCurrentTiStep,
           All.DustEnableCreation, All.DustEnableDrag,
           All.DustEnableGrowth,   All.DustEnableCoagulation,
           All.DustEnableSputtering, All.DustEnableShockDestruction,
           All.DustEnableAstration,  All.DustEnableRadiationPressure,
           All.DustEnableClumping, All.DustEnableCooling);
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
    rebuild_feedback_spatial_hash(Sp, 10.0, 0.1, Communicator);
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
    if(Sp->P[i].getMass() <= DUST_MASS_TO_DESTROY) continue;

    double dist_kpc = -1.0; 
    int nearest_gas = find_nearest_gas_particle(Sp, i, 5.0, &dist_kpc);
    if(nearest_gas < 0) continue;

    // Temperature update runs whenever cooling OR drag is active
    if(All.DustEnableCooling || All.DustEnableDrag)
        update_dust_temperature(Sp, i, nearest_gas, dt * 10);

    dust_gas_interaction(Sp, i, nearest_gas, dt * 10);  // handles drag internally

    if(All.DustEnableRadiationPressure)
      dust_radiation_pressure(Sp, i, nearest_gas, dt * 10);
    
    if(dist_kpc <= 2.0) {
      if(All.DustEnableGrowth)
        dust_grain_growth_subgrid(Sp, i, nearest_gas, dt * 10);

      if(All.DustEnableCoagulation)
        dust_grain_coagulation(Sp, i, nearest_gas, dt * 10);

      if(All.DustEnableShattering)
        dust_grain_shattering(Sp, i, nearest_gas, dt * 10);
    }
  }

  // ============================================================
  // Periodic diagnostics (every 500 steps, task 0 only)
  // ============================================================
  if(All.NumCurrentTiStep % 500 == 0) {
    print_dust_statistics(Sp, Communicator);
    analyze_dust_gas_coupling_local(Sp);
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
        if(Sp->P[i].getType() != DUST_PARTICLE_TYPE || Sp->P[i].getMass() <= DUST_MASS_TO_DESTROY) continue;
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
 */
void destroy_dust_particles(simparticles *Sp)
{
  //static long long destroy_call_count = 0;
  //destroy_call_count++;
  //printf("[DESTROY_DUST_CALLED|T=%d|Step=%d] Call #%lld, scanning %d particles\n",
  //       All.ThisTask, All.NumCurrentTiStep, destroy_call_count, Sp->NumPart);

  int dust_destroyed = 0;
  
  for(int i = 0; i < Sp->NumPart; i++) {
    if((Sp->P[i].getType() == DUST_PARTICLE_TYPE || Sp->P[i].getType() == 3) && 
      (Sp->P[i].getMass() <= DUST_MASS_TO_DESTROY || Sp->P[i].ID.get() == 0)) {
      dust_destroyed++;
    }
  }
  
  if(dust_destroyed == 0) {
    return;
  }
  
  int new_num_gas = 0;
  for(int i = 0; i < Sp->NumGas; i++) {
      // Keep valid gas particles AND any non-dust particles in gas region
      // (converted stars live here temporarily until domain decomp reorders them)
      bool is_dead_dust = (Sp->P[i].getType() == DUST_PARTICLE_TYPE || Sp->P[i].getType() == 3)
                          && (Sp->P[i].getMass() <= DUST_MASS_TO_DESTROY || Sp->P[i].ID.get() == 0);
      if(!is_dead_dust) {
          if(new_num_gas != i) {
              Sp->P[new_num_gas] = Sp->P[i];
              Sp->SphP[new_num_gas] = Sp->SphP[i];
          }
          memset(&Sp->DustP[new_num_gas], 0, sizeof(dust_data));
          new_num_gas++;
      }
  }
  
  int new_num_part = new_num_gas;
  for(int i = Sp->NumGas; i < Sp->NumPart; i++) {
    if((Sp->P[i].getType() == DUST_PARTICLE_TYPE || Sp->P[i].getType() == 3) && 
      (Sp->P[i].getMass() <= DUST_MASS_TO_DESTROY || Sp->P[i].ID.get() == 0)) {
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

    // ── NEW: failure mode diagnostics ─────────────────────────────
    static long long fail_no_neighbor  = 0;  // hash returned -1 outright
    static long long fail_too_far      = 0;  // found something but > max_r_kpc
    static long long fail_wrong_type   = 0;  // stale hash, non-gas particle
    static long long diag_total        = 0;
    // ──────────────────────────────────────────────────────────────

    if(nearest < 0) {
      HashSearchesFailed++;
      fail_no_neighbor++;
      diag_total++;
    } else if(nearest_dist > max_r_kpc) {
      HashSearchesFailed++;
      fail_too_far++;
      diag_total++;
    } else if(Sp->P[nearest].getType() != 0) {
      HashSearchesFailed++;
      fail_wrong_type++;
      diag_total++;
    } else {
      if(out_dist_kpc) *out_dist_kpc = nearest_dist;
      return nearest;
    }

    // Print breakdown every 5 million failures
    if(diag_total % 5000000 == 0 && All.ThisTask == 0)
      printf("[HASH_FAIL_DIAG|Step=%d] total_failed=%lld  "
             "no_neighbor=%lld(%.1f%%)  too_far=%lld(%.1f%%)  "
             "wrong_type=%lld(%.1f%%)  max_r=%.2f kpc  "
             "cell_size=%.3f kpc  n_cells=%d\n",
             All.NumCurrentTiStep, diag_total,
             fail_no_neighbor, 100.0*fail_no_neighbor/diag_total,
             fail_too_far,     100.0*fail_too_far/diag_total,
             fail_wrong_type,  100.0*fail_wrong_type/diag_total,
             max_r_kpc, gas_hash.cell_size, gas_hash.n_cells_per_dim);

    return -1;
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

void print_dust_statistics(simparticles *Sp, MPI_Comm Communicator)
{
  // ============================================================
  // Gather all per-task counters to global values on rank 0
  // Must happen on ALL tasks before the rank-0 early return.
  // ============================================================

  // Local particle stats (loop on all tasks, reduce to rank 0)
  int local_dust_count = 0;
  double local_dust_mass = 0.0, local_avg_size = 0.0, local_avg_temp = 0.0;
  int local_bins[6] = {0};  // cmb_floor, very_cold, cold_ism, warm_ism, hot_ism, pre_sublimation

  for(int i = 0; i < Sp->NumPart; i++) {
    if(Sp->P[i].getType() != DUST_PARTICLE_TYPE || Sp->P[i].getMass() <= DUST_MASS_TO_DESTROY) continue;
    local_dust_count++;
    local_dust_mass  += Sp->P[i].getMass();
    local_avg_size   += Sp->DustP[i].GrainRadius;
    local_avg_temp   += Sp->DustP[i].DustTemperature;
    double T = Sp->DustP[i].DustTemperature;
    if     (T < 10.0)   local_bins[0]++;
    else if(T < 50.0)   local_bins[1]++;
    else if(T < 100.0)  local_bins[2]++;
    else if(T < 500.0)  local_bins[3]++;
    else if(T < 1000.0) local_bins[4]++;
    else                local_bins[5]++;
  }

  int    global_dust_count = 0;
  double global_dust_mass  = 0.0, global_avg_size = 0.0, global_avg_temp = 0.0;
  int    global_bins[6]    = {0};

  MPI_Reduce(&local_dust_count, &global_dust_count, 1, MPI_INT,    MPI_SUM, 0, Communicator);
  MPI_Reduce(&local_dust_mass,  &global_dust_mass,  1, MPI_DOUBLE, MPI_SUM, 0, Communicator);
  MPI_Reduce(&local_avg_size,   &global_avg_size,   1, MPI_DOUBLE, MPI_SUM, 0, Communicator);
  MPI_Reduce(&local_avg_temp,   &global_avg_temp,   1, MPI_DOUBLE, MPI_SUM, 0, Communicator);
  MPI_Reduce(local_bins, global_bins, 6, MPI_INT, MPI_SUM, 0, Communicator);

  // Module-level counters are already per-task accumulators — reduce them too
  long long global_NDustCreated             = 0, global_NDustCreatedBySNII      = 0;
  long long global_NDustCreatedByAGB        = 0;
  long long global_NDustDestroyedByThermal  = 0, global_NDustDestroyedByShock   = 0;
  long long global_NDustDestroyedByAstration= 0, global_NDustDestroyedByCleanup = 0;
  long long global_NDustDestroyedByCorruption=0, global_NDustDestroyedByBadGasIndex=0;
  long long global_NGrainGrowthEvents       = 0, global_NGrainErosionEvents     = 0;
  long long global_NCoagulationEvents       = 0;
  long long global_HashSearches             = 0, global_HashSearchesFailed      = 0;
  double global_TotalMassGrown              = 0.0;
  double global_TotalMassDestroyedByThermal = 0.0, global_TotalMassDestroyedByShock = 0.0;
  double global_TotalMassErodedByThermal    = 0.0, global_TotalMassErodedByShock    = 0.0;
  double global_TotalDustMassAstrated       = 0.0;
  long long global_NShatteringEvents = 0;
  double global_TotalSizeReductionShattering = 0.0;

  MPI_Reduce(&NShatteringEvents, &global_NShatteringEvents, 1, MPI_LONG_LONG, MPI_SUM, 0, Communicator);
  MPI_Reduce(&TotalSizeReductionShattering, &global_TotalSizeReductionShattering, 1, MPI_DOUBLE, MPI_SUM, 0, Communicator);
  MPI_Reduce(&NDustCreated,              &global_NDustCreated,              1, MPI_LONG_LONG, MPI_SUM, 0, Communicator);
  MPI_Reduce(&NDustCreatedBySNII,        &global_NDustCreatedBySNII,        1, MPI_LONG_LONG, MPI_SUM, 0, Communicator);
  MPI_Reduce(&NDustCreatedByAGB,         &global_NDustCreatedByAGB,         1, MPI_LONG_LONG, MPI_SUM, 0, Communicator);
  MPI_Reduce(&NDustDestroyedByThermal,   &global_NDustDestroyedByThermal,   1, MPI_LONG_LONG, MPI_SUM, 0, Communicator);
  MPI_Reduce(&NDustDestroyedByShock,     &global_NDustDestroyedByShock,     1, MPI_LONG_LONG, MPI_SUM, 0, Communicator);
  MPI_Reduce(&NDustDestroyedByAstration, &global_NDustDestroyedByAstration, 1, MPI_LONG_LONG, MPI_SUM, 0, Communicator);
  MPI_Reduce(&NDustDestroyedByCleanup,   &global_NDustDestroyedByCleanup,   1, MPI_LONG_LONG, MPI_SUM, 0, Communicator);
  MPI_Reduce(&NDustDestroyedByCorruption,&global_NDustDestroyedByCorruption,1, MPI_LONG_LONG, MPI_SUM, 0, Communicator);
  MPI_Reduce(&NDustDestroyedByBadGasIndex,&global_NDustDestroyedByBadGasIndex,1,MPI_LONG_LONG,MPI_SUM, 0, Communicator);
  MPI_Reduce(&NGrainGrowthEvents,        &global_NGrainGrowthEvents,        1, MPI_LONG_LONG, MPI_SUM, 0, Communicator);
  MPI_Reduce(&NGrainErosionEvents,       &global_NGrainErosionEvents,       1, MPI_LONG_LONG, MPI_SUM, 0, Communicator);
  MPI_Reduce(&NCoagulationEvents,        &global_NCoagulationEvents,        1, MPI_LONG_LONG, MPI_SUM, 0, Communicator);
  MPI_Reduce(&HashSearches,              &global_HashSearches,              1, MPI_LONG_LONG, MPI_SUM, 0, Communicator);
  MPI_Reduce(&HashSearchesFailed,        &global_HashSearchesFailed,        1, MPI_LONG_LONG, MPI_SUM, 0, Communicator);
  MPI_Reduce(&TotalMassGrown,            &global_TotalMassGrown,            1, MPI_DOUBLE,    MPI_SUM, 0, Communicator);
  MPI_Reduce(&TotalMassDestroyedByThermal,&global_TotalMassDestroyedByThermal,1,MPI_DOUBLE,   MPI_SUM, 0, Communicator);
  MPI_Reduce(&TotalMassDestroyedByShock, &global_TotalMassDestroyedByShock, 1, MPI_DOUBLE,    MPI_SUM, 0, Communicator);
  MPI_Reduce(&TotalMassErodedByThermal,  &global_TotalMassErodedByThermal,  1, MPI_DOUBLE,    MPI_SUM, 0, Communicator);
  MPI_Reduce(&TotalMassErodedByShock,    &global_TotalMassErodedByShock,    1, MPI_DOUBLE,    MPI_SUM, 0, Communicator);
  MPI_Reduce(&TotalDustMassAstrated,     &global_TotalDustMassAstrated,     1, MPI_DOUBLE,    MPI_SUM, 0, Communicator);

  // Now rank 0 can print global values
  if(All.ThisTask != 0) return;

  if(global_dust_count > 0) {
    global_avg_size /= global_dust_count;
    global_avg_temp /= global_dust_count;
  }

  DUST_PRINT("=== STATISTICS (global) ===\n");
  DUST_PRINT("STATISTICS Particles: %d  Mass: %.3e Msun\n", global_dust_count, global_dust_mass);
  DUST_PRINT("STATISTICS Avg grain size: %.2f nm\n", global_avg_size);
  DUST_PRINT("STATISTICS Avg temperature: %.1f K\n", global_avg_temp);
  if(global_dust_count > 0) {
    DUST_PRINT("STATISTICS  < 10 K (CMB floor):        %d (%.1f%%)\n", global_bins[0], 100.0*global_bins[0]/global_dust_count);
    DUST_PRINT("STATISTICS  10-50 K (Cold clouds):     %d (%.1f%%)\n", global_bins[1], 100.0*global_bins[1]/global_dust_count);
    DUST_PRINT("STATISTICS  50-100 K (Cool ISM):       %d (%.1f%%)\n", global_bins[2], 100.0*global_bins[2]/global_dust_count);
    DUST_PRINT("STATISTICS  100-500 K (Warm ISM):      %d (%.1f%%)\n", global_bins[3], 100.0*global_bins[3]/global_dust_count);
    DUST_PRINT("STATISTICS  500-1000 K (Hot ISM):      %d (%.1f%%)\n", global_bins[4], 100.0*global_bins[4]/global_dust_count);
    DUST_PRINT("STATISTICS  1000-2000 K (Near sublim): %d (%.1f%%)\n", global_bins[5], 100.0*global_bins[5]/global_dust_count);
  }
  DUST_PRINT("========================\n");
  DUST_PRINT("STATISTICS Hash searches:       %lld\n", global_HashSearches);
  if(global_HashSearches > 0) {
    DUST_PRINT("STATISTICS Hash success rate:    %.1f%%\n",
               100.0 * (global_HashSearches - global_HashSearchesFailed) / global_HashSearches);
  }
  if(global_HashSearchesFailed > 0) {
    DUST_PRINT("STATISTICS [WARNING] Failed searches: %lld (%.1f%%)\n",
               global_HashSearchesFailed, 100.0 * global_HashSearchesFailed / global_HashSearches);
  }
  DUST_PRINT("STATISTICS Growth events: %lld (%.2e Msun grown)\n", global_NGrainGrowthEvents, global_TotalMassGrown);
  DUST_PRINT("STATISTICS Partial erosion events: %lld\n", global_NGrainErosionEvents);
  DUST_PRINT("========================\n");

  long long global_total_by_physics  = global_NDustDestroyedByThermal + global_NDustDestroyedByShock + global_NDustDestroyedByAstration;
  long long global_total_by_internal = global_NDustDestroyedByCleanup + global_NDustDestroyedByCorruption + global_NDustDestroyedByBadGasIndex;
  long long global_total_destroyed   = global_total_by_physics + global_total_by_internal;

  DUST_PRINT("=== DESTRUCTION AUDIT (global) ===\n");
  DUST_PRINT("  --- Physics mechanisms (gated by enable flags) ---\n");
  DUST_PRINT("  Thermal sputtering:     %lld  (flag=%d)\n"
             "    full destructions:    %.2e Msun\n"
             "    partial erosion:      %.2e Msun\n",
             global_NDustDestroyedByThermal, All.DustEnableSputtering,
             global_TotalMassDestroyedByThermal, global_TotalMassErodedByThermal);
  DUST_PRINT("  Shock destruction:      %lld  (flag=%d)\n"
             "    full destructions:    %.2e Msun\n"
             "    partial erosion:      %.2e Msun\n",
             global_NDustDestroyedByShock, All.DustEnableShockDestruction,
             global_TotalMassDestroyedByShock, global_TotalMassErodedByShock);
  DUST_PRINT("  Astration:              %lld  (flag=%d, mass=%.2e Msun)\n",
             global_NDustDestroyedByAstration, All.DustEnableAstration, global_TotalDustMassAstrated);
  DUST_PRINT("  --- Internal / unconditional paths ---\n");
  DUST_PRINT("  cleanup_invalid():      %lld  ← domain exchange corruption?\n", global_NDustDestroyedByCleanup);
  DUST_PRINT("  growth corruption:      %lld  ← zeroed DustP state in growth loop\n", global_NDustDestroyedByCorruption);
  DUST_PRINT("  bad gas index (stale):  %lld  ← hash returned non-gas type\n", global_NDustDestroyedByBadGasIndex);
  DUST_PRINT("  --- Totals ---\n");
  DUST_PRINT("  By physics:             %lld\n", global_total_by_physics);
  DUST_PRINT("  By internal paths:      %lld  ← Ideally this should be 0\n", global_total_by_internal);
  DUST_PRINT("  GRAND TOTAL destroyed:  %lld\n", global_total_destroyed);
  DUST_PRINT("  Total created:          %lld\n", global_NDustCreated);
  DUST_PRINT("  Created by SNII:        %lld\n", global_NDustCreatedBySNII);
  DUST_PRINT("  Created by AGB:         %lld\n", global_NDustCreatedByAGB);
  DUST_PRINT("  Net (created-destroyed):%lld\n", global_NDustCreated - global_total_destroyed);
  DUST_PRINT("  Current live particles: %d\n",   global_dust_count);
  DUST_PRINT("=======================================\n");
    if(global_dust_count > 0 && global_NDustDestroyedByShock > 0) {
    DUST_PRINT("  Shock events / live particles:  %.3f  (target: < 0.5)\n",
               (double)global_NDustDestroyedByShock / global_dust_count);
    }
  DUST_PRINT("STATISTICS Coagulation events: %lld \n",
             global_NCoagulationEvents);
  DUST_PRINT("STATISTICS Shattering events: %lld (avg da=%.2f nm/event)\n",
           global_NShatteringEvents,
           global_NShatteringEvents > 0 
             ? global_TotalSizeReductionShattering / global_NShatteringEvents 
             : 0.0);
  DUST_PRINT("========================\n");
}

void analyze_dust_gas_coupling_local(simparticles *Sp)
{
  if(All.ThisTask != 0) return;
  
  double total_vel_diff = 0.0;
  int dust_count = 0;
  double max_vel_diff = 0.0;
    
  for(int i = 0; i < Sp->NumPart; i++) {
    if(Sp->P[i].getType() == DUST_PARTICLE_TYPE && Sp->P[i].getMass() > DUST_MASS_TO_DESTROY) {
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
 * Shock destruction efficiency as a function of shock velocity and grain composition.
 *
 * Separate piecewise linear curves for carbonaceous and silicate grains,
 * interpolated from Bocchio et al. 2014 Table 6 ("this study" row).
 * Mixed grains interpolate linearly between the two curves by CarbonFraction.
 *
 * Carbonaceous grains are far more vulnerable at low velocities (77% at 50 km/s)
 * while silicates require stronger shocks to achieve significant destruction.
 */
double get_shock_destruction_efficiency(double v, double carbon_fraction)
{
  // -----------------------------------------------------------------------
  // Carbonaceous curve (Bocchio+2014 Table 6, carbonaceous row)
  // -----------------------------------------------------------------------
  double eps_carb;
  if     (v <  50.0) eps_carb = 0.0;
  else if(v <  75.0) eps_carb = 0.77 + 0.06  * (v -  50.0) / 25.0;
  else if(v < 100.0) eps_carb = 0.83 + 0.08  * (v -  75.0) / 25.0;
  else if(v < 125.0) eps_carb = 0.91 + 0.05  * (v - 100.0) / 25.0;
  else if(v < 150.0) eps_carb = 0.96 + 0.03  * (v - 125.0) / 25.0;
  else if(v < 175.0) eps_carb = 0.99 + 0.01  * (v - 150.0) / 25.0;
  else               eps_carb = 1.00;

  // -----------------------------------------------------------------------
  // Silicate curve (Bocchio+2014 Table 6, silicate row)
  // -----------------------------------------------------------------------
  double eps_sil;
  if     (v <  50.0) eps_sil = 0.0;
  else if(v <  75.0) eps_sil = 0.02 + 0.10  * (v -  50.0) / 25.0;
  else if(v < 100.0) eps_sil = 0.12 + 0.17  * (v -  75.0) / 25.0;
  else if(v < 125.0) eps_sil = 0.29 + 0.17  * (v - 100.0) / 25.0;
  else if(v < 150.0) eps_sil = 0.46 + 0.07  * (v - 125.0) / 25.0;
  else if(v < 175.0) eps_sil = 0.53 + 0.14  * (v - 150.0) / 25.0;
  else if(v < 200.0) eps_sil = 0.67 + 0.0   * (v - 175.0) / 25.0;
  else               eps_sil = 0.67;  // Bocchio table ends at 200 km/s — hold flat

  // -----------------------------------------------------------------------
  // Blend by carbon fraction: pure silicate at CF=0, pure carbon at CF=1
  // -----------------------------------------------------------------------
  double cf = carbon_fraction;
  if(cf < 0.0) cf = 0.0;
  if(cf > 1.0) cf = 1.0;

  return cf * eps_carb + (1.0 - cf) * eps_sil;
}

/**
 * Destroy or erode dust grains within the blast radius of a Type II supernova.
 *
 * Physics approach: Sedov-Taylor blast wave with explicit separation between
 * the physical shock radius and the hash search radius.
 *
 * The physical shock radius is computed from the local gas density via the
 * self-similar Sedov-Taylor solution at a characteristic time of 0.3 Myr.
 * The shock velocity is derived self-consistently from this PHYSICAL radius
 * and density — not from the inflated search radius. This is critical: since
 * v ∝ R^(-3/2), using an inflated radius for the velocity calculation would
 * drop velocities to ~0.1 km/s, far below the 50 km/s destruction threshold.
 *
 * At 1024³ and higher resolutions, the physical shock radius (~0.03-0.05 kpc)
 * is smaller than the dust hash cell size (~0.5-1.4 kpc). A wider effective
 * search radius is therefore used purely for neighbor finding — this is a
 * subgrid volume correction that allows the hash to locate dust representative
 * of the local shocked ISM. All dust found within the effective search radius
 * is treated as a subgrid representative of the shocked volume and receives
 * the full physical shock velocity.
 *
 * The mass-based subgrid approach (Steps 7-8) computes the EXPECTED MASS
 * destroyed within the physical shock volume and distributes it proportionally
 * across all representative grains in the search volume:
 *
 *   M_destroy = M_local × f_vol × ε(v, CF)
 *
 * where f_vol = (R_phys / R_search)³ is the volume correction factor and
 * ε(v, CF) is the Bocchio+2014 efficiency blended by carbon fraction.
 * Grains eroded below DUST_MIN_GRAIN_SIZE are fully destroyed; survivors
 * have their radius updated self-consistently via a ∝ m^(1/3).
 *
 * Called once per SN feedback event from the stellar feedback loop in feedback.cc.
 *
 * References:
 *   McKee & Ostriker 1977: SN energetics (E = 10^51 erg)
 *   Sedov 1959: Self-similar blast wave solution
 *   Jones et al. 1994: Grain shattering threshold (~50 km/s)
 *   Bocchio et al. 2014: Grain destruction efficiencies
 */
void destroy_dust_from_sn_shocks(simparticles *Sp, int sn_star_idx,
                                  double sn_energy, double metals_produced,
                                  MPI_Comm Communicator)
{
  if(!All.DustEnableShockDestruction) return;

  // =========================================================================
  // Guard: dust hash must be built and populated before we can find neighbors.
  // Early timesteps before the first dust creation will legitimately hit this.
  // =========================================================================
  static long long sn_skipped_no_hash = 0;
  if(!dust_hash.is_built || dust_hash.total_particles == 0) {
    sn_skipped_no_hash++;
    if(sn_skipped_no_hash <= 20 && All.ThisTask == 0)
      DUST_PRINT("[SN_SKIP] Call skipped: hash built=%d total=%d\n",
                 dust_hash.is_built, dust_hash.total_particles);
    return;
  }

  // =========================================================================
  // Step 1: Find nearest gas particle and read its physical density.
  //
  // This density is used for both the Sedov radius and velocity calculations.
  // A fallback of ~1 cm^-3 is used if no gas neighbor is found.
  // =========================================================================
  double nearest_dist  = -1.0;
  int    sn_nearest_gas = gas_hash.find_nearest_particle(Sp, sn_star_idx,
                                                          50.0, &nearest_dist);

  // Guard: stale hash may return a converted star instead of gas
  if(sn_nearest_gas >= 0 && Sp->P[sn_nearest_gas].getType() != 0)
    sn_nearest_gas = -1;

  double gas_density_cgs = 1.0 * PROTONMASS;  // fallback: ~1 cm^-3

  if(sn_nearest_gas >= 0 && sn_nearest_gas < Sp->NumGas) {
    double gas_density_code = Sp->SphP[sn_nearest_gas].Density * All.cf_a3inv;
    double measured         = gas_density_code * All.UnitDensity_in_cgs;
    if(measured > 0.01 * PROTONMASS)
      gas_density_cgs = measured;
  }

  // =========================================================================
  // Step 2: Compute the PHYSICAL Sedov-Taylor shock radius.
  //
  // Evaluated at characteristic_time_myr = 0.3 Myr, which gives a
  // representative Sedov-phase radius of ~30-80 pc at typical ISM densities.
  // At 1024³ resolution this radius (~0.03-0.05 kpc) is typically smaller
  // than the dust hash cell size — see Step 3 for how this is handled.
  // =========================================================================
  const double sn_energy_erg         = 1e51;
  const double characteristic_time_myr = 0.3;

  // Use the actual local gas density for the Sedov-Taylor radius, so dense
  // star-forming environments naturally produce smaller shock radii and
  // higher velocities. DustShockAmbientDensity is a FLOOR that prevents
  // unphysically large radii, making shock destruction resolution-independent.
  double rho_sedov = std::min(gas_density_cgs,
                              All.DustShockAmbientDensity * PROTONMASS);

  double physical_radius_kpc = calculate_sn_shock_radius(sn_energy_erg,
                                                          rho_sedov,
                                                          characteristic_time_myr);
  if(physical_radius_kpc < 0.001) physical_radius_kpc = 0.001;  // 1 pc floor

  // =========================================================================
  // Step 3: Compute shock velocity from the PHYSICAL radius.
  //
  // CRITICAL: velocity must use the physical radius, not the search radius.
  // From the Sedov-Taylor solution:
  //   v = (2/5) * xi^(5/2) * sqrt(E/rho) * R^(-3/2)
  //
  // Since v ∝ R^(-3/2), using an inflated search radius (e.g. 0.5 kpc instead
  // of 0.035 kpc) would reduce the velocity by a factor of ~60, dropping it
  // from ~40 km/s to ~0.7 km/s — well below any destruction threshold.
  //
  // Dense environments produce smaller radii and higher velocities, naturally
  // encoding faster deceleration in high-density ISM.
  // =========================================================================
  double shock_velocity_km_s = calculate_sedov_velocity_from_radius(physical_radius_kpc, rho_sedov);

  // =========================================================================
  // Step 4: Effective search radius for hash neighbor finding.
  // =========================================================================
  // Physical shock radius (~0.03-0.07 kpc) is always smaller than the
  // dust hash cell size at current resolutions, so we widen the search
  // to cover at least 2 cell widths. All dust found is treated as a
  // subgrid representative of the local shocked ISM volume.
  // The velocity is derived from the PHYSICAL radius, not this search radius.
    double effective_search_radius = physical_radius_kpc;
    if(effective_search_radius < 2.0 * dust_hash.cell_size)
        effective_search_radius = 2.0 * dust_hash.cell_size;
    if(effective_search_radius > 3.0)
        effective_search_radius = 3.0;


  // =========================================================================
  // Step 5: Log shock parameters for the first 20 SN events (debug mode).
  // =========================================================================
  if(All.DustDebugLevel > 0) {
    static int sn_call_count = 0;
    sn_call_count++;
    if((sn_call_count <= 20 || sn_call_count % 500 == 0) && All.ThisTask == 0)
      DUST_PRINT("[SN_SHOCK_DEBUG] Call #%d: "
                "physical_radius=%.3f kpc  effective_search=%.3f kpc  "
                "shock_velocity=%.1f km/s  "
                "rho_local=%.3e g/cm3  rho_sedov=%.3e g/cm3 %s\n",
                sn_call_count, physical_radius_kpc, effective_search_radius,
                shock_velocity_km_s, gas_density_cgs, rho_sedov,
                (rho_sedov < gas_density_cgs) ? "[DENSITY_CAPPED]" : "[LOCAL_DENSITY]");
  }

// =========================================================================
  // Step 6: Find all dust superparticles within the effective search radius.
  //
  // The 2048 cap is a per-SN safety limit. Even when saturated, the mass-based
  // approach in Step 7 remains correct — we are sampling the densest part of
  // the local dust distribution, which is a conservative (not over-destructive)
  // approximation.
  // =========================================================================
  int    neighbors[2048];
  double distances[2048];
  int    n_found = 0;

  dust_hash.find_neighbors(Sp, sn_star_idx, effective_search_radius,
                           neighbors, distances, &n_found, 2048);

  if(n_found == 0) return;

  // =========================================================================
  // Step 7: Mass-based subgrid shock destruction.
  //
  // MOTIVATION:
  // At 2048³, the physical shock radius (~20-50 pc) is much smaller than
  // the dust hash cell size (~7 kpc). This subgrid treatment is to compute the EXPECTED MASS destroyed
  // within the physical shock volume and distribute that mass loss across all
  // dust grains in the search volume.
  //
  // ALGORITHM:
  //   1. Sum total dust mass M_local in the search volume.
  //   2. Compute the fraction physically within R_ST: f_vol = (R_phys/R_search)³
  //   3. Apply Bocchio+2014 efficiency at shock_velocity to get M_destroy.
  //   4. Distribute M_destroy proportionally across all found grains by mass.
  //   5. Grains eroded below DUST_MIN_GRAIN_SIZE are fully destroyed.
  //
  // RESOLUTION INDEPENDENCE:
  // At 512³ with large hash cells, f_vol ~ 0.01-0.1 and few grains are found.
  // At 2048³ with small cells, f_vol ~ 1e-5 but many grains are found.
  // In both cases: M_destroy = M_local × f_vol × efficiency converges to the
  // same physical answer — the mass fraction destroyed per SN event in the
  // local ISM.
  //
  // NOTE: We use a single blended efficiency at the mean grain composition
  // rather than per-grain Bocchio curves. This is appropriate for the subgrid
  // treatment since we are distributing a bulk mass loss, not tracking
  // individual grain trajectories through the shock front.
  // =========================================================================

  // --- Accumulate local dust mass and mean carbon fraction ---
  double M_dust_local  = 0.0;
  double CF_sum        = 0.0;
  int    n_dust_found  = 0;

  for(int k = 0; k < n_found; k++) {
    int i = neighbors[k];
    if(Sp->P[i].getType() != DUST_PARTICLE_TYPE) continue;
    if(Sp->P[i].getMass() <= 0.0) continue;
    double m = Sp->P[i].getMass();
    M_dust_local += m;
    CF_sum       += Sp->DustP[i].CarbonFraction * m;  // mass-weighted CF
    n_dust_found++;
  }

  if(M_dust_local <= 0.0 || n_dust_found == 0) return;

  double CF_mean = CF_sum / M_dust_local;

  // --- Volume correction factor ---
  double f_vol = pow(physical_radius_kpc / effective_search_radius, 3.0);
  if(f_vol > 1.0) f_vol = 1.0;
  if(f_vol < 0.0) f_vol = 0.0;

  // --- Bocchio+2014 destruction efficiency at shock velocity ---
  double bocchio_eff = get_shock_destruction_efficiency(shock_velocity_km_s, CF_mean);

  // --- Expected mass to destroy ---
  // M_destroy = (dust mass in physical shock volume) × (destruction efficiency)
  double M_to_destroy = M_dust_local * f_vol * bocchio_eff;

  // =========================================================================
  // Step 8: Distribute mass destruction proportionally across found grains.
  //
  // Each grain loses mass proportional to its share of M_dust_local.
  // Grain radius is updated self-consistently: m ∝ a³ → a_new = a*(m_new/m)^(1/3).
  // Grains eroded below DUST_MIN_GRAIN_SIZE are fully destroyed and their
  // remaining mass is returned to the nearest gas cell as metals.
  // =========================================================================
  static long long sn_total_calls = 0;
  static long long sn_found_dust  = 0;
  sn_total_calls++;
  if(n_dust_found > 0) sn_found_dust++;

  int    dust_destroyed  = 0;
  int    dust_eroded     = 0;
  double M_actually_lost = 0.0;

  if(M_to_destroy > 0.0) {
    for(int k = 0; k < n_found; k++) {
      int i = neighbors[k];
      if(Sp->P[i].getType() != DUST_PARTICLE_TYPE) continue;
      double m = Sp->P[i].getMass();
      if(m <= 0.0) continue;

      // Proportional mass loss for this grain
      double mass_loss = M_to_destroy * (m / M_dust_local);
      double new_mass  = m - mass_loss;

      // Update grain radius self-consistently: a ∝ m^(1/3)
      double a_old = Sp->DustP[i].GrainRadius;
      double a_new = a_old * cbrt(new_mass / m);

      if(a_new < DUST_MIN_GRAIN_SIZE || new_mass <= 0.0) {
        // Grain eroded below minimum size — destroy fully
        M_actually_lost += m;
        destroy_dust_particle_to_gas(Sp, i, sn_nearest_gas,
                                     &NDustDestroyedByShock,
                                     &TotalMassDestroyedByShock);
        dust_destroyed++;
      } else {
        // Partial erosion — update mass and radius
        Sp->P[i].setMass(new_mass);
        Sp->DustP[i].GrainRadius = a_new;

        // Return eroded mass to nearest gas as metals
        if(sn_nearest_gas >= 0) {
          double gas_mass = Sp->P[sn_nearest_gas].getMass();
          if(gas_mass > 0.0)
            Sp->SphP[sn_nearest_gas].Metallicity += mass_loss / gas_mass;
          #ifdef STARFORMATION
          Sp->SphP[sn_nearest_gas].MassMetallicity =
              Sp->P[sn_nearest_gas].getMass()
              * Sp->SphP[sn_nearest_gas].Metallicity;
          #endif
        }

        LocalDustMassChange -= mass_loss;
        M_actually_lost     += mass_loss;
        TotalMassErodedByShock += mass_loss;
        NGrainErosionEvents++;
        dust_eroded++;
      }
    }
  }

  // =========================================================================
  // Diagnostics
  // =========================================================================
  if(sn_total_calls % 1000 == 0 && All.ThisTask == 0)
    DUST_PRINT("[SN_RATE] %lld SN calls, %lld found dust (%.1f%%)\n",
               sn_total_calls, sn_found_dust,
               100.0 * sn_found_dust / sn_total_calls);

  if(All.ThisTask == 0 && M_to_destroy > 0.0)
    DUST_PRINT("[DUST_SN] physical_r=%.4f kpc  search_r=%.3f kpc  "
               "v=%.1f km/s  f_vol=%.3e  eff=%.3f  "
               "M_local=%.3e  M_target=%.3e  M_lost=%.3e Msun  "
               "n_dust=%d  destroyed=%d  eroded=%d\n",
               physical_radius_kpc, effective_search_radius,
               shock_velocity_km_s, f_vol, bocchio_eff,
               M_dust_local, M_to_destroy, M_actually_lost,
               n_dust_found, dust_destroyed, dust_eroded);

  // Periodic debug: verify f_vol and efficiency are in expected ranges
  if(All.DustDebugLevel > 0) {
    static int sn_debug_count = 0;
    sn_debug_count++;
    if((sn_debug_count <= 20 || sn_debug_count % 500 == 0) && All.ThisTask == 0)
      DUST_PRINT("[SN_DEBUG] Call #%d: "
                 "physical_r=%.4f kpc  search_r=%.3f kpc  "
                 "v=%.1f km/s  f_vol=%.3e  bocchio_eff=%.3f  "
                 "CF_mean=%.2f  M_local=%.3e  M_to_destroy=%.3e  "
                 "rho_local=%.3e  rho_sedov=%.3e %s\n",
                 sn_debug_count,
                 physical_radius_kpc, effective_search_radius,
                 shock_velocity_km_s, f_vol, bocchio_eff,
                 CF_mean, M_dust_local, M_to_destroy,
                 gas_density_cgs, rho_sedov,
                 (rho_sedov < gas_density_cgs) ? "[DENSITY_CAPPED]" : "[LOCAL_DENSITY]");
  }
}

/**
 * Find nearest dust particle to a gas cell (O(N) brute force). DEFUNCT! should not use.
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
       Sp->P[i].getMass() > DUST_MASS_TO_DESTROY) {
      
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
 * Dust grain coagulation in dense cold gas.
 *
 * Grain-grain collisions in dense environments cause smaller grains to
 * stick together, growing toward larger sizes without accreting gas-phase
 * metals (unlike grain growth in dust_grain_growth_subgrid). Coagulation
 * is only active above a density threshold and below a temperature ceiling,
 * targeting molecular cloud environments where collision rates are high.
 *
 * Timescale: tau_coag ~ 10 Myr × (100/n_eff) × (0.1 um/a)
 * Reference: Hirashita & Kuo 2011, Zhukovska et al. 2008
 *
 * NOTE: The 3000 K temperature ceiling is deliberately generous relative to
 * the true molecular cloud temperature (~10-50 K) because at 512^3-1024^3
 * resolution the SPH kernel averages over unresolved cold clumps. The
 * coagulation timescale formula still encodes the correct density dependence
 * for the actual grain physics.
 */
void dust_grain_coagulation(simparticles *Sp, int dust_idx, int gas_idx, double dt)
{
  if(!All.DustEnableCoagulation) return;

  // -----------------------------------------------------------------------
  // Diagnostics
  // -----------------------------------------------------------------------
  static int coag_calls       = 0;
  static int coag_failed_dens = 0;
  static int coag_failed_temp = 0;
  static int coag_failed_size = 0;
  static int coag_passed      = 0;
  coag_calls++;

  if(coag_calls % 50000 == 0 && All.ThisTask == 0)
    DUST_PRINT("[COAG_DIAG] calls=%d  failed: dens=%d(%.1f%%) temp=%d(%.1f%%) "
               "size=%d(%.1f%%)  passed=%d(%.1f%%)\n",
               coag_calls,
               coag_failed_dens, 100.0 * coag_failed_dens / coag_calls,
               coag_failed_temp, 100.0 * coag_failed_temp / coag_calls,
               coag_failed_size, 100.0 * coag_failed_size / coag_calls,
               coag_passed,      100.0 * coag_passed      / coag_calls);

  // -----------------------------------------------------------------------
  // Gate 1: density — coagulation only active in dense environments.
  // Gate on n_eff (clumping-weighted) for resolution-independence.
  // Physical target: molecular cloud conditions, n_H ~ 100 cm^-3.
  // -----------------------------------------------------------------------
  double gas_density_code = Sp->SphP[gas_idx].Density * All.cf_a3inv;
  double gas_density_cgs  = gas_density_code * All.UnitDensity_in_cgs;
  double n_H              = (gas_density_cgs * HYDROGEN_MASSFRAC) / PROTONMASS;

  double DustClumpingFactor = dust_clumping_factor(n_H, Sp->SphP[gas_idx].Sfr > DUST_SFR_EPS);
  double n_eff              = n_H * DustClumpingFactor;

  if(n_eff < All.DustCollisionDensityThresh) { coag_failed_dens++; return; }

  // -----------------------------------------------------------------------
  // Gate 2: temperature — target cold ISM/molecular cloud environments.
  // 3000 K ceiling is generous relative to true molecular cloud temps
  // (~10-50 K) to account for SPH kernel smoothing at finite resolution.
  // -----------------------------------------------------------------------
  double T_gas = get_temperature_from_entropy(Sp, gas_idx);

  if(T_gas > 3000.0) { coag_failed_temp++; return; }

  // -----------------------------------------------------------------------
  // Gate 3: grain validity and size cap
  // -----------------------------------------------------------------------
  double a      = Sp->DustP[dust_idx].GrainRadius;
  double M_dust = Sp->P[dust_idx].getMass();

  if(a <= 0.0 || M_dust <= 0.0 || !isfinite(a) || !isfinite(M_dust)) return;
  if(a >= All.DustCoagulationMaxSize) { coag_failed_size++; return; }

  coag_passed++;

  // -----------------------------------------------------------------------
  // Coagulation timescale: tau ~ 10 Myr × (100/n_eff) × (0.1 um / a)
  // Smaller grains coagulate faster (more surface area per unit mass).
  // Reference: Hirashita & Kuo (2011).
  // -----------------------------------------------------------------------
  double a_micron    = a / 1000.0;  // nm → micron
  double tau_coag_yr = 1e7 * (100.0 / n_eff) * (0.1 / a_micron);
  tau_coag_yr       *= All.DustCoagulationCalibration;
  tau_coag_yr        = std::max(tau_coag_yr, 1e6);
  tau_coag_yr        = std::min(tau_coag_yr, 1e9);

  double tau_coag = tau_coag_yr * SEC_PER_YEAR / All.UnitTime_in_s;

  // -----------------------------------------------------------------------
  // Size update — mass-conserving.
  //
  // Coagulation is a continuous process (many grain-grain collisions per
  // timestep in dense gas), so unlike shattering we use the deterministic
  // exponential approach. The superparticle represents N grains of radius a;
  // after coagulation it represents N/growth_factor^3 grains of radius a_new.
  // Total dust mass is identical — only the representative radius changes.
  // No call to setMass().
  //
  // The radius growth factor derives from: if a fraction f of grain volume
  // is swept up in time dt, the new radius is a*(1 + f)^(1/3).
  // Here f = 1 - exp(-dt/tau_coag) is the fractional volume gained.
  // Cap at 1.2× per call to prevent unphysical jumps when dt >> tau_coag.
  // -----------------------------------------------------------------------
  double swept_fraction = 1.0 - exp(-dt / tau_coag);
  double size_ratio     = pow(1.0 + swept_fraction, 1.0 / 3.0);
  size_ratio            = std::min(size_ratio, 1.2);  // cap: at most 20% radius growth per call

  double a_new = a * size_ratio;
  if(a_new > All.DustCoagulationMaxSize) a_new = All.DustCoagulationMaxSize;
  if(!isfinite(a_new) || a_new <= a) return;

  // Mass is conserved — only radius changes
  Sp->DustP[dust_idx].GrainRadius = a_new;

  NCoagulationEvents++;

  if(All.ThisTask == 0 && (NCoagulationEvents <= 100 || NCoagulationEvents % 10000 == 0))
    DUST_PRINT("[COAGULATION] Event #%lld: a=%.1f→%.1f nm  M=%.3e Msun (conserved)  "
               "n_H=%.2f n_eff=%.2f (C=%.0f) cm^-3  T=%.0f K  "
               "tau=%.1f Myr  swept_f=%.3e\n",
               NCoagulationEvents, a, a_new, M_dust,
               n_H, n_eff, DustClumpingFactor,
               T_gas, tau_coag_yr / 1e6, swept_fraction);
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

    Sp->P[nearest_dust].setMass(DUST_MASS_TO_DESTROY);
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

  // Use clumping-boosted n_eff directly for growth timescale.
  // Floor is a numerical safety net only — set far below any physical value
  // so clumping can meaningfully accelerate growth in dense environments.
  double n_eff_for_growth = n_eff_cm3;
  if(n_eff_for_growth < 1e-3) n_eff_for_growth = 1e-3;

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