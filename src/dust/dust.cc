/*! \file dust.cc
 *  \brief Implementation of on-the-fly dust evolution model for CosmicGrain.
 *
 *  This module creates dust superparticles from stellar feedback and evolves
 *  them through gas drag, grain growth, thermal erosion, shock destruction,
 *  coagulation, shattering, radiation pressure, and dust temperature updates.
 *
 *  Each dust "superparticle" represents a population of grains of a single
 *  representative radius. Grain physics is applied per-superparticle; mass and
 *  radius are updated self-consistently (m ∝ a³).
 *
 *  ── PHYSICS REFERENCES ──────────────────────────────────────────────────────
 *
 *  DUST CREATION
 *    Todini & Ferrara 2001        — SN dust yields
 *    Ferrarotti & Gail 2006       — AGB dust yields
 *    Nozawa et al. 2003           — Dust condensation in SN ejecta
 *
 *  GRAIN GROWTH (accretion)
 *    Hirashita & Kuo 2011 (HK11)  — Subgrid accretion timescales
 *    Asano et al. 2013            — Accretion + coagulation framework
 *
 *  THERMAL SPUTTERING
 *    Draine & Salpeter 1979       — Thermal sputtering physics
 *    McKinnon et al. 2017         — Sputtering timescale formula (Eq. 2)
 *    Tsai & Mathews 1995          — Sputtering in hot gas
 *
 *  SHOCK DESTRUCTION
 *    McKee & Ostriker 1977        — SN energetics (10^51 erg standard)
 *    Sedov 1959                   — Self-similar blast wave solution
 *    Jones et al. 1994, 1996      — Grain shattering threshold (~50 km/s)
 *    Bocchio et al. 2014          — Grain destruction efficiencies (Table 6)
 *
 *  DRAG COUPLING
 *    McKinnon et al. 2018         — Epstein drag (eqs. 8–9)
 *
 *  GRAIN TEMPERATURE
 *    Hollenbach & McKee 1979      — Gas-grain collisional coupling
 *    Draine & Lee 1984            — Modified blackbody Q_abs opacity law
 *    Mathis et al. 1983           — ISRF parametrisation
 *    Draine & Li 2007             — Grain opacity and composition
 *
 *  RADIATION PRESSURE
 *    Draine & Lee 1984            — Q_pr radiation pressure efficiency
 *    Draine & Li 2007             — Grain opacity and composition treatment
 *
 *  GENERAL FRAMEWORK
 *    Dwek 1998                    — Dust evolution in the ISM
 *    McKinnon et al. 2016         — Dust in cosmological simulations
 *
 *  ── PERFORMANCE NOTES ───────────────────────────────────────────────────────
 *
 *  At 2048³, the dust population grows to ~15M superparticles by z~3. To
 *  avoid these dominating the gravity tree, newly spawned dust is assigned a gravity timebin of
    max(DUST_MIN_TIMEBIN, HighestActiveTimeBin), ensuring it lands
    on a bin that is synchronized with the current hierarchy.
    At late times HighestActiveTimeBin ≤ 15 so DUST_MIN_TIMEBIN
    dominates; at early times (z > 10) HighestActiveTimeBin can be
    21+, and without the clamp dust would spawn on an unsynchronized
    bin causing a collective gravity hang on multi-node runs.
 *
 *  IMPORTANT: spawn_dust_particle caps the timebin at birth, but Gadget's
 *  ongoing timestep criterion (timestep.cc: get_timestep_grav) can migrate
 *  dust back to short bins after the first gravity force evaluation.
 *  get_timestep_grav() in timestep.cc returns TIMEBASE-1 for all dust particles,
 *  causing timebins_get_bin_and_do_validity_checks to assign the highest
 *  currently-synchronized bin. Dust migrates upward naturally from DUST_MIN_TIMEBIN
 *  within a few sync-points after spawning.
 *
 *  Dust physics (drag, growth, sputtering, etc.) runs every 10 gravity steps
 *  via the cadence guard in update_dust_dynamics(), with dt scaled by 10
 *  to compensate. All routines that receive this scaled dt use the exact
 *  analytical form (1 − exp(−dt/τ)) rather than linear approximations,
 *  so the result is independent of step size as long as τ is well resolved.
 *  ─────────────────────────────────────────────────────────────────────────────
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

// ── Mass / size thresholds ────────────────────────────────────────────────────
// Minimum mass to bother creating a dust particle. Set well below any
// physically meaningful grain mass to allow tiny initial particles, but
// large enough to prevent creating enormous numbers from trace metal ejecta.
#define MIN_DUST_PARTICLE_MASS  1e-10

// Threshold below which a dust particle is considered destroyed and will be
// removed from the particle array by destroy_dust_particles(). Must be
// positive and much smaller than any real grain mass.
#define DUST_MASS_TO_DESTROY    1e-30

// Grain radius limits (nanometres). DUST_MAX_GRAIN_SIZE should match
// All.DustCoagulationMaxSize in the parameter file.
#define DUST_MIN_GRAIN_SIZE     1.0     // nm
#define DUST_MAX_GRAIN_SIZE     200.0   // nm

// ── Miscellaneous physics constants ──────────────────────────────────────────
// Minimum SFR to consider a gas cell actively star-forming for clumping / f_mol.
#define DUST_SFR_EPS            1e-14

// Informational: single-grain mass for a silicate grain at DUST_MIN_GRAIN_SIZE.
// m = (4/3)π(0.5e-7 cm)³ × 2.4 g/cm³ / 1.989e33 g/M☉
#define DUST_SINGLE_GRAIN_MASS_MSUN  6.32e-55   // M☉

// ── Diagnostic print macro ────────────────────────────────────────────────────
// Prepends task, scale factor, and redshift to every DUST_PRINT line.
// Controlled by All.DustDebugLevel so production runs stay quiet.
#define DUST_PRINT(...) do{ if(All.DustDebugLevel){ \
  printf("[DUST|T=%d|a=%.6g z=%.3f] ", All.ThisTask, (double)All.Time, 1.0/All.Time-1.0); \
  printf(__VA_ARGS__); } }while(0)

extern double get_random_number(void);

// ── Forward declarations ──────────────────────────────────────────────────────
void dust_grain_growth_subgrid(simparticles *Sp, int dust_idx, int gas_idx, double dt);
static int destroy_dust_particle_to_gas(simparticles *Sp, int dust_idx,
                                         int nearest_gas, long long *counter,
                                         double *mass_counter);
double dust_clumping_factor(double n_H, int is_star_forming);

// ── External hash and rebuild declarations (defined in feedback.cc) ───────────
extern spatial_hash_zoom gas_hash;
extern spatial_hash_zoom star_hash;
extern spatial_hash_zoom dust_hash;
extern void rebuild_feedback_spatial_hash(simparticles *Sp, double dust_search_radius, MPI_Comm comm);

// ── Module-level statistics ───────────────────────────────────────────────────
// These accumulate over the entire run on each task and are reduced to
// global values inside print_dust_statistics().
long long NDustCreated               = 0;
long long NDustCreatedBySNII         = 0;
long long NDustCreatedByAGB          = 0;
long long NDustDestroyed             = 0;
double    TotalDustMass              = 0.0;
long long LocalDustCreatedThisStep   = 0;
long long LocalDustDestroyedThisStep = 0;
double    LocalDustMassChange        = 0.0;
int       DustNeedsSynchronization   = 0;
long long GlobalDustCount            = 0;
long long NShatteringEvents          = 0;
double    TotalSizeReductionShattering = 0.0;  // Σ(a_old − a_new) across shattering events

// Hash usage
static long long HashSearches        = 0;
static long long HashSearchesFailed  = 0;

// Destruction by mechanism
long long NDustDestroyedByThermal    = 0;
long long NDustDestroyedByShock      = 0;
long long NDustDestroyedByAstration  = 0;
double    TotalDustMassAstrated      = 0.0;

// Destruction via internal / non-physics paths — ideally all zero in
// a clean run. Non-zero values flag domain-exchange bugs or numerical issues.
long long NDustDestroyedByCleanup    = 0;  // cleanup_invalid_dust_particles()
long long NDustDestroyedByCorruption = 0;  // dust_grain_growth_subgrid() corruption
long long NDustDestroyedByBadGasIndex= 0;  // hash returned a non-gas particle

// Growth / erosion
long long NGrainGrowthEvents         = 0;
long long NGrainErosionEvents        = 0;
double    TotalMassGrown             = 0.0;
double    TotalMassDestroyedByThermal= 0.0;
double    TotalMassDestroyedByShock  = 0.0;
double    TotalMassErodedByThermal   = 0.0;
double    TotalMassErodedByShock     = 0.0;

// Coagulation
long long NCoagulationEvents         = 0;


// ═══════════════════════════════════════════════════════════════════════════════
// cleanup_invalid_dust_particles
// ═══════════════════════════════════════════════════════════════════════════════
/**
 * Scan all dust particles and remove any with unphysical state.
 *
 * Two known sources of corruption:
 *  1. DOMAIN_EXCHANGE VICTIM — GrainRadius, CarbonFraction, and DustTemperature
 *     are all exactly 0.0 (Gadget zeroed the DustP[] slot with memset during
 *     domain exchange but the P[] slot survived). The particle position and
 *     velocity are still intact.
 *  2. GENUINE NUMERICAL CORRUPTION — NaN, Inf, or negative values propagated
 *     from growth/drag/sputtering. Typically accompanied by bad velocity too.
 *
 * Corrupted particles are converted to type 3 with zero mass and ID so they
 * are swept up by destroy_dust_particles() on the next call.
 *
 * Prints per-reason tallies on any task that finds corrupted particles,
 * with detailed per-particle reports for the first 50 per call.
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
  int detail_count   = 0;  // per-call cap on detailed print lines

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
    if(!isfinite(mass) || mass <= 0.0)                                    { n_bad_mass++;   is_corrupt = true; }
    if(!isfinite(pos[0]) || !isfinite(pos[1]) || !isfinite(pos[2]))       { n_bad_pos++;    is_corrupt = true; }
    if(!isfinite(Sp->P[i].Vel[0]) || !isfinite(Sp->P[i].Vel[1]) ||
       !isfinite(Sp->P[i].Vel[2]))                                        { n_bad_vel++;    is_corrupt = true; }
    if(!isfinite(cf) || cf < 0.0 || cf > 1.0)                             { n_bad_carbon++; is_corrupt = true; }
    if(!isfinite(temp) || temp < 0.0)                                     { n_bad_temp++;   is_corrupt = true; }

    if(is_corrupt) {
      // Distinguish the two corruption subtypes for diagnostics.
      bool is_domain_exchange_victim = (a == 0.0 && cf == 0.0 && temp == 0.0);

      if(is_domain_exchange_victim && mass > DUST_MASS_TO_DESTROY)
        n_zero_radius_nonzero_mass++;  // DustP lost but P[] intact — exchange timing bug
      else if(is_domain_exchange_victim && mass < DUST_MASS_TO_DESTROY)
        n_fully_zeroed++;              // both arrays zeroed — catastrophic

      if(detail_count < 50) {
        printf("[CORRUPT_DETAIL|T=%d|Step=%d] idx=%d ID=%lld: "
               "a=%.3e mass=%.3e cf=%.3f T=%.1f "
               "pos=(%.2f,%.2f,%.2f) vel=(%.2f,%.2f,%.2f) %s\n",
               All.ThisTask, All.NumCurrentTiStep, i,
               (long long)Sp->P[i].ID.get(),
               a, mass, cf, temp,
               pos[0], pos[1], pos[2],
               Sp->P[i].Vel[0], Sp->P[i].Vel[1], Sp->P[i].Vel[2],
               is_domain_exchange_victim ? "[LIKELY_DOMAIN_EXCHANGE_VICTIM]"
                                         : "[GENUINE_CORRUPTION]");
        detail_count++;
      }

      log_dust_particle_event(Sp, i, -1, DUST_EVENT_CLEANUP);

      // Neutralise the particle so destroy_dust_particles() will remove it.
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


// ═══════════════════════════════════════════════════════════════════════════════
// get_temperature_from_entropy
// ═══════════════════════════════════════════════════════════════════════════════
/**
 * Convert stored entropy to temperature for a gas particle, accounting for
 * ionisation state via the electron fraction ne (from SphP[].Ne).
 *
 * Mean molecular weight: μ = (1 + 4Y) / (1 + Y + ne)
 * where Y = (1 − XH) / (4 XH) is the helium-to-hydrogen number ratio.
 */
double get_temperature_from_entropy(simparticles *Sp, int idx)
{
  double utherm = Sp->get_utherm_from_entropy(idx);
  double ne     = Sp->SphP[idx].Ne;

  double XH = HYDROGEN_MASSFRAC;
  double Y  = (1.0 - XH) / (4.0 * XH);
  double mu = (1.0 + 4.0 * Y) / (1.0 + Y + ne);

  double temp = (GAMMA - 1.0) * utherm
                * (All.UnitEnergy_in_cgs / All.UnitMass_in_g)
                / BOLTZMANN * PROTONMASS * mu;
  return temp;
}


// ═══════════════════════════════════════════════════════════════════════════════
// dust_grain_shattering
// ═══════════════════════════════════════════════════════════════════════════════
/**
 * Stochastically shatter a dust grain in warm/diffuse turbulent gas.
 *
 * Shattering occurs when the turbulent (proxy: sound speed) velocity exceeds
 * the material-dependent threshold from Jones et al. (1994):
 *   v_shatter = 1 km/s (silicate) to 2 km/s (pure carbon)
 *
 * The event probability follows a Poisson process: P = 1 − exp(−dt/τ_shatter).
 * When a shattering event occurs, the grain radius is reduced by a factor of 3
 * (consistent with MRN a^−3.5 fragment distribution giving <a> ~ a/3,
 * Jones et al. 1996). Total superparticle mass is conserved — the grain simply
 * represents a population of smaller grains after fragmentation.
 *
 * Active only in diffuse gas (n_eff < DustCollisionDensityThresh) where
 * shattering dominates over coagulation.
 */
void dust_grain_shattering(simparticles *Sp, int dust_idx, int gas_idx, double dt)
{
  // Diagnostic counters (static: accumulated over full run on this task)
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

  // ── Gas properties ────────────────────────────────────────────────────────
  double gas_density_code = Sp->SphP[gas_idx].Density * All.cf_a3inv;
  double gas_density_cgs  = gas_density_code * All.UnitDensity_in_cgs;
  double n_H              = (gas_density_cgs * HYDROGEN_MASSFRAC) / PROTONMASS;
  double T_gas            = get_temperature_from_entropy(Sp, gas_idx);

  // ── Gate 1: turbulent velocity threshold ─────────────────────────────────
  // Use sound speed as a proxy for turbulent velocity — conservative but
  // reasonable in the warm diffuse ISM where shattering is active.
  // Threshold: Jones et al. (1994) — 1 km/s for silicates, 2 km/s for carbon.
  double cs_cgs        = sqrt(BOLTZMANN * T_gas / (0.6 * PROTONMASS));
  double v_turb_kms    = cs_cgs / 1e5;
  double CF            = Sp->DustP[dust_idx].CarbonFraction;
  double v_shatter_kms = 1.0 + CF;  // linear blend: 1 km/s silicate → 2 km/s carbon

  if(v_turb_kms < v_shatter_kms) { shat_failed_vel++; return; }

  // ── Gate 2: density regime ────────────────────────────────────────────────
  // Shattering operates in warm/diffuse gas; dense gas is dominated by
  // coagulation. Gate on clumping-factor-weighted n_eff for resolution
  // independence — the same physical threshold applies at all resolutions.
  double DustClumpingFactor = dust_clumping_factor(n_H, Sp->SphP[gas_idx].Sfr > DUST_SFR_EPS);
  double n_eff              = n_H * DustClumpingFactor;

  if(n_eff > All.DustCollisionDensityThresh) { shat_failed_dens++; return; }

  // ── Gate 3: grain validity ────────────────────────────────────────────────
  double a      = Sp->DustP[dust_idx].GrainRadius;
  double M_dust = Sp->P[dust_idx].getMass();

  if(a <= DUST_MIN_GRAIN_SIZE || M_dust <= 0.0 || !isfinite(a))
    { shat_failed_size++; return; }

  // ── Shattering timescale ──────────────────────────────────────────────────
  // Hirashita & Kuo (2011), eq. 9. Scales with grain size (larger grains have
  // more cross-section) and inversely with dust-to-gas ratio and velocity excess.
  double dust_to_gas     = M_dust / Sp->P[gas_idx].getMass();
  double v_excess        = v_turb_kms - v_shatter_kms;
  double velocity_factor = 1.0 + v_excess / v_shatter_kms;
  double a_micron        = a / 1000.0;  // nm → µm

  double tau_shat_yr = 1e8 * (1.0 / dust_to_gas) * (a_micron / 0.1) / velocity_factor;
  tau_shat_yr       *= All.DustShatteringCalibration;
  tau_shat_yr        = std::max(tau_shat_yr, 1e6);   // 1 Myr floor
  tau_shat_yr        = std::min(tau_shat_yr, 1e10);  // 10 Gyr ceiling

  double tau_shat = tau_shat_yr * SEC_PER_YEAR / All.UnitTime_in_s;

  // ── Stochastic shattering event ───────────────────────────────────────────
  // Shattering is catastrophic, not continuous: a grain either suffers a
  // high-velocity collision this timestep or it does not.
  // Poisson probability of at least one event in dt: P = 1 − exp(−dt/τ).
  double P_shatter = 1.0 - exp(-dt / tau_shat);

  if(get_random_number() >= P_shatter) return;  // no event this timestep

  // Fragment radius: MRN a^−3.5 distribution gives mean fragment radius ~a/3
  // (Jones et al. 1996). Total superparticle mass is conserved.
  double a_new = a * 0.33;
  if(a_new < DUST_MIN_GRAIN_SIZE) {
    // Grain shattered below the physical minimum size — the fragment population
    // is too small to survive and dissolves into the gas phase. Return the full
    // superparticle mass to the nearest gas cell as metals. Without this destruction
    // path, heavily eroded grains accumulate at just above DUST_MIN_GRAIN_SIZE
    // and never escape
    log_dust_particle_event(Sp, dust_idx, gas_idx, DUST_EVENT_SHATTERING);
    destroy_dust_particle_to_gas(Sp, dust_idx, gas_idx,
                                &NDustDestroyedByShock,
                                &TotalMassDestroyedByShock);
    return;
  }
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


// ═══════════════════════════════════════════════════════════════════════════════
// erode_dust_grain_thermal
// ═══════════════════════════════════════════════════════════════════════════════
/**
 * Erode or destroy a dust grain via thermal sputtering.
 *
 * Uses the McKinnon et al. (2017) sputtering timescale (Eq. 2):
 *   τ_sp = 0.17 Gyr × (a_-1 / ρ_-27) × [(T0/T)^ω + 1]
 * where a_-1 = a / 0.1 µm, ρ_-27 = ρ_gas / (1e-27 g/cm³), T0 = 2e6 K, ω = 2.5.
 *
 * Composition correction: carbonaceous grains sputter ~1.5× faster than
 * silicates due to lower binding energy (4 eV vs 6 eV).
 *
 * IMPORTANT — stochastic form: the grain radius change is computed via the
 * exact exponential form da/a = 1 − exp(−dt/τ), NOT a linear approximation.
 * This is essential because update_dust_dynamics passes dt × 10 (cadence
 * guard), and the linear form da = (a/τ) × dt would give unphysical results
 * when dt approaches τ. The stochastic treatment is self-consistent for any
 * ratio of dt/τ.
 *
 * Returns 1 if the particle was fully destroyed, 0 otherwise.
 */
int erode_dust_grain_thermal(simparticles *Sp, int dust_idx, int nearest_gas_input,
                              double T_gas, double dt)
{
  if(!All.DustEnableSputtering) return 0;

  double a = Sp->DustP[dust_idx].GrainRadius;

  // ── Sublimation: grain temperature exceeds material melting point ─────────
  // Silicates sublimate at ~1550 K; pure carbon grains at ~2000 K.
  // Treat as instantaneous once T_dust > T_sublimate.
  double T_sublimate = 1500.0 + 500.0 * Sp->DustP[dust_idx].CarbonFraction;
  if(Sp->DustP[dust_idx].DustTemperature > T_sublimate) {
    DUST_PRINT("[SUBLIMATION] Dust destroyed at T_dust=%.0f K\n",
               Sp->DustP[dust_idx].DustTemperature);
    log_dust_particle_event(Sp, dust_idx, nearest_gas_input, DUST_EVENT_SUBLIMATION);
    return destroy_dust_particle_to_gas(Sp, dust_idx, nearest_gas_input,
                                        &NDustDestroyedByThermal,
                                        &TotalMassDestroyedByThermal);
  }

  // ── Below sputtering threshold: no action ────────────────────────────────
  if(T_gas < All.DustThermalSputteringTemp) return 0;

  // ── McKinnon+2017 sputtering timescale ───────────────────────────────────
  const double T0    = 2e6;   // K, characteristic sputtering temperature
  const double omega = 2.5;   // power-law index (Tielens et al. 1994)

  double a_cgs       = a * 1e-7;            // nm → cm
  double a_minus1    = a_cgs / 1e-5;        // units of 0.1 µm
  double rho_cgs     = Sp->SphP[nearest_gas_input].Density
                       * All.UnitDensity_in_cgs * All.cf_a3inv;
  double rho_minus27 = rho_cgs / 1e-27;

  // Guard against zero/unphysical gas density
  if(rho_minus27 <= 0.0 || !isfinite(rho_minus27))
    rho_minus27 = 1e-4 / 1e-27 * PROTONMASS / 1e-27;  // safe fallback

  double tau_sputter_yr = 0.17e9 * (a_minus1 / rho_minus27)
                          * (pow(T0 / T_gas, omega) + 1.0);

  // Composition correction: carbonaceous grains (lower binding energy) sputter faster.
  // Blend linearly between silicate (CF=0, U=6 eV) and carbon (CF=1, U=4 eV).
  double CF  = Sp->DustP[dust_idx].CarbonFraction;
  double U_eff              = (1.0 - CF) * 6.0 + CF * 4.0;
  double composition_factor = 6.0 / U_eff;  // ~1 for silicate, ~1.5 for carbon
  tau_sputter_yr /= composition_factor;

  // Clamp to physically motivated range
  if(tau_sputter_yr < 1e6) tau_sputter_yr = 1e6;   // 1 Myr floor (very hot gas)
  if(tau_sputter_yr > 1e9) tau_sputter_yr = 1e9;   // 1 Gyr ceiling (cool gas)

  double tau_sputter = tau_sputter_yr * SEC_PER_YEAR / All.UnitTime_in_s;

  // ── Grain radius change — stochastic/exact exponential form ──────────────
  // Uses 1 − exp(−dt/τ) rather than the linear approximation dt/τ.
  // This is exact for any ratio of dt/τ and avoids over-destruction when
  // the cadence guard passes dt × 10 to this routine.
  //
  // Physical picture: da/a = ∫₀^dt (1/τ) dt' = 1 − exp(−dt/τ)
  // → a_new = a × exp(−dt/τ)
  double erosion_fraction = 1.0 - exp(-dt / tau_sputter);
  double a_new = a * (1.0 - erosion_fraction);

  // Full destruction if new radius falls below minimum
  if(a_new <= DUST_MIN_GRAIN_SIZE || erosion_fraction >= 1.0) {
    log_dust_particle_event(Sp, dust_idx, nearest_gas_input, DUST_EVENT_THERMAL);
    return destroy_dust_particle_to_gas(Sp, dust_idx, nearest_gas_input,
                                        &NDustDestroyedByThermal,
                                        &TotalMassDestroyedByThermal);
  }

  if(!isfinite(a_new) || a_new <= 0.0) {
    DUST_PRINT("[BUG] Thermal erosion created invalid a_new=%.3e (a=%.3e ef=%.3e)\n",
               a_new, a, erosion_fraction);
    log_dust_particle_event(Sp, dust_idx, nearest_gas_input, DUST_EVENT_THERMAL);
    return destroy_dust_particle_to_gas(Sp, dust_idx, nearest_gas_input,
                                        &NDustDestroyedByThermal,
                                        &TotalMassDestroyedByThermal);
  }

  // ── Update grain radius and return eroded mass to gas ────────────────────
  Sp->DustP[dust_idx].GrainRadius = a_new;

  double mass_ratio = pow(a_new / a, 3.0);  // m ∝ a³
  if(!isfinite(mass_ratio) || mass_ratio < 0.0 || mass_ratio > 1.5) {
    DUST_PRINT("[ERROR] Invalid mass_ratio=%.3e in thermal erosion (a=%.3e→%.3e)\n",
               mass_ratio, a, a_new);
    return 0;
  }

  double old_mass  = Sp->P[dust_idx].getMass();
  double new_mass  = old_mass * mass_ratio;
  double mass_lost = old_mass - new_mass;

  Sp->P[dust_idx].setMass(new_mass);

  // Return sputtered mass to nearest gas as metals and update gas metallicity
  if(nearest_gas_input >= 0) {
      double gas_mass     = Sp->P[nearest_gas_input].getMass();
      double old_Z        = Sp->SphP[nearest_gas_input].Metallicity;
      double new_gas_mass = gas_mass + mass_lost;
      Sp->P[nearest_gas_input].setMass(new_gas_mass);
      double new_Z = std::min(1.0, (gas_mass * old_Z + mass_lost) / new_gas_mass);
      Sp->SphP[nearest_gas_input].Metallicity = new_Z;
      #ifdef STARFORMATION
      Sp->SphP[nearest_gas_input].MassMetallicity = new_gas_mass * new_Z;
      #endif
  }

  LocalDustMassChange -= mass_lost;
  NGrainErosionEvents++;
  TotalMassErodedByThermal += mass_lost;

  static int erosion_count = 0;
  erosion_count++;
  if(erosion_count % 10000 == 0 && All.ThisTask == 0)
    DUST_PRINT("[EROSION] Grain shrunk: %.2f → %.2f nm (dm=%.2e, T=%.0f K)\n",
               a, a_new, mass_lost, T_gas);

  return 0;
}


// ═══════════════════════════════════════════════════════════════════════════════
// destroy_dust_particle_to_gas
// ═══════════════════════════════════════════════════════════════════════════════
/**
 * Destroy a dust particle and return its mass to the nearest gas cell.
 *
 * This is the single destruction path for all physics-driven dust removal
 * (thermal sputtering, shock destruction). Astration uses a separate path
 * that does proportional mass removal rather than full destruction.
 *
 * The particle is converted to type 3 with zero mass and ID so it will be
 * swept up by destroy_dust_particles() on the next call.
 *
 * Returns 1 to signal the caller that the particle was destroyed.
 */
static int destroy_dust_particle_to_gas(simparticles *Sp, int dust_idx,
                                         int nearest_gas, long long *counter,
                                         double *mass_counter)
{
  double dust_mass = Sp->P[dust_idx].getMass();

  if(nearest_gas >= 0) {
    double gas_mass = Sp->P[nearest_gas].getMass();
    double old_Z = Sp->SphP[nearest_gas].Metallicity;
    double new_gas_mass = gas_mass + dust_mass;
    Sp->P[nearest_gas].setMass(new_gas_mass);  // Return dust mass to the gas particle
    double new_Z = std::min(1.0, (gas_mass * old_Z + dust_mass) / new_gas_mass);
    if(new_Z > 1.0) new_Z = 1.0;
    Sp->SphP[nearest_gas].Metallicity = new_Z;
    #ifdef STARFORMATION
    Sp->SphP[nearest_gas].MassMetallicity = new_gas_mass * new_Z;
    #endif
  }

  Sp->P[dust_idx].setMass(DUST_MASS_TO_DESTROY);
  Sp->P[dust_idx].setType(3);
  Sp->P[dust_idx].ID.set(0);
  memset(&Sp->DustP[dust_idx], 0, sizeof(dust_data));
  Sp->DustP[dust_idx].GrainRadius = DUST_MIN_GRAIN_SIZE;

  LocalDustMassChange -= dust_mass;
  LocalDustDestroyedThisStep++;
  if(counter)      (*counter)++;
  if(mass_counter) (*mass_counter) += dust_mass;

  return 1;
}


// ═══════════════════════════════════════════════════════════════════════════════
// calculate_sedov_velocity_from_radius
// ═══════════════════════════════════════════════════════════════════════════════
/**
 * Compute the Sedov-Taylor blast-wave velocity at a given physical radius.
 *
 * From the self-similar solution (Sedov 1959):
 *   v = (2/5) × ξ^(5/2) × √(E/ρ) × R^(−3/2)
 *
 * where ξ = 1.033 is the Sedov dimensionless constant.
 * Returns velocity in km/s.
 */
double calculate_sedov_velocity_from_radius(double radius_kpc, double rho_cgs)
{
  const double xi = 1.033;
  const double E  = 1e51;  // erg — standard SN energy

  double R_cm      = radius_kpc * 1000.0 * PARSEC;
  double xi_factor = pow(xi, 2.5);
  double v_cm_s    = (2.0/5.0) * xi_factor * sqrt(E / rho_cgs) * pow(R_cm, -1.5);

  return v_cm_s / 1e5;  // km/s
}


// ═══════════════════════════════════════════════════════════════════════════════
// erode_dust_grain_shock
// ═══════════════════════════════════════════════════════════════════════════════
/**
 * Erode or destroy a single dust grain via a Sedov-Taylor SN shock.
 *
 * Three-step process:
 *   1. Stochastic outright shattering if local_velocity > 50 km/s (Jones 1994).
 *   2. Partial erosion for survivors using Bocchio+2014 Table 6 efficiency curves
 *      blended by CarbonFraction.
 *   3. Eroded mass returned to nearest gas cell as metals.
 *
 * The local shock velocity is attenuated by distance (Sedov: v ∝ r^{−3/2})
 * when the grain lies within the physical shock radius. Grains beyond the
 * shock radius should not reach this function — the [BUG] branch will
 * Terminate() immediately to catch any upstream logic error.
 *
 * Returns 1 if the particle was destroyed, 0 otherwise.
 */
int erode_dust_grain_shock(simparticles *Sp, int dust_idx, double shock_velocity_km_s,
                           double distance_to_sn, double shock_radius,
                           int nearest_gas_hint)
{
  double a = Sp->DustP[dust_idx].GrainRadius;

  // ── Attenuate local shock velocity by distance ────────────────────────────
  // Grains at the shock front see full velocity; grains near the centre see
  // the initial, faster blast. Clamp r/R to [0,1].
  double local_velocity;
  if(distance_to_sn <= shock_radius) {
    double r_frac = (shock_radius > 0) ? (distance_to_sn / shock_radius) : 0.0;
    r_frac = std::max(0.0, std::min(1.0, r_frac));

    double velocity_attenuation = pow(1.0 - 0.7 * r_frac, 1.5);
    if(velocity_attenuation < 0.3) velocity_attenuation = 0.3;

    local_velocity = shock_velocity_km_s * velocity_attenuation;
  } else {
    // distance_to_sn > shock_radius should never happen: destroy_dust_from_sn_shocks
    // only passes grains found within effective_search_radius, which always equals
    // shock_radius (never smaller). Any caller that passes a grain beyond the shock
    // radius has a logic error that must be caught immediately.
    Terminate("[BUG] erode_dust_grain_shock: distance_to_sn=%.4f > shock_radius=%.4f kpc. "
              "Check destroy_dust_from_sn_shocks call site.\n",
              distance_to_sn, shock_radius);
  }

  // ── Size-dependent destruction factor ────────────────────────────────────
  // Smaller grains are more easily destroyed (more surface area per mass).
  double size_factor;
  if     (a < 20.0)  size_factor = 1.5;
  else if(a < 50.0)  size_factor = 1.2;
  else if(a > 100.0) size_factor = 0.7;
  else               size_factor = 1.0;

  // ── Find nearest gas once — reused for all three paths ───────────────────
  int nearest_gas = (nearest_gas_hint >= 0) ? nearest_gas_hint
                                             : find_nearest_gas_particle(Sp, dust_idx, 5.0, NULL);

  // ── Step 1: Outright shattering (stochastic, velocity-gated) ─────────────
  if(local_velocity > 50.0) {
    double velocity_factor = std::min(1.0, (local_velocity - 50.0) / 350.0);
    double destr_factor    = std::max(0.3, std::min(3.0, 50.0 / a));
    double destruction_prob = std::min(0.9, velocity_factor * destr_factor);

    if(get_random_number() < destruction_prob) {
      log_dust_particle_event(Sp, dust_idx, nearest_gas, DUST_EVENT_SHOCK);
      return destroy_dust_particle_to_gas(Sp, dust_idx, nearest_gas,
                                          &NDustDestroyedByShock,
                                          &TotalMassDestroyedByShock);
    }
  }

  // ── Step 2: Partial erosion (Bocchio+2014 Table 6) ───────────────────────
  double base_efficiency = get_shock_destruction_efficiency(local_velocity,
                                                            Sp->DustP[dust_idx].CarbonFraction);
  double erosion_fraction = std::min(0.95, base_efficiency * size_factor);

  double a_new = a * (1.0 - erosion_fraction * 0.8);
  if(a_new <= 0.0 || !isfinite(a_new)) a_new = 0.0;

  if(a_new < DUST_MIN_GRAIN_SIZE) {
    return destroy_dust_particle_to_gas(Sp, dust_idx, nearest_gas,
                                        &NDustDestroyedByShock,
                                        &TotalMassDestroyedByShock);
  }

  // ── Step 3: Grain survived — update size, return eroded mass to gas ───────
  Sp->DustP[dust_idx].GrainRadius = a_new;

  double mass_ratio = pow(a_new / a, 3.0);
  if(!isfinite(mass_ratio) || mass_ratio < 0.0 || mass_ratio > 1.5) {
    DUST_PRINT("[ERROR] Invalid mass_ratio=%.3e in shock erosion (a=%.3e→%.3e)\n",
               mass_ratio, a, a_new);
    return 0;
  }

  double old_mass  = Sp->P[dust_idx].getMass();
  double mass_lost = old_mass * (1.0 - mass_ratio);
  Sp->P[dust_idx].setMass(old_mass - mass_lost);

  if(nearest_gas >= 0) {
      double gas_mass     = Sp->P[nearest_gas].getMass();
      double old_Z        = Sp->SphP[nearest_gas].Metallicity;
      double new_gas_mass = gas_mass + mass_lost;
      Sp->P[nearest_gas].setMass(new_gas_mass);
      double new_Z = std::min(1.0, (gas_mass * old_Z + mass_lost) / new_gas_mass);
      Sp->SphP[nearest_gas].Metallicity = new_Z;
      #ifdef STARFORMATION
      Sp->SphP[nearest_gas].MassMetallicity = new_gas_mass * new_Z;
      #endif
  }

  LocalDustMassChange    -= mass_lost;
  NGrainErosionEvents++;
  TotalMassErodedByShock += mass_lost;

  return 0;
}


// ═══════════════════════════════════════════════════════════════════════════════
// calculate_drag_timescale
// ═══════════════════════════════════════════════════════════════════════════════
/**
 * Compute the Epstein drag stopping timescale for a dust grain.
 *
 * Implements McKinnon et al. (2018), equations 8–9.
 *
 * t_stop (subsonic) = √(π γ / 8) × (a ρ_grain) / (ρ_gas c_s)
 *
 * Supersonic correction: multiply by 1 / √(1 + 9π M² / 128), where M
 * is the grain–gas Mach number. Active for Mach > 0.1.
 *
 * Returns the stopping time in Myr, clamped to [0.001, 50] Myr.
 * The upper clamp prevents dust from acquiring unrealistically long stopping
 * times in the CGM that would then place it on very short gravity timebins
 * via the drag velocity kick. For the terminal-velocity correction in
 * dust_radiation_pressure, the unclamped t_stop is computed separately.
 */
static double calculate_drag_timescale(double grain_radius_nm, double grain_density,
                                       double gas_density_cgs, double gas_temperature,
                                       double relative_velocity_cgs, double mu_gas,
                                       double gamma_gas)
{
  const double k_B      = 1.38064852e-16;  // erg/K
  const double m_p      = 1.6726219e-24;   // g
  const double s_per_myr= 3.15576e13;      // s/Myr

  double a_cm = grain_radius_nm * 1e-7;
  double c_s  = sqrt(gamma_gas * k_B * gas_temperature / (mu_gas * m_p));

  // Sanity guard: cannot compute a meaningful t_stop in vacuum or near-vacuum
  if(c_s < 1e3 || gas_density_cgs < 1e-30) return 50.0;

  double t_stop_sub = (sqrt(M_PI * gamma_gas) * a_cm * grain_density)
                      / (2.0 * sqrt(2.0) * gas_density_cgs * c_s);

  double mach = relative_velocity_cgs / c_s;
  double supersonic_factor = 1.0;
  if(mach > 0.1)
    supersonic_factor = 1.0 / sqrt(1.0 + (9.0 * M_PI / 128.0) * mach * mach);

  double t_stop_myr = (t_stop_sub * supersonic_factor) / s_per_myr;

  return std::max(0.001, std::min(50.0, t_stop_myr));
}


// ═══════════════════════════════════════════════════════════════════════════════
// dust_gas_interaction
// ═══════════════════════════════════════════════════════════════════════════════
/**
 * Apply Epstein drag and thermal sputtering to a dust–gas pair.
 *
 * DRAG (Step 4): Velocity is updated analytically — the exact exponential
 * solution (Δv = v_rel × (1 − exp(−dt/t_stop))) is used rather than a linear
 * approximation, so the result is correct for any ratio of dt/t_stop including
 * the dt×10 cadence multiplier.
 *
 * SPUTTERING (Step 5): Delegated to erode_dust_grain_thermal(), which also
 * uses the exact exponential form.
 *
 * Returns 1 if the dust particle was destroyed by sputtering, 0 otherwise.
 */
int dust_gas_interaction(simparticles *Sp, int dust_idx, int nearest_gas, double dt)
{
  if(!All.DustEnableDrag) return 0;
  if(nearest_gas < 0) return 0;

  // ── Guard: stale hash returning a converted non-gas particle ─────────────
  // Reading SphP[] for a non-gas particle is undefined behaviour and can
  // produce garbage T_gas values that spuriously trigger sputtering.
  if(Sp->P[nearest_gas].getType() != 0) {
    static long long bad_gas_warns = 0;
    bad_gas_warns++;
    NDustDestroyedByBadGasIndex++;
    if(bad_gas_warns <= 50)
      printf("[BAD_GAS_IDX|T=%d|Step=%d] dust_idx=%d nearest_gas=%d "
             "has type=%d (expected 0!), mass=%.3e, ID=%lld "
             "| RunningBadCount=%lld\n",
             All.ThisTask, All.NumCurrentTiStep,
             dust_idx, nearest_gas,
             Sp->P[nearest_gas].getType(), Sp->P[nearest_gas].getMass(),
             (long long)Sp->P[nearest_gas].ID.get(), bad_gas_warns);
    return 0;
  }

  // ── Step 1–2: Gas properties ──────────────────────────────────────────────
  double gas_vel[3]     = { Sp->P[nearest_gas].Vel[0],
                             Sp->P[nearest_gas].Vel[1],
                             Sp->P[nearest_gas].Vel[2] };
  double gas_density    = Sp->SphP[nearest_gas].Density * All.cf_a3inv;
  double gas_density_cgs= gas_density * All.UnitDensity_in_cgs;
  double n_H            = gas_density_cgs / PROTONMASS;
  double T_gas          = get_temperature_from_entropy(Sp, nearest_gas);

  // ── Step 3: Relative velocity ─────────────────────────────────────────────
  double vrel_x = Sp->P[dust_idx].Vel[0] - gas_vel[0];
  double vrel_y = Sp->P[dust_idx].Vel[1] - gas_vel[1];
  double vrel_z = Sp->P[dust_idx].Vel[2] - gas_vel[2];
  double vrel   = sqrt(vrel_x*vrel_x + vrel_y*vrel_y + vrel_z*vrel_z);
  double vrel_cgs = vrel * All.UnitVelocity_in_cm_per_s;

  // ── Step 4: Epstein drag timescale ───────────────────────────────────────
  double drag_timescale_myr = calculate_drag_timescale(
      Sp->DustP[dust_idx].GrainRadius, 2.4,
      gas_density_cgs, T_gas, vrel_cgs, 0.6, 5.0/3.0);

  double drag_timescale = drag_timescale_myr * 1e6 * SEC_PER_YEAR / All.UnitTime_in_s;

  // ── Step 5: Apply drag — exact analytical solution ───────────────────────
  // Δv_k = (v_gas,k − v_dust,k) × (1 − exp(−dt/t_stop))
  // This is exact for any dt/t_stop and is safe with the ×10 cadence multiplier.
  double drag_factor = 1.0 - exp(-dt / drag_timescale);
  for(int k = 0; k < 3; k++)
    Sp->P[dust_idx].Vel[k] += drag_factor * (gas_vel[k] - Sp->P[dust_idx].Vel[k]);

  // ── Step 6: Thermal sputtering ────────────────────────────────────────────
  if(T_gas > All.DustThermalSputteringTemp) {
    int destroyed = erode_dust_grain_thermal(Sp, dust_idx, nearest_gas, T_gas, dt);
    return destroyed;
  }

  // ── Step 7: Occasional diagnostic sample (1% of particles, first 500) ────
  if(All.ThisTask == 0 && Sp->P[dust_idx].ID.get() % 100 == 0) {
    static int drag_samples = 0;
    if(drag_samples < 500) {
      double vel_kms    = sqrt(Sp->P[dust_idx].Vel[0]*Sp->P[dust_idx].Vel[0] +
                               Sp->P[dust_idx].Vel[1]*Sp->P[dust_idx].Vel[1] +
                               Sp->P[dust_idx].Vel[2]*Sp->P[dust_idx].Vel[2])
                          * All.UnitVelocity_in_cm_per_s / 1e5;
      double cs_cgs     = sqrt((5.0/3.0) * BOLTZMANN * T_gas / (0.6 * PROTONMASS));
      double mach       = vrel_cgs / cs_cgs;
      double dt_myr     = dt * All.UnitTime_in_s / (1e6 * SEC_PER_YEAR);

      DUST_PRINT("[DUST_DRAG] vel=%.1f km/s Δv=%.1f km/s "
                 "nH=%.3e cm^-3 T=%.1e K Mach=%.2f "
                 "t_drag=%.2f Myr dt=%.2f Myr f_drag=%.3f "
                 "a=%.2f nm\n",
                 vel_kms, vrel_cgs / 1e5,
                 n_H, T_gas, mach,
                 drag_timescale_myr, dt_myr, drag_factor,
                 Sp->DustP[dust_idx].GrainRadius);
      drag_samples++;
    }
  }

  return 0;
}


// ═══════════════════════════════════════════════════════════════════════════════
// create_dust_particles_from_feedback
// ═══════════════════════════════════════════════════════════════════════════════
/**
 * Spawn dust superparticles from a stellar feedback event (SNII or AGB).
 *
 * metals_produced:  total metal mass ejected by this event (M☉ code units)
 * feedback_type:    1 = SNII, 2 = AGB
 *
 * The total dust mass is metals_produced × dust_yield_fraction. This is
 * divided equally among n_dust_particles superparticles, each placed at a
 * random offset from the parent star within [DustOffsetMin, DustOffsetMax] kpc
 * and given an initial outward velocity kick. The parent gas cell's metallicity
 * is reduced by the amount locked in grains.
 */
void create_dust_particles_from_feedback(simparticles *Sp, int star_idx,
                                          double metals_produced, int feedback_type)
{
  if(!All.DustEnableCreation) return;

  double dust_yield_fraction, velocity_scale;
  if(feedback_type == 1) {
    dust_yield_fraction = All.DustYieldSNII;
    velocity_scale      = All.DustVelocitySNII;
  } else if(feedback_type == 2) {
    dust_yield_fraction = All.DustYieldAGB;
    velocity_scale      = All.DustVelocityAGB;
  } else {
    return;
  }

  double total_dust_mass = metals_produced * dust_yield_fraction;
  if(total_dust_mass < MIN_DUST_PARTICLE_MASS) return;

  int    n_dust_particles      = (feedback_type == 1) ? All.DustParticlesPerSNII
                                                       : All.DustParticlesPerAGB;
  double dust_mass_per_particle= total_dust_mass / n_dust_particles;
  if(dust_mass_per_particle < 1e-15) {
    if(All.ThisTask == 0)
      DUST_PRINT("[CREATION_SKIP] Per-particle mass %.3e below floor, skipping\n",
                 dust_mass_per_particle);
    return;
  }

  // ── Reduce parent gas mass and metallicity: metals are now locked in grains ─
  int nearest_gas = find_nearest_gas_particle(Sp, star_idx, 2.0, NULL);
  if(nearest_gas >= 0) {
      double gas_mass     = Sp->P[nearest_gas].getMass();
      double old_Z        = Sp->SphP[nearest_gas].Metallicity;
      double new_gas_mass = gas_mass - total_dust_mass;
      if(new_gas_mass <= 0.0) new_gas_mass = gas_mass * 0.01;  // safety floor
      Sp->P[nearest_gas].setMass(new_gas_mass);
      // Metal mass locked in dust is removed from gas; remaining metals stay
      double metal_mass_remaining = gas_mass * old_Z - total_dust_mass;
      double new_Z = std::max(0.0, metal_mass_remaining / new_gas_mass);
      if(new_Z > 1.0) new_Z = 1.0;
      Sp->SphP[nearest_gas].Metallicity = new_Z;
      #ifdef STARFORMATION
      Sp->SphP[nearest_gas].MassMetallicity = new_gas_mass * new_Z;
      #endif
  }

  if(feedback_type == 1) NDustCreatedBySNII += n_dust_particles;
  else                   NDustCreatedByAGB  += n_dust_particles;

  for(int n = 0; n < n_dust_particles; n++) {
    if(Sp->NumPart >= Sp->MaxPart) {
      if(All.ThisTask == 0)
        DUST_PRINT("[WARNING] Cannot create dust particle — particle array full\n");
      break;
    }

    double theta = acos(2.0 * get_random_number() - 1.0);
    double phi   = 2.0 * M_PI * get_random_number();

    double offset_min = (feedback_type == 1) ? All.DustOffsetMinSNII : All.DustOffsetMinAGB;
    double offset_max = (feedback_type == 1) ? All.DustOffsetMaxSNII : All.DustOffsetMaxAGB;
    double r = offset_min + (offset_max - offset_min) * get_random_number();

    double offset_kpc[3] = { r * sin(theta) * cos(phi),
                              r * sin(theta) * sin(phi),
                              r * cos(theta) };

    double initial_velocity[3];
    initial_velocity[0] = Sp->P[star_idx].Vel[0]
                          + velocity_scale * sin(theta)*cos(phi)
                            / All.UnitVelocity_in_cm_per_s * 1e5;
    initial_velocity[1] = Sp->P[star_idx].Vel[1]
                          + velocity_scale * sin(theta)*sin(phi)
                            / All.UnitVelocity_in_cm_per_s * 1e5;
    initial_velocity[2] = Sp->P[star_idx].Vel[2]
                          + velocity_scale * cos(theta)
                            / All.UnitVelocity_in_cm_per_s * 1e5;

    spawn_dust_particle(Sp, offset_kpc, dust_mass_per_particle,
                        initial_velocity, star_idx, feedback_type);

    int new_idx = Sp->NumPart - 1;
    if(feedback_type == 1) {
      Sp->DustP[new_idx].GrainRadius     = 10.0;
      Sp->DustP[new_idx].CarbonFraction  = 0.1;
      Sp->DustP[new_idx].GrainType       = 0;
    } else {
      Sp->DustP[new_idx].GrainRadius     = 100.0;
      Sp->DustP[new_idx].CarbonFraction  = 0.6;
      Sp->DustP[new_idx].GrainType       = 1;
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
                              Sp->P[new_idx].Vel[2]*Sp->P[new_idx].Vel[2])
                         * All.UnitVelocity_in_cm_per_s / 1e5;
        double star_vel_mag = sqrt(Sp->P[star_idx].Vel[0]*Sp->P[star_idx].Vel[0] +
                                   Sp->P[star_idx].Vel[1]*Sp->P[star_idx].Vel[1] +
                                   Sp->P[star_idx].Vel[2]*Sp->P[star_idx].Vel[2])
                              * All.UnitVelocity_in_cm_per_s / 1e5;

        double rho         = 1.0;
        double gas_vel_mag = 0.0;
        if(nearest_gas >= 0) {
          rho = Sp->SphP[nearest_gas].Density * All.cf_a3inv
                * All.UnitDensity_in_cgs / PROTONMASS;
          gas_vel_mag = sqrt(Sp->P[nearest_gas].Vel[0]*Sp->P[nearest_gas].Vel[0] +
                             Sp->P[nearest_gas].Vel[1]*Sp->P[nearest_gas].Vel[1] +
                             Sp->P[nearest_gas].Vel[2]*Sp->P[nearest_gas].Vel[2])
                        * All.UnitVelocity_in_cm_per_s / 1e5;
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


// ═══════════════════════════════════════════════════════════════════════════════
// dust_global_synchronization
// ═══════════════════════════════════════════════════════════════════════════════
/**
 * Update global dust counters and redistribute the particle ID block across tasks.
 *
 * ID BLOCK SCHEME: Between synchronisation calls, each task independently
 * increments All.MaxID as it spawns dust particles. To prevent inter-task ID
 * collisions, each task is assigned an exclusive block of size block_size
 * after every sync. block_size must exceed the maximum number of dust particles
 * any single task can create between syncs — if the overflow warning in
 * spawn_dust_particle fires, increase it.
 */
void dust_global_synchronization(simparticles *Sp, MPI_Comm Communicator,
                                  long long dust_created, long long dust_destroyed,
                                  double dust_mass_change)
{
  NDustCreated  += dust_created;
  NDustDestroyed+= dust_destroyed;
  TotalDustMass += dust_mass_change;

  LocalDustCreatedThisStep    = 0;
  LocalDustDestroyedThisStep  = 0;
  LocalDustMassChange         = 0.0;
  DustNeedsSynchronization    = 0;

  // Find the global maximum ID across all tasks
  MyIDType local_max  = All.MaxID;
  MyIDType global_max = 0;
  MPI_Allreduce(&local_max, &global_max, 1, MPI_MYIDTYPE, MPI_MAX, Communicator);

  // Assign each task a non-overlapping ID block for the next sync interval
  const MyIDType block_size = 200000;
  All.MaxID         = global_max + 1 + (MyIDType)All.ThisTask * block_size;
  All.MaxIDBlockEnd = All.MaxID + block_size - 1;
}


// ═══════════════════════════════════════════════════════════════════════════════
// spawn_dust_particle
// ═══════════════════════════════════════════════════════════════════════════════
// Spawn bin: max(DUST_MIN_TIMEBIN, HighestActiveTimeBin).
// At late times (z<5), HighestActiveTimeBin is typically ≤15 so
// DUST_MIN_TIMEBIN dominates — same behavior as before.
// At early times (z>10), HighestActiveTimeBin can be 21+, so without
// this clamp dust spawns on bin 15 which is unsynchronized in the
// current hierarchy, causing a collective gravity hang on multi-node runs.
void spawn_dust_particle(simparticles *Sp, double offset_kpc[3], double dust_mass,
                          double initial_velocity[3], int star_idx, int feedback_type)
{
  if(Sp->NumPart >= Sp->MaxPart) {
    static int warning_count = 0;
    if(warning_count < 10 && All.ThisTask == 0)
      printf("[DUST_ERROR] T=%d: Cannot create dust — array full (NumPart=%d MaxPart=%d)\n",
             All.ThisTask, Sp->NumPart, Sp->MaxPart);
    warning_count++;
    return;
  }

  int new_idx = Sp->NumPart;

  // ── Position: parent star + random offset ─────────────────────────────────
  double star_pos[3];
  Sp->intpos_to_pos(Sp->P[star_idx].IntPos, star_pos);

  // Store birth position (= star position, ignoring offset) for tracking
  // dust transport relative to its progenitor's location.
  Sp->DustP[new_idx].BirthPos[0] = star_pos[0];
  Sp->DustP[new_idx].BirthPos[1] = star_pos[1];
  Sp->DustP[new_idx].BirthPos[2] = star_pos[2];

  double dust_pos[3];
  for(int d = 0; d < 3; d++) {
    dust_pos[d] = star_pos[d] + offset_kpc[d];
    // Periodic wrap
    while(dust_pos[d] <  0.0)          dust_pos[d] += All.BoxSize;
    while(dust_pos[d] >= All.BoxSize)  dust_pos[d] -= All.BoxSize;
  }
  Sp->pos_to_intpos(dust_pos, Sp->P[new_idx].IntPos);

  // Small integer jitter prevents tree crashes when multiple dust particles
  // spawn at nearly identical IntPos from the same star event. Sub-resolution
  // jitter has no physical effect on dynamics.
  Sp->P[new_idx].IntPos[0] += (MySignedIntPosType)((int)(get_random_number() * 4) - 2);
  Sp->P[new_idx].IntPos[1] += (MySignedIntPosType)((int)(get_random_number() * 4) - 2);
  Sp->P[new_idx].IntPos[2] += (MySignedIntPosType)((int)(get_random_number() * 4) - 2);

  // ── Particle properties ───────────────────────────────────────────────────
  Sp->P[new_idx].setType(DUST_PARTICLE_TYPE);
  Sp->P[new_idx].setMass(dust_mass);
  Sp->P[new_idx].Metallicity = 1.0;  // dust is pure metal

  Sp->P[new_idx].Vel[0] = initial_velocity[0];
  Sp->P[new_idx].Vel[1] = initial_velocity[1];
  Sp->P[new_idx].Vel[2] = initial_velocity[2];

  // ── ID assignment from per-task block ─────────────────────────────────────
  if(All.MaxIDBlockEnd > 0 && All.MaxID >= All.MaxIDBlockEnd)
    printf("[DUST_ID_OVERFLOW|T=%d|Step=%d] ID block exhausted! "
           "MaxID=%lld BlockEnd=%lld — increase block_size in dust_global_synchronization\n",
           All.ThisTask, All.NumCurrentTiStep,
           (long long)All.MaxID, (long long)All.MaxIDBlockEnd);

  Sp->P[new_idx].ID.set(All.MaxID + 1);
  All.MaxID++;

  // ── Softening class: use dust type if defined, else inherit from stars ────
  if(DUST_PARTICLE_TYPE < NTYPES && All.SofteningClassOfPartType[DUST_PARTICLE_TYPE] >= 0)
    Sp->P[new_idx].setSofteningClass(All.SofteningClassOfPartType[DUST_PARTICLE_TYPE]);
  else
    Sp->P[new_idx].setSofteningClass(All.SofteningClassOfPartType[4]);

  // ── Placeholders — overwritten in create_dust_particles_from_feedback ─────
  Sp->DustP[new_idx].GrainRadius    = 10.0;
  Sp->DustP[new_idx].CarbonFraction = 0.3;
  Sp->DustP[new_idx].GrainType      = 2;

  // Initial dust temperature: CMB floor (T_CMB ∝ (1+z))
  Sp->DustP[new_idx].DustTemperature = 2.7 / All.Time;

  Sp->P[new_idx].StellarAge  = All.Time;
  Sp->P[new_idx].Ti_Current  = All.Ti_Current;
  Sp->P[new_idx].TimeBinHydro = 0;  // dust does not participate in SPH hydro

  // Spawn at DUST_MIN_TIMEBIN floor — dust migrates to longer bins immediately
  // via get_timestep_grav returning TIMEBASE-1 (capped by Ti_nextoutput).
  // Spawning on the highest synchronized bin caused newly created dust to jump
  // over snapshot output times before get_timestep_grav could apply the cap.
  int dust_spawn_bin = std::max(DUST_MIN_TIMEBIN, (int)All.HighestActiveTimeBin);
  Sp->P[new_idx].TimeBinGrav = dust_spawn_bin;
  Sp->TimeBinsGravity.timebin_add_particle(new_idx, star_idx, dust_spawn_bin,
                                            Sp->TimeBinSynchronized[dust_spawn_bin]);

  // Sanity check — GrainRadius must be set by caller before use
  if(Sp->DustP[new_idx].GrainRadius <= 0.0 || !isfinite(Sp->DustP[new_idx].GrainRadius)) {
    DUST_PRINT("[SPAWN_BUG] GrainRadius invalid (%.3e) for new particle idx=%d — resetting\n",
               Sp->DustP[new_idx].GrainRadius, new_idx);
    Sp->DustP[new_idx].GrainRadius = 10.0;
  }

  Sp->NumPart++;
  GlobalDustCount++;
}


// ═══════════════════════════════════════════════════════════════════════════════
// analyze_grain_size_distribution
// ═══════════════════════════════════════════════════════════════════════════════
/**
 * Print a histogram of grain sizes across 6 bins (task 0 only, local particles).
 * Called periodically from update_dust_dynamics diagnostics block.
 */
void analyze_grain_size_distribution(simparticles *Sp)
{
  if(All.ThisTask != 0) return;

  const int    NBINS      = 6;
  double bin_edges[NBINS+1] = {0.0, 10.0, 50.0, 100.0, 150.0, 200.0, 500.0};
  int    bin_counts[NBINS]  = {0};
  double bin_masses[NBINS]  = {0.0};
  int    total_grains       = 0;
  double total_mass         = 0.0;

  for(int i = 0; i < Sp->NumPart; i++) {
    if(Sp->P[i].getType() != DUST_PARTICLE_TYPE ||
       Sp->P[i].getMass() <= DUST_MASS_TO_DESTROY) continue;
    double mass = Sp->P[i].getMass();
    total_grains++;
    total_mass += mass;
    for(int b = 0; b < NBINS; b++) {
      if(Sp->DustP[i].GrainRadius >= bin_edges[b] &&
         Sp->DustP[i].GrainRadius <  bin_edges[b+1]) {
        bin_counts[b]++;
        bin_masses[b] += mass;
        break;
      }
    }
  }

  if(total_grains == 0) return;

  DUST_PRINT("=== GRAIN SIZE DISTRIBUTION ===\n");
  DUST_PRINT("  Total: %d grains, %.3e Msun\n", total_grains, total_mass);
  for(int b = 0; b < NBINS; b++) {
    if(bin_counts[b] > 0)
      DUST_PRINT("  [%.0f-%.0f nm]: %d grains (%.1f%%), %.2e Msun (%.1f%%)\n",
                 bin_edges[b], bin_edges[b+1], bin_counts[b],
                 100.0*bin_counts[b]/total_grains, bin_masses[b],
                 100.0*bin_masses[b]/total_mass);
  }
  DUST_PRINT("================================\n");
}


// ═══════════════════════════════════════════════════════════════════════════════
// consume_dust_by_astration
// ═══════════════════════════════════════════════════════════════════════════════
/**
 * Remove dust mass proportional to the local dust-to-gas ratio when a star
 * forms (astration — dust is incorporated into the new star).
 *
 * The search radius is set to the gas cell radius (derived from density and
 * mass), capped at the SPH smoothing length. All dust within the radius is
 * weighted by 1/distance and consumed proportionally to match stellar_mass_formed
 * × local D/G ratio. Grains eroded below DUST_MASS_TO_DESTROY are fully removed.
 */
void consume_dust_by_astration(simparticles *Sp, int gas_idx, double stellar_mass_formed,
                                int star_idx, double hsml)
{
  if(!All.DustEnableAstration) return;

  double gas_mass  = Sp->P[gas_idx].getMass();
  double rho_code  = Sp->SphP[gas_idx].Density * All.cf_a3inv;
  double cell_radius = cbrt(3.0 * gas_mass / (4.0 * M_PI * rho_code));

  double search_radius = cell_radius;
  double max_radius    = std::max(cell_radius, (double)Sp->SphP[gas_idx].Hsml);
  if(search_radius > max_radius) search_radius = max_radius;

  const int MAX_NEIGHBORS = 100;
  int    neighbor_indices[MAX_NEIGHBORS];
  double neighbor_distances[MAX_NEIGHBORS];
  int    n_neighbors = 0;

  dust_hash.find_neighbors(Sp, gas_idx, search_radius,
                           neighbor_indices, neighbor_distances,
                           &n_neighbors, MAX_NEIGHBORS);

  if(n_neighbors == 0) return;

  // Sum total dust mass within the search volume
  double total_dust_mass = 0.0;
  for(int i = 0; i < n_neighbors; i++) {
    int di = neighbor_indices[i];
    if(Sp->P[di].getType() == DUST_PARTICLE_TYPE)
      total_dust_mass += Sp->P[di].getMass();
  }
  if(total_dust_mass < DUST_MASS_TO_DESTROY) return;

  // Dust mass to consume = stellar mass formed × local D/G
  double local_DG       = total_dust_mass / gas_mass;
  double dust_to_consume= std::min(stellar_mass_formed * local_DG, total_dust_mass);

  // Weight by 1/distance for proximity preference
  double weight_sum = 0.0;
  for(int i = 0; i < n_neighbors; i++)
    if(neighbor_distances[i] > 0)
      weight_sum += 1.0 / neighbor_distances[i];

  int    dust_consumed_count = 0;
  double dust_consumed_mass  = 0.0;

  for(int i = 0; i < n_neighbors; i++) {
    int di = neighbor_indices[i];
    if(Sp->P[di].getType() != DUST_PARTICLE_TYPE) continue;

    double weight          = (neighbor_distances[i] > 0)
                             ? (1.0 / neighbor_distances[i]) : 1.0;
    double this_fraction   = weight / weight_sum;
    double mass_loss       = dust_to_consume * this_fraction;
    double current_mass    = Sp->P[di].getMass();
    double new_mass        = current_mass - mass_loss;

    if(new_mass < DUST_MASS_TO_DESTROY) {
      log_dust_particle_event(Sp, di, -1, DUST_EVENT_ASTRATION);
      Sp->P[di].setMass(0.0);
      Sp->P[di].setType(3);
      Sp->P[di].ID.set(0);
      memset(&Sp->DustP[di], 0, sizeof(dust_data));
      dust_consumed_count++;
      dust_consumed_mass += current_mass;
    } else {
      Sp->P[di].setMass(new_mass);
      dust_consumed_mass += mass_loss;
    }
  }

  NDustDestroyedByAstration += dust_consumed_count;
  TotalDustMassAstrated     += dust_consumed_mass;

  // Transfer consumed dust mass to the new star, update mass and metallicity
  if(star_idx >= 0 && dust_consumed_mass > 0) {
    double star_mass = Sp->P[star_idx].getMass();
    if(star_mass > 0) {
      double new_star_mass = star_mass + dust_consumed_mass;
      Sp->P[star_idx].setMass(new_star_mass);
      Sp->P[star_idx].Metallicity = (star_mass * Sp->P[star_idx].Metallicity + dust_consumed_mass) / new_star_mass;
    }
  }

  // ── Diagnostics ──────────────────────────────────────────────────────────
  static int astration_count = 0;
  astration_count++;

  if(astration_count <= 20)
    DUST_PRINT("[ASTRATION_CHECK|T=%d] #%d: "
               "hsml=%.3e search_r=%.3e cell_size=%.3e "
               "SF=%.3e gas_mass=%.3e D/G=%.3e "
               "dust_to_consume=%.3e n=%d consumed=%d mass=%.3e\n",
               All.ThisTask, astration_count,
               hsml, search_radius, dust_hash.cell_size,
               stellar_mass_formed, gas_mass, local_DG,
               dust_to_consume, n_neighbors, dust_consumed_count, dust_consumed_mass);

  if(astration_count % 100 == 0)
    DUST_PRINT("[ASTRATION] Event #%d: "
               "search_r=%.2e (hsml=%.2e cell=%.2e) "
               "D/G=%.3e consumed=%d (%.2e Msun)\n",
               astration_count,
               search_radius, hsml, dust_hash.cell_size,
               local_DG, dust_consumed_count, dust_consumed_mass);

  if(dust_consumed_count > 20)
    DUST_PRINT("[ASTRATION_LARGE|T=%d|Step=%d] consumed=%d neighbors=%d "
               "search_r=%.3e hsml=%.3e cell_size=%.3e D/G=%.3e\n",
               All.ThisTask, All.NumCurrentTiStep,
               dust_consumed_count, n_neighbors,
               search_radius, hsml, dust_hash.cell_size, local_DG);
}


// ═══════════════════════════════════════════════════════════════════════════════
// radiation_pressure_efficiency (Q_pr)
// ═══════════════════════════════════════════════════════════════════════════════
/**
 * Compute the radiation pressure efficiency Q_pr for a grain of radius a_nm
 * and carbon fraction CF.
 *
 * Parametrisation: Lorentzian ramp calibrated to luminosity-weighted OB
 * population (T_eff ~ 23,000 K, peak λ ~ 125 nm → a₀ = 35 nm):
 *
 *   Q_pr = [a / (a + a₀)] × [0.8 + 0.4 × CF]
 *
 * Silicates (CF=0): species_factor = 0.8   (lower absorption cross-section)
 * Pure carbon (CF=1): species_factor = 1.2 (higher absorption)
 *
 * Reference: Draine & Lee 1984.
 */
double radiation_pressure_efficiency(double a_nm, double carbon_fraction)
{
  const double a0_nm = 35.0;  // Transition scale: grains << a0 are inefficient absorbers

  double Q_pr = a_nm / (a_nm + a0_nm);  // Lorentzian ramp to order unity

  // Linear composition blend: silicate (CF=0) → carbon (CF=1)
  double species_factor = 0.8 + 0.4 * carbon_fraction;
  Q_pr *= species_factor;

  return std::max(0.0, std::min(2.0, Q_pr));
}


// ═══════════════════════════════════════════════════════════════════════════════
// stellar_luminosity
// ═══════════════════════════════════════════════════════════════════════════════
/**
 * Estimate luminosity of a stellar particle based on its current age.
 *
 * Simplified piecewise L/M model appropriate for single-age stellar populations.
 * O/B stars dominate until ~10 Myr; by 40 Myr only A/F stars remain.
 * Particles older than 100 Myr (enforced upstream) contribute negligible UV.
 *
 * NOTE: This is a placeholder. Production-quality runs should use a
 * population synthesis table (Starburst99, BPASS, or FSPS) interpolated
 * by age and metallicity.
 */
double stellar_luminosity(simparticles *Sp, int star_idx)
{
  const double L_sun_cgs = 3.828e33;
  const double M_sun_cgs = 1.989e33;

  double M_star_sun  = Sp->P[star_idx].getMass() * All.UnitMass_in_g / M_sun_cgs;
  double age_yr      = (All.Time - Sp->P[star_idx].StellarAge)
                       * All.UnitTime_in_s / SEC_PER_YEAR;

  double L_over_M;
  if     (age_yr < 3e6)   L_over_M = 200.0;
  else if(age_yr < 10e6)  L_over_M = 200.0 - 100.0 * (age_yr - 3e6)  / 7e6;
  else if(age_yr < 40e6)  L_over_M = 100.0 -  80.0 * (age_yr - 10e6) / 30e6;
  else                    L_over_M = std::max(0.0, 20.0 - 18.0 * (age_yr - 40e6) / 60e6);

  return L_over_M * M_star_sun * L_sun_cgs;
}


// ═══════════════════════════════════════════════════════════════════════════════
// dust_radiation_pressure
// ═══════════════════════════════════════════════════════════════════════════════
/**
 * Apply radiation pressure from young nearby stars to a dust grain.
 *
 * FORCE MODEL:
 *   F_rad = (L / 4πr²c) × Q_pr × πa²
 *
 * where L is the stellar luminosity, r the star–grain distance, Q_pr the
 * radiation pressure efficiency (Draine & Lee 1984), and a the grain radius.
 * The optically thin limit is assumed between star and grain.
 *
 * TERMINAL VELOCITY CORRECTION:
 *   Grains are not in free flight — they are drag-coupled to gas. In the
 *   terminal velocity regime (t_drag << t_cross), grains drift at a velocity
 *   set by the balance of radiation pressure and drag rather than accelerating
 *   freely. The correction factor is:
 *
 *     f_coupling = (1 − exp(−dt/t_stop)) × t_stop / dt
 *
 *   → 1 when dt << t_stop (grain responds fully to the kick this step)
 *   → t_stop/dt << 1 when dt >> t_stop (terminal velocity limit, suppresses kick)
 *
 *   Note: t_stop here is computed WITHOUT the 50 Myr cap used in
 *   calculate_drag_timescale(), because the cap is a numerical safeguard for
 *   the drag velocity update and should not suppress the physical terminal
 *   velocity correction in the CGM.
 *
 * Only stars younger than 20 Myr are included (OB-dominated UV).
 */
void dust_radiation_pressure(simparticles *Sp, int dust_idx, int nearest_gas, double dt)
{
  if(!All.DustEnableRadiationPressure) return;
  if(!star_hash.is_built) return;

  // Diagnostic counters
  static long long radp_calls       = 0;
  static long long radp_with_stars  = 0;
  static long long radp_with_young  = 0;
  static double    radp_total_accel = 0.0;
  static double    radp_total_coupling = 0.0;
  radp_calls++;

  // ── Search radius ─────────────────────────────────────────────────────────
  // Use gas smoothing length as the local ISM scale (scales with resolution).
  // Floor at one star hash cell width so the search always finds neighbours.
  // Hard cap at 10 kpc beyond which the 1/r² flux is negligible.
  double search_radius = (nearest_gas >= 0)
      ? std::max(2.0 * (double)Sp->SphP[nearest_gas].Hsml, 2.0 * star_hash.cell_size)
      : 2.0 * star_hash.cell_size;
  search_radius = std::min(search_radius, 10.0);

  const int MAX_STAR_NEIGHBORS = 50;
  int    neighbor_indices[MAX_STAR_NEIGHBORS];
  double neighbor_distances[MAX_STAR_NEIGHBORS];
  int    n_neighbors = 0;

  star_hash.find_neighbors(Sp, dust_idx, search_radius,
                           neighbor_indices, neighbor_distances,
                           &n_neighbors, MAX_STAR_NEIGHBORS);

  if(n_neighbors == 0) return;
  radp_with_stars++;

  // ── Grain properties ──────────────────────────────────────────────────────
  double a_nm  = Sp->DustP[dust_idx].GrainRadius;
  double a_cm  = a_nm * 1e-7;
  double CF    = Sp->DustP[dust_idx].CarbonFraction;
  double Q_pr  = radiation_pressure_efficiency(a_nm, CF);

  const double c_cgs          = 2.998e10;   // cm/s
  const double rho_grain_cgs  = 2.4;        // g/cm³ (silicate)
  double m_single_grain_cgs   = (4.0/3.0) * M_PI * a_cm*a_cm*a_cm * rho_grain_cgs;

  // Accumulate radiation acceleration from all young neighbours
  double a_rad[3] = {0.0, 0.0, 0.0};

  for(int i = 0; i < n_neighbors; i++) {
    int si = neighbor_indices[i];
    if(Sp->P[si].getType() != 4) continue;  // only stellar particles

    // Age cut: only OB-dominated populations (< 20 Myr)
    double age_yr = (All.Time - Sp->P[si].StellarAge) * All.UnitTime_in_s / SEC_PER_YEAR;
    if(age_yr > 20e6) continue;

    double L_cgs = stellar_luminosity(Sp, si);
    if(L_cgs <= 0.0) continue;

    // Convert code distance to physical cm
    double r_code = neighbor_distances[i];
    double r_cgs  = r_code * (All.Time / All.HubbleParam) * 3.086e21;
    if(r_cgs <= 0.0) continue;

    // Vector from star to grain: nearest_image_intpos_to_pos(a, b) computes a - b,
    // so this is dust_pos - star_pos, pointing radially outward from the star.
    // Dividing by r_code (the code-unit separation) normalises to a unit vector.
    // Radiation pressure acts along this direction (photons push dust away from source)
    double dxyz[3];
    Sp->nearest_image_intpos_to_pos(Sp->P[dust_idx].IntPos, Sp->P[si].IntPos, dxyz);

    // Radiation pressure acceleration on the single grain:
    //   F = (L / 4πr²c) × Q_pr × πa²
    //   a = F / m_grain
    double flux_cgs = L_cgs / (4.0 * M_PI * r_cgs * r_cgs * c_cgs);
    double accel    = flux_cgs * Q_pr * M_PI * a_cm * a_cm
                      / m_single_grain_cgs
                      * All.DustRadiationPressureEfficiency;

    for(int k = 0; k < 3; k++)
      a_rad[k] += accel * dxyz[k] / r_code;
  }

  // ── Terminal velocity correction ──────────────────────────────────────────
  // Compute TRUE (unclamped) stopping time for the coupling factor.
  // The 50 Myr cap in calculate_drag_timescale() is a numerical safeguard for
  // the drag velocity update and must NOT be applied here.
  double t_stop_code = 0.0;
  if(nearest_gas >= 0 && Sp->P[nearest_gas].getType() == 0) {
    double rho_cgs = Sp->SphP[nearest_gas].Density * All.cf_a3inv * All.UnitDensity_in_cgs;
    double T_gas   = get_temperature_from_entropy(Sp, nearest_gas);
    if(rho_cgs > 0.0 && T_gas > 0.0) {
      double c_s_cgs  = sqrt((5.0/3.0) * BOLTZMANN * T_gas / (0.6 * PROTONMASS));
      double t_stop_s = (sqrt(M_PI * 5.0/3.0 / 8.0) * a_cm * 2.4)
                        / (rho_cgs * c_s_cgs);
      t_stop_code = t_stop_s / All.UnitTime_in_s;
    }
  }

  // f_coupling → 1 for dt << t_stop (grain accelerates freely this step)
  // f_coupling → t_stop/dt for dt >> t_stop (terminal velocity regime)
  double coupling_factor = 1.0;
  if(t_stop_code > 0.0 && dt > 0.0)
    coupling_factor = (1.0 - exp(-dt / t_stop_code)) * t_stop_code / dt;

  radp_total_coupling += coupling_factor;

  // ── Apply velocity kick with terminal velocity suppression ────────────────
  double accel_code = All.UnitVelocity_in_cm_per_s / All.UnitTime_in_s;
  for(int k = 0; k < 3; k++)
    Sp->P[dust_idx].Vel[k] += (a_rad[k] / accel_code) * dt * coupling_factor;

  // Update diagnostics
  double accel_mag = sqrt(a_rad[0]*a_rad[0] + a_rad[1]*a_rad[1] + a_rad[2]*a_rad[2]);
  if(accel_mag > 0.0) radp_with_young++;
  radp_total_accel += accel_mag;

  if(radp_calls % 500000 == 0 && All.ThisTask == 0)
    DUST_PRINT("[RADIATION_PRESSURE_RATE] calls=%lld  with_any_stars=%.1f%%  "
               "with_young_stars=%.1f%%  avg_coupling=%.3e avg_|a|=%.2e code\n",
               radp_calls,
               100.0 * radp_with_stars  / radp_calls,
               100.0 * radp_with_young  / radp_calls,
               radp_total_coupling      / radp_calls,
               radp_total_accel         / radp_calls);
}


// ═══════════════════════════════════════════════════════════════════════════════
// update_dust_temperature
// ═══════════════════════════════════════════════════════════════════════════════
/**
 * Solve for equilibrium dust temperature via modified-blackbody energy balance.
 *
 * Equilibrium condition:
 *   P_emit(T_eq) = P_CMB + P_ISRF + P_gas
 *
 * where:
 *   P_emit  = C_emit × T^(4+β)               [modified blackbody emission]
 *   P_CMB   = C_emit × T_CMB^(4+β)           [CMB photon bath]
 *   P_ISRF  = C_emit × T_ISRF^(4+β)          [interstellar radiation field]
 *   P_gas   = f_eff × 2 n_H k_B (T_gas−T_dust) α_T v_th π a²  [collisional coupling]
 *
 * Emissivity uses Draine & Lee (1984) modified blackbody: Q_abs ∝ a T^β, β=2.
 *
 * Since grain cooling times are << simulation timestep, we solve analytically
 * for T_eq and then relax T_dust → T_eq exponentially over the grain cooling
 * timescale τ_cool = C_grain × T_dust / P_emit (Hollenbach & McKee 1979).
 *
 * Back-reaction on gas: the energy transferred dust→gas (or gas→dust) is
 * applied each call, with a 20% per-call drain limiter to prevent multi-particle
 * accumulation from over-cooling a gas cell faster than the cooling solver can
 * update ne.
 *
 * References: Hollenbach & McKee 1979, Draine & Lee 1984, Mathis+1983, Draine+2007.
 */
void update_dust_temperature(simparticles *Sp, int dust_idx, int gas_idx, double dt)
{
  // If dust cooling is disabled, only enforce the CMB floor
  if(!All.DustEnableCooling) {
    double T_CMB_floor = 2.7 / All.Time;
    if(Sp->DustP[dust_idx].DustTemperature < T_CMB_floor)
      Sp->DustP[dust_idx].DustTemperature = T_CMB_floor;
    return;
  }

  // ── Constants and grain properties ───────────────────────────────────────
  const double beta      = 2.0;        // emissivity spectral index (silicate)
  const double rho_grain = 2.4;        // g/cm³, silicate density
  const double alpha_T   = 0.1;        // thermal accommodation coefficient (Burke & Hollenbach 1983)
  const double sigma_SB  = 5.6704e-5;  // erg/cm²/s/K⁴
  const double T_ref     = 100.0;      // K, Q_abs reference temperature
  const double a_ref_cm  = 1e-5;       // cm (= 0.1 µm), Q_abs reference size
  const double Q_ref     = 1.3e-4;     // Q_abs(a_ref, T_ref) from Draine & Lee 1984

  double a_nm   = Sp->DustP[dust_idx].GrainRadius;
  double a_cm   = a_nm * 1e-7;
  double CF     = Sp->DustP[dust_idx].CarbonFraction;

  // Carbonaceous grains: lower β (~1.5) and higher Q_abs normalisation (Draine 2003)
  double beta_eff = beta - 0.5 * CF;
  double Q_eff    = Q_ref * (1.0 + CF) * (a_cm / a_ref_cm);

  double T_CMB  = 2.7 / All.Time;
  double T_dust = Sp->DustP[dust_idx].DustTemperature;
  if(T_dust <= 0.0 || !isfinite(T_dust)) T_dust = T_CMB;

  double T_gas       = get_temperature_from_entropy(Sp, gas_idx);
  double rho_gas_cgs = Sp->SphP[gas_idx].Density * All.cf_a3inv * All.UnitDensity_in_cgs;
  double n_H         = rho_gas_cgs * HYDROGEN_MASSFRAC / PROTONMASS;

  // ── Emission coefficient C_emit ───────────────────────────────────────────
  double C_emit = 4.0 * M_PI * a_cm * a_cm * Q_eff * sigma_SB / pow(T_ref, beta_eff);

  // ── Heating: CMB ─────────────────────────────────────────────────────────
  double P_CMB = C_emit * pow(T_CMB, 4.0 + beta_eff);

  // ── Heating: ISRF ─────────────────────────────────────────────────────────
  // T_ISRF ~ 17 K at z=0 (Reach+1995, Mathis+1983); scale weakly with redshift
  // as more young stars heat the background field.
  double z_now  = 1.0 / All.Time - 1.0;
  double T_ISRF = 17.0 * pow(1.0 + z_now, 0.25);
  double P_ISRF = C_emit * pow(T_ISRF, 4.0 + beta_eff);

  // ── Heating/cooling: gas-grain collisional coupling ───────────────────────
  // P_coll = 2 n_H k_B (T_gas − T_dust) α_T v_th π a²
  // Effective coupling factor f_eff from Hollenbach-McKee (1979) Fig. 1:
  //   cold neutral gas (T < 10³ K):  f_eff ~ 1.0  (neutrals)
  //   warm ionised gas (10³–10⁴ K):  f_eff ~ 1–2.5 (ions begin contributing)
  //   hot gas (> 10⁴ K):             f_eff ~ 2.5  (electrons dominate)
  double f_eff;
  if     (T_gas < 1e3) f_eff = 1.0;
  else if(T_gas < 1e4) f_eff = 1.0 + 1.5 * (T_gas - 1e3) / 9e3;
  else                 f_eff = 2.5;

  double v_th  = sqrt(8.0 * BOLTZMANN * T_gas / (M_PI * PROTONMASS));
  double P_gas = f_eff * 2.0 * n_H * BOLTZMANN * (T_gas - T_dust)
                 * alpha_T * v_th * M_PI * a_cm * a_cm;
  double dt_cgs = dt * All.UnitTime_in_s;

  // ── Back-reaction: remove energy from gas ─────────────────────────────────
  // P_gas > 0 → heat flows gas→dust → gas cools.
  // Limiter: no more than 20% drain per call to prevent over-cooling faster
  // than the cooling solver can update the ionisation state.
  double m_grain_cgs = (4.0/3.0) * M_PI * a_cm*a_cm*a_cm * rho_grain;
  if(m_grain_cgs > 0.0) {
    double M_dust_cgs = Sp->P[dust_idx].getMass() * All.UnitMass_in_g;
    double N_grains   = M_dust_cgs / m_grain_cgs;
    double dE_gas_cgs = -P_gas * N_grains * dt_cgs;
    double M_gas_cgs  = Sp->P[gas_idx].getMass() * All.UnitMass_in_g;
    double du_code    = (dE_gas_cgs / M_gas_cgs)
                        / (All.UnitVelocity_in_cm_per_s * All.UnitVelocity_in_cm_per_s);
    double u_old = Sp->get_utherm_from_entropy(gas_idx);
    double u_new = u_old + du_code;

    // 5% drain limiter
    double max_drain = 0.05 * u_old;
    if((u_old - u_new) > max_drain) u_new = u_old - max_drain;

    // CMB temperature floor: gas cannot cool below the CMB 
    double u_CMB_floor = (1.5 * BOLTZMANN * T_CMB) / (0.6 * PROTONMASS)
                         / (All.UnitVelocity_in_cm_per_s * All.UnitVelocity_in_cm_per_s);
    if(u_new < u_CMB_floor) u_new = u_CMB_floor;

  if(u_new > 0.0 && isfinite(u_new) && u_new != u_old) {
        Sp->set_entropy_from_utherm(u_new, gas_idx);
        if(std::abs(u_new - u_old) > 0.30 * u_old)
          Sp->SphP[gas_idx].Ne = 0.0;
        set_thermodynamic_variables_safe(Sp, gas_idx);
      }
    }

  // ── Solve for equilibrium temperature ────────────────────────────────────
  // C_emit × T_eq^(4+β) = P_CMB + P_ISRF + P_gas
  double P_total = std::max(P_CMB + P_ISRF + P_gas, P_CMB);  // clamp to CMB floor
  double T_eq    = pow(P_total / C_emit, 1.0 / (4.0 + beta_eff));

  // ── Relax toward T_eq over the cooling timescale τ_cool ──────────────────
  // Prevents numerical oscillations when T_dust is far from T_eq.
  const double c_v = 7e6;  // erg/g/K (Draine & Li 2001, valid T > 20 K)
  double C_grain        = m_grain_cgs * c_v;
  double P_emit_current = C_emit * pow(T_dust + 1.0, 4.0 + beta_eff);
  double tau_cool       = (P_emit_current > 0.0)
                          ? (C_grain * T_dust / P_emit_current) : 1e10;

  double relax = 1.0 - exp(-dt_cgs / tau_cool);
  double T_new = T_dust + relax * (T_eq - T_dust);

  // ── Safety bounds ─────────────────────────────────────────────────────────
  if(T_new < T_CMB)                T_new = T_CMB;
  double T_sublimate_local = 1500.0 + 500.0 * CF;
  if(T_new > T_sublimate_local) T_new = T_sublimate_local;
  if(!isfinite(T_new))             T_new = T_CMB;

  Sp->DustP[dust_idx].DustTemperature = T_new;
}


// ═══════════════════════════════════════════════════════════════════════════════
// update_dust_dynamics  (main per-step entry point)
// ═══════════════════════════════════════════════════════════════════════════════
/**
 * Apply all dust physics processes to all live dust particles.
 *
 * CADENCE GUARD: Physics runs every 10 gravity steps. The actual elapsed
 * time dt is multiplied by 10 before being passed to all sub-routines. All
 * routines use exact exponential forms (1 − exp(−dt/τ)) that are valid for
 * any ratio of dt/τ, so the 10× multiplier does not introduce errors.
 *
 * PROCESS ORDER per particle:
 *   1. Dust temperature equilibration (if DustEnableCooling or DustEnableDrag)
 *   2. Gas drag + sputtering (dust_gas_interaction)
 *   3. Radiation pressure from young stars (if DustEnableRadiationPressure)
 *   4. Grain growth via accretion from gas (if within 2 kpc of gas)
 *   5. Grain coagulation in dense cold gas
 *   6. Grain shattering in turbulent diffuse gas
 */
void update_dust_dynamics(simparticles *Sp, double dt, MPI_Comm Communicator)
{

  // ── One-time flag verification (confirms parameter file was read) ─────────
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

  // ── One-time log open ─────────────────────────────────────────────────────
  static bool log_opened = false;
  if(!log_opened) {
    open_dust_particle_log(Communicator);
    log_opened = true;
  }

  // ── Cadence guard: skip 9 out of 10 steps, no MPI needed ─────────────────
  if(All.NumCurrentTiStep % 10 != 0)
    return;

  // ── Collective check: any dust exists across all tasks? ───────────────────
  long long local_count  = GlobalDustCount;
  long long global_count = 0;
  MPI_Allreduce(&local_count, &global_count, 1, MPI_LONG_LONG, MPI_SUM, Communicator);
  if(global_count == 0) return;

  // ── Timing ───────────────────────────────────────────────────────────────
  static double total_time_in_dust = 0.0;
  static int    dust_call_count    = 0;
  double t_start = MPI_Wtime();

  // ── Ensure spatial hash is built ─────────────────────────────────────────
  int need_hash_rebuild = 0;
  if(All.ThisTask == 0) need_hash_rebuild = !gas_hash.is_built;
  MPI_Bcast(&need_hash_rebuild, 1, MPI_INT, 0, Communicator);

  if(need_hash_rebuild) {
    if(All.ThisTask == 0)
      DUST_PRINT("WARNING: Hash not built, building now for dust operations\n");
    rebuild_feedback_spatial_hash(Sp, 0.1, Communicator);
  }

  // ── One-time hash verification ────────────────────────────────────────────
  static bool verified = false;
  if(!verified && All.ThisTask == 0) {
    DUST_PRINT("=== HASH VERIFICATION ===\n");
    DUST_PRINT("  Hash built:    %s\n",      gas_hash.is_built ? "YES" : "NO");
    DUST_PRINT("  Cells per dim: %d\n",      gas_hash.n_cells_per_dim);
    DUST_PRINT("  Cell size:     %.3f kpc\n",gas_hash.cell_size);
    DUST_PRINT("  Gas particles: %d\n",      gas_hash.total_particles);
    DUST_PRINT("=========================\n");
    verified = true;
  }

  // ── Main dust loop ────────────────────────────────────────────────────────
  for(int i = 0; i < Sp->NumPart; i++) {
    if(Sp->P[i].getType() != DUST_PARTICLE_TYPE) continue;
    if(Sp->P[i].getMass() <= DUST_MASS_TO_DESTROY) continue;

    double dist_kpc = -1.0;
    int nearest_gas = find_nearest_gas_particle(Sp, i, 10.0, &dist_kpc);
    if(nearest_gas < 0) continue;

    // Temperature update — needs gas properties, so runs whenever drag or
    // cooling is active (both read the gas entropy).
    if(All.DustEnableCooling || All.DustEnableDrag)
      update_dust_temperature(Sp, i, nearest_gas, dt * 10);

    // Drag coupling + sputtering (internally checks DustEnableDrag)
    dust_gas_interaction(Sp, i, nearest_gas, dt * 10);

    if(All.DustEnableRadiationPressure)
      dust_radiation_pressure(Sp, i, nearest_gas, dt * 10);

    // Growth, coagulation, shattering only when grain is within 2 kpc of gas
    if(dist_kpc <= 2.0) {
      if(All.DustEnableGrowth)
        dust_grain_growth_subgrid(Sp, i, nearest_gas, dt * 10);
      if(All.DustEnableCoagulation)
        dust_grain_coagulation(Sp, i, nearest_gas, dt * 10);
      if(All.DustEnableShattering)
        dust_grain_shattering(Sp, i, nearest_gas, dt * 10);
    }
  }

  // ── Periodic diagnostics (every 500 steps, task 0) ───────────────────────
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
      DUST_PRINT("[GAS_BUDGET] cold<10^4=%.1f%%  warm<10^5=%.1f%%  hot<10^6=%.1f%%  vhot=%.1f%%\n",
                 100*M_cold/M_tot, 100*M_warm/M_tot, 100*M_hot/M_tot, 100*M_vhot/M_tot);

      int printed = 0;
      for(int i = 0; i < Sp->NumPart && printed < 3; i++) {
        if(Sp->P[i].getType() != DUST_PARTICLE_TYPE ||
           Sp->P[i].getMass() <= DUST_MASS_TO_DESTROY) continue;
        DUST_PRINT("[DUST_SAMPLE] i=%d ID=%lld M=%.2e a=%.2f nm CF=%.2f GT=%d\n",
                   i, (long long)Sp->P[i].ID.get(), Sp->P[i].getMass(),
                   Sp->DustP[i].GrainRadius, Sp->DustP[i].CarbonFraction,
                   Sp->DustP[i].GrainType);
        printed++;
      }
    }
  }

  // ── Timing summary (every 100 calls) ─────────────────────────────────────
  double dt_dust = MPI_Wtime() - t_start;
  total_time_in_dust += dt_dust;
  dust_call_count++;
  if(dust_call_count % 100 == 0 && All.ThisTask == 0)
    printf("[DUST_TIMING] Called %d times, avg %.3f sec/call, total %.1f sec\n",
           dust_call_count, total_time_in_dust / dust_call_count, total_time_in_dust);
}


// ═══════════════════════════════════════════════════════════════════════════════
// destroy_dust_particles
// ═══════════════════════════════════════════════════════════════════════════════
/**
 * Compact the particle array by removing all dust particles marked for
 * destruction (type 3 or mass ≤ DUST_MASS_TO_DESTROY or ID = 0).
 *
 * Maintains the invariant that gas particles (type 0) occupy indices
 * [0, NumGas) and all other particles (stars, dust, etc.) follow.
 */
void destroy_dust_particles(simparticles *Sp)
{
  int dust_destroyed = 0;
  for(int i = 0; i < Sp->NumPart; i++) {
    if((Sp->P[i].getType() == DUST_PARTICLE_TYPE || Sp->P[i].getType() == 3) &&
       (Sp->P[i].getMass() <= DUST_MASS_TO_DESTROY || Sp->P[i].ID.get() == 0))
      dust_destroyed++;
  }
  if(dust_destroyed == 0) return;

  // Compact gas region [0, NumGas)
  int new_num_gas = 0;
  for(int i = 0; i < Sp->NumGas; i++) {
    bool is_dead = (Sp->P[i].getType() == DUST_PARTICLE_TYPE ||
                    Sp->P[i].getType() == 3)
                   && (Sp->P[i].getMass() <= DUST_MASS_TO_DESTROY ||
                       Sp->P[i].ID.get() == 0);
    if(!is_dead) {
      if(new_num_gas != i) {
        Sp->P[new_num_gas]    = Sp->P[i];
        Sp->SphP[new_num_gas] = Sp->SphP[i];
      }
      memset(&Sp->DustP[new_num_gas], 0, sizeof(dust_data));
      new_num_gas++;
    }
  }

  // Compact non-gas region [NumGas, NumPart)
  int new_num_part = new_num_gas;
  for(int i = Sp->NumGas; i < Sp->NumPart; i++) {
    if((Sp->P[i].getType() == DUST_PARTICLE_TYPE || Sp->P[i].getType() == 3) &&
       (Sp->P[i].getMass() <= DUST_MASS_TO_DESTROY || Sp->P[i].ID.get() == 0))
      continue;

    if(new_num_part != i) Sp->P[new_num_part] = Sp->P[i];

    if(Sp->P[i].getType() == DUST_PARTICLE_TYPE) {
      if(new_num_part != i) Sp->DustP[new_num_part] = Sp->DustP[i];
    } else {
      memset(&Sp->DustP[new_num_part], 0, sizeof(dust_data));
    }
    new_num_part++;
  }

  Sp->NumPart = new_num_part;
  Sp->NumGas  = new_num_gas;
}


// ═══════════════════════════════════════════════════════════════════════════════
// find_nearest_gas_particle
// ═══════════════════════════════════════════════════════════════════════════════
/**
 * Return the index of the nearest gas particle (type 0) within max_r_kpc,
 * using the pre-built gas_hash for O(1) lookup.
 *
 * Failure modes tracked separately for diagnostics:
 *   fail_no_neighbor  — hash returned -1 (no cell within radius)
 *   fail_too_far      — nearest found is beyond max_r_kpc
 *   fail_wrong_type   — stale hash slot contains a converted non-gas particle
 *
 * Returns -1 on failure; sets *out_dist_kpc if non-NULL.
 */
int find_nearest_gas_particle(simparticles *Sp, int dust_idx,
                               double max_r_kpc, double *out_dist_kpc)
{
  if(out_dist_kpc) *out_dist_kpc = -1.0;
  if(Sp->NumGas == 0 || max_r_kpc <= 0) return -1;

  if(!gas_hash.is_built) return -1;

  HashSearches++;

  // Failure mode counters (static: accumulated over full run on this task)
  static long long fail_no_neighbor = 0;
  static long long fail_too_far     = 0;
  static long long fail_wrong_type  = 0;
  static long long diag_total       = 0;

  double nearest_dist = -1.0;
  int    nearest      = gas_hash.find_nearest_particle(Sp, dust_idx, max_r_kpc, &nearest_dist);

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

  // Print failure breakdown every 5 million failures
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


// ═══════════════════════════════════════════════════════════════════════════════
// calculate_velocity_difference
// ═══════════════════════════════════════════════════════════════════════════════
/** Return |v_dust − v_gas| in code velocity units. */
double calculate_velocity_difference(simparticles *Sp, int dust_idx, int gas_idx)
{
  double dv[3] = { Sp->P[dust_idx].Vel[0] - Sp->P[gas_idx].Vel[0],
                   Sp->P[dust_idx].Vel[1] - Sp->P[gas_idx].Vel[1],
                   Sp->P[dust_idx].Vel[2] - Sp->P[gas_idx].Vel[2] };
  return sqrt(dv[0]*dv[0] + dv[1]*dv[1] + dv[2]*dv[2]);
}


// ═══════════════════════════════════════════════════════════════════════════════
// print_dust_statistics
// ═══════════════════════════════════════════════════════════════════════════════
/**
 * Gather and print global dust statistics from all tasks to rank 0.
 *
 * Covers: particle count and mass, grain size/temperature distributions,
 * hash search success rates, destruction pathways (physics vs. internal),
 * growth events, coagulation, shattering.
 *
 * Must be called collectively (all tasks), but only rank 0 prints.
 */
void print_dust_statistics(simparticles *Sp, MPI_Comm Communicator)
{
  // ── Per-task accumulation ─────────────────────────────────────────────────
  int    local_dust_count = 0;
  double local_dust_mass  = 0.0, local_avg_size = 0.0, local_avg_temp = 0.0;
  int    local_bins[6]    = {0};

  for(int i = 0; i < Sp->NumPart; i++) {
    if(Sp->P[i].getType() != DUST_PARTICLE_TYPE ||
       Sp->P[i].getMass() <= DUST_MASS_TO_DESTROY) continue;
    local_dust_count++;
    local_dust_mass += Sp->P[i].getMass();
    local_avg_size  += Sp->DustP[i].GrainRadius;
    local_avg_temp  += Sp->DustP[i].DustTemperature;
    double T = Sp->DustP[i].DustTemperature;
    if     (T < 10.0)   local_bins[0]++;
    else if(T < 50.0)   local_bins[1]++;
    else if(T < 100.0)  local_bins[2]++;
    else if(T < 500.0)  local_bins[3]++;
    else if(T < 1000.0) local_bins[4]++;
    else                local_bins[5]++;
  }

  // ── MPI reductions ────────────────────────────────────────────────────────
  int    global_dust_count = 0;
  double global_dust_mass  = 0.0, global_avg_size = 0.0, global_avg_temp = 0.0;
  int    global_bins[6]    = {0};

  MPI_Reduce(&local_dust_count, &global_dust_count, 1, MPI_INT,    MPI_SUM, 0, Communicator);
  MPI_Reduce(&local_dust_mass,  &global_dust_mass,  1, MPI_DOUBLE, MPI_SUM, 0, Communicator);
  MPI_Reduce(&local_avg_size,   &global_avg_size,   1, MPI_DOUBLE, MPI_SUM, 0, Communicator);
  MPI_Reduce(&local_avg_temp,   &global_avg_temp,   1, MPI_DOUBLE, MPI_SUM, 0, Communicator);
  MPI_Reduce(local_bins, global_bins, 6, MPI_INT, MPI_SUM, 0, Communicator);

  long long g_NDustCreated              = 0, g_NDustCreatedBySNII       = 0;
  long long g_NDustCreatedByAGB         = 0;
  long long g_NDustDestrByThermal       = 0, g_NDustDestrByShock        = 0;
  long long g_NDustDestrByAstration     = 0, g_NDustDestrByCleanup      = 0;
  long long g_NDustDestrByCorruption    = 0, g_NDustDestrByBadGas       = 0;
  long long g_NGrowth                   = 0, g_NErosion                 = 0;
  long long g_NCoag                     = 0;
  long long g_HashSearches              = 0, g_HashFailed               = 0;
  long long g_NShatter                  = 0;
  double g_MassGrown                    = 0.0;
  double g_MassDestrThermal             = 0.0, g_MassDestrShock         = 0.0;
  double g_MassErodThermal              = 0.0, g_MassErodShock          = 0.0;
  double g_MassAstrated                 = 0.0;
  double g_SizeReductionShattering      = 0.0;

  MPI_Reduce(&NDustCreated,              &g_NDustCreated,              1, MPI_LONG_LONG, MPI_SUM, 0, Communicator);
  MPI_Reduce(&NDustCreatedBySNII,        &g_NDustCreatedBySNII,        1, MPI_LONG_LONG, MPI_SUM, 0, Communicator);
  MPI_Reduce(&NDustCreatedByAGB,         &g_NDustCreatedByAGB,         1, MPI_LONG_LONG, MPI_SUM, 0, Communicator);
  MPI_Reduce(&NDustDestroyedByThermal,   &g_NDustDestrByThermal,       1, MPI_LONG_LONG, MPI_SUM, 0, Communicator);
  MPI_Reduce(&NDustDestroyedByShock,     &g_NDustDestrByShock,         1, MPI_LONG_LONG, MPI_SUM, 0, Communicator);
  MPI_Reduce(&NDustDestroyedByAstration, &g_NDustDestrByAstration,     1, MPI_LONG_LONG, MPI_SUM, 0, Communicator);
  MPI_Reduce(&NDustDestroyedByCleanup,   &g_NDustDestrByCleanup,       1, MPI_LONG_LONG, MPI_SUM, 0, Communicator);
  MPI_Reduce(&NDustDestroyedByCorruption,&g_NDustDestrByCorruption,    1, MPI_LONG_LONG, MPI_SUM, 0, Communicator);
  MPI_Reduce(&NDustDestroyedByBadGasIndex,&g_NDustDestrByBadGas,       1, MPI_LONG_LONG, MPI_SUM, 0, Communicator);
  MPI_Reduce(&NGrainGrowthEvents,        &g_NGrowth,                   1, MPI_LONG_LONG, MPI_SUM, 0, Communicator);
  MPI_Reduce(&NGrainErosionEvents,       &g_NErosion,                  1, MPI_LONG_LONG, MPI_SUM, 0, Communicator);
  MPI_Reduce(&NCoagulationEvents,        &g_NCoag,                     1, MPI_LONG_LONG, MPI_SUM, 0, Communicator);
  MPI_Reduce(&NShatteringEvents,         &g_NShatter,                  1, MPI_LONG_LONG, MPI_SUM, 0, Communicator);
  MPI_Reduce(&HashSearches,              &g_HashSearches,              1, MPI_LONG_LONG, MPI_SUM, 0, Communicator);
  MPI_Reduce(&HashSearchesFailed,        &g_HashFailed,                1, MPI_LONG_LONG, MPI_SUM, 0, Communicator);
  MPI_Reduce(&TotalMassGrown,            &g_MassGrown,                 1, MPI_DOUBLE,    MPI_SUM, 0, Communicator);
  MPI_Reduce(&TotalMassDestroyedByThermal,&g_MassDestrThermal,         1, MPI_DOUBLE,    MPI_SUM, 0, Communicator);
  MPI_Reduce(&TotalMassDestroyedByShock, &g_MassDestrShock,            1, MPI_DOUBLE,    MPI_SUM, 0, Communicator);
  MPI_Reduce(&TotalMassErodedByThermal,  &g_MassErodThermal,           1, MPI_DOUBLE,    MPI_SUM, 0, Communicator);
  MPI_Reduce(&TotalMassErodedByShock,    &g_MassErodShock,             1, MPI_DOUBLE,    MPI_SUM, 0, Communicator);
  MPI_Reduce(&TotalDustMassAstrated,     &g_MassAstrated,              1, MPI_DOUBLE,    MPI_SUM, 0, Communicator);
  MPI_Reduce(&TotalSizeReductionShattering,&g_SizeReductionShattering, 1, MPI_DOUBLE,    MPI_SUM, 0, Communicator);

  // Print coagulation histogram (every 500 steps, all tasks)
  print_coag_histogram(Communicator);
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
  DUST_PRINT("STATISTICS Hash searches: %lld\n", g_HashSearches);
  if(g_HashSearches > 0)
    DUST_PRINT("STATISTICS Hash success rate: %.1f%%\n",
               100.0 * (g_HashSearches - g_HashFailed) / g_HashSearches);
  if(g_HashFailed > 0)
    DUST_PRINT("STATISTICS [WARNING] Failed searches: %lld (%.1f%%)\n",
               g_HashFailed, 100.0 * g_HashFailed / g_HashSearches);
  DUST_PRINT("STATISTICS Growth events: %lld (%.2e Msun grown)\n", g_NGrowth, g_MassGrown);
  DUST_PRINT("STATISTICS Partial erosion events: %lld\n", g_NErosion);
  DUST_PRINT("========================\n");

  long long total_by_physics  = g_NDustDestrByThermal + g_NDustDestrByShock + g_NDustDestrByAstration;
  long long total_by_internal = g_NDustDestrByCleanup + g_NDustDestrByCorruption + g_NDustDestrByBadGas;
  long long total_destroyed   = total_by_physics + total_by_internal;

  DUST_PRINT("=== DESTRUCTION AUDIT (global) ===\n");
  DUST_PRINT("  --- Physics mechanisms ---\n");
  DUST_PRINT("  Thermal sputtering:     %lld  (flag=%d)\n"
             "    full destructions:    %.2e Msun\n"
             "    partial erosion:      %.2e Msun\n",
             g_NDustDestrByThermal, All.DustEnableSputtering,
             g_MassDestrThermal, g_MassErodThermal);
  DUST_PRINT("  Shock destruction:      %lld  (flag=%d)\n"
             "    full destructions:    %.2e Msun\n"
             "    partial erosion:      %.2e Msun\n",
             g_NDustDestrByShock, All.DustEnableShockDestruction,
             g_MassDestrShock, g_MassErodShock);
  DUST_PRINT("  Astration:              %lld  (flag=%d, mass=%.2e Msun)\n",
             g_NDustDestrByAstration, All.DustEnableAstration, g_MassAstrated);
  DUST_PRINT("  --- Internal paths (should be 0 in a clean run) ---\n");
  DUST_PRINT("  cleanup_invalid():      %lld  ← domain exchange corruption?\n", g_NDustDestrByCleanup);
  DUST_PRINT("  growth corruption:      %lld  ← zeroed DustP in growth loop\n", g_NDustDestrByCorruption);
  DUST_PRINT("  bad gas index (stale):  %lld  ← hash returned non-gas type\n", g_NDustDestrByBadGas);
  DUST_PRINT("  --- Totals ---\n");
  DUST_PRINT("  By physics:             %lld\n", total_by_physics);
  DUST_PRINT("  By internal paths:      %lld\n", total_by_internal);
  DUST_PRINT("  GRAND TOTAL destroyed:  %lld\n", total_destroyed);
  DUST_PRINT("  Total created:          %lld  (SNII=%lld  AGB=%lld)\n",
             g_NDustCreated, g_NDustCreatedBySNII, g_NDustCreatedByAGB);
  DUST_PRINT("  Net (created-destroyed):%lld\n", g_NDustCreated - total_destroyed);
  DUST_PRINT("  Current live particles: %d\n",  global_dust_count);
  DUST_PRINT("=======================================\n");
  if(global_dust_count > 0 && g_NDustDestrByShock > 0)
    DUST_PRINT("  Shock events / live particles: %.3f  (target < 0.5)\n",
               (double)g_NDustDestrByShock / global_dust_count);
  DUST_PRINT("STATISTICS Coagulation events: %lld\n", g_NCoag);
  DUST_PRINT("STATISTICS Shattering events: %lld (avg da=%.2f nm/event)\n",
             g_NShatter,
             g_NShatter > 0 ? g_SizeReductionShattering / g_NShatter : 0.0);
  DUST_PRINT("========================\n");
}


// ═══════════════════════════════════════════════════════════════════════════════
// analyze_dust_gas_coupling_local
// ═══════════════════════════════════════════════════════════════════════════════
/**
 * Sample the dust-gas velocity offset on task 0 and print average and max.
 *
 * Only every 100th particle is sampled to bound the cost — at 16M dust
 * particles split across ~240 tasks, task 0 holds ~67k particles, so a full
 * loop would do ~67k hash searches purely for diagnostics. Sampling 1%
 * (~670 particles) gives statistically representative output at negligible cost.
 *
 * This is a local (task 0 only) diagnostic; for a global MPI-reduced velocity
 * statistic, see the [COUPLING] line in print_dust_statistics().
 */
void analyze_dust_gas_coupling_local(simparticles *Sp)
{
  if(All.ThisTask != 0) return;

  double total_vel_diff = 0.0;
  double max_vel_diff   = 0.0;
  int    dust_count     = 0;

  for(int i = 0; i < Sp->NumPart; i++) {
    // Sample every 100th particle to keep diagnostic cost O(N/100)
    if(i % 100 != 0) continue;

    if(Sp->P[i].getType() != DUST_PARTICLE_TYPE ||
       Sp->P[i].getMass() <= DUST_MASS_TO_DESTROY) continue;

    int nearest_gas = find_nearest_gas_particle(Sp, i, 10.0, NULL);
    if(nearest_gas < 0) continue;

    double vd = calculate_velocity_difference(Sp, i, nearest_gas);
    total_vel_diff += vd;
    if(vd > max_vel_diff) max_vel_diff = vd;
    dust_count++;
  }

  if(dust_count > 0)
    DUST_PRINT("[COUPLING] %d sampled particles (1%%), avg Δv=%.3f, max=%.3f (Rank 0)\n",
               dust_count, total_vel_diff / dust_count, max_vel_diff);
}


// ═══════════════════════════════════════════════════════════════════════════════
// calculate_sn_shock_radius
// ═══════════════════════════════════════════════════════════════════════════════
/**
 * Compute the Sedov-Taylor shock radius at time_myr after the SN explosion.
 *
 * R(t) = ξ × [E t² / ρ]^(1/5)    (Sedov 1959, ξ = 1.033)
 *
 * Returns radius in kpc.
 */
double calculate_sn_shock_radius(double sn_energy_erg, double gas_density_cgs,
                                  double time_myr)
{
  const double xi     = 1.033;
  double time_sec     = time_myr * 1e6 * SEC_PER_YEAR;
  double radius_cm    = xi * pow(sn_energy_erg * time_sec * time_sec / gas_density_cgs, 0.2);
  return radius_cm / (1000.0 * PARSEC);  // cm → kpc
}


// ═══════════════════════════════════════════════════════════════════════════════
// get_shock_destruction_efficiency
// ═══════════════════════════════════════════════════════════════════════════════
/**
 * Grain destruction efficiency as a function of shock velocity and composition.
 *
 * Piecewise-linear interpolation of Bocchio et al. (2014), Table 6, blended
 * linearly between the silicate and carbonaceous curves by CarbonFraction.
 *
 * Key values:
 *   Carbonaceous at 50 km/s: 0.77  (highly vulnerable to weak shocks)
 *   Silicate at 50 km/s:     0.02  (requires strong shocks for destruction)
 *   Both converge toward 1 at v > 175 km/s.
 *
 * Note: the silicate curve is held flat at 0.67 above 200 km/s because
 * Bocchio+2014 Table 6 ends at 200 km/s.
 */
double get_shock_destruction_efficiency(double v, double carbon_fraction)
{
  // ── Carbonaceous curve ────────────────────────────────────────────────────
  double eps_carb;
  if     (v <  50.0) eps_carb = 0.0;
  else if(v <  75.0) eps_carb = 0.77 + 0.06  * (v -  50.0) / 25.0;
  else if(v < 100.0) eps_carb = 0.83 + 0.08  * (v -  75.0) / 25.0;
  else if(v < 125.0) eps_carb = 0.91 + 0.05  * (v - 100.0) / 25.0;
  else if(v < 150.0) eps_carb = 0.96 + 0.03  * (v - 125.0) / 25.0;
  else if(v < 175.0) eps_carb = 0.99 + 0.01  * (v - 150.0) / 25.0;
  else               eps_carb = 1.00;

  // ── Silicate curve ────────────────────────────────────────────────────────
  double eps_sil;
  if     (v <  50.0) eps_sil = 0.0;
  else if(v <  75.0) eps_sil = 0.02 + 0.10  * (v -  50.0) / 25.0;
  else if(v < 100.0) eps_sil = 0.12 + 0.17  * (v -  75.0) / 25.0;
  else if(v < 125.0) eps_sil = 0.29 + 0.17  * (v - 100.0) / 25.0;
  else if(v < 150.0) eps_sil = 0.46 + 0.07  * (v - 125.0) / 25.0;
  else if(v < 175.0) eps_sil = 0.53 + 0.14  * (v - 150.0) / 25.0;
  else if(v < 200.0) eps_sil = 0.67;
  else               eps_sil = 0.67;  // Table ends at 200 km/s — hold flat

  double cf = std::max(0.0, std::min(1.0, carbon_fraction));
  return cf * eps_carb + (1.0 - cf) * eps_sil;
}


// ═══════════════════════════════════════════════════════════════════════════════
// destroy_dust_from_sn_shocks
// ═══════════════════════════════════════════════════════════════════════════════
/**
 * Apply subgrid SN shock destruction to dust within the blast radius.
 *
 * MOTIVATION:
 *   At 2048³, the physical Sedov-Taylor shock radius (~20–50 pc) is smaller
 *   than the dust hash cell size (~1–7 kpc). Direct grain-by-grain destruction
 *   at the physical radius would find zero grains per event. The subgrid
 *   treatment computes the EXPECTED mass destroyed within the physical shock
 *   volume and distributes it proportionally across all dust in a wider search
 *   volume.
 *
 * ALGORITHM:
 *   1. Find local gas density → compute physical shock radius R_phys and
 *      shock velocity v_shock from Sedov-Taylor at t = 0.3 Myr.
 *   2. Set effective_search_radius = max(R_phys, 2 × dust hash cell size).
 *   3. Find all dust within effective_search_radius.
 *   4. Compute f_vol = (R_phys / R_search)³ — volume correction factor.
 *   5. M_destroy = M_local × f_vol × ε(v_shock, CF_mean) [Bocchio+2014]
 *   6. Distribute M_destroy proportionally; grains below DUST_MIN_GRAIN_SIZE
 *      are fully destroyed; survivors updated via a ∝ m^(1/3).
 *
 * RESOLUTION INDEPENDENCE:
 *   At coarse resolution, few grains are found but f_vol ~ 1 (cell ≈ shock).
 *   At fine resolution, many grains are found but f_vol << 1 (cell << shock).
 *   In both cases M_destroy converges to the same physical value.
 *
 * CRITICAL: shock velocity is always derived from R_phys, NOT from
 * effective_search_radius. Using the inflated search radius would reduce v
 * by a factor ~(R_search/R_phys)^(3/2) >> 1, dropping it below destruction
 * thresholds.
 */
void destroy_dust_from_sn_shocks(simparticles *Sp, int sn_star_idx,
                                   double sn_energy, double metals_produced,
                                   MPI_Comm Communicator)
{
  if(!All.DustEnableShockDestruction) return;

  // ── Guard: hash must be built ─────────────────────────────────────────────
  static long long sn_skipped_no_hash = 0;
  if(!dust_hash.is_built || dust_hash.total_particles == 0) {
    sn_skipped_no_hash++;
    if(sn_skipped_no_hash <= 20 && All.ThisTask == 0)
      DUST_PRINT("[SN_SKIP] Call skipped: hash built=%d total=%d\n",
                 dust_hash.is_built, dust_hash.total_particles);
    return;
  }

  // ── Step 1: Local gas density ─────────────────────────────────────────────
  double nearest_dist   = -1.0;
  int    sn_nearest_gas = gas_hash.find_nearest_particle(Sp, sn_star_idx, 50.0, &nearest_dist);

  // Guard: stale hash may return a converted non-gas particle
  if(sn_nearest_gas >= 0 && Sp->P[sn_nearest_gas].getType() != 0)
    sn_nearest_gas = -1;

  double gas_density_cgs = 1.0 * PROTONMASS;  // fallback: ~1 cm^-3
  if(sn_nearest_gas >= 0 && sn_nearest_gas < Sp->NumGas) {
    double measured = Sp->SphP[sn_nearest_gas].Density * All.cf_a3inv
                      * All.UnitDensity_in_cgs;
    if(measured > 0.01 * PROTONMASS) gas_density_cgs = measured;
  }

  // ── Step 2: Physical shock radius at characteristic time 0.3 Myr ─────────
  // DustShockAmbientDensity is a floor that makes the radius resolution-
  // independent in very dense environments (prevents R → 0 in star-forming gas).
  double rho_sedov = std::min(gas_density_cgs,
                               All.DustShockAmbientDensity * PROTONMASS);
  double physical_radius_kpc = calculate_sn_shock_radius(1e51, rho_sedov, 0.3);
  if(physical_radius_kpc < 0.001) physical_radius_kpc = 0.001;  // 1 pc floor

  // ── Step 3: Shock velocity from PHYSICAL radius ───────────────────────────
  // Must use physical radius — see header comment for why using search radius
  // would neuterise the destruction.
  double shock_velocity_km_s = calculate_sedov_velocity_from_radius(physical_radius_kpc,
                                                                      rho_sedov);

  // ── Step 4: Effective search radius for hash lookup ───────────────────────
  double effective_search_radius = physical_radius_kpc;
  if(effective_search_radius < 2.0 * dust_hash.cell_size)
    effective_search_radius = 2.0 * dust_hash.cell_size;
  if(effective_search_radius > 3.0) effective_search_radius = 3.0;

  // ── Step 5: Diagnostic logging ────────────────────────────────────────────
  if(All.DustDebugLevel > 0) {
    static int sn_call_count = 0;
    sn_call_count++;
    if((sn_call_count <= 20 || sn_call_count % 500 == 0) && All.ThisTask == 0)
      DUST_PRINT("[SN_SHOCK_DEBUG] Call #%d: "
                 "physical_r=%.3f kpc  search_r=%.3f kpc  "
                 "v_shock=%.1f km/s  rho_local=%.3e  rho_sedov=%.3e %s\n",
                 sn_call_count, physical_radius_kpc, effective_search_radius,
                 shock_velocity_km_s, gas_density_cgs, rho_sedov,
                 (rho_sedov < gas_density_cgs) ? "[DENSITY_CAPPED]" : "[LOCAL_DENSITY]");
  }

  // ── Step 6: Find dust in search volume ───────────────────────────────────
  int    neighbors[2048];
  double distances[2048];
  int    n_found = 0;

  dust_hash.find_neighbors(Sp, sn_star_idx, effective_search_radius,
                           neighbors, distances, &n_found, 2048);

  if(n_found == 0) return;

  // ── Step 7: Mass-weighted subgrid approach ────────────────────────────────
  double M_dust_local = 0.0;
  double CF_sum       = 0.0;
  int    n_dust_found = 0;

  for(int k = 0; k < n_found; k++) {
    int i = neighbors[k];
    if(Sp->P[i].getType() != DUST_PARTICLE_TYPE || Sp->P[i].getMass() <= 0.0) continue;
    double m  = Sp->P[i].getMass();
    M_dust_local += m;
    CF_sum       += Sp->DustP[i].CarbonFraction * m;  // mass-weighted
    n_dust_found++;
  }

  if(M_dust_local <= 0.0 || n_dust_found == 0) return;

  double CF_mean = CF_sum / M_dust_local;

  // Volume correction: fraction of search volume that is physically shocked
  double f_vol = std::max(0.0, std::min(1.0,
                   pow(physical_radius_kpc / effective_search_radius, 3.0)));

  double bocchio_eff = get_shock_destruction_efficiency(shock_velocity_km_s, CF_mean);
  double M_to_destroy= M_dust_local * f_vol * bocchio_eff;

  // ── Step 8: Distribute mass destruction across found grains ──────────────
  static long long sn_total_calls = 0, sn_found_dust = 0;
  sn_total_calls++;
  if(n_dust_found > 0) sn_found_dust++;

  int    dust_destroyed   = 0, dust_eroded = 0;
  double M_actually_lost  = 0.0;

  if(M_to_destroy > 0.0) {
    for(int k = 0; k < n_found; k++) {
      int i = neighbors[k];
      if(Sp->P[i].getType() != DUST_PARTICLE_TYPE) continue;
      double m = Sp->P[i].getMass();
      if(m <= 0.0) continue;

      double mass_loss = M_to_destroy * (m / M_dust_local);
      double new_mass  = m - mass_loss;
      double a_old     = Sp->DustP[i].GrainRadius;
      double a_new     = a_old * cbrt(new_mass / m);

      if(a_new < DUST_MIN_GRAIN_SIZE || new_mass <= 0.0) {
        log_dust_particle_event(Sp, i, sn_nearest_gas, DUST_EVENT_SHOCK);
        M_actually_lost += m;
        destroy_dust_particle_to_gas(Sp, i, sn_nearest_gas,
                                     &NDustDestroyedByShock, &TotalMassDestroyedByShock);
        dust_destroyed++;
      } else {
        Sp->P[i].setMass(new_mass);
        Sp->DustP[i].GrainRadius = a_new;

        if(sn_nearest_gas >= 0) {
            double cur_gas_mass = Sp->P[sn_nearest_gas].getMass();
            double cur_Z        = Sp->SphP[sn_nearest_gas].Metallicity;
            double new_gas_mass = cur_gas_mass + mass_loss;
            Sp->P[sn_nearest_gas].setMass(new_gas_mass);
            double new_Z = std::min(1.0, (cur_gas_mass * cur_Z + mass_loss) / new_gas_mass);
            Sp->SphP[sn_nearest_gas].Metallicity = new_Z;
            #ifdef STARFORMATION
            Sp->SphP[sn_nearest_gas].MassMetallicity = new_gas_mass * new_Z;
            #endif
        }

        LocalDustMassChange    -= mass_loss;
        M_actually_lost        += mass_loss;
        TotalMassErodedByShock += mass_loss;
        NGrainErosionEvents++;
        dust_eroded++;
      }
    }
  }

  // ── Diagnostics ──────────────────────────────────────────────────────────
  if(sn_total_calls % 1000 == 0 && All.ThisTask == 0)
    DUST_PRINT("[SN_RATE] %lld SN calls, %lld found dust (%.1f%%)\n",
               sn_total_calls, sn_found_dust, 100.0 * sn_found_dust / sn_total_calls);

  if(All.ThisTask == 0 && M_to_destroy > 0.0)
    DUST_PRINT("[DUST_SN] physical_r=%.4f kpc  search_r=%.3f kpc  "
               "v=%.1f km/s  f_vol=%.3e  eff=%.3f  "
               "M_local=%.3e  M_target=%.3e  M_lost=%.3e Msun  "
               "n_dust=%d  destroyed=%d  eroded=%d\n",
               physical_radius_kpc, effective_search_radius,
               shock_velocity_km_s, f_vol, bocchio_eff,
               M_dust_local, M_to_destroy, M_actually_lost,
               n_dust_found, dust_destroyed, dust_eroded);
}


// ═══════════════════════════════════════════════════════════════════════════════
// HK11 accretion timescale helper
// ═══════════════════════════════════════════════════════════════════════════════
/**
 * Return the Hirashita & Kuo (2011) grain accretion timescale in years.
 *
 * Eq. from HK11; prefactor differs by species:
 *   silicate  (species=0): pref = 6.30e7 yr
 *   carbonaceous (species=1): pref = 5.59e7 yr
 *
 * τ_acc = pref × (a/0.1 µm) × (Zsun/Z) × (1000/nH) × (50K/T)^0.5 × (0.3/S)
 *
 * Returns HUGE_VAL on invalid input to allow the caller to bail out cleanly.
 */
static inline double tau_acc_yr_HK11(double nH_cm3, double T_K,
                                      double Z_massfrac, double Zsun_massfrac,
                                      double a_cm, double S,
                                      int species)
{
  const double pref = (species == 0) ? 6.30e7 : 5.59e7;

  if(nH_cm3 <= 0 || T_K <= 0 || Z_massfrac <= 0 || Zsun_massfrac <= 0 ||
     a_cm <= 0  || S <= 0)
    return HUGE_VAL;

  const double a_um  = a_cm * 1e4;
  const double a01   = a_um  / 0.1;
  const double n3    = nH_cm3 / 1e3;
  const double T50   = T_K    / 50.0;
  const double Zrat  = Z_massfrac / Zsun_massfrac;
  const double S03   = S / 0.3;

  return pref * a01 * (1.0/Zrat) * (1.0/n3) * pow(T50, -0.5) * (1.0/S03);
}


// ═══════════════════════════════════════════════════════════════════════════════
// dust_clumping_factor
// ═══════════════════════════════════════════════════════════════════════════════
/**
 * Return the subgrid clumping factor C for gas at hydrogen number density n_H.
 *
 * The clumping factor accounts for unresolved ISM density structure that
 * enhances grain growth rates. Dense molecular gas is highly clumped (C~30);
 * diffuse gas is essentially smooth (C~1).
 *
 * The density thresholds are expressed as fractions of CritPhysDensity (the
 * star formation threshold) so they scale consistently across resolution levels.
 *
 * Values: {1.5, 3, 10, 30} are fixed regardless of resolution
 */
double dust_clumping_factor(double n_H, int is_star_forming)
{
  if(!All.DustEnableClumping) return 1.0;

  if(is_star_forming) return 30.0;

  const double n_sf = All.CritPhysDensity;
  if(n_H > 0.5  * n_sf) return 10.0;
  if(n_H > 0.2  * n_sf) return 3.0;
  if(n_H > 0.05 * n_sf) return 1.5;
  return 1.0;
}


// ═══════════════════════════════════════════════════════════════════════════════
// dust_grain_coagulation
// ═══════════════════════════════════════════════════════════════════════════════
/**
 * Grow grains via coagulation in dense cold gas.
 *
 * Grain-grain collisions in molecular clouds cause small grains to stick into
 * larger ones without consuming gas-phase metals (unlike accretion growth).
 * Active only when n_eff > DustCollisionDensityThresh AND T < 3000 K.
 *
 * Timescale:  τ_coag ≈ 10 Myr × (100/n_eff) × (0.1 µm / a)  [HK11]
 *
 * The continuous deterministic form is used (not stochastic) since many
 * grain-grain collisions occur per timestep in dense gas. Size update:
 *   a_new = a × (1 + swept_fraction)^(1/3)
 * capped at 1.2× per call to prevent unphysical jumps. Total superparticle
 * mass is conserved — only the representative radius changes.
 *
 * The 3000 K temperature ceiling is deliberately generous relative to true
 * molecular cloud temperatures (~10–50 K) to account for SPH kernel averaging
 * at finite resolution.
 */
void dust_grain_coagulation(simparticles *Sp, int dust_idx, int gas_idx, double dt)
{
  if(!All.DustEnableCoagulation) return;

  // Diagnostic counters
  static int coag_calls = 0, coag_failed_dens = 0, coag_failed_temp = 0;
  static int coag_failed_size = 0, coag_passed = 0;
  coag_calls++;

  if(coag_calls % 50000 == 0 && All.ThisTask == 0)
    DUST_PRINT("[COAG_DIAG] calls=%d  failed: dens=%d(%.1f%%) temp=%d(%.1f%%) "
               "size=%d(%.1f%%)  passed=%d(%.1f%%)\n",
               coag_calls,
               coag_failed_dens, 100.0 * coag_failed_dens / coag_calls,
               coag_failed_temp, 100.0 * coag_failed_temp / coag_calls,
               coag_failed_size, 100.0 * coag_failed_size / coag_calls,
               coag_passed,      100.0 * coag_passed      / coag_calls);

  // ── Gate 1: density ───────────────────────────────────────────────────────
  double gas_density_cgs  = Sp->SphP[gas_idx].Density * All.cf_a3inv * All.UnitDensity_in_cgs;
  double n_H              = (gas_density_cgs * HYDROGEN_MASSFRAC) / PROTONMASS;
  double DustClumpingFactor= dust_clumping_factor(n_H, Sp->SphP[gas_idx].Sfr > DUST_SFR_EPS);
  double n_eff            = n_H * DustClumpingFactor;

  if(n_eff < All.DustCollisionDensityThresh) { coag_failed_dens++; return; }

  // ── Gate 2: temperature ───────────────────────────────────
  double T_gas = get_temperature_from_entropy(Sp, gas_idx);
  if(T_gas > 3000.0) { coag_failed_temp++; return; } // in reality, 3000K is way too high for coagulation, but we allow it to account for SPH kernel averaging at finite resolution

  // ── Gate 3: grain validity and size cap ───────────────────────────────────
  double a      = Sp->DustP[dust_idx].GrainRadius;
  double M_dust = Sp->P[dust_idx].getMass();

  if(a <= 0.0 || M_dust <= 0.0 || !isfinite(a) || !isfinite(M_dust)) return;
  if(a >= All.DustCoagulationMaxSize) { coag_failed_size++; return; }

  coag_passed++;

  // ── Coagulation timescale ─────────────────────────────────────────────────
  double a_micron    = a / 1000.0;  // nm → µm
  double tau_coag_yr = 1e7 * (100.0 / n_eff) * (0.1 / a_micron);
  tau_coag_yr       *= All.DustCoagulationCalibration;
  tau_coag_yr        = std::max(tau_coag_yr, 1e6);
  tau_coag_yr        = std::min(tau_coag_yr, 1e9);

  double tau_coag = tau_coag_yr * SEC_PER_YEAR / All.UnitTime_in_s;

  // ── Size update — mass-conserving ─────────────────────────────────────────
  // swept_fraction = fraction of grain volume accreted this timestep.
  // size_ratio capped at 1.2 to prevent unphysical jumps when dt >> τ_coag.
  double swept_fraction = 1.0 - exp(-dt / tau_coag);
  double size_ratio     = std::min(1.2, pow(1.0 + swept_fraction, 1.0/3.0));

  double a_new = a * size_ratio;
  if(a_new > All.DustCoagulationMaxSize) a_new = All.DustCoagulationMaxSize;
  if(!isfinite(a_new) || a_new <= a) return;

  Sp->DustP[dust_idx].GrainRadius = a_new;  // mass conserved; only radius changes

  NCoagulationEvents++;
  record_coagulation_event(n_H, n_eff);

  if(All.ThisTask == 0 && (NCoagulationEvents <= 100 || NCoagulationEvents % 10000 == 0))
    DUST_PRINT("[COAGULATION] Event #%lld: a=%.1f→%.1f nm  M=%.3e Msun (conserved)  "
               "n_H=%.2f n_eff=%.2f (C=%.0f) cm^-3  T=%.0f K  "
               "tau=%.1f Myr  swept_f=%.3e\n",
               NCoagulationEvents, a, a_new, M_dust,
               n_H, n_eff, DustClumpingFactor,
               T_gas, tau_coag_yr / 1e6, swept_fraction);
}


// ═══════════════════════════════════════════════════════════════════════════════
// dust_grain_growth_subgrid
// ═══════════════════════════════════════════════════════════════════════════════
/**
 * Grow a dust grain by accreting gas-phase metals (Hirashita & Kuo 2011).
 *
 * The HK11 accretion timescale τ_acc depends on gas density (via n_eff),
 * temperature, metallicity, grain size, and sticking coefficient. The
 * effective molecular fraction f_mol boosts growth in denser environments
 * where molecules shorten the effective accretion length.
 *
 * Grain size update: a → a × exp(f_mol × dt / τ_acc)  [continuous form]
 *
 * D/Z cap: a redshift-dependent maximum dust-to-metal ratio prevents runaway
 * growth at early times when the ISM is metal-poor:
 *   max_DZ = 0.5 / (1 + 0.15 z)    (interpolates ~0.05 at z=6 → 0.5 at z=0)
 *
 * Metal accounting: accreted mass dm is removed from Sp->SphP[gas_idx].Metallicity.
 *
 * Multiple corruption guards detect and remove grains with unphysical state
 * from domain exchange or numerical blow-up.
 */
void dust_grain_growth_subgrid(simparticles *Sp, int dust_idx, int gas_idx, double dt)
{
  if(!All.DustEnableGrowth) return;

  // Diagnostic counters (static: accumulated over full run on this task)
  static int total_calls = 0, failed_hot = 0, failed_no_metals = 0;
  static int failed_low_density = 0, failed_low_fmol = 0;
  static int failed_no_dust = 0, failed_too_far = 0;
  static int failed_max_dz = 0, failed_bad_tau = 0, passed_all = 0;
  static int used_species_sil = 0, used_species_carb = 0;
  static int fmol_diffuse = 0, fmol_moderate = 0, fmol_dense = 0, fmol_sf = 0;
  total_calls++;

  if(total_calls % 500000 == 0 && All.ThisTask == 0) {
    DUST_PRINT("=== HK11 GROWTH DIAGNOSTICS (after %d attempts) Rank 0 ===\n", total_calls);
    DUST_PRINT("  Failed hot:          %6d (%.1f%%)\n", failed_hot,          100.0*failed_hot/total_calls);
    DUST_PRINT("  Failed no metals:    %6d (%.1f%%)\n", failed_no_metals,     100.0*failed_no_metals/total_calls);
    DUST_PRINT("  Failed low density:  %6d (%.1f%%)\n", failed_low_density,   100.0*failed_low_density/total_calls);
    DUST_PRINT("  Failed f_mol low:    %6d (%.1f%%)\n", failed_low_fmol,      100.0*failed_low_fmol/total_calls);
    DUST_PRINT("  Failed no dust:      %6d (%.1f%%)\n", failed_no_dust,       100.0*failed_no_dust/total_calls);
    DUST_PRINT("  Failed too far:      %6d (%.1f%%)\n", failed_too_far,       100.0*failed_too_far/total_calls);
    DUST_PRINT("  Failed max D/Z:      %6d (%.1f%%)\n", failed_max_dz,        100.0*failed_max_dz/total_calls);
    DUST_PRINT("  Failed bad tau:      %6d (%.1f%%)\n", failed_bad_tau,       100.0*failed_bad_tau/total_calls);
    DUST_PRINT("  PASSED:              %6d (%.1f%%)\n", passed_all,           100.0*passed_all/total_calls);
    DUST_PRINT("  Species: sil=%d  carb=%d\n", used_species_sil, used_species_carb);
    DUST_PRINT("  f_mol: diffuse=%d  moderate=%d  dense=%d  sf=%d\n",
               fmol_diffuse, fmol_moderate, fmol_dense, fmol_sf);
    DUST_PRINT("  Total mass grown: %.3e Msun  Growth events: %lld\n",
               TotalMassGrown, NGrainGrowthEvents);
    DUST_PRINT("===========================================\n");
  }

  // ── Gate 1: temperature — hot gas inhibits accretion ─────────────────────
  const double T_gas = get_temperature_from_entropy(Sp, gas_idx);
  if(T_gas > All.DustThermalSputteringTemp) { failed_hot++; return; }

  // ── Gate 2: metallicity — need metals to accrete ──────────────────────────
  const double Z_gas = Sp->SphP[gas_idx].Metallicity;
  if(Z_gas < 1e-4) { failed_no_metals++; return; }

  // ── Gate 3: density — bail early in diffuse gas ───────────────────────────
  double gas_density_cgs    = Sp->SphP[gas_idx].Density * All.cf_a3inv * All.UnitDensity_in_cgs;
  double n_H                = (gas_density_cgs * HYDROGEN_MASSFRAC) / PROTONMASS;
  double DustClumpingFactor = dust_clumping_factor(n_H, Sp->SphP[gas_idx].Sfr > DUST_SFR_EPS);
  double n_eff_cm3          = n_H * DustClumpingFactor;

  if(n_eff_cm3 < 0.1) { failed_low_density++; return; }

  // ── Gate 4: D/Z cap ───────────────────────────────────────────────────────
  // Redshift-dependent cap prevents runaway growth in low-metallicity early gas.
  double z_now          = 1.0 / All.Time - 1.0;
  double max_dust_to_metal = std::max(0.05, 0.5 / (1.0 + 0.15 * z_now));

  double M_gas_quick  = Sp->P[gas_idx].getMass();
  double M_dust_quick = Sp->P[dust_idx].getMass();
  if(M_dust_quick >= M_gas_quick * Z_gas * 0.5) { failed_max_dz++; return; }

  // ── Molecular fraction ────────────────────────────────────────────────────
  // f_mol boosts accretion in denser/colder environments where the gas is
  // molecular. Values are observationally motivated proxies at each n_eff bin.
  double f_mol = 0.05;
  #ifdef STARFORMATION
    if(Sp->SphP[gas_idx].Sfr > DUST_SFR_EPS)  { f_mol = 0.8; fmol_sf++;       }
    else if(n_eff_cm3 > 100.0)                 { f_mol = 0.5; fmol_dense++;    }
    else if(n_eff_cm3 > 10.0)                  { f_mol = 0.2; fmol_moderate++; }
    else                                        {              fmol_diffuse++;  }
  #else
    if     (n_H > 100.0) { f_mol = 0.5; fmol_dense++;    }
    else if(n_H > 10.0)  { f_mol = 0.2; fmol_moderate++; }
    else                 {              fmol_diffuse++;   }
  #endif

  // Boost f_mol in metal-rich gas (self-shielding promotes molecular phase)
  if(Z_gas > 0.01) { f_mol = std::min(1.0, f_mol * 1.5); }

  if(f_mol < 0.01) { failed_low_fmol++; return; }

  // ── Dust particle validity ────────────────────────────────────────────────
  const int nearest_dust = dust_idx;
  if(nearest_dust < 0) { failed_no_dust++; return; }

  double dxyz[3];
  Sp->nearest_image_intpos_to_pos(Sp->P[gas_idx].IntPos,
                                  Sp->P[nearest_dust].IntPos, dxyz);
  const double dist_kpc = sqrt(dxyz[0]*dxyz[0] + dxyz[1]*dxyz[1] + dxyz[2]*dxyz[2]);
  if(dist_kpc > 5.0) { failed_too_far++; return; }

  const double a      = Sp->DustP[nearest_dust].GrainRadius;
  const double M_dust = Sp->P[nearest_dust].getMass();

  // ── Corruption check ──────────────────────────────────────────────────────
  // Two known failure modes:
  //  - DOMAIN_EXCHANGE: a == 0, CF == 0, T_dust == 0 (DustP not sent in exchange)
  //  - NUMERICAL: NaN or negative values from upstream physics
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

  // ── HK11 accretion timescale ──────────────────────────────────────────────
  const double CF         = Sp->DustP[nearest_dust].CarbonFraction;
  const int    species    = (CF >= 0.5) ? 1 : 0;  // 0=silicate, 1=carbon
  if(species == 1) used_species_carb++; else used_species_sil++;

  const double T_eff_K       = 20.0;   // K — effective grain surface temperature
  const double Zsun_massfrac = 0.02;
  const double S_stick       = 0.3;    // sticking coefficient

  // Use clumping-boosted n_eff; floor is a numerical safety net only
  double n_eff_for_growth = std::max(n_eff_cm3, 1e-3);
  double a_cm      = a * 1e-7;  // nm → cm

  double tau_acc_yr = tau_acc_yr_HK11(n_eff_for_growth, T_eff_K,
                                       Z_gas, Zsun_massfrac, a_cm, S_stick, species);
  tau_acc_yr *= All.DustGrowthCalibration;

  if(!isfinite(tau_acc_yr) || tau_acc_yr <= 0.0) { failed_bad_tau++; return; }
  tau_acc_yr = std::max(1e6,  tau_acc_yr);
  tau_acc_yr = std::min(5e9,  tau_acc_yr);

  const double tau_acc_code = tau_acc_yr * SEC_PER_YEAR / All.UnitTime_in_s;

  // ── Grain size and mass update ─────────────────────────────────────────────
  double a_new = a * exp(f_mol * dt / tau_acc_code);
  double da    = a_new - a;
  if(!isfinite(da) || da <= 0.0) return;
  if(a_new > DUST_MAX_GRAIN_SIZE) { a_new = DUST_MAX_GRAIN_SIZE; da = a_new - a; }
  if(da <= 0.0 || a_new < DUST_MIN_GRAIN_SIZE || a_new > DUST_MAX_GRAIN_SIZE) return;

  // Mass change: exact form from m ∝ a³
  // dm = M_dust × ((a_new/a)³ − 1)
  // Previously used the linear approximation 3×(da/a), which overestimates
  // dm when da/a is not negligible, causing a mysterious mass leak at peak SF!
  double dm = M_dust * (pow(a_new/a, 3.0) - 1.0);
  
  if(!isfinite(dm) || dm <= 0.0) return;

  const double M_gas    = Sp->P[gas_idx].getMass();
  const double M_metals = M_gas * Z_gas;
  const double M_dust_max = M_metals * max_dust_to_metal;

  if(M_dust >= M_dust_max) { failed_max_dz++; return; }

  // Fine-grained cap: don't overshoot D/Z within this timestep
  if(M_dust + dm > M_dust_max) {
    dm = M_dust_max - M_dust;
    if(dm <= 0.0) return;
    da    = (dm / M_dust) * a / 3.0;
    a_new = a + da;
    if(a_new < DUST_MIN_GRAIN_SIZE || a_new > DUST_MAX_GRAIN_SIZE) return;
  }

  if(dm > M_metals) dm = 0.99 * M_metals;

  // Per-step growth cap: no more than 20% mass gain per call
  const double max_dm_per_step = 0.2 * M_dust;
  if(dm > max_dm_per_step) dm = max_dm_per_step;
  if(dm <= 0.0) return;

  passed_all++;

  static int dt_printed = 0;
  if(dt_printed < 10 && All.ThisTask == 0) {
    double dt_myr = dt * All.UnitTime_in_s / (1e6 * SEC_PER_YEAR);
    DUST_PRINT("[GROWTH_DEBUG] dt=%.3e code = %.3f Myr | da=%.3f nm dm=%.3e Msun\n",
               dt, dt_myr, da, dm);
    dt_printed++;
  }

  // ── Apply changes ─────────────────────────────────────────────────────────
  // Remove mass from gas particle
  double M_gas_new = M_gas - dm;
  if(M_gas_new <= 0.0) return;  // should be unreachable given upstream caps
  Sp->P[gas_idx].setMass(M_gas_new);

  // Update dust particle mass and grain radius
  Sp->P[nearest_dust].setMass(M_dust + dm);
  Sp->DustP[nearest_dust].GrainRadius = a_new;

  // Update metallicities: remove accreted metals from gas particle
  double Z_new = std::max(1e-5, (M_gas * Z_gas - dm) / M_gas_new);
  Sp->SphP[gas_idx].Metallicity = Z_new;
  #ifdef STARFORMATION
  Sp->SphP[gas_idx].MassMetallicity = M_gas_new * Z_new;
  #endif

  NGrainGrowthEvents++;
  TotalMassGrown += dm;

  static int growth_count = 0;
  growth_count++;
  if(growth_count % 10000 == 0 && All.ThisTask == 0) {
    DUST_PRINT("[HK11_GROWTH] Event #%d: species=%s CF=%.2f f_mol=%.3f "
               "n_H=%.1f → n_eff=%.1f cm^-3 (C=%.0f)\n",
               growth_count, (species==1 ? "carb" : "sil"), CF, f_mol,
               n_H, n_eff_cm3, DustClumpingFactor);
    DUST_PRINT("[HK11_GROWTH] tau_acc=%.2e yr | n_eff=%.0f cm^-3 T_eff=%.0f K Z=%.4f\n",
               tau_acc_yr, n_eff_for_growth, T_eff_K, Z_gas);
    DUST_PRINT("[HK11_GROWTH] a=%.2f→%.2f nm | dm=%.3e (M: %.3e→%.3e) | Z=%.4f→%.4f\n",
               a, a_new, dm, M_dust, M_dust + dm, Z_gas, Z_new);
  }
}

#endif /* DUST */