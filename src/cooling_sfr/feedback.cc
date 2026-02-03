/* ============================================================================
 * feedback.cc — Stellar feedback with Type II SNe and AGB winds
 *
 * Features:
 *   - Cosmology-aware stellar ages using Friedmann t(a) lookup table
 *   - ΔT-target stochastic heating (EAGLE-like)
 *   - Energy reservoir system with intelligent release threshold
 *   - SPH kernel support with 2h neighbor search via spatial binning (this really helped the speed!)
 *   - Metal enrichment from Type II SNe (3-40 Myr)
 *   - Metal enrichment from AGB stars using MESA yield tables from Huscher et al 2025 (>100 Myr)
 *   - Stochastic kernel-weighted energy/metal deposition
 *
 * Build-time toggles (set in Config.sh):
 *   FEEDBACK_LIMIT_DULOG   - Clamp |Δlog10 u| per heating event
 *   FEEDBACK_T_CAP         - Soft temperature cap for debugging
 *
 * Notes:
 *   - Stars store birth scale factor in P[i].StellarAge
 *   - Reservoir threshold tunable at line ~320 (currently 0.3*E_need)
 * ============================================================================ */

#include "gadgetconfig.h"
#ifdef FEEDBACK

#include <math.h>
#include <algorithm>
#include <vector>
#include <mpi.h>
#include <stdio.h>
#include <set>

#include "../cooling_sfr/feedback.h"
#include "../cooling_sfr/spatial_hash_zoom.h"
#include "../data/allvars.h"
#include "../data/dtypes.h"
#include "../data/mymalloc.h"
#include "../logs/logs.h"
#include "../ngbtree/ngbtree.h"
#include "../domain/domain.h"
#include "../time_integration/timestep.h"
#include "../time_integration/driftfac.h"

#ifdef DUST
#include "../dust/dust.h"
#endif

#include "agb_yields.h"

constexpr int spatial_hash_config::MIN_CELLS_PER_DIM;
constexpr int spatial_hash_config::MAX_CELLS_PER_DIM;
constexpr int spatial_hash_config::TARGET_PARTICLES_PER_CELL;
constexpr double spatial_hash_config::CELL_SIZE_SAFETY_FACTOR;
//constexpr int spatial_hash_config::MAX_TOTAL_CELLS;

// ------------------------------- Constants ----------------------------------
static const double ESN_ERG                  = 1.0e51;  // should probably be 1e51; SN energy in erg
static const double NSNE_PER_MSUN_VAL        = 0.0085;  // ~0.0085; Number of SNe per solar mass formed; calculated with Chabrier IMF and SN mass range 8-100 Msun
static const double METAL_YIELD_PER_SN_MSUN  = 2.0;     // Mass of metal yield in a Type II SN (Woosley & Weaver 1995, Nomoto+ 2006, Chieffi & Limongi 2004)
static const double AGB_METAL_YIELD_PER_MSUN = 9.956112e-03;  // Integrated over IMF
static const double AGB_E_ERG                = 1.0e47;  // 20 km/s typical winds; energy per solar mass of metals ejected (E_kinetic = 0.5 × M_total_eject × v_wind²)

// ΔT-target heating parameters
static const double DELTA_T_TARGET        = 1e7;  // 3.162277660e7 would be 10^7.5 K (EAGLE-like)
static const int    HEAT_N_MIN            = 32;
static const int    HEAT_N_MAX            = 256;
static const int    MAX_KERNEL_NEIGHBORS  = 512;

#ifdef FEEDBACK_LIMIT_DULOG
static const double FEEDBACK_MAX_DULOG    = 1.5; // Limit |Δlog10 u| per event (2.0 would be ~×100)
#endif
#ifdef FEEDBACK_T_CAP
static const double FEEDBACK_T_CAP_K      = 1.5e7; // Debug soft cap
#endif

// AGB diagnostics (global for all tasks)
static int agb_event_count = 0;
static double agb_total_metals_g = 0.0;
static double agb_total_energy_erg = 0.0;
static int agb_stars_checked = 0;
static int agb_stars_eligible = 0;
// Reservoir diagnostics
static int reservoir_release_attempts = 0;
static int reservoir_release_successes = 0;
static double reservoir_max_size = 0.0;
static int stars_with_reservoir = 0;
static double reservoir_total_energy = 0.0;

// Global spatial hash instance
spatial_hash_zoom gas_hash;
static int last_rebuild_step = -1;

#define FB_PRINT(...) do{ if(All.FeedbackDebugLevel){ \
  printf("[FEEDBACK|T=%d|a=%.6g z=%.3f] ", All.ThisTask, (double)All.Time, 1.0/All.Time-1.0); \
  printf(__VA_ARGS__); } }while(0)

// ---------------------------- Diagnostics -----------------------------------
template<typename T> static inline T fb_clamp(T v, T lo, T hi){ return v<lo?lo:(v>hi?hi:v); }

FeedbackDiagLocal FbDiag;

// Hot-path helpers (NO MPI here)
static inline void diag_add_sn(double E_erg){ FbDiag.n_SNII++; FbDiag.E_SN_erg += E_erg; }
static inline void diag_add_agb(double E_erg){ FbDiag.n_AGB++; FbDiag.E_AGB_erg += E_erg; }
static inline void diag_add_Edep(double v){ FbDiag.E_deposited_erg += v; }
static inline void diag_add_EtoRes(double v){ FbDiag.E_to_reservoir_erg += v; }
static inline void diag_add_EfromRes(double v){ FbDiag.E_from_reservoir_erg += v; }
static inline void diag_track_dulog(double a){ if(a>FbDiag.max_abs_dulog) FbDiag.max_abs_dulog=a; }


// -------------------- Feedback Spatial Hash Rebuild --------------------------
// Note: We share this hash between feedback and dust modules to avoid redundant
// neighbor searches. The hash automatically detects where gas exists and only
// creates cells over that region, saving 99%+ memory in zoom simulations.
void rebuild_feedback_spatial_hash(simparticles *Sp, double /*max_feedback_radius*/)
{
  static long long last_rebuild_step = -1;
  const int REBUILD_EVERY_N_STEPS = 20;
  
  // ============================================================
  // CRITICAL: All tasks must agree whether to rebuild or skip!
  // ============================================================
  int need_rebuild = 0;
  
  // Only task 0 decides
  if(All.ThisTask == 0) {
    if(!gas_hash.is_built || last_rebuild_step < 0 || 
       (All.NumCurrentTiStep - last_rebuild_step) >= REBUILD_EVERY_N_STEPS) {
      need_rebuild = 1;
    }
  }
  
  // Broadcast decision to all tasks
  MPI_Bcast(&need_rebuild, 1, MPI_INT, 0, MPI_COMM_WORLD);
  
  // Now all tasks agree - safe to return
  if(!need_rebuild) {
    return;
  }
  
  // ============================================================
  // All tasks proceed to rebuild (no deadlock possible)
  // ============================================================
  
  const double max_search_radius = 10.0;  // kpc
  double softening = All.ForceSoftening[0];
  
  gas_hash.build(Sp, max_search_radius, softening);
  
  last_rebuild_step = All.NumCurrentTiStep;
  
  if(All.ThisTask == 0 && gas_hash.is_built) {
    gas_hash.print_stats();
  }
}


void feedback_diag_try_flush(MPI_Comm comm, int cadence)
{
  // Every rank calls this each step; only flush when cadence matches
  if (cadence <= 0) return;
  if ((All.NumCurrentTiStep % cadence) != 0) return;

  // --- Start reductions (now we are in a synchronized phase) ---
  struct Local {
    long long n_SNII, n_AGB;
    double E_SN, E_AGB, E_dep, E_to_res, E_from_res, max_dulog;
  } in {
    FbDiag.n_SNII, FbDiag.n_AGB,
    FbDiag.E_SN_erg, FbDiag.E_AGB_erg,
    FbDiag.E_deposited_erg, FbDiag.E_to_reservoir_erg, FbDiag.E_from_reservoir_erg,
    FbDiag.max_abs_dulog
  }, out{};

  // Sum scalars
  MPI_Reduce(&in.n_SNII, &out.n_SNII, 1, MPI_LONG_LONG, MPI_SUM, 0, comm);
  MPI_Reduce(&in.n_AGB,  &out.n_AGB,  1, MPI_LONG_LONG, MPI_SUM, 0, comm);
  MPI_Reduce(&in.E_SN,   &out.E_SN,   1, MPI_DOUBLE,    MPI_SUM, 0, comm);
  MPI_Reduce(&in.E_AGB,  &out.E_AGB,  1, MPI_DOUBLE,    MPI_SUM, 0, comm);
  MPI_Reduce(&in.E_dep,  &out.E_dep,  1, MPI_DOUBLE,    MPI_SUM, 0, comm);
  MPI_Reduce(&in.E_to_res,&out.E_to_res,1, MPI_DOUBLE,  MPI_SUM, 0, comm);
  MPI_Reduce(&in.E_from_res,&out.E_from_res,1, MPI_DOUBLE, MPI_SUM, 0, comm);
  // Max of max_dulog
  MPI_Reduce(&in.max_dulog, &out.max_dulog, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

  if (All.ThisTask == 0) {
    if (out.n_SNII || out.n_AGB || out.E_dep || out.E_from_res) {
      FB_PRINT("events: SNII=%lld (%.3e erg)  AGB=%lld (%.3e erg)\n",
               out.n_SNII, out.E_SN, out.n_AGB, out.E_AGB);
      FB_PRINT("energy: deposited=%.3e erg  to_reservoir=%.3e erg  from_reservoir=%.3e erg\n",
               out.E_dep, out.E_to_res, out.E_from_res);
      if (out.max_dulog > 0.0)
        FB_PRINT("        max|Δlog10 u|=%.3f\n", out.max_dulog);
    }
  }

  // Reset local counters for next window
  FbDiag.reset();
}



// ----------------------- Stellar Age Calculation ----------------------------
//
// IMPORTANT CONTEXT:
//
// In Gadget runs with comoving integration, `All.Time` is the *scale factor* a,
// not a physical time. In your star particles you store:
//
//   Sp->P[i].StellarAge = All.Time;
//
// Despite the name "StellarAge", that field is actually:
//   -> a_birth  (the scale factor when the star formed)
//
// Therefore, to get the star's age in Myr, we need:
//
//   age = t(a_now) - t(a_birth)
//
// where t(a) is the *cosmic time* from the Friedmann equation:
//
//   t(a) = (1/H0) ∫[0..a] da' / (a' E(a'))
//   E(a) = sqrt( Ωm/a'^3 + Ωk/a'^2 + ΩΛ )
//
// CRITICAL NOTE:
//
// Driftfac.get_drift_factor(...) is NOT cosmic time.
// It's an integration helper for coordinate drift (an integral involving a),
// so treating it as Δt causes huge, unphysical ages. That was the source of
// your "AGB first" / CarbonFraction=0.6 early problem.
//
// PERFORMANCE NOTE:
//
// We precompute a table of t(a) once, then use linear interpolation.
// This makes age lookups O(1) and cheap even with many stars.
//

// Table for cosmic time t(a) [Gyr], sampled uniformly in a.
// We build it once in init_stellar_feedback().
static bool   cosmic_time_table_built = false;
static int    cosmic_time_N = 0;
static double cosmic_time_a_min = 0.0;
static double cosmic_time_a_max = 1.0;
static std::vector<double> cosmic_time_tGyr;  // t(a_i) in Gyr

// Return E(a) = H(a)/H0 in standard LCDM cosmology.
// Uses Gadget-style params: Omega0 (matter), OmegaLambda (vacuum), plus curvature.
static inline double E_of_a(double a)
{
  // Protect against nonsense.
  if(a <= 0.0) return 1e30;

  const double Om = All.Omega0;
  const double Ol = All.OmegaLambda;
  const double Ok = 1.0 - Om - Ol;  // curvature term

  // For a flat run, Ok==0.
  // Terms: matter ~ a^-3, curvature ~ a^-2, lambda ~ const.
  return sqrt(Om/(a*a*a) + Ok/(a*a) + Ol);
}

// Convert H0 to 1/Gyr.
// H0 = 100 h km/s/Mpc.
// 1 km/s/Mpc = 3.240779289e-20 s^-1.
// 1 Gyr = 3.15576e16 s.
static inline double H0_in_invGyr()
{
  const double km_s_Mpc_to_inv_s = 3.240779289e-20;
  const double sec_per_Gyr = 3.15576e16;

  double H0_inv_s = 100.0 * All.HubbleParam * km_s_Mpc_to_inv_s;
  return H0_inv_s * sec_per_Gyr;  // [1/Gyr]
}

// Build t(a) table by numerically integrating dt/da = 1/(a H(a)).
// We do the integral progressively from a_min->a_max with trapezoid rule.
//
// Why trapezoid is fine here:
// - We're building a smooth monotonic mapping.
// - Table resolution can be increased cheaply.
// - This is done once at startup, not in the hot path.
//
static void build_cosmic_time_table()
{
  // If not comoving, All.Time is already "time" in code units.
  // In that case we don't need a cosmology table at all.
  if(!All.ComovingIntegrationOn) {
    cosmic_time_table_built = false;
    return;
  }

  // Choose integration range:
  // - a_min: start of the simulation (All.TimeBegin)
  // - a_max: 1.0 (z=0) is safe even if you stop earlier.
  cosmic_time_a_min = All.TimeBegin;
  cosmic_time_a_max = 1.0;

  // Resolution: higher = more accurate interpolation.
  // 2000 is usually plenty; feel free to raise to 5000 if you want.
  cosmic_time_N = 2000;

  cosmic_time_tGyr.assign(cosmic_time_N, 0.0);

  const double H0Gyr = H0_in_invGyr();

  // Step in a.
  const double da = (cosmic_time_a_max - cosmic_time_a_min) / (cosmic_time_N - 1);

  // Integrate from a_min upward: t(a_min) = 0 by definition for our table.
  // (We only care about time differences anyway.)
  double t = 0.0;
  cosmic_time_tGyr[0] = t;

  for(int i = 1; i < cosmic_time_N; i++) {
    double a0 = cosmic_time_a_min + (i - 1) * da;
    double a1 = cosmic_time_a_min + i * da;

    // dt/da = 1 / (a * H(a)) = 1 / (a * H0 * E(a))
    // where H0 is in 1/Gyr, so dt comes out in Gyr.
    double f0 = 1.0 / (a0 * H0Gyr * E_of_a(a0));
    double f1 = 1.0 / (a1 * H0Gyr * E_of_a(a1));

    // Trapezoid step
    t += 0.5 * (f0 + f1) * da;

    cosmic_time_tGyr[i] = t;
  }

  cosmic_time_table_built = true;

  if(All.ThisTask == 0) {
    FB_PRINT("[AGE] Built cosmic time table: a in [%.6g, %.6g], N=%d\n",
             cosmic_time_a_min, cosmic_time_a_max, cosmic_time_N);
    FB_PRINT("[AGE] Table endpoint: t(a=1) - t(a_begin) ≈ %.3f Gyr (relative)\n",
             cosmic_time_tGyr.back());
  }
}

// Interpolate t(a) from the table, returning cosmic time in Gyr (relative to a_min).
static inline double cosmic_time_Gyr_from_table(double a)
{
  // If table isn't built (e.g. non-comoving run), return 0.
  if(!cosmic_time_table_built || cosmic_time_N < 2) return 0.0;

  // Clamp input to table range
  if(a <= cosmic_time_a_min) return cosmic_time_tGyr.front();
  if(a >= cosmic_time_a_max) return cosmic_time_tGyr.back();

  const double da = (cosmic_time_a_max - cosmic_time_a_min) / (cosmic_time_N - 1);
  double x = (a - cosmic_time_a_min) / da;
  int i = (int)floor(x);
  if(i < 0) i = 0;
  if(i > cosmic_time_N - 2) i = cosmic_time_N - 2;

  double frac = x - i;
  return cosmic_time_tGyr[i] * (1.0 - frac) + cosmic_time_tGyr[i + 1] * frac;
}

// Public helper used by feedback logic.
// Input is *birth scale factor* a_birth stored in P[i].StellarAge.
double get_stellar_age_Myr(double a_birth, double /*unused*/)
{
  // Common failure mode: unset / corrupted birth time.
  // If we can't trust it, return 0 so the star does not trigger feedback.
  if(a_birth <= 0.0 || a_birth > 1.0) return 0.0;

  // Another failure mode: some formation code might set StellarAge = All.Time
  // even for newly created stars, which is fine. If a_birth >= a_now, age=0.
  double a_now = All.Time;
  if(a_birth >= a_now) return 0.0;

  // Case A: Cosmological run (All.Time is scale factor)
  if(All.ComovingIntegrationOn) {
    // Ensure table is built (should be done in init, but this makes it robust).
    if(!cosmic_time_table_built) build_cosmic_time_table();

    double t_now_Gyr   = cosmic_time_Gyr_from_table(a_now);
    double t_birth_Gyr = cosmic_time_Gyr_from_table(a_birth);

    double age_Gyr = t_now_Gyr - t_birth_Gyr;
    double age_Myr = age_Gyr * 1000.0;

    // Debug first few calls on rank 0.
    static int debug_count = 0;
    if(debug_count < 5 && All.ThisTask == 0) {
      printf("[AGE_DEBUG] a_birth=%.6f a_now=%.6f | t_birth=%.6f Gyr t_now=%.6f Gyr | age=%.3f Myr\n",
             a_birth, a_now, t_birth_Gyr, t_now_Gyr, age_Myr);
      debug_count++;
    }

    return (age_Myr > 0.0) ? age_Myr : 0.0;
  }

  // Case B: Non-comoving run (All.Time is already "time" in code units)
  //
  // Here All.Time is not a scale factor; it’s a time-like variable advanced
  // directly. But you're still storing "StellarAge = All.Time" at birth, so:
  // age_code = All.Time - birth_time_code.
  //
  // Convert code time to seconds using UnitLength/UnitVelocity (standard Gadget units):
  // UnitTime = UnitLength / UnitVelocity.
  double dt_code = a_now - a_birth;
  if(dt_code <= 0.0) return 0.0;

  double unit_time_s = All.UnitLength_in_cm / All.UnitVelocity_in_cm_per_s;
  double age_s  = dt_code * unit_time_s;
  double age_Myr = age_s / (SEC_PER_YEAR * 1.0e6);

  return (age_Myr > 0.0) ? age_Myr : 0.0;
}


// ----------------------- Neighbor Finding -----------------------------------

/* Get local smoothing length for feedback kernel
 * 
 * Strategy: Use nearby gas Hsml for star particles, fallback to softening
 */
double get_local_smoothing_length_tree(simparticles *Sp, ngbtree * /*Tree*/, int star_idx)
{
  double h = 0.0;
  
  // For gas particles, use their own Hsml
  if (Sp->P[star_idx].getType() == 0 && Sp->SphP[star_idx].Hsml > 0) {
    h = Sp->SphP[star_idx].Hsml;
  } 
  // For star particles, sample nearby gas to get typical scale
  else if (Sp->P[star_idx].getType() == 4) {
    double best_r2 = 1e300;
    for (int i = 0, scanned = 0; i < Sp->NumGas && scanned < 256; ++i, ++scanned) {
      double d[3]; 
      Sp->nearest_image_intpos_to_pos(Sp->P[i].IntPos, Sp->P[star_idx].IntPos, d);
      double r2 = d[0]*d[0] + d[1]*d[1] + d[2]*d[2];
      if (r2 < best_r2 && Sp->SphP[i].Hsml > 0) { 
        best_r2 = r2; 
        h = Sp->SphP[i].Hsml; 
      }
    }
  }
  
  // Fallback: use gas softening
  if (h <= 0.0) h = std::max(All.SofteningTable[0], 1e-6);
  return h;
}


/* Find feedback neighbors within 2h kernel support
 * 
 * Uses spatial hash for average-case neighbor finding.
 * Falls back to error if hash not built (should not happen in normal operation).
 */
void find_feedback_neighbors_tree(simparticles *Sp, ngbtree * /*Tree*/, 
                                  domain<simparticles> * /*D*/,
                                  int star_idx,
                                  int *ngb_list, double *distances, int *n_ngb,
                                  double *smoothing_length, int max_ngb)
{
  *n_ngb = 0;
  double h = *smoothing_length;
  if (h <= 0.0) h = get_local_smoothing_length_tree(Sp, nullptr, star_idx);
  if (h <= 0.0) { *smoothing_length = 0.0; return; }

  const double search_radius = 2.0 * h;

    if(!gas_hash.is_built) {
      FB_PRINT("ERROR: Spatial hash not built! This should not happen.\n");
      *smoothing_length = h;
      return;
    }

  //gas_hash.find_neighbors(Sp, star_idx, search_radius, ngb_list, distances, n_ngb, max_ngb);
  
  for(int i = 0; i < Sp->NumGas; i++) {
  double dxyz[3];
  Sp->nearest_image_intpos_to_pos(Sp->P[i].IntPos, Sp->P[star_idx].IntPos, dxyz);
  double r2 = dxyz[0]*dxyz[0] + dxyz[1]*dxyz[1] + dxyz[2]*dxyz[2];
  if(r2 <= search_radius*search_radius) {
    // Found neighbor
  }
}
  
  *smoothing_length = h;
}

/* Wrapper for vector-based interface */
static void gather_neighbors(simparticles *Sp, ngbtree *Tree, domain<simparticles>*D,
                             int star_i, std::vector<int>& ngb, 
                             std::vector<double>& dist, double &hsml)
{
  int n_found=0; 
  hsml = get_local_smoothing_length_tree(Sp, Tree, star_i);
  if(hsml<=0) return;
  
  ngb.resize(MAX_KERNEL_NEIGHBORS); 
  dist.resize(MAX_KERNEL_NEIGHBORS);
  
  find_feedback_neighbors_tree(Sp, Tree, D, star_i, ngb.data(), dist.data(), 
                               &n_found, &hsml, MAX_KERNEL_NEIGHBORS);
  
  ngb.resize(n_found); 
  dist.resize(n_found);
}

// ----------------------- SPH Kernel -----------------------------------------

double cubic_spline_kernel(double r, double h)
{
  if(h<=0) return 0.0;
  double q = fabs(r)/h;
  const double sig = 8.0/(M_PI*h*h*h);
  if(q<1.0) return sig*(1.0 - 1.5*q*q + 0.75*q*q*q);
  if(q<2.0){ double t=2.0-q; return sig*0.25*t*t*t; }
  return 0.0;
}

// ----------------------- Thermodynamics -------------------------------------

static inline double mu_default(double T){ return (T>1.5e4? 0.62 : 1.22); }
static inline double c_v(double mu){ return 1.5*BOLTZMANN/(mu*PROTONMASS); }

static inline double ucode_to_TK(double u_code){
  double u_cgs = u_code * All.UnitVelocity_in_cm_per_s * All.UnitVelocity_in_cm_per_s;
  double mu = mu_default(1e4);
  return u_cgs / c_v(mu);
}

static inline double TK_to_ucode(double T_K){
  double mu = mu_default(T_K);
  double u_cgs = c_v(mu) * T_K;
  return u_cgs / (All.UnitVelocity_in_cm_per_s * All.UnitVelocity_in_cm_per_s);
}

// ----------------------- Energy Deposition ----------------------------------

/* Deposit energy to neighbors using stochastic kernel-weighted heating
 * 
 * Algorithm:
 *   1. Determine Nheat = number of particles to heat (1/8 of neighbors, clamped to [32,256])
 *   2. Select Nheat neighbors stochastically, weighted by SPH kernel
 *   3. Distribute ALL available energy among selected neighbors, proportional to kernel-weighted mass
 *   4. Distribute metals the same way (kernel-weighted)
 *   5. Apply optional safety caps on temperature and Δlog10(u)
 *   6. Return total energy actually deposited
 * 
 * Notes:
 *   - Closer particles (higher kernel weight) receive more energy
 *   - Energy distribution: E_i = E_total × (m_i × W_i) / Σ(m_j × W_j)
 *   - Metal distribution uses same kernel weighting as energy
 *   - Does NOT target a specific ΔT - deposits all available energy
 * 
 * Parameters:
 *   E_avail_code - Total energy available to deposit (code units)
 *   MZ_code      - Total metal mass to distribute (code units)
 * 
 * Returns:
 *   Total energy successfully deposited (code units)
 */
static double deposit_energy_stochastic(simparticles *Sp,
                                        const std::vector<int> &ngb_list,
                                        const std::vector<double> &distances,
                                        double hsml, int star_idx,
                                        double E_avail_code, double MZ_code,
                                        int &particles_heated_global,      // ← ADD
                                        int max_particles_allowed)   
{
  if(ngb_list.empty() || E_avail_code<=0) return 0.0;

      // Track particles heated this timestep to prevent multi-heating
      // When multiple stars explode near the same gas particle, this ensures each
      // gas particle is only heated once per timestep, preventing runaway temperatures.
      // Critical for zoom simulations where star density can be very high.
      static std::set<int> heated_this_step;
      static long long last_step = -1;
      
      // Reset tracking at new timestep
      if(All.NumCurrentTiStep != last_step) {
        heated_this_step.clear();
        last_step = All.NumCurrentTiStep;
      }

  // Determine number of particles to heat
  int Nheat = fb_clamp((int)(ngb_list.size()/8), HEAT_N_MIN, HEAT_N_MAX);
  if(Nheat > (int)ngb_list.size()) Nheat = ngb_list.size();
  if(Nheat < 1) Nheat = 1;

  // Compute kernel weights for probabilistic selection
  std::vector<double> weights(ngb_list.size());
  double weight_sum = 0.0;
  for(size_t i=0; i<ngb_list.size(); ++i){
    weights[i] = cubic_spline_kernel(distances[i], hsml);
    weight_sum += weights[i];
  }
  if(weight_sum<=0) return 0.0;

  // Normalize to cumulative distribution
  std::vector<double> cumul(ngb_list.size());
  cumul[0] = weights[0]/weight_sum;
  for(size_t i=1; i<ngb_list.size(); ++i) 
    cumul[i] = cumul[i-1] + weights[i]/weight_sum;

  // Stochastically select Nheat neighbors
  std::vector<int> selected;
  for(int k=0; k<Nheat; ++k){
    double rnd = get_random_number(Sp->P[star_idx].ID.get() + k + All.NumCurrentTiStep);
    for(size_t i=0; i<cumul.size(); ++i){
      if(rnd <= cumul[i]){
        bool already = false;
        for(int s : selected) if(s==(int)i){ already=true; break; }
        if(!already) selected.push_back(i);
        break;
      }
    }
  }
  if(selected.empty()) return 0.0;

  // ✅ Compute total kernel-weighted mass for normalization
  double total_kernel_weighted_mass = 0.0;
  for(int idx : selected) {
    int j = ngb_list[idx];
    double w = cubic_spline_kernel(distances[idx], hsml);
    total_kernel_weighted_mass += Sp->P[j].getMass() * w;
  }
  
  if(total_kernel_weighted_mass <= 0) return 0.0;  // Safety check

  // ✅ Distribute energy AND metals proportional to kernel-weighted mass
  double E_used_code = 0.0;  // Track actual energy deposited
  
  for(int idx : selected) {
    int j = ngb_list[idx];

        // Skip if already heated this timestep (prevent multi-heating)
        // This prevents the same gas particle from being heated by multiple nearby SNe
        if(heated_this_step.count(j) > 0) {
          continue;  // Already heated by another star this timestep, skip
        }


                // ✅ NEW: Check global rate limit BEFORE heating
        if(particles_heated_global >= max_particles_allowed) {
            // Hit the rate limit - stop heating
            // Remaining energy will go to reservoir (handled by caller)
            break;
        }


    double m_g = Sp->P[j].getMass();
    double w = cubic_spline_kernel(distances[idx], hsml);
    double fraction = (m_g * w) / total_kernel_weighted_mass;
    
    // Energy for this particle (in code units)
    double dE_code = E_avail_code * fraction;
    
    // Convert to specific internal energy change
    double du_cgs = (dE_code * All.UnitEnergy_in_cgs) / (m_g * All.UnitMass_in_g);
    double du_code = du_cgs / (All.UnitVelocity_in_cm_per_s * All.UnitVelocity_in_cm_per_s);

    double u_old = Sp->get_utherm_from_entropy(j);
    if(u_old <= 0) {
      FB_PRINT("WARNING: Particle %d has u_old=%g, skipping feedback\n", j, u_old);
      continue;  // Skip this particle
    }

    double u_new = u_old + du_code;

#ifdef FEEDBACK_LIMIT_DULOG
    double du_log10 = log10(u_new) - log10(u_old);
    if(fabs(du_log10) > FEEDBACK_MAX_DULOG){
      double clamped = fb_clamp(du_log10, -FEEDBACK_MAX_DULOG, FEEDBACK_MAX_DULOG);
      u_new = u_old * pow(10.0, clamped);
      
      diag_track_dulog(fabs(clamped));  // ✅ Track CLAMPED value
    }
#endif

#ifdef FEEDBACK_T_CAP
    double T_new = ucode_to_TK(u_new);
    if(T_new > FEEDBACK_T_CAP_K){
      double u_cap_code = TK_to_ucode(FEEDBACK_T_CAP_K);
      if(u_cap_code < u_new) u_new = u_cap_code;
    }
#endif

    // Safety cap
    const double u_max_particle = TK_to_ucode(1.0e8);  // Was 1.0e4 code units
    if(u_new > u_max_particle) u_new = u_max_particle;

    Sp->set_entropy_from_utherm(u_new, j);
    set_thermodynamic_variables_safe(Sp, j);
    
    // 🛡️ CRITICAL: Verify the cap stuck after entropy conversion
    double u_verify = Sp->get_utherm_from_entropy(j);
    
    #ifdef FEEDBACK_T_CAP
        double u_cap_verify = TK_to_ucode(FEEDBACK_T_CAP_K);  // Use the ACTUAL cap
    #else
        double u_cap_verify = u_max_particle;  // Use safety cap
    #endif
    
    if(u_verify > u_cap_verify * 1.1) {  // ← Use the correct cap!
        // Entropy roundtrip broke the cap - force it back
        Sp->set_entropy_from_utherm(u_cap_verify, j);  // ← Use correct cap!
        set_thermodynamic_variables_safe(Sp, j);
        
        static int cap_failures = 0;
        if(cap_failures < 10 && All.ThisTask == 0) {
            printf("[FEEDBACK_CAP_FAIL] Entropy roundtrip broke cap! Particle %d: u_capped=%g -> u_verify=%g, re-capping at %.2e K\n",
                  j, u_new, u_verify, 
                  #ifdef FEEDBACK_T_CAP
                      FEEDBACK_T_CAP_K
                  #else
                      1.0e8
                  #endif
                  );
            cap_failures++;
        }
        
        u_new = u_cap_verify;  // ← Update for the final temperature check below
    }
        
        // 🏁 Mark particle as heated this timestep
        heated_this_step.insert(j);


        particles_heated_global++;  // ← INCREMENT GLOBAL COUNTER

    // 🔍 DEBUG: Check if cap is working
    double T_final = ucode_to_TK(Sp->get_utherm_from_entropy(j));
    static int cap_debug_count = 0;
    if(T_final > 6.0e7 && cap_debug_count < 20 && All.ThisTask == 0) {
        printf("[CAP_DEBUG] Particle %d: T_final = %.2e K, Hsml = %.4f kpc\n",
               j, T_final, Sp->SphP[j].Hsml);
        printf("            u_old=%.2e, u_new=%.2e, u_verify=%.2e\n",
               u_old, u_new, Sp->get_utherm_from_entropy(j));
        
        // Calculate expected timestep
        double u_current = Sp->get_utherm_from_entropy(j);
        double cs_cgs = sqrt(GAMMA * u_current * 
                            All.UnitVelocity_in_cm_per_s * All.UnitVelocity_in_cm_per_s);
        double cs_kms = cs_cgs / 1e5;
        double hsml_code = Sp->SphP[j].Hsml;
        
        // Courant timestep estimate: dt = CourantFac × h / cs
        // hsml is in code units (kpc/h comoving), cs in km/s
        // Need to convert to consistent units
        double hsml_physical_kpc = hsml_code / All.HubbleParam * All.Time;  // Physical kpc
        double hsml_km = hsml_physical_kpc * 3.086e16;  // Convert kpc to km
        double dt_est_sec = 0.075 * hsml_km / cs_kms;  // seconds
        double dt_est_code = dt_est_sec / (All.UnitLength_in_cm / All.UnitVelocity_in_cm_per_s);
        
        printf("            cs=%.0f km/s, dt_estimate=%.2e (code units)\n", 
               cs_kms, dt_est_code);
        printf("            This would be timebin %d\n", 
               (int)(log2(All.Timebase_interval / dt_est_code)));
        cap_debug_count++;
    }





    // Metals distributed the same way as energy (kernel-weighted)
    #ifdef METALS
        if(MZ_code > 0) { 
          double metal_mass_code = MZ_code * fraction;  // Kernel-weighted metals
          Sp->SphP[j].MassMetallicity += metal_mass_code;
          Sp->SphP[j].Metallicity = Sp->SphP[j].MassMetallicity / Sp->P[j].getMass();
        }
    #endif

    // Track actual energy used
    E_used_code += dE_code;
  }

  return E_used_code;  // Return in code units
}

// -------------------- Feedback Event Handling -------------------------------

/* Apply a feedback event (SNII or AGB) and manage reservoir */
static void apply_feedback_event(simparticles *Sp, ngbtree *Tree, 
                                 domain<simparticles>*D,
                                 int star_i, bool is_SNII, 
                                 double E_code, double MZ_code,
                                 int &particles_heated_global,      // ← ADD
                                 int max_particles_allowed)  
{
  std::vector<int> ngb; 
  std::vector<double> dist; 
  double hsml=0;
  gather_neighbors(Sp, Tree, D, star_i, ngb, dist, hsml);
  if(ngb.empty()) {
      FB_PRINT("WARNING: Star %d found no neighbors, storing energy in reservoir\n", star_i);
      
      // Store ALL energy in reservoir for later
      Sp->P[star_i].EnergyReservoir += E_code;
      
      // Track event and energy accounting
      if (is_SNII) diag_add_sn(E_code * All.UnitEnergy_in_cgs);
      else         diag_add_agb(E_code * All.UnitEnergy_in_cgs);
      diag_add_EtoRes(E_code * All.UnitEnergy_in_cgs);
      
      return;
  }

    double E_used_code = deposit_energy_stochastic(Sp, ngb, dist, hsml, 
                                                   star_i, E_code, MZ_code,
                                                   particles_heated_global,
                                                   max_particles_allowed);


  const double E_used_erg    = E_used_code * All.UnitEnergy_in_cgs;
  const double E_leftover_erg = std::max(0.0, (E_code - E_used_code) * All.UnitEnergy_in_cgs);

  if (is_SNII) diag_add_sn(E_code * All.UnitEnergy_in_cgs);
  else         diag_add_agb(E_code * All.UnitEnergy_in_cgs);

  diag_add_Edep(E_used_erg);
  diag_add_EtoRes(E_leftover_erg);


  // Store any leftover energy in reservoir
  double leftover = std::max(0.0, E_code - E_used_code);
  Sp->P[star_i].EnergyReservoir += leftover;

  if(leftover > 0 && All.FeedbackDebugLevel >= 2) {
      printf("[RESERVOIR_ADD|T=%d] Star %d: added %.2e to reservoir (total now %.2e)\n",
            All.ThisTask, star_i, leftover, Sp->P[star_i].EnergyReservoir);
  }
}

/* Try to release accumulated reservoir energy
 * 
 * Only releases when reservoir exceeds threshold (90% of energy needed
 * to heat Nheat neighbors by ΔT). This prevents wasteful small deposits.
 * 
 * Tuning: Adjust the 0.3 factor at line ~320 to change aggressiveness:
 *   - Lower (0.3-0.5): More frequent releases, good for isolated stars
 *   - Higher (0.3): Wait for larger reservoirs, good for clusters
 */
static void try_release_reservoir(simparticles *Sp, ngbtree *Tree, 
                                  domain<simparticles>*D, int star_i,
                                  int &particles_heated_global,      // ← ADD
                                  int max_particles_allowed)    
{
  double E_code = Sp->P[star_i].EnergyReservoir;
  if(E_code<=0) return;

  // Track that we're trying
  reservoir_release_attempts++;
  
  // Track reservoir statistics
  if(E_code > reservoir_max_size) reservoir_max_size = E_code;
  
  std::vector<int> ngb; 
  std::vector<double> dist; 
  double hsml=0;
  gather_neighbors(Sp, Tree, D, star_i, ngb, dist, hsml);
  if(ngb.empty()) {
    // Isolated star - vent energy as radiation instead of accumulating forever
    if(E_code > 0.5) {  // Only vent if reservoir is significant
        double vent_fraction = 0.02;  // Vent 2% per timestep (slow drain)
        double vented = E_code * vent_fraction;
        Sp->P[star_i].EnergyReservoir -= vented;
        
        diag_add_EfromRes(vented * All.UnitEnergy_in_cgs);
        
        static int vent_count = 0;
        if(vent_count < 10 && All.ThisTask == 0) {
            printf("[RESERVOIR_VENT] Star %d: isolated, venting %.2e\n", star_i, vented);
            vent_count++;
        }
    }
    return;
}

  // Compute threshold
  int Nheat = fb_clamp((int)(ngb.size()/8), HEAT_N_MIN, HEAT_N_MAX);
  if(Nheat<1) Nheat=1;
  
  double total_mass = 0.0;
  int n_sample = std::min(Nheat, (int)ngb.size());
  for(int i = 0; i < n_sample; i++) {
      total_mass += Sp->P[ngb[i]].getMass();
  }
  double m_g_code = total_mass / n_sample;
    
  double E_per_g  = c_v(mu_default(DELTA_T_TARGET))*DELTA_T_TARGET;
  double E_one    = E_per_g * (m_g_code*All.UnitMass_in_g);
  double E_need   = (E_one * Nheat) / All.UnitEnergy_in_cgs;

  // Debug output (occasional)
  static int release_debug_count = 0;
  if(release_debug_count < 5 && All.ThisTask == 0) {
      printf("[RESERVOIR_DEBUG] Star %d: E_res=%.2e (code), E_need=%.2e, ratio=%.3f\n",
             star_i, E_code, E_need, E_code/E_need);
      release_debug_count++;
  }

  // Only release if we have meaningful energy
  if(E_code < 0.0001*E_need) return;  // Needs 0.806 code units
  
  // We passed the threshold!
  reservoir_release_successes++;

  double E_to_release = std::min(E_code, E_need * 2.0);
    double used = deposit_energy_stochastic(Sp, ngb, dist, hsml, star_i, 
                                           E_to_release, 0.0,
                                           particles_heated_global,
                                           max_particles_allowed); 
  
  diag_add_EfromRes(used * All.UnitEnergy_in_cgs);
  diag_add_Edep(used * All.UnitEnergy_in_cgs);
  
  Sp->P[star_i].EnergyReservoir = std::max(0.0, E_code - used);
}

// ---------------------------- Public API ------------------------------------

void init_stellar_feedback(void)
{
  #ifdef FEEDBACK
    if(All.ThisTask == 0) {
      FB_PRINT("FEEDBACK enabled; ΔT_target=%.2e K, Nheat=[%d,%d]\n", 
              DELTA_T_TARGET, HEAT_N_MIN, HEAT_N_MAX);
      
      // Load AGB yield table from parameter file path
      if(!AGB_Yields.load_from_file(All.AGByieldFile)) {
        printf("[FEEDBACK] ERROR: Failed to load AGB yields from '%s'\n", 
               All.AGByieldFile);
        printf("[FEEDBACK] AGB feedback will be DISABLED\n");
      } else {
        printf("[FEEDBACK] AGB yields loaded successfully from '%s'\n",
               All.AGByieldFile);
      }
    }

      // Build cosmic time lookup once so get_stellar_age_Myr() is cheap.
      // Only used if All.ComovingIntegrationOn == 1.
      build_cosmic_time_table();
  #endif
}

/* Utilizes a spatial binning approach so that we don't have to loop over ever gas particle 
to find the neighbors to a star that would receive feedback. */
void apply_stellar_feedback(double /*current_time*/, simparticles *Sp, 
                           ngbtree *Tree, domain<simparticles> *D)
{
  double t_start = MPI_Wtime();
    
    // Rate limiting to prevent timebin cascades
    static int particles_heated_this_step = 0;
    static long long last_heating_step = -1;
    
    if(All.NumCurrentTiStep != last_heating_step) {
        particles_heated_this_step = 0;
        last_heating_step = All.NumCurrentTiStep;
    }
    
    const int MAX_PARTICLES_HEATED_PER_STEP = Sp->NumGas / 100;  // 1% of gas per step
    
    // Build star index list FIRST (very fast)
    static std::vector<int> star_indices;
    star_indices.clear();
    star_indices.reserve(Sp->NumPart / 10);
    
    for(int p = 0; p < Sp->NumPart; ++p) {
        if(Sp->P[p].getType() == 4) {
            star_indices.push_back(p);
        }
    }
    
    int n_stars_processed = star_indices.size();
    
    // Early exit BEFORE expensive calculations
    int local_has_work = 0;

    bool any_reservoir = false;
    for (int p : star_indices) {
      if (Sp->P[p].EnergyReservoir > 0) { any_reservoir = true; break; }
    }

    if (n_stars_processed > 0 || any_reservoir) {
        local_has_work = 1;
    }

    // Check if ANY task has work to do
    int global_has_work = 0;
    MPI_Allreduce(&local_has_work, &global_has_work, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    // All tasks return together, or all continue together
    if (!global_has_work) {
        return;  // Now safe - all tasks agree to exit
    }

    // NOW calculate max_radius (only if stars exist)
    double max_radius = 10.0;
    if(Sp->NumGas > 0) {
        double h_sum = 0;
        int h_count = 0;
        for(int i = 0; i < std::min(100, Sp->NumGas); i += std::max(1, Sp->NumGas/100)) {
            if(Sp->SphP[i].Hsml > 0) {
                h_sum += Sp->SphP[i].Hsml;
                h_count++;
            }
        }
        if(h_count > 0) {
            double typical_h = h_sum / h_count;
            max_radius = 2.5 * typical_h;
        }
    }

    // Build spatial hash (only if stars exist will we get this far)
    rebuild_feedback_spatial_hash(Sp, max_radius);

    // ====================================================================
    // 1) Try to release reservoir energy (ONLY non-empty reservoirs)
    // ====================================================================
    reservoir_release_attempts = 0;
    reservoir_release_successes = 0;
    reservoir_max_size = 0.0;
    stars_with_reservoir = 0;
    reservoir_total_energy = 0.0;

    for(int p : star_indices) {
        if(Sp->P[p].EnergyReservoir > 0) {
            stars_with_reservoir++;
            reservoir_total_energy += Sp->P[p].EnergyReservoir;
                        try_release_reservoir(Sp, Tree, D, p,
                                 particles_heated_this_step,
                                 MAX_PARTICLES_HEATED_PER_STEP);
        }
    }
    
    // ====================================================================
    // 2) Trigger new feedback events (SNe AND AGB in same loop)
    // ====================================================================
    for(int p : star_indices) {
        double age_Myr = get_stellar_age_Myr(Sp->P[p].StellarAge, 0.0);
        
        // ----------------------------------------------------------------
        // 2a) Type II SNe (3-40 Myr window)
        // ----------------------------------------------------------------
        if(age_Myr >= MIN_TYPEII_TIME && age_Myr < MAX_TYPEII_TIME && 
           !(Sp->P[p].FeedbackFlag & 1)) {
            
            double m_star_g = Sp->P[p].getMass() * All.UnitMass_in_g;
            double n_SN = NSNE_PER_MSUN_VAL * (m_star_g / SOLAR_MASS);
            double E_erg = n_SN * ESN_ERG;
            
            // Metal mass = number of SNe × yield per SN
            double MZ_g = n_SN * METAL_YIELD_PER_SN_MSUN * SOLAR_MASS;
            double MZ_code = MZ_g / All.UnitMass_in_g; 
            
            static int snii_event_count = 0;
            snii_event_count++;
            if(snii_event_count % 10 == 0 && All.ThisTask == 0) {
                FB_PRINT("[SNII_EVENT] #%d: Star %d, age=%.1f Myr, M=%.3f Msun\n",
                        snii_event_count, p, age_Myr, m_star_g/SOLAR_MASS);
                FB_PRINT("            n_SN=%.3f, MZ=%.3e Msun (%.3e code)\n",
                        n_SN, MZ_g/SOLAR_MASS, MZ_code);
            }

            double E_code = E_erg / All.UnitEnergy_in_cgs;
                        apply_feedback_event(Sp, Tree, D, p, /*SNII*/true, E_code, MZ_code,
                                particles_heated_this_step,       // ← ADD
                                MAX_PARTICLES_HEATED_PER_STEP);  // ← ADD

            #ifdef DUST
              if(snii_event_count % 10 == 0 && All.ThisTask == 0) {
                    FB_PRINT("            Calling dust creation with MZ=%.3e, type=1\n", MZ_code);
              }

                // Create dust from SN ejecta
                create_dust_particles_from_feedback(Sp, p, MZ_code, 1);  // 1 = Type II SN
                
                // Destroy nearby dust in shock
                destroy_dust_from_sn_shocks(Sp, p, E_code, MZ_code);
            #endif

            Sp->P[p].FeedbackFlag |= 1;
        }
        
        // ----------------------------------------------------------------
        // 2b) AGB enrichment (one-time event using MESA yield table)
        // ----------------------------------------------------------------
        // Note: We load the AGB yield table for future use (individual elements),
        //       but currently use a pre-integrated total yield for simplicity.
        //            double C_msun = integrate_element_over_imf(AGB_Yields, "C", m_star_msun, Z_star);
        //            double N_msun = integrate_element_over_imf(AGB_Yields, "N", m_star_msun, Z_star);
        if(AGB_Yields.is_table_loaded()) {
            
            agb_stars_checked++;  // Count all stars checked for AGB
            
            // Check if star is old enough for AGB
            if(age_Myr >= MIN_AGB_TIME && !(Sp->P[p].FeedbackFlag & 2)) {
                
                agb_stars_eligible++;  // Count eligible stars
                
                // Get stellar properties
                double m_star_g = Sp->P[p].getMass() * All.UnitMass_in_g;
                double m_star_msun = m_star_g / SOLAR_MASS;
                
                // Get metallicity (default to solar if not tracked)
                double Z_star = 0.02;  // Solar default
                #ifdef METALS
                    // If you track particle metallicity, use it:
                    // Z_star = Sp->P[p].Metallicity;
                #endif
                
                // Use our pre-calculated value of metal yields per solar mass formed with Chabrier IMF
                double MZ_msun = AGB_METAL_YIELD_PER_MSUN * m_star_msun;
                
                if(MZ_msun > 0.0) {  // Only proceed if there's a yield
                    
                    // Convert to code units
                    double MZ_g = MZ_msun * SOLAR_MASS;
                    double MZ_code = MZ_g / All.UnitMass_in_g;
                    
                    // Energy (simple scaling)
                    double E_erg = MZ_msun * AGB_E_ERG;
                    double E_code = E_erg / All.UnitEnergy_in_cgs;
                    
                    // Debug output for first few events on task 0
                    if(agb_event_count < 5 && All.ThisTask == 0) {
                      FB_PRINT("[AGB_DEBUG|T=%d] Star %d: age=%.1f Myr, M=%.3e Msun, Z=%.4f | "
                            "MZ=%.3e Msun (%.2e code) | E=%.2e erg\n",
                            All.ThisTask, p, age_Myr, m_star_msun, Z_star, 
                            MZ_msun, MZ_code, E_erg);
                    }
                    
                    // Apply feedback event
                    apply_feedback_event(Sp, Tree, D, p, /*is_SNII=*/false, E_code, MZ_code,
                    particles_heated_this_step, MAX_PARTICLES_HEATED_PER_STEP);
                    
                    #ifdef DUST
                        // Create dust from AGB winds (no shock destruction)
                        create_dust_particles_from_feedback(Sp, p, MZ_code, 2);  // 2 = AGB
                    #endif

                    // Mark as done
                    Sp->P[p].FeedbackFlag |= 2;
                    
                    // Update diagnostics
                    agb_event_count++;
                    agb_total_metals_g += MZ_g;
                    agb_total_energy_erg += E_erg;
                    
                    // Add to domain-local diagnostics
                    diag_add_agb(E_erg);
                }
            }
        }
        
    } // End of star_indices loop

    // ====================================================================
    // 3) Print diagnostics
    // ====================================================================
    static int feedback_call_count = 0;
    feedback_call_count++;

    if(feedback_call_count % 10 == 0 && All.ThisTask == 0) {
        FB_PRINT("\n[AGB_STATS|T=0|Step=%d|a=%.6f z=%.3f]\n", 
              feedback_call_count, All.Time, 1.0/All.Time - 1.0);
        FB_PRINT(" Stars checked: %d, Eligible for AGB: %d\n",
              agb_stars_checked, agb_stars_eligible);
        FB_PRINT(" AGB events: %d\n", agb_event_count);
        FB_PRINT(" Particles heated this step: %d / %d (%.1f%% of limit)\n",
          particles_heated_this_step, MAX_PARTICLES_HEATED_PER_STEP,
          100.0 * particles_heated_this_step / MAX_PARTICLES_HEATED_PER_STEP);
        

        if(agb_event_count > 0) {
            FB_PRINT(" AGB Total metals: %.3e Msun\n", agb_total_metals_g / SOLAR_MASS);
            FB_PRINT(" AGB Total energy: %.3e erg\n", agb_total_energy_erg);
            FB_PRINT(" AGB Avg metal/event: %.4f Msun\n", 
                  (agb_total_metals_g / SOLAR_MASS) / agb_event_count);
        }
        
        FB_PRINT(" Stars with reservoirs: %d / %d (%.1f%%)\n",
              stars_with_reservoir, n_stars_processed,
              100.0 * stars_with_reservoir / std::max(1, n_stars_processed));
        
        if(stars_with_reservoir > 0) {
            double avg_reservoir = reservoir_total_energy / stars_with_reservoir;
            FB_PRINT(" Reservoir total energy: %.3e (code units)\n", reservoir_total_energy);
            FB_PRINT(" Reservoir average: %.3e (code units)\n", avg_reservoir);
            FB_PRINT(" Reservoir max: %.3e (code units)\n", reservoir_max_size);
            FB_PRINT(" Reservoir release attempts: %d\n", reservoir_release_attempts);
            FB_PRINT(" Reservoir release successes: %d (%.1f%%)\n", 
                  reservoir_release_successes,
                  100.0 * reservoir_release_successes / std::max(1, reservoir_release_attempts));
        } else {
            FB_PRINT("  No non-empty reservoirs\n");
        }
        
        static int last_printed_count = 0;
        // Only print if lookups increased by at least 100
        if(AGB_Yields.get_lookup_count() - last_printed_count >= 100) {
            AGB_Yields.print_diagnostics();
            last_printed_count = AGB_Yields.get_lookup_count();
        }
        printf("\n");
    }

    double t_end = MPI_Wtime();

}

// -------------------------- RNG ---------------------------------------------
double get_random_number(unsigned long long id)
{
  // Simple LCG + Park-Miller RNG seeded by particle ID and timestep
  unsigned int seed = (unsigned int)(id ^ (unsigned long long)(All.NumCurrentTiStep*2654435761u));
  int ia=16807, im=2147483647, iq=127773, ir=2836; 
  int k = seed/iq; 
  int temp = ia*(seed - k*iq) - ir*k; 
  if(temp<0) temp+=im; 
  return temp/(double)im;
}

#endif // FEEDBACK