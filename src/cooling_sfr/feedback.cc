/* ============================================================================
 * feedback.cc — Stellar feedback with Type II SNe and AGB winds
 *
 * Features:
 *   - Cosmology-aware stellar ages using Friedmann t(a) lookup table
 *   - ΔT-target stochastic heating (EAGLE-like)
 *   - Energy reservoir system with intelligent release threshold
 *   - SPH kernel support with 2h neighbor search via spatial binning
 *   - Metal enrichment from Type II SNe (3-40 Myr)
 *   - Metal enrichment from AGB stars (>100 Myr)
 *   - Single-event SNII firing at window entry (3-40 Myr)
 *
 * Build-time toggles (set in Config.sh):
 *   FEEDBACK_LIMIT_DULOG   - Clamp |Δlog10 u| per heating event
 *   FEEDBACK_T_CAP         - Soft temperature cap for debugging
 *
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

// ------------------------------- Constants ----------------------------------
static const double ESN_ERG                  = 1.0e51;       // SN energy in erg
static const double NSNE_PER_MSUN_VAL        = 0.011;        // SNe per solar mass; Kroupa IMF, 8-100 Msun
static const double METAL_YIELD_PER_SN_MSUN  = 2.0;          // Metal yield per Type II SN in Msun
static const double AGB_METAL_YIELD_PER_MSUN = 5.495072e-03; // Integrated over Kroupa IMF using MESA AGB yields from Huscher et al 2025, 1-8 Msun, Z=0.0001-0.03
static const double AGB_E_ERG                = 1.0e47;       // Energy per Msun of metals ejected (20 km/s winds)

// ΔT-target heating parameters
static const double DELTA_T_TARGET        = 1e7;   // Target heating temperature (K)
static const int    HEAT_N_MIN            = 32;
static const int    HEAT_N_MAX            = 256;
static const int    MAX_KERNEL_NEIGHBORS  = 512;

#ifdef FEEDBACK_LIMIT_DULOG
static const double FEEDBACK_MAX_DULOG    = 1.5;  // Limit |Δlog10 u| per event
#endif
#ifdef FEEDBACK_T_CAP
static const double FEEDBACK_T_CAP_K      = 1.5e7;  // Debug soft cap (K)
#endif

// Per-window diagnostic counters (reset each flush cadence via agb_diag_reset())
static int    agb_event_count     = 0;
static double agb_total_metals_g  = 0.0;
static double agb_total_energy_erg= 0.0;
static int    agb_stars_checked   = 0;
static int    agb_stars_eligible  = 0;

static void agb_diag_reset()
{
  agb_event_count      = 0;
  agb_total_metals_g   = 0.0;
  agb_total_energy_erg = 0.0;
  agb_stars_checked    = 0;
  agb_stars_eligible   = 0;
}

// Reservoir diagnostics (reset each call to apply_stellar_feedback)
static int    reservoir_release_attempts  = 0;
static int    reservoir_release_successes = 0;
static double reservoir_max_size          = 0.0;
static int    stars_with_reservoir        = 0;
static double reservoir_total_energy      = 0.0;

// Global spatial hash instance
spatial_hash_zoom gas_hash;
spatial_hash_zoom star_hash;
spatial_hash_zoom dust_hash;

#define FB_PRINT(...) do{ if(All.FeedbackDebugLevel){ \
  printf("[FEEDBACK|T=%d|a=%.6g z=%.3f] ", All.ThisTask, (double)All.Time, 1.0/All.Time-1.0); \
  printf(__VA_ARGS__); } }while(0)

// ---------------------------- Diagnostics -----------------------------------
template<typename T> static inline T fb_clamp(T v, T lo, T hi){ return v<lo?lo:(v>hi?hi:v); }

FeedbackDiagLocal FbDiag;

// Hot-path helpers (NO MPI here)
static inline void diag_add_sn(double E_erg)    { FbDiag.n_SNII++; FbDiag.E_SN_erg += E_erg; }
static inline void diag_add_agb(double E_erg)   { FbDiag.n_AGB++;  FbDiag.E_AGB_erg += E_erg; }
static inline void diag_add_Edep(double v)      { FbDiag.E_deposited_erg += v; }
static inline void diag_add_EtoRes(double v)    { FbDiag.E_to_reservoir_erg += v; }
static inline void diag_add_EfromRes(double v)  { FbDiag.E_from_reservoir_erg += v; }
static inline void diag_track_dulog(double a)   { if(a > FbDiag.max_abs_dulog) FbDiag.max_abs_dulog = a; }


// ============================================================================
// rebuild_feedback_spatial_hash()
//
// Builds three spatial hashes — gas, star, and dust — used by both the
// stellar feedback and dust physics modules for neighbour finding.
//
// ── REBUILD CADENCE ──────────────────────────────────────────────────────────
//
// All tasks must agree whether to rebuild (task 0 decides, then broadcasts)
// to prevent MPI hangs from tasks diverging. Rebuilds whenever:
//   - The hash has never been built (first call)
//   - More than REBUILD_EVERY_DLOGA log-scale factor has elapsed since last rebuild
//
// REBUILD_EVERY_DLOGA = 0.002 corresponds roughly to ~100 Myr at z~2,
// balancing hash staleness against the cost of three full rebuilds per call.
//
// ── HASH ORDER ───────────────────────────────────────────────────────────────
//
// Gas hash is always built first. Its bbox (the zoom-region extent derived
// from gas positions) is then copied to the dust hash before the dust hash
// is built. This is essential — see dust bbox note below.
//
// ── DUST BBOX ────────────────────────────────────────────────────────────────
//
// Dust superparticles are ejected by radiation pressure and SN kicks and can
// travel far beyond the zoom region over time. If the dust bbox is computed
// from dust positions (detect_extent_collective on type 6), escaped grains
// inflate the bbox to > 100% of the box volume, producing cell sizes of
// ~100 kpc and physically meaningless nearest-gas searches.
//
// Solution: force the dust hash to use the gas hash bbox. Gas traces the
// actual zoom region extent; dust that has escaped the gas distribution is
// physically decoupled and would fail to find a gas neighbor anyway (the
// gas hash search for that grain will return -1, which is the correct result).
//
// To achieve this without a second MPI bbox reduction, build() is called
// with preset_bbox=true after copying gas_hash.bbox_* into dust_hash.
// The dust hash then only populates cells — it does not recompute the bbox.
// zoom_mass_threshold is set to 1e30 (no mass filter) since all dust
// particles are in the zoom region by construction.
//
// ── CELL SIZE OVERRIDES ───────────────────────────────────────────────────────
//
// Per-call overrides (max_cells_override) cap grid resolution independently
// of the global MAX_CELLS_PER_DIM backstop. Recommended values at 2048³:
//   gas hash:  768  → cell_size ~ 13 kpc, appropriate for feedback radii
//   star hash: 768  → same reasoning as gas
//   dust hash: 512  → dust search radii are shorter; fewer cells needed
//
// ── TIMING ───────────────────────────────────────────────────────────────────
//
// [HASH_TIMING] is printed after each rebuild with the per-rebuild wall time
// and running average. gas_hash.print_stats() follows, showing the grid
// dimensions, allocated cell count, and max/avg occupancy.
// ============================================================================

void rebuild_feedback_spatial_hash(simparticles *Sp, double dust_search_radius, MPI_Comm comm)
{
  static double total_rebuild_time = 0.0;
  static int    rebuild_count      = 0;

  // Rebuild when scale factor has advanced by more than this since last build.
  // ~0.002 in log-a corresponds to ~100 Myr at z~2, ~50 Myr at z~5.
  static constexpr double REBUILD_EVERY_DLOGA = 0.002;
  static double last_rebuild_a = -1.0;

  // ── Collective rebuild decision ───────────────────────────────────────────
  // Task 0 decides; all others follow. Prevents MPI hangs from divergence.
  int need_rebuild = 0;
  if(All.ThisTask == 0) {
    if(!gas_hash.is_built || last_rebuild_a < 0.0 ||
       (All.Time - last_rebuild_a) >= REBUILD_EVERY_DLOGA)
      need_rebuild = 1;
  }
  MPI_Bcast(&need_rebuild, 1, MPI_INT, 0, comm);

  if(!need_rebuild) return;

  double t_start = MPI_Wtime();

  // ── Step 1: Gas hash ──────────────────────────────────────────────────────
  // Use the dust search radius as the cell sizing target — it's the smallest
  // search radius used by any module, so it's the binding constraint.
  // This is resolution-independent: at coarser resolution dust_search_radius
  // will be larger, naturally producing fewer cells; at finer resolution it
  // will be smaller, naturally requesting more cells (capped by the override).
  gas_hash.build(Sp, dust_search_radius, All.SofteningTable[0], comm, 0, 768);

  // ── Step 2: Star hash ─────────────────────────────────────────────────
  // Guard: skip if no stars exist yet (e.g. z > 15). Building with zero
  // particles produces a degenerate 8^3 grid with cell_size=0 which causes
  // MPI_ERR_TRUNCATE in subsequent collectives on split communicators.
  int n_stars_local = 0;
  for(int i = 0; i < Sp->NumPart; i++)
      if(Sp->P[i].getType() == 4) n_stars_local++;
  int n_stars_global = 0;
  MPI_Allreduce(&n_stars_local, &n_stars_global, 1, MPI_INT, MPI_SUM, comm);

  if(n_stars_global > 0)
      star_hash.build(Sp, dust_search_radius, All.SofteningTable[4], comm, 4, 768);
  else
      star_hash.is_built = false;  // explicitly mark as not built

  // ── Step 3: Dust hash — inherits bbox from gas hash ───────────────────────
  //
  // This can be a key place that causes hangs and crashes if not handled carefully, 
  // note the MPI_Allreduce in here.
  //
  // Guard: skip if no dust exists yet. Building with zero particles produces
  // a degenerate grid (same problem as the star hash guard above) which causes
  // MPI_ERR_TRUNCATE on split communicators.
  //
  // When dust does exist, we force the dust hash to use the gas hash bbox
  // rather than computing it from dust positions. Radiation-pressure and
  // SN-kicked grains can scatter across the full box, inflating the bbox to
  // >> 100% of box volume and producing cell sizes of ~100 kpc. At that
  // resolution, dust-gas coupling, sputtering, and growth are physically
  // meaningless (grains couple to gas particles 50+ kpc away).
  //
  // Solution: copy the gas hash bbox (which correctly traces the zoom region)
  // into dust_hash, then build cells only — skip detect_extent_collective.
  // Dust that has escaped the gas distribution will not find a gas neighbor
  // via the hash (correct: decoupled grains should return -1).
  //
  // zoom_mass_threshold = 1e30: all dust is in the zoom region by construction
  // (spawned from stars inside the zoom volume), so no mass filter is needed.
  // build() is called with preset_bbox=true to skip detect_extent_collective.

  int n_dust_local = 0;
  for(int i = 0; i < Sp->NumPart; i++)
      if(Sp->P[i].getType() == 6) n_dust_local++;
  int n_dust_global = 0;
  MPI_Allreduce(&n_dust_local, &n_dust_global, 1, MPI_INT, MPI_SUM, comm);

  if(n_dust_global > 0) {
      for(int d = 0; d < 3; d++) {
          dust_hash.bbox_min[d]  = gas_hash.bbox_min[d];
          dust_hash.bbox_max[d]  = gas_hash.bbox_max[d];
          dust_hash.bbox_size[d] = gas_hash.bbox_size[d];
      }
      dust_hash.zoom_mass_threshold = 1e30;
      dust_hash.build(Sp, dust_search_radius, All.SofteningTable[6],
                      comm, 6, 512, /*preset_bbox=*/true);
  } else {
      dust_hash.is_built = false;
  }

  double t_end = MPI_Wtime();
  total_rebuild_time += (t_end - t_start);
  rebuild_count++;
  last_rebuild_a = All.Time;

  if(All.ThisTask == 0) {
    printf("[HASH_TIMING] Rebuild #%d took %.3f sec, avg %.3f sec\n",
           rebuild_count, t_end - t_start, total_rebuild_time / rebuild_count);
    gas_hash.print_stats();
  }
}

// ----------------------- Poisson Sampler ------------------------------------
// NOT CURRENTLY USING THIS, BUT WAS EXPERIMENTING WITH POISSON FIRING FOR SNII INSTEAD OF DETERMINISTIC AVERAGE RATES. 
// STILL KEEPING THIS FUNCTION IN CASE WE WANT TO REVISIT POISSON FIRING IN THE FUTURE.
//
// Draw a random integer from a Poisson distribution with mean lambda.
// Uses Knuth's algorithm for lambda < 30 (exact), and a normal approximation
// with continuity correction for lambda >= 30 (fast, accurate to <1%).
// This is used for stochastic SNII firing — see apply_stellar_feedback().
//
static int poisson_draw(double lambda, unsigned long long seed)
{
  if(lambda <= 0.0) return 0;

  if(lambda < 30.0) {
    // Knuth's algorithm: exact Poisson sampling
    double L = exp(-lambda);
    double p = 1.0;
    int k = 0;
    do {
      k++;
      p *= get_random_number(seed + k);
    } while(p > L);
    return k - 1;
  } else {
    // Normal approximation: N(lambda, sqrt(lambda))
    // Box-Muller transform for Gaussian deviate
    double u1 = get_random_number(seed);
    double u2 = get_random_number(seed + 1);
    if(u1 < 1e-300) u1 = 1e-300;
    double z = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    int result = (int)floor(lambda + sqrt(lambda) * z + 0.5);
    return (result < 0) ? 0 : result;
  }
}

void feedback_diag_try_flush(MPI_Comm comm, int cadence)
{
  if(cadence <= 0) return;
  if((All.NumCurrentTiStep % cadence) != 0) return;

  struct Local {
    long long n_SNII, n_AGB;
    double E_SN, E_AGB, E_dep, E_to_res, E_from_res, max_dulog;
  } in {
    FbDiag.n_SNII, FbDiag.n_AGB,
    FbDiag.E_SN_erg, FbDiag.E_AGB_erg,
    FbDiag.E_deposited_erg, FbDiag.E_to_reservoir_erg, FbDiag.E_from_reservoir_erg,
    FbDiag.max_abs_dulog
  }, out{};

  MPI_Reduce(&in.n_SNII,     &out.n_SNII,     1, MPI_LONG_LONG, MPI_SUM, 0, comm);
  MPI_Reduce(&in.n_AGB,      &out.n_AGB,      1, MPI_LONG_LONG, MPI_SUM, 0, comm);
  MPI_Reduce(&in.E_SN,       &out.E_SN,       1, MPI_DOUBLE,    MPI_SUM, 0, comm);
  MPI_Reduce(&in.E_AGB,      &out.E_AGB,      1, MPI_DOUBLE,    MPI_SUM, 0, comm);
  MPI_Reduce(&in.E_dep,      &out.E_dep,      1, MPI_DOUBLE,    MPI_SUM, 0, comm);
  MPI_Reduce(&in.E_to_res,   &out.E_to_res,   1, MPI_DOUBLE,    MPI_SUM, 0, comm);
  MPI_Reduce(&in.E_from_res, &out.E_from_res, 1, MPI_DOUBLE,    MPI_SUM, 0, comm);
  MPI_Reduce(&in.max_dulog,  &out.max_dulog,  1, MPI_DOUBLE,    MPI_MAX, 0, comm);

  if(All.ThisTask == 0) {
    if(out.n_SNII || out.n_AGB || out.E_dep || out.E_from_res) {
      FB_PRINT("events: SNII=%lld (%.3e erg)  AGB=%lld (%.3e erg)\n",
               out.n_SNII, out.E_SN, out.n_AGB, out.E_AGB);
      FB_PRINT("energy: deposited=%.3e erg  to_reservoir=%.3e erg  from_reservoir=%.3e erg\n",
               out.E_dep, out.E_to_res, out.E_from_res);
      if(out.max_dulog > 0.0)
        FB_PRINT("        max|Δlog10 u|=%.3f\n", out.max_dulog);
    }
  }

  FbDiag.reset();
  agb_diag_reset();  // Reset per-window AGB counters alongside FbDiag
}


// ----------------------- Stellar Age Calculation ----------------------------
//
// All.Time is the scale factor a in comoving runs. P[i].StellarAge stores
// a_birth, so age = t(a_now) - t(a_birth) via Friedmann integration.
//
// Driftfac.get_drift_factor() is NOT cosmic time — it is a drift integral.
//

static bool   cosmic_time_table_built = false;
static int    cosmic_time_N           = 0;
static double cosmic_time_a_min       = 0.0;
static double cosmic_time_a_max       = 1.0;
static std::vector<double> cosmic_time_tGyr;

static inline double E_of_a(double a)
{
  if(a <= 0.0) return 1e30;
  const double Om = All.Omega0;
  const double Ol = All.OmegaLambda;
  const double Ok = 1.0 - Om - Ol;
  return sqrt(Om/(a*a*a) + Ok/(a*a) + Ol);
}

static inline double H0_in_invGyr()
{
  const double km_s_Mpc_to_inv_s = 3.240779289e-20;
  const double sec_per_Gyr       = 3.15576e16;
  return 100.0 * All.HubbleParam * km_s_Mpc_to_inv_s * sec_per_Gyr;
}

static void build_cosmic_time_table()
{
  if(!All.ComovingIntegrationOn) {
    cosmic_time_table_built = false;
    return;
  }

  cosmic_time_a_min = All.TimeBegin;
  cosmic_time_a_max = 1.0;
  cosmic_time_N     = 2000;
  cosmic_time_tGyr.assign(cosmic_time_N, 0.0);

  const double H0Gyr = H0_in_invGyr();
  const double da    = (cosmic_time_a_max - cosmic_time_a_min) / (cosmic_time_N - 1);

  double t = 0.0;
  cosmic_time_tGyr[0] = t;
  for(int i = 1; i < cosmic_time_N; i++) {
    double a0 = cosmic_time_a_min + (i - 1) * da;
    double a1 = cosmic_time_a_min + i * da;
    double f0 = 1.0 / (a0 * H0Gyr * E_of_a(a0));
    double f1 = 1.0 / (a1 * H0Gyr * E_of_a(a1));
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

static inline double cosmic_time_Gyr_from_table(double a)
{
  if(!cosmic_time_table_built || cosmic_time_N < 2) return 0.0;
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

double get_stellar_age_Myr(double a_birth, double /*unused*/)
{
  if(a_birth <= 0.0 || a_birth > 1.0) return 0.0;

  double a_now = All.Time;
  if(a_birth >= a_now) return 0.0;

  if(All.ComovingIntegrationOn) {
    if(!cosmic_time_table_built) build_cosmic_time_table();
    double age_Myr = (cosmic_time_Gyr_from_table(a_now) - cosmic_time_Gyr_from_table(a_birth)) * 1000.0;
    return (age_Myr > 0.0) ? age_Myr : 0.0;
  }

  // Non-comoving: All.Time is already a time variable in code units
  double dt_code = a_now - a_birth;
  if(dt_code <= 0.0) return 0.0;
  double unit_time_s = All.UnitLength_in_cm / All.UnitVelocity_in_cm_per_s;
  double age_Myr = (dt_code * unit_time_s) / (SEC_PER_YEAR * 1.0e6);
  return (age_Myr > 0.0) ? age_Myr : 0.0;
}


// ----------------------- Neighbor Finding -----------------------------------

double get_local_smoothing_length_tree(simparticles *Sp, ngbtree * /*Tree*/, int star_idx)
{
  double h = 0.0;

  if(Sp->P[star_idx].getType() == 0 && Sp->SphP[star_idx].Hsml > 0) {
    h = Sp->SphP[star_idx].Hsml;
  } else if(Sp->P[star_idx].getType() == 4) {
    double best_r2 = 1e300;
    for(int i = 0, scanned = 0; i < Sp->NumGas && scanned < 256; ++i, ++scanned) {
      double d[3];
      Sp->nearest_image_intpos_to_pos(Sp->P[i].IntPos, Sp->P[star_idx].IntPos, d);
      double r2 = d[0]*d[0] + d[1]*d[1] + d[2]*d[2];
      if(r2 < best_r2 && Sp->SphP[i].Hsml > 0) {
        best_r2 = r2;
        h = Sp->SphP[i].Hsml;
      }
    }
  }

  if(h <= 0.0) h = std::max(All.SofteningTable[0], 1e-6);
  return h;
}

void find_feedback_neighbors_tree(simparticles *Sp, ngbtree * /*Tree*/,
                                  domain<simparticles> * /*D*/,
                                  int star_idx,
                                  int *ngb_list, double *distances, int *n_ngb,
                                  double *smoothing_length, int max_ngb)
{
  *n_ngb = 0;
  double h = *smoothing_length;
  if(h <= 0.0) h = get_local_smoothing_length_tree(Sp, nullptr, star_idx);
  if(h <= 0.0) { *smoothing_length = 0.0; return; }

  if(!gas_hash.is_built) {
    FB_PRINT("ERROR: Spatial hash not built — this should not happen.\n");
    *smoothing_length = h;
    return;
  }

  const double search_radius = 2.0 * h;
  gas_hash.find_neighbors(Sp, star_idx, search_radius, ngb_list, distances, n_ngb, max_ngb);
  *smoothing_length = h;
}

static void gather_neighbors(simparticles *Sp, ngbtree *Tree, domain<simparticles> *D,
                             int star_i, std::vector<int> &ngb,
                             std::vector<double> &dist, double &hsml)
{
  int n_found = 0;
  hsml = get_local_smoothing_length_tree(Sp, Tree, star_i);
  if(hsml <= 0) return;

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
  if(h <= 0) return 0.0;
  double q = fabs(r) / h;
  const double sig = 8.0 / (M_PI * h * h * h);
  if(q < 1.0) return sig * (1.0 - 1.5*q*q + 0.75*q*q*q);
  if(q < 2.0) { double t = 2.0 - q; return sig * 0.25 * t*t*t; }
  return 0.0;
}

// ----------------------- Thermodynamics -------------------------------------

static inline double mu_default(double T) { return (T > 1.5e4 ? 0.62 : 1.22); }
static inline double c_v(double mu)       { return 1.5 * BOLTZMANN / (mu * PROTONMASS); }

static inline double ucode_to_TK(double u_code)
{
  double u_cgs = u_code * All.UnitVelocity_in_cm_per_s * All.UnitVelocity_in_cm_per_s;
  return u_cgs / c_v(mu_default(1e4));
}

static inline double TK_to_ucode(double T_K)
{
  double u_cgs = c_v(mu_default(T_K)) * T_K;
  return u_cgs / (All.UnitVelocity_in_cm_per_s * All.UnitVelocity_in_cm_per_s);
}

// ----------------------- Energy Deposition ----------------------------------

/*
 * Deposit energy to neighbors using stochastic kernel-weighted heating.
 *
 * Algorithm:
 *   1. Nheat = clamp(N_ngb/8, HEAT_N_MIN, HEAT_N_MAX)
 *   2. Select Nheat neighbors stochastically by kernel weight
 *   3. Distribute energy and metals proportional to kernel-weighted mass
 *   4. Apply temperature and Δlog10(u) caps
 *
 * Returns: total energy deposited (code units)
 */
static double deposit_energy_stochastic(simparticles *Sp,
                                        const std::vector<int>   &ngb_list,
                                        const std::vector<double> &distances,
                                        double hsml, int star_idx,
                                        double E_avail_code, double MZ_code,
                                        int &particles_heated_global,
                                        int  max_particles_allowed)
{
  if(ngb_list.empty() || E_avail_code <= 0) return 0.0;

  // Per-step tracking to prevent the same gas particle being heated by multiple
  // nearby stars in a single timestep, which causes runaway temperatures.
  static std::set<int>  heated_this_step;
  static long long      last_step = -1;
  if(All.NumCurrentTiStep != last_step) {
    heated_this_step.clear();
    last_step = All.NumCurrentTiStep;
  }

  int Nheat = fb_clamp((int)(ngb_list.size() / 8), HEAT_N_MIN, HEAT_N_MAX);
  if(Nheat > (int)ngb_list.size()) Nheat = ngb_list.size();
  if(Nheat < 1) Nheat = 1;

  // Kernel weights → cumulative CDF for stochastic selection
  std::vector<double> weights(ngb_list.size());
  double weight_sum = 0.0;
  for(size_t i = 0; i < ngb_list.size(); ++i) {
    weights[i]  = cubic_spline_kernel(distances[i], hsml);
    weight_sum += weights[i];
  }
  if(weight_sum <= 0) return 0.0;

  std::vector<double> cumul(ngb_list.size());
  cumul[0] = weights[0] / weight_sum;
  for(size_t i = 1; i < ngb_list.size(); ++i)
    cumul[i] = cumul[i-1] + weights[i] / weight_sum;

  // Stochastic selection of Nheat targets
  std::vector<int> selected;
  selected.reserve(Nheat);
  for(int k = 0; k < Nheat; ++k) {
    double rnd = get_random_number(Sp->P[star_idx].ID.get() + k + All.NumCurrentTiStep);
    for(size_t i = 0; i < cumul.size(); ++i) {
      if(rnd <= cumul[i]) {
        bool already = false;
        for(int s : selected) if(s == (int)i) { already = true; break; }
        if(!already) selected.push_back(i);
        break;
      }
    }
  }
  if(selected.empty()) return 0.0;

  // Total kernel-weighted mass for normalization
  double total_kernel_weighted_mass = 0.0;
  for(int idx : selected) {
    int j = ngb_list[idx];
    total_kernel_weighted_mass += Sp->P[j].getMass() * cubic_spline_kernel(distances[idx], hsml);
  }
  if(total_kernel_weighted_mass <= 0) return 0.0;

  double E_used_code = 0.0;

  for(int idx : selected) {
    int j = ngb_list[idx];

    if(heated_this_step.count(j) > 0)
      continue;

    if(particles_heated_global >= max_particles_allowed)
      break;

    double m_g      = Sp->P[j].getMass();
    double w        = cubic_spline_kernel(distances[idx], hsml);
    double fraction = (m_g * w) / total_kernel_weighted_mass;

    // Deposit metals unconditionally — before any skip checks
    // Add ejected metal mass to the gas particle — this increases both the
    // total gas mass and the metal mass!
    #ifdef METALS
    if(MZ_code > 0) {
      double metal_mass_code = MZ_code * fraction;
      double new_gas_mass = Sp->P[j].getMass() + metal_mass_code;
      Sp->P[j].setMass(new_gas_mass);
      Sp->SphP[j].MassMetallicity += metal_mass_code;
      Sp->SphP[j].Metallicity = Sp->SphP[j].MassMetallicity / new_gas_mass;
    }
    #endif

    double dE_code  = E_avail_code * fraction;
    double du_cgs   = (dE_code * All.UnitEnergy_in_cgs) / (m_g * All.UnitMass_in_g);
    double du_code  = du_cgs / (All.UnitVelocity_in_cm_per_s * All.UnitVelocity_in_cm_per_s);

    double u_old = Sp->get_utherm_from_entropy(j);
    if(u_old <= 0) {
      FB_PRINT("WARNING: Particle %d has u_old=%g, skipping\n", j, u_old);
      continue;
    }

    double u_new = u_old + du_code;



    #ifdef FEEDBACK_LIMIT_DULOG
        double du_log10 = log10(u_new) - log10(u_old);
        if(fabs(du_log10) > FEEDBACK_MAX_DULOG) {
          double clamped = fb_clamp(du_log10, -FEEDBACK_MAX_DULOG, FEEDBACK_MAX_DULOG);
          u_new = u_old * pow(10.0, clamped);
          diag_track_dulog(fabs(clamped));
        }
    #endif

    #ifdef FEEDBACK_T_CAP
        {
          double T_new = ucode_to_TK(u_new);
          if(T_new > FEEDBACK_T_CAP_K) {
            double u_cap = TK_to_ucode(FEEDBACK_T_CAP_K);
            if(u_cap < u_new) u_new = u_cap;
          }
        }
    #endif

    const double u_max_particle = TK_to_ucode(1.0e8);
    if(u_new > u_max_particle) u_new = u_max_particle;

    Sp->set_entropy_from_utherm(u_new, j);
    set_thermodynamic_variables_safe(Sp, j);

    // Verify the cap survived entropy round-trip
    double u_verify = Sp->get_utherm_from_entropy(j);
#ifdef FEEDBACK_T_CAP
    double u_cap_verify = TK_to_ucode(FEEDBACK_T_CAP_K);
#else
    double u_cap_verify = u_max_particle;
#endif
    if(u_verify > u_cap_verify * 1.1) {
      Sp->set_entropy_from_utherm(u_cap_verify, j);
      set_thermodynamic_variables_safe(Sp, j);
      static int cap_failures = 0;
      if(cap_failures < 10 && All.ThisTask == 0) {
        printf("[FEEDBACK_CAP_FAIL] Particle %d: entropy roundtrip broke cap "
               "(u_capped=%.2e -> u_verify=%.2e), re-capped\n", j, u_new, u_verify);
        cap_failures++;
      }
    }

    heated_this_step.insert(j);
    particles_heated_global++;
    E_used_code += dE_code;
  }

  return E_used_code;
}

// -------------------- Feedback Event Handling -------------------------------

static void apply_feedback_event(simparticles *Sp, ngbtree *Tree,
                                 domain<simparticles> *D,
                                 int star_i, bool is_SNII,
                                 double E_code, double MZ_code,
                                 int &particles_heated_global,
                                 int  max_particles_allowed)
{
  std::vector<int>    ngb;
  std::vector<double> dist;
  double hsml = 0;
  gather_neighbors(Sp, Tree, D, star_i, ngb, dist, hsml);

  if(ngb.empty()) {
    // No neighbors — bank all energy for later
    Sp->P[star_i].EnergyReservoir += E_code;
    if(is_SNII) diag_add_sn(E_code * All.UnitEnergy_in_cgs);
    else        diag_add_agb(E_code * All.UnitEnergy_in_cgs);
    diag_add_EtoRes(E_code * All.UnitEnergy_in_cgs);
    return;
  }

  double E_used_code = deposit_energy_stochastic(Sp, ngb, dist, hsml,
                                                 star_i, E_code, MZ_code,
                                                 particles_heated_global,
                                                 max_particles_allowed);

  if(is_SNII) diag_add_sn(E_code * All.UnitEnergy_in_cgs);
  else        diag_add_agb(E_code * All.UnitEnergy_in_cgs);

  diag_add_Edep(E_used_code * All.UnitEnergy_in_cgs);
  double leftover = std::max(0.0, E_code - E_used_code);
  diag_add_EtoRes(leftover * All.UnitEnergy_in_cgs);

  Sp->P[star_i].EnergyReservoir += leftover;

  if(leftover > 0 && All.FeedbackDebugLevel >= 2)
    printf("[RESERVOIR_ADD|T=%d] Star %d: added %.2e to reservoir (total %.2e)\n",
           All.ThisTask, star_i, leftover, Sp->P[star_i].EnergyReservoir);
}

/* This function attempts to release energy from a star's reservoir. 
It was originally intended to diagnose trouble with feedback imparting too much energy
too quickly and crashing the simulation, but it may be somewhat extraneous now. */
static void try_release_reservoir(simparticles *Sp, ngbtree *Tree,
                                  domain<simparticles> *D, int star_i,
                                  int &particles_heated_global,
                                  int  max_particles_allowed)
{
  double E_code = Sp->P[star_i].EnergyReservoir;
  if(E_code <= 0) return;

  const double E_floor_code = 1e46 / All.UnitEnergy_in_cgs;  // ~1e-7 code units
  if(E_code < E_floor_code) return;

  // Track diagnostics for the periodic FB_STATS printout
  reservoir_release_attempts++;
  if(E_code > reservoir_max_size) reservoir_max_size = E_code;

  // =========================================================================
  // Step 1: Find gas neighbors around this star.
  //
  // gather_neighbors() uses the spatial hash to find all gas particles within
  // 2×hsml of the star. If none are found the star is isolated (e.g. kicked
  // into a void by a previous feedback event), and we handle that separately.
  // =========================================================================
  std::vector<int>    ngb;
  std::vector<double> dist;
  double hsml = 0;
  gather_neighbors(Sp, Tree, D, star_i, ngb, dist, hsml);

  if(ngb.empty()) {
    // Isolated star: no neighbors to heat. Rather than letting the reservoir
    // grow without bound, slowly vent 2% per call. This energy is lost, but
    // an isolated star has no physical channel to deposit it anyway.
    // The "half a supernova" threshold prevents venting tiny numerical residuals.
    if(E_code > (0.5 * (1e51 / All.UnitEnergy_in_cgs))) {
      double vented = E_code * 0.02;
      Sp->P[star_i].EnergyReservoir -= vented;
      diag_add_EfromRes(vented * All.UnitEnergy_in_cgs);
      static int vent_count = 0;
      if(vent_count < 10 && All.ThisTask == 0)
        printf("[RESERVOIR_VENT] Star %d: isolated, venting %.2e (remaining %.2e)\n",
               star_i, vented, Sp->P[star_i].EnergyReservoir);
      vent_count++;
    }
    return;
  }

  // =========================================================================
  // Step 2: Compute E_need — the energy required to heat Nheat neighbors by
  // DELTA_T_TARGET (the ΔT-target heating scheme from EAGLE).
  //
  // This is computed FIRST because it is needed by both the overflow vent
  // check (Step 3) and the release threshold check (Step 4). Computing it
  // once here avoids the use-before-initialization bug that results from
  // placing the overflow check before this calculation.
  //
  // Nheat is the same clamp used in deposit_energy_stochastic so that
  // E_need represents the actual heating budget, not a hypothetical one.
  // =========================================================================
  int Nheat = fb_clamp((int)(ngb.size() / 8), HEAT_N_MIN, HEAT_N_MAX);
  if(Nheat < 1) Nheat = 1;

  // Sample the first Nheat neighbors to estimate a representative gas mass.
  // These are already sorted by distance from the star by the hash search,
  // so this is a reasonable local average.
  double total_mass = 0.0;
  int n_sample = std::min(Nheat, (int)ngb.size());
  for(int i = 0; i < n_sample; i++) total_mass += Sp->P[ngb[i]].getMass();
  double m_g_code = total_mass / n_sample;  // mean gas particle mass [code units]

  // E_need: total thermal energy to raise Nheat particles by DELTA_T_TARGET.
  // c_v(mu) = (3/2) k_B / (mu m_p), the specific heat at constant volume.
  // The factor (m_g_code × UnitMass) converts mass to cgs for the c_v product.
  double E_need = (c_v(mu_default(DELTA_T_TARGET)) * DELTA_T_TARGET *
                   (m_g_code * All.UnitMass_in_g) * Nheat) / All.UnitEnergy_in_cgs;

  // =========================================================================
  // Step 3: Overflow vent — prevent indefinite reservoir accumulation in
  // dense star-forming regions.
  //
  // Overflow vent — prevent indefinite reservoir accumulation for isolated
  // stars embedded in dense clumps where all neighbors hit MAX_PARTICLES_HEATED_PER_STEP.
  //
  // If the reservoir exceeds MAX_RESERVOIR_FACTOR × E_need, we vent 5% per
  // call directly to the largest-h (most diffuse) neighbor, bypassing the
  // Courant floor. The T_cap is still respected. This is a deliberate
  // numerical approximation: a slow continuous drain is physically and
  // numerically better than an eventual catastrophic dump.
  // =========================================================================
  const double E_overflow = 100.0 * (1e51 / All.UnitEnergy_in_cgs);
  if(E_code > E_overflow) {

    double vented = E_code * 0.05;  // 5% per call — slow enough to be stable
    Sp->P[star_i].EnergyReservoir -= vented;

    // Find the neighbor with the largest smoothing length (most diffuse gas).
    // Heating diffuse gas produces the longest post-heating Courant step and
    // therefore causes the least damage to the timebin hierarchy.
    int    best_j = ngb[0];
    double best_h = 0.0;
    for(int i = 0; i < (int)ngb.size(); i++) {
      if(Sp->SphP[ngb[i]].Hsml > best_h) {
        best_h = Sp->SphP[ngb[i]].Hsml;
        best_j = ngb[i];
      }
    }

    double u_old  = Sp->get_utherm_from_entropy(best_j);
    double du_cgs = (vented * All.UnitEnergy_in_cgs)
                    / (Sp->P[best_j].getMass() * All.UnitMass_in_g);
    double du     = du_cgs / (All.UnitVelocity_in_cm_per_s * All.UnitVelocity_in_cm_per_s);

    // Still respect the temperature cap even during forced venting
    #ifdef FEEDBACK_T_CAP
      double u_cap = TK_to_ucode(FEEDBACK_T_CAP_K);
    #else
      double u_cap = TK_to_ucode(1.0e8);  // same hard cap used elsewhere
    #endif
    double u_new = std::min(u_old + du, u_cap);

    Sp->set_entropy_from_utherm(u_new, best_j);
    set_thermodynamic_variables_safe(Sp, best_j);

    diag_add_EfromRes(vented * All.UnitEnergy_in_cgs);
    diag_add_Edep(vented * All.UnitEnergy_in_cgs);

    static long long overflow_vents = 0;
    overflow_vents++;
    if(overflow_vents % 500 == 0 && All.ThisTask == 0)
      printf("[RESERVOIR_OVERFLOW|T=%d|Step=%d] %lld overflow vents: "
             "star=%d E_res=%.2e E_need=%.2e ratio=%.1f "
             "vented_to=j=%d h=%.3f kpc T_after=%.2e K\n",
             All.ThisTask, All.NumCurrentTiStep, overflow_vents,
             star_i, E_code, E_need, E_code / E_need,
             best_j, best_h, ucode_to_TK(u_new));

    // Return after venting — don't also attempt a normal release this call.
    // This keeps the logic clean: each call either vents (overflow) or
    // does a normal stochastic release, never both.
    return;
  }

  reservoir_release_successes++;

  // =========================================================================
  // Step 4: Stochastic deposit.
  //
  // Release at most 2× E_need per call. This prevents a large accumulated
  // reservoir from dumping all its energy in a single step (which would heat
  // particles to extreme temperatures). The remainder stays in the reservoir
  // and is released on subsequent calls.
  //
  // deposit_energy_stochastic() handles the Courant-skip logic internally:
  // any energy it refuses to deposit (because the target particle's post-
  // heating Courant step would be too short) is NOT reflected in `used`,
  // so it naturally ends up back in the reservoir via the max(0, E_code-used)
  // line below. No double-banking occurs.
  // =========================================================================
  double E_to_release = std::min(E_code, E_need * 2.0);
  double used = deposit_energy_stochastic(Sp, ngb, dist, hsml, star_i,
                                          E_to_release, /*MZ_code=*/0.0,
                                          particles_heated_global,
                                          max_particles_allowed);

  diag_add_EfromRes(used * All.UnitEnergy_in_cgs);
  diag_add_Edep(used * All.UnitEnergy_in_cgs);

  // Update reservoir: subtract only what was actually deposited.
  // Courant-skipped energy (not in `used`) stays in the reservoir automatically.
  Sp->P[star_i].EnergyReservoir = std::max(0.0, E_code - used);
}


// ---------------------------- Public API ------------------------------------

void init_stellar_feedback(void)
{
#ifdef FEEDBACK
  // Print banner on task 0
  if(All.ThisTask == 0)
    FB_PRINT("FEEDBACK enabled; ΔT_target=%.2e K, Nheat=[%d,%d]\n",
         DELTA_T_TARGET, HEAT_N_MIN, HEAT_N_MAX);

  // All tasks load AGB yields (not just task 0)
  if(!AGB_Yields.load_from_file(All.AGByieldFile)) {
    if(All.ThisTask == 0) {
      printf("[FEEDBACK] ERROR: Failed to load AGB yields from '%s'\n", All.AGByieldFile);
      printf("[FEEDBACK] AGB feedback will be DISABLED\n");
    }
  } else {
    if(All.ThisTask == 0)
      printf("[FEEDBACK] AGB yields loaded from '%s'\n", All.AGByieldFile);
  }

  // Build cosmic time table on all tasks
  build_cosmic_time_table();
#endif
}

void apply_stellar_feedback(double /*current_time*/, simparticles *Sp,
                            ngbtree *Tree, domain<simparticles> *D, MPI_Comm comm)
{
  // Per-step rate limiting to prevent timebin cascades
  static int       particles_heated_this_step = 0;
  static long long last_heating_step          = -1;
  if(All.NumCurrentTiStep != last_heating_step) {
    particles_heated_this_step = 0;
    last_heating_step          = All.NumCurrentTiStep;
  }
  const int MAX_PARTICLES_HEATED_PER_STEP = Sp->NumGas / 100;  // 1% of local gas

  // Build local star index list
  static std::vector<int> star_indices;
  star_indices.clear();
  star_indices.reserve(Sp->NumPart / 10);
  for(int p = 0; p < Sp->NumPart; ++p)
    if(Sp->P[p].getType() == 4)
      star_indices.push_back(p);

  int n_stars_local = (int)star_indices.size();

  // Collective early exit: if no task has stars, skip everything
  int global_has_stars = 0;
  MPI_Allreduce(&n_stars_local, &global_has_stars, 1, MPI_INT, MPI_MAX, comm);
  if(!global_has_stars)
    return;

  double max_radius = 10.0;
  if(Sp->NumGas > 0) {
    double h_sum  = 0;
    int    h_count = 0;
    int sample_stride = std::max(1, Sp->NumGas / 100);
    for(int i = 0; i < Sp->NumGas && h_count < 100; i += sample_stride) {
        if(Sp->SphP[i].Hsml > 0) { h_sum += Sp->SphP[i].Hsml; h_count++; }
    }
    if(h_count > 0) max_radius = 2.5 * h_sum / h_count;
  }

  // max_radius is used for actual neighbor searches below, not for hash sizing.
  // Hash cell sizing uses dust_search_radius (0.1 kpc) — the finest search
  // any module performs — ensuring cells are small enough for all use cases.
  rebuild_feedback_spatial_hash(Sp, 0.1, comm);

  // Reset per-call reservoir diagnostics
  reservoir_release_attempts  = 0;
  reservoir_release_successes = 0;
  reservoir_max_size          = 0.0;
  stars_with_reservoir        = 0;
  reservoir_total_energy      = 0.0;

  // ====================================================================
  // 1) Release reservoir energy from stars that have it
  // ====================================================================
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
  // 2) Trigger new feedback events (SNII and AGB)
  // ====================================================================

  for(int p : star_indices) {
    double age_Myr = get_stellar_age_Myr(Sp->P[p].StellarAge, 0.0);

// --- 2a) Type II SNe — single-event firing at window entry ---
    //
    // Physical motivation: a stellar particle of mass M_star represents a
    // population of stars. The 0.011 × M_star / M_sun SNe fire over the
    // 3-40 Myr window set by the stellar IMF and mass-lifetime relation.
    //
    // Numerical implementation: all SNe from this stellar particle are fired
    // in a single event when the star first enters the 3-40 Myr window.
    // This is a deliberate simplification — physically, SNe fire continuously
    // over the window, but single-event firing was found to produce a stable
    // ISM at the resolution of these runs (512³-2048³) whereas distributed
    // stochastic firing increased the number of neighbor searches by ~2600×
    // per star, making the simulation prohibitively slow and producing
    // similar or worse ISM evacuation due to interactions with the
    // heated_this_step deduplication and MAX_PARTICLES_HEATED_PER_STEP
    // rate limiting. Stochastic firing is documented in the git history
    // and deferred to future work with explicit wind feedback.
    //
    // Algorithm:
    //   1. Star enters window (age >= MIN_TYPEII_TIME)
    //   2. Fire all N_SN = NSNE_PER_MSUN × M_star SNe at once
    //   3. Deposit total energy E = N_SN × ESN_ERG to neighbors
    //   4. Deposit total metals MZ = N_SN × METAL_YIELD_PER_SN to neighbors
    //   5. Set FeedbackFlag bit 0 — never fires again

   if(age_Myr >= MIN_TYPEII_TIME && age_Myr < MAX_TYPEII_TIME &&
   !(Sp->P[p].FeedbackFlag & 1))
{
    double m_star_g   = Sp->P[p].getMass() * All.UnitMass_in_g;
    double n_SN_total = NSNE_PER_MSUN_VAL * (m_star_g / SOLAR_MASS);
    double E_erg      = n_SN_total * ESN_ERG;
    double MZ_g       = n_SN_total * METAL_YIELD_PER_SN_MSUN * SOLAR_MASS;
    double MZ_code    = MZ_g / All.UnitMass_in_g;
    double E_code     = E_erg / All.UnitEnergy_in_cgs;

    apply_feedback_event(Sp, Tree, D, p, true, E_code, MZ_code,
                        particles_heated_this_step,
                        MAX_PARTICLES_HEATED_PER_STEP);

    // Debit ejected metal mass from star particle — mass conservation.
    // The metals are now in the gas phase; the star loses this mass.
    {
      double new_star_mass = Sp->P[p].getMass() - MZ_code;
      if(new_star_mass > 0.0) Sp->P[p].setMass(new_star_mass);
    }

    #ifdef DUST
    create_dust_particles_from_feedback(Sp, p, MZ_code, 1);
    destroy_dust_from_sn_shocks(Sp, p, E_code, MZ_code, comm);
    #endif

    Sp->P[p].FeedbackFlag |= 1;
}

  if(age_Myr >= MAX_TYPEII_TIME && !(Sp->P[p].FeedbackFlag & 1))
  {
      Sp->P[p].FeedbackFlag |= 1;
  }

    // --- 2b) AGB enrichment (>100 Myr, one-time per star) ---
    if(AGB_Yields.is_table_loaded() && age_Myr >= MIN_AGB_TIME &&
       !(Sp->P[p].FeedbackFlag & 2))
    {
      agb_stars_eligible++;

      double m_star_g   = Sp->P[p].getMass() * All.UnitMass_in_g;
      double m_star_msun = m_star_g / SOLAR_MASS;
      double MZ_msun    = AGB_METAL_YIELD_PER_MSUN * m_star_msun;

      if(MZ_msun > 0.0) {
        double MZ_code = (MZ_msun * SOLAR_MASS) / All.UnitMass_in_g;
        double E_erg   = MZ_msun * AGB_E_ERG;
        double E_code  = E_erg / All.UnitEnergy_in_cgs;

        apply_feedback_event(Sp, Tree, D, p, /*is_SNII=*/false, E_code, MZ_code,
                             particles_heated_this_step, MAX_PARTICLES_HEATED_PER_STEP);

        // Debit ejected metal mass from star particle — mass conservation.
        {
          double new_star_mass = Sp->P[p].getMass() - MZ_code;
          if(new_star_mass > 0.0) Sp->P[p].setMass(new_star_mass);
        }

        #ifdef DUST
            create_dust_particles_from_feedback(Sp, p, MZ_code, 2);
        #endif

        Sp->P[p].FeedbackFlag |= 2;

        agb_event_count++;
        agb_total_metals_g   += MZ_msun * SOLAR_MASS;
        agb_total_energy_erg += E_erg;
      }
    }

    // Track all stars checked for AGB eligibility
    if(AGB_Yields.is_table_loaded())
      agb_stars_checked++;

  }  // end star loop

  // ====================================================================
  // 3) Periodic diagnostics (task 0 only, guarded by FeedbackDebugLevel)
  // ====================================================================
  static int feedback_call_count = 0;
  feedback_call_count++;

  if(feedback_call_count % 10 == 0 && All.ThisTask == 0) {
    FB_PRINT("\n[FB_STATS|Step=%d|a=%.6f z=%.3f]\n",
             feedback_call_count, All.Time, 1.0 / All.Time - 1.0);
    FB_PRINT(" Stars local: %d | Particles heated: %d / %d (%.1f%% of limit)\n",
             n_stars_local, particles_heated_this_step, MAX_PARTICLES_HEATED_PER_STEP,
             100.0 * particles_heated_this_step / std::max(1, MAX_PARTICLES_HEATED_PER_STEP));
    FB_PRINT(" AGB (this window): checked=%d eligible=%d events=%d\n",
             agb_stars_checked, agb_stars_eligible, agb_event_count);
    if(agb_event_count > 0) {
      FB_PRINT("   metals=%.3e Msun  energy=%.3e erg  avg=%.4f Msun/event\n",
               agb_total_metals_g / SOLAR_MASS, agb_total_energy_erg,
               (agb_total_metals_g / SOLAR_MASS) / agb_event_count);
    }
    FB_PRINT(" Reservoirs: %d stars  total=%.3e  max=%.3e  "
             "release attempts=%d successes=%d\n",
             stars_with_reservoir, reservoir_total_energy, reservoir_max_size,
             reservoir_release_attempts, reservoir_release_successes);
  }
}

// -------------------------- RNG ---------------------------------------------
double get_random_number(unsigned long long id)
{
  unsigned int seed = (unsigned int)(id ^ (unsigned long long)(All.NumCurrentTiStep * 2654435761u));
  int ia = 16807, im = 2147483647, iq = 127773, ir = 2836;
  int k    = seed / iq;
  int temp = ia * (seed - k * iq) - ir * k;
  if(temp < 0) temp += im;
  return temp / (double)im;
}

#endif // FEEDBACK