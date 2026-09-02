/* ============================================================================
 * feedback.cc — Stellar feedback with Type II SNe and AGB winds
 *
 * Features:
 *   - Cosmology-aware stellar ages using Friedmann t(a) lookup table
 *   - ΔT-target stochastic heating (EAGLE-like)
 *   - Fallback energy reservoir for temporarily unprocessed feedback
 *   - SPH kernel support with 2h neighbor search via spatial binning
 *   - Metal enrichment from Type II SNe (3-40 Myr)
 *   - Metal enrichment from AGB stars (>100 Myr)
 *   - Time-distributed SNII/HN feedback across the 3-40 Myr window
 *   - SNII dust sampling distributed across temporal tranches without
 *     multiplying DustParticlesPerSNII by SNII_TIME_BINS
 *   - LRN dust retained as one full-SSP injection on the first SNII tranche
 *
 * ============================================================================ */

#include "gadgetconfig.h"
#ifdef FEEDBACK

#include <math.h>
#include <algorithm>
#include <vector>
#include <mpi.h>
#include <stdio.h>

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
#include "agb_yields.h"
#include "snii_yields.h"

#ifdef DUST
#include "../dust/dust.h"
#endif

constexpr int spatial_hash_config::MIN_CELLS_PER_DIM;
constexpr int spatial_hash_config::MAX_CELLS_PER_DIM;
constexpr int spatial_hash_config::TARGET_PARTICLES_PER_CELL;
constexpr double spatial_hash_config::CELL_SIZE_SAFETY_FACTOR;

// ------------------------------- Constants ----------------------------------
static const double AGB_E_ERG                = 1.0e47;  // Energy per Msun of metals ejected (20 km/s winds)
static const double CCSN_NUMBER_PER_MSUN     = 0.011;   // Number of CCSN progenitors per Msun formed; Kroupa IMF, 8-100 Msun. (needed for LRN-rate)

// ΔT-target heating parameters
static const double DELTA_T_TARGET        = 1.0e7; // 3.162277660168379e7;   // 10^7.5 K
static const int    MAX_KERNEL_NEIGHBORS  = 512;

// Adaptive feedback-kernel controls.
//
// Feedback begins with the smoothing length inherited from the nearest gas
// particle.  If that star-centered kernel contains very few gas particles,
// enlarge it until a minimally resolved local environment is obtained.
//
// We deliberately use a MINIMUM rather than forcing every event to exactly
// DesNumNgb.  Kernels that are already adequately populated are left alone.
static const int    MIN_FEEDBACK_NEIGHBORS       = 16;
static const double FEEDBACK_KERNEL_GROWTH        = 1.25;
static const double FEEDBACK_MAX_RADIUS_FACTOR    = 4.0;
static const double FEEDBACK_MAX_RADIUS_PKPC      = 20.0;
static const int    FEEDBACK_MAX_KERNEL_ITERS     = 16;

// SNII/HN temporal-distribution experiment.
//
// Instead of depositing the entire IMF-integrated CCSN/HN budget the first
// time an SSP enters the 3-40 Myr window, divide that budget into equal
// time tranches.  One overdue tranche is allowed to fire per feedback call.
//
// Bits 0 and 1 of FeedbackFlag retain their existing meanings:
//   bit 0 (value 1): all SNII/HN tranches complete
//   bit 1 (value 2): AGB feedback complete
//
// Bits 2..5 record the four individual SNII/HN tranches.
static const int    SNII_TIME_BINS             = 4;
static const int    SNII_FIRST_TRANCHE_BIT     = 2;

static inline unsigned int snii_tranche_mask(int ibin)
{
    return 1u << (SNII_FIRST_TRANCHE_BIT + ibin);
}

static inline double snii_tranche_start_Myr(int ibin)
{
    const double dt =
        (MAX_TYPEII_TIME - MIN_TYPEII_TIME) /
        (double)SNII_TIME_BINS;

    return MIN_TYPEII_TIME + ibin * dt;
}

static inline bool all_snii_tranches_complete(unsigned int flag)
{
    for(int ibin = 0; ibin < SNII_TIME_BINS; ibin++)
        if(!(flag & snii_tranche_mask(ibin)))
            return false;

    return true;
}

struct ElementYields
{
  double C  = 0.0;
  double N  = 0.0;
  double O  = 0.0;
  double Ne = 0.0;
  double Mg = 0.0;
  double Si = 0.0;
  double Fe = 0.0;
};

// Return the immutable initial SSP mass represented by a PartType4 particle.
// IMF-integrated SNII/HN, LRN, and AGB rates/yields are all normalized per
// unit initial stellar-population mass, not per unit surviving stellar mass.
static inline double stellar_birth_mass_msun(
    simparticles *Sp,
    int star_idx)
{
    const double birth_mass_code =
        (double)Sp->P[star_idx].StellarBirthMass;

    if(!std::isfinite(birth_mass_code) ||
       birth_mass_code <= 0.0)
    {
        printf(
            "[SSP_BIRTH_MASS_ERROR|T=%d] "
            "star=%d starID=%llu birth=%.17g current=%.17g\n",
            All.ThisTask,
            star_idx,
            (unsigned long long)Sp->P[star_idx].ID.get(),
            birth_mass_code,
            (double)Sp->P[star_idx].getMass());

        Terminate(
            "Invalid StellarBirthMass for feedback-eligible star");
    }

    return
        birth_mass_code *
        All.UnitMass_in_g /
        SOLAR_MASS;
}

#ifdef DUST

// Silicate elemental composition used throughout CosmicGrain.
// These must remain identical to the values used in dust.cc.
static constexpr double FB_SIL_FRAC_O  = 0.69876708;
static constexpr double FB_SIL_FRAC_MG = 0.07680773;
static constexpr double FB_SIL_FRAC_SI = 0.08097301;
static constexpr double FB_SIL_FRAC_FE = 0.14345218;

// Must match MIN_DUST_PARTICLE_MASS in dust.cc.
static constexpr double FB_MIN_STELLAR_DUST_MASS = 1e-10;

struct StellarDustPartition
{
    double carbon_dust   = 0.0;
    double silicate_dust = 0.0;
    double total_dust    = 0.0;
    double gas_metals    = 0.0;
};

// AGB condensation diagnostics. These are local-task counters and are
// reported with the existing per-window feedback diagnostics below.
static long long agb_dust_partition_count       = 0;
static long long agb_dust_carbon_limited_count  = 0;
static long long agb_dust_silicate_target_count = 0;
static long long agb_dust_silicate_O_count      = 0;
static long long agb_dust_silicate_Mg_count     = 0;
static long long agb_dust_silicate_proxy_count  = 0;
static long long agb_dust_below_min_count       = 0;


/*
 * Partition stellar metal ejecta into:
 *
 *   1. stellar-condensed dust
 *   2. remaining gas-phase metal ejecta
 *
 * elem_gas is passed in containing the TOTAL stellar elemental ejecta.
 * This function subtracts the material condensed into dust so that
 * elem_gas contains only the gas-phase remainder on return.
 *
 * For SNII/HN, silicate formation is explicitly constrained by the
 * available O/Mg/Si/Fe ejecta.
 *
 * The current AGB tables explicitly provide O and Mg but not Si and Fe.
 * For AGB events, use_element_resolved_silicate_limits=false therefore
 * constrains silicate formation by explicit O, explicit Mg, and the
 * untracked-metal residual used as a combined proxy for Si+Fe. This keeps
 * the sum of explicitly tracked gas-phase elements no larger than the
 * remaining total gas-phase metal mass.
 */
static StellarDustPartition partition_stellar_ejecta_for_dust(
    double MZ_total_code,
    ElementYields &elem_gas,
    double dust_yield_fraction,
    double target_carbon_fraction,
    bool use_element_resolved_silicate_limits)
{
    StellarDustPartition result;

    result.gas_metals = MZ_total_code;

    if(!All.DustEnableCreation)
        return result;

    if(MZ_total_code <= 0.0)
        return result;

    if(!std::isfinite(MZ_total_code) ||
       !std::isfinite(dust_yield_fraction) ||
       !std::isfinite(target_carbon_fraction))
    {
        Terminate(
            "Non-finite input to stellar ejecta dust partition");
    }

    double dust_fraction =
        std::max(0.0,
                 std::min(1.0, dust_yield_fraction));

    double target_CF =
        std::max(0.0,
                 std::min(1.0, target_carbon_fraction));

    double requested_total_dust =
        MZ_total_code * dust_fraction;

    if(requested_total_dust <
       FB_MIN_STELLAR_DUST_MASS)
    {
        return result;
    }

    if(use_element_resolved_silicate_limits)
    {
        // ------------------------------------------------------------
        // SNII/HN:
        // preserve the requested carbon/silicate birth composition,
        // but constrain each component by the actual stellar ejecta.
        // ------------------------------------------------------------

        double requested_carbon =
            requested_total_dust * target_CF;

        double requested_silicate =
            requested_total_dust * (1.0 - target_CF);

        result.carbon_dust =
            std::min(
                requested_carbon,
                std::max(0.0, elem_gas.C));

        double silicate_available = HUGE_VAL;

        if(FB_SIL_FRAC_O > 0.0)
            silicate_available =
                std::min(
                    silicate_available,
                    std::max(0.0, elem_gas.O) /
                    FB_SIL_FRAC_O);

        if(FB_SIL_FRAC_MG > 0.0)
            silicate_available =
                std::min(
                    silicate_available,
                    std::max(0.0, elem_gas.Mg) /
                    FB_SIL_FRAC_MG);

        if(FB_SIL_FRAC_SI > 0.0)
            silicate_available =
                std::min(
                    silicate_available,
                    std::max(0.0, elem_gas.Si) /
                    FB_SIL_FRAC_SI);

        if(FB_SIL_FRAC_FE > 0.0)
            silicate_available =
                std::min(
                    silicate_available,
                    std::max(0.0, elem_gas.Fe) /
                    FB_SIL_FRAC_FE);

        result.silicate_dust =
            std::min(
                requested_silicate,
                silicate_available);

        result.silicate_dust =
            std::max(
                0.0,
                result.silicate_dust);
    }
    else
    {
        // AGB birth composition is prescribed as
        // 60% carbonaceous / 40% silicate by dust mass.
        //
        // Carbon is explicitly constrained by the tracked C ejecta.
        // Silicate is constrained by explicit O, explicit Mg, and the
        // untracked-metal residual, which serves as a combined proxy for
        // the unavailable Si+Fe ejecta.

        double requested_carbon =
            requested_total_dust * target_CF;

        double requested_silicate =
            requested_total_dust * (1.0 - target_CF);

        result.carbon_dust =
            std::min(
                requested_carbon,
                std::max(0.0, elem_gas.C));

        agb_dust_partition_count++;

        if(result.carbon_dust < requested_carbon * (1.0 - 1e-12))
            agb_dust_carbon_limited_count++;

        const double tracked_metals =
            std::max(0.0, elem_gas.C)  +
            std::max(0.0, elem_gas.N)  +
            std::max(0.0, elem_gas.O)  +
            std::max(0.0, elem_gas.Ne) +
            std::max(0.0, elem_gas.Mg) +
            std::max(0.0, elem_gas.Si) +
            std::max(0.0, elem_gas.Fe);

        double untracked_metals =
            MZ_total_code - tracked_metals;

        if(untracked_metals < 0.0)
        {
            if(untracked_metals >
               -1e-10 * std::max(MZ_total_code, 1e-30))
                untracked_metals = 0.0;
            else
                Terminate(
                    "AGB tracked elemental ejecta exceed total metal ejecta");
        }

        const double silicate_from_O =
            (FB_SIL_FRAC_O > 0.0) ?
            std::max(0.0, elem_gas.O) / FB_SIL_FRAC_O :
            HUGE_VAL;

        const double silicate_from_Mg =
            (FB_SIL_FRAC_MG > 0.0) ?
            std::max(0.0, elem_gas.Mg) / FB_SIL_FRAC_MG :
            HUGE_VAL;

        const double proxy_fraction =
            FB_SIL_FRAC_SI + FB_SIL_FRAC_FE;

        const double silicate_from_proxy =
            (proxy_fraction > 0.0) ?
            std::max(0.0, untracked_metals) / proxy_fraction :
            HUGE_VAL;

        double silicate_available =
            std::min(
                silicate_from_O,
                std::min(
                    silicate_from_Mg,
                    silicate_from_proxy));

        result.silicate_dust =
            std::min(
                requested_silicate,
                std::max(0.0, silicate_available));

        if(requested_silicate <= silicate_available)
            agb_dust_silicate_target_count++;
        else if(silicate_from_O <= silicate_from_Mg &&
                silicate_from_O <= silicate_from_proxy)
            agb_dust_silicate_O_count++;
        else if(silicate_from_Mg <= silicate_from_proxy)
            agb_dust_silicate_Mg_count++;
        else
            agb_dust_silicate_proxy_count++;
    }

    result.total_dust =
        result.carbon_dust +
        result.silicate_dust;

    // The requested mass may exceed the numerical particle-mass floor while
    // the reservoir-limited mass does not. In that case leave all ejecta in
    // the gas phase rather than removing metals without spawning dust.
    if(result.total_dust > 0.0 &&
       result.total_dust < FB_MIN_STELLAR_DUST_MASS)
    {
        if(!use_element_resolved_silicate_limits)
            agb_dust_below_min_count++;

        result.carbon_dust   = 0.0;
        result.silicate_dust = 0.0;
        result.total_dust    = 0.0;
        result.gas_metals    = MZ_total_code;

        return result;
    }

    if(result.total_dust <= 0.0)
    {
        result.carbon_dust   = 0.0;
        result.silicate_dust = 0.0;
        result.total_dust    = 0.0;
        result.gas_metals    = MZ_total_code;

        return result;
    }

    if(result.total_dust >
       MZ_total_code * 1.000001)
    {
        Terminate(
            "Stellar dust mass exceeds total stellar metal ejecta");
    }

    // ------------------------------------------------------------
    // Remove the condensed species from the ejecta that will be
    // deposited into gas.
    // ------------------------------------------------------------

    elem_gas.C -= result.carbon_dust;

    if(use_element_resolved_silicate_limits)
    {
        elem_gas.O  -=
            result.silicate_dust *
            FB_SIL_FRAC_O;

        elem_gas.Mg -=
            result.silicate_dust *
            FB_SIL_FRAC_MG;

        elem_gas.Si -=
            result.silicate_dust *
            FB_SIL_FRAC_SI;

        elem_gas.Fe -=
            result.silicate_dust *
            FB_SIL_FRAC_FE;
    }
    else
    {
        // O and Mg are explicit in the AGB tables and are removed directly.
        // The Si+Fe share is removed from the implicit untracked-metal
        // residual when result.gas_metals is reduced below.
        elem_gas.O -=
            result.silicate_dust *
            FB_SIL_FRAC_O;

        elem_gas.Mg -=
            result.silicate_dust *
            FB_SIL_FRAC_MG;
    }

    // Numerical cleanup only. A physically meaningful negative value
    // should never reach this point because the availability limiter
    // above prevents it.
    elem_gas.C  = std::max(0.0, elem_gas.C);
    elem_gas.N  = std::max(0.0, elem_gas.N);
    elem_gas.O  = std::max(0.0, elem_gas.O);
    elem_gas.Ne = std::max(0.0, elem_gas.Ne);
    elem_gas.Mg = std::max(0.0, elem_gas.Mg);
    elem_gas.Si = std::max(0.0, elem_gas.Si);
    elem_gas.Fe = std::max(0.0, elem_gas.Fe);

    result.gas_metals =
        MZ_total_code -
        result.total_dust;

    if(result.gas_metals < 0.0)
    {
        if(result.gas_metals > -1e-12)
            result.gas_metals = 0.0;
        else
            Terminate(
                "Negative gas-phase metal ejecta after dust partition");
    }

    const double tracked_gas_metals =
        elem_gas.C  + elem_gas.N  + elem_gas.O +
        elem_gas.Ne + elem_gas.Mg + elem_gas.Si + elem_gas.Fe;

    const double metal_tolerance =
        1e-10 * std::max(MZ_total_code, 1e-30);

    if(tracked_gas_metals > result.gas_metals + metal_tolerance)
    {
        Terminate(
            "Tracked gas-phase elemental ejecta exceed total gas-phase metals");
    }

    return result;
}


/*
 * Numerical sampling for time-distributed SNII dust
 * -------------------------------------------------
 * DustParticlesPerSNII is interpreted as the requested TOTAL number of
 * PartType6 representatives for the complete SSP SNII/HN episode, subject
 * to a conservation floor of one representative per active tranche.
 *
 * The SNII/HN energy and ejecta are distributed across SNII_TIME_BINS.
 * Without this helper, create_dust_particles_from_feedback() would create
 * DustParticlesPerSNII particles on EVERY tranche, multiplying the intended
 * numerical sampling by SNII_TIME_BINS.
 */
static inline int snii_dust_particles_for_tranche(int ibin)
{
    const int total = std::max(0, All.DustParticlesPerSNII);

    if(total <= 0)
        return 0;

    const int base      = total / SNII_TIME_BINS;
    const int remainder = total % SNII_TIME_BINS;

    return std::max(
        1,
        base + ((ibin < remainder) ? 1 : 0));
}


/*
 * create_dust_particles_from_feedback() currently obtains the SNII particle
 * count from All.DustParticlesPerSNII.  Feedback events are processed
 * serially within each MPI task, so temporarily overriding this local-task
 * value for the duration of one spawn call is safe.  It is restored
 * immediately afterward.
 *
 * IMPORTANT: this changes ONLY the number of numerical representatives.
 * carbon_dust_mass + silicate_dust_mass is unchanged, so the tranche dust
 * mass remains conserved.
 */
static inline double create_snii_dust_for_tranche(
    simparticles *Sp,
    int star_idx,
    double carbon_dust_mass,
    double silicate_dust_mass,
    int tranche_index)
{
    const int n_create =
        snii_dust_particles_for_tranche(tranche_index);

    if(n_create <= 0)
        return 0.0;

    const int saved_particles_per_snii =
        All.DustParticlesPerSNII;

    All.DustParticlesPerSNII = n_create;

    double created_mass =
        create_dust_particles_from_feedback(
            Sp,
            star_idx,
            carbon_dust_mass,
            silicate_dust_mass,
            /*feedback_type=*/1);

    All.DustParticlesPerSNII =
        saved_particles_per_snii;

    return created_mass;
}


#endif // DUST

// Per-window diagnostic counters (reset each flush cadence via agb_diag_reset())
static int    agb_event_count     = 0;
static int    agb_deposition_failures = 0;
static int    snii_deposition_failures = 0;
static double agb_total_metals_g  = 0.0;
static double agb_total_energy_erg= 0.0;
static int    agb_stars_checked   = 0;
static int    agb_stars_eligible  = 0;

static void agb_diag_reset()
{
  agb_event_count      = 0;
  agb_deposition_failures = 0;
  snii_deposition_failures = 0;
  agb_total_metals_g   = 0.0;
  agb_total_energy_erg = 0.0;
  agb_stars_checked    = 0;
  agb_stars_eligible   = 0;

#ifdef DUST
  agb_dust_partition_count       = 0;
  agb_dust_carbon_limited_count  = 0;
  agb_dust_silicate_target_count = 0;
  agb_dust_silicate_O_count      = 0;
  agb_dust_silicate_Mg_count     = 0;
  agb_dust_silicate_proxy_count  = 0;
  agb_dust_below_min_count       = 0;
#endif
}

// Reservoir diagnostics (reset each call to apply_stellar_feedback)
static int    stars_with_reservoir        = 0;
static double reservoir_total_energy      = 0.0;

// Global spatial hash instance
spatial_hash_zoom gas_hash;
spatial_hash_zoom star_hash;
spatial_hash_zoom dust_hash;
spatial_hash_zoom dust_hash_shock;   // reserved for a dedicated shock-destruction hash; currently unused

// Set by rebuild_feedback_spatial_hash() on every call.
//
// This lets the reservoir logic distinguish between:
//
//   (a) a genuinely refreshed gas environment, and
//   (b) another feedback call using the same cached spatial hash.
//
// Reservoir energy is retried only in case (a).  Keeping this as a file-scope
// flag avoids changing the public rebuild_feedback_spatial_hash() signature
// (and therefore avoids requiring a matching feedback.h change).
static bool feedback_hash_rebuilt_this_call = false;

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
  // Default for this invocation: hashes were NOT refreshed.
  // All MPI tasks will subsequently receive the same need_rebuild value.
  feedback_hash_rebuilt_this_call = false;

  static double total_rebuild_time = 0.0;
  static int    rebuild_count      = 0;

  // Per-hash cumulative timing, parallel to total_rebuild_time/rebuild_count.
  // Added to answer: which of gas/star/dust dominates the existing combined
  // [HASH_TIMING] number, before deciding whether a 4th (shock-only, finer)
  // dust hash is affordable to add on top.
  static double total_gas_time  = 0.0;
  static double total_star_time = 0.0;
  static double total_dust_time = 0.0;

  // Rebuild when scale factor has advanced by more than this since last build.
  static constexpr double REBUILD_EVERY_DLOGA = 0.002;
  static double last_rebuild_a = -1.0;

  int need_rebuild = 0;
  if(All.ThisTask == 0) {
    if(!gas_hash.is_built || last_rebuild_a < 0.0 ||
       (All.Time - last_rebuild_a) >= REBUILD_EVERY_DLOGA)
      need_rebuild = 1;
  }
  MPI_Bcast(&need_rebuild, 1, MPI_INT, 0, comm);

  if(!need_rebuild)
      return;

  // From this point onward every task is participating in an actual rebuild.
  feedback_hash_rebuilt_this_call = true;

  double t_start = MPI_Wtime();

  // ── Step 1: Gas hash ──────────────────────────────────────────────────────
  double t_gas_start = MPI_Wtime();
  gas_hash.build(Sp, dust_search_radius, All.SofteningTable[0], comm, 0, 768);
  double t_gas_end = MPI_Wtime();

  // ── Step 2: Star hash ─────────────────────────────────────────────────
  int n_stars_local = 0;
  for(int i = 0; i < Sp->NumPart; i++)
      if(Sp->P[i].getType() == 4) n_stars_local++;
  int n_stars_global = 0;
  MPI_Allreduce(&n_stars_local, &n_stars_global, 1, MPI_INT, MPI_SUM, comm);

  double t_star_start = MPI_Wtime();
  if(n_stars_global > 0)
      star_hash.build(Sp, dust_search_radius, All.SofteningTable[4], comm, 4, 768);
  else
      star_hash.is_built = false;
  double t_star_end = MPI_Wtime();

  // ── Step 3: Dust hash — inherits bbox from gas hash ───────────────────────
  int n_dust_local = 0;
  for(int i = 0; i < Sp->NumPart; i++)
      if(Sp->P[i].getType() == 6) n_dust_local++;
  int n_dust_global = 0;
  MPI_Allreduce(&n_dust_local, &n_dust_global, 1, MPI_INT, MPI_SUM, comm);

  double t_dust_start = MPI_Wtime();
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
  double t_dust_end = MPI_Wtime();

  double t_end = MPI_Wtime();
  total_rebuild_time += (t_end - t_start);
  total_gas_time      += (t_gas_end  - t_gas_start);
  total_star_time     += (t_star_end - t_star_start);
  total_dust_time     += (t_dust_end - t_dust_start);
  rebuild_count++;
  last_rebuild_a = All.Time;

  if(All.ThisTask == 0) {
    printf("[HASH_TIMING] Rebuild #%d took %.3f sec, avg %.3f sec\n",
           rebuild_count, t_end - t_start, total_rebuild_time / rebuild_count);
    printf("[HASH_TIMING_DETAIL] gas=%.3f (avg %.3f)  star=%.3f (avg %.3f)  "
           "dust=%.3f (avg %.3f)  [sec]\n",
           t_gas_end - t_gas_start,  total_gas_time  / rebuild_count,
           t_star_end - t_star_start, total_star_time / rebuild_count,
           t_dust_end - t_dust_start, total_dust_time / rebuild_count);
    gas_hash.print_stats();
  }
}

void feedback_diag_try_flush(MPI_Comm comm, int cadence)
{
  if(cadence <= 0) return;
  if((All.NumCurrentTiStep % cadence) != 0) return;

  struct Local {
    long long n_SNII, n_AGB;
    double E_SN, E_AGB, E_dep, E_to_res, E_from_res;
  } in {
    FbDiag.n_SNII, FbDiag.n_AGB,
    FbDiag.E_SN_erg, FbDiag.E_AGB_erg,
    FbDiag.E_deposited_erg, FbDiag.E_to_reservoir_erg, FbDiag.E_from_reservoir_erg
  }, out{};

  MPI_Reduce(&in.n_SNII,     &out.n_SNII,     1, MPI_LONG_LONG, MPI_SUM, 0, comm);
  MPI_Reduce(&in.n_AGB,      &out.n_AGB,      1, MPI_LONG_LONG, MPI_SUM, 0, comm);
  MPI_Reduce(&in.E_SN,       &out.E_SN,       1, MPI_DOUBLE,    MPI_SUM, 0, comm);
  MPI_Reduce(&in.E_AGB,      &out.E_AGB,      1, MPI_DOUBLE,    MPI_SUM, 0, comm);
  MPI_Reduce(&in.E_dep,      &out.E_dep,      1, MPI_DOUBLE,    MPI_SUM, 0, comm);
  MPI_Reduce(&in.E_to_res,   &out.E_to_res,   1, MPI_DOUBLE,    MPI_SUM, 0, comm);
  MPI_Reduce(&in.E_from_res, &out.E_from_res, 1, MPI_DOUBLE,    MPI_SUM, 0, comm);

  if(All.ThisTask == 0) {
    if(out.n_SNII || out.n_AGB || out.E_dep || out.E_from_res) {
      FB_PRINT("events: SNII=%lld (%.3e erg)  AGB=%lld (%.3e erg)\n",
               out.n_SNII, out.E_SN, out.n_AGB, out.E_AGB);
      FB_PRINT("energy: deposited=%.3e erg  to_reservoir=%.3e erg  from_reservoir=%.3e erg\n",
               out.E_dep, out.E_to_res, out.E_from_res);
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

double get_local_smoothing_length_tree(simparticles *Sp,
                                       ngbtree * /*Tree*/,
                                       int star_idx)
{
    double h = 0.0;

    // Gas particle: just use its own SPH smoothing length.
    if(Sp->P[star_idx].getType() == 0)
    {
        if(Sp->SphP[star_idx].Hsml > 0.0 &&
           std::isfinite(Sp->SphP[star_idx].Hsml))
        {
            return Sp->SphP[star_idx].Hsml;
        }
    }

    // Star particle:
    //
    // Find the ACTUAL nearest local gas particle rather than searching
    // only the first 256 entries of the gas array.  The old implementation
    // could assign a star the Hsml of an unrelated gas particle, producing
    // an unphysically large or small feedback kernel.
    if(Sp->P[star_idx].getType() == 4)
    {
        double best_r2 = HUGE_VAL;
        int best_gas = -1;

        for(int i = 0; i < Sp->NumGas; ++i)
        {
            if(Sp->P[i].getType() != 0)
                continue;

            if(!(Sp->SphP[i].Hsml > 0.0) ||
               !std::isfinite(Sp->SphP[i].Hsml))
                continue;

            double d[3];

            Sp->nearest_image_intpos_to_pos(
                Sp->P[i].IntPos,
                Sp->P[star_idx].IntPos,
                d);

            double r2 =
                d[0] * d[0] +
                d[1] * d[1] +
                d[2] * d[2];

            if(r2 < best_r2)
            {
                best_r2 = r2;
                best_gas = i;
            }
        }

        if(best_gas >= 0)
            h = Sp->SphP[best_gas].Hsml;
    }

    if(!(h > 0.0) || !std::isfinite(h))
        h = std::max(All.SofteningTable[0], 1.0e-6);

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

static void gather_neighbors(simparticles *Sp,
                             ngbtree *Tree,
                             domain<simparticles> *D,
                             int star_i,
                             std::vector<int> &ngb,
                             std::vector<double> &dist,
                             double &hsml)
{
    ngb.clear();
    dist.clear();
    hsml = 0.0;

    // ---------------------------------------------------------------------
    // 1. Initial estimate.
    //
    // For stars, get_local_smoothing_length_tree() currently returns the
    // Hsml of the actual nearest local gas particle.  This gives us a
    // sensible local starting scale, but it does NOT guarantee that a
    // star-centered 2h kernel contains a well-resolved number of gas
    // particles.
    // ---------------------------------------------------------------------

    const double initial_hsml =
        get_local_smoothing_length_tree(Sp, Tree, star_i);

    if(!(initial_hsml > 0.0) || !std::isfinite(initial_hsml))
        return;

    const double initial_radius = 2.0 * initial_hsml;

    // ---------------------------------------------------------------------
    // 2. Define a conservative maximum search radius.
    //
    // Positions/radii here are in comoving kpc/h.
    //
    // Physical pkpc = (ckpc/h) * a / h
    //
    // therefore
    //
    // ckpc/h = pkpc * h / a.
    //
    // We impose BOTH:
    //   (a) at most 4x the original search radius, and
    //   (b) at most 20 physical kpc,
    //
    // except that an already-large initial kernel is never forcibly shrunk.
    // ---------------------------------------------------------------------

    double physical_cap_ckpch;

    if(All.ComovingIntegrationOn)
    {
        const double a =
            std::max((double)All.Time, 1.0e-8);

        physical_cap_ckpch =
            FEEDBACK_MAX_RADIUS_PKPC *
            All.HubbleParam / a;
    }
    else
    {
        // In non-comoving runs, preserve the same kpc/h unit convention.
        physical_cap_ckpch =
            FEEDBACK_MAX_RADIUS_PKPC *
            All.HubbleParam;
    }

    const double factor_cap =
        FEEDBACK_MAX_RADIUS_FACTOR * initial_radius;

    double max_radius =
        std::min(factor_cap, physical_cap_ckpch);

    // Never shrink a kernel that was already larger than the safety cap.
    if(max_radius < initial_radius)
        max_radius = initial_radius;

    // ---------------------------------------------------------------------
    // 3. Allocate neighbor buffers once.
    // ---------------------------------------------------------------------

    ngb.resize(MAX_KERNEL_NEIGHBORS);
    dist.resize(MAX_KERNEL_NEIGHBORS);

    double search_radius = initial_radius;
    hsml = 0.5 * search_radius;

    int n_found = 0;
    int iter = 0;

    // ---------------------------------------------------------------------
    // 4. Initial search + adaptive enlargement for under-populated kernels.
    //
    // find_feedback_neighbors_tree() searches within 2*hsml, so setting
    //
    //       hsml = search_radius / 2
    //
    // keeps the returned smoothing length consistent with the actual
    // cubic-spline kernel support.
    // ---------------------------------------------------------------------

    while(true)
    {
        n_found = 0;
        hsml = 0.5 * search_radius;

        find_feedback_neighbors_tree(
            Sp,
            Tree,
            D,
            star_i,
            ngb.data(),
            dist.data(),
            &n_found,
            &hsml,
            MAX_KERNEL_NEIGHBORS);

        // Adequately resolved: keep this kernel.
        if(n_found >= MIN_FEEDBACK_NEIGHBORS)
            break;

        // Reached our allowed spatial extent.
        if(search_radius >= max_radius)
            break;

        // Safety against pathological looping.
        if(iter >= FEEDBACK_MAX_KERNEL_ITERS)
            break;

        double new_radius =
            search_radius * FEEDBACK_KERNEL_GROWTH;

        if(new_radius > max_radius)
            new_radius = max_radius;

        // Floating-point safety.
        if(!(new_radius > search_radius))
            break;

        search_radius = new_radius;
        iter++;
    }

    // hsml must correspond exactly to the final search support.
    hsml = 0.5 * search_radius;

    // ---------------------------------------------------------------------
    // 5. Diagnostics.
    //
    // Print both the original and final kernels.  This will let us quantify
    // exactly how often the adaptive mechanism intervenes.
    // ---------------------------------------------------------------------

    if(All.FeedbackDebugLevel >= 2)
    {
        const bool expanded =
            search_radius > initial_radius * (1.0 + 1.0e-12);

        printf(
            "[FEEDBACK_KERNEL|T=%d] "
            "star=%d "
            "hsml_initial=%.6e "
            "radius_initial=%.6e "
            "hsml_final=%.6e "
            "radius_final=%.6e "
            "Nngb=%d "
            "expanded=%d "
            "iters=%d "
            "hit_cap=%d\n",
            All.ThisTask,
            star_i,
            initial_hsml,
            initial_radius,
            hsml,
            search_radius,
            n_found,
            expanded ? 1 : 0,
            iter,
            (search_radius >= max_radius *
             (1.0 - 1.0e-12)) ? 1 : 0);
    }

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

static inline double c_v(double mu)       { return 1.5 * BOLTZMANN / (mu * PROTONMASS); }

// Actual gas mean molecular weight from the particle's own ionization
// state (Ne), matching the convention used in dust.cc's
// get_temperature_from_entropy(). Used to size stochastic heating events
// to the ACTUAL particle being heated, rather than assuming a fixed
// ionization state.
static inline double mu_from_ne(double ne)
{
  const double XH = HYDROGEN_MASSFRAC;
  const double Y  = (1.0 - XH) / (4.0 * XH);
  return (1.0 + 4.0 * Y) / (1.0 + Y + ne);
}

static inline double ucode_to_TK(double u_code, double ne)
{
  double u_cgs = u_code * All.UnitVelocity_in_cm_per_s * All.UnitVelocity_in_cm_per_s;
  return u_cgs / c_v(mu_from_ne(ne));
}

// ----------------------- Energy Deposition ----------------------------------

/*
 * Deposit feedback using stochastic ΔT heating.
 *
 * Chemistry:
 *   - distribute metal ejecta over the full SPH kernel
 *     using kernel-weighted gas mass.
 *
 * Thermal energy:
 *   - each selected gas particle receives exactly DELTA_T_TARGET, sized
 *     using THAT PARTICLE'S actual mu (from its current Ne), not a fixed
 *     assumed ionization state;
 *   - neighbors are sampled without replacement by kernel weight;
 *   - full heating packets are deposited deterministically;
 *   - a final fractional packet is handled with a Bernoulli draw;
 *   - statistically consumed fractional energy is never banked;
 *   - only genuinely unprocessed energy is returned in
 *     E_unspent_code for the reservoir.
 *
 * Diagnostics:
 *   - from_reservoir labels each stochastic heating packet as either a
 *     newly triggered stellar-feedback EVENT or a RESERVOIR retry;
 *   - debug level >= 2 prints persistent star/gas ParticleIDs and both
 *     pre- and post-heating temperatures.
 *
 * Returns the thermal energy actually deposited.
 */
static double deposit_energy_stochastic(
    simparticles *Sp,
    const std::vector<int> &ngb_list,
    const std::vector<double> &distances,
    double hsml,
    int star_idx,
    double E_avail_code,
    double MZ_code,
    const ElementYields &elem,
    double &E_unspent_code,
    bool from_reservoir)
{
    E_unspent_code = 0.0;

    if(ngb_list.empty())
    {
        E_unspent_code = E_avail_code;
        return 0.0;
    }

    const int N = (int)ngb_list.size();

    // ============================================================
    // 1. Compute kernel-weighted mass fractions.
    //
    // Chemistry is distributed over the whole feedback kernel,
    // independently of which particles happen to receive thermal
    // energy.
    // ============================================================

    std::vector<double> weights(N, 0.0);

    double weight_sum = 0.0;

    for(int i = 0; i < N; ++i)
    {
        int j = ngb_list[i];

        double W =
            cubic_spline_kernel(distances[i], hsml);

        // SPH-style mass weighting.
        weights[i] =
            Sp->P[j].getMass() * W;

        weight_sum += weights[i];
    }

    if(weight_sum <= 0.0)
    {
        E_unspent_code = E_avail_code;
        return 0.0;
    }


    // ============================================================
    // 2. Deposit metals and individual elements.
    //
    // This happens whether or not a particular particle is selected
    // for stochastic thermal heating.
    // ============================================================

#ifdef METALS

    if(MZ_code > 0.0)
    {
        for(int i = 0; i < N; ++i)
        {
            int j = ngb_list[i];

            double fraction =
                weights[i] / weight_sum;

            double old_gas_mass =
                Sp->P[j].getMass();

            double old_C =
                old_gas_mass *
                Sp->SphP[j].GasCarbonMassFraction;

            double old_N =
                old_gas_mass *
                Sp->SphP[j].GasNitrogenMassFraction;

            double old_O =
                old_gas_mass *
                Sp->SphP[j].GasOxygenMassFraction;

            double old_Ne =
                old_gas_mass *
                Sp->SphP[j].GasNeonMassFraction;

            double old_Mg =
                old_gas_mass *
                Sp->SphP[j].GasMagnesiumMassFraction;

            double old_Si =
                old_gas_mass *
                Sp->SphP[j].GasSiliconMassFraction;

            double old_Fe =
                old_gas_mass *
                Sp->SphP[j].GasIronMassFraction;


            double dMZ =
                MZ_code * fraction;

            double dC =
                elem.C * fraction;

            double dN =
                elem.N * fraction;

            double dO =
                elem.O * fraction;

            double dNe =
                elem.Ne * fraction;

            double dMg =
                elem.Mg * fraction;

            double dSi =
                elem.Si * fraction;

            double dFe =
                elem.Fe * fraction;


            double new_gas_mass =
                old_gas_mass + dMZ;

            Sp->P[j].setMass(new_gas_mass);

            Sp->SphP[j].MassMetallicity += dMZ;

            Sp->SphP[j].Metallicity =
                Sp->SphP[j].MassMetallicity /
                new_gas_mass;

            Sp->P[j].Metallicity =
                Sp->SphP[j].Metallicity;

            Sp->SphP[j].GasCarbonMassFraction =
                (old_C + dC) /
                new_gas_mass;

            Sp->SphP[j].GasNitrogenMassFraction =
                (old_N + dN) /
                new_gas_mass;

            Sp->SphP[j].GasOxygenMassFraction =
                (old_O + dO) /
                new_gas_mass;

            Sp->SphP[j].GasNeonMassFraction =
                (old_Ne + dNe) /
                new_gas_mass;

            Sp->SphP[j].GasMagnesiumMassFraction =
                (old_Mg + dMg) /
                new_gas_mass;

            Sp->SphP[j].GasSiliconMassFraction =
                (old_Si + dSi) /
                new_gas_mass;

            Sp->SphP[j].GasIronMassFraction =
                (old_Fe + dFe) /
                new_gas_mass;
        }
    }

#endif // METALS


    // ============================================================
    // 3. Nothing more to do if there is no thermal energy.
    // ============================================================

    if(E_avail_code <= 0.0)
        return 0.0;


    double E_remaining_code =
        E_avail_code;

    double E_deposited_code =
        0.0;


    // ============================================================
    // 4. Track which candidate neighbors are still available.
    //
    // We sample without replacement so one SN event cannot heat
    // the same gas particle twice.
    // ============================================================

    std::vector<int> available(N);

    for(int i = 0; i < N; ++i)
        available[i] = i;


    unsigned long long random_counter = 0;


    // ============================================================
    // 5. Stochastic thermal heating.
    // ============================================================

    while(E_remaining_code > 0.0 &&
          !available.empty())
    {
        // --------------------------------------------------------
        // Select one remaining gas neighbor according to its
        // kernel weight.
        // --------------------------------------------------------

        double available_weight_sum = 0.0;

        for(int idx : available)
            available_weight_sum += weights[idx];

        if(available_weight_sum <= 0.0)
            break;


        double rnd =
            get_random_number(
                Sp->P[star_idx].ID.get() +
                All.NumCurrentTiStep +
                random_counter++);

        double target =
            rnd * available_weight_sum;

        double cumulative = 0.0;

        int chosen_pos = -1;

        for(int a = 0;
            a < (int)available.size();
            ++a)
        {
            int idx = available[a];

            cumulative += weights[idx];

            if(cumulative >= target)
            {
                chosen_pos = a;
                break;
            }
        }

        if(chosen_pos < 0)
            chosen_pos =
                (int)available.size() - 1;


        int idx =
            available[chosen_pos];

        int j =
            ngb_list[idx];


        // Remove this candidate immediately:
        // sampling is without replacement.
        available.erase(
            available.begin() + chosen_pos);

        double u_old =
            Sp->get_utherm_from_entropy(j);

        if(u_old <= 0.0 || !std::isfinite(u_old))
        {
            FB_PRINT("WARNING: Particle %d has invalid u_old=%g\n",j, u_old);
            continue;
        }

        // --------------------------------------------------------
        // Per-particle target Δu, sized using THIS particle's
        // actual mu (from its own Ne) rather than a fixed assumed
        // ionization state.
        //
        // Previously delta_u_cgs was computed once, outside this
        // loop, via c_v(mu_default(DELTA_T_TARGET)) — which always
        // evaluated to the fully-ionized mu=0.62 since
        // DELTA_T_TARGET=3e6 > 1.5e4. That systematically undersizes
        // c_v for the mostly-neutral/partially-ionized ISM gas
        // feedback actually targets (mu~1.22), producing a real
        // temperature jump of ΔT_target * (1.22/0.62) ≈ 1.97x the
        // intended value — matching the observed ~6e6 K post-heating
        // temperatures against a 3e6 K target.
        // --------------------------------------------------------

        const double ne_before =
            Sp->SphP[j].Ne;

        double mu_j =
            mu_from_ne(ne_before);

        const double T_before =
            ucode_to_TK(
                u_old,
                ne_before);

        double delta_u_cgs_j =
            c_v(mu_j) * DELTA_T_TARGET;

        double delta_u_code_j =
            delta_u_cgs_j /
            (All.UnitVelocity_in_cm_per_s *
             All.UnitVelocity_in_cm_per_s);

        // --------------------------------------------------------
        // Energy required to increase THIS particle's temperature
        // by DELTA_T_TARGET.
        //
        // Use its actual current mass, including any metal ejecta
        // just deposited above.
        // --------------------------------------------------------

        double gas_mass_g =
            Sp->P[j].getMass() *
            All.UnitMass_in_g;

        double E_heat_erg =
            gas_mass_g *
            delta_u_cgs_j;

        double E_heat_code =
            E_heat_erg /
            All.UnitEnergy_in_cgs;

        if(E_heat_code <= 0.0 ||
           !std::isfinite(E_heat_code))
            continue;


        bool heat_particle = false;


        // --------------------------------------------------------
        // Full heating packet available:
        // heat with probability 1.
        // --------------------------------------------------------

        if(E_remaining_code >= E_heat_code)
        {
            heat_particle = true;

            E_remaining_code -=
                E_heat_code;
        }

        // --------------------------------------------------------
        // Fractional final packet:
        //
        // P = E_remaining / E_heat.
        //
        // Regardless of whether the draw succeeds, this remainder
        // is now statistically accounted for and MUST NOT later be
        // placed into the reservoir.
        // --------------------------------------------------------

        else
        {
            double probability =
                E_remaining_code /
                E_heat_code;

            probability =
                fb_clamp(probability,
                         0.0, 1.0);

            double draw =
                get_random_number(
                    Sp->P[star_idx].ID.get() +
                    All.NumCurrentTiStep +
                    random_counter++);

            if(draw < probability)
                heat_particle = true;

            // Statistically consumed either way.
            E_remaining_code = 0.0;
        }


        if(!heat_particle)
            continue;


        // ========================================================
        // 6. Apply exactly one DELTA_T_TARGET temperature jump,
        //    using this particle's own delta_u_code_j.
        // ========================================================

        double u_new =
            u_old +
            delta_u_code_j;


        Sp->set_entropy_from_utherm(
            u_new, j);

        set_thermodynamic_variables_safe(
            Sp, j);


        E_deposited_code +=
            E_heat_code;

        // Detailed diagnostic.
        //
        // Print both local array indices and persistent ParticleIDs.
        // The local indices may be reused as particles migrate between MPI
        // tasks, whereas starID/gasID identify the physical particles across
        // feedback calls and snapshots.
        //
        // T_before uses the particle's pre-heating Ne. T_after uses the
        // thermodynamic state after set_thermodynamic_variables_safe().
        if(All.FeedbackDebugLevel >= 2)
        {
            const double T_after =
                ucode_to_TK(
                    Sp->get_utherm_from_entropy(j),
                    Sp->SphP[j].Ne);

            const unsigned long long star_id =
                (unsigned long long)Sp->P[star_idx].ID.get();

            const unsigned long long gas_id =
                (unsigned long long)Sp->P[j].ID.get();

            printf(
                "[SN_STOCHASTIC|T=%d|a=%.8g|z=%.6f] "
                "source=%s "
                "star=%d gas=%d "
                "starID=%llu gasID=%llu "
                "Eheat=%.6e erg "
                "DeltaT_target=%.6e K "
                "mu=%.6f "
                "Tbefore=%.6e K "
                "Tafter=%.6e K\n",
                All.ThisTask,
                (double)All.Time,
                1.0 / All.Time - 1.0,
                from_reservoir ? "RESERVOIR" : "EVENT",
                star_idx,
                j,
                star_id,
                gas_id,
                E_heat_erg,
                DELTA_T_TARGET,
                mu_j,
                T_before,
                T_after);
        }
    }


    // ============================================================
    // 7. Truly unspent energy.
    //
    // This should normally be zero. It is non-zero only if we ran
    // out of viable gas neighbors before accounting for the event
    // energy.
    //
    // THIS quantity may safely go into EnergyReservoir.
    // ============================================================

    if(E_remaining_code > 0.0)
        E_unspent_code =
            E_remaining_code;


    return E_deposited_code;
}

// -------------------- Feedback Event Handling -------------------------------

static bool apply_feedback_event(simparticles *Sp,ngbtree *Tree,domain<simparticles> *D,
    int star_i,bool is_SNII,double E_code,double MZ_code,const ElementYields &elem)
{
  std::vector<int>    ngb;
  std::vector<double> dist;
  double hsml = 0;
  gather_neighbors(Sp, Tree, D, star_i, ngb, dist, hsml);

  if(ngb.empty())
  {
      // No gas is available to receive this event. Keep the entire event
      // pending: do not debit metals, create dust, mark the event complete,
      // or bank its full energy. Banking the energy while leaving the event
      // pending would inject it twice when the event is retried later.

      if(All.FeedbackDebugLevel >= 2)
          printf("[FEEDBACK_NO_GAS|T=%d] Star %d: no gas neighbors; "
                "retaining %.6e code mass and %.6e code energy in pending event\n",
                All.ThisTask, star_i, MZ_code, E_code);

      return false;
  }

  if(is_SNII)
      diag_add_sn(E_code * All.UnitEnergy_in_cgs);
  else
      diag_add_agb(E_code * All.UnitEnergy_in_cgs);

  double E_unspent_code = 0.0;

  double E_used_code =
      deposit_energy_stochastic(
          Sp,
          ngb,
          dist,
          hsml,
          star_i,
          E_code,
          MZ_code,
          elem,
          E_unspent_code,
          /*from_reservoir=*/false);

  diag_add_Edep(
      E_used_code *
      All.UnitEnergy_in_cgs);

  if(E_unspent_code > 0.0)
  {
      Sp->P[star_i].EnergyReservoir +=
          E_unspent_code;

      diag_add_EtoRes(
          E_unspent_code *
          All.UnitEnergy_in_cgs);
  }

  return true;

}

/*
If we have energy that couldn't previously be processed (i.e. not enough neighbors)
Try the normal stochastic algorithm again.
*/
static void try_release_reservoir(
    simparticles *Sp,
    ngbtree *Tree,
    domain<simparticles> *D,
    int star_i)
{
    double E_code =
        Sp->P[star_i].EnergyReservoir;

    if(E_code <= 0.0)
        return;

    std::vector<int> ngb;
    std::vector<double> dist;
    double hsml = 0.0;

    gather_neighbors(
        Sp, Tree, D,
        star_i,
        ngb, dist, hsml);

    if(ngb.empty())
        return;

    ElementYields no_elements;

    double E_unspent_code = 0.0;

    double E_used_code =
        deposit_energy_stochastic(
            Sp,
            ngb,
            dist,
            hsml,
            star_i,
            E_code,
            /*MZ_code=*/0.0,
            no_elements,
            E_unspent_code,
            /*from_reservoir=*/true);

    Sp->P[star_i].EnergyReservoir =
        E_unspent_code;

    diag_add_EfromRes(
        E_used_code *
        All.UnitEnergy_in_cgs);

    diag_add_Edep(
        E_used_code *
        All.UnitEnergy_in_cgs);
}


// ---------------------------- Public API ------------------------------------

void init_stellar_feedback(void)
{
#ifdef FEEDBACK
  // Print banner on task 0
  if(All.ThisTask == 0)
    FB_PRINT(
    "FEEDBACK enabled; stochastic thermal heating "
    "DeltaT=%.3e K\n",
    DELTA_T_TARGET);

  // All tasks load AGB yields (not just task 0)
  if(!AGB_Yields.load_from_file(All.AGBYieldFile)) {
    if(All.ThisTask == 0) {
      printf("[FEEDBACK] ERROR: Failed to load AGB yields from '%s'\n", All.AGBYieldFile);
      printf("[FEEDBACK] AGB feedback will be DISABLED\n");
    }
  } else {
    if(All.ThisTask == 0)
      printf("[FEEDBACK] AGB yields loaded from '%s'\n", All.AGBYieldFile);
  }

  // Load ordinary SNII yields.
  if(!SNII_Yields.load_sn_from_file(All.SNIIYieldFile))
    {
      if(All.ThisTask == 0)
        printf("[FEEDBACK] ERROR: Failed to load SNII yields from '%s'\n",
              All.SNIIYieldFile);

      Terminate("Failed to load SNII yield table");
    }

  if(All.ThisTask == 0)
    printf("[FEEDBACK] SNII yields loaded from '%s'\n",
          All.SNIIYieldFile);

  // Load hypernova yields.
  if(!SNII_Yields.load_hn_from_file(All.HypernovaYieldFile))
    {
      if(All.ThisTask == 0)
        printf("[FEEDBACK] ERROR: Failed to load hypernova yields from '%s'\n",
              All.HypernovaYieldFile);

      Terminate("Failed to load hypernova yield table");
    }

  if(All.ThisTask == 0)
    printf("[FEEDBACK] Hypernova yields loaded from '%s'\n",
          All.HypernovaYieldFile);

#ifdef DUST
  if(All.ThisTask == 0)
    {
      printf(
          "[FEEDBACK] SNII dust sampling: %d total particles per SSP "
          "distributed across %d temporal tranches\n",
          All.DustParticlesPerSNII,
          SNII_TIME_BINS);

      printf(
          "[FEEDBACK] LRN dust sampling: full SSP budget injected once "
          "with first SNII tranche (DustParticlesPerLRN=%d)\n",
          All.DustParticlesPerLRN);
    }
#endif

  // Set the SNII/HN mixture from the parameter file.
  SNII_Yields.set_hypernova_fraction(All.HypernovaFraction);

  // Now build the IMF-integrated SSP yield table using that mixture.
  SNII_Yields.build_imf_yield_table();

  // Build cosmic time table on all tasks
  build_cosmic_time_table();
#endif
}

void apply_stellar_feedback(
    double /*current_time*/,
    simparticles *Sp,
    ngbtree *Tree,
    domain<simparticles> *D,
    MPI_Comm comm)
{
    // Build local star index list.
    static std::vector<int> star_indices;

    star_indices.clear();
    star_indices.reserve(Sp->NumPart / 10);

    for(int p = 0; p < Sp->NumPart; ++p)
    {
        if(Sp->P[p].getType() == 4)
            star_indices.push_back(p);
    }

    int n_stars_local =
        (int)star_indices.size();

    // Collective early exit: if no task has stars, skip everything.
    int global_has_stars = 0;

    MPI_Allreduce(
        &n_stars_local,
        &global_has_stars,
        1,
        MPI_INT,
        MPI_MAX,
        comm);

    if(!global_has_stars)
        return;

    // Build/update spatial hashes used by feedback and dust searches.
    rebuild_feedback_spatial_hash(
        Sp, 0.1, comm);

    // Reset per-call reservoir diagnostics.
    stars_with_reservoir   = 0;
    reservoir_total_energy = 0.0;


    // ====================================================================
    // 1) Account for reservoir energy, but retry it ONLY after the
    //    feedback spatial environment has actually been refreshed.
    //
    // EnergyReservoir contains genuinely unprocessed feedback energy
    // (for example, energy left after exhausting viable gas neighbors).
    // Retrying that energy on every call can repeatedly present it with
    // essentially the same cached neighborhood.  Instead, wait until the
    // gas/star/dust spatial hashes have been rebuilt, which guarantees that
    // the neighbor environment has at least been refreshed before another
    // release attempt.
    //
    // IMPORTANT:
    //   - New feedback EVENT energy below is still processed immediately.
    //   - Existing reservoir energy is still counted every call for
    //     diagnostics, even when no release attempt is made.
    //   - A reservoir created later in THIS call will therefore wait until
    //     the NEXT genuine hash rebuild before being retried.
    // ====================================================================

    for(int p : star_indices)
    {
        if(Sp->P[p].EnergyReservoir > 0.0)
        {
            stars_with_reservoir++;

            reservoir_total_energy +=
                Sp->P[p].EnergyReservoir;

            if(feedback_hash_rebuilt_this_call)
            {
                try_release_reservoir(
                    Sp, Tree, D, p);
            }
        }
    }

    if(All.FeedbackDebugLevel >= 2 &&
       stars_with_reservoir > 0)
    {
        printf(
            "[RESERVOIR_CADENCE|T=%d|a=%.8g|z=%.6f] "
            "environment_refreshed=%d "
            "stars=%d "
            "energy=%.6e erg\n",
            All.ThisTask,
            (double)All.Time,
            1.0 / All.Time - 1.0,
            feedback_hash_rebuilt_this_call ? 1 : 0,
            stars_with_reservoir,
            reservoir_total_energy *
                All.UnitEnergy_in_cgs);
    }


    // ====================================================================
    // 2) Trigger new feedback events.
    // ====================================================================

    for(int p : star_indices)
    {
        double age_Myr =
            get_stellar_age_Myr(
                Sp->P[p].StellarAge,
                0.0);


        // ================================================================
        // 2a) Type II SN + hypernova SSP feedback, distributed in time.
        //
        // The old implementation deposited the entire IMF-integrated
        // 3-40 Myr CCSN/HN budget in one event as soon as the SSP entered
        // the SNII window.  At coarse mass resolution this makes many
        // physically distinct explosions artificially coherent.
        //
        // Here we divide the same IMF-integrated budget into four equal
        // temporal tranches spanning MIN_TYPEII_TIME..MAX_TYPEII_TIME.
        //
        // At most ONE overdue tranche is fired per call.  Thus a temporarily
        // inactive star can catch up, but cannot dump several missed bins in
        // one synchronization point.
        //
        // Every tranche is normalized to the immutable stellar birth mass.
        // The surviving particle mass may decrease as ejecta are returned,
        // but the IMF-integrated SSP budget must not shrink between tranches.
        // ================================================================

        if(age_Myr >= MIN_TYPEII_TIME &&
           !(Sp->P[p].FeedbackFlag & 1))
        {
            int tranche_to_fire = -1;

            for(int ibin = 0;
                ibin < SNII_TIME_BINS;
                ibin++)
            {
                const unsigned int mask =
                    snii_tranche_mask(ibin);

                const double t_start =
                    snii_tranche_start_Myr(ibin);

                if(age_Myr >= t_start &&
                   !(Sp->P[p].FeedbackFlag & mask))
                {
                    tranche_to_fire = ibin;
                    break;
                }
            }

            if(tranche_to_fire >= 0)
            {
                const double tranche_fraction =
                    1.0 /
                    (double)SNII_TIME_BINS;

                const unsigned int tranche_mask =
                    snii_tranche_mask(
                        tranche_to_fire);

                const double tranche_start =
                    snii_tranche_start_Myr(
                        tranche_to_fire);

                const double tranche_end =
                    (tranche_to_fire ==
                     SNII_TIME_BINS - 1)
                    ? MAX_TYPEII_TIME
                    : snii_tranche_start_Myr(
                          tranche_to_fire + 1);


                // --------------------------------------------------------
                // Stellar population represented by this particle.
                // --------------------------------------------------------

                const double m_ssp_msun =
                    stellar_birth_mass_msun(
                        Sp,
                        p);

                double Z_star =
                    Sp->P[p].Metallicity;


                // --------------------------------------------------------
                // IMF-integrated SNII + HN yields for the full SSP.
                // --------------------------------------------------------

                SNIIIMFYield sn =
                    SNII_Yields.get_imf_yields(
                        Z_star);


                // --------------------------------------------------------
                // Validate SSP yields.
                // --------------------------------------------------------

                if(!std::isfinite(sn.Z_per_Mstar)  ||
                   !std::isfinite(sn.C_per_Mstar)  ||
                   !std::isfinite(sn.N_per_Mstar)  ||
                   !std::isfinite(sn.O_per_Mstar)  ||
                   !std::isfinite(sn.Ne_per_Mstar) ||
                   !std::isfinite(sn.Mg_per_Mstar) ||
                   !std::isfinite(sn.Si_per_Mstar) ||
                   !std::isfinite(sn.Fe_per_Mstar) ||
                   !std::isfinite(sn.E51_per_Mstar))
                {
                    Terminate(
                        "Non-finite IMF-integrated SNII/HN yield");
                }

                if(sn.Z_per_Mstar  < 0.0 ||
                   sn.C_per_Mstar  < 0.0 ||
                   sn.N_per_Mstar  < 0.0 ||
                   sn.O_per_Mstar  < 0.0 ||
                   sn.Ne_per_Mstar < 0.0 ||
                   sn.Mg_per_Mstar < 0.0 ||
                   sn.Si_per_Mstar < 0.0 ||
                   sn.Fe_per_Mstar < 0.0 ||
                   sn.E51_per_Mstar <= 0.0)
                {
                    Terminate(
                        "Negative IMF-integrated SNII/HN yield");
                }


                // --------------------------------------------------------
                // This temporal tranche's ejecta and energy.
                //
                // Integrated over all four tranches, this reproduces the
                // exact IMF-integrated budget defined for the birth SSP.
                // --------------------------------------------------------

                double MZ_msun =
                    tranche_fraction *
                    sn.Z_per_Mstar *
                    m_ssp_msun;

                double C_msun =
                    tranche_fraction *
                    sn.C_per_Mstar *
                    m_ssp_msun;

                double N_msun =
                    tranche_fraction *
                    sn.N_per_Mstar *
                    m_ssp_msun;

                double O_msun =
                    tranche_fraction *
                    sn.O_per_Mstar *
                    m_ssp_msun;

                double Ne_msun =
                    tranche_fraction *
                    sn.Ne_per_Mstar *
                    m_ssp_msun;

                double Mg_msun =
                    tranche_fraction *
                    sn.Mg_per_Mstar *
                    m_ssp_msun;

                double Si_msun =
                    tranche_fraction *
                    sn.Si_per_Mstar *
                    m_ssp_msun;

                double Fe_msun =
                    tranche_fraction *
                    sn.Fe_per_Mstar *
                    m_ssp_msun;

                double E_erg =
                    tranche_fraction *
                    All.SNEfficiency *
                    sn.E51_per_Mstar *
                    m_ssp_msun *
                    1.0e51;


                // --------------------------------------------------------
                // Convert to GADGET code units.
                // --------------------------------------------------------

                double MZ_total_code =
                    MZ_msun *
                    SOLAR_MASS /
                    All.UnitMass_in_g;

                double E_code =
                    E_erg /
                    All.UnitEnergy_in_cgs;


                ElementYields elem_total;

                elem_total.C =
                    C_msun *
                    SOLAR_MASS /
                    All.UnitMass_in_g;

                elem_total.N =
                    N_msun *
                    SOLAR_MASS /
                    All.UnitMass_in_g;

                elem_total.O =
                    O_msun *
                    SOLAR_MASS /
                    All.UnitMass_in_g;

                elem_total.Ne =
                    Ne_msun *
                    SOLAR_MASS /
                    All.UnitMass_in_g;

                elem_total.Mg =
                    Mg_msun *
                    SOLAR_MASS /
                    All.UnitMass_in_g;

                elem_total.Si =
                    Si_msun *
                    SOLAR_MASS /
                    All.UnitMass_in_g;

                elem_total.Fe =
                    Fe_msun *
                    SOLAR_MASS /
                    All.UnitMass_in_g;


                // --------------------------------------------------------
                // Sanity check: tracked elements cannot exceed metals.
                // --------------------------------------------------------

                double tracked_msun =
                    C_msun  +
                    N_msun  +
                    O_msun  +
                    Ne_msun +
                    Mg_msun +
                    Si_msun +
                    Fe_msun;

                if(tracked_msun >
                   MZ_msun * 1.000001)
                {
                    printf(
                        "[SNII_YIELD_ERROR] "
                        "Z*=%.6g M*=%.6g "
                        "tranche=%d "
                        "tracked=%.6e > "
                        "total_metals=%.6e Msun\n",
                        Z_star,
                        m_ssp_msun,
                        tranche_to_fire,
                        tracked_msun,
                        MZ_msun);

                    Terminate(
                        "SNII/HN elemental ejecta exceed total metal ejecta");
                }


                // --------------------------------------------------------
                // Partition this tranche's ejecta into gas and dust.
                // --------------------------------------------------------

                ElementYields elem_gas =
                    elem_total;

                double MZ_gas_code =
                    MZ_total_code;

#ifdef DUST
                StellarDustPartition sn_dust;

                if(All.DustParticlesPerSNII > 0)
                {
                    sn_dust =
                        partition_stellar_ejecta_for_dust(
                            MZ_total_code,
                            elem_gas,
                            All.DustYieldSNII,
                            /*target_carbon_fraction=*/0.10,
                            /*use_element_resolved_silicate_limits=*/true);

                    MZ_gas_code =
                        sn_dust.gas_metals;
                }
#endif


                if(All.FeedbackDebugLevel >= 2)
                {
                    printf(
                        "[SNII_TRANCHE|T=%d|a=%.8g|z=%.6f] "
                        "star=%d starID=%llu "
                        "bin=%d/%d "
                        "age=%.6f Myr "
                        "window=[%.6f,%.6f) Myr "
                        "fraction=%.6f "
                        "E=%.6e erg "
                        "MZ=%.6e Msun\n",
                        All.ThisTask,
                        (double)All.Time,
                        1.0 / All.Time - 1.0,
                        p,
                        (unsigned long long)
                            Sp->P[p].ID.get(),
                        tranche_to_fire + 1,
                        SNII_TIME_BINS,
                        age_Myr,
                        tranche_start,
                        tranche_end,
                        tranche_fraction,
                        E_erg,
                        MZ_msun);
                }


                // --------------------------------------------------------
                // Deposit this tranche's gas-phase ejecta and energy.
                // --------------------------------------------------------

                bool ejecta_deposited =
                    apply_feedback_event(
                        Sp,
                        Tree,
                        D,
                        p,
                        /*is_SNII=*/true,
                        E_code,
                        MZ_gas_code,
                        elem_gas);


                if(ejecta_deposited)
                {
                    // Remove this tranche's TOTAL metal ejecta from star.
                    double new_star_mass =
                        Sp->P[p].getMass() -
                        MZ_total_code;

                    if(new_star_mass <= 0.0)
                    {
                        Terminate(
                            "SNII feedback tranche would remove entire stellar particle mass");
                    }

                    Sp->P[p].setMass(
                        new_star_mass);


#ifdef DUST
                    if(sn_dust.total_dust > 0.0)
                    {
                        /*
                         * Keep DustParticlesPerSNII as the TOTAL desired
                         * representation for the complete SSP SNII/HN
                         * episode, distributed across the temporal tranches.
                         */
                        double created_sn_dust =
                            create_snii_dust_for_tranche(
                                Sp,
                                p,
                                sn_dust.carbon_dust,
                                sn_dust.silicate_dust,
                                tranche_to_fire);

                        double creation_tolerance =
                            1e-12 *
                            std::max(
                                sn_dust.total_dust,
                                1e-30);

                        if(std::abs(
                               created_sn_dust -
                               sn_dust.total_dust) >
                           creation_tolerance)
                        {
                            Terminate(
                                "SNII dust creation did not realize partitioned dust mass");
                        }
                    }
#endif
#ifdef DUST
                // --------------------------------------------------------
                // Shock destruction from this tranche only.
                // --------------------------------------------------------

                erode_dust_from_sn_shocks(
                    Sp,
                    p,
                    E_code,
                    comm);


                // --------------------------------------------------------
                // LRN dust.
                //
                // LRN yields are tiny. Dividing the LRN budget into the
                // four SNII temporal tranches pushes the individual dust
                // masses below MIN_DUST_PARTICLE_MASS in dust.cc and can
                // eliminate the LRN PartType6 population entirely.
                //
                // Restore SSP-level LRN sampling: inject the FULL LRN dust
                // budget once, alongside the first SNII tranche. SNII/HN
                // thermal energy and metal ejecta remain time-distributed.
                // --------------------------------------------------------

                if(tranche_to_fire == 0)
                {
                    double n_CCSN_full =
                        CCSN_NUMBER_PER_MSUN *
                        m_ssp_msun;

                    double n_LRN_full =
                        All.DustLRNRatePerCCSN *
                        n_CCSN_full;

                    double LRN_dust_mass_g =
                        n_LRN_full *
                        All.DustLRNDustMassMsun *
                        SOLAR_MASS;

                    double LRN_dust_mass_code =
                        LRN_dust_mass_g /
                        All.UnitMass_in_g;

                    if(LRN_dust_mass_code > 0.0)
                    {
                        double LRN_carbon_dust =
                            0.10 *
                            LRN_dust_mass_code;

                        double LRN_silicate_dust =
                            0.90 *
                            LRN_dust_mass_code;

                        double created_lrn_dust =
                            create_dust_particles_from_feedback(
                                Sp,
                                p,
                                LRN_carbon_dust,
                                LRN_silicate_dust,
                                /*feedback_type=*/3);

                        if(created_lrn_dust > 0.0)
                        {
                            double post_lrn_star_mass =
                                Sp->P[p].getMass() -
                                created_lrn_dust;

                            if(post_lrn_star_mass <= 0.0)
                            {
                                Terminate(
                                    "LRN dust creation would remove entire stellar particle mass");
                            }

                            Sp->P[p].setMass(
                                post_lrn_star_mass);
                        }
                    }
                }
#endif


                // --------------------------------------------------------
                // The full transaction succeeded: gas received the ejecta,
                // the star was debited, associated dust processing occurred,
                // and the tranche may now be marked complete.
                // --------------------------------------------------------

                Sp->P[p].FeedbackFlag |=
                    tranche_mask;


                if(all_snii_tranches_complete(
                       (unsigned int)
                       Sp->P[p].FeedbackFlag))
                {
                    Sp->P[p].FeedbackFlag |= 1;

                    if(All.FeedbackDebugLevel >= 2)
                    {
                        printf(
                            "[SNII_COMPLETE|T=%d|a=%.8g|z=%.6f] "
                            "star=%d starID=%llu age=%.6f Myr\n",
                            All.ThisTask,
                            (double)All.Time,
                            1.0 / All.Time - 1.0,
                            p,
                            (unsigned long long)
                                Sp->P[p].ID.get(),
                            age_Myr);
                    }
                }
                }
                else
                {
                    // Leave this tranche pending. The complete event will be
                    // retried later without a duplicated energy-reservoir
                    // contribution or orphaned SNII/LRN dust production.
                    snii_deposition_failures++;
                }
            }
        }


        // Do NOT simply mark an SSP complete at MAX_TYPEII_TIME.
        //
        // If a star ever missed one or more scheduled bins (for example
        // because it was temporarily inactive), continue releasing at most
        // one overdue tranche per feedback call until all four are done.
        // This preserves the integrated budget without recreating a single
        // large catch-up burst.


        // ================================================================
        // 2b) AGB enrichment.
        // ================================================================

        if(AGB_Yields.is_table_loaded() &&
           age_Myr >= MIN_AGB_TIME &&
           !(Sp->P[p].FeedbackFlag & 2))
        {
            agb_stars_eligible++;


            // ------------------------------------------------------------
            // Stellar particle represents an SSP.
            // ------------------------------------------------------------

            const double m_ssp_msun =
                stellar_birth_mass_msun(
                    Sp,
                    p);

            double Z_star =
                Sp->P[p].Metallicity;


            // ------------------------------------------------------------
            // IMF-integrated MESA AGB yield.
            // ------------------------------------------------------------

            AGBIMFYield agb =
                AGB_Yields.get_imf_yields(
                    Z_star);


            // ------------------------------------------------------------
            // Validate yield table.
            // ------------------------------------------------------------

            if(!std::isfinite(agb.Z_per_Mstar)  ||
               !std::isfinite(agb.C_per_Mstar)  ||
               !std::isfinite(agb.N_per_Mstar)  ||
               !std::isfinite(agb.O_per_Mstar)  ||
               !std::isfinite(agb.Ne_per_Mstar) ||
               !std::isfinite(agb.Mg_per_Mstar))
            {
                Terminate(
                    "Non-finite IMF-integrated AGB yield");
            }

            if(agb.Z_per_Mstar  < 0.0 ||
               agb.C_per_Mstar  < 0.0 ||
               agb.N_per_Mstar  < 0.0 ||
               agb.O_per_Mstar  < 0.0 ||
               agb.Ne_per_Mstar < 0.0 ||
               agb.Mg_per_Mstar < 0.0)
            {
                Terminate(
                    "Negative IMF-integrated AGB yield");
            }


            // ------------------------------------------------------------
            // Actual AGB metal ejecta represented by this SSP.
            // ------------------------------------------------------------

            double MZ_msun =
                agb.Z_per_Mstar *
                m_ssp_msun;

            double C_msun =
                agb.C_per_Mstar *
                m_ssp_msun;

            double N_msun =
                agb.N_per_Mstar *
                m_ssp_msun;

            double O_msun =
                agb.O_per_Mstar *
                m_ssp_msun;

            double Ne_msun =
                agb.Ne_per_Mstar *
                m_ssp_msun;

            double Mg_msun =
                agb.Mg_per_Mstar *
                m_ssp_msun;


            if(MZ_msun > 0.0)
            {
                // --------------------------------------------------------
                // Total AGB metal ejecta.
                // --------------------------------------------------------

                double MZ_total_code =
                    MZ_msun *
                    SOLAR_MASS /
                    All.UnitMass_in_g;


                // Current model assumes negligible AGB thermal feedback.
                double E_erg  = 0.0;
                double E_code = 0.0;


                // --------------------------------------------------------
                // Composition-resolved AGB ejecta.
                // --------------------------------------------------------

                ElementYields elem_total;

                elem_total.C =
                    C_msun *
                    SOLAR_MASS /
                    All.UnitMass_in_g;

                elem_total.N =
                    N_msun *
                    SOLAR_MASS /
                    All.UnitMass_in_g;

                elem_total.O =
                    O_msun *
                    SOLAR_MASS /
                    All.UnitMass_in_g;

                elem_total.Ne =
                    Ne_msun *
                    SOLAR_MASS /
                    All.UnitMass_in_g;

                elem_total.Mg =
                    Mg_msun *
                    SOLAR_MASS /
                    All.UnitMass_in_g;

                // Current MESA table does not explicitly resolve Si/Fe.
                elem_total.Si = 0.0;
                elem_total.Fe = 0.0;


                // --------------------------------------------------------
                // Yield sanity check.
                // --------------------------------------------------------

                double tracked_msun =
                    C_msun +
                    N_msun +
                    O_msun +
                    Ne_msun +
                    Mg_msun;

                if(tracked_msun >
                   MZ_msun * 1.000001)
                {
                    printf(
                        "[AGB_YIELD_ERROR] "
                        "Z*=%.6g M*=%.6g "
                        "tracked=%.6e > "
                        "total_metals=%.6e Msun\n",
                        Z_star,
                        m_ssp_msun,
                        tracked_msun,
                        MZ_msun);

                    Terminate(
                        "AGB elemental ejecta exceed total metal ejecta");
                }


                // --------------------------------------------------------
                // Partition AGB ejecta.
                //
                // Current AGB tables lack Si and Fe, so we cannot make a
                // fully species-conserving silicate component yet.
                //
                // The requested AGB condensation efficiency is partitioned into the
                // prescribed 60/40 carbonaceous/silicate birth mixture.
                //
                // Carbon is capped by the explicit C ejecta. Silicate is
                // constrained by explicit O, explicit Mg, and the untracked
                // metal residual used as a combined proxy for Si+Fe.
                // --------------------------------------------------------

                ElementYields elem_gas =
                    elem_total;

                double MZ_gas_code =
                    MZ_total_code;

#ifdef DUST
                StellarDustPartition agb_dust;

                if(All.DustParticlesPerAGB > 0)
                {
                    agb_dust =
                        partition_stellar_ejecta_for_dust(
                            MZ_total_code,
                            elem_gas,
                            All.DustYieldAGB,
                            /*target_carbon_fraction=*/0.60,
                            /*use_element_resolved_silicate_limits=*/false);

                    MZ_gas_code =
                        agb_dust.gas_metals;
                }
#endif

                // --------------------------------------------------------
                // Deposit only the gas-phase AGB ejecta.
                // --------------------------------------------------------

                bool ejecta_deposited =
                    apply_feedback_event(
                        Sp,
                        Tree,
                        D,
                        p,
                        /*is_SNII=*/false,
                        E_code,
                        MZ_gas_code,
                        elem_gas);


                if(ejecta_deposited)
                {
                    // Remove TOTAL AGB metal ejecta from the star.
                    double new_star_mass =
                        Sp->P[p].getMass() -
                        MZ_total_code;

                    if(new_star_mass <= 0.0)
                    {
                        Terminate(
                            "AGB feedback would remove entire stellar particle mass");
                    }

                    Sp->P[p].setMass(
                        new_star_mass);


#ifdef DUST
                    if(agb_dust.total_dust > 0.0)
                    {
                        double created_agb_dust =
                            create_dust_particles_from_feedback(
                                Sp,
                                p,
                                agb_dust.carbon_dust,
                                agb_dust.silicate_dust,
                                /*feedback_type=*/2);

                        double creation_tolerance =
                            1e-12 *
                            std::max(
                                agb_dust.total_dust,
                                1e-30);

                        if(std::abs(
                               created_agb_dust -
                               agb_dust.total_dust) >
                           creation_tolerance)
                        {
                            Terminate(
                                "AGB dust creation did not realize partitioned dust mass");
                        }
                    }
#endif

                    // Mark the AGB event complete only after gas actually
                    // received the gas-phase ejecta and any dust was spawned.
                    Sp->P[p].FeedbackFlag |= 2;

                    agb_event_count++;

                    agb_total_metals_g +=
                        MZ_msun *
                        SOLAR_MASS;

                    agb_total_energy_erg +=
                        E_erg;

                    if(agb_event_count < 10 &&
                       All.ThisTask == 0)
                    {
                        printf(
                            "[AGB_YIELD] "
                            "Z*=%.5f Mstar=%.3e "
                            "yZ=%.3e yC=%.3e yN=%.3e "
                            "yO=%.3e yNe=%.3e yMg=%.3e\n",
                            Z_star,
                            m_ssp_msun,
                            agb.Z_per_Mstar,
                            agb.C_per_Mstar,
                            agb.N_per_Mstar,
                            agb.O_per_Mstar,
                            agb.Ne_per_Mstar,
                            agb.Mg_per_Mstar);
                    }
                }
                else
                {
                    // Leave FeedbackFlag bit 1 unset so this event can be
                    // retried after the gas-neighbor environment changes.
                    agb_deposition_failures++;
                }
            }
        }


        // Track all stars checked for AGB eligibility.
        if(AGB_Yields.is_table_loaded())
            agb_stars_checked++;

    } // end star loop


    // ====================================================================
    // 3) Periodic diagnostics.
    // ====================================================================

    static int feedback_call_count = 0;

    feedback_call_count++;

    if(feedback_call_count % 10 == 0 &&
       All.ThisTask == 0)
    {
        FB_PRINT(
            "\n[FB_STATS|Step=%d|a=%.6f z=%.3f]\n",
            feedback_call_count,
            All.Time,
            1.0 / All.Time - 1.0);

        FB_PRINT(
            " Stars local: %d\n",
            n_stars_local);

        FB_PRINT(
            " SNII pending failures (no gas): %d\n",
            snii_deposition_failures);

        FB_PRINT(
            " AGB (this window): "
            "checked=%d eligible=%d events=%d failed_no_gas=%d\n",
            agb_stars_checked,
            agb_stars_eligible,
            agb_event_count,
            agb_deposition_failures);

        if(agb_event_count > 0)
        {
            FB_PRINT(
                "   metals=%.3e Msun "
                " energy=%.3e erg "
                " avg=%.4f Msun/event\n",
                agb_total_metals_g / SOLAR_MASS,
                agb_total_energy_erg,
                (agb_total_metals_g / SOLAR_MASS) /
                    agb_event_count);
        }

#ifdef DUST
        if(agb_dust_partition_count > 0)
        {
            FB_PRINT(
                "   AGB dust partitions=%lld carbon_limited=%lld "
                "below_min=%lld\n",
                agb_dust_partition_count,
                agb_dust_carbon_limited_count,
                agb_dust_below_min_count);

            FB_PRINT(
                "   AGB silicate limiter: target=%lld O=%lld Mg=%lld "
                "SiFe_proxy=%lld\n",
                agb_dust_silicate_target_count,
                agb_dust_silicate_O_count,
                agb_dust_silicate_Mg_count,
                agb_dust_silicate_proxy_count);
        }
#endif

        FB_PRINT(
            " Reservoirs: %d stars total=%.3e\n",
            stars_with_reservoir,
            reservoir_total_energy);
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