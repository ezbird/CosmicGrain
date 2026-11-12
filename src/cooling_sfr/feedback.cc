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
#include <cfloat>
#include <cstdint>
#include <vector>
#include <utility>
#include <unordered_map>
#include <string>
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
#include "agb_feedback_math.h"
#include "cg_distributed.h"
#include <exception>
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

// Gadget code units carry h: one code mass unit is UnitMass_in_g/h grams and
// one code energy unit is UnitEnergy_in_cgs/h erg.  Ejecta and energies
// computed per solar mass must use these, or the SN energy handed to the dust
// shock model and every printed mass/energy is off by powers of h.
static inline double fb_code_mass_in_g(void) { return All.UnitMass_in_g / All.HubbleParam; }
static inline double fb_code_energy_in_erg(void) { return All.UnitEnergy_in_cgs / All.HubbleParam; }

// Counter-based feedback RNG.  The implementation is kept at the end of
// this file; this explicit tuple interface prevents collisions and correlations
// caused by adding star ID, timestep, and draw counter into one integer.
static double feedback_random_u01(
    std::uint64_t star_id,
    std::uint64_t timestep,
    std::uint64_t draw_counter,
    std::uint64_t stream_id);

// Independent deterministic streams for logically different decisions.
// Keeping them separate ensures that adding a draw to one decision path does
// not silently reuse the same random value in another path.
static constexpr std::uint64_t RNG_STREAM_NEIGHBOR_SELECTION =
    UINT64_C(0x4e47425f53454c45); // "NGB_SELE"
static constexpr std::uint64_t RNG_STREAM_FRACTIONAL_PACKET =
    UINT64_C(0x4245524e4f554c4c); // "BERNOULL"

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

struct StellarDustPartition
{
    double carbon_dust   = 0.0;
    double silicate_dust = 0.0;
    double total_dust    = 0.0;
    double gas_metals    = 0.0;
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
        fb_code_mass_in_g() /
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
    bool use_element_resolved_silicate_limits,
    bool apply_mass_floor = true,
    bool record_diagnostics = true)
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

    if(apply_mass_floor && requested_total_dust <
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

        agb_dust_partition_count += record_diagnostics ? 1 : 0;

        if(result.carbon_dust < requested_carbon * (1.0 - 1e-12))
            agb_dust_carbon_limited_count += record_diagnostics ? 1 : 0;

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
            agb_dust_silicate_target_count += record_diagnostics ? 1 : 0;
        else if(silicate_from_O <= silicate_from_Mg &&
                silicate_from_O <= silicate_from_proxy)
            agb_dust_silicate_O_count++;
        else if(silicate_from_Mg <= silicate_from_proxy)
            agb_dust_silicate_Mg_count++;
        else
            agb_dust_silicate_proxy_count += record_diagnostics ? 1 : 0;
    }

    result.total_dust =
        result.carbon_dust +
        result.silicate_dust;

    // The requested mass may exceed the numerical particle-mass floor while
    // the reservoir-limited mass does not. In that case leave all ejecta in
    // the gas phase rather than removing metals without spawning dust.
    if(apply_mass_floor && result.total_dust > 0.0 &&
       result.total_dust < FB_MIN_STELLAR_DUST_MASS)
    {
        if(!use_element_resolved_silicate_limits)
            agb_dust_below_min_count += record_diagnostics ? 1 : 0;

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

// ----------------------- Distributed feedback kernels -----------------------
//
// SNII, reservoir, and AGB events deposit into the gas within a kernel around
// the star.  That gas may belong to any MPI task, so each batch of events is
// processed collectively (see cg_distributed.h):
//
//   1. every task sees every event and reports its nearest gas, giving the
//      global nearest gas and with it the initial kernel scale;
//   2. kernels are enlarged in lockstep until they hold MIN_FEEDBACK_NEIGHBORS
//      gas particles counted over all tasks;
//   3. each task sends its gas inside the final kernel to the event owner,
//      which keeps the MAX_KERNEL_NEIGHBORS nearest (ties by ID) and makes the
//      same decisions as the former single-task code;
//   4. the resulting deposits go to the gas owners, which apply them in
//      (star ID, event kind) order.
//
// The neighbour set and the stochastic choices therefore no longer depend on
// where the domain boundaries fall.  Events in one batch see the gas state
// from before the batch; the old serial code let later events see earlier
// deposits, which only matters when two events share recipients in one step.

namespace cg_fb_dist
{
enum EventKind
{
  KIND_RESERVOIR = 0,  // retry of banked SNII energy; ordered before new SNII
  KIND_SNII      = 1,
  KIND_AGB       = 2
};

struct EventPosition
{
  MyIntPosType pos[3];
  int slot;  // index in the owner's local event list
};

struct Candidate
{
  std::uint64_t id;
  double r2, mass, u, ne;
  int task, index, slot;
};

static inline bool closer(const Candidate &a, const Candidate &b) { return a.r2 < b.r2 || (a.r2 == b.r2 && a.id < b.id); }

struct Kernel
{
  bool found = false;  // some gas exists; nearest is valid
  double radius = 0;   // final search radius = 2h
  cg_dist::NearestRecord nearest;
  std::vector<Candidate> cand;  // sorted by (r2, id), at most MAX_KERNEL_NEIGHBORS
};

// Gas eligible as a feedback recipient.  Rebuilt for every feedback call.
static cg_dist::LocalTree gas_tree;

static void build_gas_tree(simparticles *Sp)
{
  gas_tree.build(Sp, 0, Sp->NumGas, [Sp](int j) {
    return Sp->P[j].getType() == 0 && Sp->P[j].getMass() > 0 && Sp->SphP[j].Hsml > 0 && std::isfinite((double)Sp->SphP[j].Hsml);
  });
}

static double kernel_radius_cap(double initial_radius)
{
  // Positions are comoving kpc/h; FEEDBACK_MAX_RADIUS_PKPC is physical.
  const double physical_cap = All.ComovingIntegrationOn
                                  ? FEEDBACK_MAX_RADIUS_PKPC * All.HubbleParam / std::max((double)All.Time, 1.0e-8)
                                  : FEEDBACK_MAX_RADIUS_PKPC * All.HubbleParam;
  // Never shrink a kernel that already exceeds the cap.
  return std::max(initial_radius, std::min(FEEDBACK_MAX_RADIUS_FACTOR * initial_radius, physical_cap));
}

// Collective: every task must call this, with its own (possibly empty) list.
static std::vector<Kernel> find_kernels(simparticles *Sp, const std::vector<EventPosition> &mine, MPI_Comm comm)
{
  std::vector<int> owner;
  const std::vector<EventPosition> all = cg_dist::allgather(mine, comm, &owner);
  const size_t n_events                = all.size();
  std::vector<Kernel> out(mine.size());
  if(n_events == 0)
    return out;

  int ntask, this_task;
  MPI_Comm_size(comm, &ntask);
  MPI_Comm_rank(comm, &this_task);

  // 1. Global nearest gas; its smoothing length sets the initial kernel.
  std::vector<cg_dist::NearestRecord> nearest(n_events);
  for(size_t e = 0; e < n_events; e++)
    {
      cg_dist::NearestRecord &rec = nearest[e];
      rec.r2                      = DBL_MAX;
      rec.id                      = UINT64_MAX;
      rec.task = rec.index = -1;
      for(double &x : rec.payload)
        x = 0;
      int j;
      double r2;
      if(gas_tree.nearest(Sp, all[e].pos, j, r2))
        {
          rec.r2         = r2;
          rec.id         = (std::uint64_t)Sp->P[j].ID.get();
          rec.task       = this_task;
          rec.index      = j;
          rec.payload[0] = Sp->SphP[j].Hsml;
          rec.payload[1] = Sp->SphP[j].Density * All.cf_a3inv;  // physical code density
        }
    }
  cg_dist::nearest_over_tasks(nearest, comm);

  // 2. Enlarge under-populated kernels, counting gas over all tasks.
  std::vector<double> radius(n_events, 0), max_radius(n_events, 0);
  std::vector<char> growing(n_events, 0);
  for(size_t e = 0; e < n_events; e++)
    if(nearest[e].task >= 0)
      {
        radius[e]     = 2 * nearest[e].payload[0];
        max_radius[e] = kernel_radius_cap(radius[e]);
        growing[e]    = 1;
      }
  for(int iter = 0;; iter++)
    {
      std::vector<double> count(n_events, 0);
      for(size_t e = 0; e < n_events; e++)
        if(growing[e])
          gas_tree.radius(Sp, all[e].pos, radius[e], [&](int, double) { count[e] += 1; });
      cg_dist::sum_over_tasks(count, comm);
      bool any = false;
      for(size_t e = 0; e < n_events; e++)
        {
          if(!growing[e])
            continue;
          if(count[e] >= MIN_FEEDBACK_NEIGHBORS || radius[e] >= max_radius[e] || iter >= FEEDBACK_MAX_KERNEL_ITERS)
            {
              growing[e] = 0;
              continue;
            }
          const double next = std::min(radius[e] * FEEDBACK_KERNEL_GROWTH, max_radius[e]);
          if(!(next > radius[e]))
            {
              growing[e] = 0;
              continue;
            }
          radius[e] = next;
          any       = true;
        }
      if(!any)
        break;
    }

  // 3. Send each task's nearest gas inside the final kernel to the owner.
  //    The global nearest K are contained in the union of each task's K.
  std::vector<std::vector<Candidate>> send(ntask);
  std::vector<Candidate> local;
  for(size_t e = 0; e < n_events; e++)
    {
      if(nearest[e].task < 0)
        continue;
      local.clear();
      gas_tree.radius(Sp, all[e].pos, radius[e], [&](int j, double d2) {
        Candidate c;
        c.id    = (std::uint64_t)Sp->P[j].ID.get();
        c.r2    = d2;
        c.mass  = Sp->P[j].getMass();
        c.u     = Sp->get_utherm_from_entropy(j);
        c.ne    = Sp->SphP[j].Ne;
        c.task  = this_task;
        c.index = j;
        c.slot  = all[e].slot;
        local.push_back(c);
      });
      if(local.size() > (size_t)MAX_KERNEL_NEIGHBORS)
        {
          std::partial_sort(local.begin(), local.begin() + MAX_KERNEL_NEIGHBORS, local.end(), closer);
          local.resize(MAX_KERNEL_NEIGHBORS);
        }
      send[owner[e]].insert(send[owner[e]].end(), local.begin(), local.end());
    }
  const std::vector<Candidate> got = cg_dist::exchange(send, comm);

  size_t first = 0;
  while(first < n_events && owner[first] < this_task)
    first++;
  for(size_t s = 0; s < mine.size(); s++)
    {
      out[s].nearest = nearest[first + s];
      out[s].found   = nearest[first + s].task >= 0;
      out[s].radius  = radius[first + s];
    }
  for(const Candidate &c : got)
    {
      if(c.slot < 0 || (size_t)c.slot >= out.size())
        Terminate("feedback kernel candidate for unknown event slot %d", c.slot);
      out[c.slot].cand.push_back(c);
    }
  for(Kernel &k : out)
    {
      std::sort(k.cand.begin(), k.cand.end(), closer);
      if(k.cand.size() > (size_t)MAX_KERNEL_NEIGHBORS)
        k.cand.resize(MAX_KERNEL_NEIGHBORS);
    }

  // Self-check on the first batches: below the cap the owner must hold every
  // gas particle inside the kernel, and the nearest gas must come first.
  static int verify_remaining = 20;
  if(verify_remaining > 0)
    {
      verify_remaining--;
      std::vector<double> count(n_events, 0);
      for(size_t e = 0; e < n_events; e++)
        if(nearest[e].task >= 0)
          gas_tree.radius(Sp, all[e].pos, radius[e], [&](int, double) { count[e] += 1; });
      cg_dist::sum_over_tasks(count, comm);
      long long bad[3] = {0, 0, 0};
      for(size_t s = 0; s < mine.size(); s++)
        {
          const Kernel &k = out[s];
          if(!k.found)
            continue;
          const double c = count[first + s];
          if(c <= MAX_KERNEL_NEIGHBORS && (double)k.cand.size() != c)
            bad[0]++;
          if(c > MAX_KERNEL_NEIGHBORS && k.cand.size() != (size_t)MAX_KERNEL_NEIGHBORS)
            bad[1]++;
          if(k.nearest.r2 <= k.radius * k.radius && (k.cand.empty() || k.cand[0].id != k.nearest.id))
            bad[2]++;
        }
      long long n_local = mine.size(), totals[4], local4[4] = {bad[0], bad[1], bad[2], n_local};
      MPI_Allreduce(local4, totals, 4, MPI_LONG_LONG, MPI_SUM, comm);
      if(this_task == 0)
        printf("[FB_KERNEL_VERIFY] %lld events on %d tasks: %lld incomplete kernels, %lld wrong caps, %lld nearest-gas "
               "mismatches\n",
               totals[3], ntask, totals[0], totals[1], totals[2]);
    }
  return out;
}

// Local index of gas particle `id` at `index`, checked against the ID it
// had when the candidate was collected.
static inline int checked_gas_index(simparticles *Sp, int index, std::uint64_t id)
{
  if(index < 0 || index >= Sp->NumGas || Sp->P[index].getType() != 0 || (std::uint64_t)Sp->P[index].ID.get() != id)
    Terminate("feedback deposit for gas ID %llu no longer at index %d", (unsigned long long)id, index);
  return index;
}

// ---------------------------- SNII and reservoir ----------------------------

struct ThermalEvent
{
  int star;  // local index
  int kind;  // KIND_RESERVOIR or KIND_SNII
  double E_code, MZ_code;
  ElementYields elem;
};

struct GasDeposit
{
  std::uint64_t star_id, gas_id;
  int kind, index;
  int heat;  // receives one DELTA_T_TARGET jump
  double dMZ, dElem[7];  // C N O Ne Mg Si Fe
};

/*
 * Owner-side decisions for one event, unchanged from the former serial code:
 *
 * Chemistry: metals and elements are spread over the whole kernel by
 * kernel-weighted mass, whether or not a particle is heated.
 *
 * Heat: neighbours are sampled without replacement by kernel weight, in
 * (distance, ID) order, with the counter-based RNG keyed on the star ID and
 * step.  Each chosen particle receives exactly DELTA_T_TARGET, sized with its
 * own mu (from Ne) and its post-enrichment mass.  Full packets are deposited
 * deterministically, the final fractional one with a Bernoulli draw that
 * consumes it either way.  Only energy left after running out of neighbours
 * is returned in E_unspent (for the reservoir).
 *
 * Returns false (event stays pending) if the kernel holds no weight.
 */
static bool plan_thermal_event(simparticles *Sp, const Kernel &k, const ThermalEvent &ev, std::vector<std::vector<GasDeposit>> &send,
                               double &E_used, double &E_unspent)
{
  E_used = E_unspent = 0;
  const int n        = (int)k.cand.size();
  if(!k.found || n == 0)
    return false;

  const double h = 0.5 * k.radius;
  std::vector<double> w(n);
  double w_sum = 0;
  for(int i = 0; i < n; i++)
    {
      w[i] = k.cand[i].mass * cubic_spline_kernel(std::sqrt(k.cand[i].r2), h);
      if(!std::isfinite(w[i]) || w[i] < 0)
        Terminate("Invalid feedback kernel weight");
      w_sum += w[i];
    }
  if(!std::isfinite(w_sum) || !(w_sum > 0))
    return false;

  const std::uint64_t star_id = (std::uint64_t)Sp->P[ev.star].ID.get();
  std::vector<GasDeposit> dep(n);
  for(int i = 0; i < n; i++)
    {
      dep[i]         = GasDeposit{};
      dep[i].star_id = star_id;
      dep[i].gas_id  = k.cand[i].id;
      dep[i].kind    = ev.kind;
      dep[i].index   = k.cand[i].index;
    }

#ifdef METALS
  if(ev.MZ_code > 0)
    {
      const double elem[7] = {ev.elem.C, ev.elem.N, ev.elem.O, ev.elem.Ne, ev.elem.Mg, ev.elem.Si, ev.elem.Fe};
      for(int i = 0; i < n; i++)
        {
          const double f = w[i] / w_sum;
          dep[i].dMZ     = ev.MZ_code * f;
          for(int x = 0; x < 7; x++)
            dep[i].dElem[x] = elem[x] * f;
        }
    }
#endif

  if(ev.E_code > 0)
    {
      double E_remaining = ev.E_code;
      std::vector<int> available(n);
      for(int i = 0; i < n; i++)
        available[i] = i;
      unsigned long long counter = 0;

      while(E_remaining > 0 && !available.empty())
        {
          double available_sum = 0;
          for(int i : available)
            available_sum += w[i];
          if(available_sum <= 0)
            break;

          const double target =
              feedback_random_u01(star_id, (std::uint64_t)All.NumCurrentTiStep, counter++, RNG_STREAM_NEIGHBOR_SELECTION) * available_sum;
          double cumulative = 0;
          int chosen        = (int)available.size() - 1;
          for(int a = 0; a < (int)available.size(); a++)
            {
              cumulative += w[available[a]];
              if(cumulative >= target)
                {
                  chosen = a;
                  break;
                }
            }
          const int i = available[chosen];
          available.erase(available.begin() + chosen);

          const Candidate &c = k.cand[i];
          if(c.u <= 0 || !std::isfinite(c.u))
            {
              FB_PRINT("WARNING: gas ID %llu has invalid u_old=%g\n", (unsigned long long)c.id, c.u);
              continue;
            }
          const double E_heat = (c.mass + dep[i].dMZ) * fb_code_mass_in_g() * c_v(mu_from_ne(c.ne)) * DELTA_T_TARGET /
                                fb_code_energy_in_erg();
          if(E_heat <= 0 || !std::isfinite(E_heat))
            continue;

          bool heat = false;
          if(E_remaining >= E_heat)
            {
              heat = true;
              E_remaining -= E_heat;
            }
          else
            {
              const double probability = fb_clamp(E_remaining / E_heat, 0.0, 1.0);
              const double draw =
                  feedback_random_u01(star_id, (std::uint64_t)All.NumCurrentTiStep, counter++, RNG_STREAM_FRACTIONAL_PACKET);
              heat        = draw < probability;
              E_remaining = 0;  // statistically consumed either way
            }
          if(heat)
            {
              dep[i].heat = 1;
              E_used += E_heat;
            }
        }
      E_unspent = std::max(0.0, E_remaining);
    }

  for(int i = 0; i < n; i++)
    if(dep[i].dMZ > 0 || dep[i].heat)
      send[k.cand[i].task].push_back(dep[i]);
  return true;
}

static void apply_gas_deposit(simparticles *Sp, const GasDeposit &d)
{
  const int j = checked_gas_index(Sp, d.index, d.gas_id);

#ifdef METALS
  if(d.dMZ > 0)
    {
      const double old_mass = Sp->P[j].getMass();
      const double new_mass = old_mass + d.dMZ;
      MyFloat *fraction[7]  = {&Sp->SphP[j].GasCarbonMassFraction,    &Sp->SphP[j].GasNitrogenMassFraction,
                               &Sp->SphP[j].GasOxygenMassFraction,    &Sp->SphP[j].GasNeonMassFraction,
                               &Sp->SphP[j].GasMagnesiumMassFraction, &Sp->SphP[j].GasSiliconMassFraction,
                               &Sp->SphP[j].GasIronMassFraction};
      for(int x = 0; x < 7; x++)
        *fraction[x] = (old_mass * *fraction[x] + d.dElem[x]) / new_mass;
      Sp->P[j].setMass(new_mass);
      Sp->SphP[j].MassMetallicity += d.dMZ;
      Sp->SphP[j].Metallicity = Sp->SphP[j].MassMetallicity / new_mass;
      Sp->P[j].Metallicity    = Sp->SphP[j].Metallicity;
    }
#endif

  if(!d.heat)
    return;
  const double u_old = Sp->get_utherm_from_entropy(j);
  if(u_old <= 0 || !std::isfinite(u_old))
    {
      FB_PRINT("WARNING: Particle %d has invalid u_old=%g\n", j, u_old);
      return;
    }
  const double ne_before = Sp->SphP[j].Ne;
  const double mu        = mu_from_ne(ne_before);
  const double du = c_v(mu) * DELTA_T_TARGET / (All.UnitVelocity_in_cm_per_s * All.UnitVelocity_in_cm_per_s);
  Sp->set_entropy_from_utherm(u_old + du, j);
  set_thermodynamic_variables_safe(Sp, j);

  if(All.FeedbackDebugLevel >= 2)
    printf("[SN_STOCHASTIC|T=%d|a=%.8g|z=%.6f] source=%s gas=%d starID=%llu gasID=%llu "
           "DeltaT_target=%.6e K mu=%.6f Tbefore=%.6e K Tafter=%.6e K\n",
           All.ThisTask, (double)All.Time, 1.0 / All.Time - 1.0, d.kind == KIND_RESERVOIR ? "RESERVOIR" : "EVENT", j,
           (unsigned long long)d.star_id, (unsigned long long)d.gas_id, DELTA_T_TARGET, mu, ucode_to_TK(u_old, ne_before),
           ucode_to_TK(Sp->get_utherm_from_entropy(j), Sp->SphP[j].Ne));
}

// Collective.  Deposits a batch of SNII/reservoir events; ok[i] reports
// whether event i found gas (otherwise it stays pending and nothing changed),
// and kernels[i] the kernel it used.
static void deposit_thermal_events(simparticles *Sp, const std::vector<ThermalEvent> &events, std::vector<char> &ok,
                                   std::vector<Kernel> &kernels, MPI_Comm comm)
{
  int ntask;
  MPI_Comm_size(comm, &ntask);

  std::vector<EventPosition> positions(events.size());
  for(size_t i = 0; i < events.size(); i++)
    {
      for(int k = 0; k < 3; k++)
        positions[i].pos[k] = Sp->P[events[i].star].IntPos[k];
      positions[i].slot = (int)i;
    }
  kernels = find_kernels(Sp, positions, comm);

  ok.assign(events.size(), 0);
  std::vector<std::vector<GasDeposit>> send(ntask);
  for(size_t i = 0; i < events.size(); i++)
    {
      const ThermalEvent &ev = events[i];
      double E_used, E_unspent;
      ok[i] = plan_thermal_event(Sp, kernels[i], ev, send, E_used, E_unspent);
      if(!ok[i])
        {
          if(ev.kind == KIND_SNII && All.FeedbackDebugLevel >= 2)
            printf("[FEEDBACK_NO_GAS|T=%d] Star %d: no gas neighbors; retaining %.6e code mass and %.6e code energy in pending event\n",
                   All.ThisTask, ev.star, ev.MZ_code, ev.E_code);
          continue;  // reservoir stays banked; SNII tranche stays pending
        }
      diag_add_Edep(E_used * fb_code_energy_in_erg());
      if(ev.kind == KIND_RESERVOIR)
        {
          Sp->P[ev.star].EnergyReservoir = E_unspent;
          diag_add_EfromRes(E_used * fb_code_energy_in_erg());
        }
      else
        {
          diag_add_sn(ev.E_code * fb_code_energy_in_erg());
          if(E_unspent > 0)
            {
              Sp->P[ev.star].EnergyReservoir += E_unspent;
              diag_add_EtoRes(E_unspent * fb_code_energy_in_erg());
            }
        }
    }

  std::vector<GasDeposit> got = cg_dist::exchange(send, comm);
  std::stable_sort(got.begin(), got.end(), [](const GasDeposit &a, const GasDeposit &b) {
    return a.star_id < b.star_id || (a.star_id == b.star_id && a.kind < b.kind);
  });
  for(const GasDeposit &d : got)
    apply_gas_deposit(Sp, d);
}
}  // namespace cg_fb_dist

// ---------------------------- Public API ------------------------------------

// CG07_AGB_TRANSACTION_PROBE_V1: observations only, no particle writes.
// Current thermal prediction at the stored density; density-entropy SPH only.
static inline double agb_current_predicted_utherm(simparticles *Sp,int j) {
  const double factor=pow(Sp->SphP[j].Density*All.cf_a3inv,GAMMA_MINUS1)/GAMMA_MINUS1;
  return Sp->SphP[j].EntropyPred*factor;
}

namespace cg_agb_probe {
enum Metric {
  gas, metals, mix_energy, ejecta_energy, internal_added, dense_deposit,
  sf_deposit, inactive_deposit, hot_before, hot_after, recipients,
  transactions, specific_u_change_mass, prediction_gap_mass,
  velocity_prediction_gap_energy, radius_mass, relative_v2_mass,
  tracked_mass, tracked_dense_before, tracked_dense_after,
  tracked_sf_after_cooling, tracked_rho_ratio_mass,
  tracked_u_before_mass, tracked_u_after_density_mass,
  tracked_u_after_cooling_mass, tracked_count, dropped_tracking_mass,
  invalid_tracking_mass, metric_count
};
static double sums[metric_count] = {};
struct Pending {
  double dm, rho, u;
  bool dense;
  int stage;
};
static std::unordered_map<std::uint64_t,Pending> pending;
static const size_t max_pending = 8192;

static double temperature(simparticles *Sp,int j,double u) {
  const double X=HYDROGEN_MASSFRAC;
  const double mu=4.0/(1+3*X+4*X*Sp->SphP[j].Ne);
  return GAMMA_MINUS1*u*All.UnitVelocity_in_cm_per_s*
      All.UnitVelocity_in_cm_per_s*mu*PROTONMASS/BOLTZMANN;
}

static void record(simparticles *Sp,int j,double dm,double distance,
                   double uold,const agb_feedback::MixedGas &mixed,
                   const double ve[3]) {
  if(!(dm>0)) return;
  const double m=Sp->P[j].getMass();
  const double rho=Sp->SphP[j].Density;
  const double vfac=All.ComovingIntegrationOn?All.cf_a2inv:1.;
  const double a=All.ComovingIntegrationOn?All.Time:1.;
  double rel2=0,gap2=0;
  for(int k=0;k<3;++k) {
    const double dv=Sp->SphP[j].VelPred[k]-ve[k];rel2+=dv*dv;
    const double dg=Sp->P[j].Vel[k]-Sp->SphP[j].VelPred[k];gap2+=dg*dg;
  }
  const bool dense=rho*All.cf_a3inv>=All.PhysDensThresh;
  sums[gas]+=dm;
  sums[mix_energy]+=.5*m*dm/(m+dm)*rel2*vfac;
  sums[ejecta_energy]+=dm*All.MinEgySpec;
  sums[internal_added]+=(m+dm)*mixed.u-m*uold;
  sums[dense_deposit]+=dense?dm:0;
  sums[sf_deposit]+=Sp->SphP[j].Sfr>0?dm:0;
  sums[inactive_deposit]+=Sp->TimeBinSynchronized[Sp->P[j].getTimeBinHydro()]?0:dm;
  sums[hot_before]+=temperature(Sp,j,uold)>=1e5?dm:0;
  sums[hot_after]+=temperature(Sp,j,mixed.u)>=1e5?dm:0;
  sums[recipients]+=1;
  sums[specific_u_change_mass]+=dm*(mixed.u-uold);
  sums[prediction_gap_mass]+=dm*std::abs((double)Sp->SphP[j].EntropyPred-
      (double)Sp->SphP[j].Entropy)/std::max(std::abs((double)Sp->SphP[j].Entropy),1e-100);
  sums[velocity_prediction_gap_energy]+=.5*dm*gap2*vfac;
  sums[radius_mass]+=dm*distance*a/All.HubbleParam;
  sums[relative_v2_mass]+=dm*rel2*vfac;
  const std::uint64_t id=(std::uint64_t)Sp->P[j].ID.get();
  auto it=pending.find(id);
  if(it==pending.end() && pending.size()>=max_pending) {
    sums[dropped_tracking_mass]+=dm;return;
  }
  double carried=it==pending.end()?0:it->second.dm;
  // Multiple deposits before an active update are combined. The baseline
  // state below is the latest deposit's post-mix state, not the first one.
  pending[id]={carried+dm,rho,mixed.u,dense,0};
}
}

void agb_probe_before_domain(void) {
  for(const auto &kv:cg_agb_probe::pending)
    cg_agb_probe::sums[cg_agb_probe::dropped_tracking_mass]+=kv.second.dm;
  cg_agb_probe::pending.clear();
}

void agb_probe_observe(simparticles *Sp,int stage) {
  using namespace cg_agb_probe;
  if(pending.empty()) return;
  for(int n=0;n<Sp->TimeBinsHydro.NActiveParticles;++n) {
    int j=Sp->TimeBinsHydro.ActiveParticleList[n];
    if(j<0 || j>=Sp->NumGas || Sp->P[j].getType()!=0) continue;
    auto it=pending.find((std::uint64_t)Sp->P[j].ID.get());
    if(it==pending.end()) continue;
    Pending &p=it->second;
    const double rho=Sp->SphP[j].Density,u=agb_current_predicted_utherm(Sp,j);
    if(!std::isfinite(rho)||rho<=0||!std::isfinite(u)||u<=0) {
      sums[invalid_tracking_mass]+=p.dm;pending.erase(it);continue;
    }
    if(stage==1 && p.stage==0) {
      sums[tracked_mass]+=p.dm;
      sums[tracked_dense_before]+=p.dense?p.dm:0;
      sums[tracked_dense_after]+=rho*All.cf_a3inv>=All.PhysDensThresh?p.dm:0;
      sums[tracked_rho_ratio_mass]+=p.dm*rho/p.rho;
      sums[tracked_u_before_mass]+=p.dm*p.u;
      sums[tracked_u_after_density_mass]+=p.dm*u;
      sums[tracked_count]+=1;p.stage=1;
    } else if(stage==2 && p.stage==1) {
      sums[tracked_sf_after_cooling]+=Sp->SphP[j].Sfr>0?p.dm:0;
      sums[tracked_u_after_cooling_mass]+=p.dm*u;
      pending.erase(it);
    }
  }
}

void agb_probe_flush(simparticles *Sp,MPI_Comm comm) {
  using namespace cg_agb_probe;
  static double next_a=0;
  static bool header=false;
  if(All.Time<next_a && All.Time<All.TimeMax) return;
  next_a=All.Time+.01;
  double global[metric_count]={};
  MPI_Reduce(sums,global,metric_count,MPI_DOUBLE,MPI_SUM,0,comm);
  // Instantaneous stored-state census (not a time integral or unique-event
  // count). Star mass includes pending AGB dust. This precedes dust evolution.
  double phase[9]={},gphase[9]={};
  for(int j=0;j<Sp->NumGas;++j) if(Sp->P[j].getType()==0) {
    double m=Sp->P[j].getMass();phase[0]+=m;
    if(Sp->SphP[j].Density*All.cf_a3inv>=All.PhysDensThresh)phase[1]+=m;
    if(Sp->SphP[j].Sfr>0)phase[2]+=m;
    phase[3]+=Sp->SphP[j].Sfr;
    const double u=Sp->get_utherm_from_entropy(j);
    if(std::isfinite(u) && u>0)phase[4]+=m*u;
  }
  for(int j=0;j<Sp->NumPart;++j) {
    if(Sp->P[j].getType()==4)phase[5]+=Sp->P[j].getMass();
    if(Sp->P[j].getType()==6)phase[6]+=Sp->P[j].getMass();
  }
  for(const auto &kv:pending)phase[7]+=kv.second.dm;
  phase[8]=(double)pending.size();
  MPI_Reduce(phase,gphase,9,MPI_DOUBLE,MPI_SUM,0,comm);
  if(All.ThisTask!=0)return;
  std::string file=std::string(All.OutputDir)+"/agb_coupling_diagnostics.csv";
  FILE *fp=fopen(file.c_str(),header?"a":"w");
  if(!fp)Terminate("Cannot open AGB coupling diagnostic CSV");
  const char *names[metric_count]={
    "gas_returned_Msun","gas_metals_returned_Msun","mixing_heat_erg",
    "ejecta_internal_energy_erg","recipient_internal_energy_added_erg",
    "returned_to_dense_Msun","returned_to_SFRpositive_Msun",
    "returned_to_inactive_Msun","returned_to_Tge1e5_before_Msun",
    "returned_to_Tge1e5_after_Msun","recipient_processing_events",
    "successful_transactions","dm_weighted_delta_u_erg",
    "dm_weighted_entropy_prediction_fraction_Msun",
    "gas_velocity_prediction_gap_energy_erg","dm_times_distance_Msun_pkpc",
    "dm_times_relative_v2_Msun_km2_s2","followed_return_Msun",
    "followed_dense_before_Msun","followed_dense_after_density_Msun",
    "followed_SFRpositive_after_cooling_Msun","dm_times_density_ratio_Msun",
    "dm_times_u_at_deposit_erg","dm_times_u_after_density_erg",
    "dm_times_u_after_cooling_erg","followed_gas_update_events",
    "dropped_followup_return_Msun","invalid_followup_return_Msun"};
  const double mf=fb_code_mass_in_g()/SOLAR_MASS;
  const double ef=fb_code_energy_in_erg();
  if(!header) {
    fprintf(fp,"a,mode");
    for(int i=0;i<metric_count;++i)fprintf(fp,",%s",names[i]);
    fprintf(fp,",gas_now_Msun,dense_gas_now_Msun,SFRpositive_gas_now_Msun,SFR_now_Msun_yr,gas_internal_energy_now_erg,stars_now_Msun,dust_now_Msun,pending_followup_return_Msun,pending_followup_gas_count\n");
    header=true;
  }
  fprintf(fp,"%.12g,%d",(double)All.Time,All.AGBFeedbackMode);
  for(int i=0;i<metric_count;++i) {
    double factor=mf;
    if(i==mix_energy||i==ejecta_energy||i==internal_added||i==specific_u_change_mass||
       i==velocity_prediction_gap_energy||i==tracked_u_before_mass||
       i==tracked_u_after_density_mass||i==tracked_u_after_cooling_mass)factor=ef;
    if(i==recipients||i==transactions||i==tracked_count)factor=1;
    if(i==relative_v2_mass)factor=mf*All.UnitVelocity_in_cm_per_s*All.UnitVelocity_in_cm_per_s/1e10;
    fprintf(fp,",%.12g",global[i]*factor);
  }
  for(int i=0;i<9;++i)fprintf(fp,",%.12g",gphase[i]*(i==3||i==8?1:(i==4?ef:mf)));
  fprintf(fp,"\n");fclose(fp);
}

// Age-dependent AGB transport. Pending dust remains in stellar mass until
// a fixed-mass packet is emitted; no mass disappears into an external ledger.
// Isolated correctness reference. All ranks enter events in the same ID order.
#include "rc1_gas_index.h"
#include "rc1_gas_geometry.h"
#if !defined(PERIODIC) || defined(LONG_X_BITS) || defined(LONG_Y_BITS) || defined(LONG_Z_BITS) || defined(GRAVITY_TALLBOX) || defined(ONEDIMS) || defined(TWODIMS) || defined(POSITIONS_IN_128BIT)
#error "Spatial AGB experiment supports only the frozen 3D cubic periodic configuration"
#endif
// Conservative reference policy: cumulative AGB transactions occur only in
// the end-of-step window after every occupied gravity and hydro bin has closed.
// No inactive stellar predictor or force-history reconstruction is assumed.
namespace cg_agb_sync {
static bool open=false,closed=false;
static integertime epoch=-1;
static MPI_Comm comm;
static long long commits=0;
static double max_noninitial_interval_myr=0;
}
void agb_begin_closed_feedback_window(simparticles *Sp,MPI_Comm comm) {
#if !defined(HIERARCHICAL_GRAVITY) || !defined(TREEPM_NOTIMESPLIT)
  Terminate("Synchronized AGB reference requires its audited hierarchical common-clock TreePM build");
#else
  int local=1,global=0;
  for(int p=0;p<Sp->NumPart;++p) {
    if(Sp->P[p].getMass()==0)continue;
    const int b=Sp->P[p].TimeBinGrav;
    if(b<0 || b>=TIMEBINS)Terminate("Invalid gravity bin in AGB clock gate");
    if(b>0 && (All.Ti_Current%(((integertime)1)<<b)!=0 || !Sp->TimeBinSynchronized[b]))local=0;
    if(Sp->P[p].getType()==0) {
      const int h=Sp->P[p].getTimeBinHydro();
      if(h<0 || h>=TIMEBINS)Terminate("Invalid hydro bin in AGB clock gate");
      if(h>0 && (All.Ti_Current%(((integertime)1)<<h)!=0 || !Sp->TimeBinSynchronized[h]))local=0;
    }
  }
  MPI_Allreduce(&local,&global,1,MPI_INT,MPI_MIN,comm);
  cg_agb_sync::open=true;cg_agb_sync::closed=global;cg_agb_sync::epoch=All.Ti_Current;cg_agb_sync::comm=comm;
  cg_agb_sync::commits=0;cg_agb_sync::max_noninitial_interval_myr=0;
#endif
}
void agb_end_closed_feedback_window(void) {
  if(!cg_agb_sync::open || cg_agb_sync::epoch!=All.Ti_Current)Terminate("Invalid AGB feedback window close");
  long long commits=0;double interval=0;
  MPI_Allreduce(&cg_agb_sync::commits,&commits,1,MPI_LONG_LONG_INT,MPI_SUM,cg_agb_sync::comm);
  MPI_Allreduce(&cg_agb_sync::max_noninitial_interval_myr,&interval,1,MPI_DOUBLE,MPI_MAX,cg_agb_sync::comm);
  int rank;MPI_Comm_rank(cg_agb_sync::comm,&rank);
  if(!cg_agb_sync::closed && commits)Terminate("AGB committed outside a fully closed state");
  if(rank==0)printf("[AGB_SYNC_WINDOW] ti=%lld closed=%d commits=%lld max_noninitial_interval_myr=%.12g\n",(long long)All.Ti_Current,(int)cg_agb_sync::closed,commits,interval);
  cg_agb_sync::open=false;
}

// ------------------------------ AGB (batched) -------------------------------
//
// All AGB transactions due in a closed feedback window are processed as one
// batch with the distributed kernels above, instead of one event at a time
// with several all-task collectives each.  Recipients, weights and the mixing
// are those of the former serial reference; events in the batch see the gas
// state from before the batch, and each gas owner applies its deposits in
// star-ID order (the order the serial scheduler used).

namespace cg_fb_dist
{
struct AGBPlan
{
  int star;
  double age, previous;
  double gas, metals, returned, packet;
  int max_packets;
  bool complete;
  ElementYields elem;
  StellarDustPartition dust;
  double donor_mass, ejecta_v[3];
};

struct AGBDeposit
{
  std::uint64_t star_id, gas_id;
  int index, slot;
  double dm, f, metals, distance;
  double elem[7];  // C N O Ne Mg Si Fe totals for the event
  double ejecta_v[3];
};

struct MomentumReply
{
  int slot;
  double p[3];
};

// Owner-side bookkeeping before the deposit.  Returns false if the star is
// not due.  Nothing in the star is modified here.
static bool plan_agb(simparticles *Sp, int p, double age, AGBPlan &plan)
{
  auto &star            = Sp->P[p];
  const double previous = star.AGBLastProcessedAgeMyr;
  if(!std::isfinite(age) || age < 0 || !std::isfinite(previous) || previous < 0 || previous > age + 1e-6 || (star.FeedbackFlag & 2))
    Terminate("Invalid/legacy age-dependent AGB state for star %lld", (long long)star.ID.get());
  if(age - previous < All.AGBFeedbackIntervalMyr && All.Time < All.TimeMax)
    return false;
  if(age <= previous)
    return false;
  const double Z     = star.AGBBirthMetallicity;
  const double birth = star.StellarBirthMass;
  if(!std::isfinite(Z) || Z < 0 || !std::isfinite(birth) || birth <= 0)
    Terminate("Invalid age-dependent AGB birth state");
  agb_stars_eligible++;
  AGBIMFYield current = AGB_Yields.get_cumulative_imf_yields(Z, age);
  AGBIMFYield before  = AGB_Yields.get_cumulative_imf_yields(Z, previous);
  AGBIMFYield full    = AGB_Yields.get_imf_yields(Z);

  plan          = AGBPlan{};
  plan.star     = p;
  plan.age      = age;
  plan.previous = previous;
  try
    {
      plan.metals   = birth * agb_feedback::increment(current.Z_per_Mstar, before.Z_per_Mstar);
      plan.returned = birth * agb_feedback::increment(current.M_return_per_Mstar, before.M_return_per_Mstar);
      plan.elem.C   = birth * agb_feedback::increment(current.C_per_Mstar, before.C_per_Mstar);
      plan.elem.N   = birth * agb_feedback::increment(current.N_per_Mstar, before.N_per_Mstar);
      plan.elem.O   = birth * agb_feedback::increment(current.O_per_Mstar, before.O_per_Mstar);
      plan.elem.Ne  = birth * agb_feedback::increment(current.Ne_per_Mstar, before.Ne_per_Mstar);
      plan.elem.Mg  = birth * agb_feedback::increment(current.Mg_per_Mstar, before.Mg_per_Mstar);
    }
  catch(const std::exception &e)
    {
      Terminate("AGB budget for star %lld: %s", (long long)star.ID.get(), e.what());
    }
  const ElementYields &elem = plan.elem;
  if(plan.metals > plan.returned * (1 + 1e-10) || elem.C + elem.N + elem.O + elem.Ne + elem.Mg > plan.metals * (1 + 1e-10))
    Terminate("AGB interval violates ejecta budget");
  plan.dust.gas_metals = plan.metals;
#ifdef DUST
  plan.max_packets = std::max(0, All.DustParticlesPerAGB);
  if(plan.max_packets > 0 && All.DustEnableCreation)
    {
      // Full-budget partition is an upper bound on the sum of interval
      // partitions (minimum availability limits). Use it only for packet size.
      ElementYields full_elem;
      full_elem.C   = birth * full.C_per_Mstar;
      full_elem.N   = birth * full.N_per_Mstar;
      full_elem.O   = birth * full.O_per_Mstar;
      full_elem.Ne  = birth * full.Ne_per_Mstar;
      full_elem.Mg  = birth * full.Mg_per_Mstar;
      auto full_dust = partition_stellar_ejecta_for_dust(birth * full.Z_per_Mstar, full_elem, All.DustYieldAGB, 0.60, false,
                                                         /*apply_mass_floor=*/false, /*record_diagnostics=*/false);
      plan.packet    = full_dust.total_dust / plan.max_packets;
      plan.dust = partition_stellar_ejecta_for_dust(plan.metals, plan.elem, All.DustYieldAGB, 0.60, false, /*apply_mass_floor=*/false);
    }
#endif
  plan.complete      = current.M_return_per_Mstar >= full.M_return_per_Mstar * (1 - 1e-12);
  const double total = All.AGBFeedbackMode == 2 ? plan.returned : plan.metals;
  plan.gas           = total - plan.dust.total_dust;
  if(!std::isfinite(star.AGBPendingCarbonMass) || star.AGBPendingCarbonMass < 0 || !std::isfinite(star.AGBPendingSilicateMass) ||
     star.AGBPendingSilicateMass < 0 || star.AGBDustPacketsCreated < 0 || star.AGBDustPacketsCreated > plan.max_packets)
    Terminate("Invalid AGB pending dust state");
  if(star.getMass() <= total + star.AGBPendingCarbonMass + star.AGBPendingSilicateMass)
    Terminate("AGB transaction would exhaust stellar mass");
  plan.donor_mass = star.getMass();
  for(int k = 0; k < 3; k++)
    plan.ejecta_v[k] = star.Vel[k];
  if(plan.gas != 0 && (!std::isfinite(plan.gas) || !std::isfinite(plan.dust.gas_metals) || plan.dust.gas_metals < 0 ||
                       plan.gas < plan.dust.gas_metals || plan.donor_mass <= plan.gas))
    Terminate("Invalid AGB gas/mass budget");
  return true;
}

// Gas-owner side: mix one deposit into gas particle d.index and return the
// momentum it gained, measured with the current velocity prediction.
static void apply_agb_deposit(simparticles *Sp, const AGBDeposit &d, double momentum[3])
{
  const int j = checked_gas_index(Sp, d.index, d.gas_id);
  if(!std::isfinite(Sp->SphP[j].Density) || Sp->SphP[j].Density <= 0)
    Terminate("Invalid density before AGB mass deposition");
  const double old = Sp->P[j].getMass();
  if(!std::isfinite(Sp->SphP[j].MassMetallicity) || Sp->SphP[j].MassMetallicity < 0 ||
     Sp->SphP[j].MassMetallicity > old * (1 + 1e-6))
    Terminate("Invalid gas metals before AGB deposition");
  MyFloat *fraction[7] = {&Sp->SphP[j].GasCarbonMassFraction,    &Sp->SphP[j].GasNitrogenMassFraction,
                          &Sp->SphP[j].GasOxygenMassFraction,    &Sp->SphP[j].GasNeonMassFraction,
                          &Sp->SphP[j].GasMagnesiumMassFraction, &Sp->SphP[j].GasSiliconMassFraction,
                          &Sp->SphP[j].GasIronMassFraction};
  for(MyFloat *x : fraction)
    if(!std::isfinite((double)*x) || *x < 0 || *x > 1)
      Terminate("Invalid gas elemental fraction before AGB deposition");

  // Cold unresolved AGB ejecta: configured minimum specific thermal energy.
  // No SN-like thermal energy is added. Relative bulk kinetic energy is mixed.
  const double vfac  = All.ComovingIntegrationOn ? All.cf_a2inv : 1.0;
  const double u_old = agb_current_predicted_utherm(Sp, j);
  double v[3], vp[3];
  for(int k = 0; k < 3; k++)
    {
      v[k]  = Sp->P[j].Vel[k];
      vp[k] = Sp->SphP[j].VelPred[k];
    }
  agb_feedback::MixedGas mixed;
  try
    {
      mixed = agb_feedback::mix_gas(old, d.dm, u_old, All.MinEgySpec, v, vp, d.ejecta_v, vfac);
    }
  catch(const std::exception &e)
    {
      Terminate("AGB mixing for gas ID %llu: %s", (unsigned long long)d.gas_id, e.what());
    }
  const double candidate_prediction = mixed.u * GAMMA_MINUS1 / pow(Sp->SphP[j].Density * All.cf_a3inv, GAMMA_MINUS1);
  // Reproduce storage rounding before checking the base-state impulse.
  const decltype(Sp->SphP[j].EntropyPred) stored_prediction = candidate_prediction;
  const double candidate_base = (double)Sp->SphP[j].Entropy + ((double)stored_prediction - (double)Sp->SphP[j].EntropyPred);
  if(!std::isfinite(candidate_prediction) || candidate_prediction <= 0 || !std::isfinite(candidate_base) || candidate_base <= 0)
    Terminate("Invalid time-centered AGB entropy preflight");

  cg_agb_probe::record(Sp, j, d.dm, d.distance, u_old, mixed, d.ejecta_v);
  const double mass = mixed.mass;
  for(int x = 0; x < 7; x++)
    *fraction[x] = (old * *fraction[x] + d.f * d.elem[x]) / mass;
  Sp->SphP[j].MassMetallicity += d.f * d.metals;
  Sp->SphP[j].Metallicity = Sp->SphP[j].MassMetallicity / mass;
  Sp->P[j].Metallicity    = Sp->SphP[j].Metallicity;
  Sp->P[j].setMass(mass);
  for(int k = 0; k < 3; k++)
    {
      const double old_momentum = old * Sp->SphP[j].VelPred[k];
      Sp->P[j].Vel[k]           = mixed.velocity[k];
      Sp->SphP[j].VelPred[k]    = mixed.predicted_velocity[k];
      // Debit the donor using the realized current-prediction momentum.
      // Stored midpoint momenta belong to different clocks on inactive gas.
      momentum[k] = mass * Sp->SphP[j].VelPred[k] - old_momentum;
    }
  // Apply the current-time impulse to BOTH entropy time levels; preserve
  // their drift/kick offset instead of resetting an inactive prediction.
  const double base_before       = Sp->SphP[j].Entropy;
  const double prediction_before = Sp->SphP[j].EntropyPred;
  Sp->set_entropy_from_utherm(mixed.u, j);
  const double prediction_after = Sp->SphP[j].EntropyPred;
  Sp->SphP[j].Entropy           = base_before + (prediction_after - prediction_before);
  if(!std::isfinite((double)Sp->SphP[j].Entropy) || Sp->SphP[j].Entropy <= 0 || !std::isfinite(prediction_after) ||
     prediction_after <= 0)
    Terminate("Invalid time-centered AGB entropy commit");
  set_thermodynamic_variables_safe(Sp, j);
}

// Owner side after a successful deposit: debit the star, advance its ledger,
// and release pending AGB dust.  No reference into P may survive dust
// creation, which can reallocate the particle storage.
static void finish_agb(simparticles *Sp, const AGBPlan &plan, const double momentum[3])
{
  const int p = plan.star;
  if(plan.gas != 0)
    {
      cg_agb_probe::sums[cg_agb_probe::transactions] += 1;
      cg_agb_probe::sums[cg_agb_probe::metals] += plan.dust.gas_metals;
      Sp->P[p].setMass(plan.donor_mass - plan.gas);
      const double stored_mass = Sp->P[p].getMass();
      for(int k = 0; k < 3; k++)
        Sp->P[p].Vel[k] = (plan.donor_mass * plan.ejecta_v[k] - momentum[k]) / stored_mass;
    }
  Sp->P[p].AGBPendingCarbonMass += plan.dust.carbon_dust;
  Sp->P[p].AGBPendingSilicateMass += plan.dust.silicate_dust;
  Sp->P[p].AGBCumulativeReturnedMass += plan.returned;
  Sp->P[p].AGBLastProcessedAgeMyr = plan.age;
#ifdef DUST
  const double pending = Sp->P[p].AGBPendingCarbonMass + Sp->P[p].AGBPendingSilicateMass;
  agb_feedback::DustRelease release;
  try
    {
      release = agb_feedback::dust_release(pending, plan.packet, plan.max_packets - Sp->P[p].AGBDustPacketsCreated, plan.complete);
    }
  catch(const std::exception &e)
    {
      Terminate("AGB dust release for star %lld: %s", (long long)Sp->P[p].ID.get(), e.what());
    }
  if(release.mass >= FB_MIN_STELLAR_DUST_MASS && release.packets > 0)
    {
      double cf = Sp->P[p].AGBPendingCarbonMass / pending;
      const double emitted =
          create_dust_particles_from_feedback(Sp, p, release.mass * cf, release.mass * (1 - cf), 2, release.packets, true);
      if(std::abs(emitted - release.mass) > 1e-12 * std::max(release.mass, 1e-30))
        Terminate("AGB dust packet creation failed conservation check");
      Sp->P[p].setMass(Sp->P[p].getMass() - emitted);
      // Saturation protects roundoff only; both components are debited in
      // exactly the same ratio used to construct the actual dust particles.
      Sp->P[p].AGBPendingCarbonMass   = std::max(0.0, Sp->P[p].AGBPendingCarbonMass - emitted * cf);
      Sp->P[p].AGBPendingSilicateMass = std::max(0.0, Sp->P[p].AGBPendingSilicateMass - emitted * (1 - cf));
      Sp->P[p].AGBDustPacketsCreated += release.packets;
    }
#endif
  ++cg_agb_sync::commits;
  if(plan.previous > 0)
    cg_agb_sync::max_noninitial_interval_myr = std::max(cg_agb_sync::max_noninitial_interval_myr, plan.age - plan.previous);
  agb_event_count++;
  agb_total_metals_g += plan.metals * fb_code_mass_in_g();
  diag_add_agb(0.0);
  if(All.FeedbackDebugLevel >= 2)
    printf("[AGB_AGE|T=%d] ID=%lld age=[%.9g,%.9g] mode=%d gas=%.12g metals=%.12g pending_dust=%.12g return=%.12g\n", All.ThisTask,
           (long long)Sp->P[p].ID.get(), plan.previous, plan.age, All.AGBFeedbackMode, plan.gas, plan.metals,
           Sp->P[p].AGBPendingCarbonMass + Sp->P[p].AGBPendingSilicateMass, plan.returned);
}

// Collective.  Deposits all planned AGB transactions; each plan either
// completes or leaves the star untouched (its interval stays pending).
static void deposit_agb_events(simparticles *Sp, const std::vector<AGBPlan> &plans, MPI_Comm comm)
{
  int ntask, this_task;
  MPI_Comm_size(comm, &ntask);
  MPI_Comm_rank(comm, &this_task);

  // Transactions without gas ejecta need no recipients.
  std::vector<int> with_gas;
  std::vector<EventPosition> positions;
  for(size_t i = 0; i < plans.size(); i++)
    if(plans[i].gas != 0)
      {
        EventPosition ep;
        for(int k = 0; k < 3; k++)
          ep.pos[k] = Sp->P[plans[i].star].IntPos[k];
        ep.slot = (int)i;
        positions.push_back(ep);
        with_gas.push_back((int)i);
      }
  // find_kernels() indexes results by position in its input list, so map
  // the slots back afterwards.
  std::vector<EventPosition> compact = positions;
  for(size_t n = 0; n < compact.size(); n++)
    compact[n].slot = (int)n;
  std::vector<Kernel> kernels = find_kernels(Sp, compact, comm);

  std::vector<char> ok(plans.size(), 0);
  for(size_t i = 0; i < plans.size(); i++)
    if(plans[i].gas == 0)
      ok[i] = 1;

  std::vector<std::vector<AGBDeposit>> send(ntask);
  for(size_t n = 0; n < with_gas.size(); n++)
    {
      const int i        = with_gas[n];
      const AGBPlan &pl  = plans[i];
      const Kernel &k    = kernels[n];
      if(!k.found || k.cand.empty())
        continue;
      const double h = 0.5 * k.radius;
      std::vector<double> w(k.cand.size());
      double sum = 0;
      for(size_t c = 0; c < k.cand.size(); c++)
        {
          w[c] = k.cand[c].mass * cubic_spline_kernel(std::sqrt(k.cand[c].r2), h);
          if(!std::isfinite(w[c]) || w[c] < 0)
            Terminate("Invalid distributed AGB weight");
          sum += w[c];
        }
      if(!std::isfinite(sum) || !(sum > 0))
        continue;
      ok[i] = 1;
      AGBDeposit d;
      d.star_id                = (std::uint64_t)Sp->P[pl.star].ID.get();
      d.slot                   = i;
      d.metals                 = pl.dust.gas_metals;
      const double elem[7]     = {pl.elem.C, pl.elem.N, pl.elem.O, pl.elem.Ne, pl.elem.Mg, pl.elem.Si, pl.elem.Fe};
      for(int x = 0; x < 7; x++)
        d.elem[x] = elem[x];
      for(int x = 0; x < 3; x++)
        d.ejecta_v[x] = pl.ejecta_v[x];
      for(size_t c = 0; c < k.cand.size(); c++)
        {
          if(w[c] == 0)
            continue;
          d.gas_id   = k.cand[c].id;
          d.index    = k.cand[c].index;
          d.f        = w[c] / sum;
          d.dm       = pl.gas * d.f;
          d.distance = std::sqrt(k.cand[c].r2);
          send[k.cand[c].task].push_back(d);
        }
    }

  std::vector<int> source;
  std::vector<AGBDeposit> got = cg_dist::exchange(send, comm, &source);
  std::vector<size_t> order(got.size());
  for(size_t n = 0; n < order.size(); n++)
    order[n] = n;
  std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return got[a].star_id < got[b].star_id; });
  std::vector<std::vector<MomentumReply>> replies(ntask);
  for(size_t n : order)
    {
      MomentumReply r;
      r.slot = got[n].slot;
      apply_agb_deposit(Sp, got[n], r.p);
      replies[source[n]].push_back(r);
    }
  const std::vector<MomentumReply> back = cg_dist::exchange(replies, comm);

  std::vector<double> momentum(3 * plans.size(), 0.0);
  for(const MomentumReply &r : back)
    for(int k = 0; k < 3; k++)
      momentum[3 * r.slot + k] += r.p[k];
  for(size_t i = 0; i < plans.size(); i++)
    {
      if(ok[i])
        finish_agb(Sp, plans[i], &momentum[3 * i]);
      else
        agb_deposition_failures++;  // no ledger or age advanced
    }
}
}  // namespace cg_fb_dist

static void apply_distributed_agb_pass(simparticles *Sp, MPI_Comm comm)
{
  if(!cg_agb_sync::open || cg_agb_sync::epoch != All.Ti_Current)
    Terminate("AGB scheduler called outside its post-kick feedback window");
  if(!cg_agb_sync::closed)
    return;  // Leave all mass, age and packet state pending.
  for(int j = 0; j < Sp->NumGas; ++j)
    if(Sp->P[j].getType() == 0)
      for(int k = 0; k < 3; ++k)
        {
          const double v = Sp->P[j].Vel[k], vp = Sp->SphP[j].VelPred[k];
          if(!std::isfinite(v) || !std::isfinite(vp) || std::abs(v - vp) > 1e-5 * std::max(1., std::max(std::abs(v), std::abs(vp))))
            Terminate("Gas velocity clocks disagree at full AGB synchronization");
        }
  // Geometry is evaluated at the current drift time on every gas owner.
  // Recipient velocity impulses preserve their kick offset.
  // All donor gravity kicks have closed; stored donor velocity is current.
  Sp->drift_all_particles();

  std::vector<cg_fb_dist::AGBPlan> plans;
  long long clock_local[2] = {0, 0}, clock_global[2] = {0, 0};
  for(int p = 0; p < Sp->NumPart; ++p)
    if(Sp->P[p].getType() == 4)
      {
        const double age    = get_stellar_age_Myr(Sp->P[p].StellarAge, 0.0);
        const double before = Sp->P[p].AGBLastProcessedAgeMyr;
        if(!std::isfinite(age) || !std::isfinite(before) || before < 0 || before > age + 1e-6 || (Sp->P[p].FeedbackFlag & 2))
          Terminate("Invalid distributed AGB eligibility state");
        if(!(age > before && (age - before >= All.AGBFeedbackIntervalMyr || All.Time >= All.TimeMax)))
          continue;
        if(!Sp->TimeBinSynchronized[(int)Sp->P[p].TimeBinGrav])
          ++clock_local[0];
#if defined(PMGRID) && defined(PERIODIC) && !defined(TREEPM_NOTIMESPLIT)
        if(All.PM_Ti_endstep != All.Ti_Current)
          ++clock_local[1];
#endif
        cg_fb_dist::AGBPlan plan;
        if(cg_fb_dist::plan_agb(Sp, p, age, plan))
          plans.push_back(plan);
      }
  MPI_Allreduce(clock_local, clock_global, 2, MPI_LONG_LONG, MPI_SUM, comm);
  if(All.ThisTask == 0 && (clock_global[0] || clock_global[1]))
    printf("[AGB_DONOR_CLOCK_AUDIT] ti=%lld unsynchronized_gravity=%lld between_PM_closings=%lld; stored donor velocity is not proven "
           "current\n",
           (long long)All.Ti_Current, clock_global[0], clock_global[1]);

  long long n_local = plans.size(), n_global = 0;
  MPI_Allreduce(&n_local, &n_global, 1, MPI_LONG_LONG, MPI_SUM, comm);
  if(n_global == 0)
    return;
  cg_fb_dist::build_gas_tree(Sp);
  cg_fb_dist::deposit_agb_events(Sp, plans, comm);
}

void init_stellar_feedback(void)
{
  if(All.AGBFeedbackMode != 2) Terminate("Time-centered AGB pilot requires mode 2");
  if(All.ThisTask == 0) printf("[AGB_THERMAL_UPDATE] current_prediction_increment; mixing heat retained\n");
  if(All.AGBFeedbackMode < 0 || All.AGBFeedbackMode > 2 ||
     !std::isfinite(All.AGBFeedbackIntervalMyr) || All.AGBFeedbackIntervalMyr <= 0)
    Terminate("AGBFeedbackMode must be 0..2; AGBFeedbackIntervalMyr must be positive");
  if(All.SnapFormat != 3) Terminate("AGB pilot requires HDF5 snapshots (SnapFormat=3)");
  if(All.ThisTask == 0)
    printf("[AGB_MODEL] version=4 mode=%d interval_Myr=%.9g; pending dust is included in stellar mass\n",
           All.AGBFeedbackMode,All.AGBFeedbackIntervalMyr);

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
    if(All.AGBFeedbackMode > 0) Terminate("Age-dependent AGB requires its yield table");
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
    ngbtree * /*Tree*/,
    domain<simparticles> * /*D*/,
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

    // Build/update spatial hashes used by dust searches.
    TIMER_START(CPU_FEEDBACK_HASH);
    rebuild_feedback_spatial_hash(
        Sp, 0.1, comm);
    TIMER_STOP(CPU_FEEDBACK_HASH);

    // Events of this call, deposited together further below.
    std::vector<cg_fb_dist::ThermalEvent> thermal;

    struct SNIITranche
    {
        int star, event, tranche;
        unsigned int mask;
        double MZ_total_code, E_code, m_ssp_msun, age_Myr;
        StellarDustPartition dust;
    };
    std::vector<SNIITranche> tranches;

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
                cg_fb_dist::ThermalEvent ev;
                ev.star    = p;
                ev.kind    = cg_fb_dist::KIND_RESERVOIR;
                ev.E_code  = Sp->P[p].EnergyReservoir;
                ev.MZ_code = 0.0;
                thermal.push_back(ev);
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
                fb_code_energy_in_erg());
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
                    fb_code_mass_in_g();

                double E_code =
                    E_erg /
                    fb_code_energy_in_erg();


                ElementYields elem_total;

                elem_total.C =
                    C_msun *
                    SOLAR_MASS /
                    fb_code_mass_in_g();

                elem_total.N =
                    N_msun *
                    SOLAR_MASS /
                    fb_code_mass_in_g();

                elem_total.O =
                    O_msun *
                    SOLAR_MASS /
                    fb_code_mass_in_g();

                elem_total.Ne =
                    Ne_msun *
                    SOLAR_MASS /
                    fb_code_mass_in_g();

                elem_total.Mg =
                    Mg_msun *
                    SOLAR_MASS /
                    fb_code_mass_in_g();

                elem_total.Si =
                    Si_msun *
                    SOLAR_MASS /
                    fb_code_mass_in_g();

                elem_total.Fe =
                    Fe_msun *
                    SOLAR_MASS /
                    fb_code_mass_in_g();


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


                cg_fb_dist::ThermalEvent ev;
                ev.star    = p;
                ev.kind    = cg_fb_dist::KIND_SNII;
                ev.E_code  = E_code;
                ev.MZ_code = MZ_gas_code;
                ev.elem    = elem_gas;
                thermal.push_back(ev);

                SNIITranche t;
                t.star          = p;
                t.event         = (int)thermal.size() - 1;
                t.tranche       = tranche_to_fire;
                t.mask          = tranche_mask;
                t.MZ_total_code = MZ_total_code;
                t.E_code        = E_code;
                t.m_ssp_msun    = m_ssp_msun;
                t.age_Myr       = age_Myr;
#ifdef DUST
                t.dust          = sn_dust;
#endif
                tranches.push_back(t);
            }
        }

        // Age-dependent AGB is processed collectively after the SN pass.
        if(AGB_Yields.is_table_loaded())
            agb_stars_checked++;
    } // end star loop


    // ====================================================================
    // 3) Deposit reservoir retries and SNII tranches as one batch over
    //    all tasks (see cg_fb_dist above).
    // ====================================================================

    long long n_thermal_local = (long long)thermal.size(), n_thermal = 0;
    MPI_Allreduce(&n_thermal_local, &n_thermal, 1, MPI_LONG_LONG, MPI_SUM, comm);

    // Self-check on the first calls with events: feedback moves mass
    // between stars, gas, and dust but must not create or lose any.
    static int mass_check_remaining = 50;
    const bool check_mass = mass_check_remaining > 0 && n_thermal > 0;
    const double mass_before = check_mass ? cg_dist::total_baryon_mass(Sp, comm) : 0.0;

    if(n_thermal > 0)
    {
        // Search the gas at its current positions.
        Sp->drift_all_particles();
        cg_fb_dist::build_gas_tree(Sp);

        std::vector<char> deposited;
        std::vector<cg_fb_dist::Kernel> kernels;
        cg_fb_dist::deposit_thermal_events(Sp, thermal, deposited, kernels, comm);

#ifdef DUST
        // Dust spawned below is excluded from this call's shock erosion.
        const int first_new_dust = Sp->NumPart;
        std::vector<DustShockEvent> shocks;
#endif

        // ----------------------------------------------------------------
        // The full transaction succeeded for these tranches: gas received
        // the ejecta, so debit the star, create dust, queue shock erosion,
        // and mark the tranche complete.  Failed tranches stay pending and
        // are retried later without a duplicated reservoir contribution or
        // orphaned SNII/LRN dust production.
        // ----------------------------------------------------------------
        for(const SNIITranche &t : tranches)
        {
            const int p = t.star;

            if(!deposited[t.event])
            {
                snii_deposition_failures++;
                continue;
            }

            // Remove this tranche's TOTAL metal ejecta from the star.
            const double new_star_mass = Sp->P[p].getMass() - t.MZ_total_code;
            if(new_star_mass <= 0.0)
                Terminate("SNII feedback tranche would remove entire stellar particle mass");
            Sp->P[p].setMass(new_star_mass);

#ifdef DUST
            if(t.dust.total_dust > 0.0)
            {
                // DustParticlesPerSNII is the TOTAL representation of the
                // whole SSP SNII/HN episode, distributed over the tranches.
                const double created = create_snii_dust_for_tranche(
                    Sp, p, t.dust.carbon_dust, t.dust.silicate_dust, t.tranche);
                if(std::abs(created - t.dust.total_dust) > 1e-12 * std::max(t.dust.total_dust, 1e-30))
                    Terminate("SNII dust creation did not realize partitioned dust mass");
            }

            // Shock destruction from this tranche only, applied below.
            {
                const cg_fb_dist::Kernel &k = kernels[t.event];
                DustShockEvent shock;
                for(int d = 0; d < 3; d++)
                    shock.pos[d] = Sp->P[p].IntPos[d];
                shock.key    = (std::uint64_t)Sp->P[p].ID.get();
                shock.E_code = t.E_code;
                // Nearest gas within 50 kpc sets the ambient density and
                // receives the returned dust material.
                const bool near = k.found && k.nearest.r2 <= 50.0 * 50.0;
                shock.rho_cgs   = near ? k.nearest.payload[1] * All.UnitDensity_in_cgs * All.HubbleParam * All.HubbleParam : -1.0;
                shock.gas_task  = near ? k.nearest.task : -1;
                shock.gas_index = near ? k.nearest.index : -1;
                shock.gas_id    = near ? k.nearest.id : 0;
                shocks.push_back(shock);
            }

            // LRN dust.  LRN yields are tiny: split over the four SNII
            // tranches, the individual dust masses fall below
            // MIN_DUST_PARTICLE_MASS in dust.cc and the LRN PartType6
            // population disappears.  Inject the FULL LRN dust budget once,
            // with the first SNII tranche; SNII/HN energy and metals remain
            // time-distributed.
            if(t.tranche == 0)
            {
                const double n_LRN_full         = All.DustLRNRatePerCCSN * CCSN_NUMBER_PER_MSUN * t.m_ssp_msun;
                const double LRN_dust_mass_code = n_LRN_full * All.DustLRNDustMassMsun * SOLAR_MASS / fb_code_mass_in_g();
                if(LRN_dust_mass_code > 0.0)
                {
                    const double created_lrn_dust = create_dust_particles_from_feedback(
                        Sp, p, 0.10 * LRN_dust_mass_code, 0.90 * LRN_dust_mass_code, /*feedback_type=*/3);
                    if(created_lrn_dust > 0.0)
                    {
                        const double post_lrn_star_mass = Sp->P[p].getMass() - created_lrn_dust;
                        if(post_lrn_star_mass <= 0.0)
                            Terminate("LRN dust creation would remove entire stellar particle mass");
                        Sp->P[p].setMass(post_lrn_star_mass);
                    }
                }
            }
#endif

            Sp->P[p].FeedbackFlag |= t.mask;
            if(all_snii_tranches_complete((unsigned int)Sp->P[p].FeedbackFlag))
            {
                Sp->P[p].FeedbackFlag |= 1;
                if(All.FeedbackDebugLevel >= 2)
                    printf("[SNII_COMPLETE|T=%d|a=%.8g|z=%.6f] star=%d starID=%llu age=%.6f Myr\n", All.ThisTask,
                           (double)All.Time, 1.0 / All.Time - 1.0, p, (unsigned long long)Sp->P[p].ID.get(), t.age_Myr);
            }
        }

#ifdef DUST
        erode_dust_from_sn_shocks_batch(Sp, shocks, first_new_dust, comm);
#endif
    }

    // Do NOT simply mark an SSP complete at MAX_TYPEII_TIME.  If a star
    // missed one or more scheduled bins (for example because no gas was
    // available), it keeps releasing at most one overdue tranche per
    // feedback call until all four are done.

    if(AGB_Yields.is_table_loaded() && All.AGBFeedbackMode > 0)
        apply_distributed_agb_pass(Sp, comm);

    if(check_mass)
    {
        mass_check_remaining--;
        const double mass_after = cg_dist::total_baryon_mass(Sp, comm);
        if(All.ThisTask == 0)
            printf("[FB_MASS_CHECK] step %d: %lld SNII/reservoir events, gas+star+dust mass %.17g -> %.17g (rel. change %.3g)\n",
                   All.NumCurrentTiStep, n_thermal, mass_before, mass_after, (mass_after - mass_before) / mass_before);
    }

    // ====================================================================
    // 4) Periodic diagnostics.
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
/*
 * Deterministic counter-based random numbers for stochastic feedback.
 *
 * SplitMix64 is used here as a high-quality 64-bit mixing function.  This is
 * not a mutable pseudorandom stream: each result is a pure hash of the full
 * (star ID, global timestep, draw counter, stream ID) tuple.  It is therefore
 * reproducible, inexpensive, and insensitive to adjacent numerical values.
 * Separate stream IDs decorrelate neighbor selection from fractional-packet
 * acceptance.
 */
static inline std::uint64_t splitmix64(std::uint64_t x)
{
  x += UINT64_C(0x9e3779b97f4a7c15);
  x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
  return x ^ (x >> 31);
}

static inline double uint64_to_uniform01(std::uint64_t bits)
{
  // Use the upper 53 bits, matching the precision of an IEEE-754 double.
  // The result is in the half-open interval [0,1).
  return (double)(bits >> 11) * (1.0 / 9007199254740992.0);
}

static double feedback_random_u01(
    std::uint64_t star_id,
    std::uint64_t timestep,
    std::uint64_t draw_counter,
    std::uint64_t stream_id)
{
  // Mix each tuple component independently with a distinct domain constant,
  // then mix sequentially so exchanging components cannot preserve the key.
  std::uint64_t state = UINT64_C(0x6a09e667f3bcc909);
  state = splitmix64(state ^ splitmix64(star_id +
                                       UINT64_C(0x243f6a8885a308d3)));
  state = splitmix64(state ^ splitmix64(timestep +
                                       UINT64_C(0x13198a2e03707344)));
  state = splitmix64(state ^ splitmix64(draw_counter +
                                       UINT64_C(0xa4093822299f31d0)));
  state = splitmix64(state ^ splitmix64(stream_id +
                                       UINT64_C(0x082efa98ec4e6c89)));
  return uint64_to_uniform01(splitmix64(state));
}

/*
 * Backward-compatible public wrapper retained for feedback.h and any external
 * caller.  New feedback code should use feedback_random_u01() with the full
 * tuple rather than pre-combining inputs arithmetically.
 */
double get_random_number(unsigned long long id)
{
  return feedback_random_u01(
      (std::uint64_t)id,
      (std::uint64_t)All.NumCurrentTiStep,
      UINT64_C(0),
      UINT64_C(0x4c45474143595f52)); // "LEGACY_R"
}

#endif // FEEDBACK