/* ============================================================================
 * feedback.cc — Stellar feedback with Type II SNe and AGB winds
 *
 * Features:
 *   - Cosmology-aware stellar ages using Gadget-4 drift factor tables
 *   - ΔT-target stochastic heating (EAGLE-like)
 *   - Energy reservoir system with intelligent release threshold
 *   - SPH kernel support with 2h neighbor search
 *   - Metal enrichment from SNe and AGB winds
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

#include "../cooling_sfr/feedback.h"
#include "../cooling_sfr/feedback_spatial_hash.h"
#include "../data/allvars.h"
#include "../data/dtypes.h"
#include "../data/mymalloc.h"
#include "../logs/logs.h"
#include "../ngbtree/ngbtree.h"
#include "../domain/domain.h"
#include "../time_integration/timestep.h"
#include "../time_integration/driftfac.h"

#ifdef DUST
#include "../cooling_sfr/dust.h"
#endif

// ------------------------------- Constants ----------------------------------
//#define AGB_METAL_YIELD 0.005         // Metal yield for AGB stars
//#define AGB_MASS_LOSS_RATE 0.001      // AGB mass loss rate
//#define AGB_WIND_ENERGY 1.0e48        // AGB wind energy per event (erg)
//#define ESN 1.0e50                    // should probably be 1e51; SN energy in erg
//#define NSNE_PER_MSUN 0.0085          // ~0.0085; Number of SNe per solar mass formed; calculated with Chabrier IMF and SN mass range 8-100 Msun
//#define TYPEII_METAL_YIELD 0.02       // Metal yield for Type II SNe

// Global spatial hash instance
static spatial_hash_improved gas_hash;
static int last_rebuild_step = -1;

static const double ESN_ERG                 = 1.0e50;  // should probably be 1e51; SN energy in erg
static const double NSNE_PER_MSUN_VAL       = 0.0085;  // ~0.0085; Number of SNe per solar mass formed; calculated with Chabrier IMF and SN mass range 8-100 Msun
static const double METAL_YIELD_PER_SN_MSUN = 0.1;     // Mass of metal yield in a Type II SN
static const double AGB_WIND_METALLICITY    = 0.01;    // Mass fraction of metals in AGB winds
static const double AGB_MLOSS_RATE          = 0.001;   // Msun/Msun/Myr
static const double AGB_E_ERG               = 1.0e48;  // AGB wind energy per event (erg)

// ΔT-target heating parameters (EAGLE-like)
static const double DELTA_T_TARGET        = 3.162277660e7;  // 10^7.5 K
static const int    HEAT_N_MIN            = 32;
static const int    HEAT_N_MAX            = 256;

#ifdef FEEDBACK_LIMIT_DULOG
static const double FEEDBACK_MAX_DULOG    = 1.5; // Limit |Δlog10 u| per event (2.0 would be ~×100)
#endif
#ifdef FEEDBACK_T_CAP
static const double FEEDBACK_T_CAP_K      = 5.0e7; // Debug soft cap
#endif

// Neighbor search parameters
static const int MIN_KERNEL_NEIGHBORS = 32;
static const int MAX_KERNEL_NEIGHBORS = 256;

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

// In your existing places where you had DIAG counters, replace with calls above.
// Example inside apply_feedback_event():
//   if (is_SNII) diag_add_sn(E_code * All.UnitEnergy_in_cgs);
//   else         diag_add_agb(E_code * All.UnitEnergy_in_cgs);
//   diag_add_Edep(E_used_code * All.UnitEnergy_in_cgs);
//   diag_add_EtoRes((E_code - E_used_code) * All.UnitEnergy_in_cgs);
// And where you clamp Δlog10(u):
//   diag_track_dulog(fabs(du_log10));

void rebuild_feedback_spatial_hash(simparticles *Sp, double max_feedback_radius)
{
  // Only rebuild if on a new timestep
  if(All.NumCurrentTiStep != last_rebuild_step) {
    if(All.ThisTask == 0) {
      printf("\n[FEEDBACK] Rebuilding spatial hash for step %lld\n", 
             (long long)All.NumCurrentTiStep);
    }
    
    double gas_softening = All.ForceSoftening[0];
    
    // Automatic sizing - it figures out optimal cell count
    gas_hash.build(Sp, max_feedback_radius, gas_softening, true, 0);
    
    if(All.ThisTask == 0 && last_rebuild_step < 0) {
      gas_hash.print_stats();  // Print stats on first build only
    }
    
    last_rebuild_step = All.NumCurrentTiStep;
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

/* Convert scale factor to integertime for drift factor lookup */
static inline integertime scalefactor_to_integertime(double a)
{
  if(All.ComovingIntegrationOn)
    return (integertime)(log(a / All.TimeBegin) / All.Timebase_interval);
  else
    return (integertime)((a - All.TimeBegin) / All.Timebase_interval);
}

/* Get stellar age in Myr using Gadget's drift factor tables 
 * 
 * The drift factor integrates ∫dt/a from birth to now, returning results
 * in Gyr. We multiply by 1000 to convert to Myr.
 */
double get_stellar_age_Myr(double stellar_age_field, double /*unused*/)
{
  if(stellar_age_field <= 0.0 || stellar_age_field > 1.0)
    return 0.0;
  
  integertime ti_birth = scalefactor_to_integertime(stellar_age_field);
  integertime ti_now   = scalefactor_to_integertime(All.Time);
  
  double dt_Gyr = Driftfac.get_drift_factor(ti_birth, ti_now);
  double age_Myr = dt_Gyr * 1000.0;
  
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
 * Uses brute-force scan over gas particles. For production runs with many
 * particles, consider replacing with proper tree walk.
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

  const double r2_max = (2.0*h) * (2.0*h);

  // Scan gas particles for neighbors within 2h
  for (int i = 0; i < Sp->NumGas; ++i) {
    if (i == star_idx) continue;

    double d[3]; 
    Sp->nearest_image_intpos_to_pos(Sp->P[i].IntPos, Sp->P[star_idx].IntPos, d);
    double r2 = d[0]*d[0] + d[1]*d[1] + d[2]*d[2];
    
    if (r2 <= r2_max) {
        if (*n_ngb < max_ngb) {
            ngb_list[*n_ngb]  = i;
            distances[*n_ngb] = sqrt(r2);
            (*n_ngb)++;
            
            // Early exit - we have enough neighbors
            if (*n_ngb >= max_ngb) {
                *smoothing_length = h;
                return;
            }
        }
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
  double u_cgs = u_code * All.UnitPressure_in_cgs / All.UnitDensity_in_cgs;
  double mu = mu_default(1e4);
  return u_cgs / c_v(mu);
}

static inline double TK_to_ucode(double T_K){
  double mu = mu_default(T_K);
  double u_cgs = c_v(mu) * T_K;
  return u_cgs / (All.UnitPressure_in_cgs / All.UnitDensity_in_cgs);
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
                                        double E_avail_code, double MZ_code)
{
  if(ngb_list.empty() || E_avail_code<=0) return 0.0;

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
    double m_g = Sp->P[j].getMass();
    double w = cubic_spline_kernel(distances[idx], hsml);
    double fraction = (m_g * w) / total_kernel_weighted_mass;
    
    // Energy for this particle (in code units)
    double dE_code = E_avail_code * fraction;
    
    // Convert to specific internal energy change
    double du_cgs = (dE_code * All.UnitEnergy_in_cgs) / (m_g * All.UnitMass_in_g);
    double du_code = du_cgs / (All.UnitPressure_in_cgs / All.UnitDensity_in_cgs);

    double u_old = Sp->get_utherm_from_entropy(j);
    if(u_old <= 0) {
      FB_PRINT("WARNING: Particle %d has u_old=%g, skipping feedback\n", j, u_old);
      continue;  // Skip this particle
    }

    double u_new = u_old + du_code;

#ifdef FEEDBACK_LIMIT_DULOG
    double du_log10 = log10(u_new) - log10(u_old);
    if(fabs(du_log10) > FEEDBACK_MAX_DULOG){
      u_new = u_old * pow(10.0, fb_clamp(du_log10, -FEEDBACK_MAX_DULOG, FEEDBACK_MAX_DULOG));

      diag_track_dulog(fabs(du_log10));
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
                                 double E_code, double MZ_code)
{
  std::vector<int> ngb; 
  std::vector<double> dist; 
  double hsml=0;
  gather_neighbors(Sp, Tree, D, star_i, ngb, dist, hsml);
  if(ngb.empty()) {
    // DON'T set feedback flag if no neighbors found
    FB_PRINT("WARNING: Star %d found no neighbors, skipping feedback\n", star_i);
    return;
  }

  double E_used_code = deposit_energy_stochastic(Sp, ngb, dist, hsml, 
                                                 star_i, E_code, MZ_code);


  const double E_used_erg    = E_used_code * All.UnitEnergy_in_cgs;
  const double E_leftover_erg= (E_code - E_used_code) * All.UnitEnergy_in_cgs;

  if (is_SNII) diag_add_sn(E_code * All.UnitEnergy_in_cgs);
  else         diag_add_agb(E_code * All.UnitEnergy_in_cgs);

  diag_add_Edep(E_used_erg);
  diag_add_EtoRes(E_leftover_erg);


  // Store any leftover energy in reservoir
  double leftover = std::max(0.0, E_code - E_used_code);
  Sp->P[star_i].EnergyReservoir += leftover;
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
                                  domain<simparticles>*D, int star_i)
{
  double E_code = Sp->P[star_i].EnergyReservoir;
  if(E_code<=0) return;

  std::vector<int> ngb; 
  std::vector<double> dist; 
  double hsml=0;
  gather_neighbors(Sp, Tree, D, star_i, ngb, dist, hsml);
  if(ngb.empty()) return;

  // Compute threshold: energy needed to heat Nheat neighbors by ΔT
  int Nheat = fb_clamp((int)(ngb.size()/8), HEAT_N_MIN, HEAT_N_MAX);
  if(Nheat<1) Nheat=1;
  
  double m_g_code = Sp->P[ngb[0]].getMass();
  double E_per_g  = c_v(mu_default(DELTA_T_TARGET))*DELTA_T_TARGET;
  double E_one    = E_per_g * (m_g_code*All.UnitMass_in_g);
  double E_need   = (E_one * Nheat) / All.UnitEnergy_in_cgs;

  // Only release if we have meaningful energy (90% of threshold)
  // TUNING: Adjust 0.3 here to change release frequency
  if(E_code < 0.01*E_need) return;

  // Instead of releasing all at once, release in chunks
  double E_to_release = std::min(E_code, E_need * 2.0);  // Release at most 2× threshold
  double used = deposit_energy_stochastic(Sp, ngb, dist, hsml, star_i, E_to_release, 0.0);
  

  diag_add_EfromRes(used * All.UnitEnergy_in_cgs);
  diag_add_Edep(used * All.UnitEnergy_in_cgs);
  
  Sp->P[star_i].EnergyReservoir = std::max(0.0, E_code - used);
}

// ---------------------------- Public API ------------------------------------

void init_stellar_feedback(void)
{
  #ifdef FEEDBACK
    if(All.ThisTask==0) 
      FB_PRINT("FEEDBACK enabled; ΔT_target=%.2e K, Nheat=[%d,%d]\n", 
              DELTA_T_TARGET, HEAT_N_MIN, HEAT_N_MAX);
  #endif
}

void apply_stellar_feedback(double /*current_time*/, simparticles *Sp, 
                           ngbtree *Tree, domain<simparticles> *D)
{

  double t_start = MPI_Wtime();
    
    // ✅ Build star index list ONCE
    static std::vector<int> star_indices;
    star_indices.clear();
    star_indices.reserve(Sp->NumPart / 10);
    
    for(int p = 0; p < Sp->NumPart; ++p) {
        if(Sp->P[p].getType() == 4) {
            star_indices.push_back(p);
        }
    }
    
    int n_stars_processed = star_indices.size();
    
    // ---- Early multi-node safe bail-out: do NOTHING and return, no collectives ----
    bool any_reservoir = false;
    for (int p : star_indices) {
      if (Sp->P[p].EnergyReservoir > 0) { any_reservoir = true; break; }
    }
    if (n_stars_processed == 0 && !any_reservoir) {
        return;  // absolutely no MPI here
    }

    // 1) Try to release reservoir energy (ONLY non-empty reservoirs)
    for(int p : star_indices) {
        if(Sp->P[p].EnergyReservoir > 0) {  // ✅ Skip empty reservoirs
            try_release_reservoir(Sp, Tree, D, p);
        }
    }
    
    // 2) Trigger new feedback events
    for(int p : star_indices) {
        double age_Myr = get_stellar_age_Myr(Sp->P[p].StellarAge, 0.0);
        
        // Type II SNe (10-40 Myr window)
        // SNII - Calculate metals from actual SNe
        if(age_Myr >= MIN_TYPEII_TIME && age_Myr < MAX_TYPEII_TIME && 
          !(Sp->P[p].FeedbackFlag & 1)) {
            
            double m_star_g = Sp->P[p].getMass() * All.UnitMass_in_g;
            double n_SN = NSNE_PER_MSUN_VAL * (m_star_g / SOLAR_MASS);
            double E_erg = n_SN * ESN_ERG;
            
            const double METAL_YIELD_PER_SN_MSUN = 0.1;  // M☉ of metals per SN
            double MZ_g = n_SN * METAL_YIELD_PER_SN_MSUN * SOLAR_MASS; // Metal mass = number of SN × yield per SN
            double MZ_code = MZ_g / All.UnitMass_in_g; 
            
            double E_code = E_erg / All.UnitEnergy_in_cgs;
            apply_feedback_event(Sp, Tree, D, p, /*SNII*/true, E_code, MZ_code);
            Sp->P[p].FeedbackFlag |= 1;
        }

        // AGB - Calculate metals from actual mass loss
        if(age_Myr >= MIN_AGB_TIME && age_Myr < MAX_AGB_TIME && 
          !(Sp->P[p].FeedbackFlag & 2)) {
            
            double m_star_g = Sp->P[p].getMass() * All.UnitMass_in_g;
            double m_star_msun = m_star_g / SOLAR_MASS;
            
            // Mass lost in this timestep (or event)
            double mass_lost_msun = AGB_MLOSS_RATE * m_star_msun;  // This might need dt
            
            // AGB stars enrich lost mass to ~0.01-0.02 (1-2% metals by mass)
            double MZ_g = mass_lost_msun * AGB_WIND_METALLICITY * SOLAR_MASS;
            double MZ_code = MZ_g / All.UnitMass_in_g;
            
            double E_erg = mass_lost_msun * AGB_E_ERG;
            double E_code = E_erg / All.UnitEnergy_in_cgs;
            apply_feedback_event(Sp, Tree, D, p, /*SNII*/false, E_code, MZ_code);
            Sp->P[p].FeedbackFlag |= 2;
        }
    }


    double t_end = MPI_Wtime();  // ← End timing HERE
    (void)t_start; (void)t_end; // keep locals if you want per-rank timing with FB_PRINT
    /*
    // Gather stats across MPI ranks
    int n_stars_global = 0;
    MPI_Reduce(&n_stars_processed, &n_stars_global, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    
    double t_max = 0.0;
    double t_local = t_end - t_start;
    MPI_Reduce(&t_local, &t_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if(All.ThisTask == 0 && All.NumCurrentTiStep % 100 == 0 && n_stars_global > 0) {
        FB_PRINT("TIMING: Total feedback took %.3f sec for %d stars (%.3f ms/star)\n",
               t_max, n_stars_global, (t_max * 1000.0) / n_stars_global);
    }
    */
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