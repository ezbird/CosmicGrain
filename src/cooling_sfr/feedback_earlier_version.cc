/*! \file feedback.cc
 *
 *  \brief Implements stellar feedback from Type II SNe and AGB stars
 *
 *  STABLE VERSION with:
 *  - Tree-based neighbor finding (not O(N) scan)
 *  - Timebin promotion for heated particles
 *  - Energy reservoir for gradual injection
 *  - Relative/absolute ∆u caps
 *  - All operations in code units
 *  - Extensive debug output
 *  - SAFETY CHECKS to prevent inf/NaN
 */

#include "gadgetconfig.h"

#ifdef FEEDBACK

#include <gsl/gsl_math.h>
#include <math.h>
#include <mpi.h>
#include "../mpi_utils/setcomm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../cooling_sfr/feedback.h"
#include "../data/allvars.h"
#include "../data/dtypes.h"
#include "../data/intposconvert.h"
#include "../data/mymalloc.h"
#include "../logs/logs.h"
#include "../logs/timer.h"
#include "../system/system.h"
#include "../mpi_utils/mpi_utils.h"
#include "../time_integration/timestep.h"
#include "../cooling_sfr/cooling.h"
#include "../domain/domain.h"
#include "../ngbtree/ngbtree.h"

#ifdef DUST
#include "../cooling_sfr/dust.h"
#endif

// Module-level statistics
static int NSNeInjected = 0;
static int NAGBWindsInjected = 0;
static double EnergyInjected = 0.0;
static double MetalsInjected = 0.0;
static double EnergyReservoir = 0.0;
static int NParticlesPromoted = 0;
static int NEnergyCapped = 0;

// Feedback parameters - CONSERVATIVE values to prevent inf acceleration
#define MAX_RELATIVE_DU 0.01       // Max 10% increase in u per event (PRIMARY LIMIT)
#define MAX_ABSOLUTE_DU 1.0e4     // Absolute safety cap (rarely triggered)
#define MAX_U_THRESHOLD 1.0e4     // Don't heat particles with u > this value
#define MAX_VOLUME_WEIGHT 1.0e10  // Cap on volume weight to prevent blow-up
#define MIN_KERNEL_NEIGHBORS 8    // Minimum neighbors to deposit feedback
#define MAX_KERNEL_NEIGHBORS 64   // Maximum neighbors to consider
#define KERNEL_SUPPORT_MULT 2.0   // Search radius = 2h

#define FEEDBACK_PRINT(...) \
    do { \
        if (All.FeedbackDebugLevel) { \
            printf("[FEEDBACK|T=%d|t=%.6g] ", All.ThisTask, (double)All.Time); \
            printf(__VA_ARGS__); \
        } \
    } while (0)

/**
 * Cubic spline kernel function (standard SPH kernel)
 * Returns W(r,h) normalized such that integral over all space = 1
 * r and h must be in same units (code units)
 */
double cubic_spline_kernel(double r, double h)
{
  if(h <= 0.0) return 0.0;
  
  double q = r / h;
  double norm = 8.0 / (M_PI * h * h * h);  // 3D normalization
  
  if(q < 0.5)
    return norm * (1.0 - 6.0 * q * q + 6.0 * q * q * q);
  else if(q < 1.0)
    return norm * 2.0 * (1.0 - q) * (1.0 - q) * (1.0 - q);
  else
    return 0.0;
}



/**
 * Tree-based neighbor finding for feedback module
 * Simplified version based on density.cc approach
 * Add this to feedback.cc or create feedback_Tree->cc
 */

// Structure to hold feedback neighbor data
struct feedback_ngbdata
{
  int Index;           // Index in Sp->P[] array
  MyIntPosType IntPos[3];
  double Distance;     // Distance in code units
};

// Structure for feedback search parameters
struct feedback_pinfo
{
  int target;                    // Index of star particle
  MyIntPosType searchcenter[3];  // Search center (star position)
  MyIntPosType search_min[3];    // Minimum search box corner
  MyIntPosType search_range[3];  // Search box range
  double hsml;                   // Smoothing length (code units)
  double search_radius;  // Search radius (2h)
  double hsml2;                  // hsml squared
  int numngb;                    // Number of neighbors found
};

// Global neighbor data array for feedback (reused across calls)
static feedback_ngbdata *FeedbackNgbdat = NULL;
static int FeedbackMaxNgb = 0;

/**
 * Initialize feedback neighbor finding
 * Call once at startup
 */
void init_feedback_neighbor_search(int max_neighbors)
{
  FeedbackMaxNgb = max_neighbors;
  FeedbackNgbdat = (feedback_ngbdata *)Mem.mymalloc("FeedbackNgbdat", 
                                                     FeedbackMaxNgb * sizeof(feedback_ngbdata));
}

/**
 * Free feedback neighbor data
 * Call at shutdown
 */
void free_feedback_neighbor_search(void)
{
  if(FeedbackNgbdat)
    {
      Mem.myfree(FeedbackNgbdat);
      FeedbackNgbdat = NULL;
    }
}

/**
 * Set up search parameters for feedback
 */
void feedback_get_pinfo(simparticles *Sp, int star_idx, double search_radius, feedback_pinfo &pdat)
{
  pdat.target = star_idx;
  pdat.search_radius = search_radius;           // Store 2h
  pdat.hsml = search_radius / KERNEL_SUPPORT_MULT;  // Store h (smoothing length)
  pdat.hsml2 = pdat.hsml * pdat.hsml;           // h² (not (2h)²!)
  pdat.numngb = 0;
  
  // Get star position in integer coordinates
  for(int i = 0; i < 3; i++)
    pdat.searchcenter[i] = Sp->P[star_idx].IntPos[i];
  
  // Convert search radius to integer coordinates
  double search_radius_vec[3] = {search_radius, search_radius, search_radius};
  MySignedIntPosType search_range_int[3];
  Sp->pos_to_signedintpos(search_radius_vec, search_range_int);
  
  // Set search box
  for(int i = 0; i < 3; i++)
    {
      pdat.search_min[i] = pdat.searchcenter[i] - search_range_int[i];
      pdat.search_range[i] = 2 * search_range_int[i];
    }
}

/**
 * Check if node overlaps with search region
 * Returns NODE_OPEN if we should open it, NODE_DISCARD otherwise
 */
inline int feedback_evaluate_node_opening_criterion(simparticles *Sp, feedback_pinfo &pdat, ngbnode *nop)
{
  if(nop->level <= LEVEL_ALWAYS_OPEN)
    return NODE_OPEN;
  
  // Drift node if needed
  if(nop->Ti_Current != All.Ti_Current)
    nop->drift_node(All.Ti_Current, Sp);
  
  // Check for spatial overlap with search box
  MyIntPosType left[3], right[3];
  
  left[0] = Sp->nearest_image_intpos_to_intpos_X(nop->center_offset_min[0] + nop->center[0], pdat.search_min[0]);
  right[0] = Sp->nearest_image_intpos_to_intpos_X(nop->center_offset_max[0] + nop->center[0], pdat.search_min[0]);
  if(left[0] > pdat.search_range[0] && right[0] > left[0])
    return NODE_DISCARD;
  
  left[1] = Sp->nearest_image_intpos_to_intpos_Y(nop->center_offset_min[1] + nop->center[1], pdat.search_min[1]);
  right[1] = Sp->nearest_image_intpos_to_intpos_Y(nop->center_offset_max[1] + nop->center[1], pdat.search_min[1]);
  if(left[1] > pdat.search_range[1] && right[1] > left[1])
    return NODE_DISCARD;
  
  left[2] = Sp->nearest_image_intpos_to_intpos_Z(nop->center_offset_min[2] + nop->center[2], pdat.search_min[2]);
  right[2] = Sp->nearest_image_intpos_to_intpos_Z(nop->center_offset_max[2] + nop->center[2], pdat.search_min[2]);
  if(left[2] > pdat.search_range[2] && right[2] > left[2])
    return NODE_DISCARD;
  
  return NODE_OPEN;
}

/**
 * Check if particle is within search radius and add to neighbor list
 */
inline void feedback_check_particle_interaction(simparticles *Sp, ngbtree *Tree, feedback_pinfo &pdat, 
                                                int p, unsigned char shmrank)
{
  particle_data *P = Tree->get_Pp(p, shmrank);
  
  // Only consider gas particles
  if(P->getType() != 0)
    return;
  
  // Drift particle if needed
  sph_particle_data *SphP = Tree->get_SphPp(p, shmrank);
  if(P->get_Ti_Current() != All.Ti_Current)
    Sp->drift_particle(P, SphP, All.Ti_Current);
  
  // Calculate distance using integer positions
  double posdiff[3];
  Sp->nearest_image_intpos_to_pos(P->IntPos, pdat.searchcenter, posdiff);
  
  double r2 = posdiff[0] * posdiff[0] + posdiff[1] * posdiff[1] + posdiff[2] * posdiff[2];
  
  // Early termination: only keep particles within ~1.2h where kernel has significant weight
  // pdat.hsml is the smoothing length h, so 1.44*hsml2 is (1.2h)^2
  if(r2 > pdat.hsml2)  // Beyond 1.2h (useful kernel radius)
    return;
  
  //if(pdat.numngb >= FeedbackMaxNgb)
    //  {
      //  if(pdat.numngb == FeedbackMaxNgb) {  // Only print ONCE per star
          // FEEDBACK_PRINT("WARNING: Reached max neighbors (%d), some neighbors will be missed!\n", FeedbackMaxNgb);
      //  }
      //  return;
     // }
  
  // Add to neighbor list
  int n = pdat.numngb++;
  FeedbackNgbdat[n].Index = p;
  FeedbackNgbdat[n].IntPos[0] = P->IntPos[0];
  FeedbackNgbdat[n].IntPos[1] = P->IntPos[1];
  FeedbackNgbdat[n].IntPos[2] = P->IntPos[2];
  FeedbackNgbdat[n].Distance = sqrt(r2);
}

/**
 * Recursively open a node and check all particles/subnodes
 */
void feedback_open_node(simparticles *Sp, ngbtree *Tree, domain<simparticles> *D, 
                        feedback_pinfo &pdat, ngbnode *nop)
{
  int p = nop->nextnode;
  unsigned char shmrank = nop->nextnode_shmrank;
  
  while(p != nop->sibling || (shmrank != nop->sibling_shmrank && nop->sibling >= Tree->MaxPart + D->NTopnodes))
    {
      if(p < 0)
        Terminate("feedback_open_node: p < 0");
      
      int next;
      unsigned char next_shmrank;
      
      if(p < Tree->MaxPart) // A gas particle
        {
          next = Tree->get_nextnodep(shmrank)[p];
          next_shmrank = shmrank;
          
          // Check if this particle is within search radius
          feedback_check_particle_interaction(Sp, Tree, pdat, p, shmrank);
        }
      else if(p < Tree->MaxPart + Tree->MaxNodes) // Internal node
        {
          ngbnode *nop_next = Tree->get_nodep(p, shmrank);
          next = nop_next->sibling;
          next_shmrank = nop_next->sibling_shmrank;
          
          if(nop_next->not_empty)
            {
              int openflag = feedback_evaluate_node_opening_criterion(Sp, pdat, nop_next);
              
              if(openflag == NODE_OPEN)
                {
                  // Recursively open this node
                  feedback_open_node(Sp, Tree, D, pdat, nop_next);
                }
            }
        }
      else
        {
          Terminate("feedback_open_node: unexpected node type p=%d", p);
        }
      
      p = next;
      shmrank = next_shmrank;
    }
}

/**
 * Find neighbors using tree-based search
 * This is the main function to call from feedback.cc
 * 
 * Returns number of neighbors found
 * Fills ngb_list with particle indices and distances with distances (in code units)
 */
void find_feedback_neighbors_tree(simparticles *Sp, ngbtree *Tree, domain<simparticles> *D, int star_idx, 
                                  int *ngb_list, double *distances, int *n_ngb, 
                                  double *smoothing_length, int max_ngb)
{
  *n_ngb = 0;
  *smoothing_length = 0.0;
  
  // Get adaptive smoothing length
  *smoothing_length = get_local_smoothing_length_tree(Sp, Tree, star_idx);
  
  if(*smoothing_length <= 0.0)
    {
      FEEDBACK_PRINT("Star ID=%llu: Failed to get valid smoothing length\n",
                     (unsigned long long)Sp->P[star_idx].ID.get());
      return;
    }
  
  // Set up search parameters
  feedback_pinfo pdat;
  double search_radius = KERNEL_SUPPORT_MULT * (*smoothing_length);
  feedback_get_pinfo(Sp, star_idx, search_radius, pdat);
  
  FEEDBACK_PRINT("Star ID=%llu: h=%g, search_radius=%g (code units)\n",
                 (unsigned long long)Sp->P[star_idx].ID.get(), 
                 *smoothing_length, search_radius);
  
  // Start tree walk from root
  int no = Tree->MaxPart; // Root node
  unsigned char shmrank = Shmem.Island_ThisTask;
  
  ngbnode *nop = Tree->get_nodep(no, shmrank);
  
  // Walk the tree
  feedback_open_node(Sp, Tree, D, pdat, nop);
  
  // Copy results to output arrays
  *n_ngb = (pdat.numngb < max_ngb) ? pdat.numngb : max_ngb;
  
  for(int i = 0; i < *n_ngb; i++)
    {
      ngb_list[i] = FeedbackNgbdat[i].Index;
      distances[i] = FeedbackNgbdat[i].Distance;
    }
  
  FEEDBACK_PRINT("Star ID=%llu: Found %d neighbors\n",
                 (unsigned long long)Sp->P[star_idx].ID.get(), *n_ngb);
}

/**
 * Get adaptive smoothing length from nearby gas particles using tree
 * Returns smoothing length in CODE UNITS
 */
double get_local_smoothing_length_tree(simparticles *Sp, ngbtree *Tree, int star_idx)
{
  // Sample nearby gas particles to get typical smoothing length
  int n_sample = 0;
  double hsml_samples[32];
  double min_hsml = 1e30;
  double max_hsml = 0.0;
  
  // First, do a quick scan to find a reasonable search radius
  double min_dist = 1e30;
  
  // Sample first 100 gas particles to get idea of scale
  int sample_limit = (Sp->NumGas < 100) ? Sp->NumGas : 100;
  for(int i = 0; i < sample_limit; i++) {
    // Calculate distance using integer positions
    double posdiff[3];
    Sp->nearest_image_intpos_to_pos(Sp->P[i].IntPos, Sp->P[star_idx].IntPos, posdiff);
    
    double r = sqrt(posdiff[0]*posdiff[0] + posdiff[1]*posdiff[1] + posdiff[2]*posdiff[2]);
    if(r < min_dist && r > 0.0) min_dist = r;
    
    // Collect smoothing length samples
    if(Sp->SphP[i].Hsml > 0.0 && n_sample < 32) {
      hsml_samples[n_sample] = Sp->SphP[i].Hsml;
      if(Sp->SphP[i].Hsml < min_hsml) min_hsml = Sp->SphP[i].Hsml;
      if(Sp->SphP[i].Hsml > max_hsml) max_hsml = Sp->SphP[i].Hsml;
      n_sample++;
    }
  }
  
  double h;
  
if(n_sample >= 5) {
    // Calculate median - FULL SORTING CODE
    for(int i = 0; i < n_sample - 1; i++) {
      for(int j = 0; j < n_sample - i - 1; j++) {
        if(hsml_samples[j] > hsml_samples[j + 1]) {
          double temp = hsml_samples[j];
          hsml_samples[j] = hsml_samples[j + 1];
          hsml_samples[j + 1] = temp;
        }
      }
    }
    h = hsml_samples[n_sample / 2];
    
    // CRITICAL FIX: Enforce MINIMUM smoothing length for feedback
    double h_min_feedback = 10.0 * min_hsml;
    if(h < h_min_feedback) h = h_min_feedback;
    
    // Also enforce absolute minimum (5 kpc to be VERY safe)
    double abs_min_feedback = 0.005;  // 5 kpc in code units (Mpc)
    if(h < abs_min_feedback) h = abs_min_feedback;
    
    // Relax max bound
    double max_bound = 10.0 * max_hsml;
    if(h > max_bound) h = max_bound;
    
    FEEDBACK_PRINT("[HSML] Star ID=%llu: median h=%g → enforced h_feedback=%g\n",
           (unsigned long long)Sp->P[star_idx].ID.get(), hsml_samples[n_sample/2], h);
}
  else {
    // Fallback: use minimum distance
    h = min_dist * 2.0;
    
    // Apply reasonable absolute bounds as last resort
    double abs_min = All.BoxSize / 10000.0;  // 0.01% of box
    double abs_max = All.BoxSize / 1000.0;     // 0.1% of box
    if(h < abs_min) h = abs_min;
    if(h > abs_max) h = abs_max;
    
    FEEDBACK_PRINT("[HSML] Star ID=%llu: Fallback h=%g (min_dist=%g)\n",
           (unsigned long long)Sp->P[star_idx].ID.get(), h, min_dist);
  }
  
  return h;
}

/**
 * Initialize the stellar feedback module
 */
void init_stellar_feedback(void)
{  
  // Reset statistics
  NSNeInjected = 0;
  NAGBWindsInjected = 0;
  EnergyInjected = 0.0;
  MetalsInjected = 0.0;
  EnergyReservoir = 0.0;
  NParticlesPromoted = 0;
  NEnergyCapped = 0;
  
  // Initialize tree-based neighbor search ONCE
  init_feedback_neighbor_search(MAX_KERNEL_NEIGHBORS);

  if(All.ThisTask == 0)
    {
      FEEDBACK_PRINT("=== STELLAR FEEDBACK INITIALIZATION ===\n");
      FEEDBACK_PRINT("Type II SN energy = %g erg\n", ESN);
      FEEDBACK_PRINT("SNe per Msun = %g\n", NSNE_PER_MSUN);
      FEEDBACK_PRINT("Max relative ∆u = %.1fx\n", MAX_RELATIVE_DU);
      FEEDBACK_PRINT("Max absolute ∆u = %g (code units)\n", MAX_ABSOLUTE_DU);
      FEEDBACK_PRINT("Max volume weight = %g\n", MAX_VOLUME_WEIGHT);
      FEEDBACK_PRINT("Using tree-based neighbor search\n");
      FEEDBACK_PRINT("Using SPH cubic spline kernel weighting\n");
      FEEDBACK_PRINT("Timebin promotion: ENABLED\n");
      FEEDBACK_PRINT("Energy reservoir: ENABLED (1%% per timestep)\n");
      FEEDBACK_PRINT("Safety checks: ENABLED (inf/NaN protection)\n");
      FEEDBACK_PRINT("=====================================\n");
    }
}

/**
 * Main routine to apply stellar feedback
 * Called after star formation in cooling_and_starformation
 */
void apply_stellar_feedback(double current_time, simparticles *Sp, ngbtree *Tree, domain<simparticles> *D)
{


  // DIAGNOSTIC: Check for problematic particles BEFORE doing anything

  // At the very start of apply_stellar_feedback(), before ANY feedback
// Emergency check: if any particles already have extreme values, abort feedback this timestep
int skip_feedback = 0;
int n_extreme = 0;
for(int i = 0; i < Sp->NumGas; i++)
{
  double u = Sp->get_utherm_from_entropy(i);
  double vmag = sqrt(Sp->P[i].Vel[0]*Sp->P[i].Vel[0] + 
                    Sp->P[i].Vel[1]*Sp->P[i].Vel[1] + 
                    Sp->P[i].Vel[2]*Sp->P[i].Vel[2]);
  
  if(!gsl_finite(u) || !gsl_finite(vmag) || u > 5.0e4 || vmag > 1e4)
  {
    n_extreme++;
    if(n_extreme < 10)  // Only print first few
    {
      FEEDBACK_PRINT("WARNING: Particle ID=%llu already extreme BEFORE feedback: u=%g, v=%g\n",
             (unsigned long long)Sp->P[i].ID.get(), u, vmag);
    }
  }
}

if(n_extreme > 0)
{
  FEEDBACK_PRINT("ERROR: %d particles already extreme - SKIPPING ALL FEEDBACK THIS TIMESTEP\n", n_extreme);
  skip_feedback = 1;  // Set flag instead of returning
}


  for(int i = 0; i < Sp->NumGas; i++)
    {
      if(Sp->P[i].ID.get() == 1575802)  // The crashing particle
        {
          double u = Sp->get_utherm_from_entropy(i);
          double rho = Sp->SphP[i].Density;
          double hsml = Sp->SphP[i].Hsml;
          double vmag = sqrt(Sp->P[i].Vel[0]*Sp->P[i].Vel[0] + 
                            Sp->P[i].Vel[1]*Sp->P[i].Vel[1] + 
                            Sp->P[i].Vel[2]*Sp->P[i].Vel[2]);
          
          FEEDBACK_PRINT("*** TRACKING PROBLEM PARTICLE ID=1575802 ***\n");
          FEEDBACK_PRINT("  u=%g, rho=%g, hsml=%g, v=%g\n", u, rho, hsml, vmag);
          FEEDBACK_PRINT("  Has reservoir energy? Checking nearby stars...\n");
          
          // Check if any nearby stars have reservoir energy
          for(int j = 0; j < Sp->NumPart; j++)
            {
              if(Sp->P[j].getType() == 4 && Sp->P[j].EnergyReservoir > 0.0)
                {
                  double dx = Sp->P[j].IntPos[0] - Sp->P[i].IntPos[0];
                  double dy = Sp->P[j].IntPos[1] - Sp->P[i].IntPos[1];
                  double dz = Sp->P[j].IntPos[2] - Sp->P[i].IntPos[2];
                  double dist = sqrt(dx*dx + dy*dy + dz*dz);
                  
                  if(dist < 0.1)  // Within 100 kpc
                    {
                      FEEDBACK_PRINT("  Nearby star ID=%llu at dist=%g has reservoir=%g\n",
                             (unsigned long long)Sp->P[j].ID.get(), dist, 
                             Sp->P[j].EnergyReservoir);
                    }
                }
            }
        }
    }





  int local_sne = 0;
  int local_agb = 0;
  double local_energy = 0.0;
  double local_metals = 0.0;
  int total_stars = 0;
  int stars_in_typeii_range = 0;
  int stars_with_feedback_flag = 0;
  double local_reservoir_injected = 0.0;
  
// Implementing a flag on whether to skip, to ensure all MPI tasks reach MPI collective operations
if(!skip_feedback)
  {

  // First pass: distribute any stored reservoir energy
FEEDBACK_PRINT("=== RESERVOIR INJECTION PASS ===\n");

for(int i = 0; i < Sp->NumPart; i++)
{
  if(Sp->P[i].getType() == 4 && Sp->P[i].EnergyReservoir > 0.0)
    {


      // TESTING DISABLING RESERVOIR!!!!!!
      // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      //double injected = distribute_reservoir_energy(Sp, Tree, D, i, current_time);
      //local_reservoir_injected += injected;
      //EnergyReservoir -= injected;
    }
}
  
  if(local_reservoir_injected > 0.0) {
    FEEDBACK_PRINT("Distributed %g (code units) from reservoir\n", local_reservoir_injected);
  }
  
// Diagnostic: Check h distribution
if(All.NumCurrentTiStep % 100 == 0 && All.ThisTask == 0) {
    int n_stars_sampled = 0;
    double h_min = 1e30, h_max = 0, h_sum = 0;
    
    for(int i = 0; i < Sp->NumPart && n_stars_sampled < 20; i++) {
        if(Sp->P[i].getType() == 4) {
            double h = get_local_smoothing_length_tree(Sp, Tree, i);
            if(h > 0) {
                h_min = (h < h_min) ? h : h_min;
                h_max = (h > h_max) ? h : h_max;
                h_sum += h;
                n_stars_sampled++;
            }
        }
    }
    
    if(n_stars_sampled > 0) {
        FEEDBACK_PRINT("=== SMOOTHING LENGTH DISTRIBUTION ===\n");
        FEEDBACK_PRINT("Sampled %d stars: h_min=%g, h_max=%g, h_mean=%g (code units)\n",
               n_stars_sampled, h_min, h_max, h_sum/n_stars_sampled);
        FEEDBACK_PRINT("At h_mean=%g: kernel radius=2h=%g, useful kernel~h=%g\n",
               h_sum/n_stars_sampled, 2*(h_sum/n_stars_sampled), h_sum/n_stars_sampled);
    }
}

  // Debug: Count stars and ages
  FEEDBACK_PRINT("=== STELLAR POPULATION CENSUS - Count stars and ages ===\n");
  for(int i = 0; i < Sp->NumPart; i++)
    {
      if(Sp->P[i].getType() == 4)  // Star particle
        {
          total_stars++;
          
          if(Sp->P[i].FeedbackFlag > 0)
            stars_with_feedback_flag++;
          
          double stellar_age = get_stellar_age_Myr(Sp->P[i].StellarAge, current_time);
          
          if(stellar_age >= MIN_TYPEII_TIME && stellar_age <= MAX_TYPEII_TIME)
            stars_in_typeii_range++;
          
          // Debug output for first few stars
          if(All.ThisTask == 0 && total_stars <= 3) {
            FEEDBACK_PRINT("Star ID=%llu: age=%.2f Myr, FeedbackFlag=%d, Reservoir=%g\n", 
                   (unsigned long long)Sp->P[i].ID.get(), stellar_age, 
                   Sp->P[i].FeedbackFlag, Sp->P[i].EnergyReservoir);
          }
        }
    }
  
  FEEDBACK_PRINT("Total stars: %d, In Type II range: %d, With feedback flag: %d\n", 
         total_stars, stars_in_typeii_range, stars_with_feedback_flag);
  FEEDBACK_PRINT("Type II range: %.1f - %.1f Myr\n", MIN_TYPEII_TIME, MAX_TYPEII_TIME);
  
  // Loop over all star particles for NEW feedback events
  FEEDBACK_PRINT("=== NEW FEEDBACK EVENTS ===\n");
  for(int i = 0; i < Sp->NumPart; i++)
    {
      if(Sp->P[i].getType() == 4)  // Star particle
        {
          double stellar_age = get_stellar_age_Myr(Sp->P[i].StellarAge, current_time);
          
          // ========== TYPE II SUPERNOVA FEEDBACK (CUMULATIVE) ==========
          if(stellar_age >= MIN_TYPEII_TIME && stellar_age <= MAX_TYPEII_TIME)
            {
              // Skip if this star has already done Type II feedback
              if(Sp->P[i].FeedbackFlag & 1)
                continue;
              
              // Calculate cumulative probability that this star should have exploded by now
              double time_in_sn_phase = stellar_age - MIN_TYPEII_TIME;
              double total_sn_duration = MAX_TYPEII_TIME - MIN_TYPEII_TIME;
              double cumulative_prob = time_in_sn_phase / total_sn_duration;
              
              // Use a deterministic seed based on star ID for consistency
              double rnd = get_random_number(Sp->P[i].ID.get() + 1000);
              
              if(rnd < cumulative_prob)
                {
                  // This star should explode!
                  FEEDBACK_PRINT("*** TYPE II SN TRIGGERED *** Star ID=%llu (age=%.2f Myr, prob=%.4f)\n", 
                         (unsigned long long)Sp->P[i].ID.get(), stellar_age, cumulative_prob);
                  
                  typeII_supernova_feedback(Sp, Tree, D, i, current_time);
                  local_sne++;
                  
                  double stellar_mass_msun = Sp->P[i].getMass() * (All.UnitMass_in_g / SOLAR_MASS) / All.HubbleParam;
                  local_energy += ESN * NSNE_PER_MSUN * stellar_mass_msun;
                  local_metals += TYPEII_METAL_YIELD * Sp->P[i].getMass();
                  
                  // Mark that this star has done Type II feedback
                  Sp->P[i].FeedbackFlag |= 1;
                }
            }
          
          // ========== AGB WIND FEEDBACK (CUMULATIVE) ==========
          if(stellar_age >= MIN_AGB_TIME && stellar_age <= MAX_AGB_TIME)
            {
              // Skip if this star has already done AGB feedback
              if(Sp->P[i].FeedbackFlag & 2)
                continue;
              
              // Calculate cumulative probability for AGB winds
              double time_in_agb_phase = stellar_age - MIN_AGB_TIME;
              double total_agb_duration = MAX_AGB_TIME - MIN_AGB_TIME;
              double cumulative_agb_prob = time_in_agb_phase / total_agb_duration;
              
              // Use different seed for AGB vs Type II
              double rnd = get_random_number(Sp->P[i].ID.get() + 2000);
              
              if(rnd < cumulative_agb_prob)
                {
                  // This star should have AGB winds!
                  FEEDBACK_PRINT("*** AGB WIND TRIGGERED *** Star ID=%llu (age=%.2f Myr, prob=%.4f)\n", 
                         (unsigned long long)Sp->P[i].ID.get(), stellar_age, cumulative_agb_prob);
                  
                  agb_wind_feedback(Sp, Tree, D, i, current_time);
                  local_agb++;
                  
                  double stellar_mass_msun = Sp->P[i].getMass() * (All.UnitMass_in_g / SOLAR_MASS) / All.HubbleParam;
                  double mass_loss = AGB_MASS_LOSS_RATE * stellar_mass_msun;
                  local_energy += mass_loss * AGB_WIND_ENERGY / SOLAR_MASS;
                  local_metals += AGB_METAL_YIELD * Sp->P[i].getMass();
                  
                  // Mark that this star has done AGB feedback
                  Sp->P[i].FeedbackFlag |= 2;
                }
            }
        }
    }
  
  }

  // Update global statistics
  NSNeInjected += local_sne;
  NAGBWindsInjected += local_agb;
  EnergyInjected += local_energy;
  MetalsInjected += local_metals;
  
  // Output statistics periodically
  if(All.NumCurrentTiStep % 100 == 0)
    {
      stellar_feedback_statistics(Sp->Communicator);
    }
  
  // Report if any feedback occurred
  if(local_sne > 0 || local_agb > 0) {
    FEEDBACK_PRINT("This timestep: %d Type II SNe, %d AGB winds, %d particles promoted\n", 
           local_sne, local_agb, NParticlesPromoted);
  }
}

/**
 * Distribute stored reservoir energy gradually
 * Returns amount of energy actually distributed
 */
double distribute_reservoir_energy(simparticles *Sp, ngbtree *Tree, domain<simparticles> *D, 
                                   int star_idx, double current_time)
{
  if(Sp->P[star_idx].EnergyReservoir <= 0.0) return 0.0;
  
  // Distribute a VERY SMALL fraction per timestep (0.1% instead of 10%)
  double fraction_to_distribute = 0.001;  // 0.1% per timestep
  double energy_to_inject = Sp->P[star_idx].EnergyReservoir * fraction_to_distribute;
  
  FEEDBACK_PRINT("[RESERVOIR] Star ID=%llu: reservoir=%g, injecting=%g (0.1%% per step)\n",
         (unsigned long long)Sp->P[star_idx].ID.get(), 
         Sp->P[star_idx].EnergyReservoir, energy_to_inject);
  
  // Find neighbors using tree
  int max_ngb = MAX_KERNEL_NEIGHBORS;
  int *ngb_list = (int *)Mem.mymalloc("ngb_list", max_ngb * sizeof(int));
  double *distances = (double *)Mem.mymalloc("distances", max_ngb * sizeof(double));
  int n_ngb = 0;
  double smoothing_length = 0.0;
  
  find_feedback_neighbors_tree(Sp, Tree, D, star_idx, ngb_list, distances, &n_ngb, 
                               &smoothing_length, max_ngb);
  
  double energy_deposited = 0.0;
  
  if(n_ngb >= MIN_KERNEL_NEIGHBORS && smoothing_length > 0.0)
    {
      // PRE-SCREEN: Count how many neighbors are already too hot
      int n_available = 0;
      double max_u_threshold = 1e5;  // Don't heat particles with u > 100,000
      
      for(int i = 0; i < n_ngb; i++)
        {
          int idx = ngb_list[i];
          double u = Sp->get_utherm_from_entropy(idx);
          
          if(u < max_u_threshold)
            n_available++;
        }
      
      if(n_available < MIN_KERNEL_NEIGHBORS)
        {
          FEEDBACK_PRINT("[RESERVOIR] Star ID=%llu: All neighbors too hot (<%d available), "
                 "keeping energy in reservoir\n",
                 (unsigned long long)Sp->P[star_idx].ID.get(), n_available);
          Mem.myfree(distances);
          Mem.myfree(ngb_list);
          return 0.0;
        }
      
      // Deposit with capping - but deposit_feedback_energy_capped will skip hot particles
      energy_deposited = deposit_feedback_energy_capped(Sp, star_idx, energy_to_inject, 0.0, 
                                                        ngb_list, distances, n_ngb, 
                                                        smoothing_length);
      
      // Update reservoir
      Sp->P[star_idx].EnergyReservoir -= energy_deposited;
      
      if(Sp->P[star_idx].EnergyReservoir < 1e-10)
        Sp->P[star_idx].EnergyReservoir = 0.0;
      
      FEEDBACK_PRINT("[RESERVOIR] Star ID=%llu: Deposited %g, remaining reservoir=%g\n",
             (unsigned long long)Sp->P[star_idx].ID.get(), 
             energy_deposited, Sp->P[star_idx].EnergyReservoir);
    }
  else
    {
      FEEDBACK_PRINT("[RESERVOIR] Star ID=%llu: Insufficient neighbors (%d), keeping in reservoir\n",
             (unsigned long long)Sp->P[star_idx].ID.get(), n_ngb);
    }
  
  Mem.myfree(distances);
  Mem.myfree(ngb_list);
  
  return energy_deposited;
}

void typeII_supernova_feedback(simparticles *Sp, ngbtree *Tree, domain<simparticles> *D, 
                                int star_idx, double current_time)
{
  // Calculate energy and metals to inject
  double stellar_mass_msun = Sp->P[star_idx].getMass() * (All.UnitMass_in_g / SOLAR_MASS) / All.HubbleParam;
  double n_sne = NSNE_PER_MSUN * stellar_mass_msun;
  double energy = n_sne * ESN;  // Total energy in ergs
  double metals = TYPEII_METAL_YIELD * Sp->P[star_idx].getMass();
  
  // Convert energy to code units
  energy *= All.HubbleParam / All.UnitEnergy_in_cgs;
  
  FEEDBACK_PRINT("[TYPEII] Star ID=%llu: M=%.3g Msun, N_SNe=%.3g, E_total=%g (code units)\n",
         (unsigned long long)Sp->P[star_idx].ID.get(), stellar_mass_msun, n_sne, energy);
  
  FEEDBACK_PRINT("[TYPEII] Depositing %.4g Msun of metals (yield=%.4g)\n", 
         metals * All.UnitMass_in_g / SOLAR_MASS / All.HubbleParam, TYPEII_METAL_YIELD);

  // Find neighbors using TREE
  int max_ngb = MAX_KERNEL_NEIGHBORS;
  int *ngb_list = (int *)Mem.mymalloc("ngb_list", max_ngb * sizeof(int));
  double *distances = (double *)Mem.mymalloc("distances", max_ngb * sizeof(double));
  int n_ngb = 0;
  double smoothing_length = 0.0;
  
  find_feedback_neighbors_tree(Sp, Tree, D, star_idx, ngb_list, distances, &n_ngb, &smoothing_length, max_ngb);
  
  FEEDBACK_PRINT("[TYPEII] Star ID=%llu: found %d neighbors within h=%g (code units)\n",
         (unsigned long long)Sp->P[star_idx].ID.get(), n_ngb, smoothing_length);
  
  if(n_ngb >= MIN_KERNEL_NEIGHBORS && smoothing_length > 0.0)
    {
      // KEY CHANGE: Deposit only 0.1% immediately, store 99.9% in reservoir
      double immediate_fraction = 0.0001;  // Only 0.1%!
      double immediate_energy = energy * immediate_fraction;
      
      double energy_deposited = deposit_feedback_energy_capped(Sp, star_idx, immediate_energy, metals, 
                                                               ngb_list, distances, n_ngb, smoothing_length);
      
      // Store the REST (99.9%) in reservoir for gradual injection
      double reservoir_energy = energy - energy_deposited;
      Sp->P[star_idx].EnergyReservoir += reservoir_energy;
      EnergyReservoir += reservoir_energy;
      
      FEEDBACK_PRINT("[TYPEII] Star ID=%llu: Deposited %g immediately (%.1f%%), stored %g in reservoir (%.1f%%)\n",
             (unsigned long long)Sp->P[star_idx].ID.get(), 
             energy_deposited, 100.0*energy_deposited/energy,
             reservoir_energy, 100.0*reservoir_energy/energy);
      
      #ifdef DUST
        create_dust_particles_from_feedback(Sp, star_idx, metals, 1);
        destroy_dust_from_sn_shocks(Sp, star_idx, energy, metals);
      #endif
    }
  else
    {
      // Store ALL energy in reservoir if can't deposit
      Sp->P[star_idx].EnergyReservoir += energy;
      EnergyReservoir += energy;
      FEEDBACK_PRINT("[TYPEII] Star ID=%llu: No neighbors (%d < %d)! Stored all energy in reservoir\n",
             (unsigned long long)Sp->P[star_idx].ID.get(), n_ngb, MIN_KERNEL_NEIGHBORS);
    }
  
  Mem.myfree(distances);
  Mem.myfree(ngb_list);
}

/**
 * Apply AGB wind feedback from a star particle
 */
void agb_wind_feedback(simparticles *Sp, ngbtree *Tree, domain<simparticles> *D, int star_idx, double current_time)
{
  // Calculate energy and metals to inject
  double stellar_mass_msun = Sp->P[star_idx].getMass() * (All.UnitMass_in_g / SOLAR_MASS) / All.HubbleParam;
  double mass_loss = AGB_MASS_LOSS_RATE * stellar_mass_msun;
  double energy = mass_loss * AGB_WIND_ENERGY / SOLAR_MASS;  // Total energy in ergs
  double metals = AGB_METAL_YIELD * Sp->P[star_idx].getMass();
  
  // Convert energy to code units
  energy *= All.HubbleParam / All.UnitEnergy_in_cgs;
  
  FEEDBACK_PRINT("[AGB] Star ID=%llu: M=%.3g Msun, dM=%.3g, E_total=%g (code units)\n",
         (unsigned long long)Sp->P[star_idx].ID.get(), stellar_mass_msun, mass_loss, energy);
  
  FEEDBACK_PRINT("[AGB] Depositing %.4g Msun of metals (yield=%.4g)\n",
         metals * All.UnitMass_in_g / SOLAR_MASS / All.HubbleParam, AGB_METAL_YIELD);       

  // Find neighbors using TREE
  int max_ngb = MAX_KERNEL_NEIGHBORS;
  int *ngb_list = (int *)Mem.mymalloc("ngb_list", max_ngb * sizeof(int));
  double *distances = (double *)Mem.mymalloc("distances", max_ngb * sizeof(double));
  int n_ngb = 0;
  double smoothing_length = 0.0;
  
  find_feedback_neighbors_tree(Sp, Tree, D, star_idx, ngb_list, distances, &n_ngb, &smoothing_length, max_ngb);
  
  FEEDBACK_PRINT("[AGB] Star ID=%llu: found %d neighbors within h=%g (code units)\n",
         (unsigned long long)Sp->P[star_idx].ID.get(), n_ngb, smoothing_length);
  
  if(n_ngb >= MIN_KERNEL_NEIGHBORS && smoothing_length > 0.0)
    {
      double energy_deposited = deposit_feedback_energy_capped(Sp, star_idx, energy, metals, 
                                                               ngb_list, distances, n_ngb, smoothing_length);
      
      // Store any excess in reservoir
      double excess_energy = energy - energy_deposited;
      if(excess_energy > 0.0) {
        Sp->P[star_idx].EnergyReservoir += excess_energy;
        EnergyReservoir += excess_energy;
      }
      
      #ifdef DUST
        create_dust_particles_from_feedback(Sp, star_idx, metals, 2);  // AGB
      #endif
      
      FEEDBACK_PRINT("[AGB] Star ID=%llu: Successfully deposited %g of %g energy (%.1f%%)\n",
             (unsigned long long)Sp->P[star_idx].ID.get(), energy_deposited, energy,
             100.0 * energy_deposited / energy);
    }
  else
    {
      Sp->P[star_idx].EnergyReservoir += energy;
      EnergyReservoir += energy;
      FEEDBACK_PRINT("[AGB] Star ID=%llu: No neighbors! Stored all energy in reservoir\n",
             (unsigned long long)Sp->P[star_idx].ID.get());
    }
  
  Mem.myfree(distances);
  Mem.myfree(ngb_list);
}

/**
 * Deposit feedback energy with ∆u capping and reservoir storage
 */
double deposit_feedback_energy_capped(simparticles *Sp, int star_idx, double energy, 
                                      double metals, int *ngb_list, double *distances, 
                                      int n_ngb, double smoothing_length)
{
  if(n_ngb == 0 || smoothing_length <= 0.0) return 0.0;
  
  FEEDBACK_PRINT("[DEPOSIT] Depositing E=%g, Z=%g into %d neighbors, h=%g\n", 
         energy, metals, n_ngb, smoothing_length);
  
  // ========== PHASE 1: Calculate weights with hot particle filtering ==========
  
  double *weights = (double *)Mem.mymalloc("weights", n_ngb * sizeof(double));
  double total_weight = 0.0;
  int n_too_hot = 0;
  
  for(int i = 0; i < n_ngb; i++)
    {
      int gas_idx = ngb_list[i];
      
      // Check if particle is already too hot
      double old_u = Sp->get_utherm_from_entropy(gas_idx);
      if(old_u > MAX_U_THRESHOLD)  // Use the #define directly
        {
          weights[i] = 0.0;
          n_too_hot++;
          continue;
        }
      
      // Calculate kernel weight
      double r = distances[i];
      double kernel_weight = cubic_spline_kernel(r, smoothing_length);
      
      // SPH volume weighting with safety checks
      double density = Sp->SphP[gas_idx].Density;
      if(density <= 1e-20) density = 1e-20;
      
      double volume_weight = Sp->P[gas_idx].getMass() / density;
      
      if(volume_weight > MAX_VOLUME_WEIGHT || !gsl_finite(volume_weight))
        volume_weight = MAX_VOLUME_WEIGHT;
      
      weights[i] = kernel_weight * volume_weight;
      total_weight += weights[i];
    }
  
  // Check if we have any valid targets
  if(total_weight <= 0.0 || !gsl_finite(total_weight))
    {
      if(n_too_hot == n_ngb)
        {
          FEEDBACK_PRINT("[DEPOSIT] All %d neighbors too hot (u>%.1e), cannot deposit\n", 
                 n_ngb, MAX_U_THRESHOLD);
        }
      else
        {
          FEEDBACK_PRINT("[DEPOSIT] WARNING: Invalid total_weight=%g, cannot deposit\n", 
                 total_weight);
        }
      Mem.myfree(weights);
      return 0.0;
    }
  
  // Normalize weights
  for(int i = 0; i < n_ngb; i++)
    weights[i] /= total_weight;
  
  if(n_too_hot > 0)
    {
      FEEDBACK_PRINT("[DEPOSIT] Skipped %d/%d neighbors (too hot), redistributing to cooler gas\n",
             n_too_hot, n_ngb);
    }
  
  // ========== PHASE 2: Deposit energy with safety caps ==========
  
  double total_energy_deposited = 0.0;
  double total_metals_deposited = 0.0;
  int n_capped = 0;
  int n_skipped = 0;
  
  for(int i = 0; i < n_ngb; i++)
    {
      if(weights[i] <= 0.0) continue;
      
      int idx = ngb_list[i];
      
      double old_utherm = Sp->get_utherm_from_entropy(idx);
      double du_requested = energy * weights[i] / Sp->P[idx].getMass();
      
      if(!gsl_finite(du_requested) || du_requested < 0.0)
        {
          FEEDBACK_PRINT("[DEPOSIT] WARNING: Gas idx=%d has invalid du_requested=%g, skipping\n",
                 idx, du_requested);
          n_skipped++;
          continue;
        }
      
      // Apply BOTH relative and absolute caps
      double du_max_relative = MAX_RELATIVE_DU * old_utherm;
      double du_max_absolute = MAX_ABSOLUTE_DU;
      double du_max = (du_max_relative < du_max_absolute) ? du_max_relative : du_max_absolute;
      
      double du_actual = du_requested;
      if(du_actual > du_max)
        {
          du_actual = du_max;
          n_capped++;
          NEnergyCapped++;
        }
      
      double new_utherm = old_utherm + du_actual;
      
      if(!gsl_finite(new_utherm) || new_utherm < 0.0 || new_utherm > 1e10)
        {
          FEEDBACK_PRINT("[DEPOSIT] WARNING: Gas idx=%d produced invalid new_utherm=%g "
                 "(old=%g, du=%g), skipping!\n",
                 idx, new_utherm, old_utherm, du_actual);
          n_skipped++;
          continue;
        }
      
      // Update entropy and thermodynamic variables
      Sp->set_entropy_from_utherm(new_utherm, idx);
      
      // Add metals
      #ifdef COOLING
      if(metals > 0.0)
        {
          double dm = metals * weights[i];
          Sp->P[idx].Metallicity += dm / Sp->P[idx].getMass();
          
          if(Sp->P[idx].Metallicity > 0.1)
            Sp->P[idx].Metallicity = 0.1;
          
          total_metals_deposited += dm;
        }
      #endif
      
      total_energy_deposited += du_actual * Sp->P[idx].getMass();
      
      // Update thermodynamic variables
      Sp->SphP[idx].set_thermodynamic_variables();
      
      // Debug output for first few particles
      if(i < 3 && weights[i] > 0.0)
        {
          FEEDBACK_PRINT("[DEPOSIT] Gas idx=%d: w=%.4f, du_req=%.3g, du_max=%.3g, "
                 "du_act=%.3g, old_u=%.3g, new_u=%.3g (×%.2f)\n",
                 idx, weights[i], du_requested, du_max, du_actual, 
                 old_utherm, new_utherm, new_utherm/old_utherm);
        }
    }
  
  // ========== Summary ==========
  
  if(n_skipped > 0)
    {
      FEEDBACK_PRINT("[DEPOSIT] WARNING: Skipped %d particles due to invalid values\n", 
             n_skipped);
    }
  
  FEEDBACK_PRINT("[DEPOSIT] SUMMARY: Deposited %.3g of %.3g energy (%.1f%%), "
         "%d capped, %d too hot, %d skipped\n",
         total_energy_deposited, energy, 
         (energy > 0.0) ? 100.0*total_energy_deposited/energy : 0.0,
         n_capped, n_too_hot, n_skipped);
  
  Mem.myfree(weights);
  
  return total_energy_deposited;
}

/**
 * Get stellar age in Myr
 */
double get_stellar_age_Myr(double formation_time, double current_time)
{
  double age_code = current_time - formation_time;
  
  if(All.ComovingIntegrationOn)
    {
      // Cosmological: convert scale factor to time
      double hubble_time_years = 9.77e9 / All.HubbleParam;
      double age_years = age_code * hubble_time_years * 0.01;
      return age_years / 1.0e6;
    }
  else
    {
      // Non-cosmological
      double age_years = age_code * All.UnitTime_in_s / (365.25 * 24.0 * 3600.0);
      return age_years / 1.0e6;
    }
}

/**
 * Get random number for a given seed
 */
double get_random_number(unsigned long long id)
{
  unsigned int seed = (unsigned int)(id & 0xFFFFFFFF);
  seed = seed ^ (unsigned int)(All.Time * 1000000.0);
  seed = (seed * 2654435761U) % UINT_MAX;
  
  int ia = 16807;
  int im = 2147483647;
  int iq = 127773;
  int ir = 2836;
  
  int k = seed / iq;
  unsigned int temp = ia * (seed - k * iq) - ir * k;
  if((int)temp < 0) temp += im;
  
  return temp / (double)im;
}

/**
 * Output feedback statistics
 */
void stellar_feedback_statistics(MPI_Comm Communicator)
{
  int global_sne, global_agb, global_promoted, global_capped;
  double global_energy, global_metals, global_reservoir;
  
  MPI_Reduce(&NSNeInjected, &global_sne, 1, MPI_INT, MPI_SUM, 0, Communicator);
  MPI_Reduce(&NAGBWindsInjected, &global_agb, 1, MPI_INT, MPI_SUM, 0, Communicator);
  MPI_Reduce(&EnergyInjected, &global_energy, 1, MPI_DOUBLE, MPI_SUM, 0, Communicator);
  MPI_Reduce(&MetalsInjected, &global_metals, 1, MPI_DOUBLE, MPI_SUM, 0, Communicator);
  MPI_Reduce(&EnergyReservoir, &global_reservoir, 1, MPI_DOUBLE, MPI_SUM, 0, Communicator);
  MPI_Reduce(&NParticlesPromoted, &global_promoted, 1, MPI_INT, MPI_SUM, 0, Communicator);
  MPI_Reduce(&NEnergyCapped, &global_capped, 1, MPI_INT, MPI_SUM, 0, Communicator);
  
  if(All.ThisTask == 0)
    {
      FEEDBACK_PRINT("===== CUMULATIVE STATISTICS =====\n");
      FEEDBACK_PRINT("Type II SNe:        %d\n", global_sne);
      FEEDBACK_PRINT("AGB winds:          %d\n", global_agb);
      FEEDBACK_PRINT("Energy injected:    %g erg\n", global_energy);
      FEEDBACK_PRINT("Metals injected:    %g Msun\n", 
             global_metals * All.UnitMass_in_g / SOLAR_MASS / All.HubbleParam);
      FEEDBACK_PRINT("Energy in reservoir: %g (code units)\n", global_reservoir);
      FEEDBACK_PRINT("Particles promoted:  %d\n", global_promoted);
      FEEDBACK_PRINT("Energy cappings:     %d\n", global_capped);
      FEEDBACK_PRINT("================================\n");
    }
}

#endif /* FEEDBACK */