/*! \file dust.h
 *  \brief On-the-fly dust evolution model
 *  
 *  This module implements dust particle creation, evolution, and destruction
 *  in response to stellar feedback and environmental conditions.
 */

#ifndef DUST_H
#define DUST_H

#include "gadgetconfig.h"
#include <mpi.h>                 /* for MPI_Comm */
#include "../data/simparticles.h"


#ifdef DUST
#define DUST_PARTICLE_TYPE       6
#define DUST_MIN_TIMEBIN         15

void consume_dust_by_astration(simparticles *Sp, int gas_idx, double stellar_mass_formed, int star_idx, double hsml);
void dust_grain_coagulation(simparticles *Sp, int dust_idx, int gas_idx, double dt);

// ========== GLOBAL DUST STATISTICS ==========
// Accessible from all dust modules
extern long long NDustCreated;
extern long long NDustDestroyed;
extern double    TotalDustMass;

extern long long LocalDustCreatedThisStep;
extern long long LocalDustDestroyedThisStep;
extern double    LocalDustMassChange;
extern int       DustNeedsSynchronization;
extern long long GlobalDustCount;  // Current number of dust particles

// Destruction mechanism tracking
extern long long NDustDestroyedByThermal;
extern long long NDustDestroyedByShock;

// Growth/erosion tracking
extern long long NGrainGrowthEvents;
extern long long NGrainErosionEvents;
extern double    TotalMassGrown;
extern double TotalMassDestroyedByThermal; // full thermal destructions
extern double TotalMassDestroyedByShock;   // full shock destructions
extern double TotalMassErodedByThermal;    // partial sputtering events
extern double TotalMassErodedByShock;      // partial shock erosion events

// ========== DUST LOG (dust_particle_log.cc) ==========

#define DUST_EVENT_THERMAL     0
#define DUST_EVENT_SHOCK       1
#define DUST_EVENT_ASTRATION   2
#define DUST_EVENT_SUBLIMATION 3
#define DUST_EVENT_CLEANUP     4
#define DUST_EVENT_SHATTERING  5

void open_dust_particle_log(MPI_Comm Communicator);
void close_dust_particle_log(void);
void log_dust_particle_event(simparticles *Sp, int dust_idx,
                              int nearest_gas, int event_type);
void record_coagulation_event(double n_H_cgs, double n_eff_cgs);
void print_coag_histogram(MPI_Comm Communicator);

// ========== CORE DUST FUNCTIONS (dust.cc) ==========

// Dust particle creation and destruction
void create_dust_particles_from_feedback(simparticles *Sp, int star_idx, 
                                         double metals_produced, int feedback_type);
void spawn_dust_particle(simparticles *Sp, double offset_kpc[3], double dust_mass, 
                         double initial_velocity[3], int star_idx, int feedback_type);
void destroy_dust_particles(simparticles *Sp);
void cleanup_invalid_dust_particles(simparticles *Sp);

// Dust dynamics and interaction
void update_dust_dynamics(simparticles *Sp, double dt, MPI_Comm Communicator);
void update_dust_temperature(simparticles *Sp, int dust_idx, int gas_idx, double dt);
int dust_gas_interaction(simparticles *Sp, int dust_idx, double dt);
void dust_global_synchronization(simparticles *Sp, MPI_Comm Communicator,
                                 long long dust_created,
                                 long long dust_destroyed,
                                 double dust_mass_change);

// ========== GRAIN GROWTH AND EROSION ==========

// Dust grain growth (subgrid model)
void dust_grain_growth_subgrid(simparticles *Sp, int gas_idx, double dt);

// Grain growth in cold, dense ISM
void dust_grain_growth(simparticles *Sp, int gas_idx, double dt);

// Gradual erosion functions
int erode_dust_grain_thermal(simparticles *Sp, int dust_idx, double T_gas, double dt);
int erode_dust_grain_shock(simparticles *Sp, int dust_idx, double shock_velocity_km_s, 
                           double distance_to_sn, double shock_radius, int nearest_gas_hint);

// ========== SHOCK DESTRUCTION ==========

void destroy_dust_from_sn_shocks(simparticles *Sp, int sn_star_idx, 
                                 double sn_energy, double metals_produced, MPI_Comm comm);
double calculate_sn_shock_radius(double sn_energy_erg, double gas_density_cgs, double time_myr);
double calculate_current_sn_shock_radius(simparticles *Sp, int sn_star_idx,
                                          double *out_density_cgs,
                                          int    *out_nearest_gas);
double get_shock_destruction_efficiency(double shock_velocity_km_s, double carbon_fraction);
double get_size_dependent_destruction_efficiency(double shock_velocity_km_s, 
                                                 simparticles *Sp, int dust_idx);

// ========== HELPER FUNCTIONS ==========

// Particle finding
int find_nearest_gas_particle(simparticles *Sp, int dust_idx, double max_r_kpc, double *out_dist_kpc = nullptr);
int find_nearest_dust_particle(simparticles *Sp, int gas_idx);

// Utility
double get_temperature_from_entropy(simparticles *Sp, int idx);
double calculate_velocity_difference(simparticles *Sp, int dust_idx, int gas_idx);
double get_dust_destruction_rate(double temperature, double density);

// ========== DIAGNOSTICS ==========

void print_dust_statistics(simparticles *Sp, MPI_Comm Communicator);
void analyze_dust_gas_coupling(simparticles *Sp);
void analyze_dust_gas_coupling_local(simparticles *Sp);
void analyze_grain_size_distribution(simparticles *Sp);

/**
 * Integrity check of all dust particles (PartType6). This became necessary 
 * as restarting from snapshots and restart files both can encounter frequent
 * problems regarding the new DustP structure, which is stored separately 
 * from the base particle data in P[].
 *
 * It is for debugging corruption or loss of
 * dust particle data during snapshot restart, domain decomposition,
 * particle exchange, reordering, cleanup, or timestep evolution.
 *
 * The function scans all local particles and identifies those with
 * particle type 6 (dust). For each dust particle, it validates both
 * the base particle data stored in P[] and the corresponding auxiliary
 * dust properties stored in DustP[].
 *
 * Specifically, the following conditions are checked:
 *
 *   - Dust grain radius must be finite and > 0
 *   - Dust temperature must be finite and >= 0
 *   - Carbon fraction must be finite and within [0,1]
 *   - Particle mass must be finite and > 0
 *
 * Particles failing any of these checks are classified as "invalid".
 *
 * The routine prints:
 *
 *   - Global counts of total dust particles
 *   - Number of valid dust particles
 *   - Number of invalid dust particles
 *
 * Additionally, a small sample of invalid particles is printed with:
 *
 *   - Particle index
 *   - Particle ID
 *   - Mass
 *   - Grain radius
 *   - Dust temperature
 *   - Carbon fraction
 *   - Grain type
 *
 * The diagnostic output is especially useful for determining whether:
 *
 *   - DustP[] became desynchronized from P[]
 *   - Snapshot restart failed to restore dust fields
 *   - Domain exchange/reordering corrupted dust alignment
 *   - Cleanup routines are removing particles unexpectedly
 *   - Numerical evolution generated invalid dust states
 *
 * MPI_Allreduce() is used so the reported totals reflect the entire
 * simulation across all MPI tasks.
 * */
inline void dust_integrity_check(simparticles *Sp, const char *label, 
                                  MPI_Comm Communicator)
{

    if(All.DustDebugLevel <= 0)
        return;

    long long ndust_local = 0;
    long long valid_local = 0;
    long long invalid_local = 0;

    for(int i = 0; i < Sp->NumPart; i++)
    {
        if(Sp->P[i].getType() == 6)
        {
            ndust_local++;

            bool ok = true;

            if(!std::isfinite(Sp->DustP[i].GrainRadius) ||
               Sp->DustP[i].GrainRadius <= 0)
                ok = false;

            if(!std::isfinite(Sp->DustP[i].DustTemperature) ||
               Sp->DustP[i].DustTemperature < 0)
                ok = false;

            if(!std::isfinite(Sp->DustP[i].CarbonFraction) ||
               Sp->DustP[i].CarbonFraction < 0 ||
               Sp->DustP[i].CarbonFraction > 1)
                ok = false;

            if(Sp->P[i].getMass() <= 0 ||
               !std::isfinite(Sp->P[i].getMass()))
                ok = false;

            if(ok)
                valid_local++;
            else
            {
                invalid_local++;

                if(invalid_local < 10)
                {
                    printf("[BAD_DUST|Task=%d|%s] "
                           "i=%d ID=%llu "
                           "mass=%g radius=%g temp=%g cf=%g type=%d\n",
                           All.ThisTask,
                           label,
                           i,
                           (unsigned long long)Sp->P[i].ID.get(),
                           Sp->P[i].getMass(),
                           Sp->DustP[i].GrainRadius,
                           Sp->DustP[i].DustTemperature,
                           Sp->DustP[i].CarbonFraction,
                           Sp->DustP[i].GrainType);
                }
            }
        }
    }

    long long send[3] = {ndust_local, valid_local, invalid_local};
    long long recv[3];

    MPI_Allreduce(send, recv, 3, MPI_LONG_LONG, MPI_SUM, Communicator);

    int ThisTask;
    MPI_Comm_rank(Communicator, &ThisTask);
    if(ThisTask == 0)
    {
        printf("[DUST_CHECK|%s] "
               "ndust=%lld valid=%lld invalid=%lld\n",
               label,
               recv[0],
               recv[1],
               recv[2]);
    }
}




#endif /* DUST */

#endif /* DUST_H */