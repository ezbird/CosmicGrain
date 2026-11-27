/*! \file dust.h
 *  \brief On-the-fly dust evolution model
 *
 *  Public interface of the dust module. Dust lives in its own particles
 *  (PartType 6). Stars create them through feedback; the surrounding gas then
 *  grows, erodes, and destroys them as the simulation runs.
 */

#ifndef DUST_H
#define DUST_H

#include "gadgetconfig.h"
#include <mpi.h>                 /* for MPI_Comm */
#include <cstdint>
#include <vector>
#include "../data/simparticles.h"


#ifdef DUST
#define DUST_PARTICLE_TYPE       6
#define DUST_MIN_TIMEBIN         15

// Astration = dust locked into a newly formed star. consume_dust_by_astration()
// only queues the request for the star formed from gas_idx. The dust is removed
// later by apply_queued_dust_astration(), which every task must call once all
// stars of the step exist (it is collective).
void consume_dust_by_astration(simparticles *Sp, int gas_idx, double stellar_mass_formed, int star_idx, double hsml);
void apply_queued_dust_astration(simparticles *Sp, MPI_Comm comm);
void dust_grain_coagulation(simparticles *Sp, int dust_idx, int gas_idx, double dt,
                            double local_DTG, double T_gas);
void dust_grain_shattering(simparticles *Sp, int dust_idx, int gas_idx, double dt,
                           double local_DTG, double T_gas);

// ========== GLOBAL DUST STATISTICS ==========
// Run-wide counters shared by all dust source files.
extern long long NDustCreated;
extern long long NDustDestroyed;
extern double    TotalDustMass;

extern long long LocalDustCreatedThisStep;
extern long long LocalDustDestroyedThisStep;
extern double    LocalDustMassChange;
extern int       DustNeedsSynchronization;
extern long long GlobalDustCount;  // Current number of dust particles

// Particles destroyed, by mechanism
extern long long NDustDestroyedByThermal;
extern long long NDustDestroyedByShock;

// Growth and erosion: event counts and mass (code units)
extern long long NGrainGrowthEvents;
extern long long NGrainErosionEvents;
extern double    TotalMassGrown;
extern double TotalMassDestroyedByThermal; // whole particles destroyed by sputtering
extern double TotalMassDestroyedByShock;   // whole particles destroyed by SN shocks
extern double TotalMassErodedByThermal;    // mass lost to partial sputtering
extern double TotalMassErodedByShock;      // mass lost to partial shock erosion

// ========== DUST LOG (dust_particle_log.cc) ==========
// Event codes written in the last column of the per-particle event log.

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

// Creating and removing dust particles
double create_dust_particles_from_feedback(
    simparticles *Sp,
    int star_idx,
    double carbon_dust_mass,
    double silicate_dust_mass,
    int feedback_type,
    int particle_count_override = -1,
    bool conserve_agb_momentum = false);
void spawn_dust_particle(simparticles *Sp, double offset_kpc[3], double dust_mass,
                         double initial_velocity[3], int star_idx, int feedback_type);
void destroy_dust_particles(simparticles *Sp);
void cleanup_invalid_dust_particles(simparticles *Sp);

// Dust motion and dust-gas processes
void update_dust_dynamics(simparticles *Sp, double dt, MPI_Comm Communicator);
// T_gas is the nearest gas particle's temperature. update_dust_dynamics()
// computes it once per dust particle and passes it to every process below.
void update_dust_temperature(simparticles *Sp, int dust_idx, int gas_idx, double dt,
                             double T_gas);
int dust_gas_interaction(simparticles *Sp, int dust_idx, int nearest_gas, double dt,
                         double T_gas);
void dust_global_synchronization(simparticles *Sp, MPI_Comm Communicator,
                                 long long dust_created,
                                 long long dust_destroyed,
                                 double dust_mass_change);

// ========== GRAIN GROWTH AND EROSION ==========

// Grain growth by accretion of gas-phase metals (subgrid model)
void dust_grain_growth_subgrid(simparticles *Sp, int dust_idx, int gas_idx, double dt,
                               double T_gas);

// Thermal sputtering (gradual erosion; also handles sublimation).
// Returns 1 if the particle was destroyed, 0 otherwise.
int erode_dust_grain_thermal(simparticles *Sp, int dust_idx, int nearest_gas_input,
                              double T_gas, double dt);

// ========== SHOCK DESTRUCTION ==========

// One SN tranche (one time slice of a star particle's supernovae) whose shock
// erodes the surrounding dust.
struct DustShockEvent
{
  MyIntPosType pos[3];
  std::uint64_t key;    // star ID; events are applied in key order
  double E_code;        // SN energy of the tranche, code units
  double rho_cgs;       // physical density of the nearest gas; <= 0 if none
  std::uint64_t gas_id; // nearest gas, which receives the eroded material
  int gas_task, gas_index;  // gas_task < 0 if no gas within 50 kpc
};

// Collective (all tasks must call): erode dust around the SN events of every
// task. Dust at indices >= first_new_dust was just created in this feedback
// call and is left alone.
void erode_dust_from_sn_shocks_batch(simparticles *Sp, const std::vector<DustShockEvent> &events,
                                     int first_new_dust, MPI_Comm comm);
double calculate_sn_shock_radius(double sn_energy_erg, double gas_density_cgs, double time_myr);
void get_shock_destruction_efficiency(double shock_velocity_km_s,
                                       double *eps_carb, double *eps_sil);

// ========== HELPER FUNCTIONS ==========

// Particle finding
int find_nearest_gas_particle(simparticles *Sp, int dust_idx, double max_r_kpc, double *out_dist_kpc = nullptr);

// Utility
double get_temperature_from_entropy(simparticles *Sp, int idx);
double calculate_velocity_difference(simparticles *Sp, int dust_idx, int gas_idx);

// ========== DIAGNOSTICS ==========

void print_dust_statistics(simparticles *Sp, MPI_Comm Communicator);
void analyze_dust_gas_coupling_local(simparticles *Sp);
void analyze_grain_size_distribution(simparticles *Sp);

/**
 * Sanity check of every dust particle (PartType 6), for debugging.
 *
 * Why: dust properties live in DustP[], a separate array from the base
 * particle data in P[]. Restarts from snapshots and from restart files often
 * broke the link between the two. This check catches dust data that was
 * lost or scrambled by a restart, domain decomposition, particle exchange,
 * reordering, cleanup, or the time integration itself.
 *
 * A local dust particle is "invalid" if any of these fail:
 *   - grain radius finite and > 0
 *   - dust temperature finite and >= 0
 *   - carbon mass fraction finite and in [0,1]
 *   - particle mass (in P[]) finite and > 0
 *
 * Each task prints its first few invalid particles ([BAD_DUST] lines: index,
 * ID, mass, radius, temperature, carbon fraction, DustSource). The global
 * totals are summed with MPI_Allreduce, so every task must call this; the
 * line that prints them is commented out below.
 *
 * Does nothing when DustDebugLevel <= 0.
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

            if(!std::isfinite(Sp->DustP[i].CarbonMassFraction) ||
               Sp->DustP[i].CarbonMassFraction < 0 ||
               Sp->DustP[i].CarbonMassFraction > 1)
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
                           Sp->DustP[i].CarbonMassFraction,
                           Sp->DustP[i].DustSource);
                }
            }
        }
    }

    long long send[3] = {ndust_local, valid_local, invalid_local};
    long long recv[3];

    MPI_Allreduce(send, recv, 3, MPI_LONG_LONG, MPI_SUM, Communicator);

    // Uncomment to print the global totals on task 0 when hunting corruption.
    // int ThisTask;
    // MPI_Comm_rank(Communicator, &ThisTask);
    // if(ThisTask == 0)
    // {
    //    printf("[DUST_CHECK|%s] "
    //           "ndust=%lld valid=%lld invalid=%lld\n",
    //           label,
    //           recv[0],
    //           recv[1],
    //           recv[2]);
    // }
}


#endif /* DUST */

#endif /* DUST_H */