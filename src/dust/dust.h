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
extern double    TotalMassEroded;

// ========== CORE DUST FUNCTIONS (dust.cc) ==========

// Dust particle creation and destruction
void create_dust_particles_from_feedback(simparticles *Sp, int star_idx, 
                                         double metals_produced, int feedback_type);
void spawn_dust_particle(simparticles *Sp, double offset_kpc[3], double dust_mass, 
                         double initial_velocity[3], int star_idx);
void destroy_dust_particles(simparticles *Sp);
void cleanup_invalid_dust_particles(simparticles *Sp);

// Dust dynamics and interaction
void update_dust_dynamics(simparticles *Sp, double dt, MPI_Comm Communicator);
int dust_gas_interaction(simparticles *Sp, int dust_idx, double dt);
void dust_global_synchronization(simparticles *Sp, MPI_Comm Communicator,
                                 long long dust_created,
                                 long long dust_destroyed,
                                 double dust_mass_change);

// ========== GRAIN GROWTH AND EROSION ==========

// Dust grain growth (subgrid model)
void dust_grain_growth_subgrid(simparticles *Sp, int gas_idx, double dt);
double estimate_molecular_fraction(double n_H, double Z, double T);

// Grain growth in cold, dense ISM
void dust_grain_growth(simparticles *Sp, int gas_idx, double dt);

// Gradual erosion functions
int erode_dust_grain_thermal(simparticles *Sp, int dust_idx, double T_gas, double dt);
int erode_dust_grain_shock(simparticles *Sp, int dust_idx, double shock_velocity_km_s, 
                           double distance_to_sn, double shock_radius);

// ========== SHOCK DESTRUCTION ==========

void destroy_dust_from_sn_shocks(simparticles *Sp, int sn_star_idx, 
                                 double sn_energy, double metals_produced);
double calculate_sn_shock_radius(double sn_energy_erg, double gas_density_cgs, double time_myr);
double calculate_current_sn_shock_radius(simparticles *Sp, int sn_star_idx);
double calculate_shock_velocity(double sn_energy_erg, double gas_density_cgs, double time_myr);
double get_shock_destruction_efficiency(double shock_velocity_km_s);
double get_size_dependent_destruction_efficiency(double shock_velocity_km_s, 
                                                 simparticles *Sp, int dust_idx);

// ========== HELPER FUNCTIONS ==========

// Particle finding
int find_nearest_gas_particle(simparticles *Sp, int dust_idx);
int find_nearest_dust_particle(simparticles *Sp, int gas_idx);

// Utility
double get_temperature_from_entropy(simparticles *Sp, int idx);
double calculate_velocity_difference(simparticles *Sp, int dust_idx, int gas_idx);
double get_dust_destruction_rate(double temperature, double density);

// ========== DIAGNOSTICS ==========

void print_dust_statistics(simparticles *Sp);
void analyze_dust_gas_coupling(simparticles *Sp);
void analyze_dust_gas_coupling_global(simparticles *Sp);
void analyze_grain_size_distribution(simparticles *Sp);

#endif /* DUST */

#endif /* DUST_H */