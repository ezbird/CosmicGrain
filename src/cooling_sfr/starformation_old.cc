/*******************************************************************************
 * \copyright   This file is part of the GADGET4 N-body/SPH code developed
 * \copyright   by Volker Springel. Copyright (C) 2014-2020 by Volker Springel
 * \copyright   (vspringel@mpa-garching.mpg.de) and all contributing authors.
 *******************************************************************************/

/*! \file starformation.cc
 *
 *  \brief Generic creation routines for creating star particles
 */

#include "gadgetconfig.h"

#ifdef STARFORMATION

#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../cooling_sfr/cooling.h"
#include "../data/allvars.h"
#include "../data/dtypes.h"
#include "../data/mymalloc.h"
#include "../logs/logs.h"
#include "../logs/timer.h"
#include "../system/system.h"
#include "../time_integration/timestep.h"

#define SF_PRINT(...) do{ if(All.StarFormationDebugLevel){ \
  printf("[STAR_FORM|T=%d|a=%.6g z=%.3f] ", All.ThisTask, (double)All.Time, 1.0/All.Time-1.0); \
  printf(__VA_ARGS__); } }while(0)

// Forward declarations for diagnostic functions
static void check_particle_validity(simparticles *Sp, int idx, const char *context, int task);
static void check_for_overlaps(simparticles *Sp, int idx, int task);

/** \brief This routine creates star/wind particles according to their respective rates.
 *
 *  This function loops over all the active gas cells. If in a given cell the SFR is
 *  greater than zero, the probability of forming a star or a wind particle is computed
 *  and the corresponding particle is created stichastically according to the model
 *  in Springel & Hernquist (2003, MNRAS). It also saves information about the formed stellar
 *  mass and the star formation rate in the file FdSfr.
 */
void coolsfr::sfr_create_star_particles(simparticles *Sp)
{
  TIMER_START(CPU_COOLING_SFR);

  double dt, dtime;
  MyDouble mass_of_star;
  double sum_sm, total_sm, rate, sum_mass_stars, total_sum_mass_stars;
  double p = 0, pall = 0, prob, p_decide;
  double rate_in_msunperyear;
  double totsfrrate;
  double w = 0;

  All.set_cosmo_factors_for_current_time();

  stars_spawned = stars_converted = 0;

  sum_sm = sum_mass_stars = 0;

  for(int i = 0; i < Sp->TimeBinsHydro.NActiveParticles; i++)
    {
      int target = Sp->TimeBinsHydro.ActiveParticleList[i];
      if(Sp->P[target].getType() == 0)
        {
          if(Sp->P[target].getMass() == 0 && Sp->P[target].ID.get() == 0)
            continue; /* skip cells that have been swallowed or eliminated */

          dt = (Sp->P[target].getTimeBinHydro() ? (((integertime)1) << Sp->P[target].getTimeBinHydro()) : 0) * All.Timebase_interval;
          /*  the actual time-step */

          dtime = All.cf_atime * dt / All.cf_atime_hubble_a;

          mass_of_star = 0;
          prob         = 0;
          p            = 0;

          if(Sp->SphP[target].Sfr > 0)
            {
              // SphP[target].Sfr is in Msun/yr; convert to code-mass/time units first
              double sfr_code_units = Sp->SphP[target].Sfr *
                        (SOLAR_MASS / All.UnitMass_in_g) *
                        (All.UnitTime_in_s / SEC_PER_YEAR);

              p = sfr_code_units * dtime / Sp->P[target].getMass();


              pall = p;
              sum_sm += Sp->P[target].getMass() * (1 - exp(-p));

              w = get_random_number();

              Sp->SphP[target].Metallicity += w * METAL_YIELD * (1 - exp(-p));
              Sp->SphP[target].MassMetallicity = Sp->SphP[target].Metallicity * Sp->P[target].getMass();
              Sp->P[target].Metallicity        = Sp->SphP[target].Metallicity;

              mass_of_star = Sp->P[target].getMass();

              prob = Sp->P[target].getMass() / mass_of_star * (1 - exp(-pall));
            }

          if(prob == 0)
            continue;

          if(prob < 0)
            Terminate("prob < 0");

          /* decide what process to consider (currently available: make a star or kick to wind) */
          p_decide = get_random_number();

          if(p_decide < p / pall) /* ok, a star formation is considered */
            {
              // Check particle before star formation
              check_particle_validity(Sp, target, "BEFORE_STAR_FORMATION", ThisTask);
              
              make_star(Sp, target, prob, mass_of_star, &sum_mass_stars);
              
              // Check particles after star formation
              if(Sp->P[target].getType() == STAR_TYPE)
                {
                  // Particle was converted to star
                  check_particle_validity(Sp, target, "AFTER_CONVERT_TO_STAR", ThisTask);
                  check_for_overlaps(Sp, target, ThisTask);
                }
              else if(Sp->P[target].getType() == 0)
                {
                  // Particle spawned a star - check both gas and new star
                  check_particle_validity(Sp, target, "AFTER_SPAWN_GAS_PARENT", ThisTask);
                  if(stars_spawned > 0)
                    {
                      int new_star_idx = Sp->NumPart + stars_spawned - 1;
                      check_particle_validity(Sp, new_star_idx, "AFTER_SPAWN_NEW_STAR", ThisTask);
                      check_for_overlaps(Sp, new_star_idx, ThisTask);
                    }
                }
            }

          if(Sp->SphP[target].Sfr > 0)
            {
              if(Sp->P[target].getType() == 0) /* to protect using a particle that has been turned into a star */
                {
                  Sp->SphP[target].Metallicity += (1 - w) * METAL_YIELD * (1 - exp(-p));
                  Sp->SphP[target].MassMetallicity = Sp->SphP[target].Metallicity * Sp->P[target].getMass();
                }
            }
          Sp->P[target].Metallicity = Sp->SphP[target].Metallicity;
        }
    } /* end of main loop over active gas particles */

  MPI_Allreduce(&stars_spawned, &tot_stars_spawned, 1, MPI_INT, MPI_SUM, Communicator);
  MPI_Allreduce(&stars_converted, &tot_stars_converted, 1, MPI_INT, MPI_SUM, Communicator);

  if(tot_stars_spawned > 0 || tot_stars_converted > 0)
    {
      mpi_printf("SFR: spawned %d stars, converted %d gas particles into stars\n", tot_stars_spawned, tot_stars_converted);
    }

  tot_altogether_spawned = tot_stars_spawned;
  altogether_spawned     = stars_spawned;
  if(tot_altogether_spawned)
    {
      /* need to assign new unique IDs to the spawned stars */

      if(All.MaxID == 0) /* MaxID not calculated yet */
        {
          /* determine maximum ID */
          MyIDType maxid = 0;
          for(int i = 0; i < Sp->NumPart; i++)
            if(Sp->P[i].ID.get() > maxid)
              {
                maxid = Sp->P[i].ID.get();
              }

          MyIDType *tmp = (MyIDType *)Mem.mymalloc("tmp", NTask * sizeof(MyIDType));

          MPI_Allgather(&maxid, sizeof(MyIDType), MPI_BYTE, tmp, sizeof(MyIDType), MPI_BYTE, Communicator);

          for(int i = 0; i < NTask; i++)
            if(tmp[i] > maxid)
              maxid = tmp[i];

          All.MaxID = maxid;

          Mem.myfree(tmp);
        }

      int *list = (int *)Mem.mymalloc("list", NTask * sizeof(int));

      MPI_Allgather(&altogether_spawned, 1, MPI_INT, list, 1, MPI_INT, Communicator);

      MyIDType newid = All.MaxID + 1;

      for(int i = 0; i < ThisTask; i++)
        newid += list[i];

      Mem.myfree(list);

      for(int i = 0; i < altogether_spawned; i++)
        Sp->P[Sp->NumPart + i].ID.set(newid++);

      All.MaxID += tot_altogether_spawned;
    }

  /* Note: New tree construction can be avoided because of  `force_add_star_to_tree()' */
  if(tot_stars_spawned > 0 || tot_stars_converted > 0)
    {
      Sp->TotNumPart += tot_stars_spawned;
      Sp->TotNumGas -= tot_stars_converted;
      Sp->NumPart += stars_spawned;
    }

  double sfrrate = 0;
  for(int bin = 0; bin < TIMEBINS; bin++)
    if(Sp->TimeBinsHydro.TimeBinCount[bin])
      sfrrate += Sp->TimeBinSfr[bin];

  MPI_Allreduce(&sfrrate, &totsfrrate, 1, MPI_DOUBLE, MPI_SUM, Communicator);

  MPI_Reduce(&sum_sm, &total_sm, 1, MPI_DOUBLE, MPI_SUM, 0, Communicator);
  MPI_Reduce(&sum_mass_stars, &total_sum_mass_stars, 1, MPI_DOUBLE, MPI_SUM, 0, Communicator);
  if(ThisTask == 0)
    {
      if(All.TimeStep > 0)
        rate = total_sm / (All.TimeStep / All.cf_atime_hubble_a);
      else
        rate = 0;

      /* compute the cumulative mass of stars */
      cum_mass_stars += total_sum_mass_stars;

      /* convert to solar masses per yr */
      rate_in_msunperyear = rate * (All.UnitMass_in_g / SOLAR_MASS) / (All.UnitTime_in_s / SEC_PER_YEAR);

      fprintf(Logs.FdSfr, "%14e %14e %14e %14e %14e %14e\n", All.Time, total_sm, totsfrrate, rate_in_msunperyear, total_sum_mass_stars,
              cum_mass_stars);
      myflush(Logs.FdSfr);
    }

  TIMER_STOP(CPU_COOLING_SFR);
}

/** \brief Check particle for invalid or extreme values that could cause crashes
 *
 *  \param Sp pointer to simparticles
 *  \param idx index of particle to check
 *  \param context string describing when/where the check is happening
 *  \param task MPI task number
 */
static void check_particle_validity(simparticles *Sp, int idx, const char *context, int task)
{
  if(idx < 0 || idx >= Sp->NumPart + 100) // Allow some buffer for newly spawned
    return;
    
  particle_data *P = &Sp->P[idx];
  
  // Get position in physical coordinates
  double pos[3];
  Sp->intpos_to_pos(P->IntPos, pos);
  
  // Check velocities for NaN/Inf
  for(int k = 0; k < 3; k++)
    {
      if(!isfinite(P->Vel[k]))
        {
          SF_PRINT("STAR_FORM_CHECK|T=%d|%s: Particle ID=%lld has INVALID velocity[%d]=%g\n",
                 task, context, (long long)P->ID.get(), k, P->Vel[k]);
          SF_PRINT("STAR_FORM_CHECK: Pos=(%g,%g,%g), Mass=%g, Type=%d\n",
                 pos[0], pos[1], pos[2], P->getMass(), P->getType());
          fflush(stdout);
        }
    }
  
  // Check for extreme velocities
  double vmag = sqrt(P->Vel[0]*P->Vel[0] + P->Vel[1]*P->Vel[1] + P->Vel[2]*P->Vel[2]);
  if(vmag > 1000.0)
    {
      SF_PRINT("STAR_FORM_CHECK|T=%d|%s: Particle ID=%lld has EXTREME velocity=%g\n",
             task, context, (long long)P->ID.get(), vmag);
      SF_PRINT("STAR_FORM_CHECK: Vel=(%g,%g,%g), Pos=(%g,%g,%g), Mass=%g, Type=%d\n",
             P->Vel[0], P->Vel[1], P->Vel[2], pos[0], pos[1], pos[2], 
             P->getMass(), P->getType());
      fflush(stdout);
    }
  
  // Check mass
  if(!isfinite(P->getMass()) || P->getMass() <= 0)
    {
      SF_PRINT("STAR_FORM_CHECK|T=%d|%s: Particle ID=%lld has INVALID mass=%g\n",
             task, context, (long long)P->ID.get(), P->getMass());
      fflush(stdout);
    }
  
  // Check for extreme accelerations via GravAccel if available
  #ifdef EVALPOTENTIAL
  double amag = sqrt(P->GravAccel[0]*P->GravAccel[0] + 
                    P->GravAccel[1]*P->GravAccel[1] + 
                    P->GravAccel[2]*P->GravAccel[2]);
  if(!isfinite(amag) || amag > 1e10)
    {
      SF_PRINT("STAR_FORM_CHECK|T=%d|%s: Particle ID=%lld has EXTREME acceleration=%g\n",
             task, context, (long long)P->ID.get(), amag);
      fflush(stdout);
    }
  #endif
}

/** \brief Check if newly created particle overlaps with nearby particles
 *
 *  \param Sp pointer to simparticles
 *  \param idx index of particle to check
 *  \param task MPI task number
 */
static void check_for_overlaps(simparticles *Sp, int idx, int task)
{
  if(idx < 0 || idx >= Sp->NumPart + 100)
    return;
    
  particle_data *P = &Sp->P[idx];
  double pos1[3];
  Sp->intpos_to_pos(P->IntPos, pos1);
  
  double min_sep = 1e30;
  int closest_idx = -1;
  
  // Check against nearby particles (just check first 1000 to avoid too much overhead)
  int check_max = (Sp->NumPart < 1000) ? Sp->NumPart : 1000;
  
  for(int j = 0; j < check_max; j++)
    {
      if(j == idx) continue;
      
      double pos2[3];
      Sp->intpos_to_pos(Sp->P[j].IntPos, pos2);
      
      double dx = pos1[0] - pos2[0];
      double dy = pos1[1] - pos2[1];
      double dz = pos1[2] - pos2[2];
      double r = sqrt(dx*dx + dy*dy + dz*dz);
      
      if(r < min_sep)
        {
          min_sep = r;
          closest_idx = j;
        }
      
      // Flag very close particles
      if(r < 0.001) // Less than 1 pc
        {
          printf("OVERLAP_WARNING|T=%d: Particles ID=%lld and ID=%lld separated by only %g!\n",
                 task, (long long)P->ID.get(), (long long)Sp->P[j].ID.get(), r);
          printf("OVERLAP_WARNING: P1: Pos=(%g,%g,%g), Type=%d, Mass=%g\n",
                 pos1[0], pos1[1], pos1[2], P->getType(), P->getMass());
          printf("OVERLAP_WARNING: P2: Pos=(%g,%g,%g), Type=%d, Mass=%g\n",
                 pos2[0], pos2[1], pos2[2], 
                 Sp->P[j].getType(), Sp->P[j].getMass());
          fflush(stdout);
        }
    }
  
  if(closest_idx >= 0 && min_sep < 0.01) // Less than 10 pc
    {
      printf("CLOSE_PARTICLE|T=%d: Newly created particle ID=%lld closest neighbor is %g away (ID=%lld)\n",
             task, (long long)P->ID.get(), min_sep, (long long)Sp->P[closest_idx].ID.get());
      fflush(stdout);
    }
}

/** \brief Convert a SPH particle into a star.
 *
 *  This function convertss an active star-forming gas particle into a star.
 *  The particle information of the gas is copied to the
 *  location istar and the fields necessary for the creation of the star
 *  particle are initialized.
 *
 *  \param i index of the gas particle to be converted
 *  \param birthtime time of birth (in code units) of the stellar particle
 */
void coolsfr::convert_sph_particle_into_star(simparticles *Sp, int i, double birthtime)
{
  Sp->P[i].setType(STAR_TYPE);
#if NSOFTCLASSES > 1
  Sp->P[i].setSofteningClass(All.SofteningClassOfPartType[Sp->P[i].getType()]);
#endif
#ifdef INDIVIDUAL_GRAVITY_SOFTENING
  if(((1 << Sp->P[i].getType()) & (INDIVIDUAL_GRAVITY_SOFTENING)))
    Sp->P[i].setSofteningClass(Sp->get_softening_type_from_mass(Sp->P[i].getMass()));
#endif

  Sp->TimeBinSfr[Sp->P[i].getTimeBinHydro()] -= Sp->SphP[i].Sfr;

  Sp->P[i].StellarAge = birthtime;

  #ifdef FEEDBACK
    Sp->P[i].FeedbackFlag = 0;      // No feedback done yet
    Sp->P[i].EnergyReservoir = 0.0; // No stored energy
  #endif

  return;
}

/** \brief Spawn a star particle from a SPH gas particle.
 *
 *  This function spawns a star particle from an active star-forming
 *  SPH gas particle. The particle information of the gas is copied to the
 *  location istar and the fields necessary for the creation of the star
 *  particle are initialized. The total mass of the gas particle is split
 *  between the newly spawned star and the gas particle.
 *  (This function is probably unecessary)
 *
 *  \param igas index of the gas cell from which the star is spawned
 *  \param birthtime time of birth (in code units) of the stellar particle
 *  \param istar index of the spawned stellar particle
 *  \param mass_of_star the mass of the spawned stellar particle
 */
void coolsfr::spawn_star_from_sph_particle(simparticles *Sp, int igas, double birthtime, int istar, MyDouble mass_of_star)
{
  Sp->P[istar] = Sp->P[igas];

  // ADD SMALL RANDOM OFFSET TO PREVENT OVERLAP
  // This was creating huge accelerations with feedback turned on.
  // Offset by ~0.1 * softening length in a random direction
  double offset_mag = 0.2 * All.SofteningTable[Sp->P[istar].getSofteningClass()]; // 0.4 pc for 2pc softening
  double theta = 2.0 * M_PI * get_random_number();
  double phi = acos(2.0 * get_random_number() - 1.0);
  
  double offset[3];
  offset[0] = offset_mag * sin(phi) * cos(theta);
  offset[1] = offset_mag * sin(phi) * sin(theta);
  offset[2] = offset_mag * cos(phi);
  
  // Convert to integer positions and add offset
  MyIntPosType offset_intpos[3];
  Sp->pos_to_intpos(offset, offset_intpos);
  
  for(int k = 0; k < 3; k++) {
    Sp->P[istar].IntPos[k] += offset_intpos[k];
  }
    printf("SPAWN_OFFSET|T=%d: Star ID will be %u, offset=(%g,%g,%g), mag=%g\n",
       All.ThisTask,
       (unsigned)Sp->P[istar].ID.get(),
       offset[0], offset[1], offset[2], offset_mag);


  Sp->P[istar].setType(STAR_TYPE);
#if NSOFTCLASSES > 1
  Sp->P[istar].setSofteningClass(All.SofteningClassOfPartType[Sp->P[istar].getType()]);
#endif
#ifdef INDIVIDUAL_GRAVITY_SOFTENING
  if(((1 << Sp->P[istar].getType()) & (INDIVIDUAL_GRAVITY_SOFTENING)))
    Sp->P[istar].setSofteningClass(Sp->get_softening_type_from_mass(Sp->P[istar].getMass()));
#endif

  Sp->TimeBinsGravity.ActiveParticleList[Sp->TimeBinsGravity.NActiveParticles++] = istar;

  Sp->TimeBinsGravity.timebin_add_particle(istar, igas, Sp->P[istar].TimeBinGrav, Sp->TimeBinSynchronized[Sp->P[istar].TimeBinGrav]);

  Sp->P[istar].setMass(mass_of_star);

  Sp->P[istar].StellarAge = birthtime;

if (All.FeedbackDebugLevel && All.ThisTask==0) {
  printf("[SF|a=%.6g z=%.2f] new star: ID=%u M=%.3e a_birth=%.6g\n",
         (double)All.Time, 1.0/All.Time-1.0,
         (unsigned)Sp->P[istar].ID.get(),
         Sp->P[istar].getMass(),
         Sp->P[istar].StellarAge);
}

  #ifdef FEEDBACK
    Sp->P[istar].FeedbackFlag = 0;      // No feedback done yet
    Sp->P[istar].EnergyReservoir = 0.0; // No stored energy
  #endif

  /* now change the conserved quantities in the cell in proportion */
  double fac = (Sp->P[igas].getMass() - Sp->P[istar].getMass()) / Sp->P[igas].getMass();

  Sp->P[igas].setMass(fac * Sp->P[igas].getMass());

  return;
}

/** \brief Make a star particle from a SPH gas particle.
 *
 *  Given a gas cell where star formation is active and the probability
 *  of forming a star, this function selectes either to convert the gas
 *  particle into a star particle or to spawn a star depending on the
 *  target mass for the star.
 *
 *  \param i index of the gas cell
 *  \param prob probability of making a star
 *  \param mass_of_star desired mass of the star particle
 *  \param sum_mass_stars holds the mass of all the stars created at the current time-step (for the local task)
 */
void coolsfr::make_star(simparticles *Sp, int i, double prob, MyDouble mass_of_star, double *sum_mass_stars)
{
  if(mass_of_star > Sp->P[i].getMass())
    Terminate("mass_of_star > P[i].Mass");

  if(get_random_number() < prob)
    {
      if(mass_of_star == Sp->P[i].getMass())
        {
          /* here we turn the gas particle itself into a star particle */
          stars_converted++;

          *sum_mass_stars += Sp->P[i].getMass();

          convert_sph_particle_into_star(Sp, i, All.Time);
        }
      else
        {
          /* in this case we spawn a new star particle, only reducing the mass in the cell by mass_of_star */
          altogether_spawned = stars_spawned;
          if(Sp->NumPart + altogether_spawned >= Sp->MaxPart)
            Terminate("NumPart=%d spwawn %d particles no space left (Sp.MaxPart=%d)\n", Sp->NumPart, altogether_spawned, Sp->MaxPart);

          int j = Sp->NumPart + altogether_spawned; /* index of new star */

          spawn_star_from_sph_particle(Sp, i, All.Time, j, mass_of_star);

          *sum_mass_stars += mass_of_star;
          stars_spawned++;
        }
    }
}

#endif /* closes SFR */