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
#include <vector>
#include <algorithm>

#include "../cooling_sfr/cooling.h"
#include "../data/allvars.h"
#include "../data/dtypes.h"
#include "../data/mymalloc.h"
#include "../logs/logs.h"
#include "../logs/timer.h"
#include "../system/system.h"
#include "../time_integration/timestep.h"

// Queue **IDs** (not indices) of gas marked for full conversion this step
namespace { std::vector<MyIDType> SF_convert_queue; }

// Debug printing macros for star formation
// Level 0: No debug output
// Level 1: Only print when stars actually form (summaries only, from task 0)
// Level 2: Print summaries from all tasks that have SF activity
// Level 3: Print detailed per-particle info

#define SF_PRINT_DETAILED(...) do{ if(All.StarformationDebugLevel >= 3){ \
  printf("[STARFORMATION|T=%d|a=%.6g z=%.3f] ", ThisTask, (double)All.Time, 1.0/All.Time-1.0); \
  printf(__VA_ARGS__); fflush(stdout); } }while(0)

#define SF_PRINT_SUMMARY(...) do{ if(All.StarformationDebugLevel >= 2){ \
  printf("[STARFORMATION|T=%d|a=%.6g z=%.3f] ", ThisTask, (double)All.Time, 1.0/All.Time-1.0); \
  printf(__VA_ARGS__); fflush(stdout); } }while(0)

#define SF_PRINT_TASK0(...) do{ if(All.StarformationDebugLevel >= 1 && ThisTask == 0){ \
  printf("[STARFORMATION|T=%d|a=%.6g z=%.3f] ", ThisTask, (double)All.Time, 1.0/All.Time-1.0); \
  printf(__VA_ARGS__); fflush(stdout); } }while(0)

// --- local helper: resolve ID -> local index (O(N) fallback) ---
static inline int find_index_by_id(simparticles* Sp, MyIDType id)
{
  for (int i = 0; i < Sp->NumPart; i++)
    if (Sp->P[i].ID.get() == id) return i;
  return -1;
}

/** \brief Finalize all pending star conversions (run at a safe end-of-step barrier).
 *
 *  IMPORTANT: Call this from the timestep driver AFTER hydro/cooling/SF for the
 *  current global step are finished, and BEFORE any next-step hydro/density/tree
 *  structures are (re)built.
 *
 *  This routine flips Type from GAS->STAR in-place for all queued IDs, after we
 *  quarantined them earlier in the step. It avoids mid-step SphP access on stars.
 */
void coolsfr::flush_star_conversions(simparticles* Sp, double time_now)
{
  if (SF_convert_queue.empty()) return;

  std::sort(SF_convert_queue.begin(), SF_convert_queue.end());
  SF_convert_queue.erase(std::unique(SF_convert_queue.begin(), SF_convert_queue.end()),
                         SF_convert_queue.end());

  int locally_converted = 0;

  for (MyIDType id : SF_convert_queue)
  {
    int i = find_index_by_id(Sp, id);
    if (i < 0) continue;
    if (Sp->P[i].getType() != 0) continue; // only gas

    // Flip type to STAR at the safe barrier
    Sp->P[i].setType(STAR_TYPE);

#if NSOFTCLASSES > 1
    Sp->P[i].setSofteningClass(All.SofteningClassOfPartType[Sp->P[i].getType()]);
#endif
#ifdef INDIVIDUAL_GRAVITY_SOFTENING
    if(((1 << Sp->P[i].getType()) & (INDIVIDUAL_GRAVITY_SOFTENING)))
      Sp->P[i].setSofteningClass(Sp->get_softening_type_from_mass(Sp->P[i].getMass()));
#endif

    // Ensure no hydro re-activation
    Sp->P[i].TimeBinHydro = 0;

    locally_converted++;
  }

  SF_convert_queue.clear();

  if (All.StarformationDebugLevel >= 2 && locally_converted > 0)
    SF_PRINT_SUMMARY("Finalized %d in-place gas->star conversions on task %d\n", locally_converted, ThisTask);
}


/** \brief This routine creates star/wind particles according to their respective rates.
 *
 *  Loops over active gas. Stochastic SF following Springel & Hernquist (2003).
 *  NOTE: Full conversions are now only **marked/quarantined** here; the actual
 *  Type flip happens later in flush_star_conversions() at a safe barrier.
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

  int n_sf_candidates = 0;
  int n_stars_attempted = 0;
  double max_sfr = 0;
  double total_sf_gas_mass = 0;
  double max_prob = 0;

  // Only print if level >= 2
  if(All.StarformationDebugLevel >= 2 && ThisTask == 0)
    printf("[STARFORMATION|T=%d|a=%.6g z=%.3f] Starting star formation evaluation\n",
           ThisTask, (double)All.Time, 1.0/All.Time-1.0);

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
              n_sf_candidates++;

              p = Sp->SphP[target].Sfr / ((All.UnitMass_in_g / SOLAR_MASS) / (All.UnitTime_in_s / SEC_PER_YEAR)) * dtime /
                  Sp->P[target].getMass();
              pall = p;
              sum_sm += Sp->P[target].getMass() * (1 - exp(-p));

              // Track statistics
              if(Sp->SphP[target].Sfr > max_sfr)
                max_sfr = Sp->SphP[target].Sfr;
              total_sf_gas_mass += Sp->P[target].getMass();

              w = get_random_number();

              // Metallicity enrichment (first part)
              double metal_added = w * METAL_YIELD * (1 - exp(-p));
              Sp->SphP[target].Metallicity += metal_added;
              Sp->SphP[target].MassMetallicity = Sp->SphP[target].Metallicity * Sp->P[target].getMass();
              Sp->P[target].Metallicity        = Sp->SphP[target].Metallicity;

              SF_PRINT_DETAILED("  Particle %llu: SFR=%.3e Msun/yr, mass=%.3e Msun, p=%.4f, Z+=%.3e\n",
                       (unsigned long long)Sp->P[target].ID.get(),
                       Sp->SphP[target].Sfr * (All.UnitMass_in_g / SOLAR_MASS) / (All.UnitTime_in_s / SEC_PER_YEAR),
                       Sp->P[target].getMass() * (All.UnitMass_in_g / SOLAR_MASS) / All.HubbleParam,
                       p, metal_added);

              mass_of_star = Sp->P[target].getMass();

              prob = Sp->P[target].getMass() / mass_of_star * (1 - exp(-pall));

              if(prob > max_prob)
                max_prob = prob;
            }

          if(prob == 0)
            continue;

          if(prob < 0)
            Terminate("prob < 0");

          n_stars_attempted++;

          /* decide what process to consider (currently available: make a star or kick to wind) */
          p_decide = get_random_number();

          SF_PRINT_DETAILED("  Particle %llu attempting SF: prob=%.4f, p_decide=%.4f\n",
                   (unsigned long long)Sp->P[target].ID.get(), prob, p_decide);

          if(p_decide < p / pall) /* ok, a star formation is considered */
            {
              SF_PRINT_DETAILED("  -> Star formation selected for particle %llu\n",
                       (unsigned long long)Sp->P[target].ID.get());
              make_star(Sp, target, prob, mass_of_star, &sum_mass_stars);
            }

          if(Sp->SphP[target].Sfr > 0)
            {
              if(Sp->P[target].getType() == 0) /* still gas */
                {
                  // Metallicity enrichment (second part)
                  double metal_added2 = (1 - w) * METAL_YIELD * (1 - exp(-p));
                  Sp->SphP[target].Metallicity += metal_added2;
                  Sp->SphP[target].MassMetallicity = Sp->SphP[target].Metallicity * Sp->P[target].getMass();

                  SF_PRINT_DETAILED("  Particle %llu still gas, adding Z+=%.3e\n",
                           (unsigned long long)Sp->P[target].ID.get(), metal_added2);
                }
            }
          Sp->P[target].Metallicity = Sp->SphP[target].Metallicity;
        }
    } /* end of main loop over active gas particles */

  // NOTE: We intentionally **do not** finalize conversions here.
  // Call flush_star_conversions(Sp, All.Time) once per global step from the timestep driver,
  // after hydro/cooling/SF and before any next-step hydro/density/tree builds.

  // Only print local summary if there was actual SF activity
  if(All.StarformationDebugLevel >= 2 && (n_sf_candidates > 0 || stars_spawned > 0 || stars_converted > 0))
    {
      printf("[STARFORMATION|T=%d|a=%.6g z=%.3f] Local summary: SF candidates=%d, attempts=%d, spawned=%d, converted=%d\n",
             ThisTask, (double)All.Time, 1.0/All.Time-1.0,
             n_sf_candidates, n_stars_attempted, stars_spawned, stars_converted);
      printf("[STARFORMATION|T=%d|a=%.6g z=%.3f]   Max SFR=%.3e Msun/yr, Max prob=%.4f, Total SF gas mass=%.3e Msun\n",
             ThisTask, (double)All.Time, 1.0/All.Time-1.0,
             max_sfr * (All.UnitMass_in_g / SOLAR_MASS) / (All.UnitTime_in_s / SEC_PER_YEAR),
             max_prob,
             total_sf_gas_mass * (All.UnitMass_in_g / SOLAR_MASS) / All.HubbleParam);
      fflush(stdout);
    }

  MPI_Allreduce(&stars_spawned, &tot_stars_spawned, 1, MPI_INT, MPI_SUM, Communicator);
  MPI_Allreduce(&stars_converted, &tot_stars_converted, 1, MPI_INT, MPI_SUM, Communicator);

  if(tot_stars_spawned > 0 || tot_stars_converted > 0)
    {
      mpi_printf("SFR: spawned %d stars, converted %d gas particles into stars\n", tot_stars_spawned, tot_stars_converted);

      // Additional detailed output
      int tot_candidates = 0, tot_attempts = 0;
      double global_max_sfr = 0, global_sf_mass = 0;
      MPI_Allreduce(&n_sf_candidates, &tot_candidates, 1, MPI_INT, MPI_SUM, Communicator);
      MPI_Allreduce(&n_stars_attempted, &tot_attempts, 1, MPI_INT, MPI_SUM, Communicator);
      MPI_Allreduce(&max_sfr, &global_max_sfr, 1, MPI_DOUBLE, MPI_MAX, Communicator);
      MPI_Allreduce(&total_sf_gas_mass, &global_sf_mass, 1, MPI_DOUBLE, MPI_SUM, Communicator);

      mpi_printf("SFR: Total SF candidates=%d, attempts=%d, success rate=%.1f%%\n",
                 tot_candidates, tot_attempts,
                 100.0 * (tot_stars_spawned + tot_stars_converted) / (tot_attempts > 0 ? tot_attempts : 1));
      mpi_printf("SFR: Max SFR=%.3e Msun/yr, Total SF gas mass=%.3e Msun\n",
                 global_max_sfr * (All.UnitMass_in_g / SOLAR_MASS) / (All.UnitTime_in_s / SEC_PER_YEAR),
                 global_sf_mass * (All.UnitMass_in_g / SOLAR_MASS) / All.HubbleParam);
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

      SF_PRINT_SUMMARY("Assigned IDs from %llu to %llu for %d spawned stars\n",
               (unsigned long long)(All.MaxID + 1), (unsigned long long)newid, altogether_spawned);

      All.MaxID += tot_altogether_spawned;
    }

  /* Note: New tree construction can be avoided because of  `force_add_star_to_tree()' */
  if(tot_stars_spawned > 0 || tot_stars_converted > 0)
    {
      Sp->TotNumPart += tot_stars_spawned;  // only spawned stars increase NumPart
      Sp->TotNumGas  -= tot_stars_converted; // conversions reduce gas count
      Sp->NumPart    += stars_spawned;

      SF_PRINT_TASK0("Updated particle counts: TotNumPart=%lld (+%d), TotNumGas=%lld (-%d)\n",
                 (long long)Sp->TotNumPart, tot_stars_spawned,
                 (long long)Sp->TotNumGas,  tot_stars_converted);
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

      if(All.StarformationDebugLevel)
        {
          printf("[STARFORMATION|T=%d|a=%.6g z=%.3f] SFR file: rate=%.3e Msun/yr, mass_formed=%.3e Msun, cum_mass=%.3e Msun\n",
                 ThisTask, (double)All.Time, 1.0/All.Time-1.0,
                 rate_in_msunperyear,
                 total_sum_mass_stars * (All.UnitMass_in_g / SOLAR_MASS) / All.HubbleParam,
                 cum_mass_stars * (All.UnitMass_in_g / SOLAR_MASS) / All.HubbleParam);
        }
    }

  TIMER_STOP(CPU_COOLING_SFR);
}

/** \brief Quarantine a SPH particle that is selected for full conversion this step.
 *
 *  NOTE: This function no longer flips Type. It only:
 *   - marks the particle,
 *   - makes SPH thermodynamics benign,
 *   - removes it from hydro time bins for the rest of the step,
 *   - updates SFR bookkeeping,
 *   - records the stellar birth time.
 *  The actual Type flip to STAR is done in flush_star_conversions() at the barrier.
 */
void coolsfr::convert_sph_particle_into_star(simparticles *Sp, int i, double birthtime)
{
  double star_mass_msun = Sp->P[i].getMass() * (All.UnitMass_in_g / SOLAR_MASS) / All.HubbleParam;

  SF_PRINT_DETAILED("CONVERTING (quarantine) particle %llu: mass=%.3e Msun, Z=%.4f, birthtime=%.6f\n",
           (unsigned long long)Sp->P[i].ID.get(),
           star_mass_msun,
           Sp->SphP[i].Metallicity,
           birthtime);

  Sp->P[i].StellarAge = birthtime;

  // ⬇️ capture original hydro bin first
  int oldbin = Sp->P[i].getTimeBinHydro();

  // keep it out of hydro for the rest of the step
  Sp->P[i].TimeBinHydro = 0;
  
  // subtract its SFR from the correct bin
  if (oldbin >= 0)
    Sp->TimeBinSfr[oldbin] -= Sp->SphP[i].Sfr;

  Sp->SphP[i].Sfr = 0;

  int *alist = Sp->TimeBinsHydro.ActiveParticleList;
  int  nact  = Sp->TimeBinsHydro.NActiveParticles;
  for (int k = 0; k < nact; ++k)
  {
    if (alist[k] == i)
    {
      // move last active into this slot and shrink
      alist[k] = alist[nact - 1];
      Sp->TimeBinsHydro.NActiveParticles = nact - 1;
      break;
    }
  }


}


/** \brief Spawn a star particle from a SPH gas particle (partial conversion).
 *
 *  Copies the parent to a new particle j with STAR_TYPE, reduces parent gas mass,
 *  and puts the new star on the gravity time bin. No manual ActiveParticleList pokes.
 */
void coolsfr::spawn_star_from_sph_particle(simparticles *Sp, int igas, double birthtime, int istar, MyDouble mass_of_star)
{
  double star_mass_msun   = mass_of_star * (All.UnitMass_in_g / SOLAR_MASS) / All.HubbleParam;
  double gas_mass_before  = Sp->P[igas].getMass() * (All.UnitMass_in_g / SOLAR_MASS) / All.HubbleParam;

  Sp->P[istar] = Sp->P[igas];
  Sp->P[istar].setType(STAR_TYPE);
#if NSOFTCLASSES > 1
  Sp->P[istar].setSofteningClass(All.SofteningClassOfPartType[Sp->P[istar].getType()]);
#endif
#ifdef INDIVIDUAL_GRAVITY_SOFTENING
  if(((1 << Sp->P[istar].getType()) & (INDIVIDUAL_GRAVITY_SOFTENING)))
    Sp->P[istar].setSofteningClass(Sp->get_softening_type_from_mass(Sp->P[istar].getMass()));
#endif

  // Place on gravity time bin using the API; do NOT touch ActiveParticleList directly
  Sp->TimeBinsGravity.timebin_add_particle(istar, igas, Sp->P[istar].TimeBinGrav,
                                           Sp->TimeBinSynchronized[Sp->P[istar].TimeBinGrav]);

  Sp->P[istar].setMass(mass_of_star);
  Sp->P[istar].StellarAge = birthtime;

  /* now change the conserved quantities in the cell in proportion */
  double fac = (Sp->P[igas].getMass() - Sp->P[istar].getMass()) / Sp->P[igas].getMass();
  Sp->P[igas].setMass(fac * Sp->P[igas].getMass());

  double gas_mass_after = Sp->P[igas].getMass() * (All.UnitMass_in_g / SOLAR_MASS) / All.HubbleParam;

  SF_PRINT_DETAILED("SPAWNING star from particle %llu: star_mass=%.3e Msun, gas_before=%.3e Msun, gas_after=%.3e Msun, Z=%.4f\n",
           (unsigned long long)Sp->P[igas].ID.get(),
           star_mass_msun,
           gas_mass_before,
           gas_mass_after,
           Sp->P[igas].Metallicity);

  return;
}

/** \brief Decide to spawn or (fully) convert a gas particle to a star.
 *
 *  Full conversions:
 *    - increment counters and sum_mass_stars
 *    - queue the particle’s **ID** for end-of-step finalize
 *    - quarantine the particle now (safe SPH + remove from hydro bins)
 *
 *  Partial conversions (spawn):
 *    - append a new STAR particle at the end
 *    - leave parent gas with reduced mass
 */
void coolsfr::make_star(simparticles *Sp, int i, double prob, MyDouble mass_of_star, double *sum_mass_stars)
{
  if(mass_of_star > Sp->P[i].getMass())
    Terminate("mass_of_star > P[i].Mass");

  double random_val = get_random_number();

  SF_PRINT_DETAILED("make_star for particle %llu: prob=%.4f, random=%.4f, decision=%s\n",
           (unsigned long long)Sp->P[i].ID.get(),
           prob, random_val,
           (random_val < prob) ? "FORM STAR" : "NO STAR");

  if (random_val < prob)
  {
    if (mass_of_star == Sp->P[i].getMass())
    {
      // --- Full conversion: defer actual type flip to end-of-step ---
      SF_PRINT_DETAILED("  -> Full conversion (deferred in-place at barrier)\n");
      stars_converted++;

      *sum_mass_stars += Sp->P[i].getMass();

      // Queue by ID and quarantine immediately
      SF_convert_queue.push_back(Sp->P[i].ID.get());
      convert_sph_particle_into_star(Sp, i, All.Time);
      return;
    }
    else
    {
      // --- Spawn new star, gas remains (reduced mass) ---
      SF_PRINT_DETAILED("  -> Spawning new star (gas particle remains)\n");

      altogether_spawned = stars_spawned;
      if (Sp->NumPart + altogether_spawned >= Sp->MaxPart)
        Terminate("NumPart=%d spawn %d particles: no space left (Sp.MaxPart=%d)\n",
                  Sp->NumPart, altogether_spawned, Sp->MaxPart);

      int j = Sp->NumPart + altogether_spawned;  // index of new star
      spawn_star_from_sph_particle(Sp, i, All.Time, j, mass_of_star);

      *sum_mass_stars += mass_of_star;
      stars_spawned++;
    }
  }
}

#endif /* closes STARFORMATION */
