/*******************************************************************************
 * \copyright   This file is part of the GADGET4 N-body/SPH code developed
 * \copyright   by Volker Springel. Copyright (C) 2014-2020 by Volker Springel
 * \copyright   (vspringel@mpa-garching.mpg.de) and all contributing authors.
 *******************************************************************************/

/*! \file energy_ledger.h
 *
 *  \brief Gas energy bookkeeping per phase of the main loop.
 *
 *  The 2048^3 halo441 run gained ~100x more thermal energy in ~70 Myr than
 *  its supernovae could supply.  This ledger measures the total gas thermal
 *  and kinetic energy at fixed points of each step and attributes every
 *  change to the phase between two points (drift, density, hydro, gravity,
 *  cooling/star formation, feedback, dust).  Totals are printed every
 *  LEDGER_PRINT_STEPS steps; a single phase that raises the thermal energy
 *  by more than LEDGER_JUMP_FRACTION is reported at once, with the hottest
 *  gas particle.
 */

#ifndef ENERGY_LEDGER_H
#define ENERGY_LEDGER_H

#include <mpi.h>
#include <cmath>
#include <cstdio>

#include "../data/allvars.h"
#include "../data/simparticles.h"

long long feedback_hot_skips(void);

namespace cg_ledger
{
enum Phase
{
  PH_DRIFT,
  PH_DENSITY,
  PH_HYDRO2,
  PH_GRAV2,
  PH_COOLSFR,
  PH_FEEDBACK,
  PH_DUST,
  PH_KICK1_HYDRO1,
  PH_NPHASES
};
static const char *phase_name[PH_NPHASES] = {"drift", "density", "hydro2", "grav2", "coolsfr", "feedback", "dust", "kick1+hydro1"};

static const int LEDGER_PRINT_STEPS          = 500;
static const double LEDGER_JUMP_FRACTION     = 0.005;
static const double LEDGER_HOT_REPORT_K      = 1.0e9;

static double Last[2], Acc[PH_NPHASES][2];
static bool HaveLast = false;
static int StepsSincePrint = 0;

struct Hottest
{
  double T;
  int task;
};

static inline double gas_temperature(simparticles *Sp, int i, double u)
{
  const double mu = 4.0 / (1.0 + 3.0 * HYDROGEN_MASSFRAC + 4.0 * HYDROGEN_MASSFRAC * Sp->SphP[i].Ne);
  return GAMMA_MINUS1 * u * All.UnitVelocity_in_cm_per_s * All.UnitVelocity_in_cm_per_s * mu * PROTONMASS / BOLTZMANN;
}

/* Global gas thermal and kinetic energy (code units) and the hottest particle. */
static void measure(simparticles *Sp, MPI_Comm comm, double E[2], Hottest &hot, int &hot_index)
{
  double local[2] = {0, 0};
  Hottest mine    = {0, 0};
  MPI_Comm_rank(comm, &mine.task);
  hot_index = -1;
  for(int i = 0; i < Sp->NumGas; i++)
    {
      if(Sp->P[i].getType() != 0 || Sp->P[i].getMass() <= 0)
        continue;
      const double m = Sp->P[i].getMass();
      const double u = Sp->get_utherm_from_entropy(i);
      double v2      = 0;
      for(int k = 0; k < 3; k++)
        v2 += (double)Sp->P[i].Vel[k] * Sp->P[i].Vel[k];
      local[0] += m * u;
      local[1] += 0.5 * m * v2 * All.cf_a2inv;
      const double T = gas_temperature(Sp, i, u);
      if(T > mine.T)
        {
          mine.T    = T;
          hot_index = i;
        }
    }
  MPI_Allreduce(local, E, 2, MPI_DOUBLE, MPI_SUM, comm);
  MPI_Allreduce(&mine, &hot, 1, MPI_DOUBLE_INT, MPI_MAXLOC, comm);
  if(hot.task != mine.task)
    hot_index = -1;
}

static void report_hottest(simparticles *Sp, int i, const char *why)
{
  if(i < 0)
    return;
  const double u  = Sp->get_utherm_from_entropy(i);
  const double nH = HYDROGEN_MASSFRAC * Sp->SphP[i].Density * All.cf_a3inv * All.UnitDensity_in_cgs * All.HubbleParam *
                    All.HubbleParam / PROTONMASS;
  double v2 = 0, pos[3];
  for(int k = 0; k < 3; k++)
    v2 += (double)Sp->P[i].Vel[k] * Sp->P[i].Vel[k];
  Sp->intpos_to_pos(Sp->P[i].IntPos, pos);
  printf("[ENERGY_HOTTEST|%s|a=%.6f] gas ID %lld T=%.3e K nH=%.3e v=%.0f km/s Hsml=%.3e mass=%.3e SFR=%.3e bin=%d pos=(%.1f|%.1f|%.1f)\n",
         why, (double)All.Time, (long long)Sp->P[i].ID.get(), gas_temperature(Sp, i, u), nH, sqrt(v2) * All.cf_ainv,
         (double)Sp->SphP[i].Hsml, Sp->P[i].getMass(), (double)Sp->SphP[i].Sfr, (int)Sp->P[i].getTimeBinHydro(), pos[0], pos[1],
         pos[2]);
  fflush(stdout);
}

/* Measure the energies and attribute the change since the previous point to phase p. */
static void point(simparticles *Sp, MPI_Comm comm, Phase p)
{
  double E[2];
  Hottest hot;
  int hot_index;
  measure(Sp, comm, E, hot, hot_index);
  if(HaveLast)
    {
      const double dE[2] = {E[0] - Last[0], E[1] - Last[1]};
      Acc[p][0] += dE[0];
      Acc[p][1] += dE[1];
      if(dE[0] > LEDGER_JUMP_FRACTION * Last[0])  // gains only: expansion cools gas every drift
        {
          if(All.ThisTask == 0)
            printf("[ENERGY_JUMP|step=%d|a=%.6f] phase %s changed gas thermal energy by %+.3e (%+.2f%% of %.4e), kinetic by %+.3e; Tmax=%.3e K\n",
                   All.NumCurrentTiStep, (double)All.Time, phase_name[p], dE[0], 100 * dE[0] / Last[0], Last[0], dE[1], hot.T);
          report_hottest(Sp, hot_index, phase_name[p]);
        }
    }
  static double reported_T = LEDGER_HOT_REPORT_K;
  if(hot.T > reported_T)
    {
      report_hottest(Sp, hot_index, "new-Tmax");
      reported_T = 2 * hot.T;
    }
  Last[0]  = E[0];
  Last[1]  = E[1];
  HaveLast = true;

  if(p == PH_KICK1_HYDRO1 && ++StepsSincePrint >= LEDGER_PRINT_STEPS)
    {
      long long skips = feedback_hot_skips(), skips_sum = 0;
      MPI_Reduce(&skips, &skips_sum, 1, MPI_LONG_LONG, MPI_SUM, 0, comm);
      if(All.ThisTask == 0)
        {
          printf("[ENERGY_LEDGER|step=%d|a=%.6f] Eth=%.5e Ekin=%.5e Tmax=%.3e K hot_skips=%lld; dEth/dEkin by phase over %d steps:", All.NumCurrentTiStep,
                 (double)All.Time, E[0], E[1], hot.T, skips_sum, StepsSincePrint);
          for(int q = 0; q < PH_NPHASES; q++)
            printf(" %s=%+.3e/%+.3e", phase_name[q], Acc[q][0], Acc[q][1]);
          printf("\n");
        }
      for(int q = 0; q < PH_NPHASES; q++)
        Acc[q][0] = Acc[q][1] = 0;
      StepsSincePrint = 0;
    }
}
}  // namespace cg_ledger

#endif
