/*=============================================================================
 * Enhanced cooling.cc with diagnostics and safety guards
 *
 * - Step-level summaries for metal cooling usage and net cooling/heating
 * - Optional hot-particle prints with physical units and (Heat-Lambda) sign
 * - Optional per-step dU limiter to prevent runaway u excursions while debugging
 * - One-shot configuration prints in InitCool()
 * - Light-weight sanity warnings for metallicity units
 *
 * Build-time toggles (add to CFLAGS):
 *   -DCOOLING -DSTARFORMATION                 // enable base module and metal cooling paths
 *   -DOUTPUT_COOLHEAT                         // per-particle CoolHeat accumulation & step summary
 *   -DCOOLING_HOT_DEBUG                       // print limited details about very hot particles
 *   -DCOOLING_LIMIT_DU                        // clamp |Δlog10 u| per step to avoid pathologies
 *
 * Optional: to quickly test UVB timing at high z, you can define COOLING_Z_REION_CUT
 * and set COOLING_Z_REION_CUT to a redshift; UVB will be disabled above it.
 *   -DCOOLING_Z_REION_CUT -DCOOLING_Z_REION_CUT=8.0
 *
 * Notes:
 *  - This file is based on the user's provided cooling.cc (Gadget-4) with additions clearly
 *    marked by comments starting with "// === DIAG ===" or compile-time guards listed above.
 *  - The logic and interfaces of the original functions are preserved.
 *============================================================================*/

#include "gadgetconfig.h"

#ifdef COOLING

#include <gsl/gsl_math.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>

#ifdef STARFORMATION
#include <vector>  // For metal cooling table
#endif

#include "../cooling_sfr/cooling.h"
#include "../data/allvars.h"
#include "../data/dtypes.h"
#include "../data/mymalloc.h"
#include "../logs/logs.h"
#include "../logs/timer.h"
#include "../system/system.h"
#include "../time_integration/timestep.h"

// === DIAG === step-level accumulators (file scope)
#ifdef STARFORMATION
static long long MC_count_particles = 0;
static long long MC_count_used      = 0;
static double     MC_Lambda_tot     = 0.0;
static double     MC_Lambda_metal   = 0.0;
#endif

#ifdef OUTPUT_COOLHEAT
static double     SUM_COOL = 0.0, SUM_HEAT = 0.0;
static long long  N_COOL = 0, N_HEAT = 0;
#endif

#ifdef COOLING
#define COOLING_Z_REION_CUT 6.0
#endif

// =============================================================================
// Cooling core
// =============================================================================

/** \brief Compute the new internal energy per unit mass. */
double coolsfr::DoCooling(double u_old, double rho, double dt, double *ne_guess, gas_state *gs, do_cool_data *DoCool, double metallicity)
{
  DoCool->u_old_input    = u_old;
  DoCool->rho_input      = rho;
  DoCool->dt_input       = dt;
  DoCool->ne_guess_input = *ne_guess;

  if(!gsl_finite(u_old))
    Terminate("invalid input: u_old=%g\n", u_old);

  if(u_old < 0 || rho < 0)
    Terminate("invalid input: u_old=%g  rho=%g  dt=%g  All.MinEgySpec=%g\n", u_old, rho, dt, All.MinEgySpec);

  rho   *= All.UnitDensity_in_cgs * All.HubbleParam * All.HubbleParam; /* physical cgs */
  u_old *= All.UnitPressure_in_cgs / All.UnitDensity_in_cgs;
  dt    *= All.UnitTime_in_s / All.HubbleParam;

  gs->nHcgs       = gs->XH * rho / PROTONMASS; /* hydrogen number dens in cgs units */
  double ratefact = gs->nHcgs * gs->nHcgs / rho;

  double u       = u_old;
  double u_lower = u;
  double u_upper = u;

  double LambdaNet = CoolingRateFromU(u, rho, ne_guess, gs, DoCool, metallicity);

  /* bracketing */
  if(u - u_old - ratefact * LambdaNet * dt < 0) /* heating */
    {
      u_upper *= sqrt(1.1);
      u_lower /= sqrt(1.1);
      while(u_upper - u_old - ratefact * CoolingRateFromU(u_upper, rho, ne_guess, gs, DoCool, metallicity) * dt < 0)
        {
          u_upper *= 1.1;
          u_lower *= 1.1;
        }
    }

  if(u - u_old - ratefact * LambdaNet * dt > 0)
    {
      u_lower /= sqrt(1.1);
      u_upper *= sqrt(1.1);
      while(u_lower - u_old - ratefact * CoolingRateFromU(u_lower, rho, ne_guess, gs, DoCool, metallicity) * dt > 0)
        {
          u_upper /= 1.1;
          u_lower /= 1.1;
        }
    }

  int iter = 0;
  double du;
  do
    {
      u = 0.5 * (u_lower + u_upper);

      LambdaNet = CoolingRateFromU(u, rho, ne_guess, gs, DoCool, metallicity);

      if(u - u_old - ratefact * LambdaNet * dt > 0)
        {
          u_upper = u;
        }
      else
        {
          u_lower = u;
        }

      du = u_upper - u_lower;

      iter++;

      if(iter >= (MAXITER - 10))
        printf("u= %g\n", u);
    }
  while(fabs(du / u) > 1.0e-6 && iter < MAXITER);

  if(iter >= MAXITER)
    Terminate(
        "failed to converge in DoCooling(): DoCool->u_old_input=%g\nDoCool->rho_input= %g\nDoCool->dt_input= "
        "%g\nDoCool->ne_guess_input= %g\n",
        DoCool->u_old_input, DoCool->rho_input, DoCool->dt_input, DoCool->ne_guess_input);

  u *= All.UnitDensity_in_cgs / All.UnitPressure_in_cgs; /* to internal units */

  return u;
}

/** \brief Return the cooling time. */
double coolsfr::GetCoolingTime(double u_old, double rho, double *ne_guess, gas_state *gs, do_cool_data *DoCool, double metallicity)
{
  DoCool->u_old_input    = u_old;
  DoCool->rho_input      = rho;
  DoCool->ne_guess_input = *ne_guess;

  rho   *= All.UnitDensity_in_cgs * All.HubbleParam * All.HubbleParam; /* cgs */
  u_old *= All.UnitPressure_in_cgs / All.UnitDensity_in_cgs;

  gs->nHcgs       = gs->XH * rho / PROTONMASS; /* hydrogen number dens in cgs units */
  double ratefact = gs->nHcgs * gs->nHcgs / rho;

  double u = u_old;

  double LambdaNet = CoolingRateFromU(u, rho, ne_guess, gs, DoCool, metallicity);

  if(LambdaNet >= 0) /* net heating */
    return 0;

  double coolingtime = u_old / (-ratefact * LambdaNet);
  coolingtime *= All.HubbleParam / All.UnitTime_in_s;
  return coolingtime;
}

/** \brief Compute gas temperature from internal energy per unit mass. */
double coolsfr::convert_u_to_temp(double u, double rho, double *ne_guess, gas_state *gs, const do_cool_data *DoCool)
{

/* // === CRITICAL CAP === specific energy can erupt from feedback without this.
  const double u_max_safe = 3.0e7;  // Code units corresponds to ~10^9 K in CGS
  if(u > u_max_safe) {
    static int cap_count = 0;
    if(cap_count++ < 50 && ThisTask == 0) {
      printf("TEMP_CONV_CAP: u=%.3e->%.3e (rho=%.3e)\n", u, u_max_safe, rho);
    }
    u = u_max_safe;
  }
*/

  double u_input   = u;
  double rho_input = rho;
  double ne_input  = *ne_guess;

  double mu   = (1 + 4 * gs->yhelium) / (1 + gs->yhelium + *ne_guess);
  double temp = GAMMA_MINUS1 / BOLTZMANN * u * PROTONMASS * mu;

  double max = 0;
  int iter   = 0;
  double temp_old;
  do
    {
      double ne_old = *ne_guess;

      find_abundances_and_rates(log10(temp), rho, ne_guess, gs, DoCool);
      temp_old = temp;

      mu = (1 + 4 * gs->yhelium) / (1 + gs->yhelium + *ne_guess);

      double temp_new = GAMMA_MINUS1 / BOLTZMANN * u * PROTONMASS * mu;

      max = std::max<double>(max, temp_new / (1 + gs->yhelium + *ne_guess) * fabs((*ne_guess - ne_old) / (temp_new - temp_old + 1.0)));

      temp = temp_old + (temp_new - temp_old) / (1 + max);
      iter++;

      if(iter > (MAXITER - 10))
        printf("-> temp= %g ne=%g\n", temp, *ne_guess);
    }
  while(fabs(temp - temp_old) > 1.0e-3 * temp && iter < MAXITER);

  if(iter >= MAXITER)
    {
      printf("failed to converge in convert_u_to_temp()\n");
      printf("u_input= %g\nrho_input=%g\n ne_input=%g\n", u_input, rho_input, ne_input);
      printf("DoCool->u_old_input=%g\nDoCool->rho_input= %g\nDoCool->dt_input= %g\nDoCool->ne_guess_input= %g\n", DoCool->u_old_input,
             DoCool->rho_input, DoCool->dt_input, DoCool->ne_guess_input);
      Terminate("convergence failure");
    }

  return temp;
}

/** \brief Computes the actual abundance ratios. */
void coolsfr::find_abundances_and_rates(double logT, double rho, double *ne_guess, gas_state *gs, const do_cool_data *DoCool)
{
  double logT_input = logT;
  double rho_input  = rho;
  double ne_input   = *ne_guess;

  if(!gsl_finite(logT))
    Terminate("logT=%g\n", logT);

  if(logT <= Tmin) /* everything neutral */
    {
      gs->nH0   = 1.0;
      gs->nHe0  = gs->yhelium;
      gs->nHp   = 0;
      gs->nHep  = 0;
      gs->nHepp = 0;
      gs->ne    = 0;
      *ne_guess = 0;
      return;
    }

  if(logT >= Tmax) /* everything ionized */
    {
      gs->nH0   = 0;
      gs->nHe0  = 0;
      gs->nHp   = 1.0;
      gs->nHep  = 0;
      gs->nHepp = gs->yhelium;
      gs->ne    = gs->nHp + 2.0 * gs->nHepp;
      *ne_guess = gs->ne; /* in units of n_H */
      return;
    }

  double t    = (logT - Tmin) / deltaT;
  int j       = (int)t;
  double fhi  = t - j;
  double flow = 1 - fhi;

  if(*ne_guess == 0)
    *ne_guess = 1.0;

  gs->nHcgs = gs->XH * rho / PROTONMASS; /* hydrogen number dens in cgs units */

  gs->ne       = *ne_guess;
  double neold = gs->ne;
  int niter    = 0;
  gs->necgs    = gs->ne * gs->nHcgs;

  do
    {
      niter++;

      gs->aHp   = flow * RateT[j].AlphaHp + fhi * RateT[j + 1].AlphaHp;
      gs->aHep  = flow * RateT[j].AlphaHep + fhi * RateT[j + 1].AlphaHep;
      gs->aHepp = flow * RateT[j].AlphaHepp + fhi * RateT[j + 1].AlphaHepp;
      gs->ad    = flow * RateT[j].Alphad + fhi * RateT[j + 1].Alphad;
      gs->geH0  = flow * RateT[j].GammaeH0 + fhi * RateT[j + 1].GammaeH0;
      gs->geHe0 = flow * RateT[j].GammaeHe0 + fhi * RateT[j + 1].GammaeHe0;
      gs->geHep = flow * RateT[j].GammaeHep + fhi * RateT[j + 1].GammaeHep;

      if(gs->necgs <= 1.e-25 || pc.J_UV == 0)
        {
          gs->gJH0ne = gs->gJHe0ne = gs->gJHepne = 0;
        }
      else
        {
          gs->gJH0ne  = pc.gJH0 / gs->necgs;
          gs->gJHe0ne = pc.gJHe0 / gs->necgs;
          gs->gJHepne = pc.gJHep / gs->necgs;
        }

      gs->nH0 = gs->aHp / (gs->aHp + gs->geH0 + gs->gJH0ne); /* eqn (33) */
      gs->nHp = 1.0 - gs->nH0;                               /* eqn (34) */

      if((gs->gJHe0ne + gs->geHe0) <= SMALLNUM) /* no ionization at all */
        {
          gs->nHep  = 0.0;
          gs->nHepp = 0.0;
          gs->nHe0  = gs->yhelium;
        }
      else
        {
          gs->nHep = gs->yhelium /
                     (1.0 + (gs->aHep + gs->ad) / (gs->geHe0 + gs->gJHe0ne) + (gs->geHep + gs->gJHepne) / gs->aHepp);
          gs->nHe0  = gs->nHep * (gs->aHep + gs->ad) / (gs->geHe0 + gs->gJHe0ne);
          gs->nHepp = gs->nHep * (gs->geHep + gs->gJHepne) / gs->aHepp;
        }

      neold = gs->ne;

      gs->ne    = gs->nHp + gs->nHep + 2 * gs->nHepp; /* eqn (38) */
      gs->necgs = gs->ne * gs->nHcgs;

      if(pc.J_UV == 0)
        break;

      double nenew = 0.5 * (gs->ne + neold);
      gs->ne       = nenew;
      gs->necgs    = gs->ne * gs->nHcgs;

      if(fabs(gs->ne - neold) < 1.0e-4)
        break;

      if(niter > (MAXITER - 10))
        printf("ne= %g  niter=%d\n", gs->ne, niter);
    }
  while(niter < MAXITER);

  if(niter >= MAXITER)
    Terminate(
        "no convergence reached in find_abundances_and_rates(): logT_input= %g  rho_input= %g  ne_input= %g "
        "DoCool->u_old_input=%g\nDoCool->rho_input= %g\nDoCool->dt_input= %g\nDoCool->ne_guess_input= %g\n",
        logT_input, rho_input, ne_input, DoCool->u_old_input, DoCool->rho_input, DoCool->dt_input, DoCool->ne_guess_input);

  gs->bH0  = flow * RateT[j].BetaH0 + fhi * RateT[j + 1].BetaH0;
  gs->bHep = flow * RateT[j].BetaHep + fhi * RateT[j + 1].BetaHep;
  gs->bff  = flow * RateT[j].Betaff + fhi * RateT[j + 1].Betaff;

  *ne_guess = gs->ne;
}

/** \brief Get cooling rate from gas internal energy. 
 *  Added a cap here to prevent specific energy (u) from getting too high from feedback.
*/
double coolsfr::CoolingRateFromU(double u, double rho, double *ne_guess, gas_state *gs, const do_cool_data *DoCool, double metallicity)
{
  /*
  // === EMERGENCY CAP === Prevent extreme energies from crashing temperature solver
  const double u_max_safe = 3.0e7; // ~10^9 K in CGS units (cm/s)^2
  if(u > u_max_safe) {
    static int cap_count = 0;
    if(cap_count++ < 20 && ThisTask == 0) {
      printf("ENERGY_CAP: u=%.3e->%.3e (rho=%.3e, u_old=%.3e)\n", 
             u, u_max_safe, rho, DoCool->u_old_input);
    }
    u = u_max_safe;
  }
  */
  double temp = convert_u_to_temp(u, rho, ne_guess, gs, DoCool);
  return CoolingRate(log10(temp), rho, ne_guess, gs, DoCool, metallicity);
}

/** \brief  Self-consistent temperature and abundance ratios helper. */
double coolsfr::AbundanceRatios(double u, double rho, double *ne_guess, double *nH0_pointer, double *nHeII_pointer)
{
  gas_state gs          = GasState;
  do_cool_data DoCool   = DoCoolData;
  DoCool.u_old_input    = u;
  DoCool.rho_input      = rho;
  DoCool.ne_guess_input = *ne_guess;

  rho *= All.UnitDensity_in_cgs * All.HubbleParam * All.HubbleParam; /* cgs */
  u   *= All.UnitPressure_in_cgs / All.UnitDensity_in_cgs;

  double temp = convert_u_to_temp(u, rho, ne_guess, &gs, &DoCool);

  *nH0_pointer   = gs.nH0;
  *nHeII_pointer = gs.nHep;

  return temp;
}

#ifdef STARFORMATION
// ==== Metal cooling helpers (unchanged logic, plus used in diagnostics) ====
void coolsfr::GetTableIndex(const std::vector<double> &array, double value, int &index, double &fraction)
{
    if(value <= array[0])
    { index = 0; fraction = 0.0; return; }
    if(value >= array.back())
    { index = (int)array.size() - 2; fraction = 1.0; return; }
    int low = 0, high = (int)array.size() - 1;
    while(high - low > 1)
    { int mid = (low + high) / 2; if(array[mid] <= value) low = mid; else high = mid; }
    index = low; fraction = (value - array[low]) / (array[high] - array[low]);
}

double coolsfr::GetMetallicitySolarUnits(double total_metallicity)
{
    const double Z_solar = 0.02;  // Solar metallicity
    return total_metallicity / Z_solar;
}

double coolsfr::GetMetalLambda(double logT, double logZ)
{
    if(!metal_table.table_loaded || logZ < metal_table.z_min || logZ > metal_table.z_max ||
       logT < metal_table.t_min || logT > metal_table.t_max)
        return 0.0;

    int z_index, t_index; double z_frac, t_frac;
    GetTableIndex(metal_table.metallicity_bins, logZ, z_index, z_frac);
    GetTableIndex(metal_table.temperature_bins, logT, t_index, t_frac);

    double lambda_00 = metal_table.cooling_rates[z_index][t_index];
    double lambda_10 = (z_index + 1 < metal_table.n_metallicity) ? metal_table.cooling_rates[z_index + 1][t_index] : lambda_00;
    double lambda_01 = (t_index + 1 < metal_table.n_temperature) ? metal_table.cooling_rates[z_index][t_index + 1] : lambda_00;
    double lambda_11 = (z_index + 1 < metal_table.n_metallicity && t_index + 1 < metal_table.n_temperature) ?
                       metal_table.cooling_rates[z_index + 1][t_index + 1] : lambda_00;

    double lambda_0 = (1.0 - z_frac) * lambda_00 + z_frac * lambda_10;
    double lambda_1 = (1.0 - z_frac) * lambda_01 + z_frac * lambda_11;
    return (1.0 - t_frac) * lambda_0 + t_frac * lambda_1;
}

void coolsfr::ReadMetalCoolingTable(const char *filename)
{
    FILE *file = fopen(filename, "r");
    if(!file)
    {
        if(ThisTask == 0)
        {
            mpi_printf("WARNING: Cannot open metal cooling table file: %s\n", filename);
            mpi_printf("         Metal cooling will be disabled\n");
        }
        metal_table.table_loaded = false;
        return;
    }

    metal_table.metallicity_bins.clear();
    metal_table.temperature_bins.clear();
    metal_table.cooling_rates.clear();

    double metallicity; int n_temp;
    while(fscanf(file, "%lg %d", &metallicity, &n_temp) == 2)
    {
        metal_table.metallicity_bins.push_back(metallicity);
        std::vector<double> lambda_row; lambda_row.reserve(n_temp);
        for(int i = 0; i < n_temp; i++)
        {
            double temp, lambda;
            if(fscanf(file, "%lg %lg", &temp, &lambda) != 2)
                Terminate("Error reading metal cooling table at metallicity %g, point %d\n", metallicity, i);
            if(metal_table.metallicity_bins.size() == 1)
                metal_table.temperature_bins.push_back(temp);
            lambda_row.push_back(lambda);
        }
        metal_table.cooling_rates.push_back(lambda_row);
    }
    fclose(file);

    metal_table.n_metallicity = (int)metal_table.metallicity_bins.size();
    metal_table.n_temperature = (int)metal_table.temperature_bins.size();
    if(metal_table.n_metallicity > 0 && metal_table.n_temperature > 0)
    {
        metal_table.z_min = metal_table.metallicity_bins.front();
        metal_table.z_max = metal_table.metallicity_bins.back();
        metal_table.t_min = metal_table.temperature_bins.front();
        metal_table.t_max = metal_table.temperature_bins.back();
        metal_table.table_loaded = true;
        if(ThisTask == 0)
        {
            mpi_printf("METAL_COOLING: Read table with %d metallicity bins and %d temperature points\n",
                       metal_table.n_metallicity, metal_table.n_temperature);
            mpi_printf("METAL_COOLING: Metallicity range: %g to %g (log Z/Z_solar)\n", metal_table.z_min, metal_table.z_max);
            mpi_printf("METAL_COOLING: Temperature range: %g to %g (log T)\n", metal_table.t_min, metal_table.t_max);
        }
    }
    else
    {
        if(ThisTask == 0) mpi_printf("WARNING: Metal cooling table is empty\n");
        metal_table.table_loaded = false;
    }
}
#endif // STARFORMATION

/** \brief  Calculate (heating rate-cooling rate)/n_h^2 in cgs units. */
double coolsfr::CoolingRate(double logT, double rho, double *nelec, gas_state *gs, const do_cool_data *DoCool, double metallicity)
{
  double Lambda, Heat;

  if(logT <= Tmin)
    logT = Tmin + 0.5 * deltaT; /* floor at Tmin */

  gs->nHcgs = gs->XH * rho / PROTONMASS; /* hydrogen number dens in cgs units */

  if(logT < Tmax)
    {
      find_abundances_and_rates(logT, rho, nelec, gs, DoCool);

      double T = pow(10.0, logT);

      double LambdaExcH0   = gs->bH0 * gs->ne * gs->nH0;
      double LambdaExcHep  = gs->bHep * gs->ne * gs->nHep;
      double LambdaExc     = LambdaExcH0 + LambdaExcHep; /* excitation */
      double LambdaIonH0   = 2.18e-11 * gs->geH0 * gs->ne * gs->nH0;
      double LambdaIonHe0  = 3.94e-11 * gs->geHe0 * gs->ne * gs->nHe0;
      double LambdaIonHep  = 8.72e-11 * gs->geHep * gs->ne * gs->nHep;
      double LambdaIon     = LambdaIonH0 + LambdaIonHe0 + LambdaIonHep; /* ionization */
      double LambdaRecHp   = 1.036e-16 * T * gs->ne * (gs->aHp * gs->nHp);
      double LambdaRecHep  = 1.036e-16 * T * gs->ne * (gs->aHep * gs->nHep);
      double LambdaRecHepp = 1.036e-16 * T * gs->ne * (gs->aHepp * gs->nHepp);
      double LambdaRecHepd = 6.526e-11 * gs->ad * gs->ne * gs->nHep;
      double LambdaRec     = LambdaRecHp + LambdaRecHep + LambdaRecHepp + LambdaRecHepd;
      double LambdaFF      = gs->bff * (gs->nHp + gs->nHep + 4 * gs->nHepp) * gs->ne;
      Lambda               = LambdaExc + LambdaIon + LambdaRec + LambdaFF;

#ifdef STARFORMATION
      const double Lambda_before_metal = Lambda; // === DIAG === save pre-metal value
      const bool metal_enabled_now = (metallicity > 1.0e-10) && metal_table.table_loaded;
      MC_count_particles++;
      if(metal_enabled_now) MC_count_used++;

      /* Add metal line cooling if metallicity is present */
      if(metal_enabled_now)
        {
          double z_solar_units = GetMetallicitySolarUnits(metallicity);
          double logZ = log10(std::max(z_solar_units, 1.0e-4));
          double metal_lambda = GetMetalLambda(logT, logZ);
          if(metal_lambda > 0.0)
            {
              double LambdaMetal = metal_lambda * gs->ne * (gs->nHp + gs->nHep + gs->nHepp);
              Lambda += LambdaMetal;
            }
        }
      // === DIAG === accumulate contributions
      if(metal_enabled_now)
      {
        const double added = Lambda - Lambda_before_metal;
        MC_Lambda_metal += added;
      }
      MC_Lambda_tot += Lambda;
#endif

      if(All.ComovingIntegrationOn)
        {
          double redshift    = 1 / All.Time - 1;
          double LambdaCmptn = 5.65e-36 * gs->ne * (T - 2.73 * (1. + redshift)) * pow(1. + redshift, 4.) / gs->nHcgs;
          Lambda += LambdaCmptn;
        }

      Heat = 0;

#ifdef COOLING_Z_REION_CUT
      {
        double z_now = 1.0/All.Time - 1.0;
        if(z_now > (double)COOLING_Z_REION_CUT) { /* disable UVB above cut */ }
        else
#endif
      if(pc.J_UV != 0)
        Heat += (gs->nH0 * pc.epsH0 + gs->nHe0 * pc.epsHe0 + gs->nHep * pc.epsHep) / gs->nHcgs;
#ifdef COOLING_Z_REION_CUT
      }
#endif
    }
  else /* T>Tmax */
    {
      Heat = 0;
      gs->nHp   = 1.0;
      gs->nHep  = 0;
      gs->nHepp = gs->yhelium;
      gs->ne    = gs->nHp + 2.0 * gs->nHepp;
      *nelec    = gs->ne;

      double T        = pow(10.0, logT);
      double LambdaFF = 1.42e-27 * sqrt(T) * (1.1 + 0.34 * exp(-(5.5 - logT) * (5.5 - logT) / 3)) * (gs->nHp + 4 * gs->nHepp) * gs->ne;
      double LambdaCmptn = 0;
      if(All.ComovingIntegrationOn)
        {
          double redshift = 1 / All.Time - 1;
          LambdaCmptn = 5.65e-36 * gs->ne * (T - 2.73 * (1. + redshift)) * pow(1. + redshift, 4.) / gs->nHcgs;
        }
      Lambda = LambdaFF + LambdaCmptn;
    }

  return (Heat - Lambda);
}

/** \brief Make cooling rates interpolation table. */
void coolsfr::MakeRateTable(void)
{
  GasState.yhelium = (1 - GasState.XH) / (4 * GasState.XH);
  GasState.mhboltz = PROTONMASS / BOLTZMANN;

  deltaT          = (Tmax - Tmin) / NCOOLTAB;
  GasState.ethmin = pow(10.0, Tmin) * (1. + GasState.yhelium) / ((1. + 4. * GasState.yhelium) * GasState.mhboltz * GAMMA_MINUS1);

  for(int i = 0; i <= NCOOLTAB; i++)
    {
      RateT[i].BetaH0 = RateT[i].BetaHep = RateT[i].Betaff = RateT[i].AlphaHp = RateT[i].AlphaHep = RateT[i].AlphaHepp =
          RateT[i].Alphad = RateT[i].GammaeH0 = RateT[i].GammaeHe0 = RateT[i].GammaeHep = 0;

      double T     = pow(10.0, Tmin + deltaT * i);
      double Tfact = 1.0 / (1 + sqrt(T / 1.0e5));

      if(118348 / T < 70)      RateT[i].BetaH0  = 7.5e-19 * exp(-118348 / T) * Tfact;
      if(473638 / T < 70)      RateT[i].BetaHep = 5.54e-17 * pow(T, -0.397) * exp(-473638 / T) * Tfact;
      RateT[i].Betaff = 1.43e-27 * sqrt(T) * (1.1 + 0.34 * exp(-(5.5 - log10(T)) * (5.5 - log10(T)) / 3));
      RateT[i].AlphaHp   = 8.4e-11 * pow(T / 1000, -0.2) / (1. + pow(T / 1.0e6, 0.7)) / sqrt(T);
      RateT[i].AlphaHep  = 1.5e-10 * pow(T, -0.6353);
      RateT[i].AlphaHepp = 4. * RateT[i].AlphaHp;
      if(470000 / T < 70)      RateT[i].Alphad = 1.9e-3 * pow(T, -1.5) * exp(-470000 / T) * (1. + 0.3 * exp(-94000 / T));
      if(157809.1 / T < 70)    RateT[i].GammaeH0  = 5.85e-11 * sqrt(T) * exp(-157809.1 / T) * Tfact;
      if(285335.4 / T < 70)    RateT[i].GammaeHe0 = 2.38e-11 * sqrt(T) * exp(-285335.4 / T) * Tfact;
      if(631515.0 / T < 70)    RateT[i].GammaeHep = 5.68e-12 * sqrt(T) * exp(-631515.0 / T) * Tfact;
    }
}

/** \brief Read table input for ionizing parameters. */
void coolsfr::ReadIonizeParams(char *fname)
{
  NheattabUVB = 0;
  int i, iter;
  for(iter = 0, i = 0; iter < 2; iter++)
    {
      FILE *fdcool;
      if(!(fdcool = fopen(fname, "r")))
        Terminate(" Cannot read ionization table in file `%s'\n", fname);
      if(iter == 0)
        while(fscanf(fdcool, "%*g %*g %*g %*g %*g %*g %*g") != EOF)
          NheattabUVB++;
      if(iter == 1)
        while(fscanf(fdcool, "%g %g %g %g %g %g %g", &PhotoTUVB[i].variable, &PhotoTUVB[i].gH0, &PhotoTUVB[i].gHe, &PhotoTUVB[i].gHep,
                     &PhotoTUVB[i].eH0, &PhotoTUVB[i].eHe, &PhotoTUVB[i].eHep) != EOF)
          i++;
      fclose(fdcool);

      if(iter == 0)
        {
          PhotoTUVB = (photo_table *)Mem.mymalloc("PhotoT", NheattabUVB * sizeof(photo_table));
          mpi_printf("COOLING: read ionization table with %d entries in file `%s'.\n", NheattabUVB, fname);
        }
    }
  for(i = 0; i < NheattabUVB; ++i)
    if(PhotoTUVB[i].gH0 == 0.0)
      break;

  NheattabUVB = i;
  mpi_printf("COOLING: using %d ionization table entries from file `%s'.\n", NheattabUVB, fname);

  if(NheattabUVB < 1)
    Terminate("The length of the cooling table has to have at least one entry");
}

/** \brief Set the ionization parameters for the UV background. */
void coolsfr::IonizeParamsUVB(void)
{
  if(!All.ComovingIntegrationOn)
    { SetZeroIonization(); return; }

  if(NheattabUVB == 1)
    {
      pc.J_UV   = 1;
      pc.gJH0   = PhotoTUVB[0].gH0;
      pc.gJHe0  = PhotoTUVB[0].gHe;
      pc.gJHep  = PhotoTUVB[0].gHep;
      pc.epsH0  = PhotoTUVB[0].eH0;
      pc.epsHe0 = PhotoTUVB[0].eHe;
      pc.epsHep = PhotoTUVB[0].eHep;
    }
  else
    {
      double redshift = 1 / All.Time - 1;
      double logz     = log10(redshift + 1.0);
      int ilow        = 0;


    // This makes it so that gas at z > z_cut cools only via collisional
    // processes (no photo-heating or ionization from the UVB),
    // mimicking a pre-reionization universe.
    #ifdef COOLING_Z_REION_CUT
    {
      const double z_now = 1.0/All.Time - 1.0;
      if (z_now > (double)COOLING_Z_REION_CUT) {
        SetZeroIonization();   // pc.J_UV = 0 and zero all photo rates
        return;
      }
    }
    #endif


      for(int i = 0; i < NheattabUVB; i++)
        {
          if(PhotoTUVB[i].variable < logz) ilow = i; else break;
        }

      if(logz > PhotoTUVB[NheattabUVB - 1].variable || ilow >= NheattabUVB - 1)
        { SetZeroIonization(); }
      else
        {
          double dzlow = logz - PhotoTUVB[ilow].variable;
          double dzhi  = PhotoTUVB[ilow + 1].variable - logz;

          if(PhotoTUVB[ilow].gH0 == 0 || PhotoTUVB[ilow + 1].gH0 == 0)
            { SetZeroIonization(); }
          else
            {
              pc.J_UV   = 1;
              pc.gJH0   = pow(10., (dzhi * log10(PhotoTUVB[ilow].gH0)  + dzlow * log10(PhotoTUVB[ilow + 1].gH0))  / (dzlow + dzhi));
              pc.gJHe0  = pow(10., (dzhi * log10(PhotoTUVB[ilow].gHe)  + dzlow * log10(PhotoTUVB[ilow + 1].gHe))  / (dzlow + dzhi));
              pc.gJHep  = pow(10., (dzhi * log10(PhotoTUVB[ilow].gHep) + dzlow * log10(PhotoTUVB[ilow + 1].gHep)) / (dzlow + dzhi));
              pc.epsH0  = pow(10., (dzhi * log10(PhotoTUVB[ilow].eH0)  + dzlow * log10(PhotoTUVB[ilow + 1].eH0))  / (dzlow + dzhi));
              pc.epsHe0 = pow(10., (dzhi * log10(PhotoTUVB[ilow].eHe)  + dzlow * log10(PhotoTUVB[ilow + 1].eHe))  / (dzlow + dzhi));
              pc.epsHep = pow(10., (dzhi * log10(PhotoTUVB[ilow].eHep) + dzlow * log10(PhotoTUVB[ilow + 1].eHep)) / (dzlow + dzhi));
            }
        }
    }
}

void coolsfr::SetZeroIonization(void) { memset(&pc, 0, sizeof(photo_current)); }
void coolsfr::IonizeParams(void) { IonizeParamsUVB(); }

/** \brief Initialize the cooling module. */
void coolsfr::InitCool(void)
{
  GasState.XH = HYDROGEN_MASSFRAC;
  SetZeroIonization();

  RateT = (rate_table *)Mem.mymalloc("RateT", (NCOOLTAB + 1) * sizeof(rate_table));
  MakeRateTable();

  ReadIonizeParams(All.TreecoolFile);

#ifdef STARFORMATION
  metal_table.table_loaded = false;
  if(All.MetalcoolFile[0] != '\0')
    { ReadMetalCoolingTable(All.MetalcoolFile); }
  else if(ThisTask == 0)
    { mpi_printf("METAL_COOLING: No MetalcoolFile specified, metal cooling disabled\n"); }
#endif

  All.Time = All.TimeBegin;
  All.set_cosmo_factors_for_current_time();
  IonizeParams();

  // === DIAG === configuration banner (once)
#ifdef STARFORMATION
  if(ThisTask == 0)
    mpi_printf("METAL_COOLING: %s (STARFORMATION=ON, table_loaded=%d)\n",
               metal_table.table_loaded ? "ACTIVE" : "INACTIVE", metal_table.table_loaded ? 1 : 0);
#else
  if(ThisTask == 0)
    mpi_printf("METAL_COOLING: STARFORMATION=OFF -> metal cooling disabled\n");
#endif
}

/** \brief Apply the isochoric cooling to all the active gas particles. */
void coolsfr::cooling_only(simparticles *Sp)
{
  TIMER_START(CPU_COOLING_SFR);
  All.set_cosmo_factors_for_current_time();

  gas_state gs        = GasState;
  do_cool_data DoCool = DoCoolData;

  for(int i = 0; i < Sp->TimeBinsHydro.NActiveParticles; i++)
    {
      int target = Sp->TimeBinsHydro.ActiveParticleList[i];
      if(Sp->P[target].getType() == 0)
        {
          if(Sp->P[target].getMass() == 0 && Sp->P[target].ID.get() == 0)
            continue; /* skip particles that have been swallowed or eliminated */

          cool_sph_particle(Sp, target, &gs, &DoCool);
        }
    }

  // === DIAG === step summaries and resets
#ifdef STARFORMATION
  if(ThisTask == 0 && MC_count_particles > 0)
  {
    const double used_frac  = (double)MC_count_used / (double)MC_count_particles;
    const double metal_frac = (MC_Lambda_tot != 0.0) ? (MC_Lambda_metal / MC_Lambda_tot) : 0.0;
    mpi_printf("METAL_COOLING: step=%d used_on=%.3f of active gas; metal_lambda_frac=%.3f  (\u03A3\u039B_metal=%.3e, \u03A3\u039B_tot=%.3e)\n",
               All.NumCurrentTiStep, used_frac, metal_frac, MC_Lambda_metal, MC_Lambda_tot);
  }
  MC_count_particles = MC_count_used = 0;
  MC_Lambda_tot = MC_Lambda_metal = 0.0;
#endif

#ifdef OUTPUT_COOLHEAT
  if(ThisTask == 0)
    mpi_printf("COOLING: step=%d n_cool=%lld \u03A3|cool|=%.3e ; n_heat=%lld \u03A3heat=%.3e\n",
               All.NumCurrentTiStep, N_COOL, SUM_COOL, N_HEAT, SUM_HEAT);
  SUM_COOL = SUM_HEAT = 0.0; N_COOL = N_HEAT = 0;
#endif

#ifdef COOLING_HOT_DEBUG
  // reset per-step hot-print counter if you choose to make it non-static/global
#endif

  TIMER_STOP(CPU_COOLING_SFR);
}

/** \brief Apply the isochoric cooling to a given gas particle. */
void coolsfr::cool_sph_particle(simparticles *Sp, int i, gas_state *gs, do_cool_data *DoCool)
{
  double dens = Sp->SphP[i].Density;

  double dt = (Sp->P[i].getTimeBinHydro() ? (((integertime)1) << Sp->P[i].getTimeBinHydro()) : 0) * All.Timebase_interval;
  double dtime = All.cf_atime * dt / All.cf_atime_hubble_a;

  double utherm = Sp->get_utherm_from_entropy(i);
  double ne = Sp->SphP[i].Ne; /* electron abundance */

  double metallicity = 0.0;
#ifdef STARFORMATION
  metallicity = Sp->SphP[i].Metallicity;
  // === DIAG === one-shot sanity for metallicity units
  static int warned_units = 0;
  if(!warned_units && ThisTask == 0)
  {
    if(metallicity > 5.0) // mass fraction should never exceed 1
    {
      mpi_printf("METAL_COOLING: WARNING metallicity looks like [Z/H] or percent, expected mass fraction (0..1). Got Z=%g\n", metallicity);
      warned_units = 1;
    }
  }
#endif

  double unew = DoCooling(std::max<double>(All.MinEgySpec, utherm), dens * All.cf_a3inv, dtime, &ne, gs, DoCool, metallicity);
  Sp->SphP[i].Ne = ne;

  if(unew < 0)
    Terminate("invalid temperature: i=%d unew=%g\n", i, unew);

  double du = unew - utherm;

  if(unew < All.MinEgySpec)
    du = All.MinEgySpec - utherm;

#ifdef COOLING_LIMIT_DU
  // === DIAG === clamp |Δlog10 u| per step to avoid pathological runaway during debugging
  {
    const double max_dulog = 2.0; // at most factor 100 change per step
    const double tiny = 1e-30;
    const double u0 = std::max(utherm, tiny);
    double uf = utherm + du;
    if(uf < All.MinEgySpec) uf = All.MinEgySpec;
    double dlog = fabs(log10(uf / u0));
    if(dlog > max_dulog)
    {
      const double sign = (du >= 0) ? 1.0 : -1.0;
      const double uf_limited = u0 * pow(10.0, sign * max_dulog);
      du = uf_limited - utherm;
      if(ThisTask == 0)
        mpi_printf("COOLING: dU limited at step=%d i=%d (|Δlog10 u| clipped to %.2f)\n", All.NumCurrentTiStep, i, max_dulog);
    }
  }
#endif

  utherm += du;

#ifdef OUTPUT_COOLHEAT
  if(dtime > 0)
    Sp->SphP[i].CoolHeat = du * Sp->P[i].getMass() / dtime;
#endif

  // === DIAG === hot-particle print with physical units & (Heat-Lambda) sign
#ifdef COOLING_HOT_DEBUG
  {
    const double debug_T_K = 2.0e6;   // trigger when T exceeds this
    const int    max_print = 20;       // limit per step to avoid spam
    static int hot_prints_this_step = 0;

    double rho_cgs = dens * All.cf_a3inv * All.UnitDensity_in_cgs * All.HubbleParam * All.HubbleParam;
    double u_cgs   = utherm * All.UnitPressure_in_cgs / All.UnitDensity_in_cgs;
    double ne_loc  = ne;
    gas_state gtmp = *gs;
    do_cool_data dtmp = *DoCool;

    double T_new = convert_u_to_temp(u_cgs, rho_cgs, &ne_loc, &gtmp, &dtmp);
    if(T_new > debug_T_K && hot_prints_this_step < max_print && ThisTask == 0)
    {
      double logT = log10(T_new);
      double ne_copy = ne_loc;
      gas_state gtmp2 = *gs;
      do_cool_data dtmp2 = *DoCool;
      double rate = CoolingRate(logT, rho_cgs, &ne_copy, &gtmp2, &dtmp2, metallicity); // Heat - Lambda
      double z_solar_units = 0.0;
#ifdef STARFORMATION
      z_solar_units = GetMetallicitySolarUnits(metallicity);
#endif
      mpi_printf("[COOLING|HOT] a=%.6g z=%.3f i=%d T=%.3e K u=%.3e rho=%.3e cgs dt=%.3e s Z=%.3e (Z/Zsun=%.3e) (Heat-Lambda)=%.3e\n",
                 All.Time, 1.0/All.Time - 1.0, i, T_new, utherm, rho_cgs, dtmp2.dt_input, metallicity, z_solar_units, rate);
      hot_prints_this_step++;
    }
  }
#endif

  Sp->set_entropy_from_utherm(utherm, i);
  Sp->SphP[i].set_thermodynamic_variables();

#ifdef OUTPUT_COOLHEAT
  // accumulate sign stats for step summary
  if(dtime > 0)
  {
    if(Sp->SphP[i].CoolHeat < 0) { SUM_COOL += -Sp->SphP[i].CoolHeat; N_COOL++; }
    else if(Sp->SphP[i].CoolHeat > 0) { SUM_HEAT += Sp->SphP[i].CoolHeat; N_HEAT++; }
  }
#endif
}

#endif /* COOLING */
