/*******************************************************************************
 * \file dust_particle_data.h
 *
 * \brief defines the structure holding the extra data for dust particles
 *******************************************************************************/

#ifndef DUSTPARTDATA_H
#define DUSTPARTDATA_H

#include "gadgetconfig.h"

#include "../data/dtypes.h"

#ifdef DUST

/** Holds data that is stored for each dust particle in addition to
    the collisionless variables in particle_data.
 */
struct dust_data
{
  float GrainRadius;        /*!< grain radius in nm (e.g., 100 = 0.1 micron) */
  float CarbonMassFraction; /*!< fraction of mass in carbonaceous grains (0-1) */
  float DustTemperature;    /*!< dust temperature in K */
  float BirthPos[3];        /*!< star birth position in comoving kpc/h */
  int DustSource;           /*!< dust source: 0=SNII, 1=AGB, 2=LRN */

  /* Fields below are NOT part of the legacy restart payload.  restart.cc
   * writes DustP[] truncated at LocalDTG (the original layout) and stores
   * these in an appended block, so pre-cache restart files remain readable.
   * New fields must be added after LocalDTG and to that appended block. */
  float LocalDTG; /*!< cached neighborhood dust-to-gas ratio; <= 0 means no valid cache */
  int DTGAge;     /*!< dust calls since LocalDTG was last recomputed */
};

#endif

#endif