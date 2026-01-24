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
  float GrainRadius;        /*!< grain radius in cm (e.g., 1e-5 = 0.1 micron) */
  float CarbonFraction;     /*!< fraction of mass in carbonaceous grains (0-1) */
  float DustTemperature;    /*!< dust temperature in K */
  int GrainType;  /*!< grain composition: 0=silicate, 1=carbon, 2=mixed */
  
  // Helper functions
  inline double get_grain_radius_microns(void) { return GrainRadius * 1e4; }  // Convert cm to microns
  inline bool is_silicate(void) { return GrainType == 0; }
  inline bool is_carbonaceous(void) { return GrainType == 1; }
};

#endif

#endif