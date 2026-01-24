/*******************************************************************************
 * \copyright   This file is part of the GADGET4 N-body/SPH code developed
 * \copyright   by Volker Springel. Copyright (C) 2014-2020 by Volker Springel
 * \copyright   (vspringel@mpa-garching.mpg.de) and all contributing authors.
 *******************************************************************************/

/*! \file  feedback.h
 *
 *  \brief API for stellar feedback (Type II SNe and AGB winds)
 */

#ifndef FEEDBACK_H
#define FEEDBACK_H

#include "../data/simparticles.h"
#include "../ngbtree/ngbtree.h"
#include "../domain/domain.h"
#include <mpi.h>

// Feedback constants
#define MIN_TYPEII_TIME 3.0           // Minimum Type II SN age (Myr)
#define MAX_TYPEII_TIME 40.0          // Maximum Type II SN age (Myr)
#define MIN_AGB_TIME 100.0            // Minimum AGB phase age (Myr); high mass AGBs do kick on by 50 Myr, but there are so few...
#define MAX_AGB_TIME 1000.0           // Maximum AGB phase age (Myr)
#define SOLAR_MASS 1.989e33           // Solar mass in grams

/* Initialize feedback system (called once at startup) */
void init_stellar_feedback(void);

/* Apply stellar feedback for current timestep 
 * 
 * This handles:
 *   - Aging stars and triggering SNII/AGB events
 *   - Energy reservoir management and release
 *   - Stochastic heating of neighboring gas
 */
void apply_stellar_feedback(double current_time, simparticles *Sp, 
                           ngbtree *Tree, domain<simparticles> *D);

/* Print feedback statistics (called at each output) */
void stellar_feedback_statistics(MPI_Comm comm);

/* RNG for stochastic heating (seeded by particle ID + timestep) */
double get_random_number(unsigned long long id);

/* Get stellar age in Myr from birth scale factor */
double get_stellar_age_Myr(double stellar_age_field, double unused);

struct FeedbackDiagLocal {
  long long n_SNII = 0;
  long long n_AGB  = 0;
  double E_SN_erg  = 0.0;
  double E_AGB_erg = 0.0;
  double E_deposited_erg = 0.0;
  double E_to_reservoir_erg = 0.0;
  double E_from_reservoir_erg = 0.0;
  double max_abs_dulog = 0.0;

  void reset() {
    n_SNII = n_AGB = 0;
    E_SN_erg = E_AGB_erg = 0.0;
    E_deposited_erg = E_to_reservoir_erg = E_from_reservoir_erg = 0.0;
    max_abs_dulog = 0.0;
  }
};

// Local counters, always compiled with FEEDBACK, independent of DIAG flag
extern FeedbackDiagLocal FbDiag;

// Call once per step by all ranks; does reductions only when cadence matches
void feedback_diag_try_flush(MPI_Comm comm, int cadence);

#endif