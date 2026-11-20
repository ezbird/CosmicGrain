/*! \file timestep_limiter.h
 *
 *  \brief SPH timestep limiter with wake-up (Saitoh & Makino 2009; Durier & Dalla Vecchia 2012).
 *
 *  Gadget-4 lets an SPH particle keep a long hydro step while its neighbours
 *  are on much shorter ones.  When feedback launches a shock, gas that chose a
 *  long step before the shock arrived stays inactive while it is overrun: in
 *  the 2048^3 halo441 runs such particles woke 100-6600x denser than their
 *  drift prediction and received single kicks of 1e4-1e5 km/s that destroyed
 *  the run (Nov 2026, jobs 389/391/413).
 *
 *  Two rules are enforced:
 *   1. Neighbour rule.  At each synchronisation point, after the new hydro
 *      steps are assigned, every active particle i asks each SPH neighbour j
 *      (found in the first-half hydro force walk) to satisfy
 *      bin_j <= bin_i + delta (delta = 2 by default, i.e. dt_j <= 4 dt_i).
 *      An active j is simply moved to the shorter bin before its first half
 *      kick.  An inactive j is woken: its step is cut to end at the next
 *      multiple of 2^bin_new after the current time, and the part of its
 *      first half kick (velocity and entropy) that lies beyond the new
 *      midpoint is removed, so that the second half kick at the new end
 *      completes an exact KDK step on the shortened interval.
 *   2. Source rule.  Gas heated or kicked by stellar feedback gets its signal
 *      velocity raised to the new sound speed (plus the kick), and is woken
 *      to the step that implies if it is inactive.
 *
 *  Cooling and star formation integrate over 2^bin; a woken particle's step
 *  is longer than that, so the difference is kept in LimiterExtraTi and added
 *  to their dt.
 *
 *  Environment: CG_TIMESTEP_LIMITER=0 disables both rules (default on);
 *  CG_LIMITER_DELTA sets delta (default 2).
 */

#ifndef TIMESTEP_LIMITER_H
#define TIMESTEP_LIMITER_H

#include <mpi.h>
#include <cstdlib>

namespace cg_limiter
{
struct request
{
  int task;
  int index;
  int bin;
};

struct counters
{
  long long requests_sent;
  long long moved_active;
  long long woke_inactive;
  long long wake_skipped;
  long long source_bumped;
  long long source_woke;
};

extern counters Count;

}  // namespace cg_limiter

class simparticles;

namespace cg_limiter
{
/* Shorten the hydro step of gas particle j to bin_new.  start_of_step: j's new step begins at the current time (first
 * half of a sync point, before its kick); otherwise j must be inactive and is woken with a kick correction.  Returns 1
 * if j's step was shortened. */
int wake(simparticles *Sp, int j, int bin_new, bool start_of_step);

/* Source rule, called right after stellar feedback with the gas state saved just before it. */
void after_sources(simparticles *Sp, int nsaved, const long long *id, const float *entropy_pred, const float *velpred);

/* Print global counters (collective). */
void report(simparticles *Sp, MPI_Comm comm);

inline bool enabled(void)
{
  static int e = -1;
  if(e < 0)
    {
      const char *s = getenv("CG_TIMESTEP_LIMITER");
      e             = (s && atoi(s) == 0) ? 0 : 1;
    }
  return e == 1;
}

inline int delta(void)
{
  static int d = -1;
  if(d < 0)
    {
      const char *s = getenv("CG_LIMITER_DELTA");
      d             = (s && atoi(s) > 0) ? atoi(s) : 2;
    }
  return d;
}
}  // namespace cg_limiter

#endif
