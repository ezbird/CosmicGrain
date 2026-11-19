/*! \file hydro_watch.h
 *
 *  \brief Watchdog for runaway SPH forces (2048^3 halo441 debug, Nov 2026).
 *
 *  Job 389 lost the run when one gas particle went from normal to
 *  T=1e12 K, v=7e5 km/s in a single kick1+hydro1 phase with no feedback
 *  acting.  This flags any particle whose hydro force would change its
 *  physical velocity by more than CG_HYDRO_WATCH_KMS (default 3000 km/s) in
 *  one half-step kick, or its entropy by more than a factor
 *  CG_HYDRO_WATCH_ENTR (default 10), and prints the particle and its
 *  strongest neighbour pairs.  Prints are capped per rank.
 */

#ifndef HYDRO_WATCH_H
#define HYDRO_WATCH_H

#include <cstdlib>

namespace cg_hydro_watch
{
static const int MAX_EVENTS_PER_RANK      = 40;
static const int MAX_KICK_EVENTS_PER_RANK = 200;
static const int MAX_NGB_LINES            = 12;

inline double threshold_kms(void)
{
  static double t = -1;
  if(t < 0)
    {
      const char *s = getenv("CG_HYDRO_WATCH_KMS");
      t             = (s && atof(s) > 0) ? atof(s) : 3000.0;
    }
  return t;
}

inline double threshold_entropy_factor(void)
{
  static double f = -1;
  if(f < 0)
    {
      const char *s = getenv("CG_HYDRO_WATCH_ENTR");
      f             = (s && atof(s) > 0) ? atof(s) : 10.0;
    }
  return f;
}
}  // namespace cg_hydro_watch

#endif
