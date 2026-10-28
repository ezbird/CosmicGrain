#ifndef AGB_FEEDBACK_MATH_H
#define AGB_FEEDBACK_MATH_H
#include "gadgetconfig.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace agb_feedback {
// Pure calculations shared by the production path and standalone tests.
inline double increment(double now, double before)
{
  const double tol = 1e-12 * std::max(1e-30, std::max(std::abs(now), std::abs(before)));
  if(!std::isfinite(now) || !std::isfinite(before) || now < -tol ||
     before < -tol || now + tol < before)
    throw std::runtime_error("Invalid/nonmonotonic cumulative AGB budget");
  return std::max(0.0, now - before);
}

struct MixedGas {
  double mass, u;
  double velocity[3], predicted_velocity[3];
};

inline MixedGas mix_gas(double m, double dm, double u, double u_ejecta,
                        const double v[3], const double vp[3], const double ve[3],
                        double velocity_to_physical_squared)
{
  if(!std::isfinite(m) || m <= 0 || !std::isfinite(dm) || dm < 0 ||
     !std::isfinite(u) || u <= 0 || !std::isfinite(u_ejecta) || u_ejecta < 0 ||
     !std::isfinite(velocity_to_physical_squared) || velocity_to_physical_squared <= 0)
    throw std::runtime_error("Invalid AGB gas mixing state");
  MixedGas out;
  out.mass = m + dm;
  double rel2 = 0;
  for(int k = 0; k < 3; ++k) {
    if(!std::isfinite(v[k]) || !std::isfinite(vp[k]) || !std::isfinite(ve[k]))
      throw std::runtime_error("Nonfinite AGB mixing velocity");
    rel2 += (vp[k]-ve[k])*(vp[k]-ve[k]);
    out.predicted_velocity[k] = (m*vp[k] + dm*ve[k]) / out.mass;
    // A current-time impulse changes both time levels by the same amount.
    // Preserve the existing stored-minus-predicted kick offset.
    out.velocity[k] = v[k] + (out.predicted_velocity[k] - vp[k]);
  }
  // Inelastic mixing conserves kinetic + thermal energy, adding only the
  // explicitly specified ejecta internal energy, using current predicted gas
  // velocity and the supplied donor velocity. Canonical velocity is a*v.
  // Caller must establish a common donor time separately.
  const double dissipated = 0.5*m*dm/out.mass*rel2*velocity_to_physical_squared;
  out.u = (m*u + dm*u_ejecta + dissipated) / out.mass;
  if(!std::isfinite(out.mass) || !std::isfinite(out.u) || out.u <= 0)
    throw std::runtime_error("Nonfinite AGB mixed gas state");
  return out;
}

struct DustRelease { int packets; double mass; };
inline DustRelease dust_release(double pending, double packet_mass,
                                int remaining, bool complete)
{
  if(!std::isfinite(pending) || pending < 0 ||
     !std::isfinite(packet_mass) || packet_mass < 0 || remaining < 0)
    throw std::runtime_error("Invalid AGB dust packet state");
  if(pending == 0 || remaining == 0) return {0, 0};
  // Reserve the final representative for the terminal remainder. Earlier
  // packets have fixed mass, so the count never depends on timestep count.
  if(complete) return {remaining, pending};
  if(packet_mass <= 0 || remaining == 1) return {0, 0};
  int n = (int)std::min((double)(remaining-1), std::floor(pending/packet_mass));
  return {n, n*packet_mass};
}
}
#endif
