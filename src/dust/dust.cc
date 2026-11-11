/*! \file dust.cc
 *  \brief Implementation of on-the-fly dust evolution model for CosmicGrain.
 *
 *  This module creates dust superparticles from stellar feedback and evolves
 *  them through gas drag, grain growth, thermal erosion, shock destruction,
 *  coagulation, shattering, radiation pressure, and dust temperature updates.
 *
 *  For more on timebins and cadence, see spawn_dust_particle(), update_dust_dynamics() and
 *  get_timestep_grav() in timestep.cc.
 *  ─────────────────────────────────────────────────────────────────────────────
 */
#include "gadgetconfig.h"
#ifdef DUST
#include <gsl/gsl_math.h>
#include <math.h>
#include <mpi.h>
#include <numeric>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "../dust/dust.h"
#include "../cooling_sfr/feedback.h"
#include "../cooling_sfr/spatial_hash_zoom.h"
#include "../data/allvars.h"
#include "../data/dtypes.h"
#include "../data/intposconvert.h"
#include "../data/mymalloc.h"
#include "../logs/logs.h"
#include "../logs/timer.h"
#include "../system/system.h"
#include "../cooling_sfr/cooling.h"

// ── Mass / size thresholds ────────────────────────────────────────────────────
// Minimum mass to bother creating a dust particle. Set well below any
// physically meaningful grain mass to allow tiny initial particles, but
// large enough to prevent creating enormous numbers from trace metal ejecta.
#define MIN_DUST_PARTICLE_MASS  1e-10

// remember that dust particles represent ensembles of grains rather than
// individual grains. If the represented dust mass falls below this value,
// the particle is considered negligible and is removed.
// This threshold should be many orders of magnitude below the typical
// simulation dust particle mass just to be safe.
#define DUST_MASS_TO_DESTROY    1e-30

// ── Miscellaneous physics constants ──────────────────────────────────────────
// Minimum SFR to consider a gas cell actively star-forming for clumping / f_mol.
#define DUST_SFR_EPS            1e-14

// Not used but informational: single-grain mass for a silicate grain at 1nm
// m = (4/3)π(0.5e-7 cm)³ × 2.4 g/cm³ / 1.989e33 g/M☉
#define DUST_SINGLE_GRAIN_MASS_MSUN  6.32e-55   // M☉

// ── Diagnostic print macro ────────────────────────────────────────────────────
// Prepends task, scale factor, and redshift to every DUST_PRINT line.
// Controlled by All.DustDebugLevel so production runs stay quiet.
#define DUST_PRINT(...) do{ if(All.DustDebugLevel){ \
  printf("[DUST|T=%d|a=%.6g z=%.3f] ", All.ThisTask, (double)All.Time, 1.0/All.Time-1.0); \
  printf(__VA_ARGS__); } }while(0)
extern double get_random_number(void);

// ── Forward declarations ──────────────────────────────────────────────────────
void dust_grain_growth_subgrid(simparticles *Sp, int dust_idx, int gas_idx, double dt,
                               double T_gas);
static int destroy_dust_particle_to_gas(simparticles *Sp, int dust_idx,
                                         int nearest_gas, long long *counter,
                                         double *mass_counter);
double dust_clumping_factor(double n_H, int is_star_forming);

static double sample_initial_grain_radius(double median_nm, double sigma_dex)
{
  if(!(sigma_dex > 0.0))
    return std::max(All.DustMinGrainSize,
                    std::min(All.DustMaxGrainSize, median_nm));

  // Box-Muller draw: log10(radius) is normal with the configured median and
  // standard deviation in dex.
  const double u1 = std::max(get_random_number(), 1.0e-12);
  const double u2 = get_random_number();
  const double standard_normal =
      std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
  const double sampled_nm =
      median_nm * std::pow(10.0, sigma_dex * standard_normal);

  if(!std::isfinite(sampled_nm))
    return All.DustMaxGrainSize;
  return std::max(All.DustMinGrainSize,
                  std::min(All.DustMaxGrainSize, sampled_nm));
}

/** Seconds per internal time unit.
 *
 * Lengths are in h^-1 kpc and velocities in km/s, so one code time unit is
 * UnitTime_in_s / h seconds (cooling.cc converts the same way).  Every
 * conversion between physical and code time in the dust module goes through
 * this helper; using UnitTime_in_s alone made all dust timescales 1/h too
 * long.
 */
static inline double dust_code_time_in_s(void)
{
  return All.UnitTime_in_s / All.HubbleParam;
}

/** Return the proper gas mass density in cgs units.
 *
 * Snapshot/SPH densities are comoving code densities.  The a^-3 and h^2
 * factors match the conversion used by cooling.cc, ensuring that dust drag,
 * sputtering, collisions, growth, and temperature calculations all use the
 * same physical density as cooling and star formation.
 */
static inline double dust_gas_density_cgs(const simparticles *Sp, int gas_idx)
{
  return Sp->SphP[gas_idx].Density * All.cf_a3inv
         * All.UnitDensity_in_cgs * All.HubbleParam * All.HubbleParam;
}

/** Return the physical hydrogen-nucleus number density in cm^-3. */
static inline double dust_gas_nH_cgs(const simparticles *Sp, int gas_idx)
{
  return HYDROGEN_MASSFRAC * dust_gas_density_cgs(Sp, gas_idx) / PROTONMASS;
}

// ── External hash and rebuild declarations (defined in feedback.cc) ───────────
extern spatial_hash_zoom gas_hash;
extern spatial_hash_zoom star_hash;
extern spatial_hash_zoom dust_hash;
extern void rebuild_feedback_spatial_hash(simparticles *Sp, double dust_search_radius, MPI_Comm comm);

struct dust_dg_query
{
  int dust_index;
  MyIntPosType center[3];
  double radius;
};

struct dust_dg_result
{
  int dust_index;
  double gas_mass;
  double dust_mass;
};

static void extend_dust_dg_bounds(simparticles *Sp, int particle_idx,
                                  double bounds[7])
{
  double pos[3];
  Sp->intpos_to_pos(Sp->P[particle_idx].IntPos, pos);
  for(int d = 0; d < 3; d++)
    {
      bounds[d] = std::min(bounds[d], pos[d]);
      bounds[d + 3] = std::max(bounds[d + 3], pos[d]);
    }
  bounds[6] = 1.0;
}

static bool dust_dg_sphere_intersects_bounds(const double center[3], double radius,
                                             const double bounds[7])
{
  if(bounds[6] == 0.0)
    return false;

  int min_shift = 0, max_shift = 0;
#ifdef PERIODIC
  min_shift = -1;
  max_shift = 1;
#endif
  const double radius2 = radius * radius;
  for(int sx = min_shift; sx <= max_shift; sx++)
    for(int sy = min_shift; sy <= max_shift; sy++)
      for(int sz = min_shift; sz <= max_shift; sz++)
        {
          double image[3] = {center[0], center[1], center[2]};
#ifdef PERIODIC
          image[0] += sx * All.BoxSize;
          image[1] += sy * All.BoxSize;
          image[2] += sz * All.BoxSize;
#endif
          double distance2 = 0.0;
          for(int d = 0; d < 3; d++)
            {
              const double nearest =
                  std::max(bounds[d], std::min(bounds[d + 3], image[d]));
              const double delta = image[d] - nearest;
              distance2 += delta * delta;
            }
          if(distance2 <= radius2)
            return true;
        }
  return false;
}

static std::vector<int> dust_dg_byte_counts(const std::vector<int> &counts,
                                             size_t record_size,
                                             std::vector<int> &displacements)
{
  std::vector<int> byte_counts(counts.size());
  displacements.resize(counts.size());
  long long total_bytes = 0;
  for(size_t i = 0; i < counts.size(); i++)
    {
      const long long bytes = (long long)counts[i] * record_size;
      if(bytes > 2147483647LL || total_bytes > 2147483647LL)
        Terminate("dust D/G exchange exceeds MPI_Alltoallv integer limits");
      byte_counts[i] = (int)bytes;
      displacements[i] = (int)total_bytes;
      total_bytes += bytes;
    }
  if(total_bytes > 2147483647LL)
    Terminate("dust D/G exchange exceeds MPI_Alltoallv integer limits");
  return byte_counts;
}

// Fine grid used only for the D/G neighborhood sums.
//
// The shared feedback hash (gas_hash, dust_hash) has 12-31 kpc cells, while
// D/G spheres are 2-10 code units, so each query used to scan every particle
// in cells hundreds of times larger than the sphere.  This grid is rebuilt
// from the hash contents at every dust call with DTG_GRID_CELL-sized cells
// and stores each cell's total mass: cells entirely inside the sphere add
// their total directly, cells entirely outside are skipped, and only cells
// crossing the sphere surface are scanned particle by particle.  The
// particle set and the inclusion test match spatial_hash_zoom::
// sum_mass_in_sphere(), so results agree up to summation-order rounding.
static constexpr double DTG_GRID_CELL = 2.0;

// Coarse cell (code units) for routing D/G queries between tasks; equal to
// the largest association radius.
static constexpr double DTG_ROUTE_CELL = 10.0;

struct dust_dg_fine_grid
{
  struct cell_data
  {
    double mass = 0.0;
    std::vector<int> particles;
  };

  double cell = 0.0;
  long long n_per_dim = 0;
  std::unordered_map<long long, cell_data> cells;

  long long key(long long ix, long long iy, long long iz) const
  {
    ix = ((ix % n_per_dim) + n_per_dim) % n_per_dim;
    iy = ((iy % n_per_dim) + n_per_dim) % n_per_dim;
    iz = ((iz % n_per_dim) + n_per_dim) % n_per_dim;
    return ix + n_per_dim * (iy + n_per_dim * iz);
  }

  void build(simparticles *Sp, const spatial_hash_zoom &hash, int part_type)
  {
    // Cells tile the periodic box exactly, so wrapped indices are consistent.
    n_per_dim = std::max(1LL, (long long)std::floor(All.BoxSize / DTG_GRID_CELL));
    cell = All.BoxSize / n_per_dim;
    cells.clear();
    for(const auto &hash_cell : hash.cells)
      for(int idx : hash_cell.second)
        {
          if(Sp->P[idx].getType() != part_type)
            continue;
          const double mass = Sp->P[idx].getMass();
          if(!std::isfinite(mass) || !(mass > 0.0))
            continue;
          double pos[3];
          Sp->intpos_to_pos(Sp->P[idx].IntPos, pos);
          cell_data &c = cells[key((long long)std::floor(pos[0] / cell),
                                   (long long)std::floor(pos[1] / cell),
                                   (long long)std::floor(pos[2] / cell))];
          c.mass += mass;
          c.particles.push_back(idx);
        }
  }

  double sum_mass_in_sphere(simparticles *Sp, const MyIntPosType *center_intpos,
                            double radius) const
  {
    if(cells.empty() || !std::isfinite(radius) || !(radius > 0.0))
      return 0.0;
    MyIntPosType center_copy[3] = {center_intpos[0], center_intpos[1], center_intpos[2]};
    double c[3];
    Sp->intpos_to_pos(center_copy, c);
    const double radius2 = radius * radius;
    // Unwrapped cell ranges around the center; a cell's unwrapped box is its
    // nearest periodic image relative to the center since radius << BoxSize/2.
    long long lo[3], hi[3];
    for(int d = 0; d < 3; d++)
      {
        lo[d] = (long long)std::floor((c[d] - radius) / cell);
        hi[d] = (long long)std::floor((c[d] + radius) / cell);
      }
    double mass_sum = 0.0;
    for(long long ix = lo[0]; ix <= hi[0]; ix++)
      for(long long iy = lo[1]; iy <= hi[1]; iy++)
        for(long long iz = lo[2]; iz <= hi[2]; iz++)
          {
            const auto it = cells.find(key(ix, iy, iz));
            if(it == cells.end())
              continue;
            const long long ic[3] = {ix, iy, iz};
            double near2 = 0.0, far2 = 0.0;
            for(int d = 0; d < 3; d++)
              {
                const double b0 = ic[d] * cell, b1 = b0 + cell;
                const double dn = c[d] - std::max(b0, std::min(b1, c[d]));
                const double df = std::max(std::fabs(c[d] - b0), std::fabs(c[d] - b1));
                near2 += dn * dn;
                far2 += df * df;
              }
            if(near2 > radius2)
              continue;
            // Margin keeps particles within rounding of the surface on the
            // exact per-particle path.
            if(far2 < radius2 * (1.0 - 1e-9))
              {
                mass_sum += it->second.mass;
                continue;
              }
            for(int idx : it->second.particles)
              {
                double dxyz[3];
                Sp->nearest_image_intpos_to_pos(Sp->P[idx].IntPos, center_intpos, dxyz);
                if(dxyz[0] * dxyz[0] + dxyz[1] * dxyz[1] + dxyz[2] * dxyz[2] <= radius2)
                  mass_sum += Sp->P[idx].getMass();
              }
          }
    return mass_sum;
  }
};

static void estimate_local_dust_to_gas_ratio(
    simparticles *Sp, const std::vector<double> &radii,
    MPI_Comm Communicator, std::vector<double> &local_DTG,
    bool occupancy_routing, long long *queries_sent)
{
  int n_task = 1;
  MPI_Comm_size(Communicator, &n_task);

  static dust_dg_fine_grid gas_grid, dust_grid;
  gas_grid.build(Sp, gas_hash, 0);
  dust_grid.build(Sp, dust_hash, DUST_PARTICLE_TYPE);

  std::vector<std::vector<dust_dg_query> > requests((size_t)n_task);

  if(occupancy_routing)
    {
      // Each task publishes the DTG_ROUTE_CELL cells holding particles it
      // sums (exactly the fine-grid members), and a query goes only to tasks
      // owning a cell the sphere touches.  Tasks dropped this way contribute
      // exactly zero, and per-task query order is unchanged, so results are
      // bit-identical to bounding-box routing.  That routing sent most
      // queries to most tasks because zoom domains are scattered, so its cost
      // grew with the number of tasks instead of being divided among them.
      const long long n_c = std::max(1LL, (long long)std::floor(All.BoxSize / DTG_ROUTE_CELL));
      const double cc = All.BoxSize / n_c;
      auto ckey = [n_c](long long ix, long long iy, long long iz) {
        ix = ((ix % n_c) + n_c) % n_c;
        iy = ((iy % n_c) + n_c) % n_c;
        iz = ((iz % n_c) + n_c) % n_c;
        return ix + n_c * (iy + n_c * iz);
      };

      std::unordered_set<long long> local_cells;
      for(const dust_dg_fine_grid *grid : {&gas_grid, &dust_grid})
        for(const auto &fine_cell : grid->cells)
          for(int idx : fine_cell.second.particles)
            {
              double pos[3];
              Sp->intpos_to_pos(Sp->P[idx].IntPos, pos);
              local_cells.insert(ckey((long long)std::floor(pos[0] / cc),
                                      (long long)std::floor(pos[1] / cc),
                                      (long long)std::floor(pos[2] / cc)));
            }

      std::vector<long long> mine(local_cells.begin(), local_cells.end());
      int n_mine = (int)mine.size();
      std::vector<int> cell_counts((size_t)n_task), cell_displs((size_t)n_task);
      MPI_Allgather(&n_mine, 1, MPI_INT, cell_counts.data(), 1, MPI_INT, Communicator);
      long long n_all = 0;
      for(int task = 0; task < n_task; task++)
        {
          cell_displs[task] = (int)n_all;
          n_all += cell_counts[task];
        }
      if(n_all > 2147483647LL)
        Terminate("dust D/G routing: too many occupied cells for MPI_Allgatherv");
      std::vector<long long> all_cells((size_t)std::max(n_all, 1LL));
      long long empty_cell = 0;
      MPI_Allgatherv(mine.empty() ? &empty_cell : mine.data(), n_mine, MPI_LONG_LONG,
                     all_cells.data(), cell_counts.data(), cell_displs.data(),
                     MPI_LONG_LONG, Communicator);

      std::unordered_map<long long, std::vector<int> > owners;
      owners.reserve((size_t)n_all);
      for(int task = 0; task < n_task; task++)
        for(int k = cell_displs[task]; k < cell_displs[task] + cell_counts[task]; k++)
          owners[all_cells[k]].push_back(task);

      std::vector<long long> last_query((size_t)n_task, -1);
      for(size_t dust_idx = 0; dust_idx < radii.size(); dust_idx++)
        {
          if(!(radii[dust_idx] > 0.0))
            continue;
          dust_dg_query query;
          query.dust_index = (int)dust_idx;
          for(int d = 0; d < 3; d++)
            query.center[d] = Sp->P[dust_idx].IntPos[d];
          query.radius = radii[dust_idx];

          double c[3];
          Sp->intpos_to_pos(query.center, c);
          const double r2 = query.radius * query.radius;
          long long lo[3], hi[3];
          for(int d = 0; d < 3; d++)
            {
              lo[d] = (long long)std::floor((c[d] - query.radius) / cc);
              hi[d] = (long long)std::floor((c[d] + query.radius) / cc);
            }
          for(long long ix = lo[0]; ix <= hi[0]; ix++)
            for(long long iy = lo[1]; iy <= hi[1]; iy++)
              for(long long iz = lo[2]; iz <= hi[2]; iz++)
                {
                  const long long ic[3] = {ix, iy, iz};
                  double near2 = 0.0;
                  for(int d = 0; d < 3; d++)
                    {
                      const double b0 = ic[d] * cc, b1 = b0 + cc;
                      const double dn = c[d] - std::max(b0, std::min(b1, c[d]));
                      near2 += dn * dn;
                    }
                  if(near2 > r2)
                    continue;
                  const auto it = owners.find(ckey(ix, iy, iz));
                  if(it == owners.end())
                    continue;
                  for(int task : it->second)
                    if(last_query[task] != (long long)dust_idx)
                      {
                        last_query[task] = (long long)dust_idx;
                        requests[task].push_back(query);
                      }
                }
        }

      // The outer loop runs in ascending dust index, as in bounding-box
      // routing, so per-task query order (and thus summation order) matches.
    }
  else
    {
      // Bounding-box routing (kept for the routing self-check).
      double local_bounds[7] = {1e30, 1e30, 1e30, -1e30, -1e30, -1e30, 0.0};
      for(const auto &cell : gas_hash.cells)
        for(int particle_idx : cell.second)
          if(Sp->P[particle_idx].getType() == 0)
            extend_dust_dg_bounds(Sp, particle_idx, local_bounds);
      for(const auto &cell : dust_hash.cells)
        for(int particle_idx : cell.second)
          if(Sp->P[particle_idx].getType() == DUST_PARTICLE_TYPE &&
             Sp->P[particle_idx].getMass() > DUST_MASS_TO_DESTROY &&
             Sp->P[particle_idx].ID.get() != 0)
            extend_dust_dg_bounds(Sp, particle_idx, local_bounds);

      std::vector<double> task_bounds((size_t)n_task * 7);
      MPI_Allgather(local_bounds, 7, MPI_DOUBLE, task_bounds.data(), 7, MPI_DOUBLE,
                    Communicator);

      for(size_t dust_idx = 0; dust_idx < radii.size(); dust_idx++)
        {
          if(!(radii[dust_idx] > 0.0))
            continue;
          dust_dg_query query;
          query.dust_index = (int)dust_idx;
          for(int d = 0; d < 3; d++)
            query.center[d] = Sp->P[dust_idx].IntPos[d];
          query.radius = radii[dust_idx];

          double center[3];
          Sp->intpos_to_pos(query.center, center);
          for(int task = 0; task < n_task; task++)
            if(dust_dg_sphere_intersects_bounds(
                   center, query.radius, &task_bounds[(size_t)task * 7]))
              requests[task].push_back(query);
        }
    }

  std::vector<int> send_counts((size_t)n_task), recv_counts((size_t)n_task);
  for(int task = 0; task < n_task; task++)
    {
      if(requests[task].size() > (size_t)2147483647)
        Terminate("too many dust D/G neighborhood requests for MPI");
      send_counts[task] = (int)requests[task].size();
    }
  MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT,
               Communicator);

  std::vector<int> send_displacements, recv_displacements;
  const std::vector<int> send_bytes =
      dust_dg_byte_counts(send_counts, sizeof(dust_dg_query), send_displacements);
  const std::vector<int> recv_bytes =
      dust_dg_byte_counts(recv_counts, sizeof(dust_dg_query), recv_displacements);
  const int n_send = std::accumulate(send_counts.begin(), send_counts.end(), 0);
  if(queries_sent)
    *queries_sent = n_send;
  const int n_recv = std::accumulate(recv_counts.begin(), recv_counts.end(), 0);
  std::vector<dust_dg_query> send_queries((size_t)n_send);
  std::vector<dust_dg_query> recv_queries((size_t)n_recv);
  dust_dg_query empty_query = {};
  for(int task = 0; task < n_task; task++)
    std::copy(requests[task].begin(), requests[task].end(),
              send_queries.begin() + send_displacements[task] /
                                         (int)sizeof(dust_dg_query));
  const void *send_query_data =
      send_queries.empty() ? static_cast<const void *>(&empty_query)
                           : static_cast<const void *>(send_queries.data());
  void *recv_query_data =
      recv_queries.empty() ? static_cast<void *>(&empty_query)
                           : static_cast<void *>(recv_queries.data());
  MPI_Alltoallv(send_query_data, send_bytes.data(), send_displacements.data(),
                MPI_BYTE, recv_query_data, recv_bytes.data(),
                recv_displacements.data(), MPI_BYTE, Communicator);

  // Self-check: for the first queries on each task, compare against the
  // shared-hash sums.  Only summation order differs, so any relative
  // difference beyond rounding indicates a bug.
  static int verify_remaining = 5000;
  static long long verify_mismatches = 0;
  static double verify_max_rel = 0.0;

  std::vector<dust_dg_result> send_results((size_t)n_recv);
  for(int i = 0; i < n_recv; i++)
    {
      send_results[i].dust_index = recv_queries[i].dust_index;
      send_results[i].gas_mass = gas_grid.sum_mass_in_sphere(
          Sp, recv_queries[i].center, recv_queries[i].radius);
      send_results[i].dust_mass = dust_grid.sum_mass_in_sphere(
          Sp, recv_queries[i].center, recv_queries[i].radius);

      if(verify_remaining > 0)
        {
          verify_remaining--;
          const double ref[2] = {
              gas_hash.sum_mass_in_sphere(Sp, recv_queries[i].center,
                                          recv_queries[i].radius, 0),
              dust_hash.sum_mass_in_sphere(Sp, recv_queries[i].center,
                                           recv_queries[i].radius, DUST_PARTICLE_TYPE)};
          const double got[2] = {send_results[i].gas_mass, send_results[i].dust_mass};
          for(int k = 0; k < 2; k++)
            {
              const double rel = std::fabs(got[k] - ref[k]) /
                                 std::max(std::fabs(ref[k]), 1e-300);
              if(ref[k] == 0.0 && got[k] == 0.0)
                continue;
              verify_max_rel = std::max(verify_max_rel, rel);
              if(rel > 1e-10 && ++verify_mismatches <= 10)
                {
                  // Brute force over every particle in the shared hash: the
                  // ground truth for which of the two methods is wrong.
                  const spatial_hash_zoom &h = (k == 0) ? gas_hash : dust_hash;
                  const int type = (k == 0) ? 0 : DUST_PARTICLE_TYPE;
                  const double r2 = recv_queries[i].radius * recv_queries[i].radius;
                  double brute = 0.0, nearest2 = 1e300;
                  for(const auto &hc : h.cells)
                    for(int idx : hc.second)
                      {
                        if(Sp->P[idx].getType() != type) continue;
                        const double mass = Sp->P[idx].getMass();
                        if(!std::isfinite(mass) || !(mass > 0.0)) continue;
                        double dxyz[3];
                        Sp->nearest_image_intpos_to_pos(Sp->P[idx].IntPos,
                                                        recv_queries[i].center, dxyz);
                        const double d2 = dxyz[0] * dxyz[0] + dxyz[1] * dxyz[1] + dxyz[2] * dxyz[2];
                        nearest2 = std::min(nearest2, d2);
                        if(d2 <= r2) brute += mass;
                      }
                  double cpos[3];
                  MyIntPosType ccopy[3] = {recv_queries[i].center[0], recv_queries[i].center[1],
                                           recv_queries[i].center[2]};
                  Sp->intpos_to_pos(ccopy, cpos);
                  printf("[DTG_GRID_MISMATCH|T=%d] %s grid=%.17g hash=%.17g brute=%.17g r=%g "
                         "nearest=%g center=(%.2f,%.2f,%.2f) hash_bbox=(%.2f..%.2f, %.2f..%.2f, "
                         "%.2f..%.2f) cell=%.3f built=%d\n",
                         All.ThisTask, k == 0 ? "gas" : "dust", got[k], ref[k], brute,
                         recv_queries[i].radius, sqrt(nearest2), cpos[0], cpos[1], cpos[2],
                         h.bbox_min[0], h.bbox_max[0], h.bbox_min[1], h.bbox_max[1],
                         h.bbox_min[2], h.bbox_max[2], h.cell_size, (int)h.is_built);
                }
            }
          if(verify_remaining == 0)
            printf("[DTG_GRID_VERIFY|T=%d] fine grid checked on 5000 queries: "
                   "%lld mismatches, max relative difference %.3g\n",
                   All.ThisTask, verify_mismatches, verify_max_rel);
        }
    }

  std::vector<int> result_send_displacements, result_recv_displacements;
  const std::vector<int> result_send_bytes =
      dust_dg_byte_counts(recv_counts, sizeof(dust_dg_result),
                          result_send_displacements);
  const std::vector<int> result_recv_bytes =
      dust_dg_byte_counts(send_counts, sizeof(dust_dg_result),
                          result_recv_displacements);
  std::vector<dust_dg_result> recv_results((size_t)n_send);
  dust_dg_result empty_result = {};
  const void *send_result_data =
      send_results.empty() ? static_cast<const void *>(&empty_result)
                           : static_cast<const void *>(send_results.data());
  void *recv_result_data =
      recv_results.empty() ? static_cast<void *>(&empty_result)
                           : static_cast<void *>(recv_results.data());
  MPI_Alltoallv(send_result_data, result_send_bytes.data(),
                result_send_displacements.data(), MPI_BYTE,
                recv_result_data, result_recv_bytes.data(),
                result_recv_displacements.data(), MPI_BYTE, Communicator);

  std::vector<double> gas_mass(radii.size(), 0.0);
  std::vector<double> dust_mass(radii.size(), 0.0);
  for(int i = 0; i < n_send; i++)
    {
      const int dust_idx = recv_results[i].dust_index;
      if(dust_idx < 0 || (size_t)dust_idx >= radii.size())
        Terminate("invalid dust index returned by D/G neighbor exchange");
      gas_mass[dust_idx] += recv_results[i].gas_mass;
      dust_mass[dust_idx] += recv_results[i].dust_mass;
    }

  local_DTG.assign(radii.size(), 0.0);
  for(size_t i = 0; i < radii.size(); i++)
    if(radii[i] > 0.0 && std::isfinite(gas_mass[i]) &&
       std::isfinite(dust_mass[i]) && gas_mass[i] > 0.0 &&
       dust_mass[i] > 0.0)
      local_DTG[i] = std::max(1e-6, std::min(0.1, dust_mass[i] / gas_mass[i]));
}

// ── Module-level statistics ───────────────────────────────────────────────────
// These accumulate over the entire run on each task and are reduced to
// global values inside print_dust_statistics().
long long NDustCreated               = 0;
long long NDustCreatedBySNII         = 0;
long long NDustCreatedByAGB          = 0;
long long NDustCreatedByLRN          = 0;
long long NDustDestroyed             = 0;
double    TotalDustMass              = 0.0;
long long LocalDustCreatedThisStep   = 0;
long long LocalDustDestroyedThisStep = 0;
double    LocalDustMassChange        = 0.0;
int       DustNeedsSynchronization   = 0;
long long GlobalDustCount            = 0;
long long NShatteringEvents          = 0;
double    TotalSizeReductionShattering = 0.0;  // Σ(a_old − a_new) across shattering events

// Hash usage
static long long HashSearches        = 0;
static long long HashSearchesFailed  = 0;

// Destruction by mechanism
long long NDustDestroyedByThermal    = 0;
long long NDustDestroyedByShock      = 0;
long long NDustDestroyedByAstration  = 0;
double    TotalDustMassAstrated      = 0.0;

// Destruction via internal / non-physics paths — ideally all zero in
// a clean run. Non-zero values flag domain-exchange bugs or numerical issues.
long long NDustDestroyedByCleanup    = 0;  // cleanup_invalid_dust_particles()
long long NDustDestroyedByCorruption = 0;  // dust_grain_growth_subgrid() corruption
long long NDustDestroyedByBadGasIndex= 0;  // hash returned a non-gas particle

// Growth / erosion
long long NGrainGrowthEvents         = 0;
long long NGrainErosionEvents        = 0;
double    TotalMassGrown             = 0.0;
double    TotalMassDestroyedByThermal= 0.0;
double    TotalMassDestroyedByShock  = 0.0;
double    TotalMassErodedByThermal   = 0.0;
double    TotalMassErodedByShock     = 0.0;

// Coagulation
long long NCoagulationEvents         = 0;


// ============================================================================
// Gas <-> dust elemental bookkeeping
//
// Dust is represented as:
//   carbonaceous mass = CF * M_dust
//   silicate mass     = (1-CF) * M_dust
//
// The silicate component is mapped onto O, Mg, Si, and Fe using fixed
// normalized mass fractions derived from the adopted solar abundance pattern.
// N and Ne are not currently constituents of the dust model.
// ============================================================================
static constexpr double SIL_FRAC_O  = 0.69876708;
static constexpr double SIL_FRAC_MG = 0.07680773;
static constexpr double SIL_FRAC_SI = 0.08097301;
static constexpr double SIL_FRAC_FE = 0.14345218;

// Synchronize the redundant gas metallicity representations.
static inline void sync_gas_metallicity(simparticles *Sp, int gas_idx)
{
    double gas_mass = Sp->P[gas_idx].getMass();
    if(gas_mass > 0.0)
    {
        Sp->SphP[gas_idx].Metallicity = Sp->SphP[gas_idx].MassMetallicity / gas_mass;
        Sp->P[gas_idx].Metallicity = Sp->SphP[gas_idx].Metallicity;
    }
}


// Return carbonaceous + silicate dust material to the gas.
//
// dM_carbon and dM_silicate are CODE MASS UNITS and must be >= 0.
//
// This routine increases:
//   gas mass
//   total gas metal mass
//   C/O/Mg/Si/Fe gas reservoirs
//
// N and Ne are unchanged.
static void return_dust_material_to_gas(
    simparticles *Sp,
    int gas_idx,
    double dM_carbon,
    double dM_silicate)
{
    if(gas_idx < 0)
        return;
    if(dM_carbon < 0.0 || dM_silicate < 0.0)
        Terminate("return_dust_material_to_gas received negative mass");
    double dM_total = dM_carbon + dM_silicate;
    if(dM_total <= 0.0)
        return;
    double old_gas_mass = Sp->P[gas_idx].getMass();
    if(old_gas_mass <= 0.0)
        Terminate("return_dust_material_to_gas: gas mass <= 0");

    // Convert existing mass fractions to absolute element masses.
    double M_C = old_gas_mass * Sp->SphP[gas_idx].GasCarbonMassFraction;
    double M_N = old_gas_mass * Sp->SphP[gas_idx].GasNitrogenMassFraction;
    double M_O = old_gas_mass * Sp->SphP[gas_idx].GasOxygenMassFraction;
    double M_Ne = old_gas_mass * Sp->SphP[gas_idx].GasNeonMassFraction;
    double M_Mg = old_gas_mass * Sp->SphP[gas_idx].GasMagnesiumMassFraction;
    double M_Si = old_gas_mass * Sp->SphP[gas_idx].GasSiliconMassFraction;
    double M_Fe = old_gas_mass * Sp->SphP[gas_idx].GasIronMassFraction;

    // Return dust material to its corresponding gas reservoirs.
    M_C  += dM_carbon;
    M_O  += dM_silicate * SIL_FRAC_O;
    M_Mg += dM_silicate * SIL_FRAC_MG;
    M_Si += dM_silicate * SIL_FRAC_SI;
    M_Fe += dM_silicate * SIL_FRAC_FE;
    double new_gas_mass = old_gas_mass + dM_total;
    Sp->P[gas_idx].setMass(new_gas_mass);
    // Total dust mass is pure metal mass.
    Sp->SphP[gas_idx].MassMetallicity += dM_total;
    // Reconstruct element mass fractions using the new gas mass.
    Sp->SphP[gas_idx].GasCarbonMassFraction = M_C / new_gas_mass;
    Sp->SphP[gas_idx].GasNitrogenMassFraction = M_N / new_gas_mass;
    Sp->SphP[gas_idx].GasOxygenMassFraction = M_O / new_gas_mass;
    Sp->SphP[gas_idx].GasNeonMassFraction = M_Ne / new_gas_mass;
    Sp->SphP[gas_idx].GasMagnesiumMassFraction = M_Mg / new_gas_mass;
    Sp->SphP[gas_idx].GasSiliconMassFraction = M_Si / new_gas_mass;
    Sp->SphP[gas_idx].GasIronMassFraction = M_Fe / new_gas_mass;
    sync_gas_metallicity(Sp, gas_idx);
}

// Remove carbonaceous + silicate material from gas and lock it into dust.
//
// requested_carbon and requested_silicate are CODE MASS UNITS.
//
// Returns the ACTUAL total mass removed. The function automatically limits
// removal to the elemental material physically available in the gas.
//
// The actually removed carbon/silicate masses are returned through the
// reference arguments.
static double remove_dust_material_from_gas(
    simparticles *Sp,
    int gas_idx,
    double requested_carbon,
    double requested_silicate,
    double &removed_carbon,
    double &removed_silicate)
{
    removed_carbon   = 0.0;
    removed_silicate = 0.0;
    if(gas_idx < 0)
        return 0.0;
    if(requested_carbon < 0.0 || requested_silicate < 0.0)
        Terminate("remove_dust_material_from_gas received negative mass");
    double old_gas_mass = Sp->P[gas_idx].getMass();
    if(old_gas_mass <= 0.0)
        return 0.0;
    // Existing absolute gas-phase element masses.
    double M_C = old_gas_mass * Sp->SphP[gas_idx].GasCarbonMassFraction;
    double M_N = old_gas_mass * Sp->SphP[gas_idx].GasNitrogenMassFraction;
    double M_O = old_gas_mass * Sp->SphP[gas_idx].GasOxygenMassFraction;
    double M_Ne = old_gas_mass * Sp->SphP[gas_idx].GasNeonMassFraction;
    double M_Mg = old_gas_mass * Sp->SphP[gas_idx].GasMagnesiumMassFraction;
    double M_Si = old_gas_mass * Sp->SphP[gas_idx].GasSiliconMassFraction;
    double M_Fe = old_gas_mass * Sp->SphP[gas_idx].GasIronMassFraction;
    // ------------------------------------------------------------
    // Carbon availability
    // ------------------------------------------------------------
    removed_carbon = std::min(requested_carbon, M_C);
    // ------------------------------------------------------------
    // Silicate availability
    //
    // The maximum silicate mass is determined by whichever required
    // element is the limiting reservoir.
    // ------------------------------------------------------------
    double silicate_available = HUGE_VAL;
    if(SIL_FRAC_O > 0.0)
        silicate_available = std::min(silicate_available, M_O / SIL_FRAC_O);
    if(SIL_FRAC_MG > 0.0)
        silicate_available = std::min(silicate_available, M_Mg / SIL_FRAC_MG);
    if(SIL_FRAC_SI > 0.0)
        silicate_available = std::min(silicate_available, M_Si / SIL_FRAC_SI);
    if(SIL_FRAC_FE > 0.0)
        silicate_available = std::min(silicate_available, M_Fe / SIL_FRAC_FE);

    removed_silicate = std::min(requested_silicate, silicate_available);
    removed_silicate = std::max(0.0, removed_silicate);

    double dM_total = removed_carbon + removed_silicate;

    if(dM_total <= 0.0)
        return 0.0;

    // ------------------------------------------------------------
    // Remove actual element masses.
    // ------------------------------------------------------------
    M_C -= removed_carbon;
    M_O  -= removed_silicate * SIL_FRAC_O;
    M_Mg -= removed_silicate * SIL_FRAC_MG;
    M_Si -= removed_silicate * SIL_FRAC_SI;
    M_Fe -= removed_silicate * SIL_FRAC_FE;
    // Numerical protection.
    M_C  = std::max(0.0, M_C);
    M_O  = std::max(0.0, M_O);
    M_Mg = std::max(0.0, M_Mg);
    M_Si = std::max(0.0, M_Si);
    M_Fe = std::max(0.0, M_Fe);
    double new_gas_mass = old_gas_mass - dM_total;
    if(new_gas_mass <= 0.0)
        Terminate("remove_dust_material_from_gas removed all gas mass");
    Sp->P[gas_idx].setMass(new_gas_mass);

    // ISM dust growth removes pure metal mass from the gas.
    Sp->SphP[gas_idx].MassMetallicity -= dM_total;
    if(Sp->SphP[gas_idx].MassMetallicity < 0.0)
    {
        if(Sp->SphP[gas_idx].MassMetallicity > -1e-12)
            Sp->SphP[gas_idx].MassMetallicity = 0.0;
        else
            Terminate("Gas metal mass became negative during dust growth");
    }
    // Reconstruct fractions.
    Sp->SphP[gas_idx].GasCarbonMassFraction = M_C / new_gas_mass;
    Sp->SphP[gas_idx].GasNitrogenMassFraction = M_N / new_gas_mass;
    Sp->SphP[gas_idx].GasOxygenMassFraction = M_O / new_gas_mass;
    Sp->SphP[gas_idx].GasNeonMassFraction = M_Ne / new_gas_mass;
    Sp->SphP[gas_idx].GasMagnesiumMassFraction = M_Mg / new_gas_mass;
    Sp->SphP[gas_idx].GasSiliconMassFraction = M_Si / new_gas_mass;
    Sp->SphP[gas_idx].GasIronMassFraction = M_Fe / new_gas_mass;
    sync_gas_metallicity(Sp, gas_idx);
    return dM_total;
}
// ═══════════════════════════════════════════════════════════════════════════════
// cleanup_invalid_dust_particles
// ═══════════════════════════════════════════════════════════════════════════════
/**
 * Scan all dust particles and remove any with unphysical state.
 *
 * Two known sources of corruption:
 *  1. DOMAIN_EXCHANGE VICTIM — GrainRadius, CarbonMassFraction, and DustTemperature
 *     are all exactly 0.0 (Gadget zeroed the DustP[] slot with memset during
 *     domain exchange but the P[] slot survived). The particle position and
 *     velocity are still intact.
 *  2. GENUINE NUMERICAL CORRUPTION — NaN, Inf, or negative values propagated
 *     from growth/drag/sputtering. Typically accompanied by bad velocity too.
 *
 * Corrupted particles are converted to type 3 with zero mass and ID so they
 * are swept up by destroy_dust_particles() on the next call.
 *
 * Prints per-reason tallies on any task that finds corrupted particles,
 * with detailed per-particle reports for the first 50 per call.
 */
void cleanup_invalid_dust_particles(simparticles *Sp)
{
  int n_bad_radius   = 0;
  int n_bad_mass     = 0;
  int n_bad_pos      = 0;
  int n_bad_vel      = 0;
  int n_bad_carbon   = 0;
  int n_bad_temp     = 0;
  int n_zero_radius_nonzero_mass = 0;
  int n_fully_zeroed             = 0;
  int cleaned        = 0;
  int detail_count   = 0;  // per-call cap on detailed print lines

  for(int i = 0; i < Sp->NumPart; i++) {
    if(Sp->P[i].getType() != DUST_PARTICLE_TYPE) continue;
    double a    = Sp->DustP[i].GrainRadius;
    double mass = Sp->P[i].getMass();
    double cf   = Sp->DustP[i].CarbonMassFraction;
    double temp = Sp->DustP[i].DustTemperature;
    double pos[3];
    Sp->intpos_to_pos(Sp->P[i].IntPos, pos);
    bool is_corrupt = false;
    if(a <= 0.0 || !isfinite(a))                                          { n_bad_radius++; is_corrupt = true; }
    if(!isfinite(mass) || mass <= 0.0)                                    { n_bad_mass++;   is_corrupt = true; }
    if(!isfinite(pos[0]) || !isfinite(pos[1]) || !isfinite(pos[2]))       { n_bad_pos++;    is_corrupt = true; }
    if(!isfinite(Sp->P[i].Vel[0]) || !isfinite(Sp->P[i].Vel[1]) ||
       !isfinite(Sp->P[i].Vel[2]))                                        { n_bad_vel++;    is_corrupt = true; }
    if(!isfinite(cf) || cf < 0.0 || cf > 1.0)                             { n_bad_carbon++; is_corrupt = true; }
    if(!isfinite(temp) || temp < 0.0)                                     { n_bad_temp++;   is_corrupt = true; }
    if(is_corrupt) {
      // Distinguish the two corruption subtypes for diagnostics.
      bool is_domain_exchange_victim = (a == 0.0 && cf == 0.0 && temp == 0.0);
      if(is_domain_exchange_victim && mass > DUST_MASS_TO_DESTROY)
        n_zero_radius_nonzero_mass++;  // DustP lost but P[] intact — exchange timing bug
      else if(is_domain_exchange_victim && mass < DUST_MASS_TO_DESTROY)
        n_fully_zeroed++;              // both arrays zeroed — catastrophic
      if(detail_count < 50) {
        printf("[CORRUPT_DETAIL|T=%d|Step=%d] idx=%d ID=%lld: "
               "a=%.3e mass=%.3e cf=%.3f T=%.1f "
               "pos=(%.2f,%.2f,%.2f) vel=(%.2f,%.2f,%.2f) %s\n",
               All.ThisTask, All.NumCurrentTiStep, i,
               (long long)Sp->P[i].ID.get(),
               a, mass, cf, temp,
               pos[0], pos[1], pos[2],
               Sp->P[i].Vel[0], Sp->P[i].Vel[1], Sp->P[i].Vel[2],
               is_domain_exchange_victim ? "[LIKELY_DOMAIN_EXCHANGE_VICTIM]"
                                         : "[GENUINE_CORRUPTION]");
        detail_count++;
      }
      log_dust_particle_event(Sp, i, -1, DUST_EVENT_CLEANUP);

      // Neutralise the particle so destroy_dust_particles() will remove it.
      Sp->P[i].setMass(DUST_MASS_TO_DESTROY);
      Sp->P[i].ID.set(0);
      Sp->P[i].setType(3);
      memset(&Sp->DustP[i], 0, sizeof(dust_data));
      Sp->DustP[i].GrainRadius = All.DustMinGrainSize;
      NDustDestroyedByCleanup++;
      cleaned++;
    }
  }
  if(cleaned > 0) {
    printf("[CLEANUP|T=%d|Step=%d|a=%.4f z=%.3f] Removed %d corrupted dust particles:\n"
           "  bad_radius=%d bad_mass=%d bad_pos=%d bad_vel=%d bad_carbon=%d bad_temp=%d\n"
           "  zero_radius_nonzero_mass=%d (DustP lost, P[] intact → exchange timing)\n"
           "  fully_zeroed=%d            (both P[] and DustP zeroed → catastrophic)\n"
           "  RunningTotal=%lld\n",
           All.ThisTask, All.NumCurrentTiStep, All.Time, 1.0/All.Time - 1.0,
           cleaned,
           n_bad_radius, n_bad_mass, n_bad_pos, n_bad_vel, n_bad_carbon, n_bad_temp,
           n_zero_radius_nonzero_mass,
           n_fully_zeroed,
           NDustDestroyedByCleanup);
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// get_temperature_from_entropy
// ═══════════════════════════════════════════════════════════════════════════════
/**
 * Convert stored entropy to temperature for a gas particle, accounting for
 * ionisation state via the electron fraction ne (from SphP[].Ne).
 *
 * Mean molecular weight: μ = (1 + 4Y) / (1 + Y + ne)
 * where Y = (1 − XH) / (4 XH) is the helium-to-hydrogen number ratio.
 */
double get_temperature_from_entropy(simparticles *Sp, int idx)
{
  double utherm = Sp->get_utherm_from_entropy(idx);
  double ne     = Sp->SphP[idx].Ne;
  double XH = HYDROGEN_MASSFRAC;
  double Y  = (1.0 - XH) / (4.0 * XH);
  double mu = (1.0 + 4.0 * Y) / (1.0 + Y + ne);
  double temp = (GAMMA - 1.0) * utherm
                * (All.UnitEnergy_in_cgs / All.UnitMass_in_g)
                / BOLTZMANN * PROTONMASS * mu;
  return temp;
}

// ═══════════════════════════════════════════════════════════════════════════════
// sample_shattering_fragment_radius
// ═══════════════════════════════════════════════════════════════════════════════
/**
 * Draw a representative post-shattering grain radius from a power-law
 * fragment MASS distribution.
 *
 * We assume the physical fragment number distribution
 *
 *     dN/da ∝ a^(-q)
 *
 * and individual grain mass m(a) ∝ a^3.  Therefore the amount of fragment
 * MASS per unit radius is
 *
 *     dM/da ∝ a^(3-q).
 *
 * Because a CosmicGrain PartType6 particle represents an ensemble of grains
 * with a fixed total represented dust mass, we sample from dM/da rather than
 * directly from dN/da.
 *
 * The resulting representative radius is restricted to
 *
 *     DustMinGrainSize <= a_new <= max_fraction * a_parent.
 *
 * Thus:
 *
 *   - total superparticle dust mass does NOT change;
 *   - CarbonMassFraction does NOT change;
 *   - no new PartType6 particle is created;
 *   - only GrainRadius changes;
 *   - different shattering events populate different fragment sizes.
 *
 * For q != 4:
 *
 *     P(a) ∝ a^(3-q)
 *
 * has inverse-CDF sampling
 *
 *     a = [ a_min^(4-q)
 *           + u (a_max^(4-q) - a_min^(4-q)) ]^(1/(4-q)).
 *
 * For q = 4 the distribution is logarithmic:
 *
 *     a = a_min * (a_max/a_min)^u.
 */
static double sample_shattering_fragment_radius(double parent_radius_nm,
                                                double q,
                                                double max_fraction,
                                                double min_radius_nm)
{
    // -------------------------------------------------------------------------
    // Basic sanity checks
    // -------------------------------------------------------------------------

    if(!isfinite(parent_radius_nm) ||
       !isfinite(q) ||
       !isfinite(max_fraction) ||
       !isfinite(min_radius_nm))
    {
        return min_radius_nm;
    }

    if(parent_radius_nm <= min_radius_nm)
        return min_radius_nm;

    // Require the largest possible fragment to be smaller than the parent.
    if(max_fraction <= 0.0 || max_fraction >= 1.0)
      return min_radius_nm;

    double a_min = min_radius_nm;
    double a_max = max_fraction * parent_radius_nm;

    // If the parent is already so small that there is no meaningful allowed
    // fragment interval, simply return the minimum grain size.
    if(a_max <= a_min)
        return a_min;

    // Random number supplied by the existing Gadget/CosmicGrain RNG.
    double u = get_random_number();

    // Numerical protection in case RNG ever returns an endpoint outside
    // the expected interval because of roundoff.
    u = std::max(0.0, std::min(1.0, u));

    double a_new;

    // -------------------------------------------------------------------------
    // Special case q = 4:
    //
    // dM/da ∝ a^-1
    // -------------------------------------------------------------------------

    if(fabs(q - 4.0) < 1.0e-10)
    {
        a_new =
            a_min *
            pow(a_max / a_min, u);
    }

    // -------------------------------------------------------------------------
    // General power-law case
    // -------------------------------------------------------------------------

    else
    {
        double beta = 4.0 - q;

        double amin_beta = pow(a_min, beta);
        double amax_beta = pow(a_max, beta);

        double sampled_beta =
            amin_beta +
            u * (amax_beta - amin_beta);

        a_new =
            pow(sampled_beta, 1.0 / beta);
    }

    // -------------------------------------------------------------------------
    // Final numerical guards
    // -------------------------------------------------------------------------

    if(!isfinite(a_new))
        a_new = a_min;

    a_new = std::max(a_min, a_new);

    // This should already be guaranteed by a_max, but explicitly enforce
    // that a shattering event can never increase grain size.
    a_new = std::min(a_new, max_fraction * parent_radius_nm);

    return a_new;
}


// ═══════════════════════════════════════════════════════════════════════════════
// dust_grain_shattering
// ═══════════════════════════════════════════════════════════════════════════════
/**
 * Stochastically shatter a dust grain in warm/diffuse turbulent gas.
 *
 * EVENT RATE / ENVIRONMENT
 * ------------------------
 * The existing CosmicGrain shattering triggering physics is retained:
 *
 *   1. Turbulent-velocity proxy must exceed the composition-dependent
 *      Jones et al. shattering threshold.
 *
 *   2. Shattering occurs only in diffuse gas:
 *
 *          n_eff < DustCollisionDensityThresh
 *
 *   3. Event probability follows a Poisson process:
 *
 *          P = 1 - exp(-dt/tau_shatter)
 *
 *
 * NEW FRAGMENT TREATMENT
 * ----------------------
 * Previously every successful event used
 *
 *          a_new = 0.33 * a_old
 *
 * so all shattering events collapsed onto one deterministic daughter size.
 *
 * We now assume a power-law fragment number distribution
 *
 *          dN/da ∝ a^(-q)
 *
 * and sample the representative CosmicGrain superparticle radius from the
 * corresponding fragment MASS distribution
 *
 *          dM/da ∝ a^(3-q).
 *
 * The fragment spectrum is controlled by runtime parameters:
 *
 *          q                  = DustShatteringFragmentSlope
 *          maximum fragment  = DustShatteringMaxFragmentFraction
 *          minimum fragment  = DustMinGrainSize
 *
 *
 * MASS / COMPOSITION CONSERVATION
 * -------------------------------
 * Shattering in this implementation changes ONLY GrainRadius.
 *
 * It does NOT call setMass().
 * It does NOT change CarbonMassFraction.
 * It does NOT create or destroy a PartType6 particle.
 *
 * Therefore:
 *
 *          M_dust,new = M_dust,old
 *
 * exactly for every successful shattering event.
 *
 * Physically, a smaller representative GrainRadius at fixed superparticle
 * mass means the superparticle represents a larger number of individual
 * physical grains.
 */
void dust_grain_shattering(simparticles *Sp, int dust_idx, int gas_idx, double dt,
                           double local_DTG, double T_gas)
{
    const double fragment_q = DustShatteringFragmentSlope;
    const double fragment_max_fraction =
        DustShatteringMaxFragmentFraction;


    // =========================================================================
    // Diagnostic counters
    // =========================================================================

    static long long shat_calls       = 0;
    static long long shat_failed_vel  = 0;
    static long long shat_failed_dens = 0;
    static long long shat_failed_size = 0;
    static long long shat_passed      = 0;
    static long long shat_failed_dtg  = 0;

    // Fragment-radius diagnostics: event counts.
    static long long frag_1_10   = 0;
    static long long frag_10_30  = 0;
    static long long frag_30_50  = 0;
    static long long frag_50_70  = 0;
    static long long frag_70_90  = 0;
    static long long frag_ge90   = 0;

    // Fragment-radius diagnostics: represented dust mass.
    static double frag_mass_1_10  = 0.0;
    static double frag_mass_10_30 = 0.0;
    static double frag_mass_30_50 = 0.0;
    static double frag_mass_50_70 = 0.0;
    static double frag_mass_70_90 = 0.0;
    static double frag_mass_ge90  = 0.0;

    // Useful for checking the actual sampled fragment population.
    static double sum_parent_radius   = 0.0;
    static double sum_fragment_radius = 0.0;

    shat_calls++;


    // =========================================================================
    // Periodic shattering diagnostic
    // =========================================================================

    if(shat_calls % 50000 == 0 &&
       All.ThisTask == 0)
    {
        DUST_PRINT(
            "[SHAT_DIAG] calls=%lld "
            "failed: vel=%lld(%.1f%%) dens=%lld(%.1f%%) "
            "size=%lld(%.1f%%) dtg=%lld(%.1f%%) "
            "passed=%lld(%.1f%%)\n",

            shat_calls,

            shat_failed_vel,
            100.0 * shat_failed_vel / shat_calls,

            shat_failed_dens,
            100.0 * shat_failed_dens / shat_calls,

            shat_failed_size,
            100.0 * shat_failed_size / shat_calls,

            shat_failed_dtg,
            100.0 * shat_failed_dtg / shat_calls,

            shat_passed,
            100.0 * shat_passed / shat_calls
        );
if(shat_passed > 0)
{
    double frag_mass_total =
        frag_mass_1_10 +
        frag_mass_10_30 +
        frag_mass_30_50 +
        frag_mass_50_70 +
        frag_mass_70_90 +
        frag_mass_ge90;

    if(frag_mass_total > 0.0)
    {
        DUST_PRINT(
            "[SHAT_FRAGMENT_DIAG] "
            "q=%.2f maxfrac=%.2f "
            "mean_parent_nm=%.2f "
            "mean_fragment_nm=%.2f "
            "counts: "
            "1-10=%lld "
            "10-30=%lld "
            "30-50=%lld "
            "50-70=%lld "
            "70-90=%lld "
            ">=90=%lld "
            "massfrac: "
            "1-10=%.3f "
            "10-30=%.3f "
            "30-50=%.3f "
            "50-70=%.3f "
            "70-90=%.3f "
            ">=90=%.3f\n",

            fragment_q,
            fragment_max_fraction,

            sum_parent_radius / (double)shat_passed,
            sum_fragment_radius / (double)shat_passed,

            frag_1_10,
            frag_10_30,
            frag_30_50,
            frag_50_70,
            frag_70_90,
            frag_ge90,

            frag_mass_1_10  / frag_mass_total,
            frag_mass_10_30 / frag_mass_total,
            frag_mass_30_50 / frag_mass_total,
            frag_mass_50_70 / frag_mass_total,
            frag_mass_70_90 / frag_mass_total,
            frag_mass_ge90  / frag_mass_total
        );
    }
}
    }

    // =========================================================================
    // Gate ordering
    // =========================================================================
    //
    // Gates run cheapest-first: grain validity is two array reads, density
    // needs only SphP[].Density, and the velocity gate (which needs T_gas and
    // several pow() calls) runs last.  Every gate is a pure early return, so
    // the order does not change which grains shatter, only which counter a
    // rejected call is attributed to.


    // =========================================================================
    // Gate 1: grain validity
    // =========================================================================

    double a =
        Sp->DustP[dust_idx].
        GrainRadius;

    double M_dust =
        Sp->P[dust_idx].
        getMass();

    if(a <= All.DustMinGrainSize ||
       M_dust <= 0.0 ||
       !isfinite(a) ||
       !isfinite(M_dust))
    {
        shat_failed_size++;
        return;
    }


    // =========================================================================
    // Additional fragment-spectrum validity check
    // =========================================================================
    //
    // If the maximum allowed fragment is at/below DustMinGrainSize, there is no resolved
    // fragment interval available. Do nothing rather than destroy the entire
    // superparticle.
    //

    double maximum_fragment_radius =
        fragment_max_fraction *
        a;

    if(maximum_fragment_radius <=
       All.DustMinGrainSize)
    {
        shat_failed_size++;
        return;
    }


    // =========================================================================
    // Gate 2: density regime
    // =========================================================================
    //
    // Shattering operates in diffuse gas.
    // Coagulation operates in dense gas.
    //

    double n_H = dust_gas_nH_cgs(Sp, gas_idx);

    double DustClumpingFactor =
        dust_clumping_factor(
            n_H,
            Sp->SphP[gas_idx].Sfr >
            DUST_SFR_EPS
        );

    double n_eff =
        n_H *
        DustClumpingFactor;

    if(n_eff >
       All.DustCollisionDensityThresh)
    {
        shat_failed_dens++;
        return;
    }


    // ── Validate local DTG ────────────────────────────────────────────────────
    // local_DTG is the same MPI-combined neighborhood estimate used by
    // coagulation, passed in from update_dust_dynamics(). Both processes
    // now use the same DTG measure for consistency.
    if(!isfinite(local_DTG) || !(local_DTG > 0.0))
    {
        shat_failed_dtg++;
        return;
    }


    // =========================================================================
    // Gate 3: size-dependent grain velocity threshold
    // =========================================================================
    //
    // Grain-grain collision velocities in turbulence depend on grain size:
    // small grains have short stopping times, stay coupled to the gas, and
    // collide slowly.  We use the turbulence-driven velocity scaling of
    // Hirashita & Aoyama (2019), based on Yan et al. (2004) and Ormel &
    // Cuzzi (2007):
    //
    //   v_gr = 1.1 km/s  M^{3/2} (a / 0.1 um)^{1/2} (T / 1e4 K)^{1/4}
    //          (n_H / 1 cm^-3)^{-1/4} (rho_gr / 3.5 g cm^-3)^{1/2}
    //
    // n_H is the unclumped density, floored at 1e-4 cm^-3 as in the
    // timescale below.  The material density blends silicate (3.5 g cm^-3)
    // and graphite (2.24 g cm^-3) by CarbonMassFraction.
    //
    // The previous gate compared the size-independent gas sound speed with
    // the threshold and never rejected anything (vel = 0.0% of calls), so
    // grains in diffuse gas shattered repeatedly down to DustMinGrainSize.
    // Grains in [DustMinGrainSize, DustMinGrainSize/0.8] then cannot shatter
    // again, and with only ~5% of their mass in coagulation-eligible gas they
    // accumulated there (12% of all dust mass at z=0 in coag6_shat2).  With
    // this gate, shattering stops at a physical, environment-dependent
    // critical size instead (~30 nm in the central galaxy ISM for M = 1).
    //
    // Shattering threshold (unchanged):
    //
    //   silicate      ~ 1 km/s
    //   pure carbon   ~ 2 km/s
    //
    // CarbonMassFraction linearly interpolates between those limits.
    //
    // The critical size scales as M^-3, so shattering_mach is the most
    // sensitive choice here.  Keep hard-coded for the first controlled run.
    //

    const double shattering_mach = 1.0;

    double CF =
        Sp->DustP[dust_idx].
        CarbonMassFraction;

    double rho_gr =
        CF * 2.24 +
        (1.0 - CF) * 3.5;

    double v_grain_kms =
        1.1 *
        pow(shattering_mach, 1.5) *
        sqrt(a / 100.0) *
        pow(T_gas / 1.0e4, 0.25) *
        pow(std::max(n_H, 1e-4), -0.25) *
        sqrt(rho_gr / 3.5);

    double v_shatter_kms =
        1.0 + CF;

    if(!(v_grain_kms >=
         v_shatter_kms))
    {
        shat_failed_vel++;
        return;
    }

    // ── Trayford et al. (2026) Eq. 9 shattering timescale ────────────────────
    // Adapted for CosmicGrain's continuous grain radius.
    //
    // τ_sh = 54.1 Myr × (DTG/0.01)^-1 × (a/0.1μm) × f(n_H)
    //
    // where f(n_H) changes slope at n_H = 1 cm^-3:
    //   f = n_H^-1       for n_H < 1 cm^-3  (collision rate ∝ density)
    //   f = n_H^(-1/3)   for n_H >= 1 cm^-3 (grains more coupled to turbulence)
    //
    // Note: n_H here is the UNCLUMPED physical density, consistent with
    // Trayford's treatment of shattering as a diffuse-gas process.
    // Coagulation uses clumped n_eff; shattering uses raw n_H.
    //
    // The velocity threshold gate above (v_grain(a) > v_shatter) handles the
    // turbulent velocity requirement explicitly via a Poisson process, which
    // is more rigorous than Trayford's implicit velocity folding into the
    // timescale normalization. We therefore drop the velocity_factor term.

    double a_micron = a / 1000.0;  // nm -> micron

    // Density factor: collision rate dependence on gas density.
    // Floor at 1e-4 cm^-3 to avoid divergence in very diffuse CGM.
    double density_factor;
    if(n_H < 1.0)
        density_factor = 1.0 / std::max(n_H, 1e-4);
    else
        density_factor = pow(n_H, -1.0 / 3.0);

    double tau_shat_yr =
        54.1e6 *
        (0.01 / local_DTG) *
        (a_micron / 0.1) *
        density_factor *
        All.DustShatteringCalibration;

    // Physical and numerical limits.
    tau_shat_yr =
        std::max(tau_shat_yr, 1.0e6);

    tau_shat_yr =
        std::min(tau_shat_yr, 1.0e10);

    double tau_shat =
        tau_shat_yr *
        SEC_PER_YEAR /
        dust_code_time_in_s();


    // =========================================================================
    // Stochastic shattering event
    // =========================================================================
    //
    // Existing Poisson event probability retained unchanged.
    //

    double P_shatter =
        1.0 -
        exp(
            -dt /
            tau_shat
        );

    if(get_random_number() >=
       P_shatter)
    {
        return;
    }


    // =========================================================================
    // NEW: sample representative fragment radius
    // =========================================================================

    double mass_before =
        M_dust;

    double CF_before =
        CF;

    double a_new =
        sample_shattering_fragment_radius(
            a,
            fragment_q,
            fragment_max_fraction,
            All.DustMinGrainSize
        );


    // =========================================================================
    // Numerical validation of sampled radius
    // =========================================================================

    if(!isfinite(a_new) ||
       a_new < All.DustMinGrainSize ||
       a_new >= a)
    {
        DUST_PRINT(
            "[SHATTERING_INVALID_SAMPLE] "
            "a_old=%.6g a_new=%.6g "
            "amin=%.6g q=%.3f maxfrac=%.3f\n",
            a,
            a_new,
            All.DustMinGrainSize,
            fragment_q,
            fragment_max_fraction
        );

        return;
    }


    // =========================================================================
    // Apply shattering
    // =========================================================================
    //
    // THIS IS THE ONLY PHYSICAL PARTICLE PROPERTY WE MODIFY.
    //

    Sp->DustP[dust_idx].
        GrainRadius =
        a_new;


    // =========================================================================
    // Explicit conservation checks
    // =========================================================================
    //
    // There is no setMass() call anywhere in this shattering path.
    // These checks are primarily here to make accidental future changes obvious.
    //

    double mass_after =
        Sp->P[dust_idx].
        getMass();

    double CF_after =
        Sp->DustP[dust_idx].
        CarbonMassFraction;

    double mass_scale =
        std::max(
            fabs(mass_before),
            1.0e-30
        );

    double relative_mass_error =
        fabs(
            mass_after -
            mass_before
        ) /
        mass_scale;

    if(relative_mass_error >
       1.0e-12)
    {
        Terminate(
            "Shattering changed dust mass: "
            "before=%g after=%g relerr=%g",
            mass_before,
            mass_after,
            relative_mass_error
        );
    }

    if(fabs(CF_after -
            CF_before) >
       1.0e-7)
    {
        Terminate(
            "Shattering changed CarbonMassFraction: "
            "before=%g after=%g",
            CF_before,
            CF_after
        );
    }


    // =========================================================================
    // Diagnostics / statistics
    // =========================================================================

    shat_passed++;

    NShatteringEvents++;

    TotalSizeReductionShattering +=
        (
            a -
            a_new
        );

    sum_parent_radius +=
        a;

    sum_fragment_radius +=
        a_new;


    // -------------------------------------------------------------------------
    // Fragment-size bins
    // -------------------------------------------------------------------------

    if(a_new < 10.0)
    {
        frag_1_10++;
        frag_mass_1_10 += M_dust;
    }
    else if(a_new < 30.0)
    {
        frag_10_30++;
        frag_mass_10_30 += M_dust;
    }
    else if(a_new < 50.0)
    {
        frag_30_50++;
        frag_mass_30_50 += M_dust;
    }
    else if(a_new < 70.0)
    {
        frag_50_70++;
        frag_mass_50_70 += M_dust;
    }
    else if(a_new < 90.0)
    {
        frag_70_90++;
        frag_mass_70_90 += M_dust;
    }
    else
    {
        frag_ge90++;
        frag_mass_ge90 += M_dust;
    }


    // =========================================================================
    // Event log
    // =========================================================================

    if(All.ThisTask == 0 &&
       (
           NShatteringEvents <= 100 ||
           NShatteringEvents % 10000 == 0
       ))
    {
        DUST_PRINT(
            "[SHATTERING] "
            "event=%lld "
            "a_nm_old=%.2f "
            "a_nm_new=%.2f "
            "ratio=%.3f "
            "q=%.2f "
            "amaxfrac=%.2f "
            "mass_before=%.6e "
            "mass_after=%.6e "
            "CF=%.4f "
            "n_H_cm3=%.3e "
            "n_eff_cm3=%.3e "
            "T_gas_K=%.0f "
            "v_grain_kms=%.2f "
            "tau_shatter_Myr=%.2f "
            "shatter_probability=%.3e\n",

            NShatteringEvents,

            a,
            a_new,
            a_new / a,

            fragment_q,
            fragment_max_fraction,

            mass_before,
            mass_after,

            CF_before,

            n_H,
            n_eff,
            T_gas,

            v_grain_kms,

            tau_shat_yr /
            1.0e6,

            P_shatter
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// erode_dust_grain_thermal
// ═══════════════════════════════════════════════════════════════════════════════
/**
 * Erode or destroy a dust grain via thermal sputtering.
 *
 * Uses the McKinnon et al. (2017) sputtering timescale (Eq. 2):
 *   τ_sp = 0.17 Gyr × (a_-1 / ρ_-27) × [(T0/T)^ω + 1]
 * where a_-1 = a / 0.1 µm, ρ_-27 = ρ_gas / (1e-27 g/cm³), T0 = 2e6 K, ω = 2.5.
 *
 * COMPOSITION EVOLUTION: sputtering is now applied to the grain's two
 * chemical species SEPARATELY rather than as one pre-blended composite rate
 * applied uniformly. Carbon sputters faster (lower binding energy, 4 eV vs
 * 6 eV silicate), so the carbon-rich portion of the grain's mass erodes away
 * faster than the silicate portion -- surviving grains drift toward more
 * silicate-dominated composition over repeated sputtering episodes. The
 * grain's radius still represents one physical geometric size (derived from
 * the new TOTAL surviving mass via a ∝ m^(1/3)); only the internal species
 * split, and hence CarbonMassFraction, evolves independently of radius.
 *
 * Returns 1 if the particle was fully destroyed, 0 otherwise.
 */
int erode_dust_grain_thermal(simparticles *Sp, int dust_idx, int nearest_gas_input,
                              double T_gas, double dt)
{
  if(!All.DustEnableSputtering) return 0;
  double a = Sp->DustP[dust_idx].GrainRadius;
  // ── Sublimation: grain temperature exceeds material melting point ─────────
  // Unchanged -- still a CF-blended threshold, not a mass/composition change.
  double T_sublimate = 1500.0 + 500.0 * Sp->DustP[dust_idx].CarbonMassFraction;
  if(Sp->DustP[dust_idx].DustTemperature > T_sublimate) {
    DUST_PRINT("[SUBLIMATION] Dust destroyed at T_dust=%.0f K\n",
               Sp->DustP[dust_idx].DustTemperature);
    log_dust_particle_event(Sp, dust_idx, nearest_gas_input, DUST_EVENT_SUBLIMATION);
    return destroy_dust_particle_to_gas(Sp, dust_idx, nearest_gas_input,
                                        &NDustDestroyedByThermal,
                                        &TotalMassDestroyedByThermal);
  }
  // ── Below sputtering threshold: no action ────────────────────────────────
  if(T_gas < All.DustThermalSputteringTemp) return 0;
  // ── McKinnon+2017 sputtering timescale ───────────────────────────────────
  const double T0    = 2e6;   // K, characteristic sputtering temperature
  const double omega = 2.5;   // power-law index (Tielens et al. 1994)
  double a_cgs       = a * 1e-7;            // nm → cm
  double a_minus1    = a_cgs / 1e-5;        // units of 0.1 µm
  double rho_cgs     = dust_gas_density_cgs(Sp, nearest_gas_input);
  double rho_minus27 = rho_cgs / 1e-27;
  // Guard against zero/unphysical gas density
  if(rho_minus27 <= 0.0 || !isfinite(rho_minus27))
    rho_minus27 = 1e-4 / 1e-27 * PROTONMASS / 1e-27;  // safe fallback
  double tau_base_yr = 0.17e9 * (a_minus1 / rho_minus27) * (pow(T0 / T_gas, omega) + 1.0);
  // ── Species-resolved composition factors ─────────────────────────────────
  // Same 6 eV (silicate) / 4 eV (carbon) binding energies as before, but no
  // longer pre-blended into one CF-weighted composite_factor -- each species
  // gets its own timescale, applied to its own share of the mass below.
  double composition_factor_carbon   = 6.0 / 4.0;  // = 1.5
  double composition_factor_silicate = 6.0 / 6.0;  // = 1.0
  double tau_sputter_carbon_yr   = tau_base_yr / composition_factor_carbon;
  double tau_sputter_silicate_yr = tau_base_yr / composition_factor_silicate;
  // Clamp each to the same physically motivated range as before
  tau_sputter_carbon_yr   = std::max(1e6, std::min(1e9, tau_sputter_carbon_yr));
  tau_sputter_silicate_yr = std::max(1e6, std::min(1e9, tau_sputter_silicate_yr));
  double tau_sputter_carbon   = tau_sputter_carbon_yr   * SEC_PER_YEAR / dust_code_time_in_s();
  double tau_sputter_silicate = tau_sputter_silicate_yr * SEC_PER_YEAR / dust_code_time_in_s();
  // ── Split current mass into species, erode each independently ────────────
double CF_old         = std::max(0.0, std::min(1.0, (double)Sp->DustP[dust_idx].CarbonMassFraction));
double old_mass       = Sp->P[dust_idx].getMass();
double M_carbon_old   = old_mass * CF_old;
double M_silicate_old = old_mass * (1.0 - CF_old);
// Exact exponential sputtering for each component.
double M_carbon_new = M_carbon_old * exp(-dt / tau_sputter_carbon);
double M_silicate_new = M_silicate_old * exp(-dt / tau_sputter_silicate);
double new_mass = M_carbon_new + M_silicate_new;
// Actual species masses returned to the gas.
double carbon_lost = M_carbon_old - M_carbon_new;
double silicate_lost = M_silicate_old - M_silicate_new;
double mass_lost = carbon_lost + silicate_lost;
// Numerical protection.
carbon_lost = std::max(0.0, carbon_lost);
silicate_lost = std::max(0.0, silicate_lost);
// Radius follows total surviving mass.
double a_new = a * cbrt(new_mass / old_mass);
// ── Full destruction ─────────────────────────────────────────────────────
//
// IMPORTANT:
// Do NOT return carbon_lost/silicate_lost here first.
//
// destroy_dust_particle_to_gas() returns the ENTIRE original surviving
// dust particle to the gas, with the composition stored in CarbonMassFraction.
// Returning the partial losses first would double count material.
if(a_new <= All.DustMinGrainSize ||
   !isfinite(a_new) ||
   new_mass <= 0.0)
{
    log_dust_particle_event(
        Sp,
        dust_idx,
        nearest_gas_input,
        DUST_EVENT_THERMAL);
    return destroy_dust_particle_to_gas(
        Sp,
        dust_idx,
        nearest_gas_input,
        &NDustDestroyedByThermal,
        &TotalMassDestroyedByThermal);
}
// ── Partial erosion ──────────────────────────────────────────────────────
// New surviving composition.
double CF_new = M_carbon_new / new_mass;
// Update the dust particle itself.
Sp->DustP[dust_idx].GrainRadius = a_new;
Sp->DustP[dust_idx].CarbonMassFraction = (float)CF_new;
Sp->P[dust_idx].setMass(
    new_mass);
// Return exactly the sputtered species to the gas.
if(nearest_gas_input >= 0)
{
    return_dust_material_to_gas(
        Sp,
        nearest_gas_input,
        carbon_lost,
        silicate_lost);
}
// Diagnostics / accounting.
LocalDustMassChange -= mass_lost;
NGrainErosionEvents++;
TotalMassErodedByThermal += mass_lost;
static int erosion_count = 0;
erosion_count++;
if(erosion_count % 10000 == 0 &&
   All.ThisTask == 0)
{
    DUST_PRINT(
        "[EROSION] Grain shrunk: %.2f -> %.2f nm  "
        "CF: %.4f -> %.4f  "
        "dmC=%.2e dmSil=%.2e dmTot=%.2e T=%.0f K\n",
        a,
        a_new,
        CF_old,
        CF_new,
        carbon_lost,
        silicate_lost,
        mass_lost,
        T_gas);
}
return 0;
}
// ═══════════════════════════════════════════════════════════════════════════════
// destroy_dust_particle_to_gas
// ═══════════════════════════════════════════════════════════════════════════════
/**
 * Destroy a dust particle and return its mass to the nearest gas cell.
 *
 * This is the single destruction path for all physics-driven dust removal
 * (thermal sputtering, shock destruction). Astration uses a separate path
 * that does proportional mass removal rather than full destruction.
 *
 * The particle is converted to type 3 with zero mass and ID so it will be
 * swept up by destroy_dust_particles() on the next call.
 *
 * Returns 1 to signal the caller that the particle was destroyed.
 */
static int destroy_dust_particle_to_gas(simparticles *Sp, int dust_idx,
                                         int nearest_gas, long long *counter,
                                         double *mass_counter)
{
  if(nearest_gas < 0)
    {
        DUST_PRINT("[DESTRUCTION_SKIP] No gas particle available for dust_idx=%d\n",
                  dust_idx);
        return 0;
    }
  double dust_mass = Sp->P[dust_idx].getMass();
  double CF = std::max(0.0, std::min(1.0, (double)Sp->DustP[dust_idx].CarbonMassFraction));
  double M_carbon = dust_mass * CF;
  double M_silicate = dust_mass * (1.0 - CF);
  if(nearest_gas >= 0)
  {
      return_dust_material_to_gas(
          Sp,
          nearest_gas,
          M_carbon,
          M_silicate);
  }
  Sp->P[dust_idx].setMass(DUST_MASS_TO_DESTROY);
  Sp->P[dust_idx].setType(3);
  Sp->P[dust_idx].ID.set(0);
  memset(&Sp->DustP[dust_idx], 0, sizeof(dust_data));
  Sp->DustP[dust_idx].GrainRadius = All.DustMinGrainSize;
  LocalDustMassChange -= dust_mass;
  LocalDustDestroyedThisStep++;
  /* The particle has changed type and must be removed from the dust/gravity
   * time-bin bookkeeping before the next force calculation or snapshot. */
  DustNeedsSynchronization = 1;
  if(counter)      (*counter)++;
  if(mass_counter) (*mass_counter) += dust_mass;
  return 1;
}
// ═══════════════════════════════════════════════════════════════════════════════
// calculate_sedov_velocity_from_radius
// ═══════════════════════════════════════════════════════════════════════════════
/**
 * Compute the Sedov-Taylor blast-wave velocity at a given physical radius.
 *
 * From the self-similar solution (Sedov 1959):
 *   v = (2/5) × ξ^(5/2) × √(E/ρ) × R^(−3/2)
 *
 * where ξ = 1.033 is the Sedov dimensionless constant.
 * Returns velocity in km/s.
 */
double calculate_sedov_velocity_from_radius(double radius_kpc, double rho_cgs, double energy_erg)
{
  const double xi = 1.033;
  double R_cm      = radius_kpc * 1000.0 * PARSEC;
  double xi_factor = pow(xi, 2.5);
  double v_cm_s    = (2.0/5.0) * xi_factor * sqrt(energy_erg / rho_cgs) * pow(R_cm, -1.5);
  return v_cm_s / 1e5;  // km/s
}
// ═══════════════════════════════════════════════════════════════════════════════
// calculate_drag_timescale
// ═══════════════════════════════════════════════════════════════════════════════
/**
 * Compute the Epstein drag stopping timescale for a dust grain.
 *
 * Implements McKinnon et al. (2018), equations 8–9.
 *
 * t_stop (subsonic) = √(π γ / 8) × (a ρ_grain) / (ρ_gas c_s)
 *
 * Supersonic correction: multiply by 1 / √(1 + 9π M² / 128), where M
 * is the grain–gas Mach number. Active for Mach > 0.1.
 *
 * Returns the stopping time in Myr, clamped to [0.001, 50] Myr.
 * The upper clamp prevents dust from acquiring unrealistically long stopping
 * times in the CGM that would then place it on very short gravity timebins
 * via the drag velocity kick. For the terminal-velocity correction in
 * dust_radiation_pressure, the unclamped t_stop is computed separately.
 */
static double calculate_drag_timescale(double grain_radius_nm, double grain_density,
                                       double gas_density_cgs, double gas_temperature,
                                       double relative_velocity_cgs, double mu_gas,
                                       double gamma_gas)
{
  const double k_B      = 1.38064852e-16;  // erg/K
  const double m_p      = 1.6726219e-24;   // g
  const double s_per_myr= 3.15576e13;      // s/Myr
  double a_cm = grain_radius_nm * 1e-7;
  double c_s  = sqrt(gamma_gas * k_B * gas_temperature / (mu_gas * m_p));
  // Sanity guard: cannot compute a meaningful t_stop in vacuum or near-vacuum
  if(c_s < 1e3 || gas_density_cgs < 1e-30) return 50.0;
  double t_stop_sub = (sqrt(M_PI * gamma_gas) * a_cm * grain_density)
                      / (2.0 * sqrt(2.0) * gas_density_cgs * c_s);
  double mach = relative_velocity_cgs / c_s;
  double supersonic_factor = 1.0;
  if(mach > 0.1)
    supersonic_factor = 1.0 / sqrt(1.0 + (9.0 * M_PI / 128.0) * mach * mach);
  double t_stop_myr = (t_stop_sub * supersonic_factor) / s_per_myr;
  return std::max(0.001, std::min(50.0, t_stop_myr));
}
// ═══════════════════════════════════════════════════════════════════════════════
// dust_gas_interaction
// ═══════════════════════════════════════════════════════════════════════════════
/**
 * Apply Epstein drag and thermal sputtering to a dust–gas pair.
 *
 * DRAG (Step 4): Velocity is updated analytically — the exact exponential
 * solution (Δv = v_rel × (1 − exp(−dt/t_stop))) is used rather than a linear
 * approximation, so the result is stable for any ratio of dt/t_stop.  The
 * caller supplies the actual physical elapsed interval in code-time units.
 *
 * SPUTTERING (Step 5): Delegated to erode_dust_grain_thermal(), which also
 * uses the exact exponential form.
 *
 * Returns 1 if the dust particle was destroyed by sputtering, 0 otherwise.
 */
int dust_gas_interaction(simparticles *Sp, int dust_idx, int nearest_gas, double dt,
                         double T_gas)
{
  if(!All.DustEnableDrag) return 0;
  if(nearest_gas < 0) return 0;
  // ── Guard: stale hash returning a converted non-gas particle ─────────────
  // Reading SphP[] for a non-gas particle is undefined behaviour and can
  // produce garbage T_gas values that spuriously trigger sputtering.
  if(Sp->P[nearest_gas].getType() != 0) {
    static long long bad_gas_warns = 0;
    bad_gas_warns++;
    NDustDestroyedByBadGasIndex++;
    if(bad_gas_warns <= 50)
      printf("[BAD_GAS_IDX|T=%d|Step=%d] dust_idx=%d nearest_gas=%d "
             "has type=%d (expected 0!), mass=%.3e, ID=%lld "
             "| RunningBadCount=%lld\n",
             All.ThisTask, All.NumCurrentTiStep,
             dust_idx, nearest_gas,
             Sp->P[nearest_gas].getType(), Sp->P[nearest_gas].getMass(),
             (long long)Sp->P[nearest_gas].ID.get(), bad_gas_warns);
    return 0;
  }
  // ── Step 1–2: Gas properties ──────────────────────────────────────────────
  double gas_vel[3]     = { Sp->P[nearest_gas].Vel[0],
                             Sp->P[nearest_gas].Vel[1],
                             Sp->P[nearest_gas].Vel[2] };
  double gas_density_cgs= dust_gas_density_cgs(Sp, nearest_gas);
  double n_H            = dust_gas_nH_cgs(Sp, nearest_gas);
  // ── Step 3: Relative velocity ─────────────────────────────────────────────
  double vrel_x = Sp->P[dust_idx].Vel[0] - gas_vel[0];
  double vrel_y = Sp->P[dust_idx].Vel[1] - gas_vel[1];
  double vrel_z = Sp->P[dust_idx].Vel[2] - gas_vel[2];
  // P[].Vel is p = a^2 dx/dt; the physical peculiar velocity difference is
  // dp / a.  Only the magnitude needs converting: the exponential relaxation
  // below acts on p directly and is unit-independent.
  double vrel   = sqrt(vrel_x*vrel_x + vrel_y*vrel_y + vrel_z*vrel_z) * All.cf_ainv;
  double vrel_cgs = vrel * All.UnitVelocity_in_cm_per_s;
  // ── Step 4: Epstein drag timescale ───────────────────────────────────────
  double drag_timescale_myr = calculate_drag_timescale(
      Sp->DustP[dust_idx].GrainRadius, 2.4,
      gas_density_cgs, T_gas, vrel_cgs, 0.6, 5.0/3.0);
  double drag_timescale = drag_timescale_myr * 1e6 * SEC_PER_YEAR / dust_code_time_in_s();
  // ── Step 5: Apply drag — exact analytical solution ───────────────────────
  // Δv_k = (v_gas,k − v_dust,k) × (1 − exp(−dt/t_stop))
  // This is exact for any dt/t_stop.  dt is the physical elapsed interval in
  // internal code-time units, converted by the caller from the cosmological
  // scale-factor step.
  double drag_factor = 1.0 - exp(-dt / drag_timescale);
  for(int k = 0; k < 3; k++)
    Sp->P[dust_idx].Vel[k] += drag_factor * (gas_vel[k] - Sp->P[dust_idx].Vel[k]);
  // ── Step 6: Thermal sputtering ────────────────────────────────────────────
  if(T_gas > All.DustThermalSputteringTemp) {
    int destroyed = erode_dust_grain_thermal(Sp, dust_idx, nearest_gas, T_gas, dt);
    return destroyed;
  }
  // ── Step 7: Occasional diagnostic sample (1% of particles, first 500) ────
  if(All.ThisTask == 0 && Sp->P[dust_idx].ID.get() % 100 == 0) {
    static int drag_samples = 0;
    if(drag_samples < 500) {
      double vel_kms    = sqrt(Sp->P[dust_idx].Vel[0]*Sp->P[dust_idx].Vel[0] +
                               Sp->P[dust_idx].Vel[1]*Sp->P[dust_idx].Vel[1] +
                               Sp->P[dust_idx].Vel[2]*Sp->P[dust_idx].Vel[2])
                          * All.UnitVelocity_in_cm_per_s / 1e5;
      double cs_cgs     = sqrt((5.0/3.0) * BOLTZMANN * T_gas / (0.6 * PROTONMASS));
      double mach       = vrel_cgs / cs_cgs;
      double dt_myr     = dt * dust_code_time_in_s() / (1e6 * SEC_PER_YEAR);
      DUST_PRINT("[DUST_DRAG] vel=%.1f km/s Δv=%.1f km/s "
                 "nH=%.3e cm^-3 T=%.1e K Mach=%.2f "
                 "t_drag=%.2f Myr dt=%.2f Myr f_drag=%.3f "
                 "a=%.2f nm\n",
                 vel_kms, vrel_cgs / 1e5,
                 n_H, T_gas, mach,
                 drag_timescale_myr, dt_myr, drag_factor,
                 Sp->DustP[dust_idx].GrainRadius);
      drag_samples++;
    }
  }
  return 0;
}
/**
 * Spawn dust superparticles from a stellar feedback event.
 *
 * carbon_dust_mass and silicate_dust_mass are the already-determined
 * stellar-condensed dust masses in code mass units. Elemental partitioning
 * between gas-phase ejecta and dust is performed by feedback.cc before this
 * routine is called.
 *
 * This routine only spawns PartType6 particles; it does not modify gas.
 *
 * feedback_type: 1 = SNII, 2 = AGB, 3 = LRN
 */
double create_dust_particles_from_feedback(
    simparticles *Sp,
    int star_idx,
    double carbon_dust_mass,
    double silicate_dust_mass,
    int feedback_type,
    int particle_count_override,
    bool conserve_agb_momentum)
{
  if(!All.DustEnableCreation)
    return 0.0;

if(carbon_dust_mass < 0.0 ||
   silicate_dust_mass < 0.0)
    Terminate("Negative stellar dust component");

double total_dust_mass =
    carbon_dust_mass + silicate_dust_mass;

if(!std::isfinite(total_dust_mass) ||
   total_dust_mass < MIN_DUST_PARTICLE_MASS)
    return 0.0;

double carbon_mass_fraction =
    carbon_dust_mass / total_dust_mass;

carbon_mass_fraction =
    std::max(0.0,
             std::min(1.0, carbon_mass_fraction));

  double velocity_scale;
  int n_dust_particles;

if(feedback_type == 1)
{
    velocity_scale =
        All.DustVelocitySNII;

    n_dust_particles =
        All.DustParticlesPerSNII;
}
else if(feedback_type == 2)
{
    velocity_scale =
        All.DustVelocityAGB;

    n_dust_particles =
        All.DustParticlesPerAGB;
}
else if(feedback_type == 3)
{
    velocity_scale =
        All.DustVelocityLRN;

    n_dust_particles =
        All.DustParticlesPerLRN;
}
else
    return 0.0;

if(particle_count_override >= 0)
    n_dust_particles = particle_count_override;

if(n_dust_particles <= 0)
    return 0.0;


// ── Choose how many superparticles should represent the dust mass ──────────
const double min_particle_mass = 1e-15;

int n_create = n_dust_particles;

if(total_dust_mass / n_create < min_particle_mass)
{
    n_create = (int)(total_dust_mass / min_particle_mass);

    if(n_create < 1)
        n_create = 1;
}

if(Sp->NumPart + n_create > Sp->MaxPart)
{
    Terminate(
        "Insufficient particle slots for stellar dust creation: "
        "need=%d available=%d",
        n_create,
        Sp->MaxPart - Sp->NumPart);
}

double dust_mass_per_particle =
    total_dust_mass / n_create;

// ── Spawn ─────────────────────────────────────────────────────────────────
int n_created = 0;
double star_mass_remaining = Sp->P[star_idx].getMass();

for(int n = 0; n < n_create; n++) {

  double theta = acos(2.0 * get_random_number() - 1.0);
  double phi   = 2.0 * M_PI * get_random_number();

  double offset_min =
      (feedback_type == 1) ? All.DustOffsetMinSNII :
      (feedback_type == 2) ? All.DustOffsetMinAGB :
                             All.DustOffsetMinLRN;

  double offset_max =
      (feedback_type == 1) ? All.DustOffsetMaxSNII :
      (feedback_type == 2) ? All.DustOffsetMaxAGB :
                             All.DustOffsetMaxLRN;

  double r =
      offset_min +
      (offset_max - offset_min) * get_random_number();

  double offset_kpc[3] = {
      r * sin(theta) * cos(phi),
      r * sin(theta) * sin(phi),
      r * cos(theta)
  };

  // P.Vel is a * physical peculiar velocity for comoving integration.
  // Apply the corrected conversion only for the new AGB transport path.
  // P[].Vel holds p = a^2 dx/dt, so a physical peculiar kick v enters as a*v.
  // Applying the factor to every source keeps DustVelocity* in physical km/s
  // at all redshifts (SNII and LRN kicks were previously 1/a too large).
  double kick_scale = All.cf_atime;
  double initial_velocity[3];

  initial_velocity[0] =
      Sp->P[star_idx].Vel[0] +
      kick_scale * velocity_scale * sin(theta) * cos(phi)
      / All.UnitVelocity_in_cm_per_s * 1e5;

  initial_velocity[1] =
      Sp->P[star_idx].Vel[1] +
      kick_scale * velocity_scale * sin(theta) * sin(phi)
      / All.UnitVelocity_in_cm_per_s * 1e5;

  initial_velocity[2] =
      Sp->P[star_idx].Vel[2] +
      kick_scale * velocity_scale * cos(theta)
      / All.UnitVelocity_in_cm_per_s * 1e5;

  spawn_dust_particle(
      Sp, offset_kpc,
      dust_mass_per_particle,
      initial_velocity,
      star_idx,
      feedback_type);

  n_created++;

  int new_idx = Sp->NumPart - 1;

  if(feedback_type == 1) {
    Sp->DustP[new_idx].GrainRadius =
        sample_initial_grain_radius(DustSNIIInitialRadius,
                                    DustSNIISizeSigmaDex);
    Sp->DustP[new_idx].DustSource  = 0;
  } else if(feedback_type == 2) {
    Sp->DustP[new_idx].GrainRadius =
        sample_initial_grain_radius(DustAGBInitialRadius,
                                    DustAGBSizeSigmaDex);
    Sp->DustP[new_idx].DustSource  = 1;
  } else {
    Sp->DustP[new_idx].GrainRadius = 10.0;
    Sp->DustP[new_idx].DustSource  = 2;
  }

  Sp->DustP[new_idx].CarbonMassFraction =
      (float)carbon_mass_fraction;

  if(conserve_agb_momentum)
    {
      const double star_mass_after = star_mass_remaining - dust_mass_per_particle;
      if(star_mass_after <= 0.0 || !std::isfinite(star_mass_after))
        Terminate("AGB dust recoil would exhaust stellar mass");
      for(int k = 0; k < 3; ++k) {
        const double star_vel_before = Sp->P[star_idx].Vel[k];
        const double dust_vel = Sp->P[new_idx].Vel[k];
        Sp->P[star_idx].Vel[k] =
            (star_mass_remaining * star_vel_before - dust_mass_per_particle * dust_vel) /
            star_mass_after;
      }
      star_mass_remaining = star_mass_after;
    }

}

if(n_created != n_create)
  Terminate("Dust creation count mismatch: created=%d expected=%d",
            n_created, n_create);


// ── Diagnostics/accounting ────────────────────────────────────────────────
if(feedback_type == 1)
  NDustCreatedBySNII += n_created;
else if(feedback_type == 2)
  NDustCreatedByAGB += n_created;
else
  NDustCreatedByLRN += n_created;

LocalDustCreatedThisStep += n_created;
LocalDustMassChange      += total_dust_mass;
DustNeedsSynchronization  = 1;

if(All.ThisTask == 0) {
    static int velocity_samples = 0;

    if(velocity_samples < 50) {
        int new_idx = Sp->NumPart - 1;

        if(new_idx >= 0 &&
           Sp->P[new_idx].getType() == DUST_PARTICLE_TYPE) {

            double vel_mag =
                sqrt(Sp->P[new_idx].Vel[0] * Sp->P[new_idx].Vel[0] +
                     Sp->P[new_idx].Vel[1] * Sp->P[new_idx].Vel[1] +
                     Sp->P[new_idx].Vel[2] * Sp->P[new_idx].Vel[2])
                * All.UnitVelocity_in_cm_per_s / 1e5;

            double star_vel_mag =
                sqrt(Sp->P[star_idx].Vel[0] * Sp->P[star_idx].Vel[0] +
                     Sp->P[star_idx].Vel[1] * Sp->P[star_idx].Vel[1] +
                     Sp->P[star_idx].Vel[2] * Sp->P[star_idx].Vel[2])
                * All.UnitVelocity_in_cm_per_s / 1e5;

            DUST_PRINT(
                "[DUST_CREATE] vel_dust=%.1f km/s "
                "vel_star=%.1f km/s "
                "grain_r=%.2f nm CF=%.4f "
                "feedback_type=%d\n",
                vel_mag,
                star_vel_mag,
                Sp->DustP[new_idx].GrainRadius,
                Sp->DustP[new_idx].CarbonMassFraction,
                feedback_type);

            velocity_samples++;
        }
    }
}

return total_dust_mass;
}

// ═══════════════════════════════════════════════════════════════════════════════
// dust_global_synchronization
// ═══════════════════════════════════════════════════════════════════════════════
/**
 * Update global dust counters and redistribute the particle ID block across tasks.
 *
 * This is called after each dust-physics update, not at every global
 * synchronization point when a reduced dust cadence is selected.
 *
 * ID BLOCK SCHEME: Between calls, each task independently
 * increments All.MaxID as it spawns dust particles. To prevent inter-task ID
 * collisions, each task is assigned an exclusive block of size block_size
 * after every sync. block_size must exceed the maximum number of dust particles
 * any single task can create between syncs — if the overflow warning in
 * spawn_dust_particle fires, increase it.
 */
void dust_global_synchronization(simparticles *Sp, MPI_Comm Communicator,
                                  long long dust_created, long long dust_destroyed,
                                  double dust_mass_change)
{
  NDustCreated  += dust_created;
  NDustDestroyed+= dust_destroyed;
  TotalDustMass += dust_mass_change;
  LocalDustCreatedThisStep    = 0;
  LocalDustDestroyedThisStep  = 0;
  LocalDustMassChange         = 0.0;
  /* Do not clear DustNeedsSynchronization here.  The main run loop owns
   * particle compaction/time-bin reconstruction and clears the flag only
   * after that collective operation has completed. */
  // Find the global maximum ID across all tasks
  MyIDType local_max  = All.MaxID;
  MyIDType global_max = 0;
  MPI_Allreduce(&local_max, &global_max, 1, MPI_MYIDTYPE, MPI_MAX, Communicator);
  // Assign each task a non-overlapping ID block for the next sync interval
  const MyIDType block_size = 200000;
  All.MaxID         = global_max + 1 + (MyIDType)All.ThisTask * block_size;
  All.MaxIDBlockEnd = All.MaxID + block_size - 1;
}
// ═══════════════════════════════════════════════════════════════════════════════
// spawn_dust_particle
// ═══════════════════════════════════════════════════════════════════════════════
// Spawn bin: max(DUST_MIN_TIMEBIN, HighestActiveTimeBin).
// At late times (z<5), HighestActiveTimeBin is typically ≤15 so
// DUST_MIN_TIMEBIN dominates — same behavior as before.
// At early times (z>10), HighestActiveTimeBin can be 21+, so without
// this clamp dust spawns on bin 15 which is unsynchronized in the
// current hierarchy, causing a collective gravity hang on multi-node runs.
void spawn_dust_particle(simparticles *Sp, double offset_kpc[3], double dust_mass,
                          double initial_velocity[3], int star_idx, int feedback_type)
{
  if(Sp->NumPart >= Sp->MaxPart) {
    static int warning_count = 0;
    if(warning_count < 10 && All.ThisTask == 0)
      printf("[DUST_ERROR] T=%d: Cannot create dust — array full (NumPart=%d MaxPart=%d)\n",
             All.ThisTask, Sp->NumPart, Sp->MaxPart);
    warning_count++;
    return;
  }
  int new_idx = Sp->NumPart;
  // ── Position: parent star + random offset ─────────────────────────────────
  double star_pos[3];
  Sp->intpos_to_pos(Sp->P[star_idx].IntPos, star_pos);
  // Store birth position (= star position, ignoring offset) for tracking
  // dust transport relative to its progenitor's location.
  Sp->DustP[new_idx].BirthPos[0] = star_pos[0];
  Sp->DustP[new_idx].BirthPos[1] = star_pos[1];
  Sp->DustP[new_idx].BirthPos[2] = star_pos[2];
  double dust_pos[3];
  for(int d = 0; d < 3; d++) {
    dust_pos[d] = star_pos[d] + offset_kpc[d];
    // Periodic wrap
    while(dust_pos[d] <  0.0)          dust_pos[d] += All.BoxSize;
    while(dust_pos[d] >= All.BoxSize)  dust_pos[d] -= All.BoxSize;
  }
  Sp->pos_to_intpos(dust_pos, Sp->P[new_idx].IntPos);
  // Small integer jitter prevents tree crashes when multiple dust particles
  // spawn at nearly identical IntPos from the same star event. Sub-resolution
  // jitter has no physical effect on dynamics.
  Sp->P[new_idx].IntPos[0] += (MySignedIntPosType)((int)(get_random_number() * 4) - 2);
  Sp->P[new_idx].IntPos[1] += (MySignedIntPosType)((int)(get_random_number() * 4) - 2);
  Sp->P[new_idx].IntPos[2] += (MySignedIntPosType)((int)(get_random_number() * 4) - 2);
  // ── Particle properties ───────────────────────────────────────────────────
  Sp->P[new_idx].setType(DUST_PARTICLE_TYPE);
  Sp->P[new_idx].setMass(dust_mass);
  Sp->P[new_idx].Metallicity = 1.0;  // dust is pure metal
  Sp->P[new_idx].Vel[0] = initial_velocity[0];
  Sp->P[new_idx].Vel[1] = initial_velocity[1];
  Sp->P[new_idx].Vel[2] = initial_velocity[2];
  // ── ID assignment from per-task block ─────────────────────────────────────
  if(All.MaxIDBlockEnd > 0 && All.MaxID >= All.MaxIDBlockEnd)
    printf("[DUST_ID_OVERFLOW|T=%d|Step=%d] ID block exhausted! "
           "MaxID=%lld BlockEnd=%lld — increase block_size in dust_global_synchronization\n",
           All.ThisTask, All.NumCurrentTiStep,
           (long long)All.MaxID, (long long)All.MaxIDBlockEnd);
  Sp->P[new_idx].ID.set(All.MaxID + 1);
  All.MaxID++;
  // ── Softening class: use dust type if defined, else inherit from stars ────
  if(DUST_PARTICLE_TYPE < NTYPES && All.SofteningClassOfPartType[DUST_PARTICLE_TYPE] >= 0)
    Sp->P[new_idx].setSofteningClass(All.SofteningClassOfPartType[DUST_PARTICLE_TYPE]);
  else
    Sp->P[new_idx].setSofteningClass(All.SofteningClassOfPartType[4]);
  // ── Placeholders — overwritten in create_dust_particles_from_feedback ─────
  Sp->DustP[new_idx].GrainRadius    = All.DustMinGrainSize;
  Sp->DustP[new_idx].CarbonMassFraction = 0.3;
  Sp->DustP[new_idx].DustSource      = 2;
  // Initial dust temperature: CMB floor (T_CMB ∝ (1+z))
  Sp->DustP[new_idx].DustTemperature = 2.7 / All.Time;
  // No cached DTG yet: computed on this particle's first dust call.
  Sp->DustP[new_idx].LocalDTG = 0.0f;
  Sp->DustP[new_idx].DTGAge   = 0;
  Sp->P[new_idx].StellarAge  = All.Time;
  Sp->P[new_idx].Ti_Current  = All.Ti_Current;
  Sp->P[new_idx].TimeBinHydro = 0;  // dust does not participate in SPH hydro
  // Spawn at DUST_MIN_TIMEBIN floor — dust migrates to longer bins immediately
  // via get_timestep_grav returning TIMEBASE-1 (capped by Ti_nextoutput).
  // Spawning on the highest synchronized bin caused newly created dust to jump
  // over snapshot output times before get_timestep_grav could apply the cap.
  int dust_spawn_bin = std::max(DUST_MIN_TIMEBIN, (int)All.HighestActiveTimeBin);
  Sp->P[new_idx].TimeBinGrav = dust_spawn_bin;
  Sp->TimeBinsGravity.timebin_add_particle(new_idx, star_idx, dust_spawn_bin,
                                            Sp->TimeBinSynchronized[dust_spawn_bin]);
  // Sanity check — GrainRadius must be set by caller before use
  if(Sp->DustP[new_idx].GrainRadius <= 0.0 || !isfinite(Sp->DustP[new_idx].GrainRadius)) {
    DUST_PRINT("[SPAWN_BUG] GrainRadius invalid (%.3e) for new particle idx=%d — resetting\n",
               Sp->DustP[new_idx].GrainRadius, new_idx);
    Sp->DustP[new_idx].GrainRadius = All.DustMinGrainSize;
  }
  Sp->NumPart++;
  GlobalDustCount++;
}
// ═══════════════════════════════════════════════════════════════════════════════
// analyze_grain_size_distribution
// ═══════════════════════════════════════════════════════════════════════════════
/**
 * Print a histogram of grain sizes across 6 bins (task 0 only, local particles).
 * Called periodically from update_dust_dynamics diagnostics block.
 */
void analyze_grain_size_distribution(simparticles *Sp)
{
  if(All.ThisTask != 0) return;
  const int    NBINS      = 6;
  double bin_edges[NBINS+1] = {0.0, 10.0, 50.0, 100.0, 150.0, 200.0, 500.0};
  int    bin_counts[NBINS]  = {0};
  double bin_masses[NBINS]  = {0.0};
  int    total_grains       = 0;
  double total_mass         = 0.0;
  for(int i = 0; i < Sp->NumPart; i++) {
    if(Sp->P[i].getType() != DUST_PARTICLE_TYPE ||
       Sp->P[i].getMass() <= DUST_MASS_TO_DESTROY) continue;
    double mass = Sp->P[i].getMass();
    total_grains++;
    total_mass += mass;
    for(int b = 0; b < NBINS; b++) {
      if(Sp->DustP[i].GrainRadius >= bin_edges[b] &&
         Sp->DustP[i].GrainRadius <  bin_edges[b+1]) {
        bin_counts[b]++;
        bin_masses[b] += mass;
        break;
      }
    }
  }
  if(total_grains == 0) return;
  DUST_PRINT("=== GRAIN SIZE DISTRIBUTION ===\n");
  DUST_PRINT("  Total: %d grains, %.3e Msun\n", total_grains, total_mass);
  for(int b = 0; b < NBINS; b++) {
    if(bin_counts[b] > 0)
      DUST_PRINT("  [%.0f-%.0f nm]: %d grains (%.1f%%), %.2e Msun (%.1f%%)\n",
                 bin_edges[b], bin_edges[b+1], bin_counts[b],
                 100.0*bin_counts[b]/total_grains, bin_masses[b],
                 100.0*bin_masses[b]/total_mass);
  }
  DUST_PRINT("================================\n");
}
// ═══════════════════════════════════════════════════════════════════════════════
// consume_dust_by_astration
// ═══════════════════════════════════════════════════════════════════════════════
/**
 * Remove dust mass proportional to the local dust-to-gas ratio when a star
 * forms (astration — dust is incorporated into the new star).
 *
 * The search radius is set to the gas cell radius (derived from density and
 * mass), capped at the SPH smoothing length. All dust within the radius is
 * weighted by 1/distance and consumed proportionally to match stellar_mass_formed
 * × local D/G ratio. Grains eroded below DUST_MASS_TO_DESTROY are fully removed.
 */
// ═══════════════════════════════════════════════════════════════════════════════
// consume_dust_by_astration
// ═══════════════════════════════════════════════════════════════════════════════
/**
 * Remove dust mass proportional to the local dust-to-gas ratio when a star
 * forms (astration — dust is incorporated into the new star).
 *
 * The search radius is the larger of the gas cell's own implied volume
 * (derived from density and mass) and its SPH smoothing length — previously
 * this was capped AT the smoothing length as an upper bound, which had no
 * effect in practice since the cell radius is almost always far smaller than
 * Hsml; the search radius therefore silently collapsed to the tiny cell
 * volume alone. Using Hsml as a floor instead gives astration access to the
 * same local ISM neighborhood used by growth, coagulation, and shattering
 * elsewhere in this file.
 *
 * All dust within the radius is weighted by 1/distance and consumed
 * proportionally to match stellar_mass_formed × local D/G ratio. Grains
 * eroded below DUST_MASS_TO_DESTROY are fully removed.
 */
void consume_dust_by_astration(simparticles *Sp, int gas_idx, double stellar_mass_formed,
                                int star_idx, double hsml)
{
  if(!All.DustEnableAstration) return;
  // Diagnostic counters (static: accumulated over full run on this task).
  // Track every attempt and where it dies, so the search-radius fix's actual
  // effect can be measured directly rather than inferred from event counts.
  static long long astration_attempts      = 0;
  static long long astration_zero_neighbor = 0;
  static long long astration_zero_dustmass = 0;
  static long long astration_overflow      = 0;
  astration_attempts++;
  double gas_mass  = Sp->P[gas_idx].getMass();
  double rho_code  = Sp->SphP[gas_idx].Density * All.cf_a3inv;
  double cell_radius = cbrt(3.0 * gas_mass / (4.0 * M_PI * rho_code));
  // FIX: search_radius now genuinely extends to the SPH smoothing length
  // rather than collapsing to the (much smaller) cell_radius alone.
  double search_radius = std::max(cell_radius, (double)Sp->SphP[gas_idx].Hsml);
  const int MAX_NEIGHBORS = 512;  // widened from 100 — the search volume is
                                   // now much larger, so more dust particles
                                   // may legitimately be found per event.
  int    neighbor_indices[MAX_NEIGHBORS];
  double neighbor_distances[MAX_NEIGHBORS];
  int    n_neighbors = 0;
  dust_hash.find_neighbors(Sp, gas_idx, search_radius,
                           neighbor_indices, neighbor_distances,
                           &n_neighbors, MAX_NEIGHBORS);
  if(n_neighbors == 0) { astration_zero_neighbor++; return; }
  if(n_neighbors == MAX_NEIGHBORS) {
    astration_overflow++;
    if(astration_overflow <= 20 && All.ThisTask == 0)
      DUST_PRINT("[ASTRATION_OVERFLOW] search may have found more than "
                 "%d dust neighbors; results may be truncated. "
                 "RunningTotal=%lld\n", MAX_NEIGHBORS, astration_overflow);
  }
  // Sum total dust mass within the search volume
  double total_dust_mass = 0.0;
  for(int i = 0; i < n_neighbors; i++) {
    int di = neighbor_indices[i];
    if(Sp->P[di].getType() == DUST_PARTICLE_TYPE)
      total_dust_mass += Sp->P[di].getMass();
  }
  if(total_dust_mass < DUST_MASS_TO_DESTROY) { astration_zero_dustmass++; return; }
  // Dust mass to consume = stellar mass formed × local D/G
  double local_DG       = total_dust_mass / gas_mass;
  // Scaled by a tunable calibration factor. Roughly calibrating by
  // destruction-timescale ratios from Jones & Nuth (2011)
  // (SN shocks ~0.3-0.6 Gyr, astration ~2-6 Gyr => astration should be
  // roughly 5-20% of the astration+shock destruction).
  double dust_to_consume = std::min(stellar_mass_formed * local_DG * All.DustAstrationCalibration,
                                      total_dust_mass);
  // Weight by 1/distance for proximity preference
  double weight_sum = 0.0;
  for(int i = 0; i < n_neighbors; i++)
    if(neighbor_distances[i] > 0)
      weight_sum += 1.0 / neighbor_distances[i];
  int    dust_consumed_count = 0;
  double dust_consumed_mass  = 0.0;
  for(int i = 0; i < n_neighbors; i++) {
    int di = neighbor_indices[i];
    if(Sp->P[di].getType() != DUST_PARTICLE_TYPE) continue;
    double weight          = (neighbor_distances[i] > 0)
                             ? (1.0 / neighbor_distances[i]) : 1.0;
    double this_fraction   = weight / weight_sum;
    double mass_loss       = dust_to_consume * this_fraction;
    double current_mass    = Sp->P[di].getMass();
    double new_mass        = current_mass - mass_loss;
    if(new_mass < DUST_MASS_TO_DESTROY) {
      log_dust_particle_event(Sp, di, -1, DUST_EVENT_ASTRATION);
      Sp->P[di].setMass(0.0);
      Sp->P[di].setType(3);
      Sp->P[di].ID.set(0);
      memset(&Sp->DustP[di], 0, sizeof(dust_data));
      dust_consumed_count++;
      dust_consumed_mass += current_mass;
    } else {
      Sp->P[di].setMass(new_mass);
      dust_consumed_mass += mass_loss;
    }
  }
  NDustDestroyedByAstration += dust_consumed_count;
  TotalDustMassAstrated     += dust_consumed_mass;  // stays in code units —
                                                     // print_dust_statistics()
                                                     // already converts this
                                                     // to Msun before printing.
  // Transfer consumed dust mass to the new star, update mass and metallicity
  if(star_idx >= 0 && dust_consumed_mass > 0) {
    double star_mass = Sp->P[star_idx].getMass();
    if(star_mass > 0) {
      double new_star_mass = star_mass + dust_consumed_mass;
      Sp->P[star_idx].setMass(new_star_mass);
      Sp->P[star_idx].Metallicity = (star_mass * Sp->P[star_idx].Metallicity + dust_consumed_mass) / new_star_mass;
    }
  }
  // ── Diagnostics ──────────────────────────────────────────────────────────
  static int astration_count = 0;
  astration_count++;
  if(astration_count <= 20)
    DUST_PRINT("[ASTRATION_CHECK|T=%d] #%d: "
               "hsml=%.3e search_r=%.3e cell_size=%.3e "
               "SF=%.3e gas_mass=%.3e D/G=%.3e "
               "dust_to_consume=%.3e n=%d consumed=%d mass=%.3e\n",
               All.ThisTask, astration_count,
               hsml, search_radius, dust_hash.cell_size,
               stellar_mass_formed, gas_mass, local_DG,
               dust_to_consume, n_neighbors, dust_consumed_count, dust_consumed_mass);
  if(astration_count % 100 == 0) {
    // FIX: dust_consumed_mass is in code mass units; convert before
    // printing, matching the mass_to_msun pattern used elsewhere in the file.
    double mass_to_msun = All.UnitMass_in_g / 1.989e33 / All.HubbleParam;
    DUST_PRINT("[ASTRATION] Event #%d: "
               "search_r=%.2e (hsml=%.2e cell=%.2e) "
               "D/G=%.3e consumed=%d (%.2e Msun)\n",
               astration_count,
               search_radius, hsml, dust_hash.cell_size,
               local_DG, dust_consumed_count, dust_consumed_mass * mass_to_msun);
  }
  if(dust_consumed_count > 20)
    DUST_PRINT("[ASTRATION_LARGE|T=%d|Step=%d] consumed=%d neighbors=%d "
               "search_r=%.3e hsml=%.3e cell_size=%.3e D/G=%.3e\n",
               All.ThisTask, All.NumCurrentTiStep,
               dust_consumed_count, n_neighbors,
               search_radius, hsml, dust_hash.cell_size, local_DG);
  // Periodic bottleneck summary — confirms whether the search-radius fix
  // actually reduced the zero-neighbor and zero-dust-mass failure rates.
  if(All.ThisTask == 0 && astration_attempts % 1000 == 0)
    DUST_PRINT("[ASTRATION_DIAG] attempts=%lld  zero_neighbor=%lld(%.1f%%)  "
               "zero_dustmass=%lld(%.1f%%)  succeeded=%d\n",
               astration_attempts,
               astration_zero_neighbor, 100.0*astration_zero_neighbor/astration_attempts,
               astration_zero_dustmass, 100.0*astration_zero_dustmass/astration_attempts,
               astration_count);
}
// ═══════════════════════════════════════════════════════════════════════════════
// radiation_pressure_efficiency (Q_pr)
// ═══════════════════════════════════════════════════════════════════════════════
/**
 * Compute the radiation pressure efficiency Q_pr for a grain of radius a_nm
 * and carbon fraction CF.
 *
 * Parametrisation: Lorentzian ramp calibrated to luminosity-weighted OB
 * population (T_eff ~ 23,000 K, peak λ ~ 125 nm → a₀ = 35 nm):
 *
 *   Q_pr = [a / (a + a₀)] × [0.8 + 0.4 × CF]
 *
 * Silicates (CF=0): species_factor = 0.8   (lower absorption cross-section)
 * Pure carbon (CF=1): species_factor = 1.2 (higher absorption)
 *
 * Reference: Draine & Lee 1984.
 */
double radiation_pressure_efficiency(double a_nm, double carbon_fraction)
{
  const double a0_nm = 35.0;  // Transition scale: grains << a0 are inefficient absorbers
  double Q_pr = a_nm / (a_nm + a0_nm);  // Lorentzian ramp to order unity
  // Linear composition blend: silicate (CF=0) → carbon (CF=1)
  double species_factor = 0.8 + 0.4 * carbon_fraction;
  Q_pr *= species_factor;
  return std::max(0.0, std::min(2.0, Q_pr));
}
// ═══════════════════════════════════════════════════════════════════════════════
// stellar_luminosity
// ═══════════════════════════════════════════════════════════════════════════════
/**
 * Estimate luminosity of a stellar particle based on its current age.
 *
 * Simplified piecewise L/M model appropriate for single-age stellar populations.
 * O/B stars dominate until ~10 Myr; by 40 Myr only A/F stars remain.
 * Particles older than 100 Myr (enforced upstream) contribute negligible UV.
 *
 * NOTE: This is a placeholder. Production-quality runs should use a
 * population synthesis table (Starburst99, BPASS, or FSPS) interpolated
 * by age and metallicity.
 */
double stellar_luminosity(simparticles *Sp, int star_idx)
{
  const double L_sun_cgs = 3.828e33;
  const double M_sun_cgs = 1.989e33;
  double M_star_sun  = Sp->P[star_idx].getMass() * All.UnitMass_in_g
                       / (All.HubbleParam * M_sun_cgs);
  double age_yr      = get_stellar_age_Myr(Sp->P[star_idx].StellarAge, All.Time) * 1.0e6;
  double L_over_M;
  if     (age_yr < 3e6)   L_over_M = 200.0;
  else if(age_yr < 10e6)  L_over_M = 200.0 - 100.0 * (age_yr - 3e6)  / 7e6;
  else if(age_yr < 40e6)  L_over_M = 100.0 -  80.0 * (age_yr - 10e6) / 30e6;
  else                    L_over_M = std::max(0.0, 20.0 - 18.0 * (age_yr - 40e6) / 60e6);
  return L_over_M * M_star_sun * L_sun_cgs;
}
// ═══════════════════════════════════════════════════════════════════════════════
// dust_radiation_pressure
// ═══════════════════════════════════════════════════════════════════════════════
/**
 * Apply radiation pressure from young nearby stars to a dust grain.
 *
 * FORCE MODEL:
 *   F_rad = (L / 4πr²c) × Q_pr × πa²
 *
 * where L is the stellar luminosity, r the star–grain distance, Q_pr the
 * radiation pressure efficiency (Draine & Lee 1984), and a the grain radius.
 * The optically thin limit is assumed between star and grain.
 *
 * TERMINAL VELOCITY CORRECTION:
 *   Grains are not in free flight — they are drag-coupled to gas. In the
 *   terminal velocity regime (t_drag << t_cross), grains drift at a velocity
 *   set by the balance of radiation pressure and drag rather than accelerating
 *   freely. The correction factor is:
 *
 *     f_coupling = (1 − exp(−dt/t_stop)) × t_stop / dt
 *
 *   → 1 when dt << t_stop (grain responds fully to the kick this step)
 *   → t_stop/dt << 1 when dt >> t_stop (terminal velocity limit, suppresses kick)
 *
 *   Note: t_stop here is computed WITHOUT the 50 Myr cap used in
 *   calculate_drag_timescale(), because the cap is a numerical safeguard for
 *   the drag velocity update and should not suppress the physical terminal
 *   velocity correction in the CGM.
 *
 * Only stars younger than 20 Myr are included (OB-dominated UV).
 */
void dust_radiation_pressure(simparticles *Sp, int dust_idx, int nearest_gas, double dt)
{
  if(!All.DustEnableRadiationPressure) return;
  if(!star_hash.is_built) return;
  // Diagnostic counters
  static long long radp_calls       = 0;
  static long long radp_with_stars  = 0;
  static long long radp_with_young  = 0;
  static double    radp_total_accel = 0.0;
  static double    radp_total_coupling = 0.0;
  radp_calls++;
  // ── Search radius ─────────────────────────────────────────────────────────
  // Use gas smoothing length as the local ISM scale (scales with resolution).
  // Floor at one star hash cell width so the search always finds neighbours.
  // Hard cap at 10 kpc beyond which the 1/r² flux is negligible.
  double search_radius = (nearest_gas >= 0)
      ? std::max(2.0 * (double)Sp->SphP[nearest_gas].Hsml, 2.0 * star_hash.cell_size)
      : 2.0 * star_hash.cell_size;
  search_radius = std::min(search_radius, 10.0);
  const int MAX_STAR_NEIGHBORS = 50;
  int    neighbor_indices[MAX_STAR_NEIGHBORS];
  double neighbor_distances[MAX_STAR_NEIGHBORS];
  int    n_neighbors = 0;
  star_hash.find_neighbors(Sp, dust_idx, search_radius,
                           neighbor_indices, neighbor_distances,
                           &n_neighbors, MAX_STAR_NEIGHBORS);
  if(n_neighbors == 0) return;
  radp_with_stars++;
  // ── Grain properties ──────────────────────────────────────────────────────
  double a_nm  = Sp->DustP[dust_idx].GrainRadius;
  double a_cm  = a_nm * 1e-7;
  double CF    = Sp->DustP[dust_idx].CarbonMassFraction;
  double Q_pr  = radiation_pressure_efficiency(a_nm, CF);
  const double c_cgs          = 2.998e10;   // cm/s
  const double rho_grain_cgs  = 2.4;        // g/cm³ (silicate)
  double m_single_grain_cgs   = (4.0/3.0) * M_PI * a_cm*a_cm*a_cm * rho_grain_cgs;
  // Accumulate radiation acceleration from all young neighbours
  double a_rad[3] = {0.0, 0.0, 0.0};
  for(int i = 0; i < n_neighbors; i++) {
    int si = neighbor_indices[i];
    if(Sp->P[si].getType() != 4) continue;  // only stellar particles
    // Age cut: only OB-dominated populations (< 20 Myr)
    double age_yr = get_stellar_age_Myr(Sp->P[si].StellarAge, All.Time) * 1.0e6;
    if(age_yr > 20e6) continue;
    double L_cgs = stellar_luminosity(Sp, si);
    if(L_cgs <= 0.0) continue;
    // Convert code distance to physical cm
    double r_code = neighbor_distances[i];
    double r_cgs  = r_code * (All.Time / All.HubbleParam) * 3.086e21;
    if(r_cgs <= 0.0) continue;
    // Vector from star to grain: nearest_image_intpos_to_pos(a, b) computes a - b,
    // so this is dust_pos - star_pos, pointing radially outward from the star.
    // Dividing by r_code (the code-unit separation) normalises to a unit vector.
    // Radiation pressure acts along this direction (photons push dust away from source)
    double dxyz[3];
    Sp->nearest_image_intpos_to_pos(Sp->P[dust_idx].IntPos, Sp->P[si].IntPos, dxyz);
    // Radiation pressure acceleration on the single grain:
    //   F = (L / 4πr²c) × Q_pr × πa²
    //   a = F / m_grain
    double flux_cgs = L_cgs / (4.0 * M_PI * r_cgs * r_cgs * c_cgs);
    double accel    = flux_cgs * Q_pr * M_PI * a_cm * a_cm
                      / m_single_grain_cgs
                      * All.DustRadiationPressureEfficiency;
    for(int k = 0; k < 3; k++)
      a_rad[k] += accel * dxyz[k] / r_code;
  }
  // ── Terminal velocity correction ──────────────────────────────────────────
  // Compute TRUE (unclamped) stopping time for the coupling factor.
  // The 50 Myr cap in calculate_drag_timescale() is a numerical safeguard for
  // the drag velocity update and must NOT be applied here.
  double t_stop_code = 0.0;
  if(nearest_gas >= 0 && Sp->P[nearest_gas].getType() == 0) {
    double rho_cgs = dust_gas_density_cgs(Sp, nearest_gas);
    double T_gas   = get_temperature_from_entropy(Sp, nearest_gas);
    if(rho_cgs > 0.0 && T_gas > 0.0) {
      double c_s_cgs  = sqrt((5.0/3.0) * BOLTZMANN * T_gas / (0.6 * PROTONMASS));
      double t_stop_s = (sqrt(M_PI * 5.0/3.0 / 8.0) * a_cm * 2.4)
                        / (rho_cgs * c_s_cgs);
      t_stop_code = t_stop_s / dust_code_time_in_s();
    }
  }
  // f_coupling → 1 for dt << t_stop (grain accelerates freely this step)
  // f_coupling → t_stop/dt for dt >> t_stop (terminal velocity regime)
  double coupling_factor = 1.0;
  if(t_stop_code > 0.0 && dt > 0.0)
    coupling_factor = (1.0 - exp(-dt / t_stop_code)) * t_stop_code / dt;
  radp_total_coupling += coupling_factor;
  // ── Apply velocity kick with terminal velocity suppression ────────────────
  double accel_code = All.UnitVelocity_in_cm_per_s / dust_code_time_in_s();
  for(int k = 0; k < 3; k++)
    // Physical acceleration g changes p = a^2 dx/dt by a * g * dt.
    Sp->P[dust_idx].Vel[k] += All.cf_atime * (a_rad[k] / accel_code) * dt * coupling_factor;
  // Update diagnostics
  double accel_mag = sqrt(a_rad[0]*a_rad[0] + a_rad[1]*a_rad[1] + a_rad[2]*a_rad[2]);
  if(accel_mag > 0.0) radp_with_young++;
  radp_total_accel += accel_mag;
  if(radp_calls % 500000 == 0 && All.ThisTask == 0)
    DUST_PRINT("[RADIATION_PRESSURE_RATE] calls=%lld  with_any_stars=%.1f%%  "
               "with_young_stars=%.1f%%  avg_coupling=%.3e avg_|a|=%.2e code\n",
               radp_calls,
               100.0 * radp_with_stars  / radp_calls,
               100.0 * radp_with_young  / radp_calls,
               radp_total_coupling      / radp_calls,
               radp_total_accel         / radp_calls);
}
// ═══════════════════════════════════════════════════════════════════════════════
// update_dust_temperature
// ═══════════════════════════════════════════════════════════════════════════════
/**
 * Estimate the equilibrium dust temperature from modified-blackbody energy
 * balance and relax the represented grain population toward that value.
 *
 * Equilibrium condition:
 *   P_emit(T_eq) = P_CMB + P_ISRF + P_gas
 *
 * where:
 *   P_emit  = C_emit × T^(4+β)               [modified blackbody emission]
 *   P_CMB   = C_emit × T_CMB^(4+β)           [CMB photon bath]
 *   P_ISRF  = C_emit × T_ISRF^(4+β)          [interstellar radiation field]
 *   P_gas   = f_eff × 2 n_H k_B (T_gas−T_dust) α_T v_th π a²  [collisional coupling]
 *
 * Emissivity uses Draine & Lee (1984) modified blackbody: Q_abs ∝ a T^β, β=2.
 *
 * The collisional term is evaluated at the dust temperature at the start of
 * the update, so T_eq is an operator-split estimate rather than the exact root
 * of the coupled gas/dust equation.  The dust temperature then relaxes toward
 * that estimate exponentially over
 * τ_cool = C_grain × T_dust / P_emit (Hollenbach & McKee 1979).
 *
 * Gas back-reaction is intentionally not applied here.  Updating the nearest
 * gas particle once per dust superparticle made the result particle-order and
 * decomposition dependent, and a per-call limiter did not bound the total
 * drain from many grains selecting the same gas particle.  A future back-
 * reaction implementation must accumulate exchanges per gas particle and
 * apply one conservative, bounded update after the dust loop.
 *
 * dt is a physical elapsed interval in internal code-time units.  In
 * cosmological runs run.cc must convert Δa using Δt = Δa/[a H(a)] before
 * calling update_dust_dynamics().
 *
 * References: Hollenbach & McKee 1979, Draine & Lee 1984, Mathis+1983, Draine+2007.
 */
void update_dust_temperature(simparticles *Sp, int dust_idx, int gas_idx, double dt,
                             double T_gas)
{
  // If dust cooling is disabled, only enforce the CMB floor
  if(!All.DustEnableCooling) {
    double T_CMB_floor = 2.7 / All.Time;
    if(Sp->DustP[dust_idx].DustTemperature < T_CMB_floor)
      Sp->DustP[dust_idx].DustTemperature = T_CMB_floor;
    return;
  }
  // ── Constants and grain properties ───────────────────────────────────────
  const double alpha_T   = All.DustAccommodationCoeff;     // thermal accommodation coefficient (Burke & Hollenbach 1983)
  const double beta      = 2.0;        // emissivity spectral index (silicate)
  const double rho_grain = 2.4;        // g/cm³, silicate density
  const double sigma_SB  = 5.6704e-5;  // erg/cm²/s/K⁴
  const double T_ref     = 100.0;      // K, Q_abs reference temperature
  const double a_ref_cm  = 1e-5;       // cm (= 0.1 µm), Q_abs reference size
  const double Q_ref     = 1.3e-4;     // Q_abs(a_ref, T_ref) from Draine & Lee 1984
  double a_nm   = Sp->DustP[dust_idx].GrainRadius;
  double a_cm   = a_nm * 1e-7;
  double CF     = Sp->DustP[dust_idx].CarbonMassFraction;
  // Carbonaceous grains: lower β (~1.5) and higher Q_abs normalisation (Draine 2003)
  double beta_eff = beta - 0.5 * CF;
  double Q_eff    = Q_ref * (1.0 + CF) * (a_cm / a_ref_cm);
  double T_CMB  = 2.7 / All.Time;
  double T_dust = Sp->DustP[dust_idx].DustTemperature;
  if(T_dust <= 0.0 || !isfinite(T_dust)) T_dust = T_CMB;
  double n_H         = dust_gas_nH_cgs(Sp, gas_idx);
  // ── Emission coefficient C_emit ───────────────────────────────────────────
  double C_emit = 4.0 * M_PI * a_cm * a_cm * Q_eff * sigma_SB / pow(T_ref, beta_eff);
  // ── Heating: CMB ─────────────────────────────────────────────────────────
  double P_CMB = C_emit * pow(T_CMB, 4.0 + beta_eff);
  // ── Heating: ISRF ─────────────────────────────────────────────────────────
  // T_ISRF ~ 17 K at z=0 (Reach+1995, Mathis+1983); scale weakly with redshift
  // as more young stars heat the background field.
  double z_now  = 1.0 / All.Time - 1.0;
  double T_ISRF = 17.0 * pow(1.0 + z_now, 0.25);
  double P_ISRF = C_emit * pow(T_ISRF, 4.0 + beta_eff);
  // ── Heating/cooling: gas-grain collisional coupling ───────────────────────
  // P_coll = 2 n_H k_B (T_gas − T_dust) α_T v_th π a²
  // Effective coupling factor f_eff from Hollenbach-McKee (1979) Fig. 1:
  //   cold neutral gas (T < 10³ K):  f_eff ~ 1.0  (neutrals)
  //   warm ionised gas (10³–10⁴ K):  f_eff ~ 1–2.5 (ions begin contributing)
  //   hot gas (> 10⁴ K):             f_eff ~ 2.5  (electrons dominate)
  double f_eff;
  if     (T_gas < 1e3) f_eff = 1.0;
  else if(T_gas < 1e4) f_eff = 1.0 + 1.5 * (T_gas - 1e3) / 9e3;
  else                 f_eff = 2.5;
  double v_th  = sqrt(8.0 * BOLTZMANN * T_gas / (M_PI * PROTONMASS));
  double P_gas = f_eff * 2.0 * n_H * BOLTZMANN * (T_gas - T_dust)
                 * alpha_T * v_th * M_PI * a_cm * a_cm;
  double dt_cgs = dt * dust_code_time_in_s();

  // Mass and heat capacity of one physical grain represented by this
  // superparticle.  These enter only the dust thermal relaxation; the gas is
  // not mutated in this routine (see the function comment above).
  double m_grain_cgs = (4.0/3.0) * M_PI * a_cm*a_cm*a_cm * rho_grain;
  // ── Solve for equilibrium temperature ────────────────────────────────────
  // C_emit × T_eq^(4+β) = P_CMB + P_ISRF + P_gas
  double P_total = std::max(P_CMB + P_ISRF + P_gas, P_CMB);  // clamp to CMB floor
  double T_eq    = pow(P_total / C_emit, 1.0 / (4.0 + beta_eff));
  // ── Relax toward T_eq over the cooling timescale τ_cool ──────────────────
  // Prevents numerical oscillations when T_dust is far from T_eq.
  const double c_v = 7e6;  // erg/g/K (Draine & Li 2001, valid T > 20 K)
  double C_grain        = m_grain_cgs * c_v;
  double P_emit_current = C_emit * pow(T_dust + 1.0, 4.0 + beta_eff);
  double tau_cool       = (P_emit_current > 0.0)
                          ? (C_grain * T_dust / P_emit_current) : 1e10;
  double relax = 1.0 - exp(-dt_cgs / tau_cool);
  double T_new = T_dust + relax * (T_eq - T_dust);
  // ── Safety bounds ─────────────────────────────────────────────────────────
  if(T_new < T_CMB)                T_new = T_CMB;
  double T_sublimate_local = 1500.0 + 500.0 * CF;
  if(T_new > T_sublimate_local) T_new = T_sublimate_local;
  if(!isfinite(T_new))             T_new = T_CMB;
  Sp->DustP[dust_idx].DustTemperature = T_new;
}
// ═══════════════════════════════════════════════════════════════════════════════
// update_dust_dynamics  (main cadence-update entry point)
// ═══════════════════════════════════════════════════════════════════════════════
/**
 * Apply all dust physics processes to all live dust particles.
 *
 * The run loop may call this less often than once per global synchronization
 * point.  dt is the sum of the exact physical intervals since the preceding
 * call, in internal code-time units.  It must never be reconstructed as the
 * current interval times a nominal cadence: adaptive cosmological intervals
 * are unequal.  Pending time is force-flushed before snapshots and at z=0.
 *
 * PROCESS ORDER per particle:
 *   1. Dust temperature equilibration (if DustEnableCooling or DustEnableDrag)
 *   2. Gas drag + sputtering (dust_gas_interaction)
 *   3. Radiation pressure from young stars (if DustEnableRadiationPressure)
 *   4. Grain growth via accretion from gas (if within 2 kpc of gas)
 *   5. Grain coagulation in dense cold gas
 *   6. Grain shattering in turbulent diffuse gas
 */
void update_dust_dynamics(simparticles *Sp, double dt, MPI_Comm Communicator)
{
  // Includes the collective existence check, so MPI waiting is not hidden.
  static double total_time_in_dust = 0.0;
  static int    dust_call_count    = 0;
  double t_start = MPI_Wtime();

  // ── One-time flag verification (confirms parameter file was read) ─────────
  static bool flags_printed = false;
  if(!flags_printed && All.ThisTask == 0) {
    printf("[DUST_FLAGS|Step=%d] Creation=%d Drag=%d Growth=%d Coagulation=%d "
           "Sputtering=%d ShockDestruction=%d Astration=%d RadPressure=%d "
           "Clumping=%d Cooling=%d\n",
           All.NumCurrentTiStep,
           All.DustEnableCreation, All.DustEnableDrag,
           All.DustEnableGrowth,   All.DustEnableCoagulation,
           All.DustEnableSputtering, All.DustEnableShockDestruction,
           All.DustEnableAstration,  All.DustEnableRadiationPressure,
           All.DustEnableClumping, All.DustEnableCooling);
    flags_printed = true;
  }
  // ── One-time log open ─────────────────────────────────────────────────────
  static bool log_opened = false;
  if(!log_opened) {
    open_dust_particle_log(Communicator);
    log_opened = true;
  }

  // ── Collective check: any live dust exists across all tasks? ──────────────
  // GlobalDustCount is a diagnostic counter, not restart-persistent truth, and
  // can become stale after compaction or particle exchange.  Recounting here
  // prevents a restarted run from silently skipping all dust evolution.
  long long local_count = 0;
  for(int i = 0; i < Sp->NumPart; i++)
    if(Sp->P[i].getType() == DUST_PARTICLE_TYPE &&
       Sp->P[i].getMass() > DUST_MASS_TO_DESTROY && Sp->P[i].ID.get() != 0)
      local_count++;
  GlobalDustCount = local_count;

  long long global_count = 0;
  MPI_Allreduce(&local_count, &global_count, 1, MPI_LONG_LONG, MPI_SUM, Communicator);
  if(global_count == 0) {
    total_time_in_dust += MPI_Wtime() - t_start;
    dust_call_count++;
    return;
  }
  // ── Ensure spatial hash is built ─────────────────────────────────────────
  int need_hash_rebuild = 0;
  if(All.ThisTask == 0) need_hash_rebuild = !gas_hash.is_built;
  MPI_Bcast(&need_hash_rebuild, 1, MPI_INT, 0, Communicator);
  if(need_hash_rebuild) {
    if(All.ThisTask == 0)
      DUST_PRINT("WARNING: Hash not built, building now for dust operations\n");
    rebuild_feedback_spatial_hash(Sp, 0.1, Communicator);
  }
  // ── One-time hash verification ────────────────────────────────────────────
  static bool verified = false;
  if(!verified && All.ThisTask == 0) {
    DUST_PRINT("=== HASH VERIFICATION ===\n");
    DUST_PRINT("  Hash built:    %s\n",      gas_hash.is_built ? "YES" : "NO");
    DUST_PRINT("  Cells per dim: %d\n",      gas_hash.n_cells_per_dim);
    DUST_PRINT("  Cell size:     %.3f kpc\n",gas_hash.cell_size);
    DUST_PRINT("  Gas particles: %d\n",      gas_hash.total_particles);
    DUST_PRINT("=========================\n");
    verified = true;
  }
  std::vector<int> nearest_gas_by_particle((size_t)Sp->NumPart, -1);
  std::vector<double> gas_distance_by_particle((size_t)Sp->NumPart, -1.0);
  std::vector<double> coagulation_radius_by_particle((size_t)Sp->NumPart, 0.0);
  for(int i = 0; i < Sp->NumPart; i++) {
    if(Sp->P[i].getType() != DUST_PARTICLE_TYPE ||
       Sp->P[i].getMass() <= DUST_MASS_TO_DESTROY)
      continue;

    double dist_kpc = -1.0;
    const int nearest_gas = find_nearest_gas_particle(Sp, i, 10.0, &dist_kpc);
    if(nearest_gas < 0)
      continue;

    nearest_gas_by_particle[i] = nearest_gas;
    gas_distance_by_particle[i] = dist_kpc;
    const double association_radius =
        std::max(2.0, std::min((double)Sp->SphP[nearest_gas].Hsml, 10.0));
    if(dist_kpc <= association_radius)
      coagulation_radius_by_particle[i] = association_radius;
  }

  // ── Local DTG with staggered per-particle caching ─────────────────────────
  // DTG changes on growth/destruction timescales (tens to hundreds of Myr),
  // much slower than the dust cadence, so each particle's neighborhood DTG is
  // recomputed only every DTG_REFRESH_INTERVAL dust calls and reused between.
  //
  // The cache lives in DustP[] (LocalDTG, DTGAge) so it travels with the
  // particle through domain exchange.  An earlier task-local map keyed by ID
  // missed ~95% of the time because the near-every-step domain decomposition
  // moves most dust to a different task between dust calls.
  //
  // DTGAge counts dust calls since the last recompute: a particle reuses its
  // cache while DTGAge < DTG_REFRESH_INTERVAL - 1, so each refreshes exactly
  // once every DTG_REFRESH_INTERVAL calls.  Particles without a valid cache
  // (new dust, snapshot starts, pre-cache restart files, failed estimates)
  // refresh immediately and take their array index as the phase, so a batch of
  // them (e.g. every particle after a restart) does not fall due together.  Unassociated particles have their cache
  // invalidated, since a value from an earlier association may be stale.
  static constexpr int DTG_REFRESH_INTERVAL = 10;

  // Cumulative histogram of |DTG_new - DTG_cached| / DTG_cached at scheduled
  // refreshes, i.e. the drift accumulated over DTG_REFRESH_INTERVAL calls.
  // Used to judge whether the interval can safely be raised.
  static constexpr int DTG_STALE_NBINS = 6;
  static const double dtg_stale_edges[DTG_STALE_NBINS - 1] = {0.01, 0.03, 0.1, 0.3, 1.0};
  static long long dtg_stale_hist[DTG_STALE_NBINS] = {0, 0, 0, 0, 0, 0};
  static double dtg_stale_sum_rel = 0.0;

  std::vector<double> local_DTG((size_t)Sp->NumPart, 0.0);
  long long n_dtg_refreshed = 0, n_dtg_cached = 0, dtg_queries_sent = 0;
  double t_dtg_start = MPI_Wtime();
  if(All.DustEnableCoagulation || All.DustEnableShattering) {
    std::vector<double> refresh_radius((size_t)Sp->NumPart, 0.0);
    std::vector<char> had_valid_cache((size_t)Sp->NumPart, 0);
    for(int i = 0; i < Sp->NumPart; i++) {
      if(Sp->P[i].getType() != DUST_PARTICLE_TYPE) continue;
      dust_data &dp = Sp->DustP[i];
      if(!(coagulation_radius_by_particle[i] > 0.0)) {
        dp.LocalDTG = 0.0f;
        dp.DTGAge = 0;
        continue;
      }
      const bool valid = isfinite(dp.LocalDTG) && dp.LocalDTG > 0.0f &&
                         dp.DTGAge >= 0;
      had_valid_cache[i] = valid;
      if(valid && dp.DTGAge < DTG_REFRESH_INTERVAL - 1) {
        local_DTG[i] = dp.LocalDTG;
        dp.DTGAge++;
        n_dtg_cached++;
      } else {
        refresh_radius[i] = coagulation_radius_by_particle[i];
        n_dtg_refreshed++;
      }
    }

    // Collective: every task must call this even if it has nothing to refresh.
    std::vector<double> fresh_DTG;
    TIMER_START(CPU_DUST_DTG);
    estimate_local_dust_to_gas_ratio(Sp, refresh_radius, Communicator, fresh_DTG,
                                     /*occupancy_routing=*/true, &dtg_queries_sent);

    // Self-check on the first calls: occupancy routing must reproduce the
    // bounding-box routing bit for bit (it only drops tasks contributing 0).
    static int route_verify_remaining = 3;
    if(route_verify_remaining > 0) {
      route_verify_remaining--;
      std::vector<double> bbox_DTG;
      long long bbox_sent = 0;
      estimate_local_dust_to_gas_ratio(Sp, refresh_radius, Communicator, bbox_DTG,
                                       /*occupancy_routing=*/false, &bbox_sent);
      long long counts[3] = {0, dtg_queries_sent, bbox_sent};
      for(int i = 0; i < Sp->NumPart; i++)
        if(refresh_radius[i] > 0.0 &&
           memcmp(&fresh_DTG[i], &bbox_DTG[i], sizeof(double)) != 0)
          counts[0]++;
      long long sums[3];
      MPI_Reduce(counts, sums, 3, MPI_LONG_LONG, MPI_SUM, 0, Communicator);
      if(All.ThisTask == 0)
        printf("[DTG_ROUTE_VERIFY] occupancy vs bounding-box routing: %lld differing "
               "D/G values; queries sent %lld vs %lld (%.1fx fewer)\n",
               sums[0], sums[1], sums[2], sums[1] > 0 ? (double)sums[2] / sums[1] : 0.0);
    }
    TIMER_STOP(CPU_DUST_DTG);

    for(int i = 0; i < Sp->NumPart; i++) {
      if(!(refresh_radius[i] > 0.0)) continue;
      local_DTG[i] = fresh_DTG[i];
      dust_data &dp = Sp->DustP[i];
      // Staleness: how far DTG drifted over one full refresh interval.
      if(had_valid_cache[i] && isfinite(fresh_DTG[i]) && fresh_DTG[i] > 0.0) {
        const double rel = std::fabs(fresh_DTG[i] - dp.LocalDTG) / dp.LocalDTG;
        int bin = 0;
        while(bin < DTG_STALE_NBINS - 1 && rel >= dtg_stale_edges[bin]) bin++;
        dtg_stale_hist[bin]++;
        dtg_stale_sum_rel += rel;
      }
      dp.LocalDTG = (isfinite(fresh_DTG[i]) && fresh_DTG[i] > 0.0)
                        ? (float)fresh_DTG[i] : 0.0f;
      dp.DTGAge = had_valid_cache[i] ? 0 : i % DTG_REFRESH_INTERVAL;
    }
  }
  double t_dtg_local = MPI_Wtime() - t_dtg_start;

  double t_dtg_max = 0.0, t_dtg_sum = 0.0;
  long long dtg_counts_local[3] = {n_dtg_refreshed, n_dtg_cached, dtg_queries_sent};
  long long dtg_counts_sum[3] = {0, 0, 0};
  int n_tasks_dtg = 1;
  MPI_Comm_size(Communicator, &n_tasks_dtg);
  MPI_Reduce(&t_dtg_local, &t_dtg_max, 1, MPI_DOUBLE, MPI_MAX, 0, Communicator);
  MPI_Reduce(&t_dtg_local, &t_dtg_sum, 1, MPI_DOUBLE, MPI_SUM, 0, Communicator);
  MPI_Reduce(dtg_counts_local, dtg_counts_sum, 3, MPI_LONG_LONG, MPI_SUM, 0,
             Communicator);
  if(All.ThisTask == 0 && dust_call_count % 10 == 0)
    printf("[DTG_TIMING|Step=%d] estimate_local_DTG: "
           "max=%.3f s  mean=%.3f s  n_dust=%lld  refreshed=%lld cached=%lld  "
           "queries_sent=%lld (%.2f per refreshed)\n",
           All.NumCurrentTiStep,
           t_dtg_max,
           t_dtg_sum / n_tasks_dtg,
           global_count,
           dtg_counts_sum[0], dtg_counts_sum[1], dtg_counts_sum[2],
           dtg_counts_sum[0] > 0 ? (double)dtg_counts_sum[2] / dtg_counts_sum[0] : 0.0);

  if(dust_call_count % 10 == 0) {
    long long stale_sum[DTG_STALE_NBINS];
    double stale_rel_sum = 0.0;
    MPI_Reduce(dtg_stale_hist, stale_sum, DTG_STALE_NBINS, MPI_LONG_LONG, MPI_SUM, 0,
               Communicator);
    MPI_Reduce(&dtg_stale_sum_rel, &stale_rel_sum, 1, MPI_DOUBLE, MPI_SUM, 0, Communicator);
    long long n_stale = 0;
    for(int b = 0; b < DTG_STALE_NBINS; b++) n_stale += stale_sum[b];
    if(All.ThisTask == 0 && n_stale > 0)
      printf("[DTG_STALENESS] after %d calls (cumulative, n=%lld): mean |dDTG|/DTG=%.3f  "
             "<1%%=%.1f%% 1-3%%=%.1f%% 3-10%%=%.1f%% 10-30%%=%.1f%% 30-100%%=%.1f%% >100%%=%.1f%%\n",
             DTG_REFRESH_INTERVAL, n_stale, stale_rel_sum / n_stale,
             100.0 * stale_sum[0] / n_stale, 100.0 * stale_sum[1] / n_stale,
             100.0 * stale_sum[2] / n_stale, 100.0 * stale_sum[3] / n_stale,
             100.0 * stale_sum[4] / n_stale, 100.0 * stale_sum[5] / n_stale);
  }

  // ── Main dust loop ────────────────────────────────────────────────────────
  for(int i = 0; i < Sp->NumPart; i++) {
    if(Sp->P[i].getType() != DUST_PARTICLE_TYPE) continue;
    if(Sp->P[i].getMass() <= DUST_MASS_TO_DESTROY) continue;
    const double dist_kpc = gas_distance_by_particle[i];
    const int nearest_gas = nearest_gas_by_particle[i];
    if(nearest_gas < 0) continue;
    // Gas temperature is shared by every process below.  Dust routines never
    // modify gas entropy, density, or Ne, so computing it once is exact.
    const double T_gas = get_temperature_from_entropy(Sp, nearest_gas);
    // Temperature update — needs gas properties, so runs whenever drag or
    // cooling is active (both read the gas entropy).
    if(All.DustEnableCooling || All.DustEnableDrag)
      update_dust_temperature(Sp, i, nearest_gas, dt, T_gas);

    dust_gas_interaction(Sp, i, nearest_gas, dt, T_gas);
    dust_radiation_pressure(Sp, i, nearest_gas, dt);
    
    // Growth, coagulation, and shattering require a resolved local gas
    // association.  A fixed 2-kpc cut is strongly resolution dependent:
    // at modest resolution a dust grain can lie well inside the nearest gas
    // particle's SPH kernel while being more than 2 kpc from its center.
    //
    // Use the nearest gas smoothing length as the resolved neighborhood scale,
    // retain 2 kpc as a floor, and cap at the existing 10-kpc nearest-gas
    // search radius so this change cannot create arbitrarily long-range
    // coupling in diffuse CGM gas.
    const double local_dust_gas_radius = coagulation_radius_by_particle[i];

    if(local_dust_gas_radius > 0.0 && dist_kpc <= local_dust_gas_radius) {
      if(All.DustEnableGrowth)
        dust_grain_growth_subgrid(Sp, i, nearest_gas, dt, T_gas);
      if(All.DustEnableCoagulation)
        dust_grain_coagulation(Sp, i, nearest_gas, dt, local_DTG[i], T_gas);
      if(All.DustEnableShattering)
        dust_grain_shattering(Sp, i, nearest_gas, dt, local_DTG[i], T_gas);
    }
  }
  // ── Periodic diagnostics (every 500 steps, task 0) ───────────────────────
  if(All.NumCurrentTiStep % 500 == 0) {
    print_dust_statistics(Sp, Communicator);
    analyze_dust_gas_coupling_local(Sp);
    analyze_grain_size_distribution(Sp);
    if(All.ThisTask == 0) {
      DUST_PRINT("[GROWTH_SUMMARY] Total growth events so far: %lld\n", NGrainGrowthEvents);
      double M_cold = 0, M_warm = 0, M_hot = 0, M_vhot = 0;
      for(int i = 0; i < Sp->NumGas; i++) {
        double T = get_temperature_from_entropy(Sp, i);
        double M = Sp->P[i].getMass();
        if     (T < 1e4) M_cold += M;
        else if(T < 1e5) M_warm += M;
        else if(T < 1e6) M_hot  += M;
        else             M_vhot += M;
      }
      double M_tot = M_cold + M_warm + M_hot + M_vhot;
      DUST_PRINT("[GAS_BUDGET] cold<10^4=%.1f%%  warm<10^5=%.1f%%  hot<10^6=%.1f%%  vhot=%.1f%%\n",
                 100*M_cold/M_tot, 100*M_warm/M_tot, 100*M_hot/M_tot, 100*M_vhot/M_tot);
      int printed = 0;
      for(int i = 0; i < Sp->NumPart && printed < 3; i++) {
        if(Sp->P[i].getType() != DUST_PARTICLE_TYPE ||
           Sp->P[i].getMass() <= DUST_MASS_TO_DESTROY) continue;
        DUST_PRINT("[DUST_SAMPLE] i=%d ID=%lld M=%.2e a=%.2f nm CF=%.2f GT=%d\n",
                   i, (long long)Sp->P[i].ID.get(), Sp->P[i].getMass(),
                   Sp->DustP[i].GrainRadius, Sp->DustP[i].CarbonMassFraction,
                   Sp->DustP[i].DustSource);
        printed++;
      }
    }
  }
  // ── Timing summary (every 100 calls) ─────────────────────────────────────
  double dt_dust = MPI_Wtime() - t_start;
  total_time_in_dust += dt_dust;
  dust_call_count++;
  if(dust_call_count % 100 == 0 && All.ThisTask == 0)
    printf("[DUST_TIMING] Called %d times, avg %.3f sec/call, total %.1f sec\n",
           dust_call_count, total_time_in_dust / dust_call_count, total_time_in_dust);
}
// ═══════════════════════════════════════════════════════════════════════════════
// destroy_dust_particles
// ═══════════════════════════════════════════════════════════════════════════════
/**
 * Compact the particle array by removing all dust particles marked for
 * destruction (type 3 or mass ≤ DUST_MASS_TO_DESTROY or ID = 0).
 *
 * Maintains the invariant that gas particles (type 0) occupy indices
 * [0, NumGas) and all other particles (stars, dust, etc.) follow.
 */
void destroy_dust_particles(simparticles *Sp)
{
  int dust_destroyed = 0;
  for(int i = 0; i < Sp->NumPart; i++) {
    if((Sp->P[i].getType() == DUST_PARTICLE_TYPE || Sp->P[i].getType() == 3) &&
       (Sp->P[i].getMass() <= DUST_MASS_TO_DESTROY || Sp->P[i].ID.get() == 0))
      dust_destroyed++;
  }
  if(dust_destroyed == 0) return;
  // Compact gas region [0, NumGas)
  int new_num_gas = 0;
  for(int i = 0; i < Sp->NumGas; i++) {
    bool is_dead = (Sp->P[i].getType() == DUST_PARTICLE_TYPE ||
                    Sp->P[i].getType() == 3)
                   && (Sp->P[i].getMass() <= DUST_MASS_TO_DESTROY ||
                       Sp->P[i].ID.get() == 0);
    if(!is_dead) {
      if(new_num_gas != i) {
        Sp->P[new_num_gas]    = Sp->P[i];
        Sp->SphP[new_num_gas] = Sp->SphP[i];
      }
      memset(&Sp->DustP[new_num_gas], 0, sizeof(dust_data));
      new_num_gas++;
    }
  }
  // Compact non-gas region [NumGas, NumPart)
  int new_num_part = new_num_gas;
  for(int i = Sp->NumGas; i < Sp->NumPart; i++) {
    if((Sp->P[i].getType() == DUST_PARTICLE_TYPE || Sp->P[i].getType() == 3) &&
       (Sp->P[i].getMass() <= DUST_MASS_TO_DESTROY || Sp->P[i].ID.get() == 0))
      continue;
    if(new_num_part != i) Sp->P[new_num_part] = Sp->P[i];
    if(Sp->P[i].getType() == DUST_PARTICLE_TYPE) {
      if(new_num_part != i) Sp->DustP[new_num_part] = Sp->DustP[i];
    } else {
      memset(&Sp->DustP[new_num_part], 0, sizeof(dust_data));
    }
    new_num_part++;
  }
  Sp->NumPart = new_num_part;
  Sp->NumGas  = new_num_gas;
}
// ═══════════════════════════════════════════════════════════════════════════════
// find_nearest_gas_particle
// ═══════════════════════════════════════════════════════════════════════════════
/**
 * Return the index of the nearest gas particle (type 0) within max_r_kpc,
 * using the pre-built gas_hash for O(1) lookup.
 *
 * Failure modes tracked separately for diagnostics:
 *   fail_no_neighbor  — hash returned -1 (no cell within radius)
 *   fail_too_far      — nearest found is beyond max_r_kpc
 *   fail_wrong_type   — stale hash slot contains a converted non-gas particle
 *
 * Returns -1 on failure; sets *out_dist_kpc if non-NULL.
 */
int find_nearest_gas_particle(simparticles *Sp, int dust_idx,
                               double max_r_kpc, double *out_dist_kpc)
{
  if(out_dist_kpc) *out_dist_kpc = -1.0;
  if(Sp->NumGas == 0 || max_r_kpc <= 0) return -1;
  if(!gas_hash.is_built) return -1;
  HashSearches++;
  // Failure mode counters (static: accumulated over full run on this task)
  static long long fail_no_neighbor = 0;
  static long long fail_too_far     = 0;
  static long long fail_wrong_type  = 0;
  static long long diag_total       = 0;
  double nearest_dist = -1.0;
  int    nearest      = gas_hash.find_nearest_particle(Sp, dust_idx, max_r_kpc, &nearest_dist);
  if(nearest < 0) {
    HashSearchesFailed++;
    fail_no_neighbor++;
    diag_total++;
  } else if(nearest_dist > max_r_kpc) {
    HashSearchesFailed++;
    fail_too_far++;
    diag_total++;
  } else if(Sp->P[nearest].getType() != 0) {
    HashSearchesFailed++;
    fail_wrong_type++;
    diag_total++;
  } else {
    if(out_dist_kpc) *out_dist_kpc = nearest_dist;
    return nearest;
  }
  // Print failure breakdown every 5 million failures
  if(diag_total % 5000000 == 0 && All.ThisTask == 0)
    printf("[HASH_FAIL_DIAG|Step=%d] total_failed=%lld  "
           "no_neighbor=%lld(%.1f%%)  too_far=%lld(%.1f%%)  "
           "wrong_type=%lld(%.1f%%)  max_r=%.2f kpc  "
           "cell_size=%.3f kpc  n_cells=%d\n",
           All.NumCurrentTiStep, diag_total,
           fail_no_neighbor, 100.0*fail_no_neighbor/diag_total,
           fail_too_far,     100.0*fail_too_far/diag_total,
           fail_wrong_type,  100.0*fail_wrong_type/diag_total,
           max_r_kpc, gas_hash.cell_size, gas_hash.n_cells_per_dim);
  return -1;
}
// ═══════════════════════════════════════════════════════════════════════════════
// calculate_velocity_difference
// ═══════════════════════════════════════════════════════════════════════════════
/** Return |v_dust − v_gas| in code velocity units. */
double calculate_velocity_difference(simparticles *Sp, int dust_idx, int gas_idx)
{
  double dv[3] = { Sp->P[dust_idx].Vel[0] - Sp->P[gas_idx].Vel[0],
                   Sp->P[dust_idx].Vel[1] - Sp->P[gas_idx].Vel[1],
                   Sp->P[dust_idx].Vel[2] - Sp->P[gas_idx].Vel[2] };
  return sqrt(dv[0]*dv[0] + dv[1]*dv[1] + dv[2]*dv[2]);
}
// ═══════════════════════════════════════════════════════════════════════════════
// print_dust_statistics
// ═══════════════════════════════════════════════════════════════════════════════
/**
 * Gather and print global dust statistics from all tasks to rank 0.
 *
 * Covers: particle count and mass, grain size/temperature distributions,
 * hash search success rates, destruction pathways (physics vs. internal),
 * growth events, coagulation, shattering.
 *
 * Must be called collectively (all tasks), but only rank 0 prints.
 */
// Cumulative coagulation diagnostics, local to each MPI task.
// Combined across tasks by print_dust_statistics().
static long long CoagCalls = 0;
static long long CoagFailedDensity = 0;
static long long CoagFailedTemperature = 0;
static long long CoagFailedSize = 0;
static long long CoagFailedDTG = 0;
static long long CoagGrown = 0;
static long long CoagLargeUpdates = 0;
static double CoagMaxFractionalIncrease = 0.0;

void print_dust_statistics(simparticles *Sp, MPI_Comm Communicator)
{
  // ── Per-task accumulation ─────────────────────────────────────────────────
  int    local_dust_count = 0;
  double local_dust_mass  = 0.0, local_avg_size = 0.0, local_avg_temp = 0.0;
  int    local_bins[6]    = {0};
  for(int i = 0; i < Sp->NumPart; i++) {
    if(Sp->P[i].getType() != DUST_PARTICLE_TYPE ||
       Sp->P[i].getMass() <= DUST_MASS_TO_DESTROY) continue;
    local_dust_count++;
    local_dust_mass += Sp->P[i].getMass();
    local_avg_size  += Sp->DustP[i].GrainRadius;
    local_avg_temp  += Sp->DustP[i].DustTemperature;
    double T = Sp->DustP[i].DustTemperature;
    if     (T < 10.0)   local_bins[0]++;
    else if(T < 50.0)   local_bins[1]++;
    else if(T < 100.0)  local_bins[2]++;
    else if(T < 500.0)  local_bins[3]++;
    else if(T < 1000.0) local_bins[4]++;
    else                local_bins[5]++;
  }
  // ── MPI reductions ────────────────────────────────────────────────────────
  int    global_dust_count = 0;
  double global_dust_mass  = 0.0, global_avg_size = 0.0, global_avg_temp = 0.0;
  int    global_bins[6]    = {0};
  MPI_Reduce(&local_dust_count, &global_dust_count, 1, MPI_INT,    MPI_SUM, 0, Communicator);
  MPI_Reduce(&local_dust_mass,  &global_dust_mass,  1, MPI_DOUBLE, MPI_SUM, 0, Communicator);
  MPI_Reduce(&local_avg_size,   &global_avg_size,   1, MPI_DOUBLE, MPI_SUM, 0, Communicator);
  MPI_Reduce(&local_avg_temp,   &global_avg_temp,   1, MPI_DOUBLE, MPI_SUM, 0, Communicator);
  MPI_Reduce(local_bins, global_bins, 6, MPI_INT, MPI_SUM, 0, Communicator);
  long long g_NDustCreated              = 0, g_NDustCreatedBySNII       = 0;
  long long g_NDustCreatedByAGB         = 0;
  long long g_NDustCreatedByLRN         = 0;
  long long g_NDustDestrByThermal       = 0, g_NDustDestrByShock        = 0;
  long long g_NDustDestrByAstration     = 0, g_NDustDestrByCleanup      = 0;
  long long g_NDustDestrByCorruption    = 0, g_NDustDestrByBadGas       = 0;
  long long g_NGrowth                   = 0, g_NErosion                 = 0;
  long long g_NCoag                     = 0;
  long long g_HashSearches              = 0, g_HashFailed               = 0;
  long long g_NShatter                  = 0;
  double g_MassGrown                    = 0.0;
  double g_MassDestrThermal             = 0.0, g_MassDestrShock         = 0.0;
  double g_MassErodThermal              = 0.0, g_MassErodShock          = 0.0;
  double g_MassAstrated                 = 0.0;
  double g_SizeReductionShattering      = 0.0;
  MPI_Reduce(&NDustCreated,              &g_NDustCreated,              1, MPI_LONG_LONG, MPI_SUM, 0, Communicator);
  MPI_Reduce(&NDustCreatedBySNII,        &g_NDustCreatedBySNII,        1, MPI_LONG_LONG, MPI_SUM, 0, Communicator);
  MPI_Reduce(&NDustCreatedByAGB,         &g_NDustCreatedByAGB,         1, MPI_LONG_LONG, MPI_SUM, 0, Communicator);
  MPI_Reduce(&NDustCreatedByLRN,         &g_NDustCreatedByLRN,         1, MPI_LONG_LONG, MPI_SUM, 0, Communicator);
  MPI_Reduce(&NDustDestroyedByThermal,   &g_NDustDestrByThermal,       1, MPI_LONG_LONG, MPI_SUM, 0, Communicator);
  MPI_Reduce(&NDustDestroyedByShock,     &g_NDustDestrByShock,         1, MPI_LONG_LONG, MPI_SUM, 0, Communicator);
  MPI_Reduce(&NDustDestroyedByAstration, &g_NDustDestrByAstration,     1, MPI_LONG_LONG, MPI_SUM, 0, Communicator);
  MPI_Reduce(&NDustDestroyedByCleanup,   &g_NDustDestrByCleanup,       1, MPI_LONG_LONG, MPI_SUM, 0, Communicator);
  MPI_Reduce(&NDustDestroyedByCorruption,&g_NDustDestrByCorruption,    1, MPI_LONG_LONG, MPI_SUM, 0, Communicator);
  MPI_Reduce(&NDustDestroyedByBadGasIndex,&g_NDustDestrByBadGas,       1, MPI_LONG_LONG, MPI_SUM, 0, Communicator);
  MPI_Reduce(&NGrainGrowthEvents,        &g_NGrowth,                   1, MPI_LONG_LONG, MPI_SUM, 0, Communicator);
  MPI_Reduce(&NGrainErosionEvents,       &g_NErosion,                  1, MPI_LONG_LONG, MPI_SUM, 0, Communicator);
  MPI_Reduce(&NCoagulationEvents,        &g_NCoag,                     1, MPI_LONG_LONG, MPI_SUM, 0, Communicator);
  MPI_Reduce(&NShatteringEvents,         &g_NShatter,                  1, MPI_LONG_LONG, MPI_SUM, 0, Communicator);
  MPI_Reduce(&HashSearches,              &g_HashSearches,              1, MPI_LONG_LONG, MPI_SUM, 0, Communicator);
  MPI_Reduce(&HashSearchesFailed,        &g_HashFailed,                1, MPI_LONG_LONG, MPI_SUM, 0, Communicator);
  MPI_Reduce(&TotalMassGrown,            &g_MassGrown,                 1, MPI_DOUBLE,    MPI_SUM, 0, Communicator);
  MPI_Reduce(&TotalMassDestroyedByThermal,&g_MassDestrThermal,         1, MPI_DOUBLE,    MPI_SUM, 0, Communicator);
  MPI_Reduce(&TotalMassDestroyedByShock, &g_MassDestrShock,            1, MPI_DOUBLE,    MPI_SUM, 0, Communicator);
  MPI_Reduce(&TotalMassErodedByThermal,  &g_MassErodThermal,           1, MPI_DOUBLE,    MPI_SUM, 0, Communicator);
  MPI_Reduce(&TotalMassErodedByShock,    &g_MassErodShock,             1, MPI_DOUBLE,    MPI_SUM, 0, Communicator);
  MPI_Reduce(&TotalDustMassAstrated,     &g_MassAstrated,              1, MPI_DOUBLE,    MPI_SUM, 0, Communicator);
  MPI_Reduce(&TotalSizeReductionShattering,&g_SizeReductionShattering, 1, MPI_DOUBLE,    MPI_SUM, 0, Communicator);
  // Global coagulation diagnostics (cumulative since initialization).
  long long coag_local[7] = {
      CoagCalls, CoagFailedDensity, CoagFailedTemperature,
      CoagFailedSize, CoagFailedDTG, CoagGrown, CoagLargeUpdates
  };
  long long coag_global[7] = {0};
  double coag_max_global = 0.0;

  MPI_Reduce(coag_local, coag_global, 7, MPI_LONG_LONG,
             MPI_SUM, 0, Communicator);
  MPI_Reduce(&CoagMaxFractionalIncrease, &coag_max_global, 1,
             MPI_DOUBLE, MPI_MAX, 0, Communicator);

  // Print coagulation histogram (every 500 steps, all tasks)
  print_coag_histogram(Communicator);
  if(All.ThisTask != 0) return;
  if(global_dust_count > 0) {
    global_avg_size /= global_dust_count;
    global_avg_temp /= global_dust_count;
  }
  // Convert to Msun
  const double mass_to_msun = All.UnitMass_in_g / 1.989e33 / All.HubbleParam;
  global_dust_mass    *= mass_to_msun;
  g_MassGrown         *= mass_to_msun;
  g_MassDestrThermal  *= mass_to_msun;
  g_MassErodThermal   *= mass_to_msun;
  g_MassDestrShock    *= mass_to_msun;
  g_MassErodShock     *= mass_to_msun;
  g_MassAstrated      *= mass_to_msun;
  DUST_PRINT("=== STATISTICS (global) ===\n");
  DUST_PRINT("STATISTICS Particles: %d  Mass: %.3e Msun\n", global_dust_count, global_dust_mass);
  DUST_PRINT("STATISTICS Avg grain size: %.2f nm\n", global_avg_size);
  DUST_PRINT("STATISTICS Avg temperature: %.1f K\n", global_avg_temp);
  if(global_dust_count > 0) {
    DUST_PRINT("STATISTICS  < 10 K (CMB floor):        %d (%.1f%%)\n", global_bins[0], 100.0*global_bins[0]/global_dust_count);
    DUST_PRINT("STATISTICS  10-50 K (Cold clouds):     %d (%.1f%%)\n", global_bins[1], 100.0*global_bins[1]/global_dust_count);
    DUST_PRINT("STATISTICS  50-100 K (Cool ISM):       %d (%.1f%%)\n", global_bins[2], 100.0*global_bins[2]/global_dust_count);
    DUST_PRINT("STATISTICS  100-500 K (Warm ISM):      %d (%.1f%%)\n", global_bins[3], 100.0*global_bins[3]/global_dust_count);
    DUST_PRINT("STATISTICS  500-1000 K (Hot ISM):      %d (%.1f%%)\n", global_bins[4], 100.0*global_bins[4]/global_dust_count);
    DUST_PRINT("STATISTICS  1000-2000 K (Near sublim): %d (%.1f%%)\n", global_bins[5], 100.0*global_bins[5]/global_dust_count);
  }
  DUST_PRINT("========================\n");
  DUST_PRINT("STATISTICS Hash searches: %lld\n", g_HashSearches);
  if(g_HashSearches > 0)
    DUST_PRINT("STATISTICS Hash success rate: %.1f%%\n",
               100.0 * (g_HashSearches - g_HashFailed) / g_HashSearches);
  if(g_HashFailed > 0)
    DUST_PRINT("STATISTICS [WARNING] Failed searches: %lld (%.1f%%)\n",
               g_HashFailed, 100.0 * g_HashFailed / g_HashSearches);
  DUST_PRINT("STATISTICS Growth events: %lld (%.2e Msun grown)\n", g_NGrowth, g_MassGrown);
  DUST_PRINT("STATISTICS Partial erosion events: %lld\n", g_NErosion);
  DUST_PRINT("========================\n");
  long long total_by_physics  = g_NDustDestrByThermal + g_NDustDestrByShock + g_NDustDestrByAstration;
  long long total_by_internal = g_NDustDestrByCleanup + g_NDustDestrByCorruption + g_NDustDestrByBadGas;
  long long total_destroyed   = total_by_physics + total_by_internal;
  DUST_PRINT("=== DESTRUCTION AUDIT (global) ===\n");
  DUST_PRINT("  --- Physics mechanisms ---\n");
  DUST_PRINT("  Thermal sputtering:     %lld  (flag=%d)\n"
             "    full destructions:    %.2e Msun\n"
             "    partial erosion:      %.2e Msun\n",
             g_NDustDestrByThermal, All.DustEnableSputtering,
             g_MassDestrThermal, g_MassErodThermal);
  DUST_PRINT("  Shock destruction:      %lld  (flag=%d)\n"
             "    full destructions:    %.2e Msun\n"
             "    partial erosion:      %.2e Msun\n",
             g_NDustDestrByShock, All.DustEnableShockDestruction,
             g_MassDestrShock, g_MassErodShock);
  DUST_PRINT("  Astration:              %lld  (flag=%d, mass=%.2e Msun)\n",
             g_NDustDestrByAstration, All.DustEnableAstration, g_MassAstrated);
  DUST_PRINT("  --- Internal paths (should be 0 in a clean run) ---\n");
  DUST_PRINT("  cleanup_invalid():      %lld  ← domain exchange corruption?\n", g_NDustDestrByCleanup);
  DUST_PRINT("  growth corruption:      %lld  ← zeroed DustP in growth loop\n", g_NDustDestrByCorruption);
  DUST_PRINT("  bad gas index (stale):  %lld  ← hash returned non-gas type\n", g_NDustDestrByBadGas);
  DUST_PRINT("  --- Totals ---\n");
  DUST_PRINT("  By physics:             %lld\n", total_by_physics);
  DUST_PRINT("  By internal paths:      %lld\n", total_by_internal);
  DUST_PRINT("  GRAND TOTAL destroyed:  %lld\n", total_destroyed);
  DUST_PRINT("  Total created:          %lld  (SNII=%lld  AGB=%lld  LRN=%lld)\n",
             g_NDustCreated, g_NDustCreatedBySNII, g_NDustCreatedByAGB, g_NDustCreatedByLRN);
  DUST_PRINT("  Net (created-destroyed):%lld\n", g_NDustCreated - total_destroyed);
  DUST_PRINT("  Current live particles: %d\n",  global_dust_count);
  DUST_PRINT("=======================================\n");
  if(global_dust_count > 0 && g_NDustDestrByShock > 0)
    DUST_PRINT("  Shock events / live particles: %.3f  (target < 0.5)\n",
               (double)g_NDustDestrByShock / global_dust_count);
  DUST_PRINT("STATISTICS Coagulation events: %lld\n", g_NCoag);
  DUST_PRINT("[COAG_GLOBAL] calls=%lld dens_reject=%lld temp_reject=%lld "
             "size_reject=%lld dtg_reject=%lld grown=%lld subcycled=%lld "
             "max_requested_da_over_a=%.3e\n",
             coag_global[0], coag_global[1], coag_global[2],
             coag_global[3], coag_global[4], coag_global[5], coag_global[6],
             coag_max_global);

  DUST_PRINT("STATISTICS Shattering events: %lld (avg da=%.2f nm/event)\n",
             g_NShatter,
             g_NShatter > 0 ? g_SizeReductionShattering / g_NShatter : 0.0);
  DUST_PRINT("========================\n");
}
// ═══════════════════════════════════════════════════════════════════════════════
// analyze_dust_gas_coupling_local
// ═══════════════════════════════════════════════════════════════════════════════
/**
 * Sample the dust-gas velocity offset on task 0 and print average and max.
 *
 * Only every 100th particle is sampled to bound the cost — at 16M dust
 * particles split across ~240 tasks, task 0 holds ~67k particles, so a full
 * loop would do ~67k hash searches purely for diagnostics. Sampling 1%
 * (~670 particles) gives statistically representative output at negligible cost.
 *
 * This is a local (task 0 only) diagnostic; for a global MPI-reduced velocity
 * statistic, see the [COUPLING] line in print_dust_statistics().
 */
void analyze_dust_gas_coupling_local(simparticles *Sp)
{
  if(All.ThisTask != 0) return;
  double total_vel_diff = 0.0;
  double max_vel_diff   = 0.0;
  int    dust_count     = 0;
  for(int i = 0; i < Sp->NumPart; i++) {
    // Sample every 100th particle to keep diagnostic cost O(N/100)
    if(i % 100 != 0) continue;
    if(Sp->P[i].getType() != DUST_PARTICLE_TYPE ||
       Sp->P[i].getMass() <= DUST_MASS_TO_DESTROY) continue;
    int nearest_gas = find_nearest_gas_particle(Sp, i, 10.0, NULL);
    if(nearest_gas < 0) continue;
    double vd = calculate_velocity_difference(Sp, i, nearest_gas);
    total_vel_diff += vd;
    if(vd > max_vel_diff) max_vel_diff = vd;
    dust_count++;
  }
  if(dust_count > 0)
    DUST_PRINT("[COUPLING] %d sampled particles (1%%), avg Δv=%.3f, max=%.3f (Rank 0)\n",
               dust_count, total_vel_diff / dust_count, max_vel_diff);
}
// ═══════════════════════════════════════════════════════════════════════════════
// calculate_sn_shock_radius
// ═══════════════════════════════════════════════════════════════════════════════
/**
 * Compute the Sedov-Taylor shock radius at time_myr after the SN explosion.
 *
 * R(t) = ξ × [E t² / ρ]^(1/5)    (Sedov 1959, ξ = 1.033)
 *
 * Returns radius in kpc.
 */
double calculate_sn_shock_radius(double sn_energy_erg, double gas_density_cgs,
                                  double time_myr)
{
  const double xi     = 1.033;
  double time_sec     = time_myr * 1e6 * SEC_PER_YEAR;
  double radius_cm    = xi * pow(sn_energy_erg * time_sec * time_sec / gas_density_cgs, 0.2);
  return radius_cm / (1000.0 * PARSEC);  // cm → kpc
}
// ═══════════════════════════════════════════════════════════════════════════════
// get_shock_destruction_efficiency
// ═══════════════════════════════════════════════════════════════════════════════
/**
 * Grain destruction efficiencies as a function of shock velocity.
 *
 * Piecewise-linear interpolation of Bocchio et al. (2014), Table 6.
 *
 * Carbonaceous and silicate destruction efficiencies are returned
 * separately so that the two components of a mixed grain can be eroded
 * independently. This allows CarbonMassFraction to evolve naturally during
 * SN shock processing.
 *
 * Key values:
 *   Carbonaceous at 50 km/s: 0.77
 *   Silicate at 50 km/s:     0.02
 *
 * Carbonaceous grains approach complete destruction by ~175 km/s.
 * The silicate curve reaches 0.67 at 175 km/s and is held constant
 * above 200 km/s because the Bocchio et al. table ends there.
 */
void get_shock_destruction_efficiency(double v,
                                      double *eps_carb,
                                      double *eps_sil)
{
  // ── Carbonaceous curve ────────────────────────────────────────────────────
  if     (v <  50.0) *eps_carb = 0.0;
  else if(v <  75.0) *eps_carb = 0.77 + 0.06 * (v -  50.0) / 25.0;
  else if(v < 100.0) *eps_carb = 0.83 + 0.08 * (v -  75.0) / 25.0;
  else if(v < 125.0) *eps_carb = 0.91 + 0.05 * (v - 100.0) / 25.0;
  else if(v < 150.0) *eps_carb = 0.96 + 0.03 * (v - 125.0) / 25.0;
  else if(v < 175.0) *eps_carb = 0.99 + 0.01 * (v - 150.0) / 25.0;
  else               *eps_carb = 1.00;
  // ── Silicate curve ────────────────────────────────────────────────────────
  if     (v <  50.0) *eps_sil = 0.0;
  else if(v <  75.0) *eps_sil = 0.02 + 0.10 * (v -  50.0) / 25.0;
  else if(v < 100.0) *eps_sil = 0.12 + 0.17 * (v -  75.0) / 25.0;
  else if(v < 125.0) *eps_sil = 0.29 + 0.17 * (v - 100.0) / 25.0;
  else if(v < 150.0) *eps_sil = 0.46 + 0.07 * (v - 125.0) / 25.0;
  else if(v < 175.0) *eps_sil = 0.53 + 0.14 * (v - 150.0) / 25.0;
  else if(v < 200.0) *eps_sil = 0.67;
  else               *eps_sil = 0.67;
  // Numerical safety
  *eps_carb = std::max(0.0, std::min(1.0, *eps_carb));
  *eps_sil  = std::max(0.0, std::min(1.0, *eps_sil));
}
// ── Shared grain-size vulnerability factor ────────────────────────────────
// Smaller grains have more surface area per unit mass and are more easily
// destroyed in a shock; larger grains are comparatively harder to destroy.
// Used to weight per-grain shock mass-loss.
static inline double shock_size_factor(double a_nm)
{
  if     (a_nm < 20.0)  return 1.5;
  else if(a_nm < 50.0)  return 1.2;
  else if(a_nm > 100.0) return 0.7;
  else                  return 1.0;
}
// ═══════════════════════════════════════════════════════════════════════════════
// erode_dust_from_sn_shocks
// ═══════════════════════════════════════════════════════════════════════════════
/**
 * Apply subgrid SN shock erosion and destruction to dust within the blast radius.
 *
 * MOTIVATION:
 *   At 2048³, the physical Sedov-Taylor shock radius (~20–50 pc) is smaller
 *   than the dust hash cell size (~1–7 kpc). Direct grain-by-grain destruction
 *   at the physical radius would find zero grains per event. The subgrid
 *   treatment computes the EXPECTED mass destroyed within the physical shock
 *   volume and distributes it proportionally across all dust in a wider search
 *   volume.
 *
 * ALGORITHM:
 *   1. Find local gas density → compute physical shock radius R_phys and
 *      shock velocity v_shock from Sedov-Taylor at t = 0.3 Myr.
 *   2. Set effective_search_radius = max(R_phys, 2 × dust hash cell size).
 *   3. Find all dust within effective_search_radius.
 *   4. Compute f_vol = (R_phys / R_search)³ — volume correction factor.
 *   5. Split local dust mass into carbonaceous and silicate components.
 *      Compute species-resolved destruction budgets:
 *
 *        M_C,destroy   = M_C,local   × f_vol × ε_C(v_shock)
 *        M_sil,destroy = M_sil,local × f_vol × ε_sil(v_shock)
 *
 *   6. Distribute each species' destruction budget using its own
 *      size-weighted mass contribution.
 *   7. Recombine the surviving species masses, update CarbonMassFraction,
 *      and update grain radius via a ∝ m^(1/3).
 *
 * RESOLUTION INDEPENDENCE:
 *   At coarse resolution, few grains are found but f_vol ~ 1 (cell ≈ shock).
 *   At fine resolution, many grains are found but f_vol << 1 (cell << shock).
 *   In both cases M_destroy converges to the same physical value.
 *
 * CRITICAL: shock velocity is always derived from R_phys, NOT from
 * effective_search_radius. Using the inflated search radius would reduce v
 * by a factor ~(R_search/R_phys)^(3/2) >> 1, dropping it below destruction
 * thresholds.
 */
void erode_dust_from_sn_shocks(simparticles *Sp, int sn_star_idx,
                                   double sn_energy, MPI_Comm Communicator)
{
  if(!All.DustEnableShockDestruction) return;
  static long long sn_total_calls = 0, sn_found_dust = 0;
  sn_total_calls++;
  if(sn_total_calls % 1000 == 0)
    DUST_PRINT("[SN_RATE] %lld SN calls, %lld found dust (%.1f%%)\n",
               sn_total_calls, sn_found_dust, 100.0 * sn_found_dust / sn_total_calls);
  // ── Guard: hash must be built ─────────────────────────────────────────────
  static long long sn_skipped_no_hash = 0;
  if(!dust_hash.is_built || dust_hash.total_particles == 0) {
    sn_skipped_no_hash++;
    if(sn_skipped_no_hash <= 20 && All.ThisTask == 0)
      DUST_PRINT("[SN_SKIP] Call skipped: hash built=%d total=%d\n",
                 dust_hash.is_built, dust_hash.total_particles);
    return;
  }

  // ── Step 1: Local gas density ─────────────────────────────────────────────
  double nearest_dist   = -1.0;
  int    sn_nearest_gas = gas_hash.find_nearest_particle(Sp, sn_star_idx, 50.0, &nearest_dist);
  if(sn_nearest_gas >= 0 && Sp->P[sn_nearest_gas].getType() != 0)
    sn_nearest_gas = -1;
  double gas_density_cgs = 1.0 * PROTONMASS;  // fallback: ~1 cm^-3
  if(sn_nearest_gas >= 0 && sn_nearest_gas < Sp->NumGas) {
    double measured = dust_gas_density_cgs(Sp, sn_nearest_gas);
    if(measured > 0.01 * PROTONMASS) gas_density_cgs = measured;
  }

  // ── Step 2: Physical shock radius at characteristic time 0.3 Myr ─────────
  // sn_energy arrives in code energy units (matches feedback.cc's E_code
  // convention); convert to erg for the Sedov-Taylor calculation. Fallback
  // only guards against a malformed call — should not trigger from feedback.cc.
  // Code energy unit is UnitEnergy_in_cgs / h erg.
  double sn_energy_erg = sn_energy * All.UnitEnergy_in_cgs / All.HubbleParam;
  if(sn_energy_erg <= 0.0 || !isfinite(sn_energy_erg)) {
    static long long bad_energy_warns = 0;
    bad_energy_warns++;
    if(bad_energy_warns <= 20 && All.ThisTask == 0)
      printf("[SN_SHOCK_WARNING|T=%d] Invalid sn_energy=%.3e code units passed to "
             "destroy_dust_from_sn_shocks — falling back to 1e51 erg\n",
             All.ThisTask, sn_energy);
    sn_energy_erg = 1e51;
  }
  double rho_sedov = std::min(gas_density_cgs,
                               All.DustShockAmbientDensity * PROTONMASS);
  double physical_radius_kpc = calculate_sn_shock_radius(sn_energy_erg, rho_sedov, 0.3);
  if(physical_radius_kpc < 0.001) physical_radius_kpc = 0.001;  // 1 pc floor

  // ── Step 3: Shock velocity from PHYSICAL radius ───────────────────────────
  // Must use physical radius — see header comment for why using search radius
  // would neuterise the destruction. Must also use the SAME sn_energy_erg as
  // the radius calculation above — R(t) and v(R) are only a valid Sedov-Taylor
  // pair when computed from the same explosion energy.
  double shock_velocity_km_s = calculate_sedov_velocity_from_radius(physical_radius_kpc,
                                                                      rho_sedov, sn_energy_erg);

  // ── Step 4: Effective search radius for hash lookup ───────────────────────
  double effective_search_radius = physical_radius_kpc;
  if(effective_search_radius < 2.0 * dust_hash.cell_size)
    effective_search_radius = 2.0 * dust_hash.cell_size;
  if(effective_search_radius > 3.0) effective_search_radius = 3.0;

  // ── Step 5: Diagnostic logging ────────────────────────────────────────────
  if(All.DustDebugLevel > 0) {
    static int sn_call_count = 0;
    sn_call_count++;
    if((sn_call_count <= 20 || sn_call_count % 500 == 0) && All.ThisTask == 0)
      DUST_PRINT("[SN_SHOCK_DEBUG] Call #%d: "
                 "physical_r=%.3f kpc  search_r=%.3f kpc  "
                 "v_shock=%.1f km/s  rho_local=%.3e  rho_sedov=%.3e %s\n",
                 sn_call_count, physical_radius_kpc, effective_search_radius,
                 shock_velocity_km_s, gas_density_cgs, rho_sedov,
                 (rho_sedov < gas_density_cgs) ? "[DENSITY_CAPPED]" : "[LOCAL_DENSITY]");
  }

  // ── Step 6: Find dust in search volume ───────────────────────────────────
  int    neighbors[2048];
  double distances[2048];
  int    n_found = 0;
  dust_hash.find_neighbors(Sp, sn_star_idx, effective_search_radius,
                           neighbors, distances, &n_found, 2048);
  if(n_found == 0) return;

  // ── Step 7: Species-resolved mass-weighted subgrid approach ────────────────
  double M_carb_local = 0.0;
  double M_sil_local  = 0.0;
  double weight_carb_sum = 0.0;
  double weight_sil_sum  = 0.0;
  int n_dust_found = 0;
  for(int k = 0; k < n_found; k++) {
    int i = neighbors[k];
    if(Sp->P[i].getType() != DUST_PARTICLE_TYPE ||
      Sp->P[i].getMass() <= 0.0)
      continue;
    double m  = Sp->P[i].getMass();
    // make sure fraction is in [0,1] range to avoid trouble
    double cf = std::max(0.0, std::min(1.0,
                                      (double)Sp->DustP[i].CarbonMassFraction));
    double sf = shock_size_factor(Sp->DustP[i].GrainRadius);
    double M_carb = m * cf;
    double M_sil  = m * (1.0 - cf);
    M_carb_local += M_carb;
    M_sil_local  += M_sil;
    weight_carb_sum += M_carb * sf;
    weight_sil_sum  += M_sil  * sf;
    n_dust_found++;
  }
  double M_dust_local = M_carb_local + M_sil_local;
  if(M_dust_local <= 0.0 || n_dust_found == 0)
    return;
  // ── Species-resolved shock destruction budgets ─────────────────────────────
  double f_vol = std::max(0.0, std::min(1.0,
                    pow(physical_radius_kpc / effective_search_radius, 3.0)));
  double eps_carb, eps_sil;
  get_shock_destruction_efficiency(shock_velocity_km_s,
                                  &eps_carb,
                                  &eps_sil);
  double M_carb_destroy = M_carb_local * f_vol * eps_carb;
  double M_sil_destroy = M_sil_local * f_vol * eps_sil;
  // ── Step 8: Distribute mass destruction across found grains ──────────────
  if(n_dust_found > 0) sn_found_dust++;
  int    dust_destroyed  = 0;
  int    dust_eroded     = 0;
  double M_actually_lost = 0.0;
  if(M_carb_destroy > 0.0 || M_sil_destroy > 0.0) {
    for(int k = 0; k < n_found; k++) {
      int i = neighbors[k];
      if(Sp->P[i].getType() != DUST_PARTICLE_TYPE)
        continue;
      double m = Sp->P[i].getMass();
      if(m <= 0.0)
        continue;
      // Handle carbonaceous and silicate mass separately so CarbonMassFraction
      // evolves naturally under composition-dependent shock destruction...
      double a_old  = Sp->DustP[i].GrainRadius;
      double cf_old = std::max(0.0, std::min(1.0,(double)Sp->DustP[i].CarbonMassFraction));
      double M_carb_old = m * cf_old;
      double M_sil_old  = m * (1.0 - cf_old);
      double sf = shock_size_factor(a_old);
      double weight_carb = M_carb_old * sf;
      double weight_sil  = M_sil_old  * sf;
      double carb_loss = (weight_carb_sum > 0.0) ? M_carb_destroy * (weight_carb / weight_carb_sum) : 0.0;
      double sil_loss = (weight_sil_sum > 0.0) ? M_sil_destroy * (weight_sil / weight_sil_sum) : 0.0;
      carb_loss = std::min(carb_loss, M_carb_old);
      sil_loss  = std::min(sil_loss, M_sil_old);
      double M_carb_new = M_carb_old - carb_loss;
      double M_sil_new  = M_sil_old  - sil_loss;
      double mass_loss = carb_loss + sil_loss;
      double new_mass  = M_carb_new + M_sil_new;
      double CF_new = (new_mass > 0.0) ? M_carb_new / new_mass : cf_old;
      double a_new = 0.0;

      if(new_mass > 0.0)
        a_new = a_old * cbrt(new_mass / m);

      if(new_mass <= 0.0 ||
        !isfinite(new_mass) ||
        !isfinite(a_new) ||
        a_new < All.DustMinGrainSize) {
          log_dust_particle_event(Sp, i, sn_nearest_gas, DUST_EVENT_SHOCK);
          M_actually_lost += m;
          destroy_dust_particle_to_gas(Sp, i, sn_nearest_gas,
                                      &NDustDestroyedByShock, &TotalMassDestroyedByShock);
          dust_destroyed++;
      } else {
        Sp->P[i].setMass(new_mass);
        Sp->DustP[i].GrainRadius = a_new;
        Sp->DustP[i].CarbonMassFraction = (float)CF_new;

        if(sn_nearest_gas >= 0)
          return_dust_material_to_gas(Sp, sn_nearest_gas, carb_loss, sil_loss);

        LocalDustMassChange    -= mass_loss;
        M_actually_lost        += mass_loss;
        TotalMassErodedByShock += mass_loss;
        NGrainErosionEvents++;
        dust_eroded++;
      }
    }
  }
  // ── Diagnostics ─────────────────────────────────────────────────────────────
  double mass_to_msun = All.UnitMass_in_g / 1.989e33 / All.HubbleParam;
  double M_target     = M_carb_destroy + M_sil_destroy;
  if(All.ThisTask == 0 && M_target > 0.0)
    DUST_PRINT("[DUST_SN] physical_r=%.4f kpc  search_r=%.3f kpc  "
              "v=%.1f km/s  f_vol=%.3e  "
              "eps_C=%.3f eps_sil=%.3f  "
              "M_C=%.3e M_sil=%.3e  "
              "M_target_C=%.3e M_target_sil=%.3e  "
              "M_target=%.3e code (%.3e Msun)  "
              "M_lost=%.3e code (%.3e Msun)  "
              "n_dust=%d destroyed=%d eroded=%d\n",
              physical_radius_kpc, effective_search_radius,
              shock_velocity_km_s, f_vol,
              eps_carb, eps_sil,
              M_carb_local, M_sil_local,
              M_carb_destroy, M_sil_destroy,
              M_target, M_target * mass_to_msun,
              M_actually_lost, M_actually_lost * mass_to_msun,
              n_dust_found, dust_destroyed, dust_eroded);
}
// ═══════════════════════════════════════════════════════════════════════════════
// HK11 accretion timescale helper
// ═══════════════════════════════════════════════════════════════════════════════
/**
 * Return the Hirashita & Kuo (2011) grain accretion timescale in years.
 *
 * Eq. from HK11; prefactor differs by species:
 *   silicate  (species=0): pref = 6.30e7 yr
 *   carbonaceous (species=1): pref = 5.59e7 yr
 *
 * τ_acc = pref × (a/0.1 µm) × (Zsun/Z) × (1000/nH) × (50K/T)^0.5 × (0.3/S)
 *
 * Returns HUGE_VAL on invalid input to allow the caller to bail out cleanly.
 */
static inline double tau_acc_yr_HK11(double nH_cm3, double T_K,
                                      double Z_massfrac, double Zsun_massfrac,
                                      double a_cm, double S,
                                      int species)
{
  const double pref = (species == 0) ? 6.30e7 : 5.59e7;
  if(nH_cm3 <= 0 || T_K <= 0 || Z_massfrac <= 0 || Zsun_massfrac <= 0 ||
     a_cm <= 0  || S <= 0)
    return HUGE_VAL;
  const double a_um  = a_cm * 1e4;
  const double a01   = a_um  / 0.1;
  const double n3    = nH_cm3 / 1e3;
  const double T50   = T_K    / 50.0;
  const double Zrat  = Z_massfrac / Zsun_massfrac;
  const double S03   = S / 0.3;
  return pref * a01 * (1.0/Zrat) * (1.0/n3) * pow(T50, -0.5) * (1.0/S03);
}
// ═══════════════════════════════════════════════════════════════════════════════
// dust_clumping_factor
// ═══════════════════════════════════════════════════════════════════════════════
/**
 * Return the subgrid clumping factor C for gas at hydrogen number density n_H.
 *
 * The clumping factor accounts for unresolved ISM density structure that
 * enhances grain growth rates. Dense molecular gas is highly clumped (C~30);
 * diffuse gas is essentially smooth (C~1).
 *
 * The density thresholds are expressed as fractions of CritPhysDensity (the
 * star formation threshold) so they scale consistently across resolution levels.
 *
 * Values: {1.5, 3, 10, 30} are fixed regardless of resolution
 */
double dust_clumping_factor(double n_H, int is_star_forming)
{
  if(!All.DustEnableClumping) return 1.0;
  if(is_star_forming) return 30.0;
  const double n_sf = All.CritPhysDensity;
  if(n_H > 0.5  * n_sf) return 10.0;
  if(n_H > 0.2  * n_sf) return 3.0;
  if(n_H > 0.05 * n_sf) return 1.5;
  return 1.0;
}

// ============================================================================
// DUST COAGULATION: continuous-radius, local-DTG Trayford-inspired model
// ============================================================================
// Literature basis:
//   Trayford et al. (2025), MNRAS, doi:10.1093/mnras/staf2040,
//   Section 3.5.2, Equation (10), adapting Aoyama et al. (2017).
//
//   tau_co = 27.1 Myr * (a / 0.1 micron) * (DTG / 0.01)^(-1)
//          * (v_co / 0.1 km/s)^(-1) * (n_H_eff / 10 cm^-3)^(-1).
//
// Trayford uses a two-size-bin mass-transfer model (a_S = 0.01 micron)
// and v_co = 0.2 km/s. CosmicGrain instead follows a continuous radius
// for each fixed-mass dust superparticle. This is NOT Trayford's exact
// two-bin algorithm or a full grain-grain collision-kernel solution.
//
// DTG is estimated from dust and gas superparticle masses within the local
// association sphere, combined across MPI ranks and clamped to [1e-6, 0.1].
// DustCoagulationCalibration multiplies the timescale (1 = uncalibrated).
//
// Representative-grain closure: da/dt = a/(3*tau_co). Because tau_co is
// proportional to a at fixed environment, integrate exactly over dt:
//   a_new = a_old * (1 + dt/(3*tau_co(a_old))).
// Each numerical substep is limited to 20% radius growth; the full timestep
// is integrated unless the physical maximum radius is reached.
// Superparticle mass and composition are unchanged: microscopic grain
// number decreases as radius increases. No fictitious carbon mixing.
// ============================================================================
void dust_grain_coagulation(simparticles *Sp, int dust_idx, int gas_idx, double dt,
                            double local_DTG, double T_gas)
{
  if(!All.DustEnableCoagulation || !(dt > 0.0) || !isfinite(dt)) return;

  ++CoagCalls;

  const double n_H = dust_gas_nH_cgs(Sp, gas_idx);
  const bool is_star_forming = Sp->SphP[gas_idx].Sfr > DUST_SFR_EPS;
  const double clumping = dust_clumping_factor(n_H, is_star_forming);
  const double n_eff = n_H * clumping;
  if(!isfinite(n_eff) || n_eff < All.DustCollisionDensityThresh ||
     !(n_eff > 0.0)) { ++CoagFailedDensity; return; }

  // Effective-EOS star-forming temperatures do not resolve cold clouds.
  // Retain the original T < 3000 K gate for non-star-forming gas.
  if(!is_star_forming && (!isfinite(T_gas) || T_gas > 3000.0)) {
    ++CoagFailedTemperature;
    return;
  }

  const double a_nm = Sp->DustP[dust_idx].GrainRadius;
  const double M_dust = Sp->P[dust_idx].getMass();
  if(!isfinite(a_nm) || !(a_nm > 0.0) || !isfinite(M_dust) ||
     !(M_dust > 0.0) ||
     a_nm >= All.DustMaxGrainSize) {
    ++CoagFailedSize;
    return;
  }
  if(!isfinite(local_DTG) || !(local_DTG > 0.0)) {
    ++CoagFailedDTG;
    return;
  }

  // Trayford et al. (2025), Eq. 10; DTG is the MPI-combined mass ratio
  // inside the resolved dust-gas association sphere.
  const double v_co_kms = 0.2;
  const double tau0_yr = 27.1e6;
  const double calibration = All.DustCoagulationCalibration;
  if(!isfinite(calibration) || !(calibration > 0.0)) return;
  const double tau_co_yr = tau0_yr * (a_nm / 100.0)
      * (0.01 / local_DTG) * (0.1 / v_co_kms)
      * (10.0 / n_eff) * calibration;
  if(!isfinite(tau_co_yr) || !(tau_co_yr > 0.0)) return;

  // update_dust_dynamics() receives elapsed physical time in code units.
  const double tau_co_code = tau_co_yr * SEC_PER_YEAR / dust_code_time_in_s();
  if(!isfinite(tau_co_code) || !(tau_co_code > 0.0)) return;

  // Subcycle so no numerical increment exceeds 20%. Since tau_co is
  // proportional to radius, this reproduces the exact fixed-environment
  // continuous-radius solution while preserving a bounded per-step update.
  const double fractional_increase = dt / (3.0 * tau_co_code);
  if(!isfinite(fractional_increase) || !(fractional_increase > 0.0))
    return;

  CoagMaxFractionalIncrease =
      std::max(CoagMaxFractionalIncrease, fractional_increase);

  const bool subcycled = fractional_increase > 0.2;
  double remaining_dt = dt;
  double a_new = a_nm;
  while(remaining_dt > 0.0 && a_new < All.DustMaxGrainSize) {
    const double tau_substep_code = tau_co_code * (a_new / a_nm);
    if(!isfinite(tau_substep_code) || !(tau_substep_code > 0.0))
      return;
    const double substep_dt = std::min(remaining_dt, 0.6 * tau_substep_code);
    const double substep_fraction = substep_dt / (3.0 * tau_substep_code);
    if(!isfinite(substep_fraction) || !(substep_fraction > 0.0))
      return;
    a_new = std::min(All.DustMaxGrainSize, a_new * (1.0 + substep_fraction));
    remaining_dt -= substep_dt;
  }

  if(!isfinite(a_new) || !(a_new > a_nm)) return;

  Sp->DustP[dust_idx].GrainRadius = a_new;
  ++NCoagulationEvents;
  ++CoagGrown;
  if(subcycled) ++CoagLargeUpdates;
  record_coagulation_event(n_H, n_eff);


}

// ═══════════════════════════════════════════════════════════════════════════════
// dust_grain_growth_subgrid
// ═══════════════════════════════════════════════════════════════════════════════
/**
 * Grow dust by accreting gas-phase metals following Hirashita & Kuo (2011).
 *
 * Growth is treated separately for the carbonaceous and silicate components
 * of each mixed grain. Each component receives its own HK11 accretion timescale,
 * so CarbonMassFraction evolves continuously rather than choosing one grain species
 * from a hard CF >= 0.5 threshold.
 *
 * For each grain:
 *
 *   M_C   = CF × M_dust
 *   M_sil = (1 - CF) × M_dust
 *
 * The two components grow independently:
 *
 *   dM_C   = M_C   [ exp(3 f_mol dt / tau_C)   - 1 ]
 *   dM_sil = M_sil [ exp(3 f_mol dt / tau_sil) - 1 ]
 *
 * The factor of 3 preserves the previous radius-growth law because M ∝ a^3.
 *
 * All existing total-mass limits (D/Z, available metals, per-step growth)
 * are applied to dM_total. If a cap reduces the requested growth, the allowed
 * mass is divided between C and silicate in the same ratio as their uncapped
 * growth requests. Thus composition and total mass remain mutually consistent.
 *
 * The final composition is:
 *
 *   CF_new = (M_C_old + dM_C) / (M_dust_old + dM_total)
 *
 * Because tau_C and tau_sil differ, accretion itself can now evolve composition.
 *
 */
void dust_grain_growth_subgrid(simparticles *Sp, int dust_idx, int gas_idx, double dt,
                               double T_gas)
{
  if(!All.DustEnableGrowth) return;
  // ── Diagnostics ────────────────────────────────────────────────────────────
  static int total_calls = 0;
  static int failed_hot = 0, failed_no_metals = 0;
  static int failed_low_density = 0, failed_low_fmol = 0;
  static int failed_no_dust = 0, failed_too_far = 0;
  static int failed_max_dz = 0, failed_bad_tau = 0;
  static int passed_all = 0;
  static int fmol_diffuse = 0, fmol_moderate = 0;
  static int fmol_dense = 0, fmol_sf = 0;
  static long long capped_by_dz = 0;
  static long long capped_by_metals = 0;
  static long long capped_by_perstep = 0;
  // Composition-change diagnostics
  static long long cf_increased = 0;
  static long long cf_decreased = 0;
  static long long cf_unchanged = 0;
  total_calls++;
  if(total_calls % 500000 == 0 && All.ThisTask == 0) {
    DUST_PRINT("=== HK11 GROWTH DIAGNOSTICS (after %d attempts) Rank 0 ===\n",
               total_calls);
    DUST_PRINT("  Failed hot:          %6d (%.1f%%)\n",
               failed_hot, 100.0 * failed_hot / total_calls);
    DUST_PRINT("  Failed no metals:    %6d (%.1f%%)\n",
               failed_no_metals, 100.0 * failed_no_metals / total_calls);
    DUST_PRINT("  Failed low density:  %6d (%.1f%%)\n",
               failed_low_density, 100.0 * failed_low_density / total_calls);
    DUST_PRINT("  Failed f_mol low:    %6d (%.1f%%)\n",
               failed_low_fmol, 100.0 * failed_low_fmol / total_calls);
    DUST_PRINT("  Failed no dust:      %6d (%.1f%%)\n",
               failed_no_dust, 100.0 * failed_no_dust / total_calls);
    DUST_PRINT("  Failed too far:      %6d (%.1f%%)\n",
               failed_too_far, 100.0 * failed_too_far / total_calls);
    DUST_PRINT("  Failed max D/Z:      %6d (%.1f%%)\n",
               failed_max_dz, 100.0 * failed_max_dz / total_calls);
    DUST_PRINT("  Failed bad tau:      %6d (%.1f%%)\n",
               failed_bad_tau, 100.0 * failed_bad_tau / total_calls);
    DUST_PRINT("  PASSED:              %6d (%.1f%%)\n",
               passed_all, 100.0 * passed_all / total_calls);
    DUST_PRINT("  Capped by D/Z:       %6lld (%.1f%% of passed)\n",
               capped_by_dz,
               passed_all > 0 ? 100.0 * capped_by_dz / passed_all : 0.0);
    DUST_PRINT("  Capped by metals:    %6lld (%.1f%% of passed)\n",
               capped_by_metals,
               passed_all > 0 ? 100.0 * capped_by_metals / passed_all : 0.0);
    DUST_PRINT("  Capped by per-step:  %6lld (%.1f%% of passed)\n",
               capped_by_perstep,
               passed_all > 0 ? 100.0 * capped_by_perstep / passed_all : 0.0);
    DUST_PRINT("  CF evolution: up=%lld down=%lld unchanged=%lld\n",
               cf_increased, cf_decreased, cf_unchanged);
    DUST_PRINT("  f_mol: diffuse=%d moderate=%d dense=%d sf=%d\n",
               fmol_diffuse, fmol_moderate, fmol_dense, fmol_sf);
    DUST_PRINT("  Total mass grown: %.3e Msun  Growth events: %lld\n",
               TotalMassGrown, NGrainGrowthEvents);
    DUST_PRINT("===========================================\n");
  }
  // ── Gate 1: hot gas inhibits accretion ─────────────────────────────────────
  if(T_gas > All.DustThermalSputteringTemp) {
    failed_hot++;
    return;
  }
  // ── Gate 2: gas must contain metals ─────────────────────────────────────────
  const double Z_gas = Sp->SphP[gas_idx].Metallicity;
  if(Z_gas < 1e-4) {
    failed_no_metals++;
    return;
  }
  // ── Gate 3: require sufficiently dense gas ──────────────────────────────────
  double n_H = dust_gas_nH_cgs(Sp, gas_idx);
  double DustClumpingFactor = dust_clumping_factor(n_H, Sp->SphP[gas_idx].Sfr > DUST_SFR_EPS);
  double n_eff_cm3 = n_H * DustClumpingFactor;
  if(n_eff_cm3 < 0.1) {
    failed_low_density++;
    return;
  }
  // ── Redshift-dependent maximum dust-to-metal ratio ──────────────────────────
  double z_now = 1.0 / All.Time - 1.0;
  double max_dust_to_metal = std::max(0.05, 0.5 / (1.0 + 0.15 * z_now));
  double M_gas_quick  = Sp->P[gas_idx].getMass();
  double M_dust_quick = Sp->P[dust_idx].getMass();
  // Use the same redshift-dependent D/Z limit as the later exact cap.
  if(M_dust_quick >=
     M_gas_quick * Z_gas * max_dust_to_metal) {
    failed_max_dz++;
    return;
  }
  // ── Molecular fraction ──────────────────────────────────────────────────────
  // Dense / star-forming gas receives stronger accretion.
  double f_mol = 0.05;
  #ifdef STARFORMATION
    if(Sp->SphP[gas_idx].Sfr > DUST_SFR_EPS) {
      f_mol = 0.8;
      fmol_sf++;
    }
    else if(n_eff_cm3 > 100.0) {
      f_mol = 0.5;
      fmol_dense++;
    }
    else if(n_eff_cm3 > 10.0) {
      f_mol = 0.2;
      fmol_moderate++;
    }
    else {
      fmol_diffuse++;
    }
  #else
    if(n_H > 100.0) {
      f_mol = 0.5;
      fmol_dense++;
    }
    else if(n_H > 10.0) {
      f_mol = 0.2;
      fmol_moderate++;
    }
    else {
      fmol_diffuse++;
    }
  #endif
  // Metal-rich gas self-shields more efficiently.
  if(Z_gas > 0.01)
    f_mol = std::min(1.0, 1.5 * f_mol);
  if(f_mol < 0.01) {
    failed_low_fmol++;
    return;
  }
  // ── Dust particle validity ──────────────────────────────────────────────────
  const int nearest_dust = dust_idx;
  if(nearest_dust < 0) {
    failed_no_dust++;
    return;
  }
  double dxyz[3];
  Sp->nearest_image_intpos_to_pos(Sp->P[gas_idx].IntPos,
                                  Sp->P[nearest_dust].IntPos,
                                  dxyz);
  const double dist_kpc =
      sqrt(dxyz[0] * dxyz[0] + dxyz[1] * dxyz[1] + dxyz[2] * dxyz[2]);

  // Safety check consistent with update_dust_dynamics(): growth is allowed
  // within the resolved SPH neighborhood of the associated gas particle,
  // with a 2-kpc floor and a 10-kpc ceiling.  The old fixed 5-kpc guard
  // could otherwise silently reject grains that passed the new Hsml-aware
  // main-loop association criterion.
  const double growth_association_radius =
      std::max(2.0,
               std::min((double)Sp->SphP[gas_idx].Hsml, 10.0));

  if(dist_kpc > growth_association_radius) {
    failed_too_far++;
    return;
  }
  const double a      = Sp->DustP[nearest_dust].GrainRadius;
  const double M_dust = Sp->P[nearest_dust].getMass();
  // ── Corruption guard ────────────────────────────────────────────────────────
  if(a <= 0.0 || !isfinite(a) ||
     M_dust <= 0.0 || !isfinite(M_dust)) {
    bool is_domain_exchange = (a == 0.0 && Sp->DustP[nearest_dust].CarbonMassFraction == 0.0 && Sp->DustP[nearest_dust].DustTemperature == 0.0);
    NDustDestroyedByCorruption++;
    static int corruption_count = 0;
    corruption_count++;
    printf("[GROWTH_CORRUPTION|T=%d|Step=%d] #%d: "
           "idx=%d a=%.3e M=%g type=%d ID=%lld "
           "subtype=%s | RunningTotal=%lld\n",
           All.ThisTask, All.NumCurrentTiStep,
           corruption_count,
           nearest_dust, a, M_dust,
           Sp->P[nearest_dust].getType(),
           (long long)Sp->P[nearest_dust].ID.get(),
           is_domain_exchange
               ? "DOMAIN_EXCHANGE_DustP_not_sent"
               : "NUMERICAL_NaN_or_negative",
           NDustDestroyedByCorruption);
    Sp->P[nearest_dust].setMass(DUST_MASS_TO_DESTROY);
    Sp->P[nearest_dust].ID.set(0);
    Sp->P[nearest_dust].setType(3);
    memset(&Sp->DustP[nearest_dust], 0, sizeof(dust_data));
    Sp->DustP[nearest_dust].GrainRadius = All.DustMinGrainSize;
    return;
  }
  // ════════════════════════════════════════════════════════════════════════════
  // Species-resolved composition growth
  // ════════════════════════════════════════════════════════════════════════════
  // Current grain composition.
  const double CF_old = std::max(0.0, std::min(1.0, (double)Sp->DustP[nearest_dust].CarbonMassFraction));
  // Split the existing superparticle mass into its two material components.
  const double M_carb_old = M_dust * CF_old;
  const double M_sil_old  = M_dust * (1.0 - CF_old);
  const double T_eff_K       = 20.0;
  const double Zsun_massfrac = 0.02;
  const double S_stick       = 0.3;
  double n_eff_for_growth = std::max(n_eff_cm3, 1e-3);
  double a_cm             = a * 1e-7;  // nm -> cm
  // ── Separate HK11 timescales for carbonaceous and silicate material ─────────
  double tau_carb_yr =
      tau_acc_yr_HK11(n_eff_for_growth,
                      T_eff_K,
                      Z_gas,
                      Zsun_massfrac,
                      a_cm,
                      S_stick,
                      1);  // carbonaceous
  double tau_sil_yr =
      tau_acc_yr_HK11(n_eff_for_growth,
                      T_eff_K,
                      Z_gas,
                      Zsun_massfrac,
                      a_cm,
                      S_stick,
                      0);  // silicate
  tau_carb_yr *= All.DustGrowthCalibration;
  tau_sil_yr  *= All.DustGrowthCalibration;
  if(!isfinite(tau_carb_yr) || tau_carb_yr <= 0.0 ||
     !isfinite(tau_sil_yr)  || tau_sil_yr  <= 0.0) {
    failed_bad_tau++;
    return;
  }
  // Same numerical limits formerly applied to the single growth timescale.
  tau_carb_yr = std::max(1e6, std::min(5e9, tau_carb_yr));
  tau_sil_yr  = std::max(1e6, std::min(5e9, tau_sil_yr));
  const double tau_carb = tau_carb_yr * SEC_PER_YEAR / dust_code_time_in_s();
  const double tau_sil = tau_sil_yr * SEC_PER_YEAR / dust_code_time_in_s();
  // ── Uncapped species growth requests ────────────────────────────────────────
  //
  // Previous model:
  //   a_new = a exp(f_mol dt / tau)
  //
  // Since M ∝ a^3:
  //   M_new / M_old = exp(3 f_mol dt / tau)
  //
  // Apply that same law separately to C and silicate.
  double dm_carb_raw = M_carb_old * (exp(3.0 * f_mol * dt / tau_carb) - 1.0);
  double dm_sil_raw = M_sil_old * (exp(3.0 * f_mol * dt / tau_sil) - 1.0);
  if(!isfinite(dm_carb_raw) || dm_carb_raw < 0.0 ||
     !isfinite(dm_sil_raw)  || dm_sil_raw  < 0.0) {
    failed_bad_tau++;
    return;
  }
  const double dm_raw = dm_carb_raw + dm_sil_raw;
  if(!isfinite(dm_raw) || dm_raw <= 0.0)
    return;
  // Start with the complete requested growth; caps below may reduce it.
  double dm = dm_raw;
  // ── Available gas metal reservoir ───────────────────────────────────────────
  const double M_gas       = Sp->P[gas_idx].getMass();
  const double M_metals    = M_gas * Z_gas;
  const double M_dust_max  = M_metals * max_dust_to_metal;
  if(M_dust >= M_dust_max) {
    failed_max_dz++;
    return;
  }
  // ── Cap 1: maximum permitted D/Z ────────────────────────────────────────────
  if(M_dust + dm > M_dust_max) {
    dm = M_dust_max - M_dust;
    if(dm <= 0.0)
      return;
    capped_by_dz++;
  }
  // ── Cap 2: cannot accrete more metal mass than exists ───────────────────────
  if(dm > M_metals) {
    dm = 0.99 * M_metals;
    capped_by_metals++;
  }
  // ── Cap 3: limit one-call growth to 20% of current dust mass ────────────────
  const double max_dm_per_step = 0.2 * M_dust;
  if(dm > max_dm_per_step) {
    dm = max_dm_per_step;
    capped_by_perstep++;
  }
  if(!isfinite(dm) || dm <= 0.0)
    return;
  // ════════════════════════════════════════════════════════════════════════════
  // Preserve species ratio after total growth caps
  // ════════════════════════════════════════════════════════════════════════════
  // Fraction of the uncapped growth request contributed by carbon.
  // Applying this fraction to the final allowed dm preserves the relative
  // C/silicate HK11 growth rates even when a total-mass cap is active.
  double carb_growth_fraction = dm_carb_raw / dm_raw;
  carb_growth_fraction = std::max(0.0, std::min(1.0, carb_growth_fraction));
  double dm_carb = dm * carb_growth_fraction;
  // Define silicate as the remainder so dm_carb + dm_sil == dm exactly.
  double dm_sil = dm - dm_carb;
    // ════════════════════════════════════════════════════════════════════════════
  // Apply species-resolved gas -> dust transfer
  // ════════════════════════════════════════════════════════════════════════════
  // First make sure the REQUESTED growth would not produce an invalid grain.
  // The helper can only reduce this growth, never increase it.
  double M_dust_requested = M_dust + dm;
  double a_requested = a * cbrt(M_dust_requested / M_dust);
  if(!isfinite(a_requested) ||
     a_requested < All.DustMinGrainSize ||
     a_requested > All.DustMaxGrainSize)
      return;
  // Remove the requested C and silicate material from the actual
  // element-resolved gas reservoirs. The helper may reduce either component
  // if the required gas-phase elements are unavailable.
  double actual_carb = 0.0;
  double actual_sil  = 0.0;
  double dm_actual =
      remove_dust_material_from_gas(
          Sp,
          gas_idx,
          dm_carb,
          dm_sil,
          actual_carb,
          actual_sil);
  if(dm_actual <= 0.0)
      return;
  // ════════════════════════════════════════════════════════════════════════════
  // Reconstruct the grain from the ACTUALLY accreted material
  // ════════════════════════════════════════════════════════════════════════════
  double M_carb_new = M_carb_old + actual_carb;
  double M_sil_new = M_sil_old + actual_sil;
  double M_dust_new = M_carb_new + M_sil_new;
  if(!isfinite(M_dust_new) ||
     M_dust_new <= M_dust)
      return;
  double CF_new = M_carb_new / M_dust_new;
  CF_new = std::max(0.0, std::min(1.0, CF_new));
  // Grain radius follows the ACTUAL new dust mass.
  double a_new = a * cbrt(M_dust_new / M_dust);
  if(!isfinite(a_new) ||
     a_new < All.DustMinGrainSize ||
     a_new > All.DustMaxGrainSize)
      Terminate("Invalid grain radius after element-resolved dust growth");

    double da = a_new - a;

    if(!isfinite(da))
        Terminate("Non-finite grain growth after gas material removal");

    // due to the possibilty of a tiny growth rounding to zero, we allow a_new == a, but not a_new < a
    if(da < 0.0)
        Terminate("Negative grain growth after gas material removal");

    if(da == 0.0)
        a_new = a;

    static long long tiny_radius_growth = 0;

    if(da == 0.0)
    {
        tiny_radius_growth++;

        if(tiny_radius_growth <= 10)
            DUST_PRINT(
                "[GROWTH_TINY] dm_actual=%.17e "
                "Mold=%.17e Mnew=%.17e "
                "rel=%.17e a=%.17e anew=%.17e\n",
                dm_actual,
                M_dust,
                M_dust_new,
                (M_dust_new - M_dust) / M_dust,
                a,
                a_new);
    }

  // ════════════════════════════════════════════════════════════════════════════
  // Update the dust particle
  // ════════════════════════════════════════════════════════════════════════════
  Sp->P[nearest_dust].setMass(
      M_dust_new);
  Sp->DustP[nearest_dust].GrainRadius = a_new;
  Sp->DustP[nearest_dust].CarbonMassFraction = (float)CF_new;
  // The event has now genuinely succeeded.
  passed_all++;
  NGrainGrowthEvents++;
  TotalMassGrown += dm_actual;
  // Track which way composition moved.
  const double dCF = CF_new - CF_old;
  if(dCF > 1e-10)
      cf_increased++;
  else if(dCF < -1e-10)
      cf_decreased++;
  else
      cf_unchanged++;
  // ── Limited timestep diagnostic ───────────────────────────────────────────
  static int dt_printed = 0;
  if(dt_printed < 10 &&
     All.ThisTask == 0)
  {
      double dt_myr = dt * dust_code_time_in_s() / (1e6 * SEC_PER_YEAR);
      DUST_PRINT(
          "[GROWTH_DEBUG] dt=%.3e code = %.3f Myr "
          "da=%.3f nm dm_req=%.3e dm_actual=%.3e "
          "dM_C_req=%.3e dM_sil_req=%.3e "
          "dM_C_actual=%.3e dM_sil_actual=%.3e "
          "CF=%.6f->%.6f\n",
          dt,
          dt_myr,
          da,
          dm,
          dm_actual,
          dm_carb,
          dm_sil,
          actual_carb,
          actual_sil,
          CF_old,
          CF_new);
      dt_printed++;
  }
  // ── Periodic composition-aware growth diagnostic ─────────────────────────
  static int growth_count = 0;
  growth_count++;
  if(growth_count % 10000 == 0 &&
     All.ThisTask == 0)
  {
      DUST_PRINT(
          "[HK11_GROWTH] event=%d "
          "CF=%.6f->%.6f dCF=%+.3e "
          "f_mol=%.3f "
          "tau_C_yr=%.3e tau_sil_yr=%.3e "
          "dM_C_req=%.3e dM_sil_req=%.3e "
          "dM_C_actual=%.3e dM_sil_actual=%.3e "
          "n_H_cm3=%.1f n_eff_cm3=%.1f clumping_factor=%.0f "
          "T_eff_K=%.0f "
          "Z_gas_before=%.4f Z_gas_after=%.4f "
          "a_nm_old=%.2f a_nm_new=%.2f "
          "dm_actual=%.3e "
          "M_dust_before=%.3e M_dust_after=%.3e\n",
          growth_count,
          CF_old,
          CF_new,
          dCF,
          f_mol,
          tau_carb_yr,
          tau_sil_yr,
          dm_carb,
          dm_sil,
          actual_carb,
          actual_sil,
          n_H,
          n_eff_cm3,
          DustClumpingFactor,
          T_eff_K,
          Z_gas,
          Sp->SphP[gas_idx].Metallicity,
          a,
          a_new,
          dm_actual,
          M_dust,
          M_dust_new);
  }
}
#endif /* DUST */