/*! \file dust.cc
 *  \brief The CosmicGrain dust model, evolved on the fly during the simulation.
 *
 *  Dust is carried by "superparticles": each one stands for a large group of grains
 *  that share a position, velocity and grain size. Stars make them (supernovae and AGB
 *  winds), and this file then evolves them through gas drag, grain growth, thermal
 *  sputtering (erosion in hot gas), shock destruction, coagulation (small grains
 *  sticking together), shattering (grains breaking apart), radiation pressure, and
 *  dust temperature updates.
 *
 *  For how dust particles get their timebins and how often they are updated, see
 *  spawn_dust_particle(), update_dust_dynamics() and get_timestep_grav() in timestep.cc.
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
#include "../cooling_sfr/cg_distributed.h"

// ── Mass / size thresholds ────────────────────────────────────────────────────
// Smallest dust mass (code units) worth turning into a new dust particle. Low enough
// that small yields still make particles, high enough that trace amounts of metal
// ejecta do not flood the run with tiny particles.
#define MIN_DUST_PARTICLE_MASS  1e-10

// A dust particle whose mass (code units) drops below this is treated as empty and
// removed. Keep it many orders of magnitude below a typical dust particle mass so
// that only truly negligible particles are removed.
#define DUST_MASS_TO_DESTROY    1e-30

// ── Miscellaneous physics constants ──────────────────────────────────────────
// A gas cell counts as star-forming (for the clumping factor and f_mol, the molecular
// fraction) only if its SFR (star formation rate) is above this.
#define DUST_SFR_EPS            1e-14

// Not used in the code; for reference only. Mass of one silicate grain 1 nm across
// (radius 0.5 nm, density 2.4 g/cm³):
// m = (4/3)π(0.5e-7 cm)³ × 2.4 g/cm³ / 1.989e33 g/M☉
#define DUST_SINGLE_GRAIN_MASS_MSUN  6.32e-55   // M☉

// ── Diagnostic print macro ────────────────────────────────────────────────────
// Prints only when All.DustDebugLevel is set, so production runs stay quiet. Each
// line starts with the task number, scale factor a, and redshift z.
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
static inline bool dust_dense_phase_model(void);
static double dust_dense_fraction(simparticles *Sp, int gas_idx, double *n_dense_cm3);

static double sample_initial_grain_radius(double median_nm, double sigma_dex)
{
  if(!(sigma_dex > 0.0))
    return std::max(All.DustMinGrainSize,
                    std::min(All.DustMaxGrainSize, median_nm));

  // Draw a log-normal grain radius in nm: log10(radius) follows a normal (bell-curve)
  // distribution with the given median and width sigma_dex (in dex, i.e. powers of
  // ten). The Box-Muller formula turns two uniform random numbers into one normal one.
  // The result is clamped to [DustMinGrainSize, DustMaxGrainSize].
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

/** Seconds per code time unit.
 *
 * Lengths are in h^-1 kpc and velocities in km/s, so one code time unit is
 * UnitTime_in_s / h seconds (cooling.cc converts the same way). Every conversion
 * between physical and code time in the dust module goes through this helper.
 * We used to use UnitTime_in_s alone, which made every dust timescale 1/h
 * (about 1.5x) too long.
 */
static inline double dust_code_time_in_s(void)
{
  return All.UnitTime_in_s / All.HubbleParam;
}

/** Physical (proper) gas mass density in g/cm^3.
 *
 * The stored SPH density is comoving and in code units. The a^-3 and h^2 factors
 * are the same ones cooling.cc uses, so dust drag, sputtering, collisions, growth
 * and dust temperature all see the same physical density as cooling and star
 * formation.
 */
static inline double dust_gas_density_cgs(const simparticles *Sp, int gas_idx)
{
  return Sp->SphP[gas_idx].Density * All.cf_a3inv
         * All.UnitDensity_in_cgs * All.HubbleParam * All.HubbleParam;
}

/** Physical number density of hydrogen nuclei, n_H, in cm^-3. */
static inline double dust_gas_nH_cgs(const simparticles *Sp, int gas_idx)
{
  return HYDROGEN_MASSFRAC * dust_gas_density_cgs(Sp, gas_idx) / PROTONMASS;
}

// ── Shared spatial hashes (defined in feedback.cc) ────────────────────────────
// gas_hash, star_hash and dust_hash bin this task's particles into cells so that
// neighbour searches only look at nearby cells.

// ── Local D/G: dust-to-gas mass ratio around each dust particle ──────────────
// For every dust particle we add up the gas mass and the dust mass inside a sphere
// around it, on all tasks, and take the ratio. A query is the sphere (owner's dust
// index, center, radius); a result is the gas and dust mass one task found inside it.
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

// Side of the fine grid cells (code units) used only for the D/G sphere sums.
//
// The shared feedback hash (gas_hash, dust_hash) has 12-31 kpc cells, but D/G spheres
// are 2-10 code units, so each query used to scan every particle in cells hundreds of
// times larger than the sphere. This fine grid is rebuilt from the hash contents at
// every dust call and stores each cell's total mass. Cells fully inside the sphere add
// their total at once, cells fully outside are skipped, and only cells cut by the
// sphere surface are checked particle by particle. It uses the same particles and the
// same inside/outside test as spatial_hash_zoom::sum_mass_in_sphere(), so the sums
// agree except for rounding from adding in a different order.
static constexpr double DTG_GRID_CELL = 2.0;

// Side of the cells used to route D/G queries between tasks, in fine-grid
// cells (DTG_GRID_CELL code units each).
static constexpr int DTG_ROUTE_RATIO = 5;

// Use occupancy routing by default from this many ranks (MPI tasks) up. On 24 ranks
// bounding-box routing was faster (0.08 vs 0.13 s per call); see the caller.
static constexpr int DTG_OCCUPANCY_MIN_TASKS = 64;

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
    // Round the cell size so a whole number of cells fits the periodic box; then
    // wrapping indices around the box edge always lands on the right cell.
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
    // Loop over cell indices around the center without wrapping; key() wraps them.
    // Because radius << BoxSize/2, the unwrapped cell position is the cell's nearest
    // periodic copy, so the distance tests below are correct.
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
            // Whole cell inside the sphere: add its total mass. The small margin sends
            // cells whose far corner is within rounding of the surface to the exact
            // per-particle check instead.
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

// Fill local_DTG[i] with the D/G ratio inside a sphere of radius radii[i] around local
// dust particle i (radius <= 0 means skip; result 0). Each task sends its spheres to
// the tasks that may hold gas or dust inside them, those tasks sum their own particles,
// and the partial sums come back and are added up. The ratio is clamped to
// [1e-6, 0.1]; it is 0 when either sum is zero.
// Collective (MPI_Alltoall/Alltoallv): all tasks must call this, even with no spheres.
// occupancy_routing picks how to choose target tasks (both give identical results).
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
      // Occupancy routing. Each task announces which coarse routing cells hold the
      // particles it sums (exactly its fine-grid members), and a query goes only to
      // tasks that hold a cell the sphere touches. The tasks skipped this way would
      // have added exactly zero, and each task still gets its queries in the same
      // order, so the results are bit-identical to bounding-box routing.
      // Why: in zoom runs each task's domain is scattered, so its bounding box is
      // large and bounding-box routing sends most queries to most tasks. Its cost
      // then grows with the number of tasks instead of being shared among them.
      // A routing cell is DTG_ROUTE_RATIO fine cells on a side, so occupancy comes
      // straight from the fine-grid keys without touching particles. The environment
      // variable CG_DTG_ROUTE_RATIO overrides the ratio for scaling tests (smaller
      // cells send fewer queries but announce more cells).
      static int route_ratio = 0;
      if(route_ratio == 0)
        {
          const char *env = getenv("CG_DTG_ROUTE_RATIO");
          route_ratio = (env && atoi(env) > 0) ? atoi(env) : DTG_ROUTE_RATIO;
          if(All.ThisTask == 0)
            printf("[DTG_ROUTE] routing cells are %d fine cells (%g code units) on a side\n",
                   route_ratio, route_ratio * gas_grid.cell);
        }
      const long long n_f = gas_grid.n_per_dim;
      const bool aligned  = n_f > 0 && n_f % route_ratio == 0;
      const long long n_c = aligned ? n_f / route_ratio
                                    : std::max(1LL, (long long)std::floor(All.BoxSize / (route_ratio * DTG_GRID_CELL)));
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
          {
            if(aligned)
              {
                const long long k = fine_cell.first;
                local_cells.insert(ckey((k % n_f) / route_ratio, ((k / n_f) % n_f) / route_ratio,
                                        (k / (n_f * n_f)) / route_ratio));
                continue;
              }
            // Box not a whole number of routing cells: bin particle positions.
            for(int idx : fine_cell.second.particles)
              {
                double pos[3];
                Sp->intpos_to_pos(Sp->P[idx].IntPos, pos);
                local_cells.insert(ckey((long long)std::floor(pos[0] / cc),
                                        (long long)std::floor(pos[1] / cc),
                                        (long long)std::floor(pos[2] / cc)));
              }
          }

      // Register the occupied cells in a distributed directory and look up only the
      // cells this task's spheres touch. (Gathering every task's cells on every task
      // cost about 0.5 s per call no matter how many tasks there were.)
      cg_dist::CellDirectory directory;
      directory.build(std::vector<long long>(local_cells.begin(), local_cells.end()), Communicator);

      // Keys of all routing cells that the sphere (center, radius) overlaps.
      auto touched_cells = [&](const MyIntPosType *center_intpos, double radius, std::vector<long long> &keys) {
        keys.clear();
        MyIntPosType center_copy[3] = {center_intpos[0], center_intpos[1], center_intpos[2]};
        double c[3];
        Sp->intpos_to_pos(center_copy, c);
        const double r2 = radius * radius;
        long long lo[3], hi[3];
        for(int d = 0; d < 3; d++)
          {
            lo[d] = (long long)std::floor((c[d] - radius) / cc);
            hi[d] = (long long)std::floor((c[d] + radius) / cc);
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
                if(near2 <= r2)
                  keys.push_back(ckey(ix, iy, iz));
              }
      };

      std::unordered_set<long long> wanted;
      std::vector<long long> keys;
      for(size_t dust_idx = 0; dust_idx < radii.size(); dust_idx++)
        if(radii[dust_idx] > 0.0)
          {
            touched_cells(Sp->P[dust_idx].IntPos, radii[dust_idx], keys);
            wanted.insert(keys.begin(), keys.end());
          }
      const std::unordered_map<long long, std::vector<int>> owners =
          directory.lookup(std::vector<long long>(wanted.begin(), wanted.end()));

      // last_query stops a sphere that touches several of one task's cells from
      // being sent to that task more than once.
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
          touched_cells(query.center, query.radius, keys);
          for(long long k : keys)
            {
              const auto it = owners.find(k);
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

      // The loop above runs in increasing dust index, as bounding-box routing does,
      // so each task gets queries (and adds results) in the same order.
    }
  else
    {
      // Bounding-box routing: every task shares the box around its gas and dust, and a
      // sphere goes to every task whose box it touches. Default below
      // DTG_OCCUPANCY_MIN_TASKS ranks, and used by the caller's routing self-check.
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

  // Exchange the queries. MPI counts are int, so check for overflow first. When a
  // buffer is empty we pass a dummy address, because .data() of an empty vector may
  // be null.
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

  // Answer the queries we received with sums over our own particles.
  // Self-check: the first 5000 queries on each task are also summed with the shared
  // hash. Only the adding order differs, so a relative difference above 1e-10 is a bug.
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
                  // Brute force over every particle in the shared hash, to tell
                  // which of the two methods is wrong.
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

  // Send the partial sums back to the tasks that asked (counts are the reverse of
  // the query exchange), then add them up per dust particle.
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
// Per-task counters that add up over the whole run. print_dust_statistics() sums
// them over all tasks.
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

// Removals for bookkeeping reasons, not physics. These should all stay zero in a clean
// run; anything else points to a domain-exchange bug or a numerical problem.
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
// Moving elements between gas and dust
//
// Each dust particle has a carbon fraction CF, so it holds:
//   carbonaceous mass = CF * M_dust
//   silicate mass     = (1-CF) * M_dust
//
// Silicate mass is split into O, Mg, Si and Fe with the fixed mass fractions below
// (from the adopted solar abundance pattern; they sum to 1). N and Ne never go
// into dust.
// ============================================================================
static constexpr double SIL_FRAC_O  = 0.69876708;
static constexpr double SIL_FRAC_MG = 0.07680773;
static constexpr double SIL_FRAC_SI = 0.08097301;
static constexpr double SIL_FRAC_FE = 0.14345218;

// The gas metallicity is stored three times (MassMetallicity, SphP.Metallicity and
// P.Metallicity). Recompute the last two from the metal mass after any change.
static inline void sync_gas_metallicity(simparticles *Sp, int gas_idx)
{
    double gas_mass = Sp->P[gas_idx].getMass();
    if(gas_mass > 0.0)
    {
        Sp->SphP[gas_idx].Metallicity = Sp->SphP[gas_idx].MassMetallicity / gas_mass;
        Sp->P[gas_idx].Metallicity = Sp->SphP[gas_idx].Metallicity;
    }
}


// Give destroyed dust back to a gas cell. dM_carbon and dM_silicate are in code mass
// units and must be >= 0. The gas mass, its metal mass and its C, O, Mg, Si and Fe
// masses all go up; N and Ne masses stay the same (their fractions drop slightly
// because the gas got heavier).
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

    // Work in element masses, not fractions, because the gas mass changes.
    double M_C = old_gas_mass * Sp->SphP[gas_idx].GasCarbonMassFraction;
    double M_N = old_gas_mass * Sp->SphP[gas_idx].GasNitrogenMassFraction;
    double M_O = old_gas_mass * Sp->SphP[gas_idx].GasOxygenMassFraction;
    double M_Ne = old_gas_mass * Sp->SphP[gas_idx].GasNeonMassFraction;
    double M_Mg = old_gas_mass * Sp->SphP[gas_idx].GasMagnesiumMassFraction;
    double M_Si = old_gas_mass * Sp->SphP[gas_idx].GasSiliconMassFraction;
    double M_Fe = old_gas_mass * Sp->SphP[gas_idx].GasIronMassFraction;

    M_C  += dM_carbon;
    M_O  += dM_silicate * SIL_FRAC_O;
    M_Mg += dM_silicate * SIL_FRAC_MG;
    M_Si += dM_silicate * SIL_FRAC_SI;
    M_Fe += dM_silicate * SIL_FRAC_FE;
    double new_gas_mass = old_gas_mass + dM_total;
    Sp->P[gas_idx].setMass(new_gas_mass);
    // Dust is made only of metals, so all of it adds to the metal mass.
    Sp->SphP[gas_idx].MassMetallicity += dM_total;
    // Back to fractions, using the new gas mass.
    Sp->SphP[gas_idx].GasCarbonMassFraction = M_C / new_gas_mass;
    Sp->SphP[gas_idx].GasNitrogenMassFraction = M_N / new_gas_mass;
    Sp->SphP[gas_idx].GasOxygenMassFraction = M_O / new_gas_mass;
    Sp->SphP[gas_idx].GasNeonMassFraction = M_Ne / new_gas_mass;
    Sp->SphP[gas_idx].GasMagnesiumMassFraction = M_Mg / new_gas_mass;
    Sp->SphP[gas_idx].GasSiliconMassFraction = M_Si / new_gas_mass;
    Sp->SphP[gas_idx].GasIronMassFraction = M_Fe / new_gas_mass;
    sync_gas_metallicity(Sp, gas_idx);
}

// Take carbon and silicate material out of a gas cell to build dust (the reverse of
// return_dust_material_to_gas). Requests are in code mass units. The gas can only give
// what it has, so the amounts may be smaller than asked. Returns the total mass
// actually removed; the carbon and silicate parts come back in removed_carbon and
// removed_silicate.
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
    // Element masses currently in the gas.
    double M_C = old_gas_mass * Sp->SphP[gas_idx].GasCarbonMassFraction;
    double M_N = old_gas_mass * Sp->SphP[gas_idx].GasNitrogenMassFraction;
    double M_O = old_gas_mass * Sp->SphP[gas_idx].GasOxygenMassFraction;
    double M_Ne = old_gas_mass * Sp->SphP[gas_idx].GasNeonMassFraction;
    double M_Mg = old_gas_mass * Sp->SphP[gas_idx].GasMagnesiumMassFraction;
    double M_Si = old_gas_mass * Sp->SphP[gas_idx].GasSiliconMassFraction;
    double M_Fe = old_gas_mass * Sp->SphP[gas_idx].GasIronMassFraction;
    // Carbon: limited by the carbon in the gas.
    removed_carbon = std::min(requested_carbon, M_C);
    // Silicate needs O, Mg, Si and Fe in fixed proportions, so the element that runs
    // out first sets the most silicate we can make.
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

    // Remove the element masses actually used.
    M_C -= removed_carbon;
    M_O  -= removed_silicate * SIL_FRAC_O;
    M_Mg -= removed_silicate * SIL_FRAC_MG;
    M_Si -= removed_silicate * SIL_FRAC_SI;
    M_Fe -= removed_silicate * SIL_FRAC_FE;
    // Guard against tiny negative values from rounding.
    M_C  = std::max(0.0, M_C);
    M_O  = std::max(0.0, M_O);
    M_Mg = std::max(0.0, M_Mg);
    M_Si = std::max(0.0, M_Si);
    M_Fe = std::max(0.0, M_Fe);
    double new_gas_mass = old_gas_mass - dM_total;
    if(new_gas_mass <= 0.0)
        Terminate("remove_dust_material_from_gas removed all gas mass");
    Sp->P[gas_idx].setMass(new_gas_mass);

    // Dust is pure metal, so the gas metal mass drops by the full amount. A tiny
    // negative result (> -1e-12) is rounding and is set to 0; anything larger stops the run.
    Sp->SphP[gas_idx].MassMetallicity -= dM_total;
    if(Sp->SphP[gas_idx].MassMetallicity < 0.0)
    {
        if(Sp->SphP[gas_idx].MassMetallicity > -1e-12)
            Sp->SphP[gas_idx].MassMetallicity = 0.0;
        else
            Terminate("Gas metal mass became negative during dust growth");
    }
    // Back to fractions, using the new gas mass.
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
 * Find dust particles with impossible values and remove them.
 *
 * We have seen two kinds of bad particle:
 *  1. DOMAIN-EXCHANGE VICTIM: GrainRadius, CarbonMassFraction and DustTemperature
 *     are all exactly 0.0. Gadget cleared the DustP[] slot (memset) during a
 *     domain exchange, but the P[] slot survived, so position and velocity
 *     still look fine.
 *  2. NUMERICAL CORRUPTION: NaN, Inf or negative values that leaked in from
 *     growth, drag or sputtering. The velocity is usually bad too.
 *
 * A bad particle is turned into type 3 with mass DUST_MASS_TO_DESTROY (1e-30)
 * and ID 0, so destroy_dust_particles() removes it on its next call.
 *
 * Any task that finds bad particles prints a tally by reason, plus a detailed
 * line for each of the first 50 particles per call.
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
  int detail_count   = 0;  // detailed lines printed this call (capped at 50)

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
      // Sort the two kinds apart for the log.
      bool is_domain_exchange_victim = (a == 0.0 && cf == 0.0 && temp == 0.0);
      if(is_domain_exchange_victim && mass > DUST_MASS_TO_DESTROY)
        n_zero_radius_nonzero_mass++;  // DustP lost, P[] intact: exchange timing bug
      else if(is_domain_exchange_victim && mass < DUST_MASS_TO_DESTROY)
        n_fully_zeroed++;              // both arrays zeroed: much worse
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

      // Make the particle harmless so destroy_dust_particles() removes it.
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
 * Gas temperature in K from the stored entropy. Free electrons lower the mean
 * mass per particle, so we use the electron abundance ne (SphP[].Ne, electrons
 * per hydrogen atom).
 *
 * Mean molecular weight: μ = (1 + 4Y) / (1 + Y + ne),
 * where Y = (1 − XH) / (4 XH) is the number of helium atoms per hydrogen atom.
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
 * Pick a new grain radius for a shattered dust particle (radii in nm).
 *
 * The fragments follow a power-law number distribution
 *
 *     dN/da ∝ a^(-q)
 *
 * and one grain's mass goes as a^3, so the fragment MASS per unit radius is
 *
 *     dM/da ∝ a^(3-q).
 *
 * A PartType6 particle stands for a fixed mass of dust, not a fixed number of
 * grains, so we draw the new radius from dM/da, not dN/da. The result lies in
 *
 *     DustMinGrainSize <= a_new <= max_fraction * a_parent.
 *
 * Only GrainRadius changes: the particle keeps its mass and CarbonMassFraction,
 * and no new particle is made. Each event draws a fresh radius, so the
 * population spreads over the whole fragment range.
 *
 * Inverse-CDF sampling with a uniform random number u in [0, 1]:
 *   q != 4:  a = [ a_min^(4-q) + u (a_max^(4-q) - a_min^(4-q)) ]^(1/(4-q))
 *   q  = 4:  dM/da ∝ 1/a, so  a = a_min * (a_max/a_min)^u.
 *
 * Any bad input returns min_radius_nm.
 */
static double sample_shattering_fragment_radius(double parent_radius_nm,
                                                double q,
                                                double max_fraction,
                                                double min_radius_nm)
{
    // -------------------------------------------------------------------------
    // Input checks
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

    // The largest fragment must be smaller than the parent.
    if(max_fraction <= 0.0 || max_fraction >= 1.0)
      return min_radius_nm;

    double a_min = min_radius_nm;
    double a_max = max_fraction * parent_radius_nm;

    // Parent too small to leave any room for fragments: return the floor size.
    if(a_max <= a_min)
        return a_min;

    double u = get_random_number();

    // Keep u inside [0, 1] in case roundoff ever pushes it out.
    u = std::max(0.0, std::min(1.0, u));

    double a_new;

    // -------------------------------------------------------------------------
    // Special case q = 4: dM/da ∝ a^-1 (log-uniform)
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
    // Final guards against roundoff
    // -------------------------------------------------------------------------

    if(!isfinite(a_new))
        a_new = a_min;

    a_new = std::max(a_min, a_new);

    // a_max should already ensure this; enforce it anyway so shattering
    // can never make a grain bigger.
    a_new = std::min(a_new, max_fraction * parent_radius_nm);

    return a_new;
}


// ═══════════════════════════════════════════════════════════════════════════════
// dust_grain_shattering
// ═══════════════════════════════════════════════════════════════════════════════
/**
 * Shatter a dust grain at random in diffuse, turbulent gas. Shattering is
 * when grains smash into each other fast enough to break into smaller pieces.
 *
 * WHEN A GRAIN CAN SHATTER
 * ------------------------
 *   1. Its turbulent collision speed v_grain(a) must reach the
 *      composition-dependent shattering threshold of Jones et al.
 *      (about 1 km/s for silicate, 2 km/s for carbon).
 *   2. The gas must be diffuse. Legacy mode: n_eff < DustCollisionDensityThresh.
 *      Dense-phase mode: the rate is scaled by the diffuse fraction 1 - f_dense.
 *   3. It then shatters with probability P = 1 - exp(-dt/tau_shatter)
 *      this step (a Poisson process).
 *
 * WHAT SIZE THE FRAGMENTS ARE
 * ---------------------------
 * We used to set a_new = 0.33 * a_old on every event, so all shattered grains
 * landed on one fixed daughter size. Now fragments follow dN/da ∝ a^(-q) and the
 * new radius is drawn from the matching mass distribution dM/da ∝ a^(3-q)
 * (see sample_shattering_fragment_radius). Runtime parameters:
 *
 *          q                  = DustShatteringFragmentSlope
 *          largest fragment   = DustShatteringMaxFragmentFraction * a_old
 *          smallest fragment  = DustMinGrainSize
 *
 * MASS AND COMPOSITION ARE CONSERVED EXACTLY
 * ------------------------------------------
 * Only GrainRadius changes. We never call setMass(), never change
 * CarbonMassFraction, and never create or delete a particle, so
 * M_dust,new = M_dust,old for every event. Same mass at a smaller radius just
 * means the particle now stands for more, smaller grains.
 */
void dust_grain_shattering(simparticles *Sp, int dust_idx, int gas_idx, double dt,
                           double local_DTG, double T_gas)
{
    const double fragment_q = DustShatteringFragmentSlope;
    const double fragment_max_fraction =
        DustShatteringMaxFragmentFraction;


    // =========================================================================
    // Diagnostic counters (per task, summed over the whole run)
    // =========================================================================

    static long long shat_calls       = 0;
    static long long shat_failed_vel  = 0;
    static long long shat_failed_dens = 0;
    static long long shat_failed_size = 0;
    static long long shat_passed      = 0;
    static long long shat_failed_dtg  = 0;

    // Number of events by new radius (nm).
    static long long frag_1_10   = 0;
    static long long frag_10_30  = 0;
    static long long frag_30_50  = 0;
    static long long frag_50_70  = 0;
    static long long frag_70_90  = 0;
    static long long frag_ge90   = 0;

    // Dust mass shattered into each radius bin.
    static double frag_mass_1_10  = 0.0;
    static double frag_mass_10_30 = 0.0;
    static double frag_mass_30_50 = 0.0;
    static double frag_mass_50_70 = 0.0;
    static double frag_mass_70_90 = 0.0;
    static double frag_mass_ge90  = 0.0;

    // Running sums for the mean parent and fragment radius.
    static double sum_parent_radius   = 0.0;
    static double sum_fragment_radius = 0.0;

    shat_calls++;


    // =========================================================================
    // Periodic report (task 0, every 50000 calls)
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
    // Gate order
    // =========================================================================
    //
    // The gates run cheapest first: grain validity is two array reads, density
    // needs only SphP[].Density, and the velocity gate (T_gas and several
    // pow() calls) comes last. Each gate just returns early, so the order does
    // not change which grains shatter, only which counter records a rejection.


    // =========================================================================
    // Gate 1: is the grain valid and above the size floor?
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
    // Is there room for fragments?
    // =========================================================================
    //
    // If the largest allowed fragment is at or below DustMinGrainSize, no
    // fragment size is possible. We leave the grain alone rather than destroy
    // the whole particle.
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
    // Gate 2: is the gas diffuse enough?
    // =========================================================================
    //
    // Shattering happens in diffuse gas; coagulation (grains sticking
    // together) happens in dense gas.
    //

    double n_H = dust_gas_nH_cgs(Sp, gas_idx);  // hydrogen number density, cm^-3

    // f_diffuse = fraction of the gas element that is diffuse.
    // Legacy (mode <= 1): all or nothing, switched at n_eff = DustCollisionDensityThresh,
    //   where n_eff = n_H times the clumping factor.
    // Dense-phase model (mode >= 2): shattering always acts on 1 - f_dense.
    double f_diffuse = 1.0;
    double n_eff = n_H;  // logged with each shattering event
    if(dust_dense_phase_model())
    {
        double n_dense_unused;
        f_diffuse = 1.0 - dust_dense_fraction(Sp, gas_idx, &n_dense_unused);
        if(!(f_diffuse >= 1.0e-4))
        {
            shat_failed_dens++;
            return;
        }
    }
    else
    {
        double DustClumpingFactor =
            dust_clumping_factor(
                n_H,
                Sp->SphP[gas_idx].Sfr >
                DUST_SFR_EPS
            );

        n_eff =
            n_H *
            DustClumpingFactor;

        if(n_eff >
           All.DustCollisionDensityThresh)
        {
            shat_failed_dens++;
            return;
        }
    }


    // ── Check the local D/G ───────────────────────────────────────────────────
    // local_DTG (D/G = dust-to-gas mass ratio) comes from update_dust_dynamics().
    // It is the same MPI-combined neighbourhood estimate that coagulation uses,
    // so both processes see the same D/G.
    if(!isfinite(local_DTG) || !(local_DTG > 0.0))
    {
        shat_failed_dtg++;
        return;
    }


    // =========================================================================
    // Gate 3: is the grain moving fast enough to shatter?
    // =========================================================================
    //
    // How fast grains hit each other in turbulence depends on their size.
    // Small grains have short stopping times, so they move with the gas and
    // collide slowly. We use the turbulent velocity of Hirashita & Aoyama
    // (2019, MNRAS 482, 2555, Eq. 18), derived following Ormel et al.
    // (2009, A&A 502, 845) and matched to Yan et al. (2004):
    //
    //   v_gr = 1.1 km/s  M^{3/2} (a / 0.1 um)^{1/2} (T / 1e4 K)^{1/4}
    //          (n_H / 1 cm^-3)^{-1/4} (rho_gr / 3.5 g cm^-3)^{1/2}
    //
    // M is the turbulent Mach number. n_H is the unclumped density, floored at
    // 1e-4 cm^-3 as in the timescale below. The grain material density rho_gr
    // blends silicate (3.5 g cm^-3) and graphite (2.24 g cm^-3) by
    // CarbonMassFraction.
    //
    // The old gate compared the gas sound speed (the same for every grain size)
    // with the threshold and never rejected anything (vel = 0.0% of calls).
    // Grains in diffuse gas shattered again and again down to DustMinGrainSize.
    // Grains in [DustMinGrainSize, DustMinGrainSize/0.8] could not shatter again,
    // and with only ~5% of their mass in gas dense enough for coagulation they
    // piled up there (12% of all dust mass at z=0 in the coag6_shat2 run).
    // With this gate, shattering stops at a physical critical size set by the
    // environment (~30 nm in the central galaxy's ISM for M = 1).
    //
    // Shattering threshold v_shatter (unchanged), linear in CarbonMassFraction:
    //   silicate      ~ 1 km/s
    //   pure carbon   ~ 2 km/s
    //
    // The critical size scales as M^-3, so shattering_mach is the most
    // sensitive choice here. Hirashita & Aoyama treat M as a tuning parameter
    // and adopt M = 3 for shattering (M = 1 for coagulation). We keep M = 1 and
    // calibrate the shattering rate with this gate in place.
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

    // ── Shattering timescale: Trayford et al. (2026) Eq. 9 ───────────────────
    // Adapted to CosmicGrain's continuous grain radius:
    //
    // τ_sh = 54.1 Myr × (DTG/0.01)^-1 × (a/0.1μm) × f(n_H)
    //   × DustShatteringCalibration / f_diffuse,  clamped to [1 Myr, 10 Gyr]
    //
    // f(n_H) changes slope at n_H = 1 cm^-3:
    //   f = n_H^-1       for n_H < 1 cm^-3  (collision rate ∝ density)
    //   f = n_H^(-1/3)   for n_H >= 1 cm^-3 (grains more coupled to turbulence)
    //
    // n_H is the UNCLUMPED density, matching Trayford's view of shattering as a
    // diffuse-gas process. (Coagulation uses the clumped n_eff.)
    //
    // Trayford et al. fold the collision speed into the 54.1 Myr normalization.
    // We test the speed directly with the v_grain(a) >= v_shatter gate above,
    // so we drop their velocity_factor term.

    double a_micron = a / 1000.0;  // nm -> micron

    // Density dependence of the collision rate. The 1e-4 cm^-3 floor keeps
    // the timescale finite in very diffuse CGM (circumgalactic medium) gas.
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
        All.DustShatteringCalibration /
        f_diffuse;

    // Keep the timescale between 1 Myr and 10 Gyr.
    tau_shat_yr =
        std::max(tau_shat_yr, 1.0e6);

    tau_shat_yr =
        std::min(tau_shat_yr, 1.0e10);

    double tau_shat =
        tau_shat_yr *
        SEC_PER_YEAR /
        dust_code_time_in_s();


    // =========================================================================
    // Roll the dice: does the grain shatter this step?
    // =========================================================================
    //
    // Poisson probability of at least one event in dt.
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
    // Draw the new grain radius
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
    // Reject a bad draw (should not happen; logged and skipped if it does)
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
    // GrainRadius is the ONLY particle property shattering changes.
    //

    Sp->DustP[dust_idx].
        GrainRadius =
        a_new;


    // =========================================================================
    // Conservation checks
    // =========================================================================
    //
    // Nothing in this path calls setMass(). These checks are here so that a
    // future edit that breaks mass or composition conservation stops the run
    // at once instead of slipping through.
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
    // Statistics
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
    // Fragment-size bins (nm; the lowest bin holds everything below 10 nm)
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
    // Event log (task 0: first 100 events, then every 10000th)
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
 * Shrink or destroy a dust grain in hot gas. In thermal sputtering, fast gas
 * ions knock atoms off the grain surface. Grains that get hotter than their
 * sublimation temperature are destroyed outright.
 *
 * Sputtering timescale from McKinnon et al. (2017), Eq. 2:
 *   τ_sp = 0.17 Gyr × (a_-1 / ρ_-27) × [(T0/T)^ω + 1]
 * where a_-1 = a / 0.1 µm, ρ_-27 = ρ_gas / (1e-27 g/cm³), T0 = 2e6 K, ω = 2.5.
 *
 * COMPOSITION: the carbon and silicate parts of the grain are sputtered
 * separately, each at its own rate. We used to apply one CF-weighted blended
 * rate to the whole grain. Carbon is held less tightly (binding energy 4 eV vs
 * 6 eV for silicate), so it erodes faster and surviving grains drift toward
 * silicate over repeated episodes. The radius still describes one grain size,
 * set by the TOTAL surviving mass (a ∝ m^(1/3)); only CarbonMassFraction
 * (CF) changes independently of the radius.
 *
 * Returns 1 if the particle was destroyed, 0 otherwise.
 */
int erode_dust_grain_thermal(simparticles *Sp, int dust_idx, int nearest_gas_input,
                              double T_gas, double dt)
{
  if(!All.DustEnableSputtering) return 0;
  double a = Sp->DustP[dust_idx].GrainRadius;
  // ── Sublimation: the grain itself is too hot to survive ───────────────────
  // Threshold 1500 K (silicate) to 2000 K (carbon), blended by CF.
  double T_sublimate = 1500.0 + 500.0 * Sp->DustP[dust_idx].CarbonMassFraction;
  if(Sp->DustP[dust_idx].DustTemperature > T_sublimate) {
    DUST_PRINT("[SUBLIMATION] Dust destroyed at T_dust=%.0f K\n",
               Sp->DustP[dust_idx].DustTemperature);
    log_dust_particle_event(Sp, dust_idx, nearest_gas_input, DUST_EVENT_SUBLIMATION);
    return destroy_dust_particle_to_gas(Sp, dust_idx, nearest_gas_input,
                                        &NDustDestroyedByThermal,
                                        &TotalMassDestroyedByThermal);
  }
  // ── Gas too cool to sputter: nothing to do ───────────────────────────────
  if(T_gas < All.DustThermalSputteringTemp) return 0;
  // ── McKinnon+2017 sputtering timescale ───────────────────────────────────
  const double T0    = 2e6;   // K, characteristic sputtering temperature
  const double omega = 2.5;   // power-law index (Tielens et al. 1994)
  double a_cgs       = a * 1e-7;            // nm → cm
  double a_minus1    = a_cgs / 1e-5;        // units of 0.1 µm
  double rho_cgs     = dust_gas_density_cgs(Sp, nearest_gas_input);
  double rho_minus27 = rho_cgs / 1e-27;
  // Fallback if the gas density is zero or not finite.
  // Known issue (kept for v1.2.0, which the Paper II run used): the fallback below
  // was meant to be n_H = 1e-4 cm^-3, i.e. rho_minus27 ~ 0.17, but the extra
  // "/ 1e-27" makes it ~1.7e26.  So a grain whose gas density is zero or NaN gets
  // the 1 Myr floor (fastest sputtering) instead of slow sputtering.  This only
  // happens for broken density values, so it should be rare.  Fix in the next version.
  if(rho_minus27 <= 0.0 || !isfinite(rho_minus27))
    rho_minus27 = 1e-4 / 1e-27 * PROTONMASS / 1e-27;  // fallback value
  double tau_base_yr = 0.17e9 * (a_minus1 / rho_minus27) * (pow(T0 / T_gas, omega) + 1.0);
  // ── Per-species speed-up factors ─────────────────────────────────────────
  // Ratio of binding energies, 6 eV (silicate) / 4 eV (carbon). These used to
  // be blended into one CF-weighted composite_factor; now each species gets
  // its own timescale, applied to its own share of the mass below.
  double composition_factor_carbon   = 6.0 / 4.0;  // = 1.5
  double composition_factor_silicate = 6.0 / 6.0;  // = 1.0
  double tau_sputter_carbon_yr   = tau_base_yr / composition_factor_carbon;
  double tau_sputter_silicate_yr = tau_base_yr / composition_factor_silicate;
  // Keep each timescale between 1 Myr and 1 Gyr.
  tau_sputter_carbon_yr   = std::max(1e6, std::min(1e9, tau_sputter_carbon_yr));
  tau_sputter_silicate_yr = std::max(1e6, std::min(1e9, tau_sputter_silicate_yr));
  double tau_sputter_carbon   = tau_sputter_carbon_yr   * SEC_PER_YEAR / dust_code_time_in_s();
  double tau_sputter_silicate = tau_sputter_silicate_yr * SEC_PER_YEAR / dust_code_time_in_s();
  // ── Split the mass into carbon and silicate; erode each on its own ───────
double CF_old         = std::max(0.0, std::min(1.0, (double)Sp->DustP[dust_idx].CarbonMassFraction));
double old_mass       = Sp->P[dust_idx].getMass();
double M_carbon_old   = old_mass * CF_old;
double M_silicate_old = old_mass * (1.0 - CF_old);
// Exact exponential decay over dt for each species.
double M_carbon_new = M_carbon_old * exp(-dt / tau_sputter_carbon);
double M_silicate_new = M_silicate_old * exp(-dt / tau_sputter_silicate);
double new_mass = M_carbon_new + M_silicate_new;
// Mass of each species that goes back to the gas.
double carbon_lost = M_carbon_old - M_carbon_new;
double silicate_lost = M_silicate_old - M_silicate_new;
double mass_lost = carbon_lost + silicate_lost;
// Guard against tiny negative values from roundoff.
carbon_lost = std::max(0.0, carbon_lost);
silicate_lost = std::max(0.0, silicate_lost);
// Radius follows the total surviving mass.
double a_new = a * cbrt(new_mass / old_mass);
// ── Full destruction ─────────────────────────────────────────────────────
//
// IMPORTANT: do NOT return carbon_lost / silicate_lost to the gas first.
// destroy_dust_particle_to_gas() returns the particle's whole current mass
// (still old_mass, split by the stored CarbonMassFraction). Returning the
// partial losses as well would count that material twice.
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
double CF_new = M_carbon_new / new_mass;
Sp->DustP[dust_idx].GrainRadius = a_new;
Sp->DustP[dust_idx].CarbonMassFraction = (float)CF_new;
Sp->P[dust_idx].setMass(
    new_mass);
// Give exactly the sputtered carbon and silicate back to the gas.
if(nearest_gas_input >= 0)
{
    return_dust_material_to_gas(
        Sp,
        nearest_gas_input,
        carbon_lost,
        silicate_lost);
}
// Bookkeeping; task 0 logs every 10000th erosion event.
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
 * Two helpers for destroying a dust particle.
 *
 * mark_dust_particle_destroyed() flags the particle for removal. It does NOT
 * give any material back to the gas, so the caller must already have done
 * that. Both thermal destruction (via destroy_dust_particle_to_gas) and SN
 * shock destruction use it. Astration (dust locked into new stars) takes a
 * separate path that removes mass in proportion instead of whole particles.
 *
 * destroy_dust_particle_to_gas() returns the particle's carbon and silicate
 * mass to the gas element nearest_gas, then marks it destroyed. Used for
 * thermal sputtering and sublimation.
 *
 * A destroyed particle becomes type 3 with mass DUST_MASS_TO_DESTROY (1e-30)
 * and ID 0, and destroy_dust_particles() removes it on its next call.
 *
 * destroy_dust_particle_to_gas() returns 1 if the particle was destroyed, and
 * 0 (particle left untouched) if there is no gas element to receive the mass.
 */
// Flag a dust particle for removal. Its material must already be accounted for.
static void mark_dust_particle_destroyed(simparticles *Sp, int dust_idx,
                                         long long *counter, double *mass_counter)
{
  const double dust_mass = Sp->P[dust_idx].getMass();
  Sp->P[dust_idx].setMass(DUST_MASS_TO_DESTROY);
  Sp->P[dust_idx].setType(3);
  Sp->P[dust_idx].ID.set(0);
  memset(&Sp->DustP[dust_idx], 0, sizeof(dust_data));
  Sp->DustP[dust_idx].GrainRadius = All.DustMinGrainSize;
  LocalDustMassChange -= dust_mass;
  LocalDustDestroyedThisStep++;
  /* The particle has changed type, so it must leave the dust and gravity
   * time-bin lists before the next force calculation or snapshot. */
  DustNeedsSynchronization = 1;
  if(counter)      (*counter)++;
  if(mass_counter) (*mass_counter) += dust_mass;
}

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
  mark_dust_particle_destroyed(Sp, dust_idx, counter, mass_counter);
  return 1;
}
// ═══════════════════════════════════════════════════════════════════════════════
// calculate_sedov_velocity_from_radius
// ═══════════════════════════════════════════════════════════════════════════════
/**
 * Speed of a supernova blast wave (Sedov-Taylor phase) when its shock front
 * has reached physical radius radius_kpc. Gas density rho_cgs in g/cm³,
 * explosion energy energy_erg in erg.
 *
 * The self-similar solution (Sedov 1959) gives R = ξ (E t²/ρ)^(1/5), so
 * v = dR/dt = (2/5) R/t, which becomes
 *   v = (2/5) × ξ^(5/2) × √(E/ρ) × R^(−3/2)
 *
 * with ξ = 1.033 the dimensionless Sedov constant.
 * Returns velocity in km/s.
 */
double calculate_sedov_velocity_from_radius(double radius_kpc, double rho_cgs, double energy_erg)
{
  // Known issue (kept for v1.2.0): 1.033 is the Sedov constant for gamma = 1.4 (air).
  // For monatomic gas (gamma = 5/3) it is about 1.15.  The shock model evaluates R and v
  // at a fixed time (0.3 Myr), where both scale with xi: about 12% larger and faster,
  // so about 1.4x the swept volume and about 1.4x (carbon) to 2x (silicate) the dust
  // destroyed per SN.  Shock destruction was calibrated with 1.033, so changing it
  // needs a recalibration.  Planned for the next version.
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
 * Stopping time of a dust grain under Epstein drag: roughly how long gas
 * drag takes to bring the grain to the gas velocity. Epstein drag applies
 * when the grain is much smaller than the gas mean free path.
 *
 * McKinnon et al. (2018), Eqs. 8-9:
 *
 * t_stop (subsonic) = √(π γ / 8) × (a ρ_grain) / (ρ_gas c_s)
 *
 * a = grain radius, ρ_grain = grain material density, c_s = gas sound speed.
 * For fast grains, multiply by 1 / √(1 + 9π M² / 128), where M is the
 * grain-gas Mach number (relative speed / c_s). Applied when M > 0.1.
 *
 * Returns t_stop in Myr, clamped to [0.001, 50] Myr; near-vacuum gas
 * (c_s < 10 km/s or ρ_gas < 1e-30 g/cm³) returns 50 Myr. The upper clamp is a
 * numerical safeguard: an unclamped stopping time in the diffuse CGM could
 * put the grain on an unrealistically short gravity timebin via the drag
 * velocity kick. dust_radiation_pressure computes its own unclamped t_stop for
 * its terminal-velocity correction.
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
  // No meaningful t_stop in (near) vacuum: return the upper limit.
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
 * Gas drag and thermal sputtering for one dust grain and its nearest gas particle.
 *
 * DRAG: Epstein drag pulls the grain's velocity toward the gas velocity. We use
 * the exact exponential answer, Δv = v_rel × (1 − exp(−dt/t_stop)), instead of a
 * linear step, so it stays stable however large dt is compared with the stopping
 * time t_stop.  dt is the real elapsed time in code-time units, supplied by the
 * caller.
 *
 * SPUTTERING: in hot gas, ions knock atoms off the grain.  This is done by
 * erode_dust_grain_thermal(), which also uses an exact exponential form.
 *
 * Both are skipped when DustEnableDrag is off.
 * Returns 1 if sputtering destroyed the dust particle, 0 otherwise.
 */
int dust_gas_interaction(simparticles *Sp, int dust_idx, int nearest_gas, double dt,
                         double T_gas)
{
  if(!All.DustEnableDrag) return 0;
  if(nearest_gas < 0) return 0;
  // ── Guard: the "nearest gas" index no longer points at gas ───────────────
  // A stale neighbour lookup can return a gas slot that has since become a star.
  // Reading SphP[] for a non-gas particle is undefined behaviour and can give a
  // garbage T_gas that triggers sputtering by mistake, so we skip the grain.
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
  // ── Gas properties ────────────────────────────────────────────────────────
  double gas_vel[3]     = { Sp->P[nearest_gas].Vel[0],
                             Sp->P[nearest_gas].Vel[1],
                             Sp->P[nearest_gas].Vel[2] };
  double gas_density_cgs= dust_gas_density_cgs(Sp, nearest_gas);
  double n_H            = dust_gas_nH_cgs(Sp, nearest_gas);
  // ── Relative velocity ─────────────────────────────────────────────────────
  double vrel_x = Sp->P[dust_idx].Vel[0] - gas_vel[0];
  double vrel_y = Sp->P[dust_idx].Vel[1] - gas_vel[1];
  double vrel_z = Sp->P[dust_idx].Vel[2] - gas_vel[2];
  // P[].Vel stores p = a^2 dx/dt (a = scale factor), so the physical velocity
  // difference is dp / a.  Only the magnitude (used for the Mach number) needs
  // converting: the exponential pull-toward-gas below works on p directly and
  // gives the same answer in any units.
  double vrel   = sqrt(vrel_x*vrel_x + vrel_y*vrel_y + vrel_z*vrel_z) * All.cf_ainv;
  double vrel_cgs = vrel * All.UnitVelocity_in_cm_per_s;
  // ── Epstein stopping time ─────────────────────────────────────────────────
  // Grain material density 2.4 g/cm^3, gas mean molecular weight 0.6, gamma 5/3.
  double drag_timescale_myr = calculate_drag_timescale(
      Sp->DustP[dust_idx].GrainRadius, 2.4,
      gas_density_cgs, T_gas, vrel_cgs, 0.6, 5.0/3.0);
  double drag_timescale = drag_timescale_myr * 1e6 * SEC_PER_YEAR / dust_code_time_in_s();
  // ── Apply drag (exact solution) ───────────────────────────────────────────
  // Δv_k = (v_gas,k − v_dust,k) × (1 − exp(−dt/t_stop)), exact for any dt/t_stop.
  // dt is the physical elapsed time in code-time units; in cosmological runs the
  // caller converts it from the scale-factor step.
  double drag_factor = 1.0 - exp(-dt / drag_timescale);
  for(int k = 0; k < 3; k++)
    Sp->P[dust_idx].Vel[k] += drag_factor * (gas_vel[k] - Sp->P[dust_idx].Vel[k]);
  // ── Thermal sputtering (only above DustThermalSputteringTemp) ─────────────
  if(T_gas > All.DustThermalSputteringTemp) {
    int destroyed = erode_dust_grain_thermal(Sp, dust_idx, nearest_gas, T_gas, dt);
    return destroyed;
  }
  // ── Diagnostic sample: task 0, 1 in 100 particle IDs, first 500 prints ────
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
 * Create dust superparticles (PartType6) from one stellar feedback event.
 * A superparticle is one simulation particle that stands for a huge number of
 * real grains with the same size and composition.
 *
 * carbon_dust_mass and silicate_dust_mass are the dust masses (code mass units)
 * that condensed in the stellar ejecta.  feedback.cc has already decided how
 * the ejected elements split between gas and dust; this routine only spawns
 * the dust particles and never touches gas.
 *
 * feedback_type: 1 = SNII (core-collapse supernova), 2 = AGB (asymptotic giant
 * branch star), 3 = LRN (luminous red nova).
 * particle_count_override >= 0 replaces the per-source particle count.
 * conserve_agb_momentum: give the star the recoil of each grain it launches.
 * Returns the total dust mass created (0 if none).
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
// Use fewer particles if splitting the mass n ways would make each one lighter
// than min_particle_mass (code units), but always at least one.
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
// Each particle goes in a random direction (uniform on the sphere), at a random
// distance between the source's DustOffsetMin* and DustOffsetMax*, and is
// kicked outward along that same direction at DustVelocity* (km/s).
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

  // P[].Vel holds p = a^2 dx/dt, so a physical kick v enters as a*v.  The factor
  // is applied to every source so DustVelocity* means physical km/s at all
  // redshifts.  (It used to be applied only on the AGB momentum-conserving path,
  // so SNII and LRN kicks were 1/a too large.)  The 1e5 converts km/s to cm/s.
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

  // Initial grain radius (nm): log-normal draw for SNII and AGB, a fixed 10 nm
  // for LRN.  DustSource: 0 = SNII, 1 = AGB, 2 = LRN.
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

  // AGB recoil: the star loses the grain's mass and momentum, so total momentum
  // is conserved.  Large recoils (> 100 km/s physical) are printed as a warning.
  if(conserve_agb_momentum)
    {
      const double star_mass_after = star_mass_remaining - dust_mass_per_particle;
      if(star_mass_after <= 0.0 || !std::isfinite(star_mass_after))
        Terminate("AGB dust recoil would exhaust stellar mass");
      double dv2 = 0.0;
      for(int k = 0; k < 3; ++k) {
        const double star_vel_before = Sp->P[star_idx].Vel[k];
        const double dust_vel = Sp->P[new_idx].Vel[k];
        Sp->P[star_idx].Vel[k] =
            (star_mass_remaining * star_vel_before - dust_mass_per_particle * dust_vel) /
            star_mass_after;
        dv2 += (Sp->P[star_idx].Vel[k] - star_vel_before) * (Sp->P[star_idx].Vel[k] - star_vel_before);
      }
      if(sqrt(dv2) * All.cf_ainv > 100.0)
        printf("[AGB_DUST_RECOIL_KICK|T=%d] star ID %llu changed velocity by %.0f km/s (dust %.3e of %.3e)\n",
               All.ThisTask, (unsigned long long)Sp->P[star_idx].ID.get(), sqrt(dv2) * All.cf_ainv,
               dust_mass_per_particle, star_mass_remaining);
      star_mass_remaining = star_mass_after;
    }

}

if(n_created != n_create)
  Terminate("Dust creation count mismatch: created=%d expected=%d",
            n_created, n_create);


// ── Bookkeeping, then a velocity sample (task 0, first 50 events) ─────────
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
 * Add this update's dust counts to the running totals and hand each task a
 * fresh block of particle IDs.  Collective: all tasks must call it (MPI_Allreduce).
 *
 * Called after each dust-physics update.  With a reduced dust cadence that is
 * not every global synchronization point.
 *
 * ID BLOCKS: between calls, each task raises its own All.MaxID as it spawns
 * dust.  So that two tasks never hand out the same ID, each task gets its own
 * range of block_size IDs above the global maximum after every sync.
 * block_size must be larger than the number of dust particles any one task can
 * create between syncs.  If spawn_dust_particle prints DUST_ID_OVERFLOW,
 * increase it.
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
  /* Do not clear DustNeedsSynchronization here.  The main run loop does the
   * particle compaction and time-bin rebuild, and clears the flag only after
   * that collective step has finished. */
  MyIDType local_max  = All.MaxID;
  MyIDType global_max = 0;
  MPI_Allreduce(&local_max, &global_max, 1, MPI_MYIDTYPE, MPI_MAX, Communicator);
  // Task t gets IDs starting at global_max + 1 + t * block_size.
  const MyIDType block_size = 200000;
  All.MaxID         = global_max + 1 + (MyIDType)All.ThisTask * block_size;
  All.MaxIDBlockEnd = All.MaxID + block_size - 1;
}
// ═══════════════════════════════════════════════════════════════════════════════
// spawn_dust_particle
// ═══════════════════════════════════════════════════════════════════════════════
// Add one dust particle at the end of the particle array, next to star_idx.
//
// Gravity time bin at birth: max(DUST_MIN_TIMEBIN, HighestActiveTimeBin).
// At late times (z < 5) HighestActiveTimeBin is usually <= 15, so the floor
// DUST_MIN_TIMEBIN wins.  At early times (z > 10) HighestActiveTimeBin can be
// 21 or more.  Without the max(), dust would be born on bin 15, which is not
// synchronized at that moment, and the collective gravity step would hang on
// multi-node runs.
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
  // Start from a full copy of the parent star.  The slot at NumPart still holds
  // whatever particle used it last (a destroyed or compacted grain, or a grain
  // visiting from another task during the dust update).  Any field not set
  // below, such as the stored gravitational acceleration used by the next
  // half-kick, would otherwise come from that old particle.  Such stale
  // accelerations once launched new grains at thousands of km/s.
  Sp->P[new_idx] = Sp->P[star_idx];
  memset(&Sp->DustP[new_idx], 0, sizeof(dust_data));
  // ── Position: parent star + random offset ─────────────────────────────────
  double star_pos[3];
  Sp->intpos_to_pos(Sp->P[star_idx].IntPos, star_pos);
  // Birth position = the star's position (without the offset), so we can later
  // measure how far the dust has travelled from where it was made.
  Sp->DustP[new_idx].BirthPos[0] = star_pos[0];
  Sp->DustP[new_idx].BirthPos[1] = star_pos[1];
  Sp->DustP[new_idx].BirthPos[2] = star_pos[2];
  double dust_pos[3];
  for(int d = 0; d < 3; d++) {
    dust_pos[d] = star_pos[d] + offset_kpc[d];
    // Wrap into the periodic box
    while(dust_pos[d] <  0.0)          dust_pos[d] += All.BoxSize;
    while(dust_pos[d] >= All.BoxSize)  dust_pos[d] -= All.BoxSize;
  }
  Sp->pos_to_intpos(dust_pos, Sp->P[new_idx].IntPos);
  // Nudge each integer coordinate by -2..+1 units.  Several grains from one star
  // event can land on almost the same IntPos, which crashes the tree build.
  // The nudge is far below resolution, so it has no physical effect.
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
  // ── ID from this task's block (see dust_global_synchronization) ───────────
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
  // ── Placeholders; create_dust_particles_from_feedback overwrites these ────
  Sp->DustP[new_idx].GrainRadius    = All.DustMinGrainSize;
  Sp->DustP[new_idx].CarbonMassFraction = 0.3;
  Sp->DustP[new_idx].DustSource      = 2;
  // Start at the CMB temperature, T_CMB = 2.7 K × (1+z) = 2.7 K / a.
  Sp->DustP[new_idx].DustTemperature = 2.7 / All.Time;
  // No cached D/G (dust-to-gas mass ratio) yet; it is computed on this
  // particle's first dust call.
  Sp->DustP[new_idx].LocalDTG = 0.0f;
  Sp->DustP[new_idx].DTGAge   = 0;
  Sp->P[new_idx].StellarAge  = All.Time;  // birth time (scale factor)
  Sp->P[new_idx].Ti_Current  = All.Ti_Current;
  Sp->P[new_idx].TimeBinHydro = 0;  // dust does not take part in SPH hydro
  // Born on the DUST_MIN_TIMEBIN floor, or on a higher active bin early on (see
  // the note above the function); get_timestep_grav then moves it to its own bin.
  int dust_spawn_bin = std::max(DUST_MIN_TIMEBIN, (int)All.HighestActiveTimeBin);
  Sp->P[new_idx].TimeBinGrav = dust_spawn_bin;
  Sp->TimeBinsGravity.timebin_add_particle(new_idx, star_idx, dust_spawn_bin,
                                            Sp->TimeBinSynchronized[dust_spawn_bin]);
  // Sanity check: the radius must be positive and finite.
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
 * Print a 6-bin histogram of grain radius (nm), by count and by mass.
 * Task 0 only, and only its own particles, so it is a sample, not a global total.
 * Called now and then from the diagnostics in update_dust_dynamics.
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
// Astration
// ═══════════════════════════════════════════════════════════════════════════════
/**
 * Astration: when gas turns into a star, the dust mixed into that gas goes into
 * the star too.  We remove dust mass = stellar_mass_formed × local D/G
 * (dust-to-gas mass ratio) and add it to the new star.
 *
 * Search radius = the larger of the gas cell's own size (the radius of a sphere
 * with its mass and density) and its SPH smoothing length, the same local ISM
 * neighbourhood used by growth, coagulation and shattering.  The mass to remove
 * is shared among all dust in the sphere with weight 1/distance, so the closest
 * grains lose the most.  Grains left below DUST_MASS_TO_DESTROY are removed.
 *
 * Two steps: star formation only queues the event (consume_dust_by_astration).
 * apply_queued_dust_astration() then consumes the dust once all of the step's
 * stars exist.  It is collective over all tasks, so grains on other tasks inside
 * the sphere take part.
 */
struct dust_astration_event
{
  MyIntPosType pos[3];
  std::uint64_t key;  // star ID; events are applied in key order
  double radius, gas_mass, stellar_mass;
  int star_index;
};
static std::vector<dust_astration_event> queued_astration;

// Called from star formation: record the event for apply_queued_dust_astration.
void consume_dust_by_astration(simparticles *Sp, int gas_idx, double stellar_mass_formed,
                                int star_idx, double hsml)
{
  if(!All.DustEnableAstration) return;
  const double gas_mass    = Sp->P[gas_idx].getMass();
  const double rho_code    = Sp->SphP[gas_idx].Density * All.cf_a3inv;
  const double cell_radius = cbrt(3.0 * gas_mass / (4.0 * M_PI * rho_code));
  dust_astration_event ev;
  for(int k = 0; k < 3; k++) ev.pos[k] = Sp->P[gas_idx].IntPos[k];
  ev.key          = 0;  // spawned stars get their IDs after the SF loop
  ev.radius       = std::max(cell_radius, hsml);
  ev.gas_mass     = gas_mass;
  ev.stellar_mass = stellar_mass_formed;
  ev.star_index   = star_idx;
  queued_astration.push_back(ev);
}

// Collective: all tasks must call this, even with no events of their own.
// Every task gets the full list of events, finds its own dust (and gas) in each
// sphere, and the sums are added up over tasks.  Events are then applied in
// star-ID order, so the result does not depend on how particles are split
// across tasks.
void apply_queued_dust_astration(simparticles *Sp, MPI_Comm Communicator)
{
  if(!All.DustEnableAstration) return;
  for(dust_astration_event &ev : queued_astration)
    ev.key = (std::uint64_t)Sp->P[ev.star_index].ID.get();
  std::vector<int> owner;
  const std::vector<dust_astration_event> all = cg_dist::allgather(queued_astration, Communicator, &owner);
  const size_t n_events = all.size();
  int this_task;
  MPI_Comm_rank(Communicator, &this_task);
  // This task's own events start at index `first` in the gathered list.
  size_t first = 0;
  while(first < n_events && owner[first] < this_task) first++;
  if(n_events == 0) { queued_astration.clear(); return; }

  static long long astration_attempts = 0, astration_zero_neighbor = 0, astration_zero_dustmass = 0;
  static long long astration_count = 0;

  cg_dist::LocalTree dust_tree;
  dust_tree.build(Sp, 0, Sp->NumPart, [Sp](int i) {
    return Sp->P[i].getType() == DUST_PARTICLE_TYPE && Sp->P[i].getMass() > 0.0 && Sp->P[i].ID.get() != 0;
  });
  // Gas in the same spheres, for the local D/G.
  cg_dist::LocalTree gas_tree;
  gas_tree.build(Sp, 0, Sp->NumGas, [Sp](int j) { return Sp->P[j].getType() == 0 && Sp->P[j].getMass() > 0.0; });
  std::vector<double> gas_in_sphere(n_events, 0.0);
  for(size_t e = 0; e < n_events; e++)
    gas_tree.radius(Sp, all[e].pos, all[e].radius, [&](int j, double) { gas_in_sphere[e] += Sp->P[j].getMass(); });
  cg_dist::sum_over_tasks(gas_in_sphere, Communicator);

  // Per event: dust mass, 1/distance weight, count; summed over all tasks.
  std::vector<double> sums(3 * n_events, 0.0);
  std::vector<std::vector<std::pair<int, double>>> found(n_events);
  // Test switch: environment variable CG_TEST_LOCAL_ASTRATION=1 brings back the
  // old behaviour, where a forming star only sees dust on its own task.
  static const bool local_only = getenv("CG_TEST_LOCAL_ASTRATION") && atoi(getenv("CG_TEST_LOCAL_ASTRATION")) == 1;
  for(size_t e = 0; e < n_events; e++)
    if(!local_only || owner[e] == this_task)
    dust_tree.radius(Sp, all[e].pos, all[e].radius, [&](int i, double d2) {
      const double d = std::sqrt(d2);
      const double w = d > 0 ? 1.0 / d : 1.0;
      sums[3 * e + 0] += Sp->P[i].getMass();
      sums[3 * e + 1] += w;
      sums[3 * e + 2] += 1.0;
      found[e].emplace_back(i, w);
    });
  cg_dist::sum_over_tasks(sums, Communicator);

  std::vector<size_t> order(n_events);
  for(size_t e = 0; e < n_events; e++) order[e] = e;
  std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return all[a].key < all[b].key; });

  std::vector<double> consumed(n_events, 0.0);
  long long n_destroyed_local = 0;
  for(size_t e : order) {
    astration_attempts++;
    if(sums[3 * e + 2] == 0) { astration_zero_neighbor++; continue; }
    const double total_dust_mass = sums[3 * e];
    if(total_dust_mass < DUST_MASS_TO_DESTROY) { astration_zero_dustmass++; continue; }
    astration_count++;
    // Dust mass to consume = stellar mass formed × local D/G × DustAstrationCalibration
    // (physically 1), capped at the dust present.  D/G is dust over gas in the
    // same sphere, counting the gas that just became the star as gas.  (D/G used
    // to be all the dust in the sphere over the single gas particle's mass, which
    // was ~DesNumNgb times too large.)
    const double gas_mass = gas_in_sphere[e] + all[e].stellar_mass;
    const double local_DG = total_dust_mass / gas_mass;
    const double dust_to_consume =
        std::min(all[e].stellar_mass * local_DG * All.DustAstrationCalibration, total_dust_mass);
    for(const auto &f : found[e]) {
      const int di = f.first;
      if(Sp->P[di].getType() != DUST_PARTICLE_TYPE) continue;  // consumed by an earlier event
      const double current_mass = Sp->P[di].getMass();
      const double mass_loss    = dust_to_consume * f.second / sums[3 * e + 1];
      const double new_mass     = current_mass - mass_loss;
      if(new_mass < DUST_MASS_TO_DESTROY) {
        log_dust_particle_event(Sp, di, -1, DUST_EVENT_ASTRATION);
        Sp->P[di].setMass(0.0);
        Sp->P[di].setType(3);
        Sp->P[di].ID.set(0);
        memset(&Sp->DustP[di], 0, sizeof(dust_data));
        DustNeedsSynchronization = 1;
        n_destroyed_local++;
        consumed[e] += current_mass;
      } else {
        Sp->P[di].setMass(new_mass);
        consumed[e] += mass_loss;
      }
    }
  }
  NDustDestroyedByAstration += n_destroyed_local;
  for(size_t e = 0; e < n_events; e++)
    TotalDustMassAstrated += consumed[e];  // code units; converted when printed
  cg_dist::sum_over_tasks(consumed, Communicator);

  // Add the consumed dust mass to each new star on this task.  Dust is pure
  // metal, so the star's metallicity rises accordingly.
  for(size_t s = 0; s < queued_astration.size(); s++) {
    const double m = consumed[first + s];
    const int star = queued_astration[s].star_index;
    const double star_mass = Sp->P[star].getMass();
    if(m > 0 && star_mass > 0) {
      const double new_star_mass = star_mass + m;
      Sp->P[star].setMass(new_star_mass);
      Sp->P[star].Metallicity = (star_mass * Sp->P[star].Metallicity + m) / new_star_mass;
    }
  }
  queued_astration.clear();

  if(All.ThisTask == 0) {
    static long long next_report = 1000;
    if(astration_attempts >= next_report) {
      DUST_PRINT("[ASTRATION_DIAG] attempts=%lld  zero_neighbor=%lld(%.1f%%)  "
                 "zero_dustmass=%lld(%.1f%%)  succeeded=%lld\n",
                 astration_attempts,
                 astration_zero_neighbor, 100.0 * astration_zero_neighbor / astration_attempts,
                 astration_zero_dustmass, 100.0 * astration_zero_dustmass / astration_attempts,
                 astration_count);
      while(next_report <= astration_attempts) next_report += 1000;
    }
  }
}
// ═══════════════════════════════════════════════════════════════════════════════
// radiation_pressure_efficiency (Q_pr)
// ═══════════════════════════════════════════════════════════════════════════════
/**
 * Radiation pressure efficiency Q_pr for a grain of radius a_nm and carbon
 * mass fraction CF.  Q_pr is how much of the light hitting the grain's cross
 * section actually pushes it (0 = none, ~1 = all).
 *
 * Grains much smaller than the light's wavelength barely feel it, so Q_pr ramps
 * up with size.  The ramp is tuned to a luminosity-weighted OB-star population
 * (T_eff ~ 23,000 K, peak λ ~ 125 nm, giving a₀ = 35 nm):
 *
 *   Q_pr = [a / (a + a₀)] × [0.8 + 0.4 × CF],   clamped to [0, 2]
 *
 * Silicates (CF=0): species_factor = 0.8   (absorb less)
 * Pure carbon (CF=1): species_factor = 1.2 (absorb more)
 *
 * Reference: Draine & Lee 1984.
 */
double radiation_pressure_efficiency(double a_nm, double carbon_fraction)
{
  const double a0_nm = 35.0;  // grains much smaller than a0 absorb poorly
  double Q_pr = a_nm / (a_nm + a0_nm);  // rises from 0 toward 1 with size
  // Linear blend from silicate (CF=0) to carbon (CF=1)
  double species_factor = 0.8 + 0.4 * carbon_fraction;
  Q_pr *= species_factor;
  return std::max(0.0, std::min(2.0, Q_pr));
}
// ═══════════════════════════════════════════════════════════════════════════════
// stellar_luminosity
// ═══════════════════════════════════════════════════════════════════════════════
/**
 * Rough luminosity (erg/s) of a star particle from its age.
 *
 * A star particle is a cluster of stars all born at once.  We use a simple
 * piecewise light-to-mass ratio L/M (in L_sun/M_sun):
 *   age < 3 Myr: 200;  3-10 Myr: falls to 100;  10-40 Myr: falls to 20;
 *   40-100 Myr: falls to 2 (floored at 0 afterwards).
 * O/B stars dominate until ~10 Myr; by 40 Myr only A/F stars remain.
 * Particles older than 100 Myr (cut upstream) give negligible UV.
 *
 * NOTE: This is a placeholder.  A possible improvement is a population
 * synthesis table (Starburst99, BPASS, or FSPS) interpolated in age and
 * metallicity.
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
 * Push a dust grain away from nearby young stars with their light.
 *
 * FORCE:
 *   F_rad = (L / 4πr²c) × Q_pr × πa²
 *
 * L = stellar luminosity, r = star-grain distance, c = speed of light,
 * Q_pr = radiation pressure efficiency (Draine & Lee 1984), a = grain radius.
 * We assume nothing absorbs the light between star and grain (optically thin).
 *
 * TERMINAL VELOCITY CORRECTION:
 *   Grains do not fly freely; gas drag holds them back.  When the drag time is
 *   short (t_drag << t_cross), a grain does not keep speeding up: it drifts at
 *   the speed where the light's push balances drag.  So we scale the kick by
 *
 *     f_coupling = (1 − exp(−dt/t_stop)) × t_stop / dt
 *
 *   → 1 when dt << t_stop (grain takes the full kick this step)
 *   → t_stop/dt << 1 when dt >> t_stop (terminal velocity limit, kick suppressed)
 *
 *   t_stop here does NOT use the 50 Myr cap in calculate_drag_timescale().
 *   That cap is a numerical safeguard for the drag velocity update and must not
 *   weaken the physical terminal-velocity correction in the CGM
 *   (circumgalactic medium, the thin gas around a galaxy).
 *
 * Only stars younger than 20 Myr count (their UV is OB-dominated).  They come
 * from a list of all tasks' young stars rebuilt at each dust update, so grains
 * near a domain boundary also see the stars on the other side.
 */
struct dust_young_star
{
  MyIntPosType pos[3];
  double L_cgs;
};
static std::vector<dust_young_star> young_stars;
static cg_dist::LocalTree young_star_tree;

// Collective: all tasks must call this.  Gathers every task's stars younger than
// 20 Myr (with positive luminosity) into young_stars and builds a search tree.
static void refresh_young_star_list(simparticles *Sp, MPI_Comm Communicator)
{
  std::vector<dust_young_star> mine;
  if(All.DustEnableRadiationPressure)
    for(int i = 0; i < Sp->NumPart; i++) {
      if(Sp->P[i].getType() != 4) continue;
      const double age_yr = get_stellar_age_Myr(Sp->P[i].StellarAge, All.Time) * 1.0e6;
      if(age_yr > 20e6) continue;
      dust_young_star y;
      y.L_cgs = stellar_luminosity(Sp, i);
      if(!(y.L_cgs > 0.0)) continue;
      for(int k = 0; k < 3; k++) y.pos[k] = Sp->P[i].IntPos[k];
      mine.push_back(y);
    }
  young_stars = cg_dist::allgather(mine, Communicator);
  young_star_tree.build_points(young_stars.size(), [](size_t k, MyIntPosType *x) {
    for(int d = 0; d < 3; d++) x[d] = young_stars[k].pos[d];
  });
}

void dust_radiation_pressure(simparticles *Sp, int dust_idx, int nearest_gas, double dt)
{
  if(!All.DustEnableRadiationPressure) return;
  if(young_stars.empty()) return;
  // Counters for the RADIATION_PRESSURE_RATE log line (this task only)
  static long long radp_calls       = 0;
  static long long radp_with_stars  = 0;
  static long long radp_with_young  = 0;
  static double    radp_total_accel = 0.0;
  static double    radp_total_coupling = 0.0;
  radp_calls++;
  // ── Search radius ─────────────────────────────────────────────────────────
  // Twice the gas smoothing length: the local ISM scale, which follows the
  // resolution.  Floor at two star-hash cell widths so the search is never
  // tiny.  Cap at 10 code length units (kpc), beyond which the 1/r² flux is
  // negligible.
  double search_radius = (nearest_gas >= 0)
      ? std::max(2.0 * (double)Sp->SphP[nearest_gas].Hsml, 2.0 * star_hash.cell_size)
      : 2.0 * star_hash.cell_size;
  search_radius = std::min(search_radius, 10.0);
  // All young stars within the radius.  (The former hash search kept the
  // first 50 stars of any age in hash order and then dropped the old ones.)
  std::vector<std::pair<int, double>> neighbors;
  young_star_tree.radius(Sp, Sp->P[dust_idx].IntPos, search_radius,
                         [&](int k, double d2) { neighbors.emplace_back(k, d2); });
  if(neighbors.empty()) return;
  radp_with_stars++;
  // ── Grain properties ──────────────────────────────────────────────────────
  double a_nm  = Sp->DustP[dust_idx].GrainRadius;
  double a_cm  = a_nm * 1e-7;
  double CF    = Sp->DustP[dust_idx].CarbonMassFraction;
  double Q_pr  = radiation_pressure_efficiency(a_nm, CF);
  const double c_cgs          = 2.998e10;   // cm/s
  const double rho_grain_cgs  = 2.4;        // g/cm³ (silicate)
  double m_single_grain_cgs   = (4.0/3.0) * M_PI * a_cm*a_cm*a_cm * rho_grain_cgs;
  // Sum the radiation acceleration (cm/s^2) from all young stars in range
  double a_rad[3] = {0.0, 0.0, 0.0};
  // A star particle is a cluster spread over its softening length, so the
  // 1/r^2 flux is evaluated no closer than that.  New grains are born
  // 10-100 pc from their parent star, where the unsoftened flux on a small
  // grain gave single-step kicks of thousands of km/s.
  const double r_min_code = All.SofteningTable[All.SofteningClassOfPartType[4]];
  for(const auto &nb : neighbors) {
    const dust_young_star &star = young_stars[nb.first];
    double L_cgs = star.L_cgs;
    double r_code = std::sqrt(nb.second);
    if(!(r_code > 0.0)) continue;
    // Softened comoving code distance (kpc/h) -> physical cm: × a / h × 3.086e21
    double r_cgs  = std::max(r_code, r_min_code) * (All.Time / All.HubbleParam) * 3.086e21;
    // nearest_image_intpos_to_pos(a, b) gives a - b, so dxyz = dust - star,
    // pointing away from the star (the direction light pushes the grain).
    // Dividing by r_code below turns it into a unit vector.
    double dxyz[3];
    Sp->nearest_image_intpos_to_pos(Sp->P[dust_idx].IntPos, star.pos, dxyz);
    // Acceleration of one grain:  F = (L / 4πr²c) × Q_pr × πa²,  accel = F / m_grain,
    // scaled by the DustRadiationPressureEfficiency parameter.
    double flux_cgs = L_cgs / (4.0 * M_PI * r_cgs * r_cgs * c_cgs);
    double accel    = flux_cgs * Q_pr * M_PI * a_cm * a_cm
                      / m_single_grain_cgs
                      * All.DustRadiationPressureEfficiency;
    for(int k = 0; k < 3; k++)
      a_rad[k] += accel * dxyz[k] / r_code;
  }
  // ── Terminal velocity correction ──────────────────────────────────────────
  // The true (unclamped) subsonic Epstein stopping time for the coupling factor.
  // The 50 Myr cap in calculate_drag_timescale() is a numerical safeguard for
  // the drag velocity update and must NOT be applied here.  If there is no
  // valid gas, t_stop_code stays 0 and the kick is not reduced.
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
  // ── Apply the kick, reduced by the coupling factor ────────────────────────
  double accel_code = All.UnitVelocity_in_cm_per_s / dust_code_time_in_s();  // code accel in cm/s^2
  // Physical velocity change this update (code velocity = km/s).  Optically
  // thin radiation pressure drives grains to at most a few hundred km/s, so a
  // bigger kick in one update is a numerical artefact (a long dust step close
  // to a luminous star).  We cap |dv| at RADP_MAX_DV_KMS; radp_capped counts
  // how often that happens.
  static constexpr double RADP_MAX_DV_KMS = 100.0;
  static long long radp_capped = 0;
  double dv[3], dv2 = 0.0;
  for(int k = 0; k < 3; k++) {
    dv[k] = (a_rad[k] / accel_code) * dt * coupling_factor;
    dv2 += dv[k] * dv[k];
  }
  double dv_scale = 1.0;
  if(dv2 > RADP_MAX_DV_KMS * RADP_MAX_DV_KMS) {
    dv_scale = RADP_MAX_DV_KMS / sqrt(dv2);
    radp_capped++;
  }
  for(int k = 0; k < 3; k++)
    // Physical acceleration g changes p = a^2 dx/dt by a * g * dt.
    Sp->P[dust_idx].Vel[k] += All.cf_atime * dv[k] * dv_scale;
  // Diagnostics; task 0 prints a summary every 500,000 calls
  double accel_mag = sqrt(a_rad[0]*a_rad[0] + a_rad[1]*a_rad[1] + a_rad[2]*a_rad[2]);
  if(accel_mag > 0.0) radp_with_young++;
  radp_total_accel += accel_mag;
  if(radp_calls % 500000 == 0 && All.ThisTask == 0)
    DUST_PRINT("[RADIATION_PRESSURE_RATE] calls=%lld  with_any_stars=%.1f%%  "
               "with_young_stars=%.1f%%  avg_coupling=%.3e avg_|a|=%.2e code  capped=%lld\n",
               radp_calls,
               100.0 * radp_with_stars  / radp_calls,
               100.0 * radp_with_young  / radp_calls,
               radp_total_coupling      / radp_calls,
               radp_total_accel         / radp_calls,
               radp_capped);
}
// ═══════════════════════════════════════════════════════════════════════════════
// update_dust_temperature
// ═══════════════════════════════════════════════════════════════════════════════
/**
 * Dust temperature.  Find the temperature at which the grain gives off as much
 * heat as it takes in (energy balance, modified blackbody), then move the
 * grain's temperature toward it.
 *
 * Balance condition:
 *   P_emit(T_eq) = P_CMB + P_ISRF + P_gas
 *
 * where (P = power, erg/s):
 *   P_emit  = C_emit × T^(4+β)               [modified blackbody emission]
 *   P_CMB   = C_emit × T_CMB^(4+β)           [heating by the CMB]
 *   P_ISRF  = C_emit × T_ISRF^(4+β)          [heating by the interstellar radiation field]
 *   P_gas   = f_eff × 2 n_H k_B (T_gas−T_dust) α_T v_th π a²  [gas atoms hitting the grain]
 *
 * Emissivity: Draine & Lee (1984) modified blackbody, Q_abs ∝ a T^β, β = 2
 * (β is lowered for carbon-rich grains below).
 *
 * P_gas uses the dust temperature at the start of the update, so T_eq is an
 * operator-split estimate, not the exact root of the coupled gas/dust problem.
 * T_dust then relaxes toward T_eq exponentially over
 * τ_cool = C_grain × T_dust / P_emit (Hollenbach & McKee 1979).
 *
 * The gas is deliberately not heated or cooled here (no back-reaction).
 * Updating the nearest gas particle once per dust superparticle made the result
 * depend on particle order and domain decomposition, and a per-call limiter did
 * not bound the total drain when many grains picked the same gas particle.  A
 * future back-reaction must add up the exchanges per gas particle and apply one
 * conservative, bounded update after the dust loop.
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
  // With dust cooling off, only keep T_dust at or above the CMB temperature.
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
  // Carbon-rich grains: lower β (1.5 for pure carbon) and higher Q_abs (Draine 2003).
  // Q_abs grows linearly with grain size.
  double beta_eff = beta - 0.5 * CF;
  double Q_eff    = Q_ref * (1.0 + CF) * (a_cm / a_ref_cm);
  double T_CMB  = 2.7 / All.Time;
  double T_dust = Sp->DustP[dust_idx].DustTemperature;
  if(T_dust <= 0.0 || !isfinite(T_dust)) T_dust = T_CMB;
  double n_H         = dust_gas_nH_cgs(Sp, gas_idx);
  // ── Emission coefficient C_emit (P_emit = C_emit × T^(4+β)) ──────────────
  double C_emit = 4.0 * M_PI * a_cm * a_cm * Q_eff * sigma_SB / pow(T_ref, beta_eff);
  // ── Heating: CMB ─────────────────────────────────────────────────────────
  double P_CMB = C_emit * pow(T_CMB, 4.0 + beta_eff);
  // ── Heating: ISRF ─────────────────────────────────────────────────────────
  // T_ISRF ~ 17 K at z=0 (Reach+1995, Mathis+1983).  It rises weakly with
  // redshift, as (1+z)^0.25, because more young stars light the background field.
  double z_now  = 1.0 / All.Time - 1.0;
  double T_ISRF = 17.0 * pow(1.0 + z_now, 0.25);
  double P_ISRF = C_emit * pow(T_ISRF, 4.0 + beta_eff);
  // ── Heating/cooling: gas-grain collisional coupling ───────────────────────
  // P_coll = 2 n_H k_B (T_gas − T_dust) α_T v_th π a²
  // Positive when the gas is hotter than the dust (heats the grain), negative
  // when colder.  v_th = mean thermal speed of hydrogen.
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

  // Mass of one real grain in this superparticle; with c_v below it gives the
  // grain's heat capacity.  Used only for the dust relaxation; the gas is not
  // changed in this routine (see the function comment above).
  double m_grain_cgs = (4.0/3.0) * M_PI * a_cm*a_cm*a_cm * rho_grain;
  // ── Solve for the balance temperature ────────────────────────────────────
  // C_emit × T_eq^(4+β) = P_CMB + P_ISRF + P_gas
  double P_total = std::max(P_CMB + P_ISRF + P_gas, P_CMB);  // never below the CMB floor
  double T_eq    = pow(P_total / C_emit, 1.0 / (4.0 + beta_eff));
  // ── Relax toward T_eq over the cooling timescale τ_cool ──────────────────
  // Jumping straight to T_eq can make T_dust swing back and forth when it is
  // far from T_eq; the exponential step does not.
  const double c_v = 7e6;  // specific heat, erg/g/K (Draine & Li 2001, valid T > 20 K)
  double C_grain        = m_grain_cgs * c_v;
  // (T_dust + 1 keeps P_emit above zero.)
  double P_emit_current = C_emit * pow(T_dust + 1.0, 4.0 + beta_eff);
  double tau_cool       = (P_emit_current > 0.0)
                          ? (C_grain * T_dust / P_emit_current) : 1e10;
  double relax = 1.0 - exp(-dt_cgs / tau_cool);
  double T_new = T_dust + relax * (T_eq - T_dust);
  // ── Safety bounds: CMB floor; sublimation ceiling 1500 K (silicate) to
  //    2000 K (carbon), above which the grain would evaporate ────────────────
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
 * Run every dust process on every live dust particle.
 *
 * The run loop may call this less often than once per global synchronization
 * point.  dt is the sum of the exact physical intervals since the previous
 * call, in internal code-time units.  Never rebuild it as "current interval ×
 * nominal cadence": adaptive cosmological intervals are not equal.  Pending time
 * is always flushed before snapshots and at z=0.
 *
 * ORDER per particle:
 *   1. Dust temperature (if DustEnableCooling or DustEnableDrag)
 *   2. Gas drag + sputtering (dust_gas_interaction)
 *   3. Radiation pressure from young stars (if DustEnableRadiationPressure)
 *   Steps 4-6 run only if the grain is within its local dust-gas association
 *   radius of the gas particle (at least 2 kpc):
 *   4. Grain growth: gas-phase metals sticking to the grain
 *   5. Coagulation: grains sticking together in dense cold gas
 *   6. Shattering: grain-grain collisions breaking grains in turbulent diffuse gas
 */
// ═══════════════════════════════════════════════════════════════════════════════
// Pairing each grain with its nearest gas, the same on any number of tasks
// ═══════════════════════════════════════════════════════════════════════════════
//
// Each grain interacts with its nearest gas particle within
// DUST_GAS_SEARCH_RADIUS.  Near a domain boundary that gas may sit on another
// MPI task, so the search has to look across tasks:
//
//   1. each task finds the nearest gas it holds itself for each of its grains;
//   2. if a grain's search sphere reaches grid cells where other tasks hold gas,
//      the grain asks those tasks for anything closer ("occupancy routing", the
//      same scheme the D/G code uses; D/G = dust-to-gas mass ratio);
//   3. if the nearest gas is on another task, the grain is copied there for this
//      update (a "visiting" grain) and then sent home, so every dust routine
//      still sees its gas as a local particle.
//
// Equal distances are settled by the lower gas ID, so the pairing does not
// depend on how particles are split over tasks.  Eligible gas is the same set
// the old gas_hash held: high-resolution gas inside the zoom bounding box.

static constexpr double DUST_GAS_SEARCH_RADIUS = 10.0;  // code length units
static constexpr double DUST_GAS_ROUTE_CELL    = 10.0;  // side of the occupancy grid cells
static constexpr long long DUST_SELFCHECK_MAX_GRAINS = 200000;  // skip the all-task self-check above this

struct dust_gas_link
{
  int gas;            // gas index on the task holding it, or -1 if none
  int task;           // task holding the nearest gas, or -1 if none
  double r2;          // squared distance to that gas
  std::uint64_t id;   // nearest gas ID
  double hsml;        // its smoothing length
};

struct dust_gas_query
{
  MyIntPosType pos[3];
  double r2_max;
  int slot;
};

struct dust_gas_reply
{
  int slot, index;
  double r2, hsml;
  std::uint64_t id;
};

// A grain is sent between tasks as raw bytes, the way Gadget's own domain
// exchange moves particles.
struct dust_particle_bytes
{
  unsigned char P[sizeof(particle_data)];
  unsigned char DustP[sizeof(dust_data)];

  void store(simparticles *Sp, int i)
  {
    memcpy(P, static_cast<const void *>(&Sp->P[i]), sizeof(particle_data));
    memcpy(DustP, static_cast<const void *>(&Sp->DustP[i]), sizeof(dust_data));
  }
  void load(simparticles *Sp, int i) const
  {
    memcpy(static_cast<void *>(&Sp->P[i]), P, sizeof(particle_data));
    memcpy(static_cast<void *>(&Sp->DustP[i]), DustP, sizeof(dust_data));
  }
};

struct dust_visitor
{
  dust_particle_bytes particle;
  double local_DTG;
  int home_slot, gas_index;
  std::uint64_t gas_id;
};

struct dust_visitor_return
{
  dust_particle_bytes particle;
  int home_slot;
};

static bool dust_gas_eligible(simparticles *Sp, int j)
{
  if(Sp->P[j].getType() != 0) return false;
  const double m = Sp->P[j].getMass();
  if(!(m > 0.0) || m > gas_hash.zoom_mass_threshold) return false;
  double pos[3];
  Sp->intpos_to_pos(Sp->P[j].IntPos, pos);
  for(int d = 0; d < 3; d++)
    if(!std::isfinite(pos[d]) || pos[d] < gas_hash.bbox_min[d] || pos[d] > gas_hash.bbox_max[d]) return false;
  return true;
}

static inline bool dust_gas_better(double r2, std::uint64_t id, const dust_gas_link &cur)
{
  return cur.task < 0 || r2 < cur.r2 || (r2 == cur.r2 && id < cur.id);
}

// Collective: all tasks must call this.  Sends each task its queries and keeps
// any remote gas that is nearer than the current link (equal distance: lower ID wins).
static void resolve_remote_gas(simparticles *Sp, const cg_dist::LocalTree &gas_tree,
                               const std::vector<std::vector<dust_gas_query>> &queries,
                               std::vector<dust_gas_link> &links, MPI_Comm Communicator)
{
  int n_task;
  MPI_Comm_size(Communicator, &n_task);
  std::vector<int> source;
  const std::vector<dust_gas_query> got = cg_dist::exchange(queries, Communicator, &source);
  std::vector<std::vector<dust_gas_reply>> replies(n_task);
  for(size_t k = 0; k < got.size(); k++) {
    int j;
    double r2;
    if(gas_tree.nearest(Sp, got[k].pos, j, r2) && r2 <= got[k].r2_max)
      replies[source[k]].push_back(dust_gas_reply{got[k].slot, j, r2, (double)Sp->SphP[j].Hsml,
                                                  (std::uint64_t)Sp->P[j].ID.get()});
  }
  std::vector<int> reply_task;
  const std::vector<dust_gas_reply> back = cg_dist::exchange(replies, Communicator, &reply_task);
  for(size_t k = 0; k < back.size(); k++) {
    dust_gas_link &l = links[back[k].slot];
    // For remote gas, l.gas is its index on the task that holds it.
    if(dust_gas_better(back[k].r2, back[k].id, l))
      l = dust_gas_link{back[k].index, reply_task[k], back[k].r2, back[k].id, back[k].hsml};
  }
}

// Collective: all tasks must call this.  For every live local grain i, links[i]
// gets the nearest eligible gas on any task.
static void associate_dust_with_gas(simparticles *Sp, const cg_dist::LocalTree &gas_tree,
                                    std::vector<dust_gas_link> &links, MPI_Comm Communicator)
{
  int n_task, this_task;
  MPI_Comm_size(Communicator, &n_task);
  MPI_Comm_rank(Communicator, &this_task);
  const double r_max2 = DUST_GAS_SEARCH_RADIUS * DUST_GAS_SEARCH_RADIUS;

  links.assign((size_t)Sp->NumPart, dust_gas_link{-1, -1, 0.0, 0, 0.0});
  for(int i = 0; i < Sp->NumPart; i++) {
    if(Sp->P[i].getType() != DUST_PARTICLE_TYPE || Sp->P[i].getMass() <= DUST_MASS_TO_DESTROY) continue;
    int j;
    double r2;
    if(gas_tree.nearest(Sp, Sp->P[i].IntPos, j, r2) && r2 <= r_max2)
      links[i] = dust_gas_link{j, this_task, r2, (std::uint64_t)Sp->P[j].ID.get(), Sp->SphP[j].Hsml};
  }
  if(n_task == 1) return;
  const std::vector<dust_gas_link> local_links = links;

  // Build a shared directory of which tasks hold eligible gas in which grid cell.
  const long long n_c = std::max(1LL, (long long)std::floor(All.BoxSize / DUST_GAS_ROUTE_CELL));
  const double cc = All.BoxSize / n_c;
  auto ckey = [n_c](long long ix, long long iy, long long iz) {
    ix = ((ix % n_c) + n_c) % n_c;
    iy = ((iy % n_c) + n_c) % n_c;
    iz = ((iz % n_c) + n_c) % n_c;
    return ix + n_c * (iy + n_c * iz);
  };
  std::unordered_set<long long> local_cells;
  for(int j = 0; j < Sp->NumGas; j++)
    if(dust_gas_eligible(Sp, j)) {
      double pos[3];
      Sp->intpos_to_pos(Sp->P[j].IntPos, pos);
      local_cells.insert(ckey((long long)std::floor(pos[0] / cc), (long long)std::floor(pos[1] / cc),
                              (long long)std::floor(pos[2] / cc)));
    }
  cg_dist::CellDirectory directory;
  directory.build(std::vector<long long>(local_cells.begin(), local_cells.end()), Communicator);

  // Grid cells that grain i's search sphere (radius^2 = r2q) overlaps.
  auto touched_cells = [&](int i, double r2q, std::vector<long long> &keys) {
    keys.clear();
    const double rq = std::sqrt(r2q);
    double c[3];
    Sp->intpos_to_pos(Sp->P[i].IntPos, c);
    long long lo[3], hi[3];
    for(int d = 0; d < 3; d++) {
      lo[d] = (long long)std::floor((c[d] - rq) / cc);
      hi[d] = (long long)std::floor((c[d] + rq) / cc);
    }
    for(long long ix = lo[0]; ix <= hi[0]; ix++)
      for(long long iy = lo[1]; iy <= hi[1]; iy++)
        for(long long iz = lo[2]; iz <= hi[2]; iz++) {
          const long long ic[3] = {ix, iy, iz};
          double near2 = 0.0;
          for(int d = 0; d < 3; d++) {
            const double b0 = ic[d] * cc, b1 = b0 + cc;
            const double dn = c[d] - std::max(b0, std::min(b1, c[d]));
            near2 += dn * dn;
          }
          if(near2 <= r2q) keys.push_back(ckey(ix, iy, iz));
        }
  };
  auto live_grain = [Sp](int i) {
    return Sp->P[i].getType() == DUST_PARTICLE_TYPE && Sp->P[i].getMass() > DUST_MASS_TO_DESTROY;
  };

  // Ask the other tasks that hold gas inside each grain's search sphere.  If a
  // local match was found, the sphere shrinks to that distance.
  std::unordered_set<long long> wanted;
  std::vector<long long> keys;
  for(int i = 0; i < Sp->NumPart; i++)
    if(live_grain(i)) {
      touched_cells(i, links[i].task >= 0 ? links[i].r2 : r_max2, keys);
      wanted.insert(keys.begin(), keys.end());
    }
  const std::unordered_map<long long, std::vector<int>> owners =
      directory.lookup(std::vector<long long>(wanted.begin(), wanted.end()));

  std::vector<std::vector<dust_gas_query>> queries(n_task);
  std::vector<int> last_query(n_task, -1);
  for(int i = 0; i < Sp->NumPart; i++) {
    if(!live_grain(i)) continue;
    const double r2q = links[i].task >= 0 ? links[i].r2 : r_max2;
    touched_cells(i, r2q, keys);
    for(long long k : keys) {
      const auto it = owners.find(k);
      if(it == owners.end()) continue;
      for(int task : it->second)
        if(task != this_task && last_query[task] != i) {
          last_query[task] = i;
          dust_gas_query q;
          for(int d = 0; d < 3; d++) q.pos[d] = Sp->P[i].IntPos[d];
          q.r2_max = r2q;
          q.slot = i;
          queries[task].push_back(q);
        }
    }
  }
  resolve_remote_gas(Sp, gas_tree, queries, links, Communicator);

  // Self-check on the first 3 calls: asking every task must give the same
  // nearest gas as asking only the tasks the directory names.
  // Asking every task costs (live grains) x (tasks) queries.  After a late
  // restart of a 2048^3 run that was ~1.5e9 queries per call (~30 s each), and
  // a restart over TCP crashed inside one.  So the check is skipped when there
  // are more than DUST_SELFCHECK_MAX_GRAINS grains in total; a fresh start,
  // with few grains, still runs it.
  static int verify_remaining = 3;
  if(verify_remaining > 0) {
    long long n_live = 0, n_live_global = 0;
    for(int i = 0; i < Sp->NumPart; i++)
      if(live_grain(i)) n_live++;
    MPI_Allreduce(&n_live, &n_live_global, 1, MPI_LONG_LONG, MPI_SUM, Communicator);
    if(n_live_global > DUST_SELFCHECK_MAX_GRAINS) {
      if(this_task == 0)
        printf("[DUST_ASSOC_VERIFY] skipped: %lld grains > %lld\n", n_live_global, (long long)DUST_SELFCHECK_MAX_GRAINS);
      verify_remaining = 0;
    }
  }
  if(verify_remaining > 0) {
    verify_remaining--;
    std::vector<dust_gas_link> all_links = local_links;
    std::vector<std::vector<dust_gas_query>> all_queries(n_task);
    for(int i = 0; i < Sp->NumPart; i++) {
      if(!live_grain(i)) continue;
      dust_gas_query q;
      for(int d = 0; d < 3; d++) q.pos[d] = Sp->P[i].IntPos[d];
      q.r2_max = local_links[i].task >= 0 ? local_links[i].r2 : r_max2;
      q.slot = i;
      for(int task = 0; task < n_task; task++)
        if(task != this_task) all_queries[task].push_back(q);
    }
    resolve_remote_gas(Sp, gas_tree, all_queries, all_links, Communicator);
    long long local[2] = {0, 0}, total[2];
    for(int i = 0; i < Sp->NumPart; i++) {
      if(!live_grain(i)) continue;
      local[1]++;
      if(all_links[i].task != links[i].task || all_links[i].id != links[i].id) local[0]++;
    }
    MPI_Reduce(local, total, 2, MPI_LONG_LONG, MPI_SUM, 0, Communicator);
    if(this_task == 0)
      printf("[DUST_ASSOC_VERIFY] directory routing vs asking every task: %lld of %lld grains differ\n",
             total[0], total[1]);
  }
}

void update_dust_dynamics(simparticles *Sp, double dt, MPI_Comm Communicator)
{
  // The timer starts before the collective "any dust?" check below, so time
  // spent waiting for other tasks there is counted too.
  static double total_time_in_dust = 0.0;
  static int    dust_call_count    = 0;
  double t_start = MPI_Wtime();

  // ── Print the process switches once (shows the parameter file was read) ───
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
  // ── Open the per-particle dust log once ───────────────────────────────────
  static bool log_opened = false;
  if(!log_opened) {
    open_dust_particle_log(Communicator);
    log_opened = true;
  }

  // ── Collective check: is there live dust on any task? ─────────────────────
  // GlobalDustCount is only a diagnostic counter.  It is not saved in restart
  // files and can go stale after compaction or particle exchange, so trusting
  // it could make a restarted run skip all dust physics without warning.  We
  // recount the grains here instead.
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
  // ── Build the gas spatial hash if it is missing ──────────────────────────
  // Task 0 decides and broadcasts, so all tasks enter the collective rebuild together.
  int need_hash_rebuild = 0;
  if(All.ThisTask == 0) need_hash_rebuild = !gas_hash.is_built;
  MPI_Bcast(&need_hash_rebuild, 1, MPI_INT, 0, Communicator);
  if(need_hash_rebuild) {
    if(All.ThisTask == 0)
      DUST_PRINT("WARNING: Hash not built, building now for dust operations\n");
    rebuild_feedback_spatial_hash(Sp, 0.1, Communicator);
  }
  // ── Print the hash setup once ─────────────────────────────────────────────
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
  // Self-check on the first 20 calls: dust updates move mass between gas and
  // grains but must not create or lose any, also for grains processed on
  // another task.  Collective: total_baryon_mass sums over all tasks.
  static int mass_check_remaining = 20;
  const bool check_mass = mass_check_remaining > 0;
  const double mass_before = check_mass ? cg_dist::total_baryon_mass(Sp, Communicator) : 0.0;

  // Move gas, grains, and stars to the current time before any distances are measured.
  Sp->drift_all_particles();
  refresh_young_star_list(Sp, Communicator);

  // Wall time per phase on this task; the timing summary reports the maximum over tasks.
  static double t_phase[4] = {0, 0, 0, 0};  // association, D/G, local loop, visiting grains
  double t_mark = MPI_Wtime();
  cg_dist::LocalTree assoc_gas_tree;
  assoc_gas_tree.build(Sp, 0, Sp->NumGas, [Sp](int j) { return dust_gas_eligible(Sp, j); });
  std::vector<dust_gas_link> links;
  associate_dust_with_gas(Sp, assoc_gas_tree, links, Communicator);
  t_phase[0] += MPI_Wtime() - t_mark;

  // Growth, coagulation, and shattering only run when the grain is truly close
  // to its gas.  A fixed 2 kpc cut depends strongly on resolution: at modest
  // resolution a grain can sit well inside the nearest gas particle's SPH
  // kernel and still be more than 2 kpc from its center.  So "close" means
  // within the gas smoothing length, with 2 kpc as a floor and the 10 kpc
  // search radius as a cap, so thin CGM gas (circumgalactic medium) cannot
  // couple to grains arbitrarily far away.
  // coagulation_radius_by_particle[i] = that radius, or 0 if the grain is too far.
  std::vector<double> coagulation_radius_by_particle((size_t)Sp->NumPart, 0.0);
  for(int i = 0; i < Sp->NumPart; i++) {
    if(Sp->P[i].getType() != DUST_PARTICLE_TYPE ||
       Sp->P[i].getMass() <= DUST_MASS_TO_DESTROY)
      continue;
    HashSearches++;
    if(links[i].task < 0) {
      HashSearchesFailed++;
      continue;
    }
    const double association_radius =
        std::max(2.0, std::min(links[i].hsml, DUST_GAS_SEARCH_RADIUS));
    if(std::sqrt(links[i].r2) <= association_radius)
      coagulation_radius_by_particle[i] = association_radius;
  }

  // ── Local D/G, cached per particle and refreshed in a staggered way ───────
  // D/G (dust-to-gas mass ratio, also written DTG) changes on growth and
  // destruction timescales (tens to hundreds of Myr), much slower than the dust
  // update cadence.  So each grain's local D/G is recomputed only every
  // DTG_REFRESH_INTERVAL dust calls and reused in between.
  //
  // The cache lives in DustP[] (LocalDTG, DTGAge) so it moves with the particle
  // between tasks.  An earlier version kept a per-task map keyed by ID; it
  // missed ~95% of the time, because the domain decomposition (redone almost
  // every step) moves most dust to another task between dust calls.
  //
  // DTGAge counts dust calls since the last recompute.  A grain reuses its
  // cache while DTGAge < DTG_REFRESH_INTERVAL - 1, so it refreshes exactly once
  // every DTG_REFRESH_INTERVAL calls.  Grains without a valid cache (new dust,
  // starts from a snapshot, restart files from before the cache existed, failed
  // estimates) refresh at once and then use their array index to set where they
  // are in the cycle.  That way a big batch (e.g. every grain after a restart)
  // does not come due on the same call again.  Grains with no close gas get
  // their cache cleared, since a value from an earlier pairing may be stale.
  static constexpr int DTG_REFRESH_INTERVAL = 10;

  // Running histogram of |DTG_new - DTG_cached| / DTG_cached at scheduled
  // refreshes, i.e. how far D/G drifted over DTG_REFRESH_INTERVAL calls.
  // It tells us whether the interval could safely be made longer.
  static constexpr int DTG_STALE_NBINS = 6;
  static const double dtg_stale_edges[DTG_STALE_NBINS - 1] = {0.01, 0.03, 0.1, 0.3, 1.0};
  static long long dtg_stale_hist[DTG_STALE_NBINS] = {0, 0, 0, 0, 0, 0};
  static double dtg_stale_sum_rel = 0.0;

  t_mark = MPI_Wtime();
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

    // Collective: every task must call the D/G estimate, even with nothing to refresh.
    std::vector<double> fresh_DTG;
    TIMER_START(CPU_DUST_DTG);
    // Two ways to decide which tasks a D/G query goes to.  Occupancy routing
    // pays a fixed cost per task to share the table of occupied cells;
    // bounding-box routing gets more expensive as the number of tasks grows.
    // On 24 ranks bounding-box routing was faster (0.08 vs 0.13 s per call), so
    // occupancy routing is used only from DTG_OCCUPANCY_MIN_TASKS ranks up.
    // The environment variable CG_DTG_ROUTING=bbox|occupancy forces either one.
    // Both give bit-identical results.
    static int use_occupancy = -1;
    if(use_occupancy < 0) {
      int n_task = 1;
      MPI_Comm_size(Communicator, &n_task);
      const char *env = getenv("CG_DTG_ROUTING");
      if(env && strcmp(env, "occupancy") == 0)      use_occupancy = 1;
      else if(env && strcmp(env, "bbox") == 0)      use_occupancy = 0;
      else use_occupancy = n_task >= DTG_OCCUPANCY_MIN_TASKS ? 1 : 0;
      if(All.ThisTask == 0)
        printf("[DTG_ROUTE] %s routing of D/G queries on %d ranks\n",
               use_occupancy ? "occupancy" : "bounding-box", n_task);
    }
    estimate_local_dust_to_gas_ratio(Sp, refresh_radius, Communicator, fresh_DTG,
                                     use_occupancy == 1, &dtg_queries_sent);

    // Self-check on the first 3 calls: run the other routing too; the two must
    // agree bit for bit, since occupancy routing only skips tasks that would add 0.
    static int route_verify_remaining = 3;
    if(route_verify_remaining > 0) {
      route_verify_remaining--;
      std::vector<double> bbox_DTG;
      long long bbox_sent = 0;
      estimate_local_dust_to_gas_ratio(Sp, refresh_radius, Communicator, bbox_DTG,
                                       use_occupancy != 1, &bbox_sent);
      long long counts[3] = {0, dtg_queries_sent, bbox_sent};
      for(int i = 0; i < Sp->NumPart; i++)
        if(refresh_radius[i] > 0.0 &&
           memcmp(&fresh_DTG[i], &bbox_DTG[i], sizeof(double)) != 0)
          counts[0]++;
      long long sums[3];
      MPI_Reduce(counts, sums, 3, MPI_LONG_LONG, MPI_SUM, 0, Communicator);
      if(All.ThisTask == 0)
        printf("[DTG_ROUTE_VERIFY] %s routing vs the other: %lld differing D/G values; "
               "queries sent %lld vs %lld\n",
               use_occupancy ? "occupancy" : "bounding-box", sums[0], sums[1], sums[2]);
    }
    TIMER_STOP(CPU_DUST_DTG);

    for(int i = 0; i < Sp->NumPart; i++) {
      if(!(refresh_radius[i] > 0.0)) continue;
      local_DTG[i] = fresh_DTG[i];
      dust_data &dp = Sp->DustP[i];
      // Staleness: how far D/G drifted over one full refresh interval.
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
  t_phase[1] += MPI_Wtime() - t_mark;

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
  // process_dust runs every dust process for grain i against its gas particle.
  // Growth, coagulation, and shattering run only if the grain lies within
  // local_dust_gas_radius (0 means "not close enough").
  auto process_dust = [&](int i, int nearest_gas, double dist_kpc, double local_dust_gas_radius,
                          double dtg) {
    // One gas temperature serves every process below.  Dust routines never
    // change the gas entropy, density, or Ne (electron abundance), so
    // computing it once gives exactly the same answer.
    const double T_gas = get_temperature_from_entropy(Sp, nearest_gas);
    // Grain temperature: needed by both drag and cooling.
    if(All.DustEnableCooling || All.DustEnableDrag)
      update_dust_temperature(Sp, i, nearest_gas, dt, T_gas);

    dust_gas_interaction(Sp, i, nearest_gas, dt, T_gas);
    dust_radiation_pressure(Sp, i, nearest_gas, dt);

    if(local_dust_gas_radius > 0.0 && dist_kpc <= local_dust_gas_radius) {
      if(All.DustEnableGrowth)
        dust_grain_growth_subgrid(Sp, i, nearest_gas, dt, T_gas);
      if(All.DustEnableCoagulation)
        dust_grain_coagulation(Sp, i, nearest_gas, dt, dtg, T_gas);
      if(All.DustEnableShattering)
        dust_grain_shattering(Sp, i, nearest_gas, dt, dtg, T_gas);
    }
  };

  t_mark = MPI_Wtime();
  int n_task_main = 1;
  MPI_Comm_size(Communicator, &n_task_main);
  std::vector<std::vector<dust_visitor>> visitors((size_t)n_task_main);
  for(int i = 0; i < Sp->NumPart; i++) {
    if(Sp->P[i].getType() != DUST_PARTICLE_TYPE) continue;
    if(Sp->P[i].getMass() <= DUST_MASS_TO_DESTROY) continue;
    const dust_gas_link &l = links[i];
    if(l.task < 0) continue;
    if(l.task != All.ThisTask) {
      // Nearest gas is on another task: send a copy of the grain there.
      dust_visitor v;
      v.particle.store(Sp, i);
      v.local_DTG = local_DTG[i];
      v.home_slot = i;
      v.gas_index = l.gas;
      v.gas_id = l.id;
      visitors[l.task].push_back(v);
      continue;
    }
    process_dust(i, l.gas, std::sqrt(l.r2), coagulation_radius_by_particle[i], local_DTG[i]);
  }

  t_phase[2] += MPI_Wtime() - t_mark;
  t_mark = MPI_Wtime();
  // Visiting grains are placed in the free slots after NumPart while they are
  // updated, so the dust routines treat them like local particles.  Then they
  // are sent home and NumPart is set back.  Collective: every task must take
  // part in both exchanges, even with no visitors.
  std::vector<int> visitor_home;
  const std::vector<dust_visitor> arrived = cg_dist::exchange(visitors, Communicator, &visitor_home);
  long long n_visiting = (long long)arrived.size();
  std::vector<std::vector<dust_visitor_return>> returns((size_t)n_task_main);
  if(!arrived.empty()) {
    const int first = Sp->NumPart;
    if(first + (long long)arrived.size() > Sp->MaxPart)
      Terminate("dust: no room for %d visiting grains (NumPart=%d MaxPart=%d)",
                (int)arrived.size(), Sp->NumPart, Sp->MaxPart);
    for(size_t k = 0; k < arrived.size(); k++) {
      arrived[k].particle.load(Sp, first + (int)k);
    }
    Sp->NumPart += (int)arrived.size();
    for(size_t k = 0; k < arrived.size(); k++) {
      const dust_visitor &v = arrived[k];
      const int i = first + (int)k;
      const int g = v.gas_index;
      if(g < 0 || g >= Sp->NumGas || Sp->P[g].getType() != 0 || (std::uint64_t)Sp->P[g].ID.get() != v.gas_id)
        Terminate("dust: visiting grain's gas ID %llu is not at index %d", (unsigned long long)v.gas_id, g);
      double d[3];
      Sp->nearest_image_intpos_to_pos(Sp->P[i].IntPos, Sp->P[g].IntPos, d);
      const double dist = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
      const double association_radius =
          std::max(2.0, std::min((double)Sp->SphP[g].Hsml, DUST_GAS_SEARCH_RADIUS));
      process_dust(i, g, dist, dist <= association_radius ? association_radius : 0.0, v.local_DTG);
      dust_visitor_return r;
      r.particle.store(Sp, i);
      r.home_slot = v.home_slot;
      returns[visitor_home[k]].push_back(r);
    }
    Sp->NumPart = first;
  }
  const std::vector<dust_visitor_return> home = cg_dist::exchange(returns, Communicator);
  for(const dust_visitor_return &r : home) {
    if(r.home_slot < 0 || r.home_slot >= Sp->NumPart)
      Terminate("dust: returning grain for invalid slot %d", r.home_slot);
    r.particle.load(Sp, r.home_slot);
  }
  t_phase[3] += MPI_Wtime() - t_mark;
  if(check_mass) {
    mass_check_remaining--;
    const double mass_after = cg_dist::total_baryon_mass(Sp, Communicator);
    if(All.ThisTask == 0)
      printf("[DUST_MASS_CHECK] step %d: gas+star+dust mass %.17g -> %.17g (rel. change %.3g)\n",
             All.NumCurrentTiStep, mass_before, mass_after, (mass_after - mass_before) / mass_before);
  }
  {
    long long n_visiting_global = 0;
    MPI_Reduce(&n_visiting, &n_visiting_global, 1, MPI_LONG_LONG, MPI_SUM, 0, Communicator);
    if(All.ThisTask == 0 && dust_call_count % 10 == 0)
      printf("[DUST_ASSOC|Step=%d] %lld of %lld grains processed on the task of their nearest gas\n",
             All.NumCurrentTiStep, n_visiting_global, global_count);
  }
  // ── Velocity watchdog ─────────────────────────────────────────────────────
  // Stars and grains should not move faster than 3000 km/s (physical peculiar
  // speed, i.e. Vel * 1/a).  Each such star is printed once; grains are only
  // counted.  Gas can be that fast in supernova-driven winds, so fast gas is
  // counted and the first ones seen are printed with temperature and density.
  {
    static constexpr double WATCH_KMS = 3000.0;
    static std::unordered_set<long long> reported_stars, reported_gas;
    long long n_fast[3] = {0, 0, 0}, n_fast_sum[3] = {0, 0, 0};
    for(int i = 0; i < Sp->NumPart; i++) {
      const int type = Sp->P[i].getType();
      if(type == 0) {
        double v2 = 0.0;
        for(int k = 0; k < 3; k++) v2 += (double)Sp->P[i].Vel[k] * Sp->P[i].Vel[k];
        const double v_kms = sqrt(v2) * All.cf_ainv;
        if(!(v_kms > WATCH_KMS)) continue;
        n_fast[2]++;
        const long long id = (long long)Sp->P[i].ID.get();
        if(reported_gas.size() < 2000 && reported_gas.insert(id).second) {
          const double u    = Sp->get_utherm_from_entropy(i);
          const double mu   = 4.0 / (1.0 + 3.0 * HYDROGEN_MASSFRAC + 4.0 * HYDROGEN_MASSFRAC * Sp->SphP[i].Ne);
          const double T    = GAMMA_MINUS1 * u * All.UnitVelocity_in_cm_per_s * All.UnitVelocity_in_cm_per_s * mu * PROTONMASS / BOLTZMANN;
          const double nH   = HYDROGEN_MASSFRAC * Sp->SphP[i].Density * All.cf_a3inv * All.UnitDensity_in_cgs * All.HubbleParam * All.HubbleParam / PROTONMASS;
          double pos[3];
          Sp->intpos_to_pos(Sp->P[i].IntPos, pos);
          printf("[FAST_GAS|T=%d|a=%.5f] gas ID %lld moves at %.0f km/s; T=%.3e K nH=%.3e cm^-3 Hsml=%.3e SFR=%.3e mass=%.3e pos=(%.1f|%.1f|%.1f)\n",
                 All.ThisTask, (double)All.Time, id, v_kms, T, nH, (double)Sp->SphP[i].Hsml, (double)Sp->SphP[i].Sfr,
                 Sp->P[i].getMass(), pos[0], pos[1], pos[2]);
        }
        continue;
      }
      if(type != 4 && type != DUST_PARTICLE_TYPE) continue;
      double v2 = 0.0;
      for(int k = 0; k < 3; k++) v2 += (double)Sp->P[i].Vel[k] * Sp->P[i].Vel[k];
      const double v_kms = sqrt(v2) * All.cf_ainv;
      if(!(v_kms > WATCH_KMS)) continue;
      if(type == 4) {
        n_fast[0]++;
        const long long id = (long long)Sp->P[i].ID.get();
        if(reported_stars.size() < 10000 && reported_stars.insert(id).second)
          printf("[FAST_STAR|T=%d|a=%.5f] star ID %lld moves at %.0f km/s; age %.1f Myr mass %.3e\n",
                 All.ThisTask, (double)All.Time, id, v_kms,
                 get_stellar_age_Myr(Sp->P[i].StellarAge, All.Time), Sp->P[i].getMass());
      } else
        n_fast[1]++;
    }
    MPI_Reduce(n_fast, n_fast_sum, 3, MPI_LONG_LONG, MPI_SUM, 0, Communicator);
    if(All.ThisTask == 0 && dust_call_count % 10 == 0)
      printf("[FAST_PARTICLES|Step=%d] above %.0f km/s: %lld stars, %lld of %lld grains, %lld gas\n",
             All.NumCurrentTiStep, WATCH_KMS, n_fast_sum[0], n_fast_sum[1], global_count, n_fast_sum[2]);
  }
  // ── Periodic diagnostics (every 500 steps) ───────────────────────────────
  // The gas budget and sample grains below cover task 0's own particles only.
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
  if(dust_call_count % 100 == 0) {
    double t_max[4];
    MPI_Reduce(t_phase, t_max, 4, MPI_DOUBLE, MPI_MAX, 0, Communicator);
    if(All.ThisTask == 0) {
      printf("[DUST_TIMING] Called %d times, avg %.3f sec/call, total %.1f sec\n",
             dust_call_count, total_time_in_dust / dust_call_count, total_time_in_dust);
      printf("[DUST_PHASE_TIMING] cumulative max over tasks [sec]: association=%.1f D/G=%.1f "
             "local=%.1f visiting=%.1f\n", t_max[0], t_max[1], t_max[2], t_max[3]);
    }
  }
}
// ═══════════════════════════════════════════════════════════════════════════════
// destroy_dust_particles
// ═══════════════════════════════════════════════════════════════════════════════
/**
 * Remove dead grains and close up the gaps in the particle array.
 *
 * A destroyed grain is marked with type 3, mass DUST_MASS_TO_DESTROY, and ID 0.
 * A particle is removed if it has type DUST_PARTICLE_TYPE or 3 AND
 * (mass ≤ DUST_MASS_TO_DESTROY or ID = 0).
 *
 * Gadget expects gas particles (type 0) at indices [0, NumGas) and all other
 * particles (stars, dust, ...) after them; the two regions are compacted
 * separately to keep it that way.  DustP is zeroed for every non-dust slot.
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
  // Compact the gas region [0, NumGas).
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
  // Compact the non-gas region [NumGas, NumPart) onto the end of the gas.
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
 * looked up in the pre-built gas_hash.  This searches only gas on this task;
 * the main dust update uses associate_dust_with_gas() above instead.
 *
 * Each kind of failure is counted separately for diagnostics:
 *   fail_no_neighbor  — the hash found nothing within the radius
 *   fail_too_far      — the nearest particle found is beyond max_r_kpc
 *   fail_wrong_type   — a stale hash slot points to a particle that is no longer gas
 *
 * Returns -1 on failure.  If out_dist_kpc is not NULL it gets the distance,
 * or -1 on failure.
 */
int find_nearest_gas_particle(simparticles *Sp, int dust_idx,
                               double max_r_kpc, double *out_dist_kpc)
{
  if(out_dist_kpc) *out_dist_kpc = -1.0;
  if(Sp->NumGas == 0 || max_r_kpc <= 0) return -1;
  if(!gas_hash.is_built) return -1;
  HashSearches++;
  // Failure counters, summed over the whole run on this task.
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
  // Print the breakdown every 5 million failures (task 0's count only).
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
/** Return |v_dust − v_gas| in internal velocity units (no 1/a factor applied). */
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
// Coagulation counters, kept separately on each MPI task since the run started.
// dust_grain_coagulation() fills them; print_dust_statistics() adds them up over tasks.
static long long CoagCalls = 0;
static long long CoagFailedDensity = 0;
static long long CoagFailedTemperature = 0;
static long long CoagFailedSize = 0;
static long long CoagFailedDTG = 0;
static long long CoagGrown = 0;
static long long CoagLargeUpdates = 0;
static double CoagMaxFractionalIncrease = 0.0;

/**
 * Collect dust statistics from all tasks and print them on rank 0.
 *
 * Reports: number and mass of dust particles, mean grain radius and temperature,
 * a temperature histogram, how often the neighbour hash search succeeded,
 * how particles were destroyed (by physics, or by internal clean-up paths
 * that should never fire), growth, coagulation and shattering.
 *
 * Collective: all tasks must call this (it uses MPI_Reduce). Only rank 0 prints.
 */
void print_dust_statistics(simparticles *Sp, MPI_Comm Communicator)
{
  // ── Sums over this task's live dust particles ─────────────────────────────
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
  // ── Add everything up on rank 0 ───────────────────────────────────────────
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
  // Coagulation counters (summed) and the largest requested da/a (maximum).
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

  // Also collective, so it must come before the non-zero ranks return.
  print_coag_histogram(Communicator);
  if(All.ThisTask != 0) return;
  if(global_dust_count > 0) {
    global_avg_size /= global_dust_count;
    global_avg_temp /= global_dust_count;
  }
  // Code mass units -> solar masses (code masses are in UnitMass/h).
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
 * Print the mean and maximum speed of dust relative to its nearest gas particle,
 * using task 0's particles only.
 *
 * Why sample: each particle costs one hash search. With 16M dust particles on
 * ~240 tasks, task 0 holds ~67k of them, too many for a diagnostic. Looking at
 * every 100th particle index (1%) keeps the cost small and still gives a fair average.
 */
void analyze_dust_gas_coupling_local(simparticles *Sp)
{
  if(All.ThisTask != 0) return;
  double total_vel_diff = 0.0;
  double max_vel_diff   = 0.0;
  int    dust_count     = 0;
  for(int i = 0; i < Sp->NumPart; i++) {
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
 * Radius of a supernova (SN) blast wave time_myr after the explosion, in kpc.
 *
 * Uses the Sedov-Taylor solution for a point explosion of energy E (erg) in gas
 * of uniform mass density rho (g cm^-3):
 *   R(t) = ξ × [E t² / ρ]^(1/5)    (Sedov 1959, ξ = 1.033)
 */
double calculate_sn_shock_radius(double sn_energy_erg, double gas_density_cgs,
                                  double time_myr)
{
  // Known issue (kept for v1.2.0): 1.033 is the Sedov constant for gamma = 1.4 (air).
  // For monatomic gas (gamma = 5/3) it is about 1.15.  The shock model evaluates R and v
  // at a fixed time (0.3 Myr), where both scale with xi: about 12% larger and faster,
  // so about 1.4x the swept volume and about 1.4x (carbon) to 2x (silicate) the dust
  // destroyed per SN.  Shock destruction was calibrated with 1.033, so changing it
  // needs a recalibration.  Planned for the next version.
  const double xi     = 1.033;
  double time_sec     = time_myr * 1e6 * SEC_PER_YEAR;
  double radius_cm    = xi * pow(sn_energy_erg * time_sec * time_sec / gas_density_cgs, 0.2);
  return radius_cm / (1000.0 * PARSEC);  // cm → kpc
}
// ═══════════════════════════════════════════════════════════════════════════════
// get_shock_destruction_efficiency
// ═══════════════════════════════════════════════════════════════════════════════
/**
 * Fraction of grain mass a shock of speed v (km/s) destroys, for carbon and
 * silicate grains separately.
 *
 * Straight-line interpolation of Bocchio et al. (2014), Table 6. The two
 * materials are returned separately so each part of a mixed grain is eroded at
 * its own rate; that is how CarbonMassFraction changes in SN shocks.
 *
 * Below 50 km/s nothing is destroyed. At 50 km/s carbon loses 0.77 and
 * silicate 0.02. Carbon reaches 1 (complete destruction) at 175 km/s.
 * Silicate reaches 0.67 at 175 km/s and stays there, since the Bocchio et al.
 * table stops at 200 km/s.
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
  // Keep both in [0, 1].
  *eps_carb = std::max(0.0, std::min(1.0, *eps_carb));
  *eps_sil  = std::max(0.0, std::min(1.0, *eps_sil));
}
// ── Grain-size weight for shock damage ────────────────────────────────────
// Small grains have more surface per unit mass, so a shock destroys them more
// easily than large ones. This weight (radius in nm) decides how an event's
// destroyed mass is shared among the grains it hits.
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
 * Erode and destroy dust hit by this step's supernova (SN) blast waves.
 *
 * WHY A SUBGRID MODEL:
 *   At 2048³ the real Sedov-Taylor shock radius (~20-50 pc) is much smaller
 *   than the dust hash cell (~1-7 kpc), so a search at the real radius would
 *   almost never find a grain. Instead we work out how much dust mass the real
 *   shock SHOULD destroy and spread that loss over all dust in a wider sphere.
 *
 * STEPS (numbered as in the code below):
 *   1. From the local gas density, get the physical shock radius R_phys and
 *      the shock speed v_shock from Sedov-Taylor at t = 0.3 Myr.
 *   2. Search radius R_search = max(R_phys, 2 × dust hash cell size), capped at 3 kpc.
 *   3. Find all dust within R_search.
 *   4. Volume factor f_vol = (R_phys / R_search)³: the fraction of the searched
 *      dust that sits inside the real shock (assuming dust is spread evenly).
 *   5. Split the found dust mass into carbon (C) and silicate (sil) and set
 *      one mass budget per material, with ε from get_shock_destruction_efficiency():
 *        M_C,destroy   = M_C,local   × f_vol × ε_C(v_shock)
 *        M_sil,destroy = M_sil,local × f_vol × ε_sil(v_shock)
 *   6. Share each budget among the grains in proportion to that material's
 *      mass times shock_size_factor().
 *   7. Put each grain back together from what is left: new CarbonMassFraction,
 *      and new radius from a ∝ m^(1/3) (same number of grains per particle).
 *   8. Give the removed material back to the gas particle nearest the SN.
 *
 * WHY IT DOES NOT DEPEND ON RESOLUTION:
 *   Coarse runs find few grains but f_vol ~ 1 (cell about the shock size).
 *   Fine runs find many grains but f_vol << 1. Either way the destroyed mass
 *   comes out the same.
 *
 * IMPORTANT: v_shock always comes from R_phys, never from R_search. Using the
 * larger search radius would lower v by ~(R_search/R_phys)^(3/2) >> 1 and push
 * it below the destruction thresholds, so nothing would be destroyed.
 *
 * PARALLEL LAYOUT: every SN event is shared with all tasks, each task searches
 * its own dust, and the per-event sums are added over all tasks. Before this,
 * each SN only saw dust on its own task, so the result depended on where the
 * domain boundaries fell.
 *
 * Collective: all tasks must call this, even with no events of their own.
 * `mine` holds this task's events; dust at index >= first_new_dust was made in
 * this same feedback call and is left alone.
 */
void erode_dust_from_sn_shocks_batch(simparticles *Sp, const std::vector<DustShockEvent> &mine,
                                     int first_new_dust, MPI_Comm Communicator)
{
  // The flag is the same on every task, so it is the only safe early exit
  // before the collectives below.
  if(!All.DustEnableShockDestruction) return;

  std::vector<int> owner;
  const std::vector<DustShockEvent> all = cg_dist::allgather(mine, Communicator, &owner);
  const size_t n_events = all.size();
  if(n_events == 0) return;

  static long long sn_total_calls = 0, sn_found_dust = 0;

  // ── Steps 1-4: shock radius, speed and search volume for each event ──────
  // Every task computes the same numbers from the shared event record.
  struct Shock { double r_phys, r_search, v, f_vol, eps_carb, eps_sil; };
  std::vector<Shock> shock(n_events);
  for(size_t e = 0; e < n_events; e++) {
    const DustShockEvent &ev = all[e];
    // Use 1 hydrogen atom per cm^3 if the recorded density is missing or tiny.
    double gas_density_cgs = 1.0 * PROTONMASS;
    if(ev.rho_cgs > 0.01 * PROTONMASS) gas_density_cgs = ev.rho_cgs;

    // E_code is in code energy units of UnitEnergy_in_cgs / h erg.
    double sn_energy_erg = ev.E_code * All.UnitEnergy_in_cgs / All.HubbleParam;
    if(sn_energy_erg <= 0.0 || !isfinite(sn_energy_erg)) {
      static long long bad_energy_warns = 0;
      if(++bad_energy_warns <= 20 && All.ThisTask == 0)
        printf("[SN_SHOCK_WARNING] Invalid sn_energy=%.3e code units — falling back to 1e51 erg\n", ev.E_code);
      sn_energy_erg = 1e51;
    }
    // Blast-wave density: the local density, but no more than
    // DustShockAmbientDensity (hydrogen atoms per cm^3).
    const double rho_sedov = std::min(gas_density_cgs, All.DustShockAmbientDensity * PROTONMASS);
    Shock &s = shock[e];
    s.r_phys = std::max(calculate_sn_shock_radius(sn_energy_erg, rho_sedov, 0.3), 0.001);  // 1 pc floor
    // Speed from the PHYSICAL radius, same energy and density: R(t) and v(R)
    // only form a valid Sedov-Taylor pair when computed together. The larger
    // search radius would drop v below the destruction thresholds.
    s.v = calculate_sedov_velocity_from_radius(s.r_phys, rho_sedov, sn_energy_erg);
    s.r_search = std::min(std::max(s.r_phys, 2.0 * dust_hash.cell_size), 3.0);
    s.f_vol = std::max(0.0, std::min(1.0, pow(s.r_phys / s.r_search, 3.0)));
    get_shock_destruction_efficiency(s.v, &s.eps_carb, &s.eps_sil);
    if(All.DustDebugLevel > 0 && All.ThisTask == 0) {
      static int sn_debug_count = 0;
      sn_debug_count++;
      if(sn_debug_count <= 20 || sn_debug_count % 500 == 0)
        DUST_PRINT("[SN_SHOCK_DEBUG] Call #%d: physical_r=%.3f kpc  search_r=%.3f kpc  "
                   "v_shock=%.1f km/s  rho_local=%.3e  rho_sedov=%.3e %s\n",
                   sn_debug_count, s.r_phys, s.r_search, s.v, gas_density_cgs, rho_sedov,
                   (rho_sedov < gas_density_cgs) ? "[DENSITY_CAPPED]" : "[LOCAL_DENSITY]");
    }
  }

  // ── Step 5: dust inside each search sphere, summed over all tasks ────────
  // The tree holds live dust only, without the dust made in this feedback call.
  cg_dist::LocalTree dust_tree;
  dust_tree.build(Sp, 0, std::min(first_new_dust, Sp->NumPart), [Sp](int i) {
    return Sp->P[i].getType() == DUST_PARTICLE_TYPE && Sp->P[i].getMass() > 0.0 && Sp->P[i].ID.get() != 0;
  });
  // Five sums per event: carbon mass, silicate mass, the same two weighted by
  // shock_size_factor(), and the number of grains found.
  std::vector<double> sums(5 * n_events, 0.0);
  std::vector<std::vector<int>> found(n_events);  // this task's grains per event
  // Test switch: CG_TEST_LOCAL_SHOCK=1 brings back the old behaviour, where an
  // SN only sees dust on its own task. Used to test how much that mattered.
  static const bool local_only = getenv("CG_TEST_LOCAL_SHOCK") && atoi(getenv("CG_TEST_LOCAL_SHOCK")) == 1;
  int this_task_shock;
  MPI_Comm_rank(Communicator, &this_task_shock);
  for(size_t e = 0; e < n_events; e++)
    if(!local_only || owner[e] == this_task_shock)
    dust_tree.radius(Sp, all[e].pos, shock[e].r_search, [&](int i, double) {
      const double m  = Sp->P[i].getMass();
      const double cf = std::max(0.0, std::min(1.0, (double)Sp->DustP[i].CarbonMassFraction));
      const double sf = shock_size_factor(Sp->DustP[i].GrainRadius);
      sums[5 * e + 0] += m * cf;
      sums[5 * e + 1] += m * (1.0 - cf);
      sums[5 * e + 2] += m * cf * sf;
      sums[5 * e + 3] += m * (1.0 - cf) * sf;
      sums[5 * e + 4] += 1.0;
      found[e].push_back(i);
    });
  cg_dist::sum_over_tasks(sums, Communicator);

  // ── Steps 6-8: share each budget among the grains, return it to gas ──────
  // Shares use the all-task sums, so each task erodes its own grains exactly as
  // one task holding every grain would. Events are applied in star-ID order
  // (same on every task); a grain hit by several events can lose no more than
  // it still has.
  std::vector<size_t> order(n_events);
  for(size_t e = 0; e < n_events; e++) order[e] = e;
  std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return all[a].key < all[b].key; });

  // Carbon and silicate mass to hand back to each event's nearest gas particle.
  std::vector<double> returned(2 * n_events, 0.0);
  std::vector<double> lost(n_events, 0.0);
  std::vector<int> n_destroyed(n_events, 0), n_eroded(n_events, 0);
  for(size_t e : order) {
    const Shock &s = shock[e];
    const double M_carb_destroy = sums[5 * e + 0] * s.f_vol * s.eps_carb;
    const double M_sil_destroy  = sums[5 * e + 1] * s.f_vol * s.eps_sil;
    const double weight_carb_sum = sums[5 * e + 2], weight_sil_sum = sums[5 * e + 3];
    if(!(M_carb_destroy > 0.0 || M_sil_destroy > 0.0)) continue;
    const bool has_gas = all[e].gas_task >= 0;
    for(int i : found[e]) {
      if(Sp->P[i].getType() != DUST_PARTICLE_TYPE) continue;  // destroyed by an earlier event
      const double m = Sp->P[i].getMass();
      if(m <= 0.0) continue;
      // Carbon and silicate lose mass separately, so CarbonMassFraction changes
      // when one material is destroyed faster than the other.
      const double a_old  = Sp->DustP[i].GrainRadius;
      const double cf_old = std::max(0.0, std::min(1.0, (double)Sp->DustP[i].CarbonMassFraction));
      const double M_carb_old = m * cf_old, M_sil_old = m * (1.0 - cf_old);
      const double sf = shock_size_factor(a_old);
      double carb_loss = weight_carb_sum > 0.0 ? M_carb_destroy * (M_carb_old * sf / weight_carb_sum) : 0.0;
      double sil_loss  = weight_sil_sum  > 0.0 ? M_sil_destroy  * (M_sil_old  * sf / weight_sil_sum)  : 0.0;
      carb_loss = std::min(carb_loss, M_carb_old);
      sil_loss  = std::min(sil_loss, M_sil_old);
      const double M_carb_new = M_carb_old - carb_loss, M_sil_new = M_sil_old - sil_loss;
      const double new_mass = M_carb_new + M_sil_new;
      const double a_new = new_mass > 0.0 ? a_old * cbrt(new_mass / m) : 0.0;

      if(new_mass <= 0.0 || !isfinite(new_mass) || !isfinite(a_new) || a_new < All.DustMinGrainSize) {
        // Too little left: the whole grain goes back to the nearest gas. If no
        // gas particle was found for this event, the grain is left untouched.
        if(!has_gas) continue;
        log_dust_particle_event(Sp, i, -1, DUST_EVENT_SHOCK);
        returned[2 * e + 0] += m * cf_old;
        returned[2 * e + 1] += m * (1.0 - cf_old);
        lost[e] += m;
        mark_dust_particle_destroyed(Sp, i, &NDustDestroyedByShock, &TotalMassDestroyedByShock);
        n_destroyed[e]++;
      } else {
        Sp->P[i].setMass(new_mass);
        Sp->DustP[i].GrainRadius = a_new;
        Sp->DustP[i].CarbonMassFraction = (float)(M_carb_new / new_mass);
        if(has_gas) {
          returned[2 * e + 0] += carb_loss;
          returned[2 * e + 1] += sil_loss;
        }
        LocalDustMassChange    -= carb_loss + sil_loss;
        lost[e]                += carb_loss + sil_loss;
        TotalMassErodedByShock += carb_loss + sil_loss;
        NGrainErosionEvents++;
        n_eroded[e]++;
      }
    }
  }
  cg_dist::sum_over_tasks(returned, Communicator);

  // The task that owns each event's nearest gas particle adds the material to it.
  // The gas must still sit at the recorded index; if not, stop rather than
  // give the metals to the wrong particle.
  for(size_t e : order)
    if(all[e].gas_task == All.ThisTask && (returned[2 * e] > 0.0 || returned[2 * e + 1] > 0.0)) {
      const int g = all[e].gas_index;
      if(g < 0 || g >= Sp->NumGas || Sp->P[g].getType() != 0 || (std::uint64_t)Sp->P[g].ID.get() != all[e].gas_id)
        Terminate("SN shock: nearest gas ID %llu is no longer at index %d", (unsigned long long)all[e].gas_id, g);
      return_dust_material_to_gas(Sp, g, returned[2 * e], returned[2 * e + 1]);
    }

  // ── Diagnostics ──────────────────────────────────────────────────────────
  for(size_t e = 0; e < n_events; e++) {
    sn_total_calls++;
    if(sums[5 * e + 4] > 0) sn_found_dust++;
  }
  if(All.ThisTask == 0) {
    const double mass_to_msun = All.UnitMass_in_g / 1.989e33 / All.HubbleParam;
    for(size_t e = 0; e < n_events; e++) {
      if(owner[e] != 0) continue;  // print only events that started on task 0
      const Shock &s = shock[e];
      const double M_target = sums[5 * e] * s.f_vol * s.eps_carb + sums[5 * e + 1] * s.f_vol * s.eps_sil;
      if(M_target > 0.0)
        DUST_PRINT("[DUST_SN] physical_r=%.4f kpc  search_r=%.3f kpc  v=%.1f km/s  f_vol=%.3e  "
                   "eps_C=%.3f eps_sil=%.3f  M_C=%.3e M_sil=%.3e  "
                   "M_target=%.3e code (%.3e Msun)  n_dust=%.0f (all tasks)  "
                   "local: lost=%.3e code destroyed=%d eroded=%d\n",
                   s.r_phys, s.r_search, s.v, s.f_vol, s.eps_carb, s.eps_sil,
                   sums[5 * e], sums[5 * e + 1], M_target, M_target * mass_to_msun,
                   sums[5 * e + 4], lost[e], n_destroyed[e], n_eroded[e]);
    }
    static long long next_report = 1000;
    if(sn_total_calls >= next_report) {
      DUST_PRINT("[SN_RATE] %lld SN events, %lld found dust (%.1f%%)\n",
                 sn_total_calls, sn_found_dust, 100.0 * sn_found_dust / sn_total_calls);
      while(next_report <= sn_total_calls) next_report += 1000;
    }
  }
}
// ═══════════════════════════════════════════════════════════════════════════════
// HK11 accretion timescale helper
// ═══════════════════════════════════════════════════════════════════════════════
/**
 * Time (in years) for a grain to grow by sticking gas-phase metals onto its
 * surface ("accretion"), from Hirashita & Kuo (2011, HK11):
 *
 *   τ_acc = pref × (a/0.1 µm) × (Zsun/Z) × (1000/nH) × (50K/T)^0.5 × (0.3/S)
 *
 * a = grain radius, Z = gas metal mass fraction, nH = hydrogen number density
 * (cm^-3), T = gas temperature, S = sticking probability. The prefactor depends
 * on the material:
 *   silicate     (species=0): pref = 6.30e7 yr
 *   carbonaceous (species=1): pref = 5.59e7 yr
 *
 * Returns HUGE_VAL (no growth) if any input is zero or negative, so the caller
 * can simply skip the grain.
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
// ============================================================================
// Dense-phase mass fraction (DustEnableClumping >= 2)
// ============================================================================
// A simulation gas particle is too big to resolve cold clouds, but grains grow
// and stick together mostly in those clouds. So we estimate what fraction of
// each gas particle's mass is in the dense (molecular) phase.
//
// Blitz & Rosolowsky (2006, ApJ 650, 933) pressure law:
//   R_mol = (P / P0)^alpha,  P0/k = 4.3e4 K cm^-3,  alpha = 0.92,
//   f_dense = R_mol / (1 + R_mol).
// P is the particle's pressure (k = Boltzmann constant). Star-forming gas sits
// on the Springel & Hernquist (2003) effective equation of state, so for it P is
// their effective multiphase pressure. Non-star-forming gas hotter than 3e4 K
// cannot hold cold clouds and gets f_dense = 0.
//
// Hydrogen density of the dense phase, used in the coagulation and accretion rates:
//   mode 2: n_dense = 100 cm^-3 (or the resolved n_H if that is higher);
//   mode 3: pressure balance, n_dense = (P/k) / T_cold with T_cold = 100 K
//           (never below the resolved n_H).
// Coagulation and accretion act on the dense part (rates x f_dense);
// shattering acts on the diffuse part (rate x (1 - f_dense)) at the resolved n_H.
// Both run every step: there is no density on/off switch.
// Mode 1 keeps the older clumping tiers (dust_clumping_factor) and the n_eff switch.
static const double DUST_BR06_P0_K_CM3   = 4.3e4;
static const double DUST_BR06_ALPHA      = 0.92;
static const double DUST_DENSE_FIXED_NH  = 100.0;
static const double DUST_DENSE_T_COLD_K  = 100.0;
static const double DUST_DENSE_T_HOT_MAX = 3.0e4;

static inline bool dust_dense_phase_model(void) { return All.DustEnableClumping >= 2; }

/** Return f_dense for gas particle gas_idx and set *n_dense_cm3 (modes >= 2 only). */
static double dust_dense_fraction(simparticles *Sp, int gas_idx, double *n_dense_cm3)
{
  const double n_H = dust_gas_nH_cgs(Sp, gas_idx);
  *n_dense_cm3     = std::max(n_H, DUST_DENSE_FIXED_NH);
  const bool is_sf = Sp->SphP[gas_idx].Sfr > DUST_SFR_EPS;
  if(!is_sf)
    {
      const double T = get_temperature_from_entropy(Sp, gas_idx);
      if(!isfinite(T) || T > DUST_DENSE_T_HOT_MAX) return 0.0;
    }
  const double u_cgs = Sp->get_utherm_from_entropy(gas_idx) * (All.UnitEnergy_in_cgs / All.UnitMass_in_g);
  const double P_over_k = GAMMA_MINUS1 * dust_gas_density_cgs(Sp, gas_idx) * u_cgs / BOLTZMANN;
  if(!isfinite(P_over_k) || !(P_over_k > 0.0)) return 0.0;
  if(All.DustEnableClumping >= 3)
    *n_dense_cm3 = std::max(n_H, P_over_k / DUST_DENSE_T_COLD_K);
  const double R = pow(P_over_k / DUST_BR06_P0_K_CM3, DUST_BR06_ALPHA);
  return R / (1.0 + R);
}

/**
 * Clumping factor C (mode 1): how much denser than the particle average the
 * gas really is, because the simulation cannot resolve small dense clumps.
 * Growth and coagulation use n_eff = C × n_H, with n_H the hydrogen number density.
 *
 * Star-forming gas gets C = 30. Other gas gets 10, 3 or 1.5 above 0.5, 0.2 and
 * 0.05 × CritPhysDensity (the star-formation density threshold), and 1 below.
 * The thresholds follow CritPhysDensity so they move with it; the C values
 * themselves are fixed at every resolution. Returns 1 when clumping is off.
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
// DUST COAGULATION: grains stick together and get bigger
// ============================================================================
// In dense, quiet gas, small grains collide gently and stick. The dust mass
// stays the same but is held in fewer, larger grains.
//
// Timescale from Trayford et al. (2026), MNRAS, doi:10.1093/mnras/staf2040,
// Section 3.5.2, Equation (10), adapted from Aoyama et al. (2017):
//
//   tau_co = 27.1 Myr * (a / 0.1 micron) * (DTG / 0.01)^(-1)
//          * (v_co / 0.1 km/s)^(-1) * (n_H_eff / 10 cm^-3)^(-1).
//
// a = grain radius, DTG = local dust-to-gas mass ratio, v_co = grain-grain
// collision speed, n_H_eff = effective hydrogen number density.
//
// Trayford et al. move mass between two size bins (small bin a_S = 0.01 micron)
// with v_co = 0.2 km/s. We keep their v_co but instead follow one continuous
// radius per dust superparticle (a simulation particle that stands for many
// grains with the same mass). So this is NOT their exact two-bin method, nor a
// full grain-grain collision calculation.
//
// DTG comes from the dust and gas particle masses in the local association
// sphere, added up over all MPI ranks and clamped to [1e-6, 0.1].
// DustCoagulationCalibration multiplies the timescale (1 = no calibration).
//
// Growth law for the one representative grain: da/dt = a/(3*tau_co). With the
// gas fixed, tau_co is proportional to a, so da/dt is constant and the exact
// answer over dt is
//   a_new = a_old * (1 + dt/(3*tau_co(a_old))).
// We still split dt into substeps of at most 20% radius growth each; the whole
// dt is used unless the grain reaches DustMaxGrainSize first.
// Particle mass and carbon fraction do not change: the grains just get fewer
// and bigger, and no carbon is mixed in from other grains.
// ============================================================================
void dust_grain_coagulation(simparticles *Sp, int dust_idx, int gas_idx, double dt,
                            double local_DTG, double T_gas)
{
  if(!All.DustEnableCoagulation || !(dt > 0.0) || !isfinite(dt)) return;

  ++CoagCalls;

  const double n_H = dust_gas_nH_cgs(Sp, gas_idx);
  const bool is_star_forming = Sp->SphP[gas_idx].Sfr > DUST_SFR_EPS;
  double n_eff;
  double f_dense = 1.0;  // mode 1: the whole particle coagulates or none of it
  if(dust_dense_phase_model()) {
    // Only the dense part coagulates: rate x f_dense at the dense-phase density.
    // Almost no dense gas (f_dense < 1e-4) counts as a density rejection.
    f_dense = dust_dense_fraction(Sp, gas_idx, &n_eff);
    if(!(f_dense >= 1.0e-4) || !isfinite(n_eff) || !(n_eff > 0.0)) {
      ++CoagFailedDensity;
      return;
    }
  } else {
    const double clumping = dust_clumping_factor(n_H, is_star_forming);
    n_eff = n_H * clumping;
    if(!isfinite(n_eff) || n_eff < All.DustCollisionDensityThresh ||
       !(n_eff > 0.0)) { ++CoagFailedDensity; return; }

    // Star-forming gas has an effective-EOS temperature that says nothing about
    // its cold clouds, so it skips the temperature test. Other gas keeps the
    // original rule: no coagulation above 3000 K.
    if(!is_star_forming && (!isfinite(T_gas) || T_gas > 3000.0)) {
      ++CoagFailedTemperature;
      return;
    }
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

  // Trayford et al. (2026), Eq. 10, with a_nm in nm (100 nm = 0.1 micron).
  // Dividing by f_dense makes the rate proportional to the dense fraction.
  const double v_co_kms = 0.2;
  const double tau0_yr = 27.1e6;
  const double calibration = All.DustCoagulationCalibration;
  if(!isfinite(calibration) || !(calibration > 0.0)) return;
  const double tau_co_yr = tau0_yr * (a_nm / 100.0)
      * (0.01 / local_DTG) * (0.1 / v_co_kms)
      * (10.0 / n_eff) * calibration / f_dense;
  if(!isfinite(tau_co_yr) || !(tau_co_yr > 0.0)) return;

  // dt is physical time in code units (as passed by update_dust_dynamics()),
  // so convert tau_co to the same units.
  const double tau_co_code = tau_co_yr * SEC_PER_YEAR / dust_code_time_in_s();
  if(!isfinite(tau_co_code) || !(tau_co_code > 0.0)) return;

  // Substeps: each is at most 0.6 tau, i.e. at most 20% radius growth. Because
  // tau_co grows with radius, the substeps add up to the exact answer above;
  // they only keep each single update small. CoagMaxFractionalIncrease records
  // the largest one-step growth asked for, before substepping.
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
 * Grow a dust particle by sticking gas-phase metals onto its grains
 * ("accretion"), following Hirashita & Kuo (2011, HK11).
 *
 * Each dust particle is a mix of carbon (C) and silicate (sil); CF is its
 * carbon mass fraction (CarbonMassFraction). Each part grows on its own HK11
 * timescale, so CF changes smoothly. (Before, one material was picked for the
 * whole grain with a hard CF >= 0.5 cut.)
 *
 *   M_C   = CF × M_dust
 *   M_sil = (1 - CF) × M_dust
 *   dM_C   = M_C   [ exp(3 f_mol dt / tau_C)   - 1 ]
 *   dM_sil = M_sil [ exp(3 f_mol dt / tau_sil) - 1 ]
 *
 * f_mol = molecular (dense) gas fraction, where accretion happens. The factor 3
 * keeps the earlier radius law a_new = a exp(f_mol dt / tau), because M ∝ a^3.
 *
 * The total-mass limits (maximum D/Z = dust-to-metal mass ratio, metals
 * available in the gas, growth per call) act on the total dM. If a limit cuts
 * the growth, the allowed mass is split between C and sil in the same ratio as
 * the uncut requests, so composition and total mass stay consistent.
 *
 *   CF_new = (M_C_old + dM_C) / (M_dust_old + dM_total)
 */
void dust_grain_growth_subgrid(simparticles *Sp, int dust_idx, int gas_idx, double dt,
                               double T_gas)
{
  if(!All.DustEnableGrowth) return;
  // ── Diagnostics: why calls fail, counted per task, printed by rank 0 ───────
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
  // Which way the carbon fraction moved
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
  // ── Gate 1: no growth in gas hot enough to sputter grains ──────────────────
  if(T_gas > All.DustThermalSputteringTemp) {
    failed_hot++;
    return;
  }
  // ── Gate 2: gas must hold metals to accrete (Z >= 1e-4) ─────────────────────
  const double Z_gas = Sp->SphP[gas_idx].Metallicity;
  if(Z_gas < 1e-4) {
    failed_no_metals++;
    return;
  }
  // ── Gate 3: gas must be dense enough (mode 1: n_eff >= 0.1 cm^-3) ──────────
  double n_H = dust_gas_nH_cgs(Sp, gas_idx);
  // Dense-phase model (modes >= 2): f_dense replaces the f_mol tiers below, and
  // the dense-phase density replaces the clumped density in the HK11 timescale.
  // There is no density gate; low f_dense is caught by the f_mol < 0.01 test.
  double f_dense_acc = -1.0;
  double n_eff_cm3;
  double DustClumpingFactor;  // logged: n_eff / n_H in either model
  if(dust_dense_phase_model()) {
    f_dense_acc = dust_dense_fraction(Sp, gas_idx, &n_eff_cm3);
    DustClumpingFactor = (n_H > 0.0) ? n_eff_cm3 / n_H : 0.0;
  } else {
    DustClumpingFactor = dust_clumping_factor(n_H, Sp->SphP[gas_idx].Sfr > DUST_SFR_EPS);
    n_eff_cm3 = n_H * DustClumpingFactor;
    if(n_eff_cm3 < 0.1) {
      failed_low_density++;
      return;
    }
  }
  // ── Maximum D/Z (dust-to-metal mass ratio), lower at high redshift ──────────
  // 0.5 at z = 0, falling as 0.5 / (1 + 0.15 z), never below 0.05.
  double z_now = 1.0 / All.Time - 1.0;
  double max_dust_to_metal = std::max(0.05, 0.5 / (1.0 + 0.15 * z_now));
  double M_gas_quick  = Sp->P[gas_idx].getMass();
  double M_dust_quick = Sp->P[dust_idx].getMass();
  // Quick early exit with the same D/Z limit as the exact cap further down.
  if(M_dust_quick >=
     M_gas_quick * Z_gas * max_dust_to_metal) {
    failed_max_dz++;
    return;
  }
  // ── Molecular fraction f_mol (mode 1 tiers) ─────────────────────────────────
  // Accretion happens in molecular gas, so denser and star-forming gas gets a
  // larger f_mol: 0.8 star-forming, 0.5 above 100 cm^-3, 0.2 above 10, else 0.05.
  // Densities are n_eff; builds without STARFORMATION use n_H and have no SF tier.
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
  // Metal-rich gas (Z > 0.01) shields its molecules from UV light better: f_mol x 1.5.
  if(Z_gas > 0.01)
    f_mol = std::min(1.0, 1.5 * f_mol);
  // Dense-phase model: the Blitz & Rosolowsky f_dense replaces the tiers above.
  if(f_dense_acc >= 0.0)
    f_mol = f_dense_acc;
  if(f_mol < 0.01) {
    failed_low_fmol++;
    return;
  }
  // ── Dust particle must exist and be close to its gas particle ──────────────
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

  // Same distance rule as update_dust_dynamics(): the dust must lie within the
  // gas particle's smoothing length Hsml, but at least 2 and at most 10 kpc.
  // This used to be a fixed 5 kpc, which quietly rejected grains that had
  // passed the Hsml-based association test in the main loop.
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
  // A zero, negative or NaN radius or mass means the particle's data is broken.
  // All-zero DustP fields point to a domain exchange that did not send DustP;
  // anything else is labelled a numerical problem. Either way the particle is
  // flagged for removal (as in mark_dust_particle_destroyed) and counted.
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
  // Grow carbon and silicate separately
  // ════════════════════════════════════════════════════════════════════════════
  const double CF_old = std::max(0.0, std::min(1.0, (double)Sp->DustP[nearest_dust].CarbonMassFraction));
  const double M_carb_old = M_dust * CF_old;
  const double M_sil_old  = M_dust * (1.0 - CF_old);
  // Fixed HK11 inputs: cold-cloud gas temperature 20 K (not the particle's T),
  // solar metal mass fraction 0.02, sticking probability 0.3.
  const double T_eff_K       = 20.0;
  const double Zsun_massfrac = 0.02;
  const double S_stick       = 0.3;
  double n_eff_for_growth = std::max(n_eff_cm3, 1e-3);
  double a_cm             = a * 1e-7;  // nm -> cm
  // ── One HK11 timescale for carbon, one for silicate ─────────────────────────
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
  // Keep each timescale between 1 Myr and 5 Gyr (the limits the old single
  // timescale had), then convert to code time units.
  tau_carb_yr = std::max(1e6, std::min(5e9, tau_carb_yr));
  tau_sil_yr  = std::max(1e6, std::min(5e9, tau_sil_yr));
  const double tau_carb = tau_carb_yr * SEC_PER_YEAR / dust_code_time_in_s();
  const double tau_sil = tau_sil_yr * SEC_PER_YEAR / dust_code_time_in_s();
  // ── Requested growth of each material, before any limit ─────────────────────
  // Old radius law a_new = a exp(f_mol dt / tau). Since M ∝ a^3, that means
  // M_new / M_old = exp(3 f_mol dt / tau). Apply it to C and silicate separately.
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
  // Start from the full request; the caps below can only lower it.
  double dm = dm_raw;
  // ── Metals in the gas particle, and the most dust that D/Z allows ───────────
  const double M_gas       = Sp->P[gas_idx].getMass();
  const double M_metals    = M_gas * Z_gas;
  const double M_dust_max  = M_metals * max_dust_to_metal;
  if(M_dust >= M_dust_max) {
    failed_max_dz++;
    return;
  }
  // ── Cap 1: stay below the maximum D/Z ───────────────────────────────────────
  if(M_dust + dm > M_dust_max) {
    dm = M_dust_max - M_dust;
    if(dm <= 0.0)
      return;
    capped_by_dz++;
  }
  // ── Cap 2: cannot take more metal than the gas has (leave 1%) ───────────────
  if(dm > M_metals) {
    dm = 0.99 * M_metals;
    capped_by_metals++;
  }
  // ── Cap 3: at most 20% of the current dust mass per call ───────────────────
  const double max_dm_per_step = 0.2 * M_dust;
  if(dm > max_dm_per_step) {
    dm = max_dm_per_step;
    capped_by_perstep++;
  }
  if(!isfinite(dm) || dm <= 0.0)
    return;
  // ════════════════════════════════════════════════════════════════════════════
  // Split the allowed growth between carbon and silicate
  // ════════════════════════════════════════════════════════════════════════════
  // Use carbon's share of the uncapped request, so the C/silicate ratio of the
  // HK11 rates survives even when a cap cut the total.
  double carb_growth_fraction = dm_carb_raw / dm_raw;
  carb_growth_fraction = std::max(0.0, std::min(1.0, carb_growth_fraction));
  double dm_carb = dm * carb_growth_fraction;
  // Silicate gets the rest, so dm_carb + dm_sil == dm exactly.
  double dm_sil = dm - dm_carb;
    // ════════════════════════════════════════════════════════════════════════════
  // Move the material from gas to dust
  // ════════════════════════════════════════════════════════════════════════════
  // First check that the REQUESTED growth gives a radius inside
  // [DustMinGrainSize, DustMaxGrainSize]. The gas helper below can only give
  // less than requested, never more, so the real result will be safe too.
  double M_dust_requested = M_dust + dm;
  double a_requested = a * cbrt(M_dust_requested / M_dust);
  if(!isfinite(a_requested) ||
     a_requested < All.DustMinGrainSize ||
     a_requested > All.DustMaxGrainSize)
      return;
  // Take the C and silicate material out of the gas particle's individual
  // element masses. If the gas lacks an element, that part comes out smaller.
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
  // Rebuild the grain from what was ACTUALLY taken from the gas
  // ════════════════════════════════════════════════════════════════════════════
  double M_carb_new = M_carb_old + actual_carb;
  double M_sil_new = M_sil_old + actual_sil;
  double M_dust_new = M_carb_new + M_sil_new;
  if(!isfinite(M_dust_new) ||
     M_dust_new <= M_dust)
      return;
  double CF_new = M_carb_new / M_dust_new;
  CF_new = std::max(0.0, std::min(1.0, CF_new));
  // Radius from the actual new mass (a ∝ m^(1/3)). Out of range here would mean
  // the check above failed, so it is a bug and we stop.
  double a_new = a * cbrt(M_dust_new / M_dust);
  if(!isfinite(a_new) ||
     a_new < All.DustMinGrainSize ||
     a_new > All.DustMaxGrainSize)
      Terminate("Invalid grain radius after element-resolved dust growth");

    double da = a_new - a;

    if(!isfinite(da))
        Terminate("Non-finite grain growth after gas material removal");

    // A very small growth can round to da = 0, so a_new == a is allowed
    // (and logged below); a shrinking grain is not.
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
  // Only now does the call count as a success.
  passed_all++;
  NGrainGrowthEvents++;
  TotalMassGrown += dm_actual;
  const double dCF = CF_new - CF_old;
  if(dCF > 1e-10)
      cf_increased++;
  else if(dCF < -1e-10)
      cf_decreased++;
  else
      cf_unchanged++;
  // ── First 10 successful growths on rank 0: print the details ──────────────
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
  // ── Every 10000th growth on rank 0: print composition and inputs ──────────
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