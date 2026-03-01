/*! \file spatial_hash_zoom.h
 *
 *  \brief Zoom-aware spatial hash for fast neighbor finding in Gadget-4
 *
 *  It can find either gas neighbors or star neighbors, depending on which particle type is used to build the hash.
 * 
 *  SPARSE IMPLEMENTATION - Only allocates non-empty cells!
 *  Optimized for zoom simulations by creating spatial bins only over the
 *  gas-occupied region, dramatically reducing memory usage and improving
 *  performance compared to full-box spatial hashing.
 *
 *  Key features:
 *  - Automatic detection of particle extent (where either PartType0 or PartType4 exists)
 *  - Adaptive cell sizing based on particle count and search radius
 *  - Efficient neighbor finding with minimal empty cell overhead
 *  - Automatically tracks gas as halo drifts/evolves
 *  - MPI-synchronized bounding box detection (uses Gadget Communicator, not MPI_COMM_WORLD)
 *  - SPARSE storage: only non-empty cells consume memory
 */

#ifndef SPATIAL_HASH_ZOOM_H
#define SPATIAL_HASH_ZOOM_H

#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <mpi.h>
#include "../data/allvars.h"
#include "../data/simparticles.h"

// Configuration constants for spatial hash behavior
struct spatial_hash_config {
  static constexpr int    MIN_CELLS_PER_DIM          = 8;
  static constexpr int    MAX_CELLS_PER_DIM          = 768;
  static constexpr int    MAX_TOTAL_CELLS             = 452984832;  // 768^3
  static constexpr int    TARGET_PARTICLES_PER_CELL  = 32;
  static constexpr double CELL_SIZE_SAFETY_FACTOR    = 2.5;
  static constexpr double BBOX_PADDING_FACTOR        = 1.2;
};

/**
 * Zoom-aware spatial hash structure (SPARSE VERSION)
 *
 * Creates a 3D grid of cells only over the region where gas exists,
 * enabling fast O(1) neighbor finding for feedback and dust operations.
 * Only non-empty cells are stored in memory (sparse hash table).
 *
 * IMPORTANT: All methods that perform MPI communication accept an explicit
 * MPI_Comm argument. Never use MPI_COMM_WORLD directly — always pass Gadget's
 * internal Communicator to avoid deadlocks when running across multiple nodes.
 */
struct spatial_hash_zoom {
  // Grid parameters
  int    n_cells_per_dim;
  double cell_size;
  double box_size;

  // Bounding box for the gas-occupied region
  double bbox_min[3];
  double bbox_max[3];
  double bbox_size[3];

  int total_particles;

  // Sparse cell storage: key = flat cell index, value = local gas particle indices
  std::unordered_map<int, std::vector<int>> cells;

  bool is_built;

  spatial_hash_zoom() : n_cells_per_dim(0), cell_size(0), box_size(0),
                        total_particles(0), is_built(false)
  {
    for(int d = 0; d < 3; d++)
      bbox_min[d] = bbox_max[d] = bbox_size[d] = 0.0;
  }

  // -------------------------------------------------------------------------
  // Calculate optimal number of cells based on particle count and search radius
  // -------------------------------------------------------------------------
  int calculate_optimal_cells(int n_particles, double search_radius, double softening)
  {
    // Method 1: target ~TARGET_PARTICLES_PER_CELL particles per cell
    int n_from_particles = (int)std::cbrt((double)n_particles /
                                          (double)spatial_hash_config::TARGET_PARTICLES_PER_CELL);

    // Method 2: cells must be >= CELL_SIZE_SAFETY_FACTOR × search_radius
    double typical_bbox = std::cbrt(bbox_size[0] * bbox_size[1] * bbox_size[2]);
    double max_cell_size = search_radius * spatial_hash_config::CELL_SIZE_SAFETY_FACTOR;
    int n_from_search = (int)std::ceil(typical_bbox / max_cell_size);

    // Method 3: cells ~ 4× softening for adequate force resolution
    int n_from_softening = (int)std::ceil(typical_bbox / (4.0 * softening));

    int n_cells = std::max({n_from_particles, n_from_search, n_from_softening});
    n_cells = std::max(spatial_hash_config::MIN_CELLS_PER_DIM, n_cells);
    n_cells = std::min(spatial_hash_config::MAX_CELLS_PER_DIM, n_cells);

    long long total_cells = (long long)n_cells * n_cells * n_cells;
    if(total_cells > spatial_hash_config::MAX_TOTAL_CELLS)
      n_cells = (int)std::cbrt((double)spatial_hash_config::MAX_TOTAL_CELLS);

    return n_cells;
  }

  // -------------------------------------------------------------------------
  // Always computes bounding box from gas particles, regardless of what
  // particle type the hash will store. Gas traces the zoom region and
  // provides the correct spatial extent for all hash types.
  // -------------------------------------------------------------------------
  void detect_extent_collective(simparticles *Sp, int local_gas, MPI_Comm comm)
  {
    double local_min[3], local_max[3];
    for(int d = 0; d < 3; d++) { local_min[d] = 1e30; local_max[d] = -1e30; }

    if(local_gas > 0) {
      for(int i = 0; i < Sp->NumGas; i++) {
        double pos[3];
        Sp->intpos_to_pos(Sp->P[i].IntPos, pos);
        for(int d = 0; d < 3; d++) {
          if(pos[d] < local_min[d]) local_min[d] = pos[d];
          if(pos[d] > local_max[d]) local_max[d] = pos[d];
        }
      }
    }
    // local_gas == 0: local_min/max stay at sentinel values,
    // which are neutral elements for MPI_MIN/MPI_MAX reductions

    double global_min[3], global_max[3];
    MPI_Allreduce(local_min, global_min, 3, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(local_max, global_max, 3, MPI_DOUBLE, MPI_MAX, comm);

    for(int d = 0; d < 3; d++) {
      bbox_min[d] = global_min[d];
      bbox_max[d] = global_max[d];
    }

    // Pad bounding box by BBOX_PADDING_FACTOR so particles near the edges aren't missed
    for(int d = 0; d < 3; d++) {
      double center     = 0.5 * (bbox_min[d] + bbox_max[d]);
      double half_width = 0.5 * (bbox_max[d] - bbox_min[d]) *
                          spatial_hash_config::BBOX_PADDING_FACTOR;
      bbox_min[d]  = center - half_width;
      bbox_max[d]  = center + half_width;
      bbox_size[d] = bbox_max[d] - bbox_min[d];
    }
  }

  // -------------------------------------------------------------------------
  // Convert 3D position to flat cell index
  // -------------------------------------------------------------------------
  int pos_to_cell_index(double pos[3]) const
  {
    int ix = (int)std::floor((pos[0] - bbox_min[0]) / cell_size);
    int iy = (int)std::floor((pos[1] - bbox_min[1]) / cell_size);
    int iz = (int)std::floor((pos[2] - bbox_min[2]) / cell_size);

    ix = std::max(0, std::min(n_cells_per_dim - 1, ix));
    iy = std::max(0, std::min(n_cells_per_dim - 1, iy));
    iz = std::max(0, std::min(n_cells_per_dim - 1, iz));

    return ix + n_cells_per_dim * (iy + n_cells_per_dim * iz);
  }

  // -------------------------------------------------------------------------
  // Build the spatial hash (collective — all tasks must call)
  // comm must be Gadget's internal Communicator, NOT MPI_COMM_WORLD
  // -------------------------------------------------------------------------
  void build(simparticles *Sp, double max_search_radius, double softening, 
            MPI_Comm comm, int part_type = 0)  // default 0 = gas
  {
    box_size = All.BoxSize;

    // Count local particles of requested type
    int local_count = 0;
    int n_total = Sp->NumPart;
    for(int i = 0; i < n_total; i++)
        if(Sp->P[i].getType() == part_type) local_count++;

    // For gas, use fast path (contiguous at front)
    if(part_type == 0) local_count = Sp->NumGas;

    int global_count = 0;
    MPI_Allreduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM, comm);
    total_particles = global_count;

    if(global_count == 0) { is_built = false; cells.clear(); return; }

    detect_extent_collective(Sp, Sp->NumGas, comm);

    n_cells_per_dim = calculate_optimal_cells(global_count, max_search_radius, softening);
    double max_bbox = std::max({bbox_size[0], bbox_size[1], bbox_size[2]});
    cell_size = max_bbox / n_cells_per_dim;

    cells.clear();

    for(int i = 0; i < Sp->NumPart; i++) {
        if(Sp->P[i].getType() != part_type) continue;
        double pos[3];
        Sp->intpos_to_pos(Sp->P[i].IntPos, pos);
        bool inside = true;
        for(int d = 0; d < 3; d++)
            if(pos[d] < bbox_min[d] || pos[d] > bbox_max[d]) { inside = false; break; }
        if(inside) cells[pos_to_cell_index(pos)].push_back(i);
    }

    is_built = true;

    if(All.ThisTask == 0) {
      double volume_ratio = (bbox_size[0] * bbox_size[1] * bbox_size[2]) /
                            (box_size * box_size * box_size);
      long long theoretical_cells = (long long)n_cells_per_dim * n_cells_per_dim * n_cells_per_dim;
      printf("[SPATIAL_HASH_ZOOM] Built: %d^3 cells, cell_size=%.3f kpc\n",
            n_cells_per_dim, cell_size);
      printf("[SPATIAL_HASH_ZOOM] Sparse: %zu/%lld cells allocated (%.1f%%)\n",
            cells.size(), theoretical_cells, 100.0 * cells.size() / theoretical_cells);
      printf("[SPATIAL_HASH_ZOOM] BBox: %.1f%% of full box volume\n",
            100.0 * volume_ratio);
    }
  }

  // -------------------------------------------------------------------------
  // Find nearest gas particle within max_search_radius (local operation, no MPI)
  // This can be a GAS particle or a STAR particle... a separate hash instance is built for each type.
  // -------------------------------------------------------------------------
  int find_nearest_particle(simparticles *Sp, int idx, double max_search_radius,
                                double *out_distance) const
  {
    if(!is_built) return -1;

    double pos[3];
    Sp->intpos_to_pos(Sp->P[idx].IntPos, pos);

    for(int d = 0; d < 3; d++) {
      if(pos[d] < bbox_min[d] || pos[d] > bbox_max[d]) {
        if(out_distance) *out_distance = -1.0;
        return -1;
      }
    }

    int n_search = (int)std::ceil(max_search_radius / cell_size) + 1;

    int home_ix = (int)std::floor((pos[0] - bbox_min[0]) / cell_size);
    int home_iy = (int)std::floor((pos[1] - bbox_min[1]) / cell_size);
    int home_iz = (int)std::floor((pos[2] - bbox_min[2]) / cell_size);

    double min_r2    = max_search_radius * max_search_radius;
    int    nearest   = -1;

    for(int dix = -n_search; dix <= n_search; dix++) {
      for(int diy = -n_search; diy <= n_search; diy++) {
        for(int diz = -n_search; diz <= n_search; diz++) {
          int ix = home_ix + dix;
          int iy = home_iy + diy;
          int iz = home_iz + diz;

          if(ix < 0 || ix >= n_cells_per_dim) continue;
          if(iy < 0 || iy >= n_cells_per_dim) continue;
          if(iz < 0 || iz >= n_cells_per_dim) continue;

          int cell_idx = ix + n_cells_per_dim * (iy + n_cells_per_dim * iz);
          auto it = cells.find(cell_idx);
          if(it == cells.end()) continue;

          for(int particle_idx : it->second) {
            if(particle_idx == idx) continue;
            double dxyz[3];
            Sp->nearest_image_intpos_to_pos(Sp->P[particle_idx].IntPos, Sp->P[idx].IntPos, dxyz);
            double r2 = dxyz[0]*dxyz[0] + dxyz[1]*dxyz[1] + dxyz[2]*dxyz[2];
            if(r2 < min_r2) { min_r2 = r2; nearest = particle_idx; }
          }
        }
      }
    }

    if(out_distance)
      *out_distance = (nearest >= 0) ? std::sqrt(min_r2) : -1.0;

    return nearest;
  }

  // -------------------------------------------------------------------------
  // Find all gas neighbors within search_radius (local operation, no MPI)
  // -------------------------------------------------------------------------
  void find_neighbors(simparticles *Sp, int idx, double search_radius,
                      int *neighbor_indices, double *neighbor_distances,
                      int *n_neighbors, int max_neighbors) const
  {
    *n_neighbors = 0;
    if(!is_built) return;

    double pos[3];
    Sp->intpos_to_pos(Sp->P[idx].IntPos, pos);

    for(int d = 0; d < 3; d++)
      if(pos[d] < bbox_min[d] || pos[d] > bbox_max[d]) return;

    if(cell_size <= 0 || !std::isfinite(cell_size)) return;

    int n_search = (int)std::ceil(search_radius / cell_size) + 1;
    n_search = std::min(n_search, n_cells_per_dim);

    int home_ix = (int)std::floor((pos[0] - bbox_min[0]) / cell_size);
    int home_iy = (int)std::floor((pos[1] - bbox_min[1]) / cell_size);
    int home_iz = (int)std::floor((pos[2] - bbox_min[2]) / cell_size);

    double search_r2 = search_radius * search_radius;
    int count = 0;

    for(int dix = -n_search; dix <= n_search; dix++) {
      for(int diy = -n_search; diy <= n_search; diy++) {
        for(int diz = -n_search; diz <= n_search; diz++) {
          int ix = home_ix + dix;
          int iy = home_iy + diy;
          int iz = home_iz + diz;

          if(ix < 0 || ix >= n_cells_per_dim) continue;
          if(iy < 0 || iy >= n_cells_per_dim) continue;
          if(iz < 0 || iz >= n_cells_per_dim) continue;

          int cell_idx = ix + n_cells_per_dim * (iy + n_cells_per_dim * iz);
          auto it = cells.find(cell_idx);
          if(it == cells.end()) continue;

          for(int particle_idx : it->second) {
            if(particle_idx == idx) continue;
            double dxyz[3];
            Sp->nearest_image_intpos_to_pos(Sp->P[particle_idx].IntPos, Sp->P[idx].IntPos, dxyz);
            double r2 = dxyz[0]*dxyz[0] + dxyz[1]*dxyz[1] + dxyz[2]*dxyz[2];
            if(r2 <= search_r2) {
              if(count < max_neighbors) {
                neighbor_indices[count]   = particle_idx;
                neighbor_distances[count] = std::sqrt(r2);
                count++;
              } else {
                *n_neighbors = count;
                return;
              }
            }
          }
        }
      }
    }

    *n_neighbors = count;
  }

  // -------------------------------------------------------------------------
  // Print statistics (task 0 only, no MPI)
  // -------------------------------------------------------------------------
  void print_stats() const
  {
    if(!is_built || All.ThisTask != 0) return;

    long long theoretical_cells = (long long)n_cells_per_dim * n_cells_per_dim * n_cells_per_dim;
    int    allocated_cells  = (int)cells.size();
    int    max_particles    = 0;
    long long total_part    = 0;

    for(const auto &kv : cells) {
      int n = (int)kv.second.size();
      total_part   += n;
      max_particles = std::max(max_particles, n);
    }

    printf("[SPATIAL_HASH_ZOOM] Statistics:\n");
    printf("  BBox: [%.1f,%.1f] x [%.1f,%.1f] x [%.1f,%.1f] kpc\n",
           bbox_min[0], bbox_max[0], bbox_min[1], bbox_max[1], bbox_min[2], bbox_max[2]);
    printf("  Grid: %d^3 = %lld theoretical cells, cell_size=%.3f kpc\n",
           n_cells_per_dim, theoretical_cells, cell_size);
    printf("  Allocated: %d cells (%.3f%%)\n",
           allocated_cells, 100.0 * allocated_cells / theoretical_cells);
    printf("  Max particles/cell: %d  |  Avg: %.1f\n",
           max_particles, (double)total_part / std::max(1, allocated_cells));
  }
};

#endif // SPATIAL_HASH_ZOOM_H