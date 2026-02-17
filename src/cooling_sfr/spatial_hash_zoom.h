/*! \file spatial_hash_zoom.h
 *
 *  \brief Zoom-aware spatial hash for fast neighbor finding in Gadget-4
 *
 *  SPARSE IMPLEMENTATION - Only allocates non-empty cells!
 *  Optimized for zoom simulations by creating spatial bins only over the
 *  gas-occupied region, dramatically reducing memory usage and improving
 *  performance compared to full-box spatial hashing.
 *
 *  Key features:
 *  - Automatic detection of gas extent (wherever PartType0 exists)
 *  - Adaptive cell sizing based on particle count and search radius
 *  - Efficient neighbor finding with minimal empty cell overhead
 *  - Automatically tracks gas as halo drifts/evolves
 *  - MPI-synchronized bounding box detection
 *  - SPARSE storage: only non-empty cells consume memory
 */

#ifndef SPATIAL_HASH_ZOOM_H
#define SPATIAL_HASH_ZOOM_H

#include <vector>
#include <unordered_map>  // ← NEW: For sparse storage
#include <algorithm>
#include <cmath>
#include <mpi.h>
#include "../data/allvars.h"
#include "../data/simparticles.h"

// Configuration constants for spatial hash behavior
struct spatial_hash_config {
  // Cell count constraints
  static constexpr int MIN_CELLS_PER_DIM = 8;       // Minimum grid resolution
  static constexpr int MAX_CELLS_PER_DIM = 768;     // Maximum grid resolution
  static constexpr int MAX_TOTAL_CELLS = 452984832;  // 768^3 max
  
  // Cell sizing parameters
  static constexpr int TARGET_PARTICLES_PER_CELL = 32;  // Optimal particles per cell
  static constexpr double CELL_SIZE_SAFETY_FACTOR = 2.5; // Cells must be ≥2.5× search radius
  
  // Bounding box padding (20% extra space around gas extent)
  static constexpr double BBOX_PADDING_FACTOR = 1.2;
};

/**
 * Zoom-aware spatial hash structure (SPARSE VERSION)
 * 
 * Creates a 3D grid of cells only over the region where gas exists,
 * enabling fast O(1) neighbor finding for feedback and dust operations.
 * Only non-empty cells are stored in memory (sparse hash table).
 */
struct spatial_hash_zoom {
  // Grid parameters
  int n_cells_per_dim;      // Number of cells per dimension
  double cell_size;         // Physical size of each cell
  double box_size;          // Full simulation box size (for periodic boundaries)
  
  // Bounding box for the spatial hash (gas-occupied region only)
  double bbox_min[3];       // Minimum coordinates of bbox
  double bbox_max[3];       // Maximum coordinates of bbox
  double bbox_size[3];      // Size of bbox in each dimension
  
  // Particle tracking
  int total_gas_particles;       // Total gas particles in simulation
  
  // SPARSE cell storage: only non-empty cells exist in map!
  // Key = cell_index, Value = list of gas particle indices
  std::unordered_map<int, std::vector<int>> cells;
  
  bool is_built;
  
  spatial_hash_zoom() : n_cells_per_dim(0), cell_size(0), box_size(0), 
                        total_gas_particles(0), is_built(false) {
    for(int d = 0; d < 3; d++) {
      bbox_min[d] = 0;
      bbox_max[d] = 0;
      bbox_size[d] = 0;
    }
  }
  
  /**
   * Calculate optimal number of cells based on resolution and search radius
   */
  int calculate_optimal_cells(int n_particles, double search_radius, double softening) {
    // Method 1: Particle-based (target ~32 particles per cell)
    int n_from_particles = (int)std::cbrt((double)n_particles /
                                     (double)spatial_hash_config::TARGET_PARTICLES_PER_CELL);
    
    // Method 2: Search radius constraint (cells must be ≥ 2.5× search radius)
    double typical_bbox_size = std::cbrt(bbox_size[0] * bbox_size[1] * bbox_size[2]);
    double max_cell_size = search_radius * spatial_hash_config::CELL_SIZE_SAFETY_FACTOR;
    int n_from_search = (int)std::ceil(typical_bbox_size / max_cell_size);
    
    // Method 3: Softening-based (cells ~4× softening for good resolution)
    int n_from_softening = (int)std::ceil(typical_bbox_size / (4.0 * softening));
    
    // Take the maximum (most conservative choice)
    int n_cells = std::max({n_from_particles, n_from_search, n_from_softening});
    
    // Apply bounds
    n_cells = std::max(spatial_hash_config::MIN_CELLS_PER_DIM, n_cells);
    n_cells = std::min(spatial_hash_config::MAX_CELLS_PER_DIM, n_cells);
    
    // Check total cell count doesn't exceed memory limit
    long long total_cells = (long long)n_cells * n_cells * n_cells;
    if(total_cells > spatial_hash_config::MAX_TOTAL_CELLS) {
      n_cells = (int)std::cbrt(spatial_hash_config::MAX_TOTAL_CELLS);
    }
    
    return n_cells;
  }
  
  /**
   * Detect gas extent across all MPI tasks (MPI-synchronized)
   */
  void detect_gas_extent_collective(simparticles *Sp, int local_gas) {
    // Local extrema
    double local_min[3], local_max[3];

    if(local_gas > 0) {
      for(int d = 0; d < 3; d++) {
        local_min[d] =  1e30;
        local_max[d] = -1e30;
      }

      for(int i = 0; i < Sp->NumGas; i++) {
        double pos[3];
        Sp->intpos_to_pos(Sp->P[i].IntPos, pos);
        for(int d = 0; d < 3; d++) {
          if(pos[d] < local_min[d]) local_min[d] = pos[d];
          if(pos[d] > local_max[d]) local_max[d] = pos[d];
        }
      }
    } else {
      // Neutral elements for MIN/MAX reductions
      for(int d = 0; d < 3; d++) {
        local_min[d] =  1e30;   // Won't win MIN unless everyone is empty
        local_max[d] = -1e30;   // Won't win MAX unless everyone is empty
      }
    }

    // MPI synchronization: get global min/max
    double global_min[3], global_max[3];
    MPI_Allreduce(local_min, global_min, 3, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(local_max, global_max, 3, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    for(int d = 0; d < 3; d++) {
      bbox_min[d] = global_min[d];
      bbox_max[d] = global_max[d];
    }

    // Apply padding
    for(int d = 0; d < 3; d++) {
      double center = 0.5 * (bbox_min[d] + bbox_max[d]);
      double half_width = 0.5 * (bbox_max[d] - bbox_min[d]);
      half_width *= spatial_hash_config::BBOX_PADDING_FACTOR;
      bbox_min[d] = center - half_width;
      bbox_max[d] = center + half_width;
      bbox_size[d] = bbox_max[d] - bbox_min[d];
    }
  }
  
  /**
   * Convert 3D position to flat cell index
   */
  int pos_to_cell_index(double pos[3]) const {
    int ix = (int)std::floor((pos[0] - bbox_min[0]) / cell_size);
    int iy = (int)std::floor((pos[1] - bbox_min[1]) / cell_size);
    int iz = (int)std::floor((pos[2] - bbox_min[2]) / cell_size);
    
    // Clamp to valid range
    ix = std::max(0, std::min(n_cells_per_dim - 1, ix));
    iy = std::max(0, std::min(n_cells_per_dim - 1, iy));
    iz = std::max(0, std::min(n_cells_per_dim - 1, iz));
    
    return ix + n_cells_per_dim * (iy + n_cells_per_dim * iz);
  }
  
  /**
   * Build the spatial hash (SPARSE VERSION - only creates non-empty cells!)
   */
  void build(simparticles *Sp, double max_search_radius, double softening) {
    box_size = All.BoxSize;

    // Count local and global gas particles
    int local_gas = Sp->NumGas;
    int global_gas = 0;
    MPI_Allreduce(&local_gas, &global_gas, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    total_gas_particles = global_gas;

    if(global_gas == 0) {
      is_built = false;
      cells.clear();
      return;
    }

    // Detect where gas actually is (MPI-synchronized)
    detect_gas_extent_collective(Sp, local_gas);

    // Calculate optimal cell count using GLOBAL particle count
    n_cells_per_dim = calculate_optimal_cells(total_gas_particles, max_search_radius, softening);

    double max_bbox_size = std::max({bbox_size[0], bbox_size[1], bbox_size[2]});
    cell_size = max_bbox_size / n_cells_per_dim;

    // SPARSE: Just clear, NO resize! Cells auto-created on first access
    cells.clear();

    // Bin local gas particles into cells
    int particles_binned = 0;
    for(int i = 0; i < Sp->NumGas; i++) {
      double pos[3];
      Sp->intpos_to_pos(Sp->P[i].IntPos, pos);

      // Check if inside bbox
      bool inside = true;
      for(int d = 0; d < 3; d++) {
        if(pos[d] < bbox_min[d] || pos[d] > bbox_max[d]) { 
          inside = false; 
          break; 
        }
      }

      if(inside) {
        int cell_idx = pos_to_cell_index(pos);
        cells[cell_idx].push_back(i);  // Auto-creates cell if doesn't exist
        particles_binned++;
      }
    }
    
    is_built = true;
    
    if(All.ThisTask == 0) {
      double volume_ratio = (bbox_size[0] * bbox_size[1] * bbox_size[2]) /
                           (box_size * box_size * box_size);
      long long theoretical_cells = (long long)n_cells_per_dim * n_cells_per_dim * n_cells_per_dim;
      printf("[SPATIAL_HASH_ZOOM] Hash built: %d^3 cells, cell_size=%.3f kpc\n",
             n_cells_per_dim, cell_size);
      printf("[SPATIAL_HASH_ZOOM] Sparse storage: %zu/%lld cells allocated (%.1f%%)\n",
             cells.size(), theoretical_cells, 100.0 * cells.size() / theoretical_cells);
      printf("[SPATIAL_HASH_ZOOM] BBox volume: %.1f%% of full box\n", 100.0 * volume_ratio);
      printf("[SPATIAL_HASH_ZOOM] Memory saved: %.1f%% (vs full box hash)\n", 
             100.0 * (1.0 - volume_ratio));
    }
  }
  
  /**
   * Find nearest gas particle to a given particle (SPARSE VERSION)
   */
  int find_nearest_gas_particle(simparticles *Sp, int idx, double max_search_radius, 
                                 double *out_distance) const {
    if(!is_built) return -1;
    
    double pos[3];
    Sp->intpos_to_pos(Sp->P[idx].IntPos, pos);
    
    // Check if search center is inside bbox
    bool inside_bbox = true;
    for(int d = 0; d < 3; d++) {
      if(pos[d] < bbox_min[d] || pos[d] > bbox_max[d]) {
        inside_bbox = false;
        break;
      }
    }
    
    if(!inside_bbox) {
      if(out_distance) *out_distance = -1.0;
      return -1;
    }
    
    // Determine how many cells we need to search
    int n_cells_to_search = (int)std::ceil(max_search_radius / cell_size) + 1;
    
    // Get home cell
    int home_ix = (int)std::floor((pos[0] - bbox_min[0]) / cell_size);
    int home_iy = (int)std::floor((pos[1] - bbox_min[1]) / cell_size);
    int home_iz = (int)std::floor((pos[2] - bbox_min[2]) / cell_size);
    
    double min_r2 = max_search_radius * max_search_radius;
    int nearest_idx = -1;
    
    // Search nearby cells
    for(int dix = -n_cells_to_search; dix <= n_cells_to_search; dix++) {
      for(int diy = -n_cells_to_search; diy <= n_cells_to_search; diy++) {
        for(int diz = -n_cells_to_search; diz <= n_cells_to_search; diz++) {
          int ix = home_ix + dix;
          int iy = home_iy + diy;
          int iz = home_iz + diz;
          
          // Skip cells outside grid
          if(ix < 0 || ix >= n_cells_per_dim) continue;
          if(iy < 0 || iy >= n_cells_per_dim) continue;
          if(iz < 0 || iz >= n_cells_per_dim) continue;
          
          int cell_idx = ix + n_cells_per_dim * (iy + n_cells_per_dim * iz);
          
          // SPARSE: Check if cell exists in map
          auto it = cells.find(cell_idx);
          if(it == cells.end()) continue;  // Cell doesn't exist = empty

          // Search particles in this cell
          for(int gas_idx : it->second) {  // Access via iterator
            if(gas_idx == idx) continue;  // Skip self
            
            double dxyz[3];
            Sp->nearest_image_intpos_to_pos(Sp->P[gas_idx].IntPos, Sp->P[idx].IntPos, dxyz);
            double r2 = dxyz[0]*dxyz[0] + dxyz[1]*dxyz[1] + dxyz[2]*dxyz[2];
            
            if(r2 < min_r2) {
              min_r2 = r2;
              nearest_idx = gas_idx;
            }
          }
        }
      }
    }
    
    if(out_distance) {
      *out_distance = (nearest_idx >= 0) ? std::sqrt(min_r2) : -1.0;
    }
    
    return nearest_idx;
  }
  
  /**
   * Find all gas neighbors within a given radius (SPARSE VERSION)
   */
  void find_neighbors(simparticles *Sp, int idx, double search_radius,
                     int *neighbor_indices, double *neighbor_distances,
                     int *n_neighbors, int max_neighbors) const {
    *n_neighbors = 0;
    
    if(!is_built) return;
    
    double pos[3];
    Sp->intpos_to_pos(Sp->P[idx].IntPos, pos);
    
    // Check if search center is inside bbox
    bool inside_bbox = true;
    for(int d = 0; d < 3; d++) {
      if(pos[d] < bbox_min[d] || pos[d] > bbox_max[d]) {
        inside_bbox = false;
        break;
      }
    }
    
    if(!inside_bbox) {
      return;  // No neighbors if outside bbox
    }
    
    // Determine how many cells we need to search
    if(cell_size <= 0 || !std::isfinite(cell_size)) return;
    int n_cells_to_search = (int)std::ceil(search_radius / cell_size) + 1;
    n_cells_to_search = std::min(n_cells_to_search, n_cells_per_dim);
    
    // Get home cell
    int home_ix = (int)std::floor((pos[0] - bbox_min[0]) / cell_size);
    int home_iy = (int)std::floor((pos[1] - bbox_min[1]) / cell_size);
    int home_iz = (int)std::floor((pos[2] - bbox_min[2]) / cell_size);
    
    double search_radius2 = search_radius * search_radius;
    int count = 0;
    
    // Search nearby cells
    for(int dix = -n_cells_to_search; dix <= n_cells_to_search; dix++) {
      for(int diy = -n_cells_to_search; diy <= n_cells_to_search; diy++) {
        for(int diz = -n_cells_to_search; diz <= n_cells_to_search; diz++) {
          int ix = home_ix + dix;
          int iy = home_iy + diy;
          int iz = home_iz + diz;
          
          // Skip cells outside grid
          if(ix < 0 || ix >= n_cells_per_dim) continue;
          if(iy < 0 || iy >= n_cells_per_dim) continue;
          if(iz < 0 || iz >= n_cells_per_dim) continue;
          
          int cell_idx = ix + n_cells_per_dim * (iy + n_cells_per_dim * iz);
          
          // SPARSE: Check if cell exists in map
          auto it = cells.find(cell_idx);
          if(it == cells.end()) continue;  // Cell doesn't exist = empty
          
          // Search particles in this cell
          for(int gas_idx : it->second) {  // Access via iterator
            if(gas_idx == idx) continue;  // Skip self
            
            double dxyz[3];
            Sp->nearest_image_intpos_to_pos(Sp->P[gas_idx].IntPos, Sp->P[idx].IntPos, dxyz);
            double r2 = dxyz[0]*dxyz[0] + dxyz[1]*dxyz[1] + dxyz[2]*dxyz[2];
            
            if(r2 <= search_radius2) {
              if(count < max_neighbors) {
                neighbor_indices[count] = gas_idx;
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

  /**
   * Print statistics about the spatial hash (SPARSE VERSION)
   */
  void print_stats() const {
    if(!is_built || All.ThisTask != 0) return;
    
    long long theoretical_cells = (long long)n_cells_per_dim * n_cells_per_dim * n_cells_per_dim;
    int allocated_cells = cells.size();
    int empty_cells = theoretical_cells - allocated_cells;  // Unallocated = empty
    int max_particles = 0;
    long long total_particles = 0;
    
    for(const auto &kv : cells) {  // Iterate over map entries
      int n = kv.second.size();
      total_particles += n;
      max_particles = std::max(max_particles, n);
    }
    
    printf("[SPATIAL_HASH_ZOOM] Statistics:\n");
    printf("  Bounding box: [%.1f,%.1f] × [%.1f,%.1f] × [%.1f,%.1f] kpc\n",
           bbox_min[0], bbox_max[0], bbox_min[1], bbox_max[1], bbox_min[2], bbox_max[2]);
    printf("  Total cells: %lld (%d^3)\n", theoretical_cells, n_cells_per_dim);
    printf("  Cell size: %.3f kpc\n", cell_size);
    printf("  Allocated cells: %d (%.3f%%)\n", allocated_cells, 
           100.0 * allocated_cells / theoretical_cells);
    printf("  Empty cells: %d (%.1f%%)\n", empty_cells, 100.0 * empty_cells / theoretical_cells);
    printf("  Max particles in cell: %d\n", max_particles);
    printf("  Avg particles per allocated cell: %.1f\n", 
           (double)total_particles / std::max(1, allocated_cells));
  }
};

#endif // SPATIAL_HASH_ZOOM_H