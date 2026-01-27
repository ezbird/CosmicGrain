/*! \file spatial_hash_improved.h
 *
 *  \brief Improved spatial hash for fast neighbor finding in Gadget-4
 *
 *  This implementation automatically determines optimal cell count based on:
 *  - Simulation resolution (particle count)
 *  - Softening length
 *  - Search radius requirements
 *
 *  Key improvements:
 *  - Adaptive cell sizing based on simulation parameters
 *  - Better memory efficiency for high-resolution runs
 *  - Robust boundary handling
 *  - Comprehensive error checking
 */

#ifndef SPATIAL_HASH_IMPROVED_H
#define SPATIAL_HASH_IMPROVED_H

#include <vector>
#include <algorithm>
#include <cmath>
#include "../data/allvars.h"
#include "../data/simparticles.h"

// Configuration for spatial hash behavior
struct spatial_hash_config {
  // Minimum and maximum allowed cells per dimension
  static constexpr int MIN_CELLS_PER_DIM = 8;
  static constexpr int MAX_CELLS_PER_DIM = 512;
  
  // Target particles per cell (for automatic sizing)
  static constexpr int TARGET_PARTICLES_PER_CELL = 32;
  
  // Safety factor: cells must be at least this many times larger than search radius
  static constexpr double CELL_SIZE_SAFETY_FACTOR = 2.5;
  
  // Maximum memory usage for hash structure (in number of cells)
  static constexpr int MAX_TOTAL_CELLS = 1000000;  // ~1 million cells max
};

//constexpr int spatial_hash_config::MIN_CELLS_PER_DIM;
//constexpr int spatial_hash_config::MAX_CELLS_PER_DIM;
//constexpr int spatial_hash_config::TARGET_PARTICLES_PER_CELL;
//constexpr double spatial_hash_config::CELL_SIZE_SAFETY_FACTOR;
//constexpr int spatial_hash_config::MAX_TOTAL_CELLS;

/**
 * Improved spatial hash structure with adaptive resolution
 */
struct spatial_hash_improved {
  int n_cells_per_dim;      // Number of cells per dimension
  double cell_size;         // Physical size of each cell
  double box_size;          // Simulation box size
  int total_gas_particles;  // Total number of gas particles
  
  // Note, using a unordered_map<CellKey,...> would be more memory efficient (handle empty cells better), could consider for the future
  std::vector<std::vector<int>> cells;  // cells[cell_index] = list of gas particle indices
  
  bool is_built;
  
  spatial_hash_improved() : n_cells_per_dim(0), cell_size(0), box_size(0), 
                           total_gas_particles(0), is_built(false) {}
  
  /**
   * Calculate optimal number of cells per dimension based on simulation parameters
   * 
   * The algorithm considers:
   * 1. Particle count (higher resolution → more cells)
   * 2. Search radius (cells must be large enough to contain neighbors)
   * 3. Softening length (natural length scale of the simulation)
   * 4. Memory constraints (don't create too many cells)
   */
  int calculate_optimal_cells(double max_search_radius, double softening_length, int num_gas) {
    if(box_size <= 0 || num_gas <= 0) {
      printf("[SPATIAL_HASH] ERROR: Invalid parameters (box_size=%g, num_gas=%d)\n", 
             box_size, num_gas);
      return spatial_hash_config::MIN_CELLS_PER_DIM;
    }
    
    // Method 1: Based on particle count
    // Target: spatial_hash_config::TARGET_PARTICLES_PER_CELL particles per cell on average
    // Total cells needed: num_gas / TARGET_PARTICLES_PER_CELL
    // Cells per dimension: cube root of total cells
    int cells_from_particle_count = (int)std::cbrt(
      (double)num_gas / spatial_hash_config::TARGET_PARTICLES_PER_CELL
    );
    
    // Method 2: Based on search radius
    // Cell size must be at least CELL_SIZE_SAFETY_FACTOR * max_search_radius
    // to ensure neighbors are in adjacent cells
    double min_cell_size = spatial_hash_config::CELL_SIZE_SAFETY_FACTOR * max_search_radius;
    int cells_from_search_radius = std::max(1, (int)(box_size / min_cell_size));
    
    // Method 3: Based on softening length
    // Use softening length as a natural scale for the simulation
    // Typically want cells to be a few softening lengths in size
    double cell_size_from_softening = 4.0 * softening_length;  // 4× softening length per cell
    int cells_from_softening = std::max(1, (int)(box_size / cell_size_from_softening));
    
    // Take the maximum to ensure all constraints are satisfied
    int n_cells = std::max({
      cells_from_particle_count,
      cells_from_search_radius,
      cells_from_softening,
      spatial_hash_config::MIN_CELLS_PER_DIM
    });
    
    // Apply upper limit based on memory constraints
    n_cells = std::min(n_cells, spatial_hash_config::MAX_CELLS_PER_DIM);
    
    // Check total cell count doesn't exceed memory limit
    long long total_cells = (long long)n_cells * n_cells * n_cells;
    if(total_cells > spatial_hash_config::MAX_TOTAL_CELLS) {
      // Scale back to stay under memory limit
      n_cells = (int)std::cbrt(spatial_hash_config::MAX_TOTAL_CELLS);
    }
    
    if(All.ThisTask == 0) {
      printf("[SPATIAL_HASH] Calculated optimal cell count:\n");
      printf("  From particle count (%d gas): %d cells/dim\n", 
             num_gas, cells_from_particle_count);
      printf("  From search radius (%g): %d cells/dim\n", 
             max_search_radius, cells_from_search_radius);
      printf("  From softening (%g): %d cells/dim\n", 
             softening_length, cells_from_softening);
      printf("  Final choice: %d cells/dim (%d^3 = %lld total cells)\n", 
             n_cells, n_cells, total_cells);
    }
    
    return n_cells;
  }
  
  /**
   * Build the spatial hash structure
   * 
   * @param Sp Simulation particles
   * @param max_search_radius Maximum distance for neighbor searches
   * @param softening_length Gravitational softening length (for automatic sizing)
   * @param auto_size If true, automatically determine cell count; if false, use provided n_cells
   * @param manual_n_cells Manual cell count (only used if auto_size=false)
   */
  void build(simparticles *Sp, double max_search_radius, double softening_length = -1.0,
             bool auto_size = true, int manual_n_cells = 0) {
    
    box_size = All.BoxSize;
    total_gas_particles = Sp->NumGas;
    
    if(box_size <= 0) {
      printf("[SPATIAL_HASH] ERROR: Invalid BoxSize=%g\n", box_size);
      is_built = false;
      return;
    }
    
    if(total_gas_particles <= 0) {
      printf("[SPATIAL_HASH] WARNING: No gas particles, skipping hash build\n");
      is_built = false;
      return;
    }
    
    // Use softening length if not provided
    if(softening_length <= 0) {
      softening_length = All.ForceSoftening[0];  // Use gas softening
    }
    
    // Determine number of cells
    if(auto_size) {
      n_cells_per_dim = calculate_optimal_cells(max_search_radius, softening_length, 
                                                total_gas_particles);
    } else {
      n_cells_per_dim = std::max(spatial_hash_config::MIN_CELLS_PER_DIM, 
                                 std::min(manual_n_cells, spatial_hash_config::MAX_CELLS_PER_DIM));
      if(All.ThisTask == 0) {
        printf("[SPATIAL_HASH] Using manual cell count: %d cells/dim\n", n_cells_per_dim);
      }
    }
    
    cell_size = box_size / (double)n_cells_per_dim;
    
    /* Verify cell size is appropriate for search radius
    if(cell_size < spatial_hash_config::CELL_SIZE_SAFETY_FACTOR * max_search_radius * 0.75) {
      printf("[SPATIAL_HASH] WARNING: Cell size (%g) is smaller than recommended for search radius (%g)\n",
             cell_size, max_search_radius);
      printf("  Consider using fewer cells or expect more cell checks during neighbor search\n");
    }
    */
   
    int total_cells = n_cells_per_dim * n_cells_per_dim * n_cells_per_dim;
    
    if(All.ThisTask == 0) {
      printf("[FEEDBACK] Building spatial bins: box=%g, cells=%d^3=%d, cell_size=%g, gas=%d, avg=%.1f/cell\n",
            box_size, n_cells_per_dim, total_cells, cell_size, 
            total_gas_particles, (double)total_gas_particles / total_cells);
    }
    
    // Allocate cell storage
    cells.clear();
    cells.resize(total_cells);
    
    // Reserve space to reduce allocations
    int avg_per_cell = std::max(1, total_gas_particles / total_cells);
    for(auto &cell : cells) {
      cell.reserve(avg_per_cell * 2);  // Reserve 2× average for safety
    }
    
    // Bin all gas particles into cells
    int particles_binned = 0;
    int out_of_bounds = 0;
    
    for(int i = 0; i < Sp->NumGas; ++i) {
      double pos[3];
      Sp->intpos_to_pos(Sp->P[i].IntPos, pos);
      
      // Check for invalid positions
      if(!std::isfinite(pos[0]) || !std::isfinite(pos[1]) || !std::isfinite(pos[2])) {
        printf("[SPATIAL_HASH] WARNING: Particle %d has invalid position (%g, %g, %g)\n",
               i, pos[0], pos[1], pos[2]);
        out_of_bounds++;
        continue;
      }
      
      // Apply periodic boundary conditions (wrap to [0, box_size))
      for(int d = 0; d < 3; d++) {
        while(pos[d] < 0) pos[d] += box_size;
        while(pos[d] >= box_size) pos[d] -= box_size;
      }
      
      // Calculate cell indices
      int cx = (int)(pos[0] / cell_size);
      int cy = (int)(pos[1] / cell_size);
      int cz = (int)(pos[2] / cell_size);
      
      // Safety clamp (should not be needed after wrapping, but just in case)
      cx = std::max(0, std::min(cx, n_cells_per_dim - 1));
      cy = std::max(0, std::min(cy, n_cells_per_dim - 1));
      cz = std::max(0, std::min(cz, n_cells_per_dim - 1));
      
      int cell_idx = cx + cy * n_cells_per_dim + cz * n_cells_per_dim * n_cells_per_dim;
      
      // Final bounds check
      if(cell_idx < 0 || cell_idx >= total_cells) {
        printf("[SPATIAL_HASH] ERROR: Invalid cell index %d for particle %d\n", cell_idx, i);
        out_of_bounds++;
        continue;
      }
      
      cells[cell_idx].push_back(i);
      particles_binned++;
    }
    
    if(out_of_bounds > 0) {
      printf("[SPATIAL_HASH] WARNING: %d particles out of bounds or invalid\n", out_of_bounds);
    }
    
    // Calculate statistics
    int max_particles_per_cell = 0;
    int empty_cells = 0;
    for(const auto &cell : cells) {
      if(cell.empty()) empty_cells++;
      max_particles_per_cell = std::max(max_particles_per_cell, (int)cell.size());
    }
    
    if(All.ThisTask == 0) {
      printf("[SPATIAL_HASH] Hash built successfully:\n");
      printf("  Particles binned: %d / %d\n", particles_binned, total_gas_particles);
      printf("  Empty cells: %d / %d (%.1f%%)\n", 
             empty_cells, total_cells, 100.0 * empty_cells / total_cells);
      printf("  Max particles in any cell: %d\n", max_particles_per_cell);
      
      // Memory usage estimate
      size_t mem_bytes = total_cells * sizeof(std::vector<int>) + 
                        particles_binned * sizeof(int);
      printf("  Estimated memory: %.2f MB\n", mem_bytes / (1024.0 * 1024.0));
    }
    
    is_built = true;
  }
  
  /**
   * Get cell index for a position
   */
  int get_cell_index(const double pos[3]) const {
    // Apply periodic wrapping
    double wrapped[3];
    for(int d = 0; d < 3; d++) {
      wrapped[d] = pos[d];
      while(wrapped[d] < 0) wrapped[d] += box_size;
      while(wrapped[d] >= box_size) wrapped[d] -= box_size;
    }
    
    int cx = (int)(wrapped[0] / cell_size);
    int cy = (int)(wrapped[1] / cell_size);
    int cz = (int)(wrapped[2] / cell_size);
    
    cx = std::max(0, std::min(cx, n_cells_per_dim - 1));
    cy = std::max(0, std::min(cy, n_cells_per_dim - 1));
    cz = std::max(0, std::min(cz, n_cells_per_dim - 1));
    
    return cx + cy * n_cells_per_dim + cz * n_cells_per_dim * n_cells_per_dim;
  }
  
  /**
   * Find neighbors within search radius using the spatial hash
   * 
   * This only searches cells that could contain particles within the search radius,
   * dramatically reducing the number of distance calculations needed.
   */
  void find_neighbors(simparticles *Sp, int star_idx, double search_radius,
                     int *ngb_list, double *distances, int *n_ngb, int max_ngb) const {
    
    if(!is_built) {
      printf("[SPATIAL_HASH] ERROR: Hash not built, cannot find neighbors\n");
      *n_ngb = 0;
      return;
    }
    
    *n_ngb = 0;
    
    // Get star position
    double star_pos[3];
    Sp->intpos_to_pos(Sp->P[star_idx].IntPos, star_pos);
    
    // Determine which cells to search
    // We need to check all cells within ceil(search_radius / cell_size) cells
    int n_cell_radius = (int)std::ceil(search_radius / cell_size);
    
    // Get star's cell
    int star_cx = (int)(star_pos[0] / cell_size);
    int star_cy = (int)(star_pos[1] / cell_size);
    int star_cz = (int)(star_pos[2] / cell_size);
    
    const double r2_max = search_radius * search_radius;
    
    // Search neighboring cells
    for(int dcx = -n_cell_radius; dcx <= n_cell_radius; dcx++) {
      for(int dcy = -n_cell_radius; dcy <= n_cell_radius; dcy++) {
        for(int dcz = -n_cell_radius; dcz <= n_cell_radius; dcz++) {
          
          // Get cell indices with periodic wrapping
          int cx = (star_cx + dcx + n_cells_per_dim) % n_cells_per_dim;
          int cy = (star_cy + dcy + n_cells_per_dim) % n_cells_per_dim;
          int cz = (star_cz + dcz + n_cells_per_dim) % n_cells_per_dim;
          
          int cell_idx = cx + cy * n_cells_per_dim + cz * n_cells_per_dim * n_cells_per_dim;
          
          if(cell_idx < 0 || cell_idx >= (int)cells.size()) continue;
          
          // Check all particles in this cell
          for(int gas_idx : cells[cell_idx]) {
            if(gas_idx == star_idx) continue;
            
            // Calculate distance with periodic boundaries
            double d[3];
            Sp->nearest_image_intpos_to_pos(Sp->P[gas_idx].IntPos, 
                                           Sp->P[star_idx].IntPos, d);
            double r2 = d[0]*d[0] + d[1]*d[1] + d[2]*d[2];
            
            if(r2 <= r2_max) {
              if(*n_ngb < max_ngb) {
                ngb_list[*n_ngb] = gas_idx;
                distances[*n_ngb] = std::sqrt(r2);
                (*n_ngb)++;
                
                // Early exit if we have enough neighbors
                if(*n_ngb >= max_ngb) {
                  return;
                }
              }
            }
          }
        }
      }
    }
  }
  
  /**
 * Find the SINGLE nearest gas particle to a given particle
 * More efficient than find_neighbors when you only need the closest one
 * This is used by the dust module to find nearest gas for dust growth, etc.
 * 
 * @param Sp Simulation particles
 * @param particle_idx Index of particle to search from
 * @param max_search_radius Maximum distance to search (in code units)
 * @param nearest_dist Output: distance to nearest particle (can be nullptr)
 * @return Index of nearest gas particle, or -1 if none found
 */
int find_nearest_gas_particle(simparticles *Sp, int particle_idx, 
                               double max_search_radius, double *nearest_dist) const {
  
  if(!is_built) {
    printf("[SPATIAL_HASH] ERROR: Hash not built\n");
    return -1;
  }
  
  double pos[3];
  Sp->intpos_to_pos(Sp->P[particle_idx].IntPos, pos);
  
  int px = (int)(pos[0] / cell_size);
  int py = (int)(pos[1] / cell_size);
  int pz = (int)(pos[2] / cell_size);
  
  int nearest_idx = -1;
  double min_r2 = max_search_radius * max_search_radius;
  
  // Search outward from particle's cell until we're sure we have the nearest
  int max_cell_radius = (int)std::ceil(max_search_radius / cell_size);
  
  for(int radius = 0; radius <= max_cell_radius; radius++) {
    bool found_in_shell = false;
    
    // Search all cells at this radius
    for(int dcx = -radius; dcx <= radius; dcx++) {
      for(int dcy = -radius; dcy <= radius; dcy++) {
        for(int dcz = -radius; dcz <= radius; dcz++) {
          
          // Only check cells on the current shell (not interior)
          if(radius > 0 && std::abs(dcx) < radius && std::abs(dcy) < radius && std::abs(dcz) < radius)
            continue;
          
          int cx = (px + dcx + n_cells_per_dim) % n_cells_per_dim;
          int cy = (py + dcy + n_cells_per_dim) % n_cells_per_dim;
          int cz = (pz + dcz + n_cells_per_dim) % n_cells_per_dim;
          
          int cell_idx = cx + cy * n_cells_per_dim + cz * n_cells_per_dim * n_cells_per_dim;
          
          if(cell_idx < 0 || cell_idx >= (int)cells.size()) continue;
          
          // Check all gas particles in this cell
          for(int gas_idx : cells[cell_idx]) {
            if(gas_idx == particle_idx) continue;  // Skip self
            
            double d[3];
            Sp->nearest_image_intpos_to_pos(Sp->P[gas_idx].IntPos, 
                                           Sp->P[particle_idx].IntPos, d);
            double r2 = d[0]*d[0] + d[1]*d[1] + d[2]*d[2];
            
            if(r2 < min_r2) {
              min_r2 = r2;
              nearest_idx = gas_idx;
              found_in_shell = true;
            }
          }
        }
      }
    }
    
    // Early exit: if we found a neighbor and the next shell can't be closer
    if(nearest_idx >= 0) {
      double best = sqrt(min_r2);
      double lower_bound_next = (radius + 0.5) * cell_size;
      if(best < lower_bound_next) break;
    }

  }
  
  if(nearest_dist != nullptr) {
    *nearest_dist = (nearest_idx >= 0) ? sqrt(min_r2) : -1.0;
  }
  
  return nearest_idx;
}

  /**
   * Print statistics about the spatial hash
   */
  void print_stats() const {
    if(!is_built) {
      printf("[SPATIAL_HASH] Hash not built\n");
      return;
    }
    
    int total_cells = cells.size();
    int empty_cells = 0;
    int max_particles = 0;
    long long total_particles = 0;
    
    std::vector<int> histogram(20, 0);  // Histogram of particles per cell
    
    for(const auto &cell : cells) {
      int n = cell.size();
      total_particles += n;
      if(n == 0) empty_cells++;
      max_particles = std::max(max_particles, n);
      
      // Add to histogram
      int bin = std::min(19, n);
      histogram[bin]++;
    }
    
    printf("[SPATIAL_HASH]  Total cells: %d (%d^3)\n", total_cells, n_cells_per_dim);
    printf("[SPATIAL_HASH]  Cell size: %g (box=%g)\n", cell_size, box_size);
    printf("[SPATIAL_HASH]  Empty cells: %d (%.1f%%)\n", empty_cells, 100.0 * empty_cells / total_cells);
    printf("[SPATIAL_HASH]  Max particles in cell: %d\n", max_particles);
    printf("[SPATIAL_HASH]  Avg particles per non-empty cell: %.1f\n", 
           (double)total_particles / (total_cells - empty_cells));
    
    printf("  Particles per cell histogram:\n");
    for(int i = 0; i < 20; i++) {
      if(histogram[i] > 0) {
        if(i < 19)
          printf("    %2d particles: %6d cells\n", i, histogram[i]);
        else
          printf("    19+ particles: %6d cells\n", histogram[i]);
      }
    }
  }
};




#endif // SPATIAL_HASH_IMPROVED_H