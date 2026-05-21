/*! \file spatial_hash_zoom.h
 *
 *  \brief Zoom-aware spatial hash for fast neighbor finding in Gadget-4.
 *
 *  Supports gas, star, and dust hash instances depending on which particle
 *  type is passed to build(). Each call site maintains its own instance.
 *
 *  SPARSE IMPLEMENTATION — Only allocates non-empty cells!
 *  Optimized for zoom simulations by creating spatial bins only over the
 *  zoom-region particle extent, dramatically reducing memory usage compared
 *  to full-box hashing.
 *
 *  ── KEY FEATURES ─────────────────────────────────────────────────────────
 *
 *  - Automatic detection of zoom-region extent via mass threshold filtering
 *    (excludes background low-res DM particles whose mass is >> zoom mass)
 *  - Adaptive cell sizing: takes the maximum of three criteria (particle
 *    count, search radius, softening) with a guard preventing the softening
 *    criterion from dominating when softening << search radius (e.g. dust)
 *  - Per-call cell count override: callers pass max_cells_override to cap
 *    grid resolution independently of the global MAX_CELLS_PER_DIM backstop
 *  - Efficient neighbor finding with O(1) cell lookup via unordered_map
 *  - MPI-synchronized bounding box (uses Gadget Communicator, never MPI_COMM_WORLD)
 *  - long long cell indices for correctness at grid sizes > 1290^3
 *
 *  ── CELL SIZING LOGIC ────────────────────────────────────────────────────
 *
 *  calculate_optimal_cells() takes the maximum of:
 *    Method 1 — particle count: n = cbrt(N / TARGET_PARTICLES_PER_CELL)
 *    Method 2 — search radius:  n = bbox / (search_radius × SAFETY_FACTOR)
 *    Method 3 — softening:      n = bbox / (4 × softening)
 *
 *  Method 3 is guarded: if n_from_softening > 4 × n_from_search, it is
 *  clamped to n_from_search. Without this guard, dust superparticles
 *  (softening ~ 0.1 kpc, search_radius ~ 5 kpc) would produce
 *  n_from_softening ~ 25000, pinning every dust hash build to
 *  MAX_CELLS_PER_DIM regardless of how many dust particles exist.
 *  This was the root cause of the pathological 4096^3 dust hash builds
 *  introduced in commit 0ffc292 (Apr 15) that caused the z~2.5 slowdown.
 *
 *  ── PER-CALL OVERRIDE ────────────────────────────────────────────────────
 *
 *  build() accepts an optional max_cells_override argument (default 0 = no
 *  override). When non-zero, n_cells_per_dim is capped at this value after
 *  calculate_optimal_cells() runs. Recommended values at 2048^3:
 *    gas hash:  768  (~13 kpc cells, sufficient for feedback search radii)
 *    star hash: 768  (same reasoning)
 *    dust hash: 512  (dust search radii are shorter; fewer cells needed)
 *  MAX_CELLS_PER_DIM = 1024 acts as a global backstop for call sites that
 *  omit the override.
 *
 *  ── BACKGROUND PARTICLE FILTERING ────────────────────────────────────────
 *
 *  detect_extent_collective() computes a mass threshold = 8 × global minimum
 *  particle mass of the requested type. Particles heavier than this are
 *  treated as background low-res DM and excluded from both the bbox and the
 *  cell population. This keeps the bbox tight around the zoom region.
 *
 *  For dust (type 6) this filter is disabled (threshold set to 1e30) because:
 *    - There are no background dust particles to exclude
 *    - Dust masses span many orders of magnitude due to growth/destruction,
 *      so global_min_mass can be near DUST_MASS_TO_DESTROY ~ 1e-30, making
 *      the 8× threshold essentially zero and incorrectly excluding real grains
 *
 *  ── MPI COMMUNICATOR ─────────────────────────────────────────────────────
 *
 *  All collective operations use the MPI_Comm passed explicitly by the caller.
 *  Never use MPI_COMM_WORLD — on multi-node Stampede runs, MPI_COMM_WORLD
 *  conflicts with the shmem I/O handler and causes deadlocks at Step=0.
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

// ── Global configuration constants ───────────────────────────────────────────
// These are backstops; per-call overrides (max_cells_override in build()) are
// the primary mechanism for controlling grid resolution at each call site.
// MAX_CELLS_PER_DIM should be set to the highest per-call override plus modest
// headroom — it exists to catch future call sites that omit the override.
struct spatial_hash_config {
  static constexpr int       MIN_CELLS_PER_DIM         = 8;
  static constexpr int       MAX_CELLS_PER_DIM         = 1024;     // backstop only
  static constexpr long long MAX_TOTAL_CELLS            = 1073741824LL;  // 1024^3
  static constexpr int       TARGET_PARTICLES_PER_CELL = 32;
  static constexpr double    CELL_SIZE_SAFETY_FACTOR   = 2.5;
  static constexpr double    BBOX_PADDING_FACTOR       = 1.2;
};


/**
 * Zoom-aware sparse spatial hash.
 *
 * One instance per particle type (gas_hash, star_hash, dust_hash in feedback.cc).
 * Build by calling build(); query with find_nearest_particle() or find_neighbors().
 * All collective methods require Gadget's internal Communicator, not MPI_COMM_WORLD.
 */
struct spatial_hash_zoom {

  // ── Grid state ─────────────────────────────────────────────────────────────
  int    n_cells_per_dim;   // cells per dimension after build()
  double cell_size;         // physical cell width (kpc)
  double box_size;          // full simulation box side length

  // ── Bounding box of zoom-region particles ──────────────────────────────────
  double bbox_min[3];
  double bbox_max[3];
  double bbox_size[3];

  // ── Particle accounting ────────────────────────────────────────────────────
  int total_particles;      // global count of particles stored in this hash

  // Mass threshold used to separate zoom-region particles from background
  // low-res DM. Set by detect_extent_collective(), reused by build() so that
  // both the bbox and the cell population use the same filter.
  // Set to 1e30 for dust (type 6) — see file header for explanation.
  double zoom_mass_threshold;

  // ── Sparse cell storage ────────────────────────────────────────────────────
  // key   = flat cell index (long long required for grids > 1290^3)
  // value = vector of local particle indices falling in that cell
  std::unordered_map<long long, std::vector<int>> cells;

  bool is_built;

  // ── Constructor ────────────────────────────────────────────────────────────
  spatial_hash_zoom() : n_cells_per_dim(0), cell_size(0), box_size(0),
                        total_particles(0), zoom_mass_threshold(1e30),
                        is_built(false)
  {
    for(int d = 0; d < 3; d++)
      bbox_min[d] = bbox_max[d] = bbox_size[d] = 0.0;
  }


  // ──────────────────────────────────────────────────────────────────────────
  // calculate_optimal_cells()
  //
  // Determine the optimal number of cells per dimension.
  //
  // Takes the maximum of three criteria, then applies bounds:
  //   Method 1 — particle density: n = cbrt(N / TARGET_PARTICLES_PER_CELL)
  //   Method 2 — search radius:    n = ceil(bbox / (search_radius × SAFETY))
  //   Method 3 — softening:        n = ceil(bbox / (4 × softening))
  //              GUARDED: capped at 4 × n_from_search to prevent tiny
  //              softening values (e.g. dust ~ 0.1 kpc) from dominating.
  //              See file header for full explanation.
  //
  // The result is then clamped to [MIN_CELLS_PER_DIM, MAX_CELLS_PER_DIM].
  // The per-call max_cells_override in build() is applied afterward and
  // takes precedence over MAX_CELLS_PER_DIM.
  // ──────────────────────────────────────────────────────────────────────────
  int calculate_optimal_cells(int n_particles, double search_radius, double softening)
  {
    // Method 1: target ~TARGET_PARTICLES_PER_CELL particles per cell
    int n_from_particles = (int)std::cbrt((double)n_particles /
                                          (double)spatial_hash_config::TARGET_PARTICLES_PER_CELL);

    // Method 2: cells must be no larger than search_radius × SAFETY_FACTOR
    double typical_bbox  = std::cbrt(bbox_size[0] * bbox_size[1] * bbox_size[2]);
    double max_cell_size = search_radius * spatial_hash_config::CELL_SIZE_SAFETY_FACTOR;
    int n_from_search    = (int)std::ceil(typical_bbox / max_cell_size);

    // Method 3: cells ~ 4× softening for adequate spatial resolution.
    // GUARD: if softening << search_radius (e.g. dust: 0.1 kpc vs 5 kpc),
    // n_from_softening ~ 25000 which would pin the hash to MAX_CELLS_PER_DIM
    // regardless of particle count, producing pathological builds with only
    // 1 allocated cell out of billions. Cap at 4 × n_from_search — beyond
    // that, softening is already sub-cell and contributes no additional benefit.
    int n_from_softening = (int)std::ceil(typical_bbox / (4.0 * softening));
    if(n_from_softening > 4 * n_from_search)
      n_from_softening = n_from_search;

    int n_cells = std::max({n_from_particles, n_from_search, n_from_softening});
    n_cells = std::max(spatial_hash_config::MIN_CELLS_PER_DIM, n_cells);
    n_cells = std::min(spatial_hash_config::MAX_CELLS_PER_DIM, n_cells);

    // Final check: total cell count must not exceed MAX_TOTAL_CELLS
    long long total_cells = (long long)n_cells * n_cells * n_cells;
    if(total_cells > spatial_hash_config::MAX_TOTAL_CELLS)
      n_cells = (int)std::cbrt((double)spatial_hash_config::MAX_TOTAL_CELLS);

    return n_cells;
  }


  // ──────────────────────────────────────────────────────────────────────────
  // detect_extent_collective()
  //
  // Compute the bounding box of zoom-region particles across all MPI tasks.
  // Also sets zoom_mass_threshold, which build() reuses to populate cells
  // with the same filter applied here, keeping bbox and cell contents consistent.
  //
  // Excludes from both bbox and cell population:
  //   1. Background low-res particles: mass > 8 × global minimum particle mass.
  //      In zoom sims these span the full box; including them would inflate the
  //      bbox to ~100% of box volume, producing a very coarse cell size.
  //      (Not applied for dust type 6 — see file header.)
  //   2. Corrupted particles: non-finite or out-of-box positions (e.g. from
  //      domain-exchange bugs that zero DustP[] positions).
  //
  // comm must be Gadget's internal Communicator, NOT MPI_COMM_WORLD.
  // ──────────────────────────────────────────────────────────────────────────
  void detect_extent_collective(simparticles *Sp, int part_type, MPI_Comm comm)
  {
    double local_min[3], local_max[3];
    for(int d = 0; d < 3; d++) { local_min[d] = 1e30; local_max[d] = -1e30; }

    // ── Step 1: mass threshold for background particle exclusion ─────────────
    double local_min_mass = 1e30;
    int n_scan_mass = (part_type == 0) ? Sp->NumGas : Sp->NumPart;
    for(int i = 0; i < n_scan_mass; i++) {
      if(Sp->P[i].getType() != part_type) continue;
      double m = Sp->P[i].getMass();
      if(m > 0.0 && std::isfinite(m) && m < local_min_mass)
        local_min_mass = m;
    }
    double global_min_mass = local_min_mass;
    MPI_Allreduce(&local_min_mass, &global_min_mass, 1, MPI_DOUBLE, MPI_MIN, comm);

    // 8× = one DM refinement level; robustly separates zoom from background
    // while tolerating natural mass scatter within the zoom region.
    zoom_mass_threshold = (global_min_mass < 1e29) ? global_min_mass * 8.0 : 1e30;

    // For dust (type 6): no background dust particles exist, and dust masses
    // span many orders of magnitude due to growth/destruction. The 8× threshold
    // could be near zero and incorrectly exclude real grains. Disable it.
    if(part_type == 6) zoom_mass_threshold = 1e30;

    // ── Step 2: bbox scan over zoom-region particles only ────────────────────
    int n_scan = (part_type == 0) ? Sp->NumGas : Sp->NumPart;
    for(int i = 0; i < n_scan; i++) {
      if(Sp->P[i].getType() != part_type) continue;
      if(Sp->P[i].getMass() > zoom_mass_threshold) continue;  // background

      double pos[3];
      Sp->intpos_to_pos(Sp->P[i].IntPos, pos);

      // Exclude corrupted/out-of-box positions
      bool valid = true;
      for(int d = 0; d < 3; d++)
        if(!std::isfinite(pos[d]) || pos[d] < 0.0 || pos[d] > box_size)
          { valid = false; break; }
      if(!valid) continue;

      for(int d = 0; d < 3; d++) {
        if(pos[d] < local_min[d]) local_min[d] = pos[d];
        if(pos[d] > local_max[d]) local_max[d] = pos[d];
      }
    }

    // ── Step 3: global reduction and bbox padding ────────────────────────────
    double global_min[3], global_max[3];
    MPI_Allreduce(local_min, global_min, 3, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(local_max, global_max, 3, MPI_DOUBLE, MPI_MAX, comm);

    for(int d = 0; d < 3; d++) {
      bbox_min[d] = global_min[d];
      bbox_max[d] = global_max[d];
    }

    // Expand by BBOX_PADDING_FACTOR so particles near the edges of the zoom
    // region are not missed due to the 1.2× margin.
    for(int d = 0; d < 3; d++) {
      double center     = 0.5 * (bbox_min[d] + bbox_max[d]);
      double half_width = 0.5 * (bbox_max[d] - bbox_min[d]) *
                          spatial_hash_config::BBOX_PADDING_FACTOR;
      bbox_min[d]  = center - half_width;
      bbox_max[d]  = center + half_width;
      bbox_size[d] = bbox_max[d] - bbox_min[d];
    }
  }


  // ──────────────────────────────────────────────────────────────────────────
  // pos_to_cell_index()
  //
  // Map a 3D physical position to a flat cell index.
  // Positions outside the bbox are clamped to the nearest edge cell.
  // Returns long long to handle grids larger than 1290^3 without overflow.
  // ──────────────────────────────────────────────────────────────────────────
  long long pos_to_cell_index(double pos[3]) const
  {
    int ix = (int)std::floor((pos[0] - bbox_min[0]) / cell_size);
    int iy = (int)std::floor((pos[1] - bbox_min[1]) / cell_size);
    int iz = (int)std::floor((pos[2] - bbox_min[2]) / cell_size);

    ix = std::max(0, std::min(n_cells_per_dim - 1, ix));
    iy = std::max(0, std::min(n_cells_per_dim - 1, iy));
    iz = std::max(0, std::min(n_cells_per_dim - 1, iz));

    return (long long)ix + n_cells_per_dim * (iy + (long long)n_cells_per_dim * iz);
  }


  // ──────────────────────────────────────────────────────────────────────────
  // build()
  //
  // Build the spatial hash for all local particles of part_type.
  // Must be called collectively — all tasks must call it together.
  //
  // Parameters
  // ----------
  // max_search_radius  : maximum radius that will ever be passed to
  //                      find_nearest_particle / find_neighbors. Used by
  //                      Method 2 in calculate_optimal_cells().
  // softening          : force softening length for this particle type. Used
  //                      by Method 3 in calculate_optimal_cells().
  // comm               : Gadget's internal Communicator (NOT MPI_COMM_WORLD)
  // part_type          : Gadget particle type (0=gas, 4=stars, 6=dust)
  // max_cells_override : if > 0, caps n_cells_per_dim to this value after
  //                      calculate_optimal_cells() runs. Takes precedence over
  //                      MAX_CELLS_PER_DIM. Recommended values at 2048^3:
  //                        gas/star hash: 768   dust hash: 512
  //                      Set to 0 to use only the global MAX_CELLS_PER_DIM.
  // ──────────────────────────────────────────────────────────────────────────
  void build(simparticles *Sp, double max_search_radius, double softening,
             MPI_Comm comm, int part_type = 0, int max_cells_override = 0,
             bool preset_bbox = false)
  {
    box_size = All.BoxSize;

    // ── Count local and global particles of the requested type ───────────────
    int local_count = (part_type == 0) ? Sp->NumGas : 0;
    if(part_type != 0) {
      for(int i = 0; i < Sp->NumPart; i++)
        if(Sp->P[i].getType() == part_type) local_count++;
    }

    int global_count = 0;
    MPI_Allreduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM, comm);
    total_particles = global_count;

    if(global_count == 0) { is_built = false; cells.clear(); return; }

    // Sets zoom_mass_threshold and bbox as side-effects, reused when populating cells.
    // Skip when preset_bbox=true — caller has already set bbox_min/max/size and
    // zoom_mass_threshold (e.g. dust hash inheriting bbox from gas hash to prevent
    // escaped grains from inflating the bbox to > 100% of box volume).
    if(!preset_bbox)
      detect_extent_collective(Sp, part_type, comm);

    // ── Determine cell grid dimensions ────────────────────────────────────────
    n_cells_per_dim = calculate_optimal_cells(global_count, max_search_radius, softening);

    // Apply per-call override if provided. This is the primary mechanism for
    // controlling resolution per hash type (gas/star/dust) independently.
    // Without this, the gas/star hashes would grow to ~1800^3 at 2048^3
    // because n_from_search dominates for large bbox / small search radius.
    if(max_cells_override > 0 && n_cells_per_dim > max_cells_override)
      n_cells_per_dim = max_cells_override;

    double max_bbox = std::max({bbox_size[0], bbox_size[1], bbox_size[2]});
    cell_size = max_bbox / n_cells_per_dim;

    // ── Populate cells ────────────────────────────────────────────────────────
    cells.clear();

    for(int i = 0; i < Sp->NumPart; i++) {
      if(Sp->P[i].getType() != part_type) continue;

      // Apply same mass filter as bbox detection for consistency
      if(Sp->P[i].getMass() > zoom_mass_threshold) continue;

      double pos[3];
      Sp->intpos_to_pos(Sp->P[i].IntPos, pos);

      // Skip non-finite positions (corrupted particles)
      bool pos_valid = true;
      for(int d = 0; d < 3; d++)
        if(!std::isfinite(pos[d])) { pos_valid = false; break; }
      if(!pos_valid) continue;

      bool inside = true;
      for(int d = 0; d < 3; d++)
        if(pos[d] < bbox_min[d] || pos[d] > bbox_max[d]) { inside = false; break; }
      if(inside) cells[pos_to_cell_index(pos)].push_back(i);
    }

    is_built = true;

    if(All.ThisTask == 0) {
      double volume_ratio    = (bbox_size[0] * bbox_size[1] * bbox_size[2]) /
                               (box_size * box_size * box_size);
      long long theory_cells = (long long)n_cells_per_dim * n_cells_per_dim * n_cells_per_dim;
      printf("[SPATIAL_HASH_ZOOM] Built: %d^3 cells, cell_size=%.3f kpc\n",
             n_cells_per_dim, cell_size);
      printf("[SPATIAL_HASH_ZOOM] Sparse: %zu/%lld cells allocated (%.1f%%)\n",
             cells.size(), theory_cells, 100.0 * cells.size() / theory_cells);
      printf("[SPATIAL_HASH_ZOOM] BBox: %.1f%% of full box volume\n",
             100.0 * volume_ratio);
    }
  }


  // ──────────────────────────────────────────────────────────────────────────
  // find_nearest_particle()
  //
  // Return the index of the nearest particle within max_search_radius,
  // or -1 if none found. Sets *out_distance if non-NULL.
  //
  // Local operation — no MPI. Works for any particle type stored in this
  // hash instance. Positions outside the bbox are clamped before searching
  // so that grains near (but not beyond) the bbox edge still find neighbours.
  // ──────────────────────────────────────────────────────────────────────────
  int find_nearest_particle(simparticles *Sp, int idx, double max_search_radius,
                             double *out_distance) const
  {
    if(!is_built) return -1;

    double pos[3];
    Sp->intpos_to_pos(Sp->P[idx].IntPos, pos);

    // Clamp to bbox so particles just outside the padded region still search
    double clamped_pos[3];
    for(int d = 0; d < 3; d++)
      clamped_pos[d] = std::max(bbox_min[d], std::min(bbox_max[d], pos[d]));

    int n_search = (int)std::ceil(max_search_radius / cell_size) + 1;
    n_search = std::min(n_search, n_cells_per_dim);

    int home_ix = (int)std::floor((clamped_pos[0] - bbox_min[0]) / cell_size);
    int home_iy = (int)std::floor((clamped_pos[1] - bbox_min[1]) / cell_size);
    int home_iz = (int)std::floor((clamped_pos[2] - bbox_min[2]) / cell_size);

    double min_r2  = max_search_radius * max_search_radius;
    int    nearest = -1;

    for(int dix = -n_search; dix <= n_search; dix++) {
      for(int diy = -n_search; diy <= n_search; diy++) {
        for(int diz = -n_search; diz <= n_search; diz++) {
          int ix = home_ix + dix;
          int iy = home_iy + diy;
          int iz = home_iz + diz;

          if(ix < 0 || ix >= n_cells_per_dim) continue;
          if(iy < 0 || iy >= n_cells_per_dim) continue;
          if(iz < 0 || iz >= n_cells_per_dim) continue;

          long long cell_idx = (long long)ix + n_cells_per_dim *
                               (iy + (long long)n_cells_per_dim * iz);
          auto it = cells.find(cell_idx);
          if(it == cells.end()) continue;

          for(int particle_idx : it->second) {
            if(particle_idx == idx) continue;
            double dxyz[3];
            Sp->nearest_image_intpos_to_pos(Sp->P[particle_idx].IntPos,
                                            Sp->P[idx].IntPos, dxyz);
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


  // ──────────────────────────────────────────────────────────────────────────
  // find_neighbors()
  //
  // Fill neighbor_indices and neighbor_distances with all particles within
  // search_radius of particle idx. Sets *n_neighbors to the count found.
  // Returns early (with *n_neighbors = max_neighbors) if the buffer fills.
  //
  // Local operation — no MPI.
  // ──────────────────────────────────────────────────────────────────────────
  void find_neighbors(simparticles *Sp, int idx, double search_radius,
                      int *neighbor_indices, double *neighbor_distances,
                      int *n_neighbors, int max_neighbors) const
  {
    *n_neighbors = 0;
    if(!is_built) return;
    if(cell_size <= 0 || !std::isfinite(cell_size)) return;

    double pos[3];
    Sp->intpos_to_pos(Sp->P[idx].IntPos, pos);

    // Clamp to bbox (same reasoning as find_nearest_particle)
    double clamped_pos[3];
    for(int d = 0; d < 3; d++)
      clamped_pos[d] = std::max(bbox_min[d], std::min(bbox_max[d], pos[d]));

    int n_search = (int)std::ceil(search_radius / cell_size) + 1;
    n_search = std::min(n_search, n_cells_per_dim);

    int home_ix = (int)std::floor((clamped_pos[0] - bbox_min[0]) / cell_size);
    int home_iy = (int)std::floor((clamped_pos[1] - bbox_min[1]) / cell_size);
    int home_iz = (int)std::floor((clamped_pos[2] - bbox_min[2]) / cell_size);

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

          long long cell_idx = (long long)ix + n_cells_per_dim *
                               (iy + (long long)n_cells_per_dim * iz);
          auto it = cells.find(cell_idx);
          if(it == cells.end()) continue;

          for(int particle_idx : it->second) {
            if(particle_idx == idx) continue;
            double dxyz[3];
            Sp->nearest_image_intpos_to_pos(Sp->P[particle_idx].IntPos,
                                            Sp->P[idx].IntPos, dxyz);
            double r2 = dxyz[0]*dxyz[0] + dxyz[1]*dxyz[1] + dxyz[2]*dxyz[2];
            if(r2 <= search_r2) {
              if(count < max_neighbors) {
                neighbor_indices[count]   = particle_idx;
                neighbor_distances[count] = std::sqrt(r2);
                count++;
              } else {
                *n_neighbors = count;
                return;  // buffer full — caller should increase max_neighbors
              }
            }
          }
        }
      }
    }

    *n_neighbors = count;
  }


  // ──────────────────────────────────────────────────────────────────────────
  // print_stats()
  //
  // Print detailed grid statistics to stdout (task 0 only, no MPI).
  // Useful for verifying cell sizing after a build().
  // ──────────────────────────────────────────────────────────────────────────
  void print_stats() const
  {
    if(!is_built || All.ThisTask != 0) return;

    long long theory_cells  = (long long)n_cells_per_dim * n_cells_per_dim * n_cells_per_dim;
    int       alloc_cells   = (int)cells.size();
    int       max_particles = 0;
    long long total_part    = 0;

    for(const auto &kv : cells) {
      int n = (int)kv.second.size();
      total_part    += n;
      max_particles  = std::max(max_particles, n);
    }

    printf("[SPATIAL_HASH_ZOOM] Statistics:\n");
    printf("  BBox: [%.1f,%.1f] x [%.1f,%.1f] x [%.1f,%.1f] kpc\n",
           bbox_min[0], bbox_max[0], bbox_min[1], bbox_max[1],
           bbox_min[2], bbox_max[2]);
    printf("  Grid: %d^3 = %lld theoretical cells, cell_size=%.3f kpc\n",
           n_cells_per_dim, theory_cells, cell_size);
    printf("  Allocated: %d cells (%.3f%%)\n",
           alloc_cells, 100.0 * alloc_cells / theory_cells);
    printf("  Max particles/cell: %d  |  Avg: %.1f\n",
           max_particles, (double)total_part / std::max(1, alloc_cells));
    printf("  Zoom mass threshold: %.3e Msun (background particles excluded)\n",
           zoom_mass_threshold);
  }
};

#endif // SPATIAL_HASH_ZOOM_H