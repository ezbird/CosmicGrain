// ============================================================================
// dust_particle_log.cc
//
// Per-particle event logging for CosmicGrain dust physics analysis.
//
// Records one line per destruction event, capturing enough information
// to reconstruct:
//   - Physical transport (displacement from birth to death)
//   - Grain growth history (radius at death vs. birth size implied by type)
//   - Destruction channel statistics (thermal vs. shock vs. astration)
//   - Radiation pressure diagnostics (grain size, carbon fraction)
//   - ISM phase at death (gas density, scale factor)
//
// ── FILE LAYOUT ──────────────────────────────────────────────────────────────
//
// One file per MPI task, placed in a dedicated subdirectory:
//   <OutputDir>/dust_logs/dust_log_task<N>.txt
//
// The subdirectory is created on RST_BEGIN. On RST_RESUME files are opened
// in append mode so the complete history across restarts is preserved.
//
// After the run, merge all tasks with:
//   cat <OutputDir>/dust_logs/dust_log_task*.txt | grep -v "^#" > all_events.txt
//
// Or in Python:
//   import pandas as pd, glob
//   df = pd.concat([pd.read_csv(f, comment='#', sep=' ', header=None)
//                   for f in glob.glob('dust_logs/dust_log_task*.txt')])
//
// ── LOG COLUMNS (space-separated) ────────────────────────────────────────────
//
//   1   ID              particle ID
//   2   birth_a         scale factor at creation
//   3   event_a         scale factor at this event
//   4   birth_x         birth position x (comoving kpc/h)
//   5   birth_y         birth position y (comoving kpc/h)
//   6   birth_z         birth position z (comoving kpc/h)
//   7   event_x         event position x (comoving kpc/h)
//   8   event_y         event position y (comoving kpc/h)
//   9   event_z         event position z (comoving kpc/h)
//  10   displacement    |event_pos - birth_pos| (comoving kpc/h)
//  11   mass            superparticle mass at event (code units)
//  12   grain_radius    grain radius at event (nm)
//  13   carbon_fraction carbonaceous mass fraction [0,1]
//  14   gas_density     local gas density (code units), 0 if unavailable
//  15   grain_type      0=SNII-silicate  1=AGB-carbon  2=mixed
//  16   event_type      see DUST_EVENT_* constants in dust.h
//
// ── BUFFERING STRATEGY ────────────────────────────────────────────────────────
//
// Each task opens its log with a 4MB userspace buffer (setvbuf _IOFBF).
// This batches writes into large chunks, avoiding per-event syscalls which
// become expensive at high dust counts (~16M particles at z~3).
//
// Explicit fflush is called:
//   - Once after the header is written on fresh runs (to ensure the header
//     survives a crash before any events are logged)
//   - Every FLUSH_INTERVAL events as a crash-safety checkpoint
//   - Once at shutdown in close_dust_particle_log()
//
// The FLUSH_INTERVAL of 10000 events balances crash safety (at most 10000
// events lost) against syscall overhead. At typical destruction rates this
// flushes roughly once per sync-point at z~3.
//
// ── BIRTH POSITION GUARD ──────────────────────────────────────────────────────
//
// BirthPos is zero-initialized by memset in the dust_data struct and set
// explicitly in spawn_dust_particle(). If it is still (0,0,0) the particle
// was either created before BirthPos tracking was added (early-run restarts)
// or suffered domain-exchange corruption that zeroed DustP[].
//
// NOTE: This guard will incorrectly skip grains born near the box origin.
// This is acceptable for Halo569 which sits far from (0,0,0). If the halo
// ever drifts near the origin, replace this with a dedicated bool flag in
// dust_data (e.g. birth_pos_set) initialized to false and set to true in
// spawn_dust_particle().
//
// ============================================================================

#include "gadgetconfig.h"

#ifdef DUST

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <mpi.h>

#include "../data/allvars.h"
#include "../dust/dust.h"

// Number of events between explicit fflush calls (crash-safety checkpoint).
// Balances crash safety (at most FLUSH_INTERVAL events lost) against the
// syscall overhead of frequent flushing. See buffering strategy note above.
static constexpr int FLUSH_INTERVAL = 10000;

// Module-level file handle — one per task, opened in open_dust_particle_log().
static FILE *dust_particle_log = NULL;


// ============================================================================
// open_dust_particle_log()
//
// Call once at simulation startup after All.OutputDir is set.
// On RST_BEGIN: creates the dust_logs/ subdirectory and opens files in
//   write mode ("w"), writing the column header.
// On RST_RESUME: opens files in append mode ("a") so the full event
//   history across restarts is preserved in a single file per task.
//
// Uses a 4MB userspace buffer to reduce syscall frequency on network
// filesystems and local SSDs alike (see buffering strategy note above).
// ============================================================================
void open_dust_particle_log(MPI_Comm Communicator)
{
    // ── Create subdirectory on task 0; all others wait ────────────────────
    if(All.ThisTask == 0)
    {
        char logdir[MAXLEN_PATH];
        snprintf(logdir, MAXLEN_PATH, "%sdust_logs", All.OutputDir);

        if(All.RestartFlag == RST_BEGIN)
        {
            if(mkdir(logdir, 02755) != 0 && errno != EEXIST)
                printf("[DUST_LOG|T=0] WARNING: could not create %s: %s\n",
                       logdir, strerror(errno));
        }
    }
    MPI_Barrier(Communicator);

    // ── Open per-task log file ────────────────────────────────────────────
    char fname[MAXLEN_PATH];
    snprintf(fname, MAXLEN_PATH, "%sdust_logs/dust_log_task%d.txt",
             All.OutputDir, All.ThisTask);

    const char *mode = (All.RestartFlag == RST_BEGIN) ? "w" : "a";
    dust_particle_log = fopen(fname, mode);

    if(dust_particle_log == NULL)
    {
        printf("[DUST_LOG|T=%d] ERROR: could not open %s: %s\n",
               All.ThisTask, fname, strerror(errno));
        return;
    }

    // 4MB userspace buffer — see buffering strategy note in file header.
    setvbuf(dust_particle_log, NULL, _IOFBF, 4 * 1024 * 1024);

    // ── Write column header on fresh runs only ────────────────────────────
    // Skipped on RST_RESUME to avoid duplicate headers in the merged file.
    if(All.RestartFlag == RST_BEGIN)
    {
        fprintf(dust_particle_log,
            "# CosmicGrain particle event log — task %d\n"
            "# One row per destruction event\n"
            "#\n"
            "# Event types:\n"
            "#   %d = thermal sputtering\n"
            "#   %d = SN shock destruction\n"
            "#   %d = shattering below minimum grain size (dissolved to gas)\n"
            "#   %d = astration (incorporated into star)\n"
            "#   %d = sublimation (T_dust > T_sublimate)\n"
            "#   %d = cleanup (corrupted particle removed)\n"
            "#\n"
            "# Columns:\n"
            "#  1  ID              particle ID\n"
            "#  2  birth_a         scale factor at creation\n"
            "#  3  event_a         scale factor at this event\n"
            "#  4  birth_x         birth position x (comoving kpc/h)\n"
            "#  5  birth_y         birth position y (comoving kpc/h)\n"
            "#  6  birth_z         birth position z (comoving kpc/h)\n"
            "#  7  event_x         event position x (comoving kpc/h)\n"
            "#  8  event_y         event position y (comoving kpc/h)\n"
            "#  9  event_z         event position z (comoving kpc/h)\n"
            "# 10  displacement    |event_pos - birth_pos| (comoving kpc/h)\n"
            "# 11  mass            superparticle mass at event (code units)\n"
            "# 12  grain_radius    grain radius at event (nm)\n"
            "# 13  carbon_fraction carbonaceous mass fraction [0,1]\n"
            "# 14  gas_density     local gas density (code units), 0=unavailable\n"
            "# 15  grain_type      0=SNII-silicate 1=AGB-carbon 2=mixed\n"
            "# 16  event_type      see event types above\n"
            "#\n",
            All.ThisTask,
            DUST_EVENT_THERMAL, DUST_EVENT_SHOCK, DUST_EVENT_SHATTERING, DUST_EVENT_ASTRATION,
            DUST_EVENT_SUBLIMATION, DUST_EVENT_CLEANUP);

        // Flush header immediately so it survives a crash before any events
        // are logged. This is the only unconditional flush outside shutdown.
        fflush(dust_particle_log);
    }

    if(All.ThisTask == 0)
        printf("[DUST_LOG] Opened dust_logs/dust_log_task*.txt (mode=%s)\n", mode);
}


// ============================================================================
// close_dust_particle_log()
//
// Flush the userspace buffer and close the file handle. Call from the
// simulation shutdown path to ensure no buffered events are lost.
// Safe to call multiple times (no-ops if already closed).
// ============================================================================
void close_dust_particle_log(void)
{
    if(dust_particle_log != NULL)
    {
        fflush(dust_particle_log);
        fclose(dust_particle_log);
        dust_particle_log = NULL;
    }
}


// ============================================================================
// log_dust_particle_event()
//
// Record one event row for a dust particle. Call immediately before any
// state-altering operation (destruction, etc.) so that the logged values
// reflect the particle's state at the moment of the event.
//
// Parameters
// ----------
// Sp          : simulation particle data
// dust_idx    : local index of the dust particle
// nearest_gas : local index of nearest gas cell (-1 if unavailable)
// event_type  : one of the DUST_EVENT_* constants defined in dust.h
//
// Silently skips:
//   - DUST_EVENT_CLEANUP events (particle state is unreliable by definition)
//   - Particles with BirthPos == (0,0,0) (uninitialized or domain-exchange
//     victims; see birth position guard note in file header)
// ============================================================================
void log_dust_particle_event(simparticles *Sp, int dust_idx,
                              int nearest_gas, int event_type)
{
    if(dust_particle_log == NULL) return;

    // Cleanup events are logged at a coarser level in print_dust_statistics().
    // Individual cleanup rows are not useful since particle state is
    // unreliable at the point cleanup_invalid_dust_particles() fires.
    if(event_type == DUST_EVENT_CLEANUP) return;

    // Skip particles whose birth position was never set. This catches grains
    // created before BirthPos tracking was added and domain-exchange victims
    // whose DustP[] was zeroed. See birth position guard note in file header.
    if(Sp->DustP[dust_idx].BirthPos[0] == 0.0 &&
       Sp->DustP[dust_idx].BirthPos[1] == 0.0 &&
       Sp->DustP[dust_idx].BirthPos[2] == 0.0) return;

    // ── Event position ────────────────────────────────────────────────────
    double event_pos[3];
    Sp->intpos_to_pos(Sp->P[dust_idx].IntPos, event_pos);

    // ── Birth position ────────────────────────────────────────────────────
    double birth_pos[3] = {
        (double)Sp->DustP[dust_idx].BirthPos[0],
        (double)Sp->DustP[dust_idx].BirthPos[1],
        (double)Sp->DustP[dust_idx].BirthPos[2]
    };

    // ── Displacement with periodic boundary wrapping ──────────────────────
    double dx = event_pos[0] - birth_pos[0];
    double dy = event_pos[1] - birth_pos[1];
    double dz = event_pos[2] - birth_pos[2];

    double half = All.BoxSize * 0.5;
    if(dx >  half) dx -= All.BoxSize;
    if(dx < -half) dx += All.BoxSize;
    if(dy >  half) dy -= All.BoxSize;
    if(dy < -half) dy += All.BoxSize;
    if(dz >  half) dz -= All.BoxSize;
    if(dz < -half) dz += All.BoxSize;

    double displacement = sqrt(dx*dx + dy*dy + dz*dz);

    // ── Local gas density (0 if no valid neighbour) ───────────────────────
    // Guard against stale hash returning a converted non-gas particle.
    double gas_density = 0.0;
    if(nearest_gas >= 0 && nearest_gas < Sp->NumGas &&
       Sp->P[nearest_gas].getType() == 0)
        gas_density = Sp->SphP[nearest_gas].Density;

    // ── Write event row ───────────────────────────────────────────────────
    fprintf(dust_particle_log,
        "%lld "             //  1  ID
        "%.6f %.6f "        //  2  birth_a   3  event_a
        "%.4f %.4f %.4f "   //  4-6  birth xyz
        "%.4f %.4f %.4f "   //  7-9  event xyz
        "%.4f "             // 10  displacement
        "%.6e "             // 11  mass
        "%.3f "             // 12  grain_radius (nm)
        "%.4f "             // 13  carbon_fraction
        "%.6e "             // 14  gas_density
        "%d %d\n",          // 15  grain_type   16  event_type
        (long long)Sp->P[dust_idx].ID.get(),
        (double)Sp->P[dust_idx].StellarAge,   // birth scale factor (a at spawn)
        (double)All.Time,                      // event scale factor
        birth_pos[0], birth_pos[1], birth_pos[2],
        event_pos[0],  event_pos[1],  event_pos[2],
        displacement,
        Sp->P[dust_idx].getMass(),
        (double)Sp->DustP[dust_idx].GrainRadius,
        (double)Sp->DustP[dust_idx].CarbonFraction,
        gas_density,
        (int)Sp->DustP[dust_idx].GrainType,
        event_type);

    // ── Periodic crash-safety flush ───────────────────────────────────────
    // The 4MB setvbuf buffer handles normal write batching. This explicit
    // flush fires every FLUSH_INTERVAL events as a checkpoint so that at
    // most FLUSH_INTERVAL events are lost if the run crashes ungracefully.
    // The counter is file-global (not per event-type) which is fine —
    // the only goal is bounding the loss window, not per-channel precision.
    static int event_count = 0;
    if(++event_count % FLUSH_INTERVAL == 0)
        fflush(dust_particle_log);
}

#endif /* DUST */