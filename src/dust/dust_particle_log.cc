// ============================================================================
// dust_particle_log.cc
//
// Per-particle event log for CosmicGrain dust, used for analysis after the run.
//
// Each time a dust particle is destroyed we write one line. From these lines
// you can reconstruct:
//   - how far dust travels (displacement from birth to death)
//   - grain growth (radius at death vs. the birth size set by its source)
//   - which process destroys dust (thermal vs. shock vs. astration, ...)
//   - grain size and carbon fraction, for radiation pressure studies
//   - the gas it died in (gas density, scale factor)
//
// ── COAGULATION HISTOGRAM ────────────────────────────────────────────────────
//
// Coagulation (grains sticking together) happens far too often for one row
// per event at 2048³. Instead each task counts events in a small histogram of
// gas density n_H (hydrogen number density, cm^-3). print_coag_histogram()
// sums the tasks and prints the result to stdout. It is called from
// print_dust_statistics(). Counts are never reset, so each print shows totals
// since the start of the run.
//
// ── SPUTTERING THROTTLE ──────────────────────────────────────────────────────
//
// Sputtering = grains worn away by hot gas. We do not log sputtering deaths of
// AGB-carbon grains (DustSource==1) lighter than AGB_SPUTTER_MASS_THRESH (code
// units). They are born at ~1e-15 code units and sputter at once in hot CGM
// (circumgalactic) gas. At 1024³ they made ~45% of log rows (more at 2048³)
// but <0.01% of the sputtered mass. Other grain types are always logged.
//
// ── FILE LAYOUT ──────────────────────────────────────────────────────────────
//
// One file per MPI task:
//   <OutputDir>/dust_logs/dust_log_task<N>.txt
//
// The directory is created on every start mode. A fresh run (RestartFlag 0)
// starts new files with a header; any restart appends, so the files keep the
// full history across restarts.
//
// To merge all tasks after the run:
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
//  15   grain_type      DustSource: 0=SNII-silicate  1=AGB-carbon  2=LRN
//                       (the file header calls type 2 "mixed")
//  16   event_type      see DUST_EVENT_* constants in dust.h
//
// The birth position is the parent star's position, without the small random
// offset the grain is placed at.
//
// ── BUFFERING ─────────────────────────────────────────────────────────────────
//
// Each task writes through a 4 MB buffer (setvbuf _IOFBF), so many events go
// out in one large write. One system call per event gets slow at high dust
// counts (~16M particles at z~3).
//
// We flush explicitly:
//   - once after the header on a fresh run
//   - every FLUSH_INTERVAL events, so a crash loses little
//   - once at shutdown in close_dust_particle_log()
//
// ── BIRTH POSITION GUARD ──────────────────────────────────────────────────────
//
// BirthPos starts at zero (DustP[] is memset when a grain is spawned) and is
// then set in spawn_dust_particle(). If it is still (0,0,0), the particle
// either predates BirthPos tracking (restarts of early runs) or had its
// DustP[] zeroed by domain-exchange corruption. Such particles are skipped.
//
// NOTE: this guard also skips real grains born exactly at the box origin.
// That is fine for Halo569, which sits far from (0,0,0). If a target halo
// ever sits near the origin, replace the guard with a bool in dust_data
// (e.g. birth_pos_set), false by default and set in spawn_dust_particle().
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

// ── Constants ────────────────────────────────────────────────────────────────

// Events between explicit fflush calls, so a crash loses at most this many.
static constexpr int FLUSH_INTERVAL = 10000;

// Mass threshold (code units) of the sputtering throttle (see top of file).
// AGB-carbon grains (DustSource==1) lighter than this are not logged when
// sputtered. They die where they were born, so no transport information is
// lost. Types 0 and 2 are always logged. Set to 0.0 to log every sputtering event.
static constexpr double AGB_SPUTTER_MASS_THRESH = 1.0e-13;

// ── Coagulation histograms ────────────────────────────────────────────────────
//
// Two histograms with 12 bins, evenly spaced in log10(n), over 0.01 to 10^4 cm^-3:
//   coag_hist_counts : n_H of the gas at each coagulation event
//   coag_neff_counts : n_eff, the effective density that sets the coagulation
//                      rate (n_H times a clumping factor, or the dense-phase
//                      density in the dense-phase model)
//
// Each task fills its own copy; print_coag_histogram() sums them.

static constexpr int    COAG_HIST_NBINS = 12;
static constexpr double COAG_HIST_LO    = 0.01;   // cm^-3
static constexpr double COAG_HIST_HI    = 1.0e4;  // cm^-3

static long long coag_hist_counts[COAG_HIST_NBINS] = {};  // raw n_H
static long long coag_neff_counts[COAG_HIST_NBINS] = {};  // n_eff
static long long coag_hist_total                   = 0;

// This task's log file; NULL if it is not open.
static FILE *dust_particle_log = NULL;


// ============================================================================
// _coag_hist_bin()
//
// Histogram bin [0, COAG_HIST_NBINS-1] for a density in cm^-3.
// Densities outside the range go into the first or last bin.
// ============================================================================
static inline int _coag_hist_bin(double n_H_cgs)
{
    if(n_H_cgs <= COAG_HIST_LO) return 0;
    if(n_H_cgs >= COAG_HIST_HI) return COAG_HIST_NBINS - 1;

    const double log_lo = log10(COAG_HIST_LO);
    const double log_hi = log10(COAG_HIST_HI);
    int bin = (int)((log10(n_H_cgs) - log_lo) / (log_hi - log_lo) * COAG_HIST_NBINS);
    if(bin < 0)                 bin = 0;
    if(bin >= COAG_HIST_NBINS)  bin = COAG_HIST_NBINS - 1;
    return bin;
}


// ============================================================================
// record_coagulation_event()
//
// Called at the end of dust_grain_coagulation() each time a grain grows.
// Adds one count to this task's histograms; no file I/O. Both densities in cm^-3.
// ============================================================================
void record_coagulation_event(double n_H_cgs, double n_eff_cgs)
{
    coag_hist_counts[_coag_hist_bin(n_H_cgs)]++;
    coag_neff_counts[_coag_hist_bin(n_eff_cgs)]++;
    coag_hist_total++;
}


// ============================================================================
// print_coag_histogram()
//
// Sum the per-task histograms onto task 0, which prints them to stdout.
// Collective (MPI_Reduce): every task must call it. Called from
// print_dust_statistics(). Counts are totals since the start of the run.
//
// Example output:
//   [COAG_HIST|a=0.4500] Cumulative coagulation events: 4,821,093
//   [COAG_HIST]  n_H edges (cm^-3):     0.01    0.032     0.10 ...
//   [COAG_HIST]  raw n_H counts   :        0        0      128 ...
//   [COAG_HIST]  n_eff counts     :        0        0      841 ...
//   [COAG_HIST]  Peak raw n_H bin : ~3.2 cm^-3  | Peak n_eff bin: ~32 cm^-3
//
// ============================================================================
void print_coag_histogram(MPI_Comm Communicator)
{

    long long global_hist[COAG_HIST_NBINS] = {};
    long long global_neff[COAG_HIST_NBINS] = {};
    long long global_total                 = 0;

    MPI_Reduce(coag_hist_counts, global_hist, COAG_HIST_NBINS,
               MPI_LONG_LONG, MPI_SUM, 0, Communicator);
    MPI_Reduce(coag_neff_counts, global_neff, COAG_HIST_NBINS,
               MPI_LONG_LONG, MPI_SUM, 0, Communicator);
    MPI_Reduce(&coag_hist_total, &global_total, 1,
               MPI_LONG_LONG, MPI_SUM, 0, Communicator);

    if(All.ThisTask != 0) return;

    const double log_lo = log10(COAG_HIST_LO);
    const double log_hi = log10(COAG_HIST_HI);

    printf("[COAG_HIST|a=%.4f] Cumulative coagulation events: %lld\n",
           (double)All.Time, global_total);

    printf("[COAG_HIST]  n_H edges (cm^-3):");
    for(int i = 0; i < COAG_HIST_NBINS; i++)
    {
        double edge = pow(10.0, log_lo + i * (log_hi - log_lo) / COAG_HIST_NBINS);
        printf("  %8.3g", edge);
    }
    printf("\n");

    printf("[COAG_HIST]  raw n_H counts   :");
    for(int i = 0; i < COAG_HIST_NBINS; i++)
        printf("  %8lld", global_hist[i]);
    printf("\n");

    printf("[COAG_HIST]  n_eff counts     :");
    for(int i = 0; i < COAG_HIST_NBINS; i++)
        printf("  %8lld", global_neff[i]);
    printf("\n");

    if(global_total > 0)
    {
        int peak_raw = 0, peak_neff = 0;
        for(int i = 1; i < COAG_HIST_NBINS; i++)
        {
            if(global_hist[i] > global_hist[peak_raw])   peak_raw  = i;
            if(global_neff[i] > global_neff[peak_neff])  peak_neff = i;
        }
        double edge_raw  = pow(10.0, log_lo + peak_raw  * (log_hi - log_lo) / COAG_HIST_NBINS);
        double edge_neff = pow(10.0, log_lo + peak_neff * (log_hi - log_lo) / COAG_HIST_NBINS);
        printf("[COAG_HIST]  Peak raw n_H bin : ~%.2g cm^-3  "
               "| Peak n_eff bin: ~%.2g cm^-3\n",
               edge_raw, edge_neff);
    }
}


// ============================================================================
// open_dust_particle_log()
//
// Collective (MPI_Barrier): every task must call it. Task 0 creates the
// directory; then each task opens its own file. If a file cannot be opened
// the task prints an error and runs on without a log.
// ============================================================================
void open_dust_particle_log(MPI_Comm Communicator)
{
    if(All.ThisTask == 0)
    {
        char logdir[MAXLEN_PATH];
        snprintf(logdir, MAXLEN_PATH, "%sdust_logs", All.OutputDir);

        // Create the directory on every start mode (fresh, restart file, or
        // snapshot restart). It used to be made only on fresh starts, so
        // restarts into a new OutputDir could not open their logs.
        if(mkdir(logdir, 02755) != 0 && errno != EEXIST)
            printf("[DUST_LOG|T=0] WARNING: could not create %s: %s\n",
                   logdir, strerror(errno));
    }
    MPI_Barrier(Communicator);

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

    setvbuf(dust_particle_log, NULL, _IOFBF, 4 * 1024 * 1024);

    if(All.RestartFlag == RST_BEGIN)
    {
        fprintf(dust_particle_log,
            "# CosmicGrain particle event log — task %d\n"
            "# One row per destruction event\n"
            "#\n"
            "# NOTE: AGB-carbon grains (type 1) below %.2e code units are\n"
            "# not logged when sputtered (immediate hot-CGM destruction,\n"
            "# negligible mass). All other types always logged.\n"
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
            AGB_SPUTTER_MASS_THRESH,
            DUST_EVENT_THERMAL, DUST_EVENT_SHOCK, DUST_EVENT_SHATTERING,
            DUST_EVENT_ASTRATION, DUST_EVENT_SUBLIMATION, DUST_EVENT_CLEANUP);

        fflush(dust_particle_log);
    }

    if(All.ThisTask == 0)
        printf("[DUST_LOG] Opened dust_logs/dust_log_task*.txt (mode=%s)\n"
               "[DUST_LOG] AGB sputtering suppressed below %.2e code units\n",
               mode, AGB_SPUTTER_MASS_THRESH);
}


// ============================================================================
// close_dust_particle_log()
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
// Write one row for a dust particle that is about to be destroyed. Call it
// before the particle's data is removed. nearest_gas is a local gas index,
// or -1 if none (then gas_density is written as 0).
// ============================================================================
void log_dust_particle_event(simparticles *Sp, int dust_idx,
                              int nearest_gas, int event_type)
{
    if(dust_particle_log == NULL) return;

    // Cleanup removes corrupted particles, whose data cannot be trusted.
    if(event_type == DUST_EVENT_CLEANUP) return;

    // ── AGB sputtering throttle (see top of file) ─────────────────────────
    if(event_type == DUST_EVENT_THERMAL              &&
       AGB_SPUTTER_MASS_THRESH > 0.0                 &&
       Sp->DustP[dust_idx].DustSource == 1            &&
       Sp->P[dust_idx].getMass() < AGB_SPUTTER_MASS_THRESH)
        return;

    // ── Birth position guard: (0,0,0) means unknown (see top of file) ─────
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

    // ── Displacement, using the nearest periodic image ────────────────────
    double dx = event_pos[0] - birth_pos[0];
    double dy = event_pos[1] - birth_pos[1];
    double dz = event_pos[2] - birth_pos[2];

    const double half = All.BoxSize * 0.5;
    if(dx >  half) dx -= All.BoxSize;
    if(dx < -half) dx += All.BoxSize;
    if(dy >  half) dy -= All.BoxSize;
    if(dy < -half) dy += All.BoxSize;
    if(dz >  half) dz -= All.BoxSize;
    if(dz < -half) dz += All.BoxSize;

    double displacement = sqrt(dx*dx + dy*dy + dz*dz);

    // ── Local gas density, only if nearest_gas is a valid local gas index ─
    double gas_density = 0.0;
    if(nearest_gas >= 0 && nearest_gas < Sp->NumGas &&
       Sp->P[nearest_gas].getType() == 0)
        gas_density = Sp->SphP[nearest_gas].Density;

    // ── Write event row (StellarAge holds the dust's birth scale factor) ──
    fprintf(dust_particle_log,
        "%lld "
        "%.6f %.6f "
        "%.4f %.4f %.4f "
        "%.4f %.4f %.4f "
        "%.4f "
        "%.6e "
        "%.3f "
        "%.4f "
        "%.6e "
        "%d %d\n",
        (long long)Sp->P[dust_idx].ID.get(),
        (double)Sp->P[dust_idx].StellarAge,
        (double)All.Time,
        birth_pos[0], birth_pos[1], birth_pos[2],
        event_pos[0],  event_pos[1],  event_pos[2],
        displacement,
        Sp->P[dust_idx].getMass(),
        (double)Sp->DustP[dust_idx].GrainRadius,
        (double)Sp->DustP[dust_idx].CarbonMassFraction,
        gas_density,
        (int)Sp->DustP[dust_idx].DustSource,
        event_type);

    static int event_count = 0;
    if(++event_count % FLUSH_INTERVAL == 0)
        fflush(dust_particle_log);
}

#endif /* DUST */