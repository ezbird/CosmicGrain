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
// ── COAGULATION HISTOGRAM ────────────────────────────────────────────────────
//
// Per-event rows for coagulation would be enormous at 2048³. Instead, a
// lightweight n_H histogram is accumulated locally per task and printed to
// stdout (merged across all tasks) whenever print_coag_histogram() is called.
// This records the gas density distribution at coagulation events without
// writing millions of rows.
//
// Call print_coag_histogram() from print_dust_statistics() or any other
// periodic diagnostic hook. The histogram accumulates across the entire run
// and is cumulative (not reset between calls) so each print shows totals.
//
// ── SPUTTERING THROTTLE ──────────────────────────────────────────────────────
//
// AGB-carbon grains (GrainType==1) below AGB_SPUTTER_MASS_THRESH code units
// are not logged. These grains are created at ~1e-15 code units and
// immediately sputtered in hot CGM gas, contributing <0.01% of sputtered
// mass but ~45% of log rows at 1024³ (and higher fractions at 2048³).
// SNII-silicate and mixed grains are always logged regardless of mass.
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
//   - Once after the header is written on fresh runs
//   - Every FLUSH_INTERVAL events as a crash-safety checkpoint
//   - Once at shutdown in close_dust_particle_log()
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

// ── Constants ────────────────────────────────────────────────────────────────

// Number of events between explicit fflush calls (crash-safety checkpoint).
static constexpr int FLUSH_INTERVAL = 10000;

// AGB-carbon grains (GrainType==1) below this mass threshold are not logged
// when sputtered. They are created at ~1e-15 code units and immediately
// destroyed in hot CGM gas — no transport information is lost by skipping them.
// SNII-silicate (type 0) and mixed (type 2) grains are always logged.
// Set to 0.0 to restore full logging of all sputtering events.
static constexpr double AGB_SPUTTER_MASS_THRESH = 1.0e-13;

// ── Coagulation histograms ────────────────────────────────────────────────────
//
// Two 12-bin log-spaced histograms covering n_H = 0.01 – 10^4 cm^-3:
//   coag_hist_counts : raw n_H at coagulation event site
//   coag_neff_counts : n_eff = n_H * clumping_factor at same event
//
// Both declared before record_coagulation_event() which references them.
// Accumulated per-task; MPI-reduced in print_coag_histogram().

static constexpr int    COAG_HIST_NBINS = 12;
static constexpr double COAG_HIST_LO    = 0.01;   // cm^-3
static constexpr double COAG_HIST_HI    = 1.0e4;  // cm^-3

static long long coag_hist_counts[COAG_HIST_NBINS] = {};  // raw n_H
static long long coag_neff_counts[COAG_HIST_NBINS] = {};  // n_eff
static long long coag_hist_total                   = 0;

// Module-level file handle — one per task.
static FILE *dust_particle_log = NULL;


// ============================================================================
// _coag_hist_bin()
//
// Return the bin index [0, COAG_HIST_NBINS-1] for a given n_H [cm^-3].
// Values outside the range are clamped to the first/last bin.
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
// Call from dust_grain_coagulation() immediately after a coagulation event
// fires. Accumulates into the per-task histogram; no file I/O occurs here.
//
// Example call site in dust_grain_coagulation():
//
//   double n_H_cgs = gas_density_cgs * HYDROGEN_MASSFRAC / PROTONMASS;
//   double n_eff   = n_H_cgs * DustClumpingFactor;
//   record_coagulation_event(n_H_cgs, n_eff);
//
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
// MPI-reduce the per-task histograms to task 0 and print a formatted
// summary to stdout. Safe to call at any time; no file I/O on non-zero tasks.
// Histogram is cumulative — each call shows totals since simulation start.
//
// Add a call to print_dust_statistics() so it fires at every sync point.
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
// ============================================================================
void open_dust_particle_log(MPI_Comm Communicator)
{
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
// ============================================================================
void log_dust_particle_event(simparticles *Sp, int dust_idx,
                              int nearest_gas, int event_type)
{
    if(dust_particle_log == NULL) return;

    // Cleanup events skipped — particle state is unreliable at that point.
    if(event_type == DUST_EVENT_CLEANUP) return;

    // ── AGB sputtering throttle ───────────────────────────────────────────
    // Skip AGB-carbon grains (type 1) below birth mass. These are created
    // at ~1e-15 code units and immediately sputtered in hot CGM gas with
    // zero displacement — they contribute ~45% of log rows but <0.01% of
    // sputtered mass. SNII and mixed grains are always logged.
    if(event_type == DUST_EVENT_THERMAL              &&
       AGB_SPUTTER_MASS_THRESH > 0.0                 &&
       Sp->DustP[dust_idx].GrainType == 1            &&
       Sp->P[dust_idx].getMass() < AGB_SPUTTER_MASS_THRESH)
        return;

    // ── Birth position guard ──────────────────────────────────────────────
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

    const double half = All.BoxSize * 0.5;
    if(dx >  half) dx -= All.BoxSize;
    if(dx < -half) dx += All.BoxSize;
    if(dy >  half) dy -= All.BoxSize;
    if(dy < -half) dy += All.BoxSize;
    if(dz >  half) dz -= All.BoxSize;
    if(dz < -half) dz += All.BoxSize;

    double displacement = sqrt(dx*dx + dy*dy + dz*dz);

    // ── Local gas density ─────────────────────────────────────────────────
    double gas_density = 0.0;
    if(nearest_gas >= 0 && nearest_gas < Sp->NumGas &&
       Sp->P[nearest_gas].getType() == 0)
        gas_density = Sp->SphP[nearest_gas].Density;

    // ── Write event row ───────────────────────────────────────────────────
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
        (double)Sp->DustP[dust_idx].CarbonFraction,
        gas_density,
        (int)Sp->DustP[dust_idx].GrainType,
        event_type);

    static int event_count = 0;
    if(++event_count % FLUSH_INTERVAL == 0)
        fflush(dust_particle_log);
}

#endif /* DUST */