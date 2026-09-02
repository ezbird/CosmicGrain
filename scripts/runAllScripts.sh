#!/usr/bin/env bash
set -euo pipefail
# ==========================
# User-configurable variables
# ==========================
SIMS_ROOT=".."   # consolidated folder containing
                                                 # all {rung}_output_{res}/ dirs
BASE_DIR="$SIMS_ROOT/S10_output_512_3e6K"
SNAP=47
RES=512              # resolution, used by plot_radial_dgr.py / plot_dz_vs_metallicity.py --res
#RMAX=127              # ckpc/h -- passed to plot_dust_histograms_agecoded.py's --rmax
                       # (matches Coordinates' native units; NOT physical pkpc)
PLOTS_DIR="$BASE_DIR/all_plots"
mkdir -p "$PLOTS_DIR"

# If user runs: bash runAllScripts.sh 1, show plots each time.
# Omitting the 1 defaults to 0 and runs each script without displaying.
# NOTE: none of the 8 figure scripts below call plt.show() (verified against
# their source), so they're all headless/batch-safe regardless of this flag --
# SHOW_PLOT is kept only in case you add a script later that does support it.
SHOW_PLOT=${1:-0}

# Zero-padded snapshot (049)
SNAP_PAD=$(printf "%03d" "$SNAP")
CATALOG="${BASE_DIR}/groups_${SNAP_PAD}/fof_subhalo_tab_${SNAP_PAD}.0.hdf5"
SNAPSHOT="${BASE_DIR}/snapdir_${SNAP_PAD}/snapshot_${SNAP_PAD}"

# Date suffix for every output filename in this run, e.g. "2026-07-10"
DATE=$(date +%Y-%m-%d)

echo "=========================================="
echo "  CosmicGrain figure + health-check pipeline"
echo "  BASE_DIR = $BASE_DIR"
echo "  SNAP     = $SNAP_PAD   RES = $RES"
echo "  Output   -> $PLOTS_DIR  (dated $DATE)"
echo "=========================================="
echo

# ==========================
# Paper figure scripts (all 8 confirmed must-keep scripts)
# ==========================

echo "--- [1/8] plot_halo_projection_full_ladder.py (dust surface density, S0-S10 physics ladder) ---"
python3 plot_halo_projection_full_ladder.py \
    --snap-pattern  "${SIMS_ROOT}/{rung}_output_${RES}/snapdir_{num}/snapshot_{num}.0.hdf5" \
    --group-pattern "${SIMS_ROOT}/{rung}_output_${RES}/groups_{num}/fof_subhalo_tab_{num}.0.hdf5" \
    --snap-num auto --rungs S0 S1 S2 S3 S4 S5 S6 S7 S8 S9 S10 \
    --axis z --view ism --depth-frac 0.5 \
    --bar-quantity gas_compare --gas-compare-rung S10 \
    --vmin-dust 1e-5 --npix 512 --dust-adaptive-k 8 --dust-adaptive-min 0.5 --gas-adaptive \
    --out "$PLOTS_DIR/surface_densities_${DATE}.png"

echo "--- [2/8] plot_radial_evolution.py (cumulative radial dust distribution) ---"
python3 plot_radial_evolution.py "$BASE_DIR" \
    --redshifts 0 0.5 1 2 3 \
    --rmax-factor 1.0 \
    --outdir "$PLOTS_DIR/radial_evolution_cache_${DATE}" \
    --summary "$PLOTS_DIR/dust_radial_evolution_${DATE}.png"

echo "--- [3/8] plot_mdust_mstar_all_halos.py (Mdust vs Mstar) ---"
python3 plot_mdust_mstar_all_halos.py "$BASE_DIR" \
    --obs-data ../data/obs_data/obs_dustmass.npz \
    --simba-catalogs ../data/simba/m50n512_151.hdf5 \
    --simba-max-epochs 1 \
    --output "$PLOTS_DIR/mdust_mstar_S10_${RES}_${DATE}.png"

echo "--- [4/8] plot_dust_histograms_agecoded.py (age-coded dust property histograms) ---"
python3 plot_dust_histograms_agecoded.py \
    --catalog  "$CATALOG" \
    --snapshot "$SNAPSHOT" \
    --out "$PLOTS_DIR/agecoded_${DATE}.png" \

echo "--- [5/8] plot_gsd_comparison.py (grain-size distribution vs MRN/WD01/THEMIS) ---"
python3 plot_gsd_comparison.py "$BASE_DIR" \
    --snap "$SNAP_PAD" \
    --output "$PLOTS_DIR/gsd_comparison_S10_${RES}_${DATE}.png"

echo "--- [6/8] plot_dust_evolution.py (5-panel evolution vs redshift) ---"
python3 plot_dust_evolution.py "$BASE_DIR" \
    --output "$PLOTS_DIR/dust_evolution_${RES}_S10_${DATE}.png"

echo "--- [7/8] plot_radial_dgr.py (radial D/G, D/Z across physics ladder) ---"
python3 plot_radial_dgr.py \
    --res "$RES" \
    --sims-root "$SIMS_ROOT" \
    --output "$PLOTS_DIR/radial_dg_dz_${RES}_${DATE}.png"

echo "--- [8/8] plot_dz_vs_metallicity.py (D/Z vs gas-phase metallicity) ---"
python3 plot_dz_vs_metallicity.py \
    --res "$RES" \
    --sims-root "$SIMS_ROOT" \
    --output "$PLOTS_DIR/DZ_vs_Z_${RES}_${DATE}.png"

# ==========================
# Health check (complete, non-sampled z=SNAP dust statistics)
# ==========================
echo
echo "--- Health check: dust_snapshot_summary.py ---"
python3 dust_snapshot_summary.py "$BASE_DIR" --snap "$SNAP" \
    | tee "$PLOTS_DIR/dust_snapshot_summary_${SNAP_PAD}_${DATE}.txt"

echo
echo "=========================================="
echo "  Done. All outputs in: $PLOTS_DIR"
echo "=========================================="
ls -la "$PLOTS_DIR"
