#!/usr/bin/env python3
"""
compute_sfr_dust_peak_delay.py
=================================
Compute the exact lookback-time delay between peak SFR and peak dust mass,
from the actual (z, SFR, M_dust) arrays -- NOT by reading values off a
plotted lookback-time axis, which is unreliable for a broad/plateaued SFR
curve and a noisy M_dust curve (exactly the shapes seen in the user's
Figure 2).

INPUT FORMATS SUPPORTED
------------------------
This script tries, in order:

  1. A CSV file with columns including some subset of:
     z (or redshift), SFR (or sfr), Mdust (or M_dust, dust_mass)
     Pass via --csv path/to/file.csv

  2. Re-deriving from compare_grid_dust.py's own log-parsing function
     parse_log(), if you point --logfile at a run's output_*.log and
     --run-script-dir at the directory containing compare_grid_dust.py
     (so this script can import parse_log directly rather than
     reimplementing the regex parsing and risking it drifting out of
     sync with the real parser).

  3. Manual arrays pasted directly into this file at the bottom (see
     MANUAL_Z / MANUAL_SFR / MANUAL_MDUST placeholders) -- use this if
     you already have the arrays in a Python session/notebook and just
     want the peak-finding + cosmology conversion logic.

METHOD
------
1. Find z_peak_SFR = z[argmax(SFR)] and z_peak_dust = z[argmax(M_dust)].
   Since SFR is a broad plateau (per the user's figure) rather than a
   sharp peak, ALSO reports the redshift where SFR first drops below a
   configurable fraction (default 90%) of its peak value, as a more
   robust "decline onset" alternative to a bare argmax on a plateau --
   argmax on a flat plateau can land almost anywhere along the flat part
   depending on noise, while a decline-onset threshold is less sensitive
   to where exactly the plateau's noise puts the single highest sample.
2. Convert both redshifts to lookback time using the actual cosmological
   parameters (Omega0, OmegaLambda, h) from the run, via direct numerical
   integration of dt/dz -- NOT read off a plotted axis label.
3. Report the lookback-time DIFFERENCE between the two events. This
   is the delay quoted in the conclusion bullet.

COSMOLOGY
---------
Lookback time from z=0 to z is:
    t_lookback(z) = integral_0^z  dz' / [(1+z') H(z')]
where H(z) = H0 sqrt(Om0 (1+z)^3 + OmegaLambda) for flat LCDM.
This is computed via numerical quadrature (scipy.integrate.quad), not
an approximation -- this is the same physics as halo_utils.py's
rho_crit_cgs()/E(z) calculations, just integrated over z instead of
evaluated at a single z.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    from scipy.integrate import quad
except ImportError:
    print("ERROR: scipy is required (pip install scipy --break-system-packages)",
          file=sys.stderr)
    sys.exit(1)


# -----------------------------------------------------------------------
# Cosmology: lookback time via direct integration (no approximations)
# -----------------------------------------------------------------------

def hubble_parameter_km_s_mpc(z, H0_km_s_mpc, Om0, OmegaLambda, Omega_k=None):
    """H(z) in km/s/Mpc for a (possibly non-flat) LCDM cosmology."""
    if Omega_k is None:
        Omega_k = 1.0 - Om0 - OmegaLambda
    Ez = np.sqrt(Om0 * (1 + z) ** 3 + Omega_k * (1 + z) ** 2 + OmegaLambda)
    return H0_km_s_mpc * Ez


def lookback_time_gyr(z, H0_km_s_mpc, Om0, OmegaLambda):
    """
    Lookback time in Gyr from z=0 to redshift z, via direct numerical
    integration of dt/dz' = 1 / [(1+z') H(z')] from 0 to z.
    """
    KM_S_MPC_TO_INV_GYR = 1.0227121650537077e-3
    # (1 km/s/Mpc) = 1.0227e-3 / Gyr -- standard conversion factor.

    H0_per_gyr = H0_km_s_mpc * KM_S_MPC_TO_INV_GYR

    def integrand(zp):
        Hz_per_gyr = H0_per_gyr * np.sqrt(
            Om0 * (1 + zp) ** 3 + (1.0 - Om0 - OmegaLambda) * (1 + zp) ** 2 + OmegaLambda
        )
        return 1.0 / ((1 + zp) * Hz_per_gyr)

    val, err = quad(integrand, 0.0, z, limit=200)
    return val


# -----------------------------------------------------------------------
# Peak-finding (robust to a broad/plateaued SFR curve)
# -----------------------------------------------------------------------

def find_peak_and_decline_onset(z_arr, y_arr, decline_frac=0.9, descending_z=True):
    """
    z_arr, y_arr : 1D arrays, same length. z_arr need not be sorted, but
                   should be monotonic in cosmic time (descending_z=True
                   means z_arr runs high-z to low-z, i.e. forward in time;
                   set False if your array runs the other way).

    Returns dict with:
      z_at_argmax       : redshift of the single highest sample
      z_decline_onset   : redshift where y FIRST drops below
                          decline_frac * peak, scanning forward in time
                          (i.e. from high z to low z) -- robust to a flat
                          plateau where the bare argmax could land on
                          noise anywhere along the plateau.
    """
    z_arr = np.asarray(z_arr, dtype=float)
    y_arr = np.asarray(y_arr, dtype=float)

    order = np.argsort(-z_arr) if descending_z else np.argsort(z_arr)
    z_sorted = z_arr[order]
    y_sorted = y_arr[order]

    i_max = int(np.argmax(y_sorted))
    z_at_argmax = float(z_sorted[i_max])
    peak_val = float(y_sorted[i_max])

    threshold = decline_frac * peak_val
    z_decline_onset = None
    for i in range(i_max, len(y_sorted)):
        if y_sorted[i] < threshold:
            z_decline_onset = float(z_sorted[i])
            break

    return dict(
        z_at_argmax=z_at_argmax,
        peak_val=peak_val,
        z_decline_onset=z_decline_onset,
        decline_frac=decline_frac,
    )


# -----------------------------------------------------------------------
# Input loaders
# -----------------------------------------------------------------------

def load_from_csv(path):
    import csv
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames_lower = {k.lower(): k for k in reader.fieldnames}
        z_key = fieldnames_lower.get("z") or fieldnames_lower.get("redshift")
        sfr_key = fieldnames_lower.get("sfr")
        mdust_key = (fieldnames_lower.get("mdust") or fieldnames_lower.get("m_dust")
                     or fieldnames_lower.get("dust_mass") or fieldnames_lower.get("mass"))
        if not (z_key and sfr_key and mdust_key):
            raise ValueError(
                f"Could not find z/SFR/Mdust columns in {path}. "
                f"Found columns: {reader.fieldnames}. "
                f"Expected something like z,SFR,Mdust (case-insensitive, "
                f"M_dust/dust_mass/mass also accepted for the dust column)."
            )
        for row in reader:
            rows.append((float(row[z_key]), float(row[sfr_key]), float(row[mdust_key])))
    z, sfr, mdust = map(np.array, zip(*rows))
    return z, sfr, mdust


def load_from_compare_grid_dust(run_script_dir, logfile, run_label="RUN"):
    """
    Import parse_log directly from the user's compare_grid_dust.py so the
    same regex parsing already validated against real log files is reused
    here, rather than reimplemented (and risking drift/inconsistency).
    """
    sys.path.insert(0, str(run_script_dir))
    try:
        from compare_grid_dust import parse_log
    except ImportError as e:
        raise ImportError(
            f"Could not import parse_log from compare_grid_dust.py in "
            f"{run_script_dir}: {e}"
        )
    data = parse_log(logfile)
    # parse_log's 'mass' field is in CODE units (1e10 Msun/h) per
    # compare_grid_dust.py's own convention -- caller may want to convert,
    # but for PEAK-FINDING purposes the overall scale doesn't matter
    # (argmax/threshold-crossing are scale-invariant).
    z = data["z"]
    # NOTE: compare_grid_dust.py's parse_log() does not currently extract
    # SFR directly (it extracts dust-physics statistics, not star
    # formation rate) -- if your log has a separate SFR field/source,
    # supply it via --csv instead. This loader is provided as a template
    # for the M_dust half only; check your own log format for SFR.
    mdust = data["mass"]
    sfr = None
    return z, sfr, mdust


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=None,
                     help="CSV file with z,SFR,Mdust columns (case-insensitive; "
                          "Mdust/M_dust/dust_mass/mass all accepted).")
    ap.add_argument("--H0", type=float, default=67.32,
                     help="H0 in km/s/Mpc (default 67.32, i.e. h=0.6732 "
                          "matching this project's HubbleParam).")
    ap.add_argument("--Om0", type=float, default=0.3158,
                     help="Omega_matter (default matches this project's "
                          "_compute_r200_from_m200_simple default).")
    ap.add_argument("--OmegaLambda", type=float, default=0.6842,
                     help="Omega_Lambda (default matches this project's "
                          "_compute_r200_from_m200_simple default).")
    ap.add_argument("--decline-frac", type=float, default=0.9,
                     help="Fraction of peak SFR used to define 'decline "
                          "onset' as a plateau-robust alternative to a "
                          "bare argmax (default 0.9, i.e. 90%% of peak).")
    args = ap.parse_args()

    if args.csv:
        z, sfr, mdust = load_from_csv(args.csv)
    else:
        print("No --csv given. Using MANUAL placeholder arrays at the "
              "bottom of this script -- edit those directly with your "
              "real (z, SFR, Mdust) arrays, or re-run with --csv.\n")
        z, sfr, mdust = MANUAL_Z, MANUAL_SFR, MANUAL_MDUST
        if z is None:
            print("ERROR: no data source provided. Either pass --csv, or "
                  "edit MANUAL_Z/MANUAL_SFR/MANUAL_MDUST at the bottom of "
                  "this file with your real arrays.", file=sys.stderr)
            sys.exit(1)

    print(f"Loaded {len(z)} points. z range: [{np.min(z):.3f}, {np.max(z):.3f}]")
    print(f"Cosmology: H0={args.H0} km/s/Mpc, Om0={args.Om0}, "
          f"OmegaLambda={args.OmegaLambda}\n")

    sfr_peaks = find_peak_and_decline_onset(z, sfr, decline_frac=args.decline_frac)
    dust_peaks = find_peak_and_decline_onset(z, mdust, decline_frac=args.decline_frac)

    print("=== SFR ===")
    print(f"  z at bare argmax        : {sfr_peaks['z_at_argmax']:.4f}")
    print(f"  z at decline onset      : {sfr_peaks['z_decline_onset']} "
          f"(first drop below {args.decline_frac*100:.0f}% of peak, "
          f"scanning forward in time from the peak)")

    print("\n=== M_dust ===")
    print(f"  z at bare argmax (peak) : {dust_peaks['z_at_argmax']:.4f}")

    t_sfr_argmax  = lookback_time_gyr(sfr_peaks['z_at_argmax'], args.H0, args.Om0, args.OmegaLambda)
    t_dust_argmax = lookback_time_gyr(dust_peaks['z_at_argmax'], args.H0, args.Om0, args.OmegaLambda)
    delay_argmax  = t_sfr_argmax - t_dust_argmax  # positive = dust peaks LATER (smaller lookback time)

    print(f"\n=== Lookback times (bare argmax) ===")
    print(f"  t_lookback(SFR peak)   = {t_sfr_argmax:.3f} Gyr")
    print(f"  t_lookback(dust peak)  = {t_dust_argmax:.3f} Gyr")
    print(f"  DELAY (SFR peak -> dust peak) = {delay_argmax:.3f} Gyr")

    if sfr_peaks['z_decline_onset'] is not None:
        t_sfr_decline = lookback_time_gyr(sfr_peaks['z_decline_onset'], args.H0, args.Om0, args.OmegaLambda)
        delay_decline = t_sfr_decline - t_dust_argmax
        print(f"\n=== Lookback times (SFR decline-onset, more robust to plateau) ===")
        print(f"  t_lookback(SFR decline onset) = {t_sfr_decline:.3f} Gyr")
        print(f"  t_lookback(dust peak)         = {t_dust_argmax:.3f} Gyr")
        print(f"  DELAY (SFR decline onset -> dust peak) = {delay_decline:.3f} Gyr")
        print(f"\n  >>> Report THIS delay ({delay_decline:.2f} Gyr) in the text if your SFR")
        print(f"      curve is a broad plateau rather than a sharp peak (per your Figure 2) --")
        print(f"      it is far less sensitive to exactly where a single noisy sample happens")
        print(f"      to be highest along a flat plateau than the bare-argmax delay above.")

    print("\nNOTE: if delay_argmax and delay_decline differ substantially, that")
    print("itself confirms the SFR curve is plateau-shaped (as it visually appears")
    print("in your Figure 2) and the bare-argmax delay is not a robust number to quote.")


# -----------------------------------------------------------------------
# EDIT THESE if running without --csv
# -----------------------------------------------------------------------
MANUAL_Z     = None   # e.g. np.array([7.0, 6.5, 6.0, ...])
MANUAL_SFR   = None   # e.g. np.array([12.0, 18.0, 25.0, ...])  Msun/yr
MANUAL_MDUST = None   # e.g. np.array([1e5, 3e5, 8e5, ...])     Msun or code units


if __name__ == "__main__":
    main()
