#!/usr/bin/env python3
"""
Integrate AGB metal yields over Kroupa (2001) IMF

Replaces the original integrate_agb_yields.py which used a Chabrier (2003) IMF.
Updated to Kroupa for consistency with the SNII rate constant (NSNE_PER_MSUN_VAL = 0.011)
which was also derived using a Kroupa IMF.

Computes:
    AGB_METAL_YIELD_PER_MSUN = ∫_{1}^{8} ξ(m) · Y(m, Z) dm
                                ─────────────────────────────
                                ∫_{0.1}^{100} ξ(m) · m dm

where ξ(m) = dN/dm is the Kroupa IMF and Y(m, Z) is the total metal yield
from the Huscher et al. (2025) MESA AGB table.

Usage:
    python3 integrate_agb_yields_kroupa.py agb_yield_table.txt

The yield table is expected to have columns:
    M_init  Z_init  <other cols>  Z_total  ...
with Z_total (total metal yield in Msun) in column index 9.
"""

import sys
import numpy as np
from scipy.interpolate import RegularGridInterpolator


# ─────────────────────────────────────────────────────────────────────────────
# Kroupa (2001) IMF
# dN/dM ∝ M^(-α) with breaks at 0.08 and 0.5 Msun
#
# Normalisation constant A is chosen so that ∫_{0.1}^{100} ξ(m)·m dm = 1
# (i.e., one solar mass of stars formed).
# ─────────────────────────────────────────────────────────────────────────────

def kroupa_imf_unnorm(m):
    """
    Unnormalised Kroupa (2001) IMF: dN/dM (not per log-mass).
    Slopes: α=1.3 for 0.08–0.5 Msun, α=2.3 for >0.5 Msun.
    Continuity enforced at break mass.
    """
    m_break = 0.5       # Msun
    alpha_lo = 1.3      # 0.08 < m < 0.5 Msun
    alpha_hi = 2.3      # m > 0.5 Msun

    if m <= 0.0:
        return 0.0
    elif m < m_break:
        return m ** (-alpha_lo)
    else:
        # Continuity at m_break
        A_hi = m_break ** (-alpha_lo) / m_break ** (-alpha_hi)
        return A_hi * m ** (-alpha_hi)


def compute_imf_normalisation(m_lo=0.1, m_hi=100.0, n_bins=10000):
    """
    Compute normalisation constant so that ∫ ξ(m)·m dm = 1 over [m_lo, m_hi].
    Returns the divisor (total mass integral of the unnormalised IMF).
    """
    masses = np.linspace(m_lo, m_hi, n_bins)
    dm = masses[1] - masses[0]
    total_mass = sum(kroupa_imf_unnorm(m) * m * dm for m in masses)
    return total_mass


# Pre-compute normalisation once
_IMF_NORM = compute_imf_normalisation()


def kroupa_imf(m):
    """
    Normalised Kroupa IMF: dN/dM per Msun of stellar population formed.
    Satisfies ∫_{0.1}^{100} ξ(m)·m dm = 1.
    """
    return kroupa_imf_unnorm(m) / _IMF_NORM


# ─────────────────────────────────────────────────────────────────────────────
# AGB yield table loader
# ─────────────────────────────────────────────────────────────────────────────

def load_agb_table(filename):
    """
    Load AGB yield table (Huscher et al. 2025 MESA format).

    Expected columns (0-indexed):
        0: M_init  (Msun)
        1: Z_init  (mass fraction)
        9: Z_total (total metal yield, Msun)

    Returns:
        masses:        sorted unique initial masses (Msun)
        metallicities: sorted unique initial metallicities
        interpolator:  RegularGridInterpolator for Y(M, Z)
    """
    data = np.loadtxt(filename)

    M_init  = data[:, 0]
    Z_init  = data[:, 1]
    Z_total = data[:, 9]

    masses        = np.unique(M_init)
    metallicities = np.unique(Z_init)

    print(f"Loaded AGB table: {filename}")
    print(f"  Mass range:        {masses.min():.2f} – {masses.max():.2f} Msun  ({len(masses)} points)")
    print(f"  Metallicity range: {metallicities.min():.5f} – {metallicities.max():.5f}  ({len(metallicities)} points)")
    print(f"  Total data points: {len(data)}")

    yields_2d = Z_total.reshape(len(masses), len(metallicities))

    interpolator = RegularGridInterpolator(
        (masses, metallicities),
        yields_2d,
        method='linear',
        bounds_error=False,
        fill_value=0.0,
    )
    return masses, metallicities, yields_2d, interpolator


# ─────────────────────────────────────────────────────────────────────────────
# Main integration
# ─────────────────────────────────────────────────────────────────────────────

def integrate_agb_yield(interpolator, Z_star=0.02, m_min=1.0, m_max=8.0, n_bins=500):
    """
    Compute AGB_METAL_YIELD_PER_MSUN by integrating over the Kroupa IMF.

        result = ∫_{m_min}^{m_max} ξ(m) · Y(m, Z_star) dm

    The denominator (normalisation) is already baked into kroupa_imf().

    Parameters:
        interpolator: RegularGridInterpolator from load_agb_table()
        Z_star:       stellar metallicity to evaluate yields at
        m_min, m_max: AGB mass range (Msun)
        n_bins:       number of quadrature points

    Returns:
        yield_per_msun: metal mass ejected per Msun of stellar population formed
    """
    masses = np.linspace(m_min, m_max, n_bins)
    dm = masses[1] - masses[0]

    total_yield = 0.0
    for m in masses:
        xi   = kroupa_imf(m)                          # dN/dM [per Msun formed]
        Y    = float(interpolator([[m, Z_star]]))      # metal yield [Msun per star]
        total_yield += xi * Y * dm

    return total_yield


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 integrate_agb_yields_kroupa.py agb_yield_table.txt")
        sys.exit(1)

    table_file = sys.argv[1]
    masses, metallicities, yields_2d, interpolator = load_agb_table(table_file)

    print("\n" + "=" * 65)
    print("AGB YIELD INTEGRATION — Kroupa (2001) IMF")
    print("=" * 65)

    results = {}
    for Z in metallicities:
        y = integrate_agb_yield(interpolator, Z_star=Z)
        results[Z] = y
        print(f"  Z = {Z:.5f}:  yield = {y:.6e} Msun/Msun  ({y*100:.4f}%)")

    # Solar metallicity result for the code constant
    Z_solar = 0.02
    # Find closest metallicity in table to Z_solar
    Z_use   = metallicities[np.argmin(np.abs(metallicities - Z_solar))]
    y_solar = results[Z_use]

    print("\n" + "=" * 65)
    print("RESULT FOR feedback.cc  (Kroupa IMF, Z ≈ Z_solar)")
    print("=" * 65)
    print(f"  Z used: {Z_use:.5f}")
    print(f"\n  static const double AGB_METAL_YIELD_PER_MSUN = {y_solar:.6e};")
    print(f"  // Integrated over Kroupa (2001) IMF, AGB mass range 1–8 Msun")
    print(f"  // Huscher et al. (2025) MESA yield tables, Z = {Z_use:.4f}")
    print("=" * 65)

    # Sanity check: compare to old Chabrier value
    old_val = 9.956112e-03
    print(f"\n  Old value (Chabrier IMF): {old_val:.6e}")
    print(f"  New value (Kroupa IMF):   {y_solar:.6e}")
    print(f"  Change: {(y_solar - old_val)/old_val * 100:+.2f}%")

    # Cross-check against SNII
    snii_metal_per_msun = 0.011 * 2.0  # NSNE_PER_MSUN_VAL × METAL_YIELD_PER_SN_MSUN
    print(f"\n  SNII metal yield per Msun: {snii_metal_per_msun:.4e} (for reference)")
    print(f"  AGB / SNII ratio:          {y_solar / snii_metal_per_msun:.3f}×")

    # Verify IMF normalisation is self-consistent
    m_arr = np.linspace(0.1, 100, 50000)
    dm = m_arr[1] - m_arr[0]
    norm_check = sum(kroupa_imf(m) * m * dm for m in m_arr)
    print(f"IMF mass normalisation check: {norm_check:.6f}  (should be ~1.0)")

if __name__ == "__main__":
    main()
