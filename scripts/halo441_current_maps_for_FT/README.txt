COSMICGRAIN HALO441 SNAPSHOT 029 — ASCII MAPS FOR INDEPENDENT FT
================================================================

Purpose
-------
Portable 2-D matrices for an independent Fourier/power-spectrum calculation.
No Python-specific format or CosmicGrain analysis code is required.

Geometry
--------
Galaxy: halo441 / CG10
Snapshot: 029 (z=0)
Map: face-on
Region: central 20 x 20 kpc
Matrix convention: rows=y, columns=x; array center is galaxy center.

Adopted simulation center (ckpc/h):
  [29459.01585206, 23170.70462420, 27860.34910496]

Face-on angular-momentum direction:
  j_hat = [0.57366824, -0.31402785, 0.75649935]

Files
-----
*_tauB.txt       B-band optical depth
*_tauV.txt       V-band optical depth
*_RV_unmasked.txt
                 R(V) = tau_V / (tau_B - tau_V)
*_dust_count.txt projected PartType6 dust-superparticle count
*_dust_surface_density_code_per_kpc2.txt
                 dust surface density, when available in source NPZ

Masking
-------
RV_unmasked has NO minimum-particle cut and NO relative tau/signal cut.
NaN is used only where the R(V) ratio is mathematically invalid.

Coarse graining
---------------
R(V) is never averaged directly. For every coarse pixel, tau_B and tau_V are
summed first and R(V) is recomputed:
  R(V)_coarse = sum(tau_V) / [sum(tau_B) - sum(tau_V)].

ASCII format
------------
Whitespace-separated matrices. Header lines begin '#'. NaN is written 'nan'.
Pixel size is recorded in every file header.

FT note
-------
For angular spatial wavenumber k, use physical pixel size to express k in
kpc^-1 (including the usual 2*pi relative to cycles/kpc). Missing-pixel/window
treatment is intentionally left to the recipient because it is part of the
independent masking test.
