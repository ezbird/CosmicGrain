# Corrected ZG mask and unsmoothed CG orientation diagnostics

The v1 extraction used correct NESTED HEALPix ordering but omitted the authors'
extinction-weighted RV display mask. It was a raw-field view, not a reproduction
of the publication footprint. This update corrects that distinction.

Primary reference: https://zenodo.org/records/11394477/files/example_query.py?download=1
The authors use 1e8 uniformly sampled positions in a 6x6x0.8 kpc box, a 2.6 kpc
circle and a 200x200 bird's-eye grid, masking RV where summed AV weights are
below 200. With an approximately 0.026 kpc display pixel this is equivalent in
expected sampling to mean AV-weight support below ~0.1065 mag/kpc. The new
script uses that normalized threshold, so it does not depend artificially on
the chosen number of Monte Carlo samples. The AV valid-sample support cut is
normalized separately. Correctly matching an exact published footprint also
requires matching display geometry, sampling and source product/version.

## Run

```bash
tar -xzf cg10_zg_1024_fourpanel_raw_20261013_v2.tar.gz

python3 scripts/cosmicgrain_zg_1024_fourpanel_raw_v2.py \
  data/Rv_map_new.h5 \
  --output-dir halo441_output_1024_mingashsml0.0 \
  --history analysis_products/cg10_tracking_fixed_ids/cg10_1024_target_history.csv \
  --snap 29 --phis 0 45 90 135 180 225 270 315 \
  --pixel 0.25 --dz 0.2 --zmax 0.4 --zg-samples 256 \
  --outdir analysis_products/zg_cg10_1024_fourpanel_raw_v2
```

No smoothing is added to the simulation. Dust and stellar orientation are read
once, then reused for every orientation. Prepared input arrays are cached;
repeating the command with new phis, display cuts or color limits avoids
snapshot/ZG rereads. --refresh forces recomputation. Dependencies and units
remain as documented for v1: corrected halo_utils, numpy, scipy, h5py,
matplotlib, astropy, astropy-healpix; nm grain radii; standard CG code mass
1e10 Msun/h. The new script uses the same uniquely named optics helper.

## Send back

- orientation_support.csv
- orientation_support.png
- map_metadata.json
- one or two raw four-panel maps, retaining their orientation labels

The CSV reports each region's particle count, occupied-pixel fraction, mean
count in occupied pixels, pixels with >=5 and >=10 dust particles, and common
ZG/CG RV pixels. Every orientation is retained rather than choosing the one
that most resembles the Milky Way.

## Interpretation

The old attractive 40-kpc-wide figure was a galaxy-wide projection. These
maps are 5-kpc-wide local patches near an observer at 8.2 kpc with only 0.8 kpc
vertical depth. Their sparse raw particle support does not mean that the full
galaxy lacks dust. Point-particle binning can also be much sparser than a
continuous reconstructed observational field. This diagnostic does not spread
each particle over a kernel or fill holes. Sparse pixels constrain what spatial
statistics can be measured at 250 pc in those regions.

The normalized ZG mask should reveal genuine low-weight regions that v1
displayed as colored pixels. It is not a guarantee of identical gap boundaries
to the authors' much finer map: v2 still displays 250 pc pixels over radius
2.5 kpc instead of the authors' ~26 pc pixels over radius 2.6 kpc. No holes are
painted in by hand. --zg-rv-weight-floor 0 disables this mask for comparison.

The observational data version must also be checked. The Zenodo description
calls Rv_map_new.h5 preliminary; a later paper figure may use a different
product. The v5 record lists its MD5 as 8ac94ff428c02fbac9f5b783568a0bf2.
File-version verification is optional here and not performed by this script.

Top maps are mean differential AV in mag/kpc; bottom maps are AV-weighted
local RV through the vertical slab, not cumulative extinction from an observer
to a star. Raw and common-footprint figures are both retained. Gray means
masked/unavailable/empty; saved CG AV zeros are not converted to fake signal.
The native physical resolutions are not equal merely because the display
pixel size is shared. The script does not compute spectra or claim convergence.

Validation: synthetic HDF5/HEALPix test confirmed the normalized RV-weight
mask, mask-off baseline, raw single-particle support, count/coverage diagnostics
and plotting without Gaussian smoothing. Real orientation support still needs
to be measured on cygnus.
