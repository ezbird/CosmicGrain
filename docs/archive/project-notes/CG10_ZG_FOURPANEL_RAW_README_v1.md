# Zhang–Green / CG10 1024 unsmoothed four-panel maps

Recovered reference: cosmicgrain_zg_healpix_validation_v3_remade.py, dated
October 8. The new script uses the same stored NESTED HEALPix convention,
AV_map_diff/R55_map_diff, and volume-average bird's-eye interpretation.
It replaces the reference's fine regular volume sampling with reproducible
uniform volume samples per display pixel (seed 7). This is a new implementation
of the recovered approach, not the unchanged October 8 script.

## Run from gadget4

```bash
tar -xzf cg10_zg_1024_fourpanel_raw_20261013_v1.tar.gz

python3 scripts/cosmicgrain_zg_1024_fourpanel_raw_v1.py \
  data/Rv_map_new.h5 \
  --output-dir halo441_output_1024_mingashsml0.0 \
  --history analysis_products/cg10_tracking_fixed_ids/cg10_1024_target_history.csv \
  --snap 29 --phis 0 \
  --pixel 0.25 --dz 0.2 --zmax 0.4 \
  --zg-samples 256 \
  --outdir analysis_products/zg_cg10_1024_fourpanel_raw_v1
```

ZG path lookup also checks data/observations/zhang_green/ and scripts/ for
Rv_map_new.h5. Supply its actual path if elsewhere. Optics default to
data/dust_optics/callqcomp.out_D16MGemt and callqcomp.out_DH16asil.

This package installs only two new uniquely named scripts and this README.
It does not overwrite halo_utils.py, previous plotting scripts or run files.
The optics helper preserves the existing multiresolution mapper's table reader,
stellar angular-momentum orientation and opacity conventions. Dependencies:
numpy, scipy, h5py, matplotlib, astropy, astropy-healpix, plus the corrected
halo_utils.py already installed. cosmicgrain.mplstyle is applied if available.

## Figure and definitions

Left column: Zhang–Green. Right: CG10 1024 at z=0.
Top row: mean differential extinction, dAV/dd, in mag/kpc.
Bottom: differential-AV-weighted mean local RV through the vertical slab.

Both maps cover a local region of radius 2.5 physical kpc and |z|<0.4 kpc.
ZG is Sun-centred. CG is centred on a mock Sun at galactocentric radius 8.2 kpc,
with x toward the Galactic centre, y tangential, and z along stellar angular
momentum measured within 20 pkpc of the audited galaxy centre.

This is not a full-halo or full-galaxy face-on map, and not an all-sky plot.
HEALPix is used to query the native observational field before Cartesian
bird's-eye display. No cumulative AV_map_int or R55_map_int is summed or used.

CG uses raw 250x250x200 pc particle histogram deposition. There is no Gaussian
convolution, interpolation between pixels, or filling of empty bins. Finite
voxel sizes are unavoidable spatial averaging and are stated explicitly.
Empty CG bins remain zero in saved arrays and appear gray in the displayed
AV map; RV is undefined there. Each nonempty cell uses the same optics
convention as previous CG maps: R55=A550/(A440-A550), RV=1.1*R55+0.07.

ZG uses mean AV_map_diff in each display column and differential-AV weighting
for RV, as in the recovered bird's-eye method. Uniform volume sampling is
averaging within the display pixel/slab, not additional Gaussian smoothing.
Native ZG resolution varies with distance and source NSIDE; the same pixel
size is not proof of matched effective resolution. The logger reports source
NSIDE values and stored ordering; it refuses a RING-labelled input instead
of silently changing its interpretation.

The projected AV-weighted mean local RV is not the ratio of vertically
integrated screen extinctions, nor cumulative line-of-sight RV from the Sun.
The labels and metadata retain that distinction. The top row is dAV/dd,
not AV in magnitudes; multiplying by the 0.8-kpc slab thickness gives the
corresponding projected column extinction under the adopted geometry.

## Outputs

- *_raw.png/pdf: each dataset's native footprint; useful for seeing support.
- *_common.png/pdf: same common finite RV footprint applied to all four panels.
- *_maps.npz: unsmoothed fields, CG particle counts, ZG valid-sample support,
  and Cartesian pixel edges, for later mask/statistics checks.
- map_metadata.json: centre, orientation, ordering, sampling, map counts and
  physical/measurement limitations.

Send the raw PNG and map_metadata.json first; common PNG is useful too.
Do not upload the original large HDF5 or rerun full history analysis.

## Choices to review before paper use

1. Orientation 0 degrees is preselected, not chosen for a visually appealing
   outcome. To assess variation, use --phis 0 45 90 135 180 225 270 315; dust
   and stars are read once and reused for all orientations.
2. Native defaults retain one-particle CG pixels and impose no AV reliability
   threshold. These show what is deposited, not statistically certified RV.
   --min-dust-count 5 can make a support diagnostic without smoothing. No
   percentile cut is silently imposed to make the maps agree.
3. Shared color scales: RV 2.5–6 initially, with extended colorbars; AV uses a
   common pooled P99 display maximum, so bright tails can saturate. Arrays are
   never clipped. Set --rv-min/--rv-max and --av-max to show alternate ranges.
4. Monte Carlo sampling can be checked with --zg-samples 1024 in a different
   outdir using the same seed. Raising sample count cannot improve the
   underlying observational resolution. Do not interpret sampling fluctuations
   as physical power on small scales.
5. Review finite support and masks before power spectra/structure functions.
   This script intentionally produces maps, not unqualified spectra.
6. The ZG support fraction measures valid sampled volume, not observational
   uncertainty or a posterior significance cut. Partial coverage may bias a
   conditional mean, so native and common footprints are both exported.

## Validation and remaining limits

Synthetic HEALPix/HDF5 tests passed stored-order checks, constant AV volume
sampling, R55-to-RV conversion, raw single-particle deposition, preservation
of empty pixels, common-footprint selection and four-panel rendering.
The existing optical-table/rotation helper was recovered and retained.
The actual ZG HDF5, optics tables and CG10 snapshots remain on cygnus, so the
real four-panel result has not been generated or visually checked here.
