# CG10 dust transition diagnostic

Copy cosmicgrain_cg10_dust_transition_v1.py into gadget4/scripts/, beside the
corrected halo_utils.py already installed. No optics tables or extra plotting
scripts are required. Dependencies are h5py, numpy, matplotlib.

Start with the 2048 event only:

```bash
python3 scripts/cosmicgrain_cg10_dust_transition_v1.py \
  halo441_output_1024_mingashsml0.0 \
  halo441_output_2048_mingashsml0.0 \
  --history-1024 analysis_products/cg10_tracking_fixed_ids/cg10_1024_target_history.csv \
  --history-2048 analysis_products/cg10_tracking_fixed_ids/cg10_2048_target_history.csv \
  --runs 2048 --snapshots 22 23 24 \
  --outdir analysis_products/cg10_dust_transition_v1
```

The first path/history is required by the command interface, but is not read
when --runs 2048 is used. Existing audited CSV centres define each galaxy;
there is no repeated dark matter tracking or SO computation. Only gas, stars
and dust are scanned, with bounded chunks and sequential disk reads. All dust
IDs/radii/masses are retained compactly to match particles globally. Memory
use depends on the dust population and requested number of epochs. Cache files
are saved per completed output, so interrupted work can be resumed by repeating
the same command. Cache validity includes file sizes/mtimes and history rows;
--refresh forces a new scan.

Then expand the comparison (completed 2048 scans are reused):

```bash
python3 scripts/cosmicgrain_cg10_dust_transition_v1.py \
  halo441_output_1024_mingashsml0.0 \
  halo441_output_2048_mingashsml0.0 \
  --history-1024 analysis_products/cg10_tracking_fixed_ids/cg10_1024_target_history.csv \
  --history-2048 analysis_products/cg10_tracking_fixed_ids/cg10_2048_target_history.csv \
  --snapshots 21 22 23 24 29 \
  --outdir analysis_products/cg10_dust_transition_v1
```

## Send back

1. nested_budgets.csv
2. dust_ID_ledger.csv
3. scan_metadata.json
4. The 2048 snapshot 022, 023 and 024 50pkpc maps, and 250pkpc maps if helpful.

Maps use fixed simulation XY/XZ axes, not stellar face-on axes, to keep the
coordinate system stable during interactions. They show mass surface density,
not AV or RV, through cubes of half-size 50 or 250 pkpc. Color normalization
is shared across outputs within each run for a given particle type; different
particle types and different runs have independent scales. White R200/cyan
30 pkpc circles are projected guides, not spherical masks. Black/masked pixels
have no deposited particle mass; maps are not smoothed.

## What the products test

- Nested inventories at 10, 20, 30, 50, 100 pkpc, R200 and 2R200 measure whether
  the central deficit is accompanied by dust elsewhere in the halo.
- A second 30 pkpc census around the unrefined catalogue centre checks the
  sensitivity to the centring choice.
- The direct 30 pkpc mass census must reproduce the audited histories to
  1e-5 relative tolerance. A disagreement stops execution rather than mixing
  incompatible definitions.
- Gas metal mass and SFR use recognized scalar snapshot fields. Missing
  fields produce NaN, not zero, and are recorded in scan_metadata.json.
- The dust ID ledger follows particles originally inside 30 pkpc to their
  positions anywhere in the next requested snapshot. Matching uses exact
  uint64 IDs and refuses duplicate IDs. No cross-resolution particle matching
  is attempted.
- Each category reports both the original mass and later mass of the matched
  particles, so mass evolution is separated from their change in radius.
- Later central IDs absent from the earlier aperture include new dust and
  inward arrivals; those two cases are not separated in this first diagnostic.

For surviving IDs, the later central mass equals the sum of
still_inside30.later_mass_Msun and
later_central_IDs_not_in_source_aperture.later_mass_Msun.
For the initial cohort, source masses sum over the five matched location bins
and ID_not_found_anywhere to its initial mass.

## Interpretation limits

Increasing radius relative to a moving galaxy centre supports redistribution
relative to that galaxy. It does not by itself identify a wind or its driver.
Missing IDs are not automatically destruction: merging, splitting, astration,
deletion, or ID changes can affect particle identity. Surviving mass changes
are net changes over the interval, not individual growth/destruction rates.
Particles beyond R200 can belong to companions or nearby galaxies. Whole-
snapshot dust mass does not belong exclusively to CG10.

Aperture budget changes can also reflect companion passages, gas phase
changes, centring, or interaction geometry. The stellar 30 pkpc dip at snapshot
023 is a reason to inspect the maps before attributing the event to feedback.
These inventories do not measure grain-size distributions or RV directly.

The old MinGasHsml=1 2048 run ends at snapshot 012, so cannot resolve this
late event. Both completed runs here have MinGasHsml=0 and different physics
parameters. A conclusion about higher resolution causing dust expulsion is
not supported by this pair alone.

## Usability decision

A dust-poor central galaxy is not itself evidence of numerical failure.
Use this diagnostic to distinguish an aperture effect from redistribution or
mass changes. Separately, a production-run numerical audit should inspect
termination/restarts, numerical warnings, mass/metal conservation, parameter
history and executable/build provenance. That audit has not been completed
by this script. The good halo identity and dust-rich earlier phase support
continuing the investigation; they do not certify the run for every paper use.

The different recorded physics settings mean this 2048 run can potentially
serve as a separate physical model, but cannot be presented as a controlled
resolution test against this 1024 run. Its near-zero z=0 SFR also needs to be
considered when choosing a Milky Way comparison model.

Validation here: synthetic HDF5 scan checked units, periodic wrapping,
constant/variable masses, missing-field handling, cache reuse, history mismatch
rejection and map rendering. ID-ledger test checked IDs above 2^63, outward
relocation, surviving mass changes, absent IDs, new central IDs and mass closure.
Real simulation files are on cygnus and have not been scanned here.
