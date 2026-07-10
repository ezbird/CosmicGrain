#!/bin/bash
# Downloads observational datasets for the M_dust vs M_star plot.
# Run from your scripts/ directory.

# Note, we also use the data/simba/m50n512_151.hdf5 file from Simba for comparison, 
# but it is too large for the repository (~200MB)

set -e
OUTDIR="obs_data"
mkdir -p "$OUTDIR"

CDS="https://cdsarc.cds.unistra.fr/ftp"

echo "=== Galliano+2021 (DustPedia 798 galaxies, J/A+A/649/A18) ==="
wget -q --show-progress -O "$OUTDIR/galliano2021_tableh1.dat" \
  "$CDS/J/A+A/649/A18/tableh1.dat"
wget -q -O "$OUTDIR/galliano2021_ReadMe" \
  "$CDS/J/A+A/649/A18/ReadMe"
echo "  -> $OUTDIR/galliano2021_tableh1.dat"

echo ""
echo "=== Remy-Ruyer+2015 (DGS+KINGFISH 109 galaxies, J/A+A/582/A121) ==="
# table4.dat = stellar masses (logM*, bytes 25-29)
# table9.dat = dust masses   (logMdust, need to check ReadMe)
wget -q --show-progress -O "$OUTDIR/remyruyer2015_table4.dat" \
  "$CDS/J/A+A/582/A121/table4.dat"
wget -q --show-progress -O "$OUTDIR/remyruyer2015_table9.dat" \
  "$CDS/J/A+A/582/A121/table9.dat"
wget -q -O "$OUTDIR/remyruyer2015_ReadMe" \
  "$CDS/J/A+A/582/A121/ReadMe"
echo "  -> $OUTDIR/remyruyer2015_table4.dat  (stellar masses)"
echo "  -> $OUTDIR/remyruyer2015_table9.dat  (dust masses)"

echo ""
echo "=== DustPedia CIGALE CSV (Nersesian+2019, 815 galaxies) ==="
wget -q --show-progress -O "$OUTDIR/dustpedia_cigale_results.csv" \
  "http://dustpedia.astro.noa.gr/Content/tempFiles/cigale/dustpedia_cigale_results_final_version.csv" \
  || echo "  [WARNING] DustPedia CIGALE CSV failed — server may require login. Skipping."

echo ""
echo "Done. Next: python parse_obs_data.py obs_data/"
