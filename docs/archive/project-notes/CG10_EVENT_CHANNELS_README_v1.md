# CG10 destruction event channels

This streams the 24 rank logs directly, without concatenating them or reading
simulation snapshots. Memory scales with missing central dust IDs, not all log
rows. It reuses the transition caches at 022, 023 and 024.

From gadget4, after extracting the archive:

```bash
tar -xzf cg10_event_channels_20261013_v1.tar.gz

python3 scripts/cosmicgrain_cg10_event_channels_v1.py \
  halo441_output_2048_mingashsml0.0 \
  --cache-dir analysis_products/cg10_dust_transition_v1/cache \
  --outdir analysis_products/cg10_event_channels_v1
```

Send channel_summary.csv and event_coverage.json. If malformed_rows is nonzero,
also send malformed_examples.txt. Detailed matches are in matched_events.csv;
there is no need to upload that larger file initially.

The default cosmology/unit values match CG10: h=0.6732, BoxSize=50000 ckpc/h,
code mass unit=1e10 Msun before dividing by h. Command-line overrides are
available if analysing different units. Dependencies: Python, numpy.

## What is measured

- Missing IDs are selected from the original central sphere, r<30 pkpc,
  against the complete later dust inventory in the cache.
- Event scale factor must lie in the appropriate snapshot interval, allowing
  5.1e-7 at boundaries because event_a was printed to six decimal places.
- Each missing particle has an exact integer ID, including IDs above 2^63
  printed as signed values by the C++ logger.
- Original snapshot mass is assigned to a channel only for an unambiguous
  record. Exact duplicate records are removed; multiple distinct records
  remain a separate ambiguous category.
- The summary includes original mass, logged mass at removal, grain radius,
  carbon fraction, gas density and approximate galaxy-centred event radius.
- Original-mass fractions refer to the MISSING cohort, not all original
  central dust and not the net aperture mass decline.
- All-rank global channel counts are raw log-row counts, not deduplicated
  unique physical events. Interpret the cohort summaries for CG10 identity.

## Logger-specific cautions

The supplied implementation does not gate event rows on DustDebugLevel. It
does skip cleanup events, thermal AGB-source particles below 1e-13 code mass,
and any particle whose stored BirthPos is exactly zero. Log headers listing
cleanup as an event type therefore do not imply that cleanup is recorded.
Missing events cannot automatically be assigned to thermal destruction.

These rows are full-removal attempts, not partial erosion. The logged mass and
radius are taken before the removal update. A radius above the 1 nm floor in
the log does not mean the predicted updated radius stayed above the floor.
The removal helper can decline removal if no gas is available, so log rows
alone are not unconditional proof of deletion; disappearance in the later
snapshot supplies additional evidence. Distinct repeated records are flagged.

Birth-to-event displacement includes galaxy bulk motion and is not a wind
distance. Event radius uses a galaxy centre interpolated linearly in scale
factor between the two audited centres, so it is only approximate over these
long intervals. Gas density remains in stored code units; zero is documented
as unavailable. There is no gas temperature column, and this analysis cannot
measure the detailed hot-gas or shock history.

Rank/time coverage is reported, but a file's min/max event times do not certify
continuous logging. Check source/build agreement and restart coverage. Fresh
starts use write mode; resumes append. This parser handles repeated exact
cohort records without counting them twice, but cannot reconstruct overwritten
records or missing ranks. Different log formats are reported as malformed.

Validation: synthetic logs/caches passed signed uint64 handling, exact-duplicate
removal, snapshot interval filtering, ambiguous channel handling, malformed
row detection, rank coverage and original mass accounting closure.
The real rank files remain on cygnus and have not been scanned here.
