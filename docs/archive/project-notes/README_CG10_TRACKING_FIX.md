# CG10 fixed-final-ID audit — version 2, 2026-10-13

The actual direct audit found:
- Final snapshot 29 group 1: 490,311 catalog-sphere HRDM IDs, all present at 12.
- Snapshot 12 group 2: zero shared final IDs (and zero shared final-core IDs).
- Snapshot 12 group 3: 46,155 shared final IDs and 39,170 shared final-core IDs.
- Snapshot 12 group 1: 18,948 shared final IDs; 95.9% of its HRDM population
  enters the selected final SO sphere.
- Snapshot 13 group 3's mutable population does not establish the dominant
  final contributor at snapshot 12. The previous checkpoint 12:2 was invalid
  for the selected final halo; spatial matching across two runs at one epoch
  was not sufficient ancestry evidence.

Version 2 retains one immutable ID cohort from the canonical particle-SO final
halo. Each requested earlier epoch is ranked independently by contribution to
that final cohort. This is the dominant final contributor at each epoch, NOT
necessarily the same branch definition as a recursive bound-particle merger
tree. A full merger tree and other contributing branches are not reconstructed.
Candidate runner-up counts, final-core overlap, and earlier-to-final purity
are exported so switches/ambiguities can be reviewed.

Validation: at least 16 shared IDs, at least 95% of the fixed IDs found globally,
positive catalog SO, canonical particle/catalog SO mass ratio between 1/3 and
3, and at least 25% of the earlier SO HRDM population in the final cohort.
Candidates within 90% of the best count are ambiguous gaps unless an explicitly
trusted checkpoint has at least half the winner's count and passes all gates.
Unlike adjacent-epoch tracking, no 5% minimum of the final halo is imposed:
early progenitors can be a small fraction of the eventual halo.

Numeric chunk ordering and fixed-aperture fixes from v1 remain. Legacy Halo 569
APIs are retained. The generic zoom history API uses the new fixed-cohort
selector. Sparse probes now visit only requested epochs/checkpoints. Full
histories visit all requested epochs once. Tracking CSVs are written atomically
after each step, including error rows, so failed probes preserve diagnostics.
No simulation data were available locally; fourteen regression tests passed.

## Install (from gadget4)

```bash
mkdir -p scripts/backups/cg10_tracking_v1_20261013
cp scripts/halo_utils.py scripts/list_stars_by_snapshot.py scripts/backups/cg10_tracking_v1_20261013/
tar -xzf cg10_tracking_fix_20261013_v2.tar.gz
python3 -m unittest discover -s tests -p 'test_tracking_audit.py' -v
```

## Probe only 012, 013, and 029

```bash
python3 scripts/list_stars_by_snapshot.py \
  halo441_output_2048_mingashsml0.0 \
  --scope target --target-mode all --target-snapshots 12 13 29 \
  --group-index 1 --reference-snap 29 --tracking-checkpoint 12:3 \
  --galaxy-aperture-pkpc 30 --tracking-verbose \
  --csv analysis_products/cg10_tracking_fixed_ids/cg10_2048_probe.csv
```

Return cg10_2048_probe.csv and cg10_2048_probe_tracking.csv. Group 3 at snapshot
12 has direct final-halo support; canonical refinement/SO and purity gates still
need to pass. Do not remove a failing checkpoint merely to get a figure.

## Compare the previous 2048 run using the SAME final-ID reference

Both 2048 runs were recorded with the same IC filename. The external identity
reference is intended only for same-IC, same-resolution runs with conserved IDs;
do not use this option for the 1024-versus-2048 comparison.

```bash
python3 scripts/list_stars_by_snapshot.py \
  halo441_output_2048 \
  --scope target --target-mode all --target-snapshots 12 \
  --identity-reference-output halo441_output_2048_mingashsml0.0 \
  --group-index 1 --reference-snap 29 \
  --galaxy-aperture-pkpc 30 --tracking-verbose \
  --csv analysis_products/cg10_tracking_fixed_ids/cg10_previous2048_probe.csv
```

Group 1 here refers to the identity reference at snapshot 29 in the completed
run. The older run need not contain snapshot 29. Its snapshot-12 contributor
index is discovered from the same final ID cohort and exported in its CSV.
This compares the corresponding ancestor rather than assuming old group 1.
Return both probe CSVs and both *_tracking.csv files.

## Full histories after the probe is reviewed

```bash
python3 scripts/list_stars_by_snapshot.py \
  halo441_output_2048_mingashsml0.0 \
  --scope target --target-mode all --history-batch-size 1 \
  --group-index 1 --reference-snap 29 --tracking-checkpoint 12:3 \
  --galaxy-aperture-pkpc 30 --tracking-verbose \
  --csv analysis_products/cg10_tracking_fixed_ids/cg10_2048_target_history.csv

python3 scripts/list_stars_by_snapshot.py \
  halo441_output_1024_mingashsml0.0 \
  --scope target --target-mode all --history-batch-size 1 \
  --group-index 1 --reference-snap 29 \
  --galaxy-aperture-pkpc 30 --tracking-verbose \
  --csv analysis_products/cg10_tracking_fixed_ids/cg10_1024_target_history.csv

python3 scripts/cosmicgrain_cg10_compare_1024_2048_audited.py \
  --csv-1024 analysis_products/cg10_tracking_fixed_ids/cg10_1024_target_history.csv \
  --csv-2048 analysis_products/cg10_tracking_fixed_ids/cg10_2048_target_history.csv \
  --output-dir CG10_1024_vs_2048_fixed_ids
```

The 1024 early ancestry has no independent checkpoint yet; review its exported
final-cohort evidence. Gaps are missing information, not zero star formation or
zero dust. Whole-volume equality does not prove dust transport from CG10.
The old 2048 comparison at snapshot 12 compared another galaxy across runs;
its floor response cannot be attributed to the selected final CG10 halo.

The supplied cg10_tracking_changes.patch compares v2 against your original
source archive. Existing cached CSVs are not modified by installation.
