# MUSIC2 IC-Suite Validation

The 12-halo CosmicGrain suite contains four MUSIC2 zoom ICs per target:
\(512^3\), \(1024^3\), \(2048^3\), and \(4096^3\)-equivalent resolution. The
suite validator checks every file independently and then tests relationships
across halos and resolutions.

## Validation checklist

Every accepted suite must pass all of the following:

1. Seven-element GADGET header consistency
2. Correct particle counts and dataset dimensions
3. Valid empty `PartType6`
4. Finite coordinates, velocities, and masses
5. Coordinates inside the simulation box
6. Positive particle masses
7. Exact global particle-ID uniqueness
8. Gas/high-resolution-DM spatial overlap
9. Cosmological high-resolution-DM-to-gas particle-mass ratio
10. Coarse-particle mass hierarchy
11. Consistent cosmology across the suite
12. Factor-of-eight particle-mass improvement between successive resolutions
13. Consistent gas and high-resolution-DM mass resolution across halos

## Running the validator

```bash
python3 ~/gadget4/scripts/validate_music2_ic_suite.py \
    --ic-root ~/gadget4/ICs
```

By default, numerical fields are scanned in chunks while the particle-ID test
is exact. Files are processed sequentially. The largest file can temporarily
require roughly 1–2 GB of memory during the uniqueness test.

The machine-readable report is written to:

```text
~/gadget4/ICs/MUSIC2_logs/ic_suite_validation.csv
```

`PASS` means that all file-level checks succeeded. `WARN` identifies a
nonfatal condition that should be understood before production. `FAIL`
identifies a structural, numerical, or suite-consistency error. The process
returns a nonzero exit status on failure; `--strict` also treats warnings as a
failed run.

## Accepted September 2026 suite

The production suite completed with:

```text
Files: 48 PASS, 0 WARN, 0 FAIL (48 total)
FINAL STATUS: PASS
```

Initial particle counts range from 2,120,948 (halo 3879 at \(512^3\)) to
517,537,024 (halo 1534 at \(4096^3\)). This range reflects the different
volumes and geometries of the refined Lagrangian regions; it is not itself a
validation failure.

## What this does not establish

The IC audit does not test:

- the production parameter file or compiled executable;
- runtime memory and load balance;
- low-resolution contamination of the evolved target halo;
- convergence of galaxy structure or dust observables; or
- whether a selected halo remains scientifically useful after evolution.

Those require startup tests, post-run zoom audits, and resolution comparisons.
