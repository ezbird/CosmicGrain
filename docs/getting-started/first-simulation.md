# First Simulation

<div class="cg-card summary" markdown="1">

A narrated walkthrough of a real CosmicGrain run — not just proving the build works, but understanding the choices behind it. This page assumes you've already completed [Quick Start](quick-start.md).

</div>

## Generating initial conditions

<div class="cg-card implementation" markdown="1">

Use one of the accepted MUSIC2 files described in [Zoom
Setup](zoom-setup.md). A dust-enabled IC begins with gas and dark matter but an
empty `PartType6`; stellar feedback creates dust during evolution.

</div>

## A minimal `.param` file

<div class="cg-card implementation" markdown="1">

Generate the file with `scripts/generate_params.py` from
`params/param_base.txt` rather than writing one by hand (see [Parameter
Files](../running/parameter-files.md)). For an inexpensive validation run:

1. use a \(512^3\)-equivalent IC (e.g. `params/param_halo441_512.txt`);
2. shorten the output list while retaining early, first-star, first-dust, and
   late checkpoints;
3. keep the physics switches and calibrations unchanged;
4. keep `CpuTimeBetRestartFile` so restart files are written; and
5. archive the parameter file (and `parameters-usedvalues`) with the output.

A \(512^3\) halo 441 run to z = 0 takes about 1.5–2 h on 24 ranks.

</div>

## Validation sequence

After startup, verify increasingly demanding milestones:

1. evolution before star formation, with no stars or dust;
2. first star formation and element enrichment;
3. first SNII/LRN dust creation;
4. delayed AGB enrichment and dust creation;
5. restart continuity across an active feedback interval; and
6. whole-box mass, element, and dust-composition closure at the final output.

The current whole-box benchmark (Halo 441 / \(1024^3\), v1.2.0) is summarized in
[Validation](../validation/index.md).
