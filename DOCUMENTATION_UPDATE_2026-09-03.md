# CosmicGrain Documentation Update — 2026-09-03

## Scope

This refresh updates the documentation from the original single-halo workflow
to the current 12-halo, four-resolution CosmicGrain program.

## Major changes

- Replaced the historical Halo 569 zoom instructions with the current parent
  census, candidate selection, Lagrangian tracing, MUSIC2 configuration,
  suite-generation, seven-type post-processing, and validation workflow.
- Added the complete 13-item IC validation checklist and recorded the accepted
  48 PASS / 0 WARN / 0 FAIL suite result.
- Documented the seven-element GADGET header and intentionally empty
  PartType6 IC layout.
- Added current delayed SNII/hypernova feedback, MESA AGB enrichment,
  stochastic heating, LRN injection, and stellar/dust metal-partition
  conventions.
- Updated grain growth, sputtering, shock destruction, coagulation, and
  shattering for evolving carbon/silicate composition and Hsml-aware local
  association.
- Filled the operational Running, Analysis, Validation, and Reference pages
  that were previously empty.
- Added aperture, unit, halo-centering, D/G, D/Z, extinction, and SKIRT
  reproducibility guidance.
- Updated MkDocs navigation and removed the missing favicon reference.

## Validation performed

- `mkdocs.yml` parsed successfully as YAML.
- All 44 navigation targets exist and are nonempty.
- All relative Markdown links resolve.
- All fenced code blocks are balanced.
- Every populated Markdown page has an H1 heading.
- No `TODO`, `under construction`, obsolete
  `run_music2_suite_updated.sh`, or obsolete `run_radial_evolution.py`
  references remain in the maintained pages.

A full MkDocs Material render was not run in the editing workspace because the
site-builder dependency was unavailable. Run `mkdocs build --strict` in the
project's documentation environment before publishing.
