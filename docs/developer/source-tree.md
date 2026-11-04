# Source Tree

The public source currently contains dust-specific code plus required interfaces to GADGET-4 subsystems such as feedback/enrichment, particle handling, time integration, and I/O. Consult the repository itself for the authoritative current paths: this page should not be used to infer that a planned modular directory already exists.

## Direction of travel

New dust code should preferentially live under the dust implementation rather than adding unrelated logic throughout the host code. The planned organization is documented in [Code Architecture](architecture.md).

When moving existing routines during refactoring, separate **code motion** from **physics changes** whenever possible. A pure reorganization should reproduce the same regression diagnostics before any new physical behavior is introduced.
