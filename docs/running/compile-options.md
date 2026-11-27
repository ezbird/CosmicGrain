# Compile-Time Options

Runtime parameters cannot activate code that was not compiled. A dust-enabled
zoom requires `NTYPES=7`, `DUST`, `IDS_64BIT`, `COOLING`, `STARFORMATION`,
`FEEDBACK`, and `METALS`.

The maintained zoom configuration, `configs/Config_zoom.sh`, also uses
periodic cosmological boundaries, mixed double precision, adaptive hydro
softening, the cubic-spline kernel, zoom-optimized TreePM
(`PMGRID=512`, `PM_ZOOM_OPTIMIZED`, `TREEPM_NOTIMESPLIT`), and FOF/SUBFIND.
Nodes running more than 64 ranks need `NUMBER_OF_MPI_LISTENERS_PER_NODE` > 1 (3 on Colibri); see
[Compilation](../getting-started/compilation.md).

Some behaviour is switched at run time by environment variables rather than
compile flags; see [Parameter Files](parameter-files.md#environment-overrides).

Archive the config file, `Makefile.systype`, the compiler identity, and the
executable checksum with every run. See the full [compile-flag
reference](../reference/compile-flags.md).
