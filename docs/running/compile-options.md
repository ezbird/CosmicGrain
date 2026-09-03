# Compile-Time Options

Runtime parameters cannot activate code that was not compiled. A standard
dust-enabled zoom requires `NTYPES=7`, `DUST`, `IDS_64BIT`, hydrodynamics,
cooling, star formation, feedback, and metal tracking.

The current production configuration also uses periodic cosmological
boundaries, double precision, adaptive hydro softening, the Wendland C4
kernel, and zoom-optimized TreePM options.

Archive `Config.sh`, `Makefile.systype`, the compiler identity, and the
executable checksum with every run. See the full [compile-flag
reference](../reference/compile-flags.md).
