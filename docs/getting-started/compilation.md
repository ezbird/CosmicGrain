# Compilation

<div class="cg-card summary" markdown="1">

This page covers configuring and building CosmicGrain itself — the steps you repeat every time you change which physics modules are active, adjust resolution, or switch between production and debug builds. It assumes [Installation](installation.md) is already done.

</div>

## Configuring `Config.sh`

<div class="cg-card implementation" markdown="1">

CosmicGrain's physics modules are enabled at compile time via `Config.sh`. The following flags are required for a standard dust-enabled run:

| Flag | Purpose |
|---|---|
| `DUST` | Enables the CosmicGrain dust physics module (creation, transport, growth, destruction, radiation pressure). |
| `FEEDBACK` | Enables stellar feedback (Type II SNe and AGB winds). |
| `STARFORMATION` | Required for metallicity- and SFR-dependent dust gating (e.g. molecular fraction, clumping). |
| `IDS_64BIT` | **Required.** 32-bit particle IDs silently overflow once dust superparticle counts grow large — this does not raise a compile or runtime error, it just corrupts particle bookkeeping. Always enable this for dust-enabled runs. |
| `NTYPES=7` | **Required.** Reserves PartType6 for live dust. |
| `METALS` | Enables metallicity and element-resolved enrichment. |
| `COOLING` | Enables the adopted gas-cooling model. |

Optional flags, useful for debugging stellar feedback energy deposition:

| Flag | Purpose |
|---|---|
| `FEEDBACK_LIMIT_DULOG` | Clamps \|Δlog₁₀ u\| per heating event — useful when diagnosing runaway temperatures from feedback. |
| `FEEDBACK_T_CAP` | Applies a soft temperature cap for debugging; not intended for production runs. |

The current zoom build also uses `PERIODIC`, `DOUBLEPRECISION=1`,
`ADAPTIVE_HYDRO_SOFTENING`, `WENDLAND_C4`, `PM_ZOOM_OPTIMIZED`,
`TREEPM_NOTIMESPLIT`, and resolution-appropriate `PMGRID`/`HRPMGRID`.

</div>

## Building

<div class="cg-card implementation" markdown="1">

CosmicGrain compiles with `mpicxx`, requiring C++11 and the libraries set up in [Installation](installation.md). A representative per-file compile invocation:

```bash
mpicxx -std=c++11 -O3 -march=native -mtune=native \
    -ffast-math -fno-math-errno -fno-trapping-math \
    -funroll-loops -Wall -Wno-format-security -Wno-unused-result -flto \
    -I/usr/include/hdf5/serial \
    -I/path/to/gsl/build/include \
    -I/path/to/fftw3/build/include \
    -Ibuild -Isrc \
    -c src/<file>.cc -o build/<file>.o
```

In practice you won't invoke this directly — the top-level `make` handles it for every source file:

```bash
make -j8
```

The executable name and location follow the target configured by the current
top-level Makefile. Record `sha256sum` of the production executable with the
run metadata rather than assuming its name proves which source was compiled.

**Optimization flag note:** `-ffast-math`, `-fno-math-errno`, and `-fno-trapping-math` relax IEEE floating-point compliance for speed. This is standard practice for production N-body/hydro codes and shouldn't affect the dust physics (which relies on exact exponential/analytical forms rather than delicate floating-point edge cases), but keep it in mind if you're ever debugging a numerical discrepancy that seems to depend on compiler flags.

</div>

## Verifying the build

<div class="cg-card implementation" markdown="1">

On startup, a successful build with dust physics enabled prints a one-time
flag summary confirming the parameter file was read correctly:

```text
[DUST_FLAGS|Step=0] Creation=1 Drag=1 Growth=1 Coagulation=1
Sputtering=1 ShockDestruction=1 Astration=1 RadPressure=1
Clumping=1 Cooling=1
```

Check that these flags match what you set in your `.param` file — a `0` where you expected `1` usually means a parameter name mismatch (e.g. a typo in `DustEnableGrowth`) rather than a compilation problem.

</div>
