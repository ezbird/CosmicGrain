# Physics Ladder

The S0–S10 ladder isolates the effect of successively enabled dust processes.
Each rung inherits all processes above it in the table.

| Run | Process added |
| --- | --- |
| S0 | Stellar dust creation |
| S1 | Dust temperature (`DustEnableCooling`) |
| S2 | Gas–dust drag |
| S3 | Astration |
| S4 | Thermal sputtering |
| S5 | ISM grain growth |
| S6 | Subgrid clumping (`DustEnableClumping`, mode 2 by default) |
| S7 | SN shock destruction |
| S8 | Coagulation |
| S9 | Shattering |
| S10 | Radiation pressure; full ladder |

Sputtering runs inside the drag routine, so S4 and above need drag on, as the
ladder already has. Clumping changes the density model of growth,
coagulation, and shattering together.

The ladder is a controlled process comparison, not a chronological claim
about which process is physically more fundamental. Compile options,
resolution, initial conditions, star formation, feedback, output cadence, and
all parameters unrelated to the newly enabled process should remain fixed.

When comparing rungs, report dust mass, D/G, D/Z, grain-size distribution,
carbon fraction, stellar source tags, radial structure, and survival in the
ISM/CGM. A difference in one observable can arise indirectly through changes
to dust abundance or spatial transport, so interpret SED and extinction
changes alongside the underlying particle distributions.
