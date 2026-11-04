# Developer Guide

CosmicGrain is a GADGET-4-based simulation code with dust physics coupled to particle creation, gas interactions, time integration, feedback/enrichment, I/O, and analysis-facing fields. It is therefore better thought of as a complete code base than as a drop-in plug-in.

The long-term development goal is to keep dust-specific physics **localized, understandable, and hackable**, with narrow interfaces to the underlying GADGET-4 machinery.

## Development principles

- keep physical prescriptions separate from generic GADGET-4 infrastructure where practical;
- group code by physical responsibility rather than by historical implementation order;
- minimize duplicated unit conversion, neighbor-search, and bookkeeping logic;
- expose new runtime parameters and output fields clearly;
- preserve restart behavior and conservation when adding mass-transfer processes; and
- add a focused validation test for every new physical mechanism.

Start with [Current Architecture](architecture.md), then see [Adding New Physics](adding-new-physics.md).
