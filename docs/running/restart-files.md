# Restart Files

Restart continuity is a required regression test because CosmicGrain creates
and destroys PartType6 particles while feedback and dust processes modify
multiple mass and element reservoirs.

## Safe restart practice

1. use the same executable, parameter file, MPI task count, and output
   directory unless the restart mode explicitly supports a change;
2. retain a complete restart set from one synchronized write;
3. inspect the log for a successful restart read before deleting older sets;
4. compare particle counts and total gas/star/dust masses immediately before
   and after restart; and
5. ensure source tags, grain radii, carbon fractions, stellar birth masses,
   feedback flags, and energy reservoirs persist.

Restart files may depend on MPI layout. When changing task count or machine,
prefer a supported snapshot-based continuation and validate it with a short
test rather than assuming binary restart portability.
