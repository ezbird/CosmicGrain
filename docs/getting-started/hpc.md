# HPC Systems

<div class="cg-card summary" markdown="1">

Platform-specific setup and known issues for the clusters CosmicGrain has been run on. If something here doesn't apply to your system, it probably belongs on [Installation](installation.md) or [Compilation](compilation.md) instead — this page is only for things that are true on one specific machine.

</div>

## TACC Stampede

<div class="cg-card implementation" markdown="1">

<!-- TODO: module load commands, SKX partition name/specs, recommended
     node/task counts, sbatch template -->

**Known issue — restart-file portability.** Restart files are not portable across different MPI task counts on Stampede; restarting with a different task count than the original run hangs. Use snapshot-based restarts instead when changing task count.

</div>

## NMSU Discovery Cluster

<div class="cg-card implementation" markdown="1">

<!-- TODO: module load commands, partition names, recommended settings -->

**Known issue — persistent hang.** A hardware/MPI-stack issue has been identified as the cause of certain hangs on this cluster; <!-- TODO: any workaround or status update -->.

</div>

## Desktop / workstation builds

<div class="cg-card implementation" markdown="1">

<!-- TODO: any settings that differ from cluster builds — e.g. MaxMemSize,
     recommended core counts for the 24-core desktop setup -->

</div>
