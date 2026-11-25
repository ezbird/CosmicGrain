#!/bin/bash
#SBATCH --account=16-cosmic_grain
#SBATCH --partition=general-cpu
#SBATCH --job-name=cg4096test
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=190
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3900M
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
##SBATCH --mail-user <your email>   (set on the cluster copy)
# First 4096^3 test of halo441: v1.2.0 release code, Paper II recipe, default SN DeltaT (10^7 K), UCX.
# Stops at TimeMax=0.2 (z=4) or TimeLimitCPU; restart files every 4 h are kept to measure their size.
# Build needs NUMBER_OF_MPI_LISTENERS_PER_NODE=3 (190 ranks per node).  $1 = param file.
export OMP_NUM_THREADS=1
module load sstack/latest
module load spack/cosmicgrain
spack load openmpi@5.0.10 fftw@3.3.10 hdf5@1.14.6 gsl@2.8
for pkg in gsl@2.8 fftw@3.3.10 hdf5@1.14.6 openmpi@5.0.10; do LD_LIBRARY_PATH="$(spack location -i $pkg)/lib:$LD_LIBRARY_PATH"; done; export LD_LIBRARY_PATH
export OMPI_MCA_pml=ucx UCX_NET_DEVICES=mlx5_0:1
unset OMPI_MCA_btl_tcp_if_include
env | grep -E "^(OMPI_MCA|UCX|CG_)" | sort
cd ~/CosmicGrain-br06
srun --export=ALL --mpi=pmix_v5 --unbuffered ~/jobexe/CosmicGrain.v120 "$1" 0
