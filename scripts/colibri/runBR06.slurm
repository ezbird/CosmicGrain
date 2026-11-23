#!/bin/bash
#SBATCH --account=16-cosmic_grain
#SBATCH --partition=general-cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3500M
#SBATCH --mail-user USER@example.edu
#SBATCH --mail-type BEGIN
#SBATCH --mail-type END
#SBATCH --mail-type FAIL
#SBATCH --ntasks-per-node=190
#SBATCH --time=20:00:00
#SBATCH --output=slurm-%x-%j.out
export OMP_NUM_THREADS=1
module load sstack/latest
module load spack/cosmicgrain
spack load openmpi@5.0.10 fftw@3.3.10 hdf5@1.14.6 gsl@2.8
GSL_PREFIX=$(spack location -i gsl@2.8)
FFTW_PREFIX=$(spack location -i fftw@3.3.10)
HDF5_PREFIX=$(spack location -i hdf5@1.14.6)
OMPI_PREFIX=$(spack location -i openmpi@5.0.10)
export LD_LIBRARY_PATH="${GSL_PREFIX}/lib:${FFTW_PREFIX}/lib:${HDF5_PREFIX}/lib:${OMPI_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# Native InfiniBand (UCX): ~8x faster than TCP over ib0 for 2-node 2048 runs (benchmark 23 Nov 2026, jobs 786 vs 1193)
export OMPI_MCA_pml=ucx UCX_NET_DEVICES=mlx5_0:1
unset OMPI_MCA_btl_tcp_if_include
cd ~/CosmicGrain-br06
JOBEXE=~/jobexe/CosmicGrain.$SLURM_JOB_ID; mkdir -p ~/jobexe; cp ./CosmicGrain $JOBEXE
srun --export=ALL --mpi=pmix_v5 --unbuffered $JOBEXE "$1" 0
