#!/bin/bash
#SBATCH --account=16-cosmic_grain
#SBATCH --partition=general-cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=190
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3500M
#SBATCH --time=10:00:00
##SBATCH --mail-user <your email>   (set on the cluster copy)
#SBATCH --mail-type FAIL
#SBATCH --output=slurm-%x-%j.out
# One run of the 1024^3 C-S calibration grid.  $1 = param file.
# Uses the binary snapshotted at submit time ($CSGRID_EXE), so rebuilding
# ~/CosmicGrain-br06 while the chain runs cannot change the grid's code.
export OMP_NUM_THREADS=1
module load sstack/latest
module load spack/cosmicgrain
spack load openmpi@5.0.10 fftw@3.3.10 hdf5@1.14.6 gsl@2.8
GSL_PREFIX=$(spack location -i gsl@2.8)
FFTW_PREFIX=$(spack location -i fftw@3.3.10)
HDF5_PREFIX=$(spack location -i hdf5@1.14.6)
OMPI_PREFIX=$(spack location -i openmpi@5.0.10)
export LD_LIBRARY_PATH="${GSL_PREFIX}/lib:${FFTW_PREFIX}/lib:${HDF5_PREFIX}/lib:${OMPI_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd ~/CosmicGrain-br06
srun --export=ALL --mpi=pmix_v5 --unbuffered "$CSGRID_EXE" "$1" 0
rc=$?
OUT=$(awk '/^OutputDir/{print $2}' "$1")
# Restart files are only needed to resume; drop them once the run reached z=0.
if [ $rc -eq 0 ] && [ -d "$OUT/snapdir_029" ] && [ -d "$OUT/groups_029" ]; then
  rm -rf "$OUT/restartfiles" && echo "removed $OUT/restartfiles"
fi
exit $rc
