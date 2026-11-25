#!/bin/bash
#SBATCH --account=16-cosmic_grain
#SBATCH --partition=general-cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=190
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3500M
#SBATCH --time=10:00:00
##SBATCH --mail-user <your email>   (set on the cluster copy)
#SBATCH --output=slurm-%x-%j.out
# 1024^3 C-S run at SN DeltaT = 10^7.5 K on 1 node.  $1 = param file.
# Binary: ~/jobexe/CosmicGrain.786 (2429f28 code = CG_FEEDBACK_DELTA_T override).
export OMP_NUM_THREADS=1 CG_FEEDBACK_DELTA_T=3.16e7
module load sstack/latest
module load spack/cosmicgrain
spack load openmpi@5.0.10 fftw@3.3.10 hdf5@1.14.6 gsl@2.8
for pkg in gsl@2.8 fftw@3.3.10 hdf5@1.14.6 openmpi@5.0.10; do LD_LIBRARY_PATH="$(spack location -i $pkg)/lib:$LD_LIBRARY_PATH"; done; export LD_LIBRARY_PATH
export OMPI_MCA_pml=ucx UCX_NET_DEVICES=mlx5_0:1
unset OMPI_MCA_btl_tcp_if_include
env | grep -E "^(OMPI_MCA|UCX|CG_)" | sort
cd ~/CosmicGrain-br06
srun --export=ALL --mpi=pmix_v5 --unbuffered ~/jobexe/CosmicGrain.786 "$1" 0
rc=$?
OUT=$(awk '/^OutputDir/{print $2}' "$1")
if [ $rc -eq 0 ] && [ -d "$OUT/snapdir_029" ] && [ -d "$OUT/groups_029" ]; then
  rm -rf "$OUT/restartfiles" && echo "removed $OUT/restartfiles"
fi
exit $rc
