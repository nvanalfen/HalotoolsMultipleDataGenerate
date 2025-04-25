#!/bin/bash
#SBATCH -J mpi_test
#SBATCH --partition=short
#SBATCH --time=5:00:00
#SBATCH --mem-per-cpu=8GB
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=3
#SBATCH --output=output_logs/mpi_example.out
#SBATCH --error=error_logs/mpi_example.err

module load openmpi/4.1.5-gcc11.1
source /work/blazek_group_storage/miniconda3/bin/activate
conda activate halotools

mpirun --mca btl vader,self -n 4 python3 training_data_generate_mpi.py config.yaml 1
