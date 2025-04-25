#!/bin/bash
#SBATCH -J mpi_test
#SBATCH --partition=short
#SBATCH --time=5:00:00
#SBATCH --mem-per-cpu=8GB
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=3
#SBATCH --output=output_logs/mpi_example.out
#SBATCH --error=error_logs/mpi_example.err

module load openmpi/4.1.5-gcc11.1
source /work/blazek_group_storage/miniconda3/bin/activate
conda activate halotools

# the 0 argument indicates use fork multiprocess start method
# the 1 argument indicates use spawn multiprocess start method
# while spawn is "safer", it is slow and since I am only using multiptrocess for
# simply getting correlations with no need to access information beyond the function
# I should be safe
mpirun --mca btl vader,self -n 5 python3 training_data_generate_mpi.py config.yaml 0
