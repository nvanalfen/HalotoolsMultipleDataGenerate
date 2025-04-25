#!/bin/bash
#SBATCH -J mp_test
#SBATCH --partition=short
#SBATCH --time=5:00:00
#SBATCH --mem=24G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --array=1-5%5
#SBATCH --output=output_logs/mp_example-%a.out
#SBATCH --error=error_logs/mp_example-%a.err

source /work/blazek_group_storage/miniconda3/bin/activate
conda activate halotools

python3 generate_median_training_data.py $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_COUNT
