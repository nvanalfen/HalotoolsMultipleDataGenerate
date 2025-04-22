#!/bin/bash
#SBATCH -J mp_test
#SBATCH --partition=express
#SBATCH --time=0:30:00
#SBATCH --mem=20G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --output=output_logs/mp_example.out
#SBATCH --error=error_logs/mp_example.err

source /work/blazek_group_storage/miniconda3/bin/activate
conda activate halotools

python3 multiprocessing_example.py
