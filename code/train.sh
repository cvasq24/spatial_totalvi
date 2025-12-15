#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --job-name=train_%j
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=32
#SBATCH --partition=beaver
#SBATCH --output=/ubc/cs/research/beaver/projects/carlos/spatial_totalvi/code/%j.out

mamba activate totalvi
python train_spatial_totalvi.py