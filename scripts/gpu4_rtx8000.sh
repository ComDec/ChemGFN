#!/bin/bash

#SBATCH --job-name=gflow
#SBATCH --output=logs/train_%j.slout
#SBATCH --error=logs/train_%j.slerr
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:rtx8000:4
#SBATCH --account=pr_133_tandon_advanced
#SBATCH --requeue

eval "$(mamba shell hook --shell bash)" && mamba activate
mamba activate gflow

python /scratch/xw3763/project/gflow/ChemGFN/chemgfn/train.py experiment="expr24_split_loss" trainer.devices=4
