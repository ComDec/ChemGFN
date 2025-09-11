#!/bin/bash

#SBATCH --job-name=gflow
#SBATCH --output=logs/train_%j.slout
#SBATCH --error=logs/train_%j.slerr
#SBATCH --cpus-per-task=8
#SBATCH --mem=256GB
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --account=pr_133_tandon_advanced
#SBATCH --requeue

# Singularity path
ext3_path=/scratch/xw3763/xw3763/overlay-50G-10M.ext3
sif_path=/scratch/xw3763/xw3763/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif

# start running
singularity exec --nv \
--overlay ${ext3_path}:ro \
${sif_path} /bin/bash -c "
    source /ext3/env.sh
    conda activate /scratch/xw3763/micromamba/envs/gflow
    python /scratch/xw3763/project/gflow/ChemGFN/chemgfn/train.py experiment="baseline" trainer.devices=1
"
