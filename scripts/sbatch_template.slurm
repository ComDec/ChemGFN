#!/bin/bash

#SBATCH --job-name=GFN_base_g8_e50
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --time=12:00:00         # max time for running this job

# Singularity path
ext3_path=/scratch/xw3763/esche/overlay-50G-10M.ext3
sif_path=/scratch/xw3763/esche/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif

# start running
singularity exec --nv \
--overlay ${ext3_path}:ro \
${sif_path} /bin/bash -c "
    source /ext3/env.sh
    conda activate llm
    cd /scratch/xw3763/project/ChemGFN/ && pip install -e .
    python /scratch/xw3763/project/ChemGFN/chemgfn/train.py exp_name="baseline-gpt2-invalid-base-g8-e50"
"
