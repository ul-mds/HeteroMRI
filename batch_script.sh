#!/bin/bash

#SBATCH --job-name="codeC"
#SBATCH --mail-user=***************
#SBATCH --mail-type=ALL
#SBATCH --time=03-00:00:00
#SBATCH --partition=*****
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mincpus=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# Setup computational environment, i.e, load desired modules
module load modenv/scs5
module load scikit-learn/1.0.2-foss-2021b
module load TensorFlow/2.4.1-fosscuda-2020b
pip install -U scikit-learn
pip install nibabel
pip install xlsxwriter
pip install openpyxl


# Execute parallel application 
srun python main.py

