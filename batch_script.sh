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
module load release/23.04  
module load GCC/11.3.0  
module load OpenMPI/4.1.4
module load scikit-learn/1.1.2
module load TensorFlow/2.11.0-CUDA-11.7.0
pip install nibabel
pip install xlsxwriter
pip install openpyxl


# Execute parallel application 
srun python MRI_main_102.py

