#!/bin/bash

#SBATCH --job-name="A"
#SBATCH --array=1-1  # Creates n parallel tasks
#SBATCH --time=70-00:00:00
#SBATCH --partition=*****
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=15G
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err
#SBATCH --account=account_name

# Setup computational environment, i.e, load desired modules
module load release/24.04
module load GCC/12.3.0
module load OpenMPI/4.1.5
module load scikit-learn/1.3.1
module load cuDNN/8.9.2.26-CUDA-12.1.1
module load TensorFlow/2.15.1-CUDA-12.1.1
module load NiBabel/5.2.0

module load modenv/scs5
pip install xlsxwriter
pip install openpyxl


MAIN="main${SLURM_ARRAY_TASK_ID}.py"

# Execute parallel application 
srun python "$MAIN"

