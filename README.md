# HeteroMRI
A novel method for classification of heterogenous brain MRIs. Here, heterogenous means the MRIs used for training the model have been acquired using different MRI scanners and acquisition protocols.

This repository contains material associated with the paper "***HeteroMRI: A method for classification of multi-scanner and multi-protocol brain MRIs with deep learning***", including:

- `MRIs_List.csv`: The list of all MRIs (from public repositories) used in the paper 
- `Experimental_Settings.xlsx`: The experimental settings implemented in the paper
- The Python code that
  - selects the necessary number of MRIs from the `MRIs_List.csv` based on the `Experimental_Settings.xlsx`
  - trains and tests the CNN classification model
  - saves the trained models
  - writes an *.xlsx file that includes the list of training, validation, and test data and the calculated metric values for each model



## Table of Contents

- [Installation](#installation)
- [Preprocessing](#preprocessing)
- [Batch Script Parameters](#batch-script-parameters)
- [Contributing](#contributing)
- [License](#license)

## Installation

Step-by-step instructions on how to get the development environment running.

```bash
# Clone the repository
git clone https://github.com/your-username/HeteroMRI.git

# Navigate to the project directory
cd HeteroMRI

# Install dependencies
npm install
```

## Preprocessing

First, preprocess all MRI files using the FlexiMRIprep pipeline. You can find FlexiMRIprep [here](https://github.com/ul-mds/FlexiMRIprep).

Use the following parameters for the preprocessing step, as detailed in our paper:

```bash
ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=8 export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS
python main.py -s "422256" -m "non" -lm "" -i "./input" -o "./output" -s2 r:1
```


## Batch Script Parameters

Below is an example of a batch script (`batch_script.sh`) used for submitting jobs to a cluster. Key parameters that users need to modify are explained.

```bash
#!/bin/bash

#SBATCH --job-name="codeC"           # Job name
#SBATCH --mail-user=***************  # Email address for notifications
#SBATCH --mail-type=ALL              # Type of notifications (BEGIN, END, FAIL, ALL)
#SBATCH --time=03-00:00:00           # Time limit (D-HH:MM:SS)
#SBATCH --partition=*****            # Partition name
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks=1                   # Number of tasks
#SBATCH --mincpus=1                  # Minimum number of CPUs
#SBATCH --cpus-per-task=16           # Number of CPUs per task
#SBATCH --gres=gpu:1                 # Number of GPUs
#SBATCH --gpus-per-task=1            # GPUs per task
#SBATCH --mem-per-cpu=8G             # Memory per CPU
#SBATCH --output=slurm-%j.out        # Standard output log file
#SBATCH --error=slurm-%j.err         # Standard error log file

# Setup computational environment, i.e, load desired modules
module load release/23.04  
module load GCC/11.3.0  
module load OpenMPI/4.1.4
module load scikit-learn/1.1.2
module load TensorFlow/2.11.0-CUDA-11.7.0

# Install required Python packages
pip install nibabel
pip install xlsxwriter
pip install openpyxl

# Execute parallel application 
srun python main.py
```

## Contributing

Please read [CONTRIBUTING.md](https://github.com/your-username/HeteroMRI/CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

