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

- [System requirements](#System-requirements)
- [Preprocessing](#preprocessing)
- [Batch Script Parameters](#batch-script-parameters)
- [License](#license)


## System requirements

This program is executed with the current architecture on at least a 40GB NVIDIA GPU with a batch size of 2. If you encounter memory limitation errors during execution, please consider changing the version of TensorFlow you are using.


## Preprocessing

First, preprocess all MRI files using the FlexiMRIprep pipeline. You can find FlexiMRIprep [here](https://github.com/ul-mds/FlexiMRIprep).

```bash
python scan_input_MRI.py
```

Next, run the `scan_input_MRI.py` script to generate a new file named `All_MRIs_List_paths_temp.csv` using the contents of `MRIs_List.csv` and `datasets_path.csv`.


## Batch Script Parameters

Please update the parameters in the `batch_script.sh` file for submitting jobs to a cluster.

This code has always been executed using SLURM. If you want to run the code directly on your local machine, you will need to manually install the required Python packages.

## License

This project is licensed under the GPL-3.0 license - see the [LICENSE](blob/main/LICENSE) file for details.

