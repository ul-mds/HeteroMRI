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

# Install python packages 

```

## Preprocessing

First, preprocess all MRI files using the FlexiMRIprep pipeline. You can find FlexiMRIprep [here](https://github.com/ul-mds/FlexiMRIprep).

```bash
python scan_input_MRI.py
```

Next, run the `scan_input_MRI.py` script to generate a new file named `All_MRIs_List_paths_temp.csv` using the contents of `MRIs_List.csv` and `datasets_path.csv`.


## Batch Script Parameters

Please update the parameters in the `batch_script.sh` file for submitting jobs to a cluster.


## Contributing

Please read [CONTRIBUTING.md](https://github.com/your-username/HeteroMRI/CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

