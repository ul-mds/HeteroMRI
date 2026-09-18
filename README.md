# HeteroMRI

**HeteroMRI** is a deep learning method for classification of heterogeneous brain MRI data based on white matter abnormalities. Here, *heterogeneous* refers to MRI data acquired using different scanners and acquisition protocols.

The method is designed to learn discriminative patterns from heterogeneous MRI datasets and can be adapted to different MRI classification tasks and applications.

This repository contains the **updated implementation of HeteroMRI**. The current implementation was used in our study applying HeteroMRI to the differentiation of **adult leukodystrophies from multiple sclerosis (MS)**. This application demonstrates the use of HeteroMRI for a clinically relevant classification problem involving heterogeneous MRI data.

The framework is not limited to this application. Researchers can utilize HeteroMRI and train it for other classification problems involving heterogeneous brain MRI data.


---

> [!CAUTION]
> **HeteroMRI is NOT a medical device and is intended for academic research use only.**
> Do NOT use HeteroMRI for diagnosis, prognosis, monitoring, or any other purposes in clinical use.

---

## HeteroMRI versions

### Current version

The `main` branch contains the **updated implementation of the HeteroMRI method**.

This version was used for the differentiation of Leukodystrophies from MS application described in the following paper:

> *Differentiating adult leukodystrophies from multiple sclerosis brain MRIs using deep learning*

The Leukodystrophies-vs-MS study provides an example of how the HeteroMRI method can be applied to a specific clinical classification problem. The method can also be adapted to and trained for other applications.

### Previous version

The implementation used in our original HeteroMRI study is preserved in the [`v1.0` branch](https://github.com/ul-mds/HeteroMRI/tree/v1.0).

The original study applied HeteroMRI to the classification of brain FLAIR MRIs **with and without white matter abnormalities**:

> Abedi, M., Shekarchizadeh, N., Bazin, P.-L., et al. *HeteroMRI: Robust white matter abnormality classification across multi-scanner MRI data.* GigaScience, 14 (2025), giaf092.
> https://doi.org/10.1093/gigascience/giaf092

The `v1.0` branch is retained to provide access to the code corresponding to that publication.


---

## What is included in this repository?

The repository contains the Python implementation and configuration files required to run HeteroMRI experiments.

The main components include:

* **MRI preprocessing and input preparation**
* **Selection of MRI data according to experimental settings**
* **Training and testing of the CNN classification model**
* **Saving trained models and experimental results**
* **Calculation of classification performance metrics**
* **Permutation-based analyses**

The specific input data, experimental settings, and model parameters can be adapted to the application of interest.

---

## Hardware requirements

The computational requirements depend on the model configuration and experimental settings.

The current CNN architecture requires a high-memory NVIDIA GPU for training. In our experiments, an NVIDIA GPU with at least **40 GB of GPU memory** (e.g., NVIDIA A100) was required (with a batch size of 2).

If a GPU out-of-memory error occurs, reducing the batch size or adjusting the software environment may help. In particular, different TensorFlow versions can have different GPU memory requirements.

---

## How to use HeteroMRI

The general HeteroMRI workflow consists of the following steps:

1. Prepare the MRI data and the corresponding input list.
2. Preprocess the MRI data.
3. Scan the input data and identify the required MRI files to make sure the files mentioned in the input list exist.
5. Configure the experimental settings. 
6. Configure the model parameters.
7. Run the HeteroMRI experiments.
8. Evaluate the resulting models and performance metrics.

The exact preprocessing and experimental configuration may differ depending on the application and dataset.

### Preparing `MRIs_List.csv`

MRI data from different datasets can be used to construct the dataset for training and evaluation.

The input list should follow the structure of `MRIs_List.csv`. The following columns are required by the code but the other columns can be left empty:

* `ID` — a unique identifier for each MRI
* `Dataset` — the source dataset name
* `Label` — the classification label
* `Subject_ID` — subject identifier (from dataset)
* `Protocol_Group` — MRI protocol group, only if required by the experimental settings
* `Selected_Cluster` — the selected intensity cluster

The labels and other dataset-specific information should be defined according to the classification task being investigated.

### MRI preprocessing

MRI files should first be processed using the required preprocessing pipeline.

Using the preprocessing workflow described in the original HeteroMRI study, MRI preprocessing was performed using the [FlexiMRIprep](https://github.com/ul-mds/FlexiMRIprep) pipeline.

In the Leukodystrophies-vs-MS study, the following command is used in FlexiMRIprep to preprocess the MRIs.The preprocessing procedure includes the necessary bias correction, registration, and intensity clustering steps required by HeteroMRI.

```
python main.py -s "49a22256" -m "non" -lm "" -i "./input" -o "./output" -s2 r:1 -s9 r:1 -sa r:1
```
This command will apply the following preprocessings on each MRI:
*N4 bias correction
*Rigid registration
*Affine registration
*3 times non-linear (deformable) registration
*Extract the white matter of the brain by multiplying the white matter template of the MNI template to the brain
*Cluster the extracted white matter into 3 intensity clusters (only one of these will be selected and used in the model)



### Dataset paths

The paths to the intensity-cluster data should be specified in:

```text
input_files/datasets_path.csv
```

### Scanning input data

Before running the model, run the `Scan_output_MRI.py` code to check whether the required input MRI data are available. Moreover, the code selects the right intensity cluster among the three cluster files available for each subject by comparing them with the `input_files/template_dice.nii.gz` file. The cluster that looks more similar (highest Dice score) to the template_dice.nii.gz is selected.

The resulting file (MRI list) contains the paths to the MRI files (the correct intensity cluster file) required for the HeteroMRI experiments.


### Configure the experimental settings
In the `input_files/Experimental_Settings.xlsx`sheet, it is possible to design the experimental setting(s). Each row is an experimental setting. There can be one or more settings. In each row, the number of MRIs taken from each dataset for the training, validation, and test sets is entered.
In the Leukodystrophies-vs-MS study,there was only one experimental setting, named 'A' as shown in `input_files/Experimental_Settings.xlsx`. In this setting, data from 4 datasets is taken. The information in this Excel sheet should match the MRI data in the `input_files/MRIs_List.csv`, i.e. the dataset names, labels, number of available MRIs in the MRI list, and number of requested MRIs in the experiment.

### Configuring the model

The parameters of the model, including the Experimental setting(s) name, number of shuffles, number of runs per shuffle, output path for writing the saved models and model evaluation results, and the number of epochs, can be set in the `main.py`.



### Running HeteroMRI

The model can be run using the provided scripts.

The experiments in our studies were performed on an HPC cluster using the Slurm workload manager and the provided `batch_script.sh`. The parameters in `batch_script.sh` should be adapted to the target HPC environment.

### Output

The output includes the trained models and experiment-specific results.



## Data

The MRI datasets used in the Leukodystrophies-vs-MS study are **not distributed with this repository**. The information regarding data access is explained in the paper.


---


---

## Citation

If you use the **HeteroMRI implementation**, please cite the following papers:

> *Differentiating adult leukodystrophies from multiple sclerosis brain MRIs using deep learning* (further information to be announced after publication of the paper)

> Abedi, M., Shekarchizadeh, N., Bazin, P.-L., Scherf, N., Lier, J., Bergner, C.-C., Köhler, W., & Kirsten, T. (2025). *HeteroMRI: Robust white matter abnormality classification across multi-scanner MRI data.* GigaScience, 14, giaf092.
> https://doi.org/10.1093/gigascience/giaf092

---

## License

This project is licensed under the **GPL-3.0 license**. See the [`LICENSE`](https://github.com/ul-mds/HeteroMRI/blob/main/LICENSE) file for details.
