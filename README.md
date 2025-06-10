# HeteroMRI
A novel method for white matter-related classification of heterogeneous brain FLAIR MRIs. Here, heterogenous means the MRIs are acquired using different MRI scanners and acquisition protocols.

This repository contains material associated with the paper "***HeteroMRI: Robust white matter abnormality classification across multi-scanner MRI data***", including:

- `MRIs_List.csv`: The list of all MRIs (from public repositories) used in the paper 
- `Experimental_Settings.xlsx`: The experimental settings implemented in the paper
- The Python code that
  - selects the necessary number of MRIs from the `MRIs_List.csv` based on the `Experimental_Settings.xlsx`
  - trains and tests the CNN classification model
  - saves the trained models
  - writes an `*.xlsx` file that includes the list of training, validation, and test data and the calculated metric values for each model

## Hardware requirements

This code with the current CNN architecture needs an NVIDIA GPU with at least 40 GB of GPU memory (such as NVIDIA A100) even with a batch size of 2. 

If you encountered a GPU out-of-memory error while training the model, please consider changing the version of your TensorFlow since some Tensorflow versions load more data than the batch size in the GPU memory!


## How to use HeteroMRI
  - ### Preparing your `MRIs_List.csv`
    You can use brain MRIs from different datasets to train and test the model. Make a list of all the data identical to `MRIs_List.csv`. The following columns of `MRIs_List.csv` are required to be filled as they are needed by the code. You can leave the other columns empty but please _do not delete_ them. 
    + `ID` (a unique arbitrary identifier)
    + `Dataset` (name of the dataset)
    + `Label` (1: if the brain has white-matter abnormalities, 0: if the brain has no white-matter abnormalities)
    + `Subject_ID` (Patient identifier)
    + `Protocol_Group` (only necessary if you have experimental settings that choose data based on the MRI protocol, such as setting _B, C, D_ in the paper) (See [Protocol naming convention](https://github.com/ul-mds/HeteroMRI#Protocol-naming-convention))
    + `Selected_Cluster` (See [Selecting the right cluster](https://github.com/ul-mds/HeteroMRI#Selecting-the-right-cluster))
   
    
  - ### Preprocessing
First, all the MRI files should be preprocessed using the [FlexiMRIprep](https://github.com/ul-mds/FlexiMRIprep) pipeline. After setting up the pipeline, run the following command to perform all the necessary MRI preprocessing steps (as detailed in the paper) on all the MRIs automatically:
```
python main.py -s "422256" -m "non" -lm "" -i "./input" -o "./output" -s2 r:1
```
For implementing the N4 inhomogenety correction, the `N4BiasFieldCorrectionImageFilter` class from the SimpleITK (version 2.1.1.2)  Python library with the default parameters is used.

The following parameters are used in the `antsRegistration` tool: 
```
--dimensionality 3, --interpolation Linear, --transform SyN[gradientStep=0.1], --metric MI[metricWeight=1, numberOfBins=32] (Mutual Information), --initial-moving-transform [initializationFeature=1] --convergence 500x400x70x30, --shrink-factors 8x4x2x1, --smoothing-sigmas 3x2x1x0vox, --use-histogram-matching 0, --winsorize-image-intensities [0.005,0.995]
```
and for the rest of the parameters the default values are used. 

In `fuzzy_cmeans` function, the parameter values of `clusters=3, max_iterations=200`, and the default values for all the other parameters are used. 

- ### Dataset(s) path
Enter the path to the folder which includes the intensity clusters of the dataset(s) in `datasets_path.csv`.

- ### Scanning input data for the model
In this step, the code checks whether the (right) intensity cluster for all MRIs of `MRIs_List.csv` exists in the dataset's path. Run the following code:
```
python scan_input_MRI.py
```
A new file named `All_MRIs_List_paths_temp.csv` is generated which is the same as `MRIs_List.csv` except that the path to the right cluster is added to its last column ("`Path_Selected_Cluster_File`").

- ### Set model parameters
  Set the parameters of the model in `main.py`, including the experimental settings names (from `Experimental_Settings.xlsx`) that you wish to run, the number of shuffles, the number of runs for each shuffle, the number of epochs, etc.
- ### Running the model
We have trained the model on an HPC cluster with the Slurm system using the script `batch_script.sh`. Please update the parameters according to the HPC cluster you are using.
Considering the [hardware requirements](https://github.com/ul-mds/HeteroMRI#hardware-requirements), you will most probably need an HPC cluster, however, if you want to run the code directly on a local machine, you will need to manually install the required Python packages.

- ### Output
In the output folder, the best-trained model for each experimental setting is saved. In addition, an `*.xlsx` file is generated for each model that includes the list of training, validation, and test data and the calculated metric values.

### MRI protocol naming convention
We have assigned a protocol name to each of the MRIs. The MRI protocol name is generated based on a naming convention. For example, consider the protocol name `Sie_TrT_30_Prot1`. The first three characters determine the MR scanner manufacturer (here, Siemens). The second three characters show the MR scanner model (here, TrioTim). The next two digits indicate the magnetic field strength of the scanner in Tesla multiplied by ten to avoid a decimal dot in the protocol name (here, 3 Tesla). The final characters are related to the acquisition time parameters (namely, TE, TR, and TI). For example, the protocol `Sie_TrT_30_Prot2` differs in the acquisition time parameters compared to the `Sie_TrT_30_Prot1` protocol. If any of the above-mentioned information is missing for an MRI, we use `NA` instead of that in the protocol name. For a list of all protocol names and their details used in this study, see the `Protocols_List` sheet in the [Experimental_Settings.xlsx](https://github.com/ul-mds/HeteroMRI/raw/refs/heads/main/Experimental_Settings.xlsx) file.

### Containerize (Docker)
#### Building the Docker Image
To build the Docker image, navigate to the directory containing the Dockerfile and run the following command:
```
sudo docker build -t heteromri .
```
#### Running the Docker Container
To run the Docker container, use the following command:
```
docker run  --gpus all --volume=/the_intensity_clusters_of_the_dataset_on_local_computer/data:/data heteromri
```
This command mounts the local directory `the_intensity_clusters_of_the_dataset_on_local_computer/data` to the /data directory in the Docker container, ensuring that your intensity cluster data is accessible within the container.

#### Copying Output Files
After the container has finished running, you can copy the output files from the container to your local machine. First, get the container ID by listing all containers:
```
sudo docker ps -a
```
Find the container ID for the `heteromri` container. Then, copy the output files using the following command:
```
docker cp CONTAINER_ID:/usr/src/app/output ./output
```
Replace `CONTAINER_ID` with the actual ID of your container. This command copies the output directory from the container to the `output` directory on your local machine.

## Citation
If this repository was helpful for your project, please cite the following paper:
```
To be announced soon
```

## License
This project is licensed under the GPL-3.0 license - see the [LICENSE](LICENSE) file for details.

