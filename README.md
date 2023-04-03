# Neurcomp Compression

[**Master's Thesis**](plots\Master_Thesis_Training_Methods_for_Memory_efficient_Volume_Scene_Representation_Networks_Maarten_Bussler.pdf)

Project for my master's thesis to research possibilities of compressing Scene Representation Networks based on a monolithic architecture with network pruning algorithms.
The network is based on [Neurcomp](https://github.com/matthewberger/neurcomp). Besides a binary masking pruning, the pruning algorithms of [Smallify](https://github.com/mitdbg/fastdeepnets) and [Variational Dropout](https://arxiv.org/pdf/1506.02557.pdf) are implemented.

## Quick Start

### Requirements
Basic functionalities, such as training and quantization of the network with 3D numpy arrays as input, as well as writing of the results as .vti files can be enabled by installing with pip (`NeurcompRequirements.txt`) or conda (`NeurcompEnv.yml`).
The resulting .vti files can be visualized with [ParaView](https://www.paraview.org/).

Optionally: [Pyrenderer](https://github.com/shamanDevel/fV-SRN) can be used as a visualization tool and to feed CVol Data into the network.
Follow the instructions under https://github.com/shamanDevel/fV-SRN for installation.

### Data
Datasets and corresponding config files for all experiments can be found in `datasets/` and `experiment-config-files/`.

### Train and run the network:
1. Install the requirements from NeurcompRequirements.txt (pip) or NeurcompEnv.yml (conda).
2. Generate a config-file for the experiment or use one under `experiment-config-files/`. Descriptions for the different parameters can be generated with `python NeurcompTraining.py --help`.
3. Use `python NeurcompTraining.py --config <Path-To-Config-File>` to start training.
4. During training, [Tensorboard](https://mlflow.org/docs/latest/quickstart.html) tracks the experiment under `mlruns/`. A checkpoint to the trained model, as well as the config-file and basic information about the training are logged to `experiments/<expname>/`. Also a .vti file for the ground-truth and model-predicted volume will be generated.
5. For inference and visualization of a trained network, run `python NeurcompVisualization.py --config <Path-To-Model-Config-File-In-Experiment-Folder>`. This will generate a .vti file for the ground-truth and model-predicted volume.

### Perform Hyperparameter Search:
In order to find the best hyperparameter for each network type and dataset, the [AX MULTI-OBJECTIVE NAS](https://ax.dev/) Algorithm is provided.
To run hyperparameter search, use `jupyter notebook` to start either the 'Multiobjective-NAS' or 'Variational NAS' jupyter notebooks.
In the first cell, define the config file of the experiment, then execute the subsequent cells to start the scheduler and visualize the results.
The Search-Space for each experiment can be configured in `Multi_Objective_NAS.py`.

### Encode and decode the network with quantization:
1. After training a model, use `python NeurcompQuantization.py --config <Path-To-Model-Config-File-In-Experiment-Folder> --filename <Encoded-Name> --quant_bits <Bits-ForQuantization>` to encode the model. The compressed model is then saved in the experiment-folder.
2. After encoding the model, use `python NeurcompDequantization.py --compressed_file <Path-to-encoded-model> --volume <Path-to-gt-dataset> --decompressed_file <Output-Name>` to decode and visualize the network. The --volume and --decompressed_file paramaters are optional, but if provided generate a .vti file of the decoded network scene.

## Project Structure
- Parsing of Arguments, as well as the entry points to training and quantization can be found in `NeurcompTraining.py` and `NeurcompQuantization.py`.
- The initialization of the network, as well as training is implemented in `training/training.py`.
- The basic model architecture can be found in `model/NeurcompModel.py` and `model/SirenLayer.py`.
- The Pruning algorithms are implemented in `model/SmallifyDropoutLayer.py`, `model/Straight_Through_Binary.py` and `model/VariationalDropoutLayer.py`.
- Data input is handled in the classes in `data/`
- Quantization is handled in the classes in `quantization/`

## Results
TODO