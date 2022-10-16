# Neurcomp Compression

[**Neurcomp**](https://github.com/matthewberger/neurcomp) | [**Targeted Dropout**](https://openreview.net/pdf?id=HkghWScuoQ) | [**Variational Dropout**](https://arxiv.org/pdf/1506.02557.pdf)

TODO

## Quick Start

### Requirements
Basic functionalities, such as training and quantization of the network with 3D numpy arrays as input, as well as writing of the results as .vti files can be enabled by installing the `requirements.txt`.
The resulting .vti files can be visualized with [ParaView](https://www.paraview.org/).
TODO: provide requirements.txt

Optionally: [Pyrenderer](https://github.com/shamanDevel/fV-SRN) is used as a visualization tool and for feeding CVol-Data for rendering support and to feed CVol Data into the network.
Follow the instructions under https://github.com/shamanDevel/fV-SRN for installation.

### Data
Datasets and corresponding config files for all experiments can be found in `datasets/` and `experiment-config-files/`.

### Train and run the network:
1. Install the requirements from requirements.txt.
2. Generate a config-file for the experiment or use one under `experiment-config-files/`. Descriptions for the different parameters can be generated with `python NeurcompTraining.py --help`.
3. Use `python NeurcompTraining.py --config <Path-To-Config-File>` to start training.
4. During training, [MLFlow](https://mlflow.org/docs/latest/quickstart.html) tracks the experiment under `mlruns/`. A checkpoint to the trained model, as well as the config-file and basic information about the training are logged to `experiments/<expname>/`.
5. For inference and visualization, run `python NeurcompVisualization.py --config <Path-To-Model-Config-File> --model <Path-To-Model-Checkpoint>`. This will generate a .vti file for the ground-truth and model-prediced volume.

### Encode and decode the network with quantization:
1. After training a model, use `python NeurcompQuantization.py --config <Path-To-Model-Config-File> --filename <Encoded-Name> --quant_bits <Bits-ForQuantization>` to encode the model. The compressed model is then saved in the experiment-folder.
2. After encoding the model, use `--compressed_file <Path-to-encoded-model> --volume <Path-to-gt-dataset> --decompressed_file <Output-Name>` to decode and visualize the network. The --volume and --decompressed_file paramaters are optional, but if provided generate a .vti file of the decoded network scene.

## Results
TODO