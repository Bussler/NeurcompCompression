import numpy as np
import torch
import os
from model.NeurcompModel import Neurcomp


# M: tries to find the correct amt of features per layer, so that we get target_size neurons
# M: TODO: better way to do this? How big is each neuron, is neuron = voxelsize?
def compute_num_neurons(num_layer, target_size, input_ch=3, output_ch=1):
    d_in = input_ch
    d_out = output_ch

    def network_size(features_each_layer):
        layer_sizes = [d_in]
        layer_sizes.extend([features_each_layer] * num_layer)
        layer_sizes.append(d_out)
        n_layers = len(layer_sizes) - 1

        n_params = 0
        for ndx in range(n_layers):
            layer_in = layer_sizes[ndx]
            layer_out = layer_sizes[ndx + 1]

            if ndx == 0 or ndx == (n_layers - 1): # M: SINE or Linear at Beginning/ End
                n_params += ((layer_in + 1) * layer_out) # M: TODO: Why + 1 here?
            else:
                n_params += (layer_in * layer_out) + layer_out # M: each res block is treated as having 2 layers
                n_params += (layer_out * layer_out) + layer_out
        return n_params

    features_each_layer = 16
    while network_size(features_each_layer) < target_size:
        features_each_layer += 1
    features_each_layer -= 1  # M: apply conservative estimate to account for biases...

    return features_each_layer


def setup_neurcomp(compression_ratio, dataset_size, n_layers, d_in, d_out, omega_0, checkpoint_path,
                   dropout_technique='', sign_variance_momentum=0.02, features_per_layer=0, featureList=None,
                   use_resnet=True, pruning_threshold=0.9, variational_init_droprate=0.5):

    if features_per_layer > 0:
        featureList = np.full(n_layers, features_per_layer)

    if featureList is not None and len(featureList) > 0:
        feature_list = featureList
    else:
        target_size = int(dataset_size / compression_ratio)  # M: Amt of neurons in whole model
        num_neurons = compute_num_neurons(num_layer=n_layers,
                                          target_size=target_size)  # M: number of neurons per layer
        if not use_resnet:
            n_layers = (n_layers*2)-1
        feature_list = np.full(n_layers, num_neurons)  # M: list holding the amt of neurons per layer

    model = Neurcomp(input_ch=d_in, output_ch=d_out, features=feature_list,
                     omega_0=omega_0, dropout_technique=dropout_technique,
                     sign_variance_momentum=sign_variance_momentum, use_resnet=use_resnet,
                     pruning_threshold=pruning_threshold, variational_init_droprate=variational_init_droprate)

    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path))

    return model


def write_dict(dictionary, filename, experiment_path=''):
    with open(os.path.join(experiment_path, filename), 'w') as f:
        for key, value in dictionary.items():
            f.write('%s = %s\n' % (key, value))


def write_list(list, filename, experiment_path=''):
    with open(os.path.join(experiment_path, filename), 'w') as f:
        for item in list:
            #f.write('%s = %s\n' % (item))
            f.write('%s\n' % (item))
