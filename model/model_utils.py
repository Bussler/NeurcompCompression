import numpy as np
import torch
from model.NeurcompModel import Neurcomp


# M: tries to find the correct amt of features per layer, so that we get target_size neurons
# M: TODO: better way to do this?
def compute_num_neurons(num_layer, target_size, input_ch=3, output_ch=1):
    d_in = input_ch
    d_out = output_ch

    def network_size(neurons):
        layers = [d_in]
        layers.extend([neurons] * num_layer)
        layers.append(d_out)
        n_layers = len(layers) - 1

        n_params = 0
        for ndx in np.arange(n_layers):
            layer_in = layers[ndx]
            layer_out = layers[ndx + 1]
            og_layer_in = max(layer_in, layer_out)

            if ndx == 0 or ndx == (n_layers - 1):
                n_params += ((layer_in + 1) * layer_out)
            else:
                is_shortcut = layer_in != layer_out
                if is_shortcut:
                    n_params += (layer_in * layer_out) + layer_out
                n_params += (layer_in * og_layer_in) + og_layer_in
                n_params += (og_layer_in * layer_out) + layer_out
        return n_params

    min_neurons = 16
    while network_size(min_neurons) < target_size:
        min_neurons += 1
    min_neurons -= 1

    return min_neurons


def setup_neurcomp(compression_ratio, dataset_size, n_layers, d_in, d_out, omega_0, checkpoint_path):
    target_size = int(dataset_size / compression_ratio)  # M: Amt of neurons in whole model
    num_neurons = compute_num_neurons(num_layer=n_layers,
                                      target_size=target_size)  # M: number of neurons per layer
    feature_list = np.full(n_layers, num_neurons)  # M: list holding the amt of neurons per layer

    model = Neurcomp(input_ch=d_in, output_ch=d_out, features=feature_list, omega_0=omega_0)

    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path))

    return model
