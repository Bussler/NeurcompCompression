import torch
from torch import nn
import torch.nn.utils.prune as prune
from model.NeurcompModel import Neurcomp


# M: Search for zero beta parameters of layer_type dropout layer
def prune_dropout_threshold(model, dropout_type, threshold=0.1):
    for module in model.net_layers.modules():
        if isinstance(module, dropout_type):
            param = module.betas  # TODO: M: Better method to get beta weights of module?
            prune_mask = (torch.abs(param) > threshold).float()
            prune.custom_from_mask(module, name='betas', mask=prune_mask)

    #named_layers = dict(model.named_modules())
    #module = named_layers['net_layers.1.linear_1']
    #prune_mask = torch.ones_like(module.weight)
    #prune_mask[:, 0] = 0
    #prune.custom_from_mask(module, name='weight', mask=prune_mask)
    #prune.remove(module, 'weight')


# M: TODO rewrite for generalization
# M: reconstruct the parameter weights according to pruning and remove dropout from network
def prune_smallify_use_resnet(model, dropout_type):

    pruned_dropout_channel = []
    state_dict = model.state_dict()
    cached_mask = None

    with torch.no_grad():
        for name, param, in model.state_dict().items():
            if 'net_layers.0.' in name and '.weight' in name:
                pruned_dropout_channel.append(param.shape[0])
            if 'betas' in name and 'betas_mask' not in name:
                layer_split = name.split('.')
                betas_name = layer_split[0] + '.' + layer_split[1] + '.' + layer_split[2] + '.betas_orig'
                mask_name = layer_split[0] + '.' + layer_split[1] + '.' + layer_split[2] + '.betas_mask'

                if 'drop1' in name:
                    layer_name = layer_split[0]+'.'+layer_split[1]+'.linear_1'
                if 'drop2' in name:
                    layer_name = layer_split[0] + '.' + layer_split[1] + '.linear_2'
                weight_name = layer_name+'.weight'
                bias_name = layer_name+'.bias'

                betas = state_dict[betas_name]

                if mask_name in state_dict:
                    mask = state_dict[mask_name]
                else:
                    mask = torch.ones(param.shape[0])

                # M: mult betas and mask to get pruned betas
                pruned_betas = betas.mul(mask)
                # M: delete pruned rows
                resized_betas = pruned_betas[mask > 0]
                pruned_dropout_channel.append(resized_betas.shape[0])

                # M: resize weight and biases and remove betas by multiplying them out with weights and biases
                resized_weights = state_dict[weight_name][mask > 0, :]
                resized_bias = state_dict[bias_name][mask > 0]

                # M: change input dimension to match previous input
                if cached_mask is not None:
                    nonzero_indices = cached_mask.nonzero().squeeze(1)
                    resized_weights = torch.index_select(resized_weights, 1, nonzero_indices)
                #else:
                    #cached_mask = mask  # M because of res net layers, we have to sum input and second output and thus can't shrink down input after here!

                # M: change size of second linear layer in case of second linear layer
                layer_name_2 = layer_split[0] + '.' + layer_split[1] + '.linear_2'
                layer_name_2_weight = layer_name_2+'.weight'
                layer_name_2_bias = layer_name_2+'.bias'
                if layer_name_2_weight in state_dict:

                    # M: output dimension
                    resized_weights_2 = state_dict[layer_name_2_weight]#[cached_mask > 0, :]
                    resized_bias_2 = state_dict[layer_name_2_bias]#[cached_mask > 0]

                    # M: input dimension
                    nonzero_indices_2 = mask.nonzero().squeeze(1)
                    resized_weights_2 = torch.index_select(resized_weights_2, 1, nonzero_indices_2)

                    state_dict[layer_name_2_weight] = resized_weights_2
                    state_dict[layer_name_2_bias] = resized_bias_2

                state_dict[weight_name] = resized_weights.mul(resized_betas[:, None])
                state_dict[bias_name] = resized_bias.mul(resized_betas)

                state_dict.pop(betas_name)
                state_dict.pop(mask_name)

    # M: fast fix: set output channel
    #last_linear_name = list(state_dict)[len(state_dict)-2]
    #nonzero_indices = cached_mask.nonzero().squeeze(1)
    #state_dict[last_linear_name] = torch.index_select(state_dict[last_linear_name], 1, nonzero_indices)

    new_model = Neurcomp(input_ch=model.d_in, output_ch=model.d_out, features=pruned_dropout_channel,
                     omega_0=model.omega_0, dropout_technique='')
    new_model.load_state_dict(state_dict)
    return new_model


def prune_smallify_no_Resnet(model, dropout_type):

    pruned_dropout_channel = []
    state_dict = model.state_dict()
    cached_mask = None

    with torch.no_grad():
        for name, param, in model.state_dict().items():
            if 'betas' in name and 'betas_mask' not in name:
                layer_split = name.split('.')
                betas_name = layer_split[0] + '.' + layer_split[1] + '.' + layer_split[2] + '.betas_orig'
                mask_name = layer_split[0] + '.' + layer_split[1] + '.' + layer_split[2] + '.betas_mask'

                layer_name = layer_split[0]+'.'+layer_split[1]+'.linear_1'

                weight_name = layer_name+'.weight'
                bias_name = layer_name+'.bias'

                betas = state_dict[betas_name]

                if mask_name in state_dict:
                    mask = state_dict[mask_name]
                else:
                    mask = torch.ones(param.shape[0])

                # M: mult betas and mask to get pruned betas
                pruned_betas = betas.mul(mask)
                # M: delete pruned rows
                resized_betas = pruned_betas[mask > 0]
                pruned_dropout_channel.append(resized_betas.shape[0])

                # M: resize weight and biases and remove betas by multiplying them out with weights and biases
                resized_weights = state_dict[weight_name][mask > 0, :]
                resized_bias = state_dict[bias_name][mask > 0]

                # M: change input dimension to match previous input
                if cached_mask is not None:
                    nonzero_indices = cached_mask.nonzero().squeeze(1)
                    resized_weights = torch.index_select(resized_weights, 1, nonzero_indices)
                cached_mask = mask

                state_dict[weight_name] = resized_weights.mul(resized_betas[:, None])
                state_dict[bias_name] = resized_bias.mul(resized_betas)

                state_dict.pop(betas_name)
                state_dict.pop(mask_name)

    # M: fast fix: set output channel
    last_linear_name = list(state_dict)[len(state_dict)-2]
    nonzero_indices = cached_mask.nonzero().squeeze(1)
    state_dict[last_linear_name] = torch.index_select(state_dict[last_linear_name], 1, nonzero_indices)

    new_model = Neurcomp(input_ch=model.d_in, output_ch=model.d_out, features=pruned_dropout_channel,
                     omega_0=model.omega_0, dropout_technique='', use_resnet=False)
    new_model.load_state_dict(state_dict)
    return new_model


# TODO
def prune_variational_dropout():
    pass
