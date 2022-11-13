from model.model_utils import compute_num_neurons


def find_CompressionValues_for_quantBits(start_comp_ratio, dataset_num_voxel, num_layers, grace_interval=4):

    # M: calculates size of model file after quantization
    def calcSizeAfterQuant(LayerSize, QuantBits, numMiddleLayers=7):
        middle = 4 * (2**QuantBits) + (((QuantBits*(LayerSize**2))) // 8) + 4 * LayerSize
        first = 4 * (3*LayerSize) + 4 * (LayerSize)
        last = 4 * (1*LayerSize) + 4 * 1
        return first + (numMiddleLayers * 2 * middle) + last

    # M: calculates compression ratio of quantized model to original data file
    def calcRatioAfterQuant(LayerSize, QuantBits, numMiddleLayers=7):
        compressed_size = calcSizeAfterQuant(LayerSize, QuantBits, numMiddleLayers)
        orig_size = dataset_num_voxel * 4
        return orig_size / compressed_size

    start_target_size = int(dataset_num_voxel / start_comp_ratio)
    start_num_features = compute_num_neurons(num_layers, start_target_size)
    start_comp_ratio_afterQuant = calcRatioAfterQuant(start_num_features, 9)
    print('Compressionratio: ', start_comp_ratio, ' for ', 9, ' Bits ', start_num_features, ' channels, orig: ',
          start_comp_ratio_afterQuant)

    # reduce available bits and search for the compressionratio to give similar results
    compratio = start_comp_ratio
    for q in [8,7,6,5,4,3,2,1]:
        while compratio > 1:
            compratio = compratio-0.1
            target_size = int(dataset_num_voxel / compratio)
            num_features = compute_num_neurons(num_layers, target_size)
            comp_ratio_afterQuant = calcRatioAfterQuant(num_features, q)

            if comp_ratio_afterQuant < start_comp_ratio_afterQuant + grace_interval \
                    and comp_ratio_afterQuant > start_comp_ratio_afterQuant - grace_interval:
                print('Compressionratio: ', compratio, ' for ', q, ' Bits ',num_features, ' channels, orig: ',
                      comp_ratio_afterQuant)
                break


if __name__ == '__main__':
    grace_interval = 1
    dataset_num_voxel = 16581375  # 3375000 # testvol #16581375  # mhd_p
    start_comp_ratio = 50
    num_layers = 8
    find_CompressionValues_for_quantBits(start_comp_ratio, dataset_num_voxel, num_layers, grace_interval)