import math
from model.NeurcompModel import Neurcomp
import struct
from sklearn.cluster import KMeans


def get_net_weights_biases(net):
    weights = []
    biases = []
    for name, param in net.named_parameters():  # M: weight matrices end with .weight; biases with .bias
        if name.endswith('.weight'):
            weights.append(param.data)
        if name.endswith('.bias'):
            biases.append(param.data)
    return weights, biases


class NetEncoder():

    def __init__(self, net: Neurcomp):
        self.net = net

    def encode(self, filename, bit_precision):
        file = open(filename, 'wb')

        # M: Get basic NW information
        n_clusters = int(math.pow(2, bit_precision))
        layer_sizes = self.net.layer_sizes
        n_layers = len(layer_sizes)  # M: haben hier noch in/ out dabei, das ist bei dem orig anders!

        all_weights, all_biases = get_net_weights_biases(self.net)

        # M: Write header for reconstruction
        # number of layers
        file.write(struct.pack('B', n_layers))
        # layer sizes
        file.write(struct.pack(''.join(['I' for _ in range(len(layer_sizes))]), *layer_sizes))
        # number of bits for clustering
        file.write(struct.pack('B', bit_precision))

        # M: First Layer: Not quantized
        weights, biases = all_weights[0].view(-1).tolist(), all_biases[0].view(-1).tolist()
        weight_format = ''.join(['f' for _ in range(len(weights))])
        bias_format = ''.join(['f' for _ in range(len(biases))])
        file.write(struct.pack(weight_format, *weights))
        file.write(struct.pack(bias_format, *biases))

        # M: Quantize middle layers with k-means clustering
        #for i in range(1, len(all_weights)-1):
        for cur_weigth, cur_bias in zip(all_weights[1:len(all_weights)-1], all_biases[1:len(all_weights)-1]):

            # M: Cluster the weights according to KMeans
            cur_weigth = cur_weigth.view(-1).unsqueeze(1).numpy()

            if n_clusters > cur_weigth.shape[0]:
                print('ERROR: Cannot perform K-Means Clustering,'
                      ' the amount of weights per layer is < than amount of clusters')
                file.close()
                return

            kmeans = KMeans(n_clusters=n_clusters, n_init=4).fit(cur_weigth)  # M: Perform k-mean clustering
            labeled_weights = kmeans.labels_.tolist()  # M: each weight as index to nearest cluster center
            cluster_centers = kmeans.cluster_centers_.reshape(n_clusters).tolist()

            # M: Write the centers
            center_format = ''.join(['f' for _ in range(len(cluster_centers))])
            file.write(struct.pack(center_format, *cluster_centers))

            # M: write the clustered weights
            weight_format = '#0'+str(bit_precision+2)+'b' #M: add 2 to account for 0b at beginning of format string
            bin_rep_labels = ''.join([format(labeled_weight, weight_format)[2:] for labeled_weight in labeled_weights])

            byte_storage = bytearray()
            n_bytes = len(bin_rep_labels) // 8
            for j in range(n_bytes):
                byte_val = bin_rep_labels[j*8: j * 8 + 8]
                byte_storage.append(int(byte_val, 2))

            if len(bin_rep_labels) % 8 != 0:
                byte_storage.append(int(bin_rep_labels[n_bytes*8:], 2))  # M: write the rest

            file.write(byte_storage)

            # M: write the unclustered biases
            cur_bias = cur_bias.view(-1).tolist()
            bias_format = ''.join(['f' for _ in range(len(cur_bias))])
            file.write(struct.pack(bias_format, *cur_bias))

        # M: Last Layer: Not quantized
        weights, biases = all_weights[-1].view(-1).tolist(), all_biases[-1].view(-1).tolist()
        weight_format = ''.join(['f' for _ in range(len(weights))])
        bias_format = ''.join(['f' for _ in range(len(biases))])
        file.write(struct.pack(weight_format, *weights))
        file.write(struct.pack(bias_format, *biases))

        file.flush()
        file.close()
