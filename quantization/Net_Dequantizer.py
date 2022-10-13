import re
import math
import struct
import torch as th
from model.NeurcompModel import Neurcomp


# M: Taken from https://github.com/matthewberger/neurcomp
class NetDecoder:
    def __init__(self):
        pass

    def decode(self,filename):
        file = open(filename,'rb')

        # header: number of layers
        self.n_layers = struct.unpack('B', file.read(1))[0]
        # header: layers
        self.layer_sizes = struct.unpack(''.join(['I' for _ in range(self.n_layers)]), file.read(4 * (self.n_layers)))
        # header: number of bits for clustering
        self.n_bits = struct.unpack('B', file.read(1))[0]

        self.n_clusters = int(math.pow(2,self.n_bits))

        self.middle_layer_sizes = self.layer_sizes[1:self.n_layers-1]
        self.d_in = self.layer_sizes[0]
        self.d_out = self.layer_sizes[-1]

        # M: TODO: make this generic
        net = Neurcomp(input_ch=self.d_in, output_ch=self.d_out, features=self.middle_layer_sizes)

        # first layer: matrix and bias
        w_pos_format = ''.join(['f' for _ in range(self.d_in * self.middle_layer_sizes[0])])
        b_pos_format = ''.join(['f' for _ in range(self.middle_layer_sizes[0])])
        w_pos = th.FloatTensor(struct.unpack(w_pos_format, file.read(4 * self.d_in * self.middle_layer_sizes[0])))
        b_pos = th.FloatTensor(struct.unpack(b_pos_format, file.read(4 * self.middle_layer_sizes[0])))

        all_ws = [w_pos]
        all_bs = [b_pos]

        # middle layers: cluster, store clusters, then map matrix indices to indices
        total_n_layers = 2*(self.n_layers-2-1) # M: TODO: generalize this for middle layers: 2* because res
        res_idx = 0
        for ldx in range(total_n_layers):
            # weights
            n_weights = self.middle_layer_sizes[res_idx] * self.middle_layer_sizes[res_idx]
            weight_size = (n_weights*self.n_bits)//8
            if (n_weights*self.n_bits)%8 != 0:
                weight_size+=1
            c_format = ''.join(['f' for _ in range(self.n_clusters)])
            centers = th.FloatTensor(struct.unpack(c_format, file.read(4*self.n_clusters)))
            inds = file.read(weight_size)
            bits = ''.join(format(byte, '0'+str(8)+'b') for byte in inds)
            w_inds = th.LongTensor([int(bits[self.n_bits*i:self.n_bits*i+self.n_bits],2) for i in range(n_weights)])

            # bias
            b_format = ''.join(['f' for _ in range(self.middle_layer_sizes[res_idx])])
            bias = th.FloatTensor(struct.unpack(b_format, file.read(4 * self.middle_layer_sizes[res_idx])))
            bias = th.squeeze(bias)

            w_quant = centers[w_inds]
            all_ws.append(w_quant)
            all_bs.append(bias)

            if ldx % 2 != 0:
                res_idx += 1

        # last layer: matrix and bias
        w_last_format = ''.join(['f' for _ in range(self.d_out * self.middle_layer_sizes[-1])])
        b_last_format = ''.join(['f' for _ in range(self.d_out)])
        w_last = th.FloatTensor(struct.unpack(w_last_format, file.read(4 * self.d_out * self.middle_layer_sizes[-1])))
        b_last = th.FloatTensor(struct.unpack(b_last_format, file.read(4 * self.middle_layer_sizes[-1])))

        all_ws.append(w_last)
        all_bs.append(b_last)

        wdx,bdx=0,0
        for name, parameters in net.named_parameters():
            if re.match(r'.*.weight', name, re.I):
                w_shape = parameters.data.shape
                parameters.data = all_ws[wdx].view(w_shape)
                wdx+=1
            if re.match(r'.*.bias', name, re.I):
                b_shape = parameters.data.shape
                parameters.data = all_bs[bdx].view(b_shape)
                bdx+=1

        return net
