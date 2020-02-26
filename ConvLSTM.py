import torch
import torch.nn as nn
from ConvLSTMCell import ConvLSTMCell


class ConvLSTM(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size,
                 num_layers=1, padding=0, stride=1, dilation=1, bias=True):
        """
        :param input_size: (h,w)
        :param input_dim: the channel of input xt in the first layer
        :param hidden_dim: the channel of state for all layers, this is a single number,
        we will extend it for the multi layers by transform it to a list
        :param num_layers: the layer of ConvLSTM
        :param padding: has the "SAME" pattern
        """

        super(ConvLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = [hidden_dim]*self.num_layers if not isinstance(hidden_dim, list) else hidden_dim
        self.kernel_size = [kernel_size]*self.num_layers if not isinstance(kernel_size, list) else hidden_dim

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_size=input_size,
                                          input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          padding=padding,
                                          stride=stride,
                                          dilation=dilation,
                                          bias=bias))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_x, init_state):
        c0, h0 = init_state

        time_

        for layer in range(self.num_layers):

