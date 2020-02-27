import torch
import torch.nn as nn
from ConvLSTMCell import ConvLSTMCell


class ConvLSTM(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size,
                 num_layers=1, padding=0, stride=1, dilation=1, bias=True, batch_first=True):
        """
        :param input_size: (h,w)
        :param input_dim: the channel of input xt in the first layer
        :param hidden_dim: the channel of state for all layers, this is a single number,
        we will extend it for the multi layers by transform it to a list
        :param num_layers: the layer of ConvLSTM
        :param padding: the padding of all th layers
        :param batch_first: batch is the first dim if true
        """

        super(ConvLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = [hidden_dim]*self.num_layers if not isinstance(hidden_dim, list) else hidden_dim
        self.kernel_size = [kernel_size]*self.num_layers if not isinstance(kernel_size, list) else hidden_dim
        self.padding = [padding]*self.num_layers if not isinstance(padding, list) else padding
        self.batch_first = batch_first

        cell_list = []
        for i in range(self.num_layers):
            cur_input_dim = input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_size=input_size,
                                          input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          padding=self.padding[i],
                                          stride=stride,
                                          dilation=dilation,
                                          bias=bias))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_x, state=None):
        """
        :param input_x: the total input data, (batch,time_steps,channel,height,weight)
        :param state: the initial state include hidden state h and cell state c
        :return: outputs (batch,time_steps,channel,height,weight), last_state (shape of h is the same with c, shape: (b,c,h,w))
        """

        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_x = input_x.permute(1, 0, 2, 3, 4)

        batch_size = input_x.size(0)
        time_step = input_x.size(1)

        state = self.init_state(batch_size)

        for layer in range(self.num_layers):
            c, h = state[layer]
            t_output = []
            for t in range(time_step):
                cur_input = input_x[:, t, :, :, :]
                c, h = self.cell_list[layer](xt=cur_input, state=(c, h))
                t_output.append(h)

            layer_output = torch.stack(t_output, dim=1)
            input_x = layer_output

        # take the last layer's output and state as output
        outputs = layer_output
        last_state = (c, h)

        return outputs, last_state

    def init_state(self, batch_size):
        init_state = []
        for layer in range(self.num_layers):
            init_state.append(self.cell_list[layer].init_state(batch_size))
        return init_state