import torch
import torch.nn as nn
from ConvLSTMCell import ConvLSTMCell


class ConvLSTM(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, padding=0, stride=1, dilation=1, bias=True):
        super(ConvLSTM, self).__init__()
        