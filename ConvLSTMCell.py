import torch
import torch.nn as nn
from torch.autograd import Variable


class ConvLSTMCell(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, padding="SAME", stride=1, dilation=1, bias=True):
        """
        :param input_size: (h,w)
        :param input_dim: the channel of input xt
        :param hidden_dim: the channel of state h and c
        :param padding: add "SAME" pattern
        """

        super(ConvLSTMCell, self).__init__()
        self.input_size = input_size if isinstance(input_size, tuple) else (input_size, input_size)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)

        # Because Pytorch has no "SAME" pattern, I add this pattern
        padding = padding if isinstance(padding, tuple) else (padding, padding)
        if padding[0] == "SAME":
            padding_h = ((self.input_size[0] - 1) * self.stride[0] - self.input_size[0] + self.kernel_size[0] + (self.kernel_size[0] - 1) \
                        * (self.dilation[0] - 1))//2
        if padding[1] == "SAME":
            padding_w = ((self.input_size[1] - 1) * self.stride[1] - self.input_size[1] + self.kernel_size[1] + (self.kernel_size[1] - 1) \
                        * (self.dilation[1] - 1))//2
        self.padding = (padding_h, padding_w)

        # in_channels, out_channels, kernel_size, stride=1, padding=0,
        # dilation=1, groups=1, bias=True, padding_mode='zeros'
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                      out_channels=4 * self.hidden_dim,
                      kernel_size=self.kernel_size,
                      padding=self.padding,
                      stride=stride,
                      dilation=dilation,
                      bias=bias),
            # nn.ReLU()
        )

    def forward(self, xt, state):
        """
        :param xt: (b,c,h,w)
        :param state: include c(t-1) and h(t-1)
        :return: c_next, h_next
        """
        c, h = state

        # concatenate h and xt along channel axis
        com_input = torch.cat([xt, h], dim=1)
        com_outputs = self.conv(com_input)
        temp_i, temp_f, temp_o, temp_g = torch.split(com_outputs, self.hidden_dim, dim=1)

        i = torch.sigmoid(temp_i)
        f = torch.sigmoid(temp_f)
        o = torch.sigmoid(temp_o)
        g = torch.tanh(temp_g)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        return c_next, h_next

    def init_state(self, batch_size):
        return (Variable(torch.zeros(batch_size, self.hidden_dim, self.input_size[0], self.input_size[1])).cuda(),
                Variable(torch.zeros(batch_size, self.hidden_dim, self.input_size[0], self.input_size[1])).cuda())
