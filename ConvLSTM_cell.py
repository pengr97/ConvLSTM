import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    def __init__(self,input_size,input_dim,hidden_dim,kernel_size,padding=0,stride=1,dilation=1,bias=True):
        super(ConvLSTMCell, self).__init__()
        self.input_size = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = padding

        # Pytorch has no "SAME" pattern
        if self.padding == "SAME":
            self.padding = (input_size-1)*stride-input_size+kernel_size+(kernel_size-1)*(dilation-1)

        # in_channels, out_channels, kernel_size, stride=1, padding=0,
        # dilation=1, groups=1, bias=True, padding_mode='zeros'
        self.conv = nn.Conv2d(in_channels=self.input_dim+self.hidden_dim,
                              out_channels=4*self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              stride=stride,
                              dilation=dilation,
                              bias=bias)

    # xt shape: b,c,h,w
    # state include c(t-1) and h(t-1)
    def forward(self, xt, state):
        c, h = state

        # concatenate h and xt along channel axis
        con_input = torch.cat([xt, h], dim=1)
        com_outputs = self.conv(con_input)
        temp_i, temp_f, temp_o, temp_g = torch.split(com_outputs, dim=1)

        i = torch.sigmoid(temp_i)
        f = torch.sigmoid(temp_f)
        o = torch.sigmoid(temp_o)
        g = torch.tanh(temp_g)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        return c_next, h_next

