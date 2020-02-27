import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from ConvLSTM import ConvLSTM
import matplotlib.pyplot as plt

EPOCH = 10
BATCH_SIZE = 10
TIME_STEP = 28//4
INPUT_SIZE = (4, 28)
LR = 0.001
DOWNLOAD_MNIST = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data = torchvision.datasets.MNIST(
    root='../MNIST',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST
)

train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE)

test_data = torchvision.datasets.MNIST(
    root='../MNIST',
    train=False,
)
test_x = Variable(torch.unsqueeze(test_data.data, dim=1)).type(torch.FloatTensor)[:200] / 255
test_y = test_data.targets.numpy()[:200]


class HandwrittenClassifier(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers, padding="SAME"):
        super(HandwrittenClassifier, self).__init__()
        self.input_size = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.padding = padding
        self.clstm = ConvLSTM(input_size=self.input_size,
                              input_dim=self.input_dim,
                              hidden_dim=self.hidden_dim,
                              kernel_size=self.kernel_size,
                              num_layers=self.num_layers,
                              padding=self.padding)
        self.out = nn.Linear(in_features=self.hidden_dim*self.input_size[0]*self.input_size[1], out_features=10)

    def forward(self, input_x):
        """
        :param input_x: (batch,time_steps,channel,height,weight)
        :return: a list filled with probability of each category
        """
        clstm_out, state = self.clstm(input_x)
        last_clstm_out = clstm_out[:, -1, :, :, :]
        # print(self.hidden_dim, self.input_size, last_clstm_out.size())
        classify_out = self.out(last_clstm_out.view(-1, self.hidden_dim*self.input_size[0]*self.input_size[1]))

        return classify_out


# print(next(iter(train_loader))[0].size())
handC = HandwrittenClassifier(input_size=INPUT_SIZE, input_dim=1, hidden_dim=3, kernel_size=3, num_layers=4).to(device=DEVICE)
optimizer = torch.optim.Adam(handC.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for batch, (train_x, train_y) in enumerate(train_loader):
        # train_x original shape: (batch_size, channel, height, width), (10, 1, 28, 28), need transform to (b,t,c,h,w)
        train_x = train_x.view(BATCH_SIZE, TIME_STEP, train_x.size(1), INPUT_SIZE[0], INPUT_SIZE[1]).to(DEVICE)
        train_y = train_y.to(DEVICE)
        out = handC(train_x)
        # print(out.size())
        # print(train_y.size())
        loss = loss_func(out, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            test_out = handC(test_x.view(200, TIME_STEP, test_x.size(1), INPUT_SIZE[0], INPUT_SIZE[1]).cuda())
            predict_y = torch.max(test_out, 1)[1].cpu().numpy()
            accuracy = sum(predict_y == test_y) / test_y.size
            print("EPOCH:", epoch, "| train loss: %.4f" % loss.item(), "test accuracy:", accuracy)
