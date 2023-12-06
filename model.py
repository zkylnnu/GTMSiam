import torch
import torch.nn as nn
import torch.nn.functional as F
import six
# import torchvision
# from torchvision import transforms
from sklearn import preprocessing
from sklearn.manifold import TSNE
import numpy as np
from scipy.io import savemat
from torch.autograd import Variable
from numba import jit
import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ConvGRU(nn.Module):
    def __init__(self, x_channels=100, channels=100):
        super(ConvGRU, self).__init__()
        self.channels = channels
        self.x_channels = x_channels

        self.conv_x_z = nn.Conv1d(in_channels=self.x_channels, out_channels=self.channels, kernel_size=1)
        self.conv_h_z = nn.Conv1d(in_channels=self.channels, out_channels=self.channels, kernel_size=1)
        self.conv_x_r = nn.Conv1d(in_channels=self.x_channels, out_channels=self.channels, kernel_size=1)

        self.conv_h_r = nn.Conv1d(in_channels=self.channels, out_channels=self.channels, kernel_size=1)
        self.conv = nn.Conv1d(in_channels=self.x_channels, out_channels=self.channels, kernel_size=1)

        self.conv_u = nn.Conv1d(in_channels=self.channels, out_channels=self.channels, kernel_size=1)
        self.lReLU = nn.LeakyReLU(0.2)

    def forward(self, x, y):

        z_t = F.sigmoid(self.conv_x_z(x) + self.conv_h_z(y))
        r_t = F.sigmoid((self.conv_x_r(x) + self.conv_h_r(y)))
        h_hat_t = self.lReLU(self.conv(x) + self.conv_u(torch.mul(r_t, y)))

        h_t = torch.mul((1 - z_t), y) + torch.mul(z_t, h_hat_t)
        return h_t


class GTMSiam(nn.Module):

    def __init__(self, max_len: int, n_channel: int, num_hidden: int, prembed_dim: int, batchsize: int):
        super(GTMSiam, self).__init__()
        self.max_len = max_len
        self.n_channel = n_channel
        self.prembed_dim = prembed_dim
        self.num_hidden = num_hidden
        self.batchsize = batchsize

        self.PE_linear = nn.Linear(self.n_channel, self.prembed_dim)
        self.PE_linear3 = nn.Linear(self.n_channel, self.prembed_dim)
        self.bn1 = nn.BatchNorm1d(25)

        self.CNN2 = nn.Conv2d(in_channels=self.prembed_dim, out_channels=self.num_hidden, kernel_size=1)
        self.CNN3 = nn.Conv2d(in_channels=self.prembed_dim, out_channels=self.num_hidden, kernel_size=3)
        self.CNN4 = nn.Conv2d(in_channels=self.prembed_dim, out_channels=self.num_hidden, kernel_size=3)
        self.CNN2A = nn.Conv2d(in_channels=self.prembed_dim, out_channels=self.num_hidden, kernel_size=1)
        self.CNN3A = nn.Conv2d(in_channels=self.prembed_dim, out_channels=self.num_hidden, kernel_size=3)
        self.CNN4A = nn.Conv2d(in_channels=self.prembed_dim, out_channels=self.num_hidden, kernel_size=3)
        self.siame_out = nn.Linear(self.prembed_dim,2)
        self.GRU1 = ConvGRU(self.prembed_dim, self.prembed_dim)
        self.GRU2 = ConvGRU(self.prembed_dim, self.prembed_dim)
        self.GRU3 = ConvGRU(self.prembed_dim, self.prembed_dim)


    def forward(self, x1, x2):

        emb_c1 = torch.cat([x1,x2],dim=2)
        emb_c2 = self.bn1(emb_c1)

        x1 = emb_c2[:,:,:self.n_channel]
        x2 = emb_c2[:,:,self.n_channel:2*self.n_channel]

        x3 = x1 - x2

        x1 = self.PE_linear3(x1)
        x1 = F.softmax(x1, -1)
        x2 = self.PE_linear3(x2)
        x2 = F.softmax(x2, -1)
        x3 = self.PE_linear(x3)
        x3 = F.softmax(x3, -1)

        x1 = torch.reshape(x1, [self.batchsize,5,5,self.prembed_dim])
        x2 = torch.reshape(x2, [self.batchsize,5,5,self.prembed_dim])
        x3 = torch.reshape(x3, [self.batchsize,5,5,self.prembed_dim])
        x1 = x1.permute(0, 3, 1, 2)
        x2 = x2.permute(0, 3, 1, 2)
        x3 = x3.permute(0, 3, 1, 2)

        x11 = self.CNN2(x1)
        x22 = self.CNN2(x2)
        x11 = torch.reshape(x11, [self.batchsize, self.prembed_dim, 25])
        x22 = torch.reshape(x22, [self.batchsize, self.prembed_dim, 25])
        x33 = self.GRU1(x11,x22)
        x3 = torch.reshape(x3, [self.batchsize, self.prembed_dim, 25])
        x3 = x3 + x33

        x11 = torch.reshape(x11, [self.batchsize, self.prembed_dim, 5, 5])
        x22 = torch.reshape(x22, [self.batchsize, self.prembed_dim, 5, 5])
        x111 = self.CNN3(x11)
        x222 = self.CNN3(x22)
        x111 = torch.reshape(x111, [self.batchsize, self.prembed_dim, 9])
        x222 = torch.reshape(x222, [self.batchsize, self.prembed_dim, 9])
        x333 = self.GRU2(x111, x222)
        x3 = torch.reshape(x3, [self.batchsize, self.prembed_dim, 5, 5])
        x3 = self.CNN3A(x3)
        x3 = torch.reshape(x3, [self.batchsize, self.prembed_dim, 9])
        x3 = x3  + x333

        x111 = torch.reshape(x111, [self.batchsize, self.prembed_dim, 3, 3])
        x222 = torch.reshape(x222, [self.batchsize, self.prembed_dim, 3, 3])
        x1111 = self.CNN4(x111)
        x2222 = self.CNN4(x222)
        x1111 = torch.reshape(x1111, [self.batchsize, self.prembed_dim, 1])
        x2222 = torch.reshape(x2222, [self.batchsize, self.prembed_dim, 1])
        x3333 = self.GRU3(x1111, x2222)
        x3 = torch.reshape(x3, [self.batchsize, self.prembed_dim, 3, 3])
        x3 = self.CNN4A(x3)
        x3 = torch.reshape(x3, [self.batchsize, self.prembed_dim, 1])
        x3 = x3 + x3333

        x3 = x3.reshape([self.batchsize, -1])
        logit = self.siame_out(x3)
        return logit
