import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioPreprocessor(nn.Module):
    def __init__(self, dim=288):
        super(AudioPreprocessor, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=dim, kernel_size=127, stride=64, bias=False)
        self.groupNorm = nn.GroupNorm(num_groups=1, num_channels=dim)
        self.conv2 = nn.Conv1d(in_channels=dim, out_channels=dim*2, kernel_size=7, stride=3, padding='valid', bias=True)
        self.conv3 = nn.Conv1d(in_channels=dim*2, out_channels=dim, kernel_size=3, stride=2, padding='valid', bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = F.tanh(x)
        x = self.groupNorm(x)
        x = self.conv2(x)
        x = F.gelu(x)
        x = self.conv3(x)
        x = F.gelu(x)

        return x

    def loadWeights(self, weights):
        conv1H5 = weights["layers/sequential/layers/conv1d/vars/0"][:] # (127, 1, 288)
        conv2H5 = weights["layers/sequential/layers/conv1d_1/vars/0"][:] # (7, 288, 576)
        conv2BiasH5 = weights["layers/sequential/layers/conv1d_1/vars/1"][:] # (576,)
        conv3H5 = weights["layers/sequential/layers/conv1d_2/vars/0"][:] # (3, 576, 288)
        conv3BiasH5 = weights["layers/sequential/layers/conv1d_2/vars/1"][:] # (288,)
        groupNormScaleH5 = weights["layers/sequential/layers/group_normalization/vars/0"][:] # (288,)
        groupNormShiftH5 = weights["layers/sequential/layers/group_normalization/vars/1"][:] # (288,)
        
        self.conv1.weight.data = torch.from_numpy(conv1H5.transpose(2, 1, 0)) # (127, 1, 288) -> (288, 1, 127)
        self.conv2.weight.data = torch.from_numpy(conv2H5.transpose(2, 1, 0)) # (7, 288, 576) -> (576, 288, 7)
        self.conv2.bias.data = torch.from_numpy(conv2BiasH5)
        self.conv3.weight.data = torch.from_numpy(conv3H5.transpose(2, 1, 0)) # (3, 576, 288) -> (288, 576, 3)
        self.conv3.bias.data = torch.from_numpy(conv3BiasH5)
        self.groupNorm.weight.data = torch.from_numpy(groupNormScaleH5)
        self.groupNorm.bias.data = torch.from_numpy(groupNormShiftH5)