import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioPreprocessor(nn.Module):
    def __init__(self, dim=288):
        super(AudioPreprocessor, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=dim, kernel_size=127, stride=64, bias=False)
        self.conv2 = nn.Conv1d(in_channels=dim, out_channels=dim*2, kernel_size=7, stride=3, padding='valid', bias=False)
        self.conv3 = nn.Conv1d(in_channels=dim*2, out_channels=dim, kernel_size=3, stride=2, padding='valid', bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = F.tanh(x)
        x = self.conv2(x)
        x = F.gelu(x)

        return x