import torch
import torch.nn as nn
import torch.nn.functional as F
from . import AudioPreprocessor
from . import EncoderBlock

class MoonShine(nn.Module):
    def __init__(self, dim=288, innerDim=288, nHead=8, nEncoder=6, nDecoder=6):
        super(MoonShine, self).__init__()
        self.preprocessor = AudioPreprocessor(dim=dim)
        maxSeqLen = 40
        self.encoders = nn.Sequential(*[EncoderBlock(dim=dim, innerDim=innerDim, nHead=nHead, maxSeqLen=maxSeqLen) for _ in range(nEncoder)])
        

    def forward(self, x):
        x = self.preprocessor(x) # (B, C, T)
        x = x.transpose(1, 2) # (B, T, C)
        x = self.encoders(x)
        
        return x