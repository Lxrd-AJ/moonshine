import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
from . import AudioPreprocessor
from . import AudioEncoders

class MoonShine(nn.Module):
    def __init__(self, dim=288, innerDim=288, nHead=8, nEncoder=6, nDecoder=6, encMlpMult=4):
        super(MoonShine, self).__init__()
        self.preprocessor = AudioPreprocessor(dim=dim)
        maxSeqLen = 40
        self.encoders = AudioEncoders(dim=dim, innerDim=innerDim, nHead=nHead, maxSeqLen=maxSeqLen, dimMult=encMlpMult)

    def forward(self, x):
        x = self.preprocessor(x) # (B, C, T)
        x = x.transpose(1, 2) # (B, T, C)
        x = self.encoders(x)
        
        return x

    def loadWeights(self, preprocessorWeights, encoderWeights, decoderWeights):
        with h5py.File(preprocessorWeights, 'r') as f:
            self.preprocessor.loadWeights(f)

        with h5py.File(encoderWeights, 'r') as f:
            self.encoders.loadWeights(f)