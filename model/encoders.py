import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer import MultiHeadSelfAttentionRoPE

class AudioEncoders(nn.Module):
    def __init__(self, dim=288, innerDim=288, nHead=8, maxSeqLen=40, dimMult=4, nEncoder=6):
        super(AudioEncoders, self).__init__()
        self.layers = nn.Sequential(
            *[EncoderBlock(dim=dim, innerDim=innerDim, nHead=nHead, maxSeqLen=maxSeqLen, dimMult=dimMult) 
                for _ in range(nEncoder)]
            )

    def forward(self, x):
        return self.layers(x)

    def loadWeights(self, weights):
        self.layers[0].loadWeights(weights["layers/functional/layers"])
        # TODO: Load the remaining layers
        # self.layers[0].loadWeights(weights["layers/functional/layers"])
        # self.layers[0].loadWeights(weights["layers/functional/layers"])
        # self.layers[0].loadWeights(weights["layers/functional/layers"])
        # self.layers[0].loadWeights(weights["layers/functional/layers"])
        # self.layers[0].loadWeights(weights["layers/functional/layers"])

class EncoderBlock(nn.Module):
    def __init__(self, dim=288, innerDim=288, nHead=8, maxSeqLen=40, dimMult=4):
        super(EncoderBlock, self).__init__()
        # inputs from audio preprocessor are of shape (B, C, T) but need to be reshaped to (B, T, C)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.sAttn = MultiHeadSelfAttentionRoPE(dim, innerDim, nHead, bias=False, maxSeqLen=maxSeqLen)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim*dimMult),
            nn.GELU(),
            nn.Linear(dim*dimMult, dim)
        )

    def forward(self, x):
        # x: (B, T, C)
        xSkip = x
        x = self.norm1(x)
        x = self.sAttn(x) # (B, T, C)
        x = x + xSkip
        xSkip = x
        x = self.norm2(x)
        x = self.mlp(x)

        return x + xSkip

    def loadWeights(self, weights):
        f1H5 = weights["functional/layers/sequential/layers/dense/vars/0"][:] # (288, 1152)
        f1BiasH5 = weights["functional/layers/sequential/layers/dense/vars/1"][:] # (1152,)
        
        # functional/layers/sequential/layers/dense_1/vars/0: shape=(1152, 288), dtype=float32
        # functional/layers/sequential/layers/dense_1/vars/1: shape=(288,), dtype=float32
        # layer_normalization/vars/0: shape=(288,), dtype=float32
        # layer_normalization_1/vars/0: shape=(288,), dtype=float32
        # mha_with_rope/key_dense/vars/0: shape=(288, 8, 36), dtype=float32
        # mha_with_rope/output_dense/vars/0: shape=(8, 36, 288), dtype=float32
        # mha_with_rope/query_dense/vars/0: shape=(288, 8, 36), dtype=float32
        # mha_with_rope/value_dense/vars/0: shape=(288, 8, 36), dtype=float32

        self.mlp[0].weight.data = torch.from_numpy(f1H5.T) # (288, 1152) -> (1152, 288)
        self.mlp[0].bias.data = torch.from_numpy(f1BiasH5)
