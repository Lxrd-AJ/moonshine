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
        self.finalNorm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.layers(x)
        return self.finalNorm(x) # (B, T, C)

    def loadWeights(self, weights):
        self.layers[0].loadWeights(weights["layers/functional/layers"])
        self.layers[1].loadWeights(weights["layers/functional_1/layers"])
        self.layers[2].loadWeights(weights["layers/functional_2/layers"])
        self.layers[3].loadWeights(weights["layers/functional_3/layers"])
        self.layers[4].loadWeights(weights["layers/functional_4/layers"])
        self.layers[5].loadWeights(weights["layers/functional_5/layers"])
        self.finalNorm.weight.data = torch.from_numpy(weights["layers/layer_normalization/vars/0"][:]) # (288,)


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
        f2H5 = weights["functional/layers/sequential/layers/dense_1/vars/0"][:] # (1152, 288)
        f2BiasH5 = weights["functional/layers/sequential/layers/dense_1/vars/1"][:] # (288,)
        norm1ScaleH5 = weights["layer_normalization/vars/0"][:] # (288,)
        norm2ScaleH5 = weights["layer_normalization_1/vars/0"][:] # (288,)
        mhaKeyH5 = weights["mha_with_rope/key_dense/vars/0"][:] # (288, 8, 36)
        mhaOutputH5 = weights["mha_with_rope/output_dense/vars/0"][:] # (8, 36, 288)
        mhaQueryH5 = weights["mha_with_rope/query_dense/vars/0"][:] # (288, 8, 36)
        mhaValueH5 = weights["mha_with_rope/value_dense/vars/0"][:] # (288, 8, 36)

        self.mlp[0].weight.data = torch.from_numpy(f1H5.T) # (288, 1152) -> (1152, 288)
        self.mlp[0].bias.data = torch.from_numpy(f1BiasH5)

        self.mlp[2].weight.data = torch.from_numpy(f2H5.T) # (1152, 288) -> (288, 1152)
        self.mlp[2].bias.data = torch.from_numpy(f2BiasH5)

        self.norm1.weight.data = torch.from_numpy(norm1ScaleH5)
        self.norm2.weight.data = torch.from_numpy(norm2ScaleH5)

        self.sAttn.kProj.weight.data = torch.from_numpy(mhaKeyH5.reshape(288,-1).T) # (288, 8, 36) -> (288, 288).T -> (288, 288)
        self.sAttn.outProj.weight.data = torch.from_numpy(mhaOutputH5.reshape(288,-1).T) # (8, 36, 288) -> (288, 288).T -> (288, 288)
        self.sAttn.qProj.weight.data = torch.from_numpy(mhaQueryH5.reshape(288,-1).T) # (288, 8, 36) -> (288, 288).T -> (288, 288)
        self.sAttn.vProj.weight.data = torch.from_numpy(mhaValueH5.reshape(288,-1).T) # (288, 8, 36) -> (288, 288).T -> (288, 288)