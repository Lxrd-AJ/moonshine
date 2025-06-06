import torch
import torch.nn as nn
import torch.nn.functional as F

@torch.no_grad()
def generateRoPE(seqLen, dim, base=10000) -> tuple["torch.Tensor", "torch.Tensor"]:
    # seqLen -> T, dim -> C
    positions = torch.arange(seqLen).unsqueeze(1) # (T, 1)
    halfChannels = dim // 2
    invFreq = 1.0 / (base ** (torch.arange(0, halfChannels)/halfChannels)) # (C/2,)
    theta = positions * invFreq # (T, 1) * (C/2,) -> (T, C/2)
    return torch.cos(theta).unsqueeze(0).unsqueeze(0), torch.sin(theta).unsqueeze(0).unsqueeze(0) # both are (1, 1, T, C/2)

class MultiHeadSelfAttentionRoPE(nn.Module):
    def __init__(self, dim=288, innerDim=288, nHead=8, bias=False, maxSeqLen=128):
        super(MultiHeadSelfAttentionRoPE, self).__init__()
        """
        Reimplement multi head attention with RoPe, adapted from 
        * https://github.com/huggingface/transformers/blob/42ef218b58de79415ab45377a1e8de8dca3929f0/src/transformers/models/roformer/modeling_roformer.py#L189
        * https://github.com/huggingface/transformers/blob/42ef218b58de79415ab45377a1e8de8dca3929f0/src/transformers/models/llama/modeling_llama.py#L197 
        * https://github.com/Lxrd-AJ/positional-encoding/blob/main/PositionalEncoding.ipynb

        Uses `cos` and `sin` to perform the rotations, not as elegant as llama's complex implementation but mathematically the 
        same
        """
        self.headDim = innerDim // nHead
        self.nHead = nHead
        self.maxSeqLen = maxSeqLen
        self.qProj = nn.Linear(dim, innerDim, bias=bias)
        self.kProj = nn.Linear(dim, innerDim, bias=bias)
        self.vProj = nn.Linear(dim, innerDim, bias=bias)
        self.outProj = nn.Linear(innerDim, dim, bias=bias)

        cosEmb, sinEmb = generateRoPE(maxSeqLen, self.headDim)
        self.register_buffer("cosinePositionEmbeddings", cosEmb, persistent=False)
        self.register_buffer("sinePositionEmbeddings", sinEmb, persistent=False)


    def forward(self, x): # x -> (B, T, C)
        B, T, _ = x.shape
        if T > self.maxSeqLen:
            raise ValueError(f"Input sequence length ({T}) exceeds maximum supported sequence length ({self.maxSeqLen}).")

        xQ = self.qProj(x).view(B, T, self.nHead, self.headDim).transpose(1, 2) # (B, T, nH, hC) -> (B, nH, T, hC)
        xK = self.kProj(x).view(B, T, self.nHead, self.headDim).transpose(1, 2) # (B, T, nH, hC) -> (B, nH, T, hC)
        xV = self.vProj(x).view(B, T, self.nHead, self.headDim).transpose(1, 2) # (B, T, nH, hC) -> (B, nH, T, hC)
        
        rQ, rK = self.applyRoPE(xQ, xK, self.cosinePositionEmbeddings, self.sinePositionEmbeddings)
        attn = F.scaled_dot_product_attention(rQ, rK, xV)
        attn = attn.transpose(1, 2).reshape(B, T, -1) # (B, nH, T, hC) -> (B, T, nH, hC) -> (B, T, C)
        out = self.outProj(attn)
        return out

    def applyRoPE(self, q, k, cos, sin):
        def rotate(x):
            # x: (B, nH, T, hC)
            # cos, sin: (1, 1, T, hC/2)
            seqLen = x.shape[2]
            cos_ = cos[..., :seqLen, :]
            sin_ = sin[..., :seqLen, :]
            xR = torch.zeros_like(x)
            xR[..., 0::2] = (x[..., 0::2] * cos_) - (x[..., 1::2] * sin_)
            xR[..., 1::2] = (x[..., 1::2] * cos_) + (x[..., 0::2] * sin_)
            return xR

        return rotate(q), rotate(k)


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