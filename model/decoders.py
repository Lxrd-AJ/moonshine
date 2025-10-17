import torch
import torch.nn as nn
from .transformer import MultiHeadSelfAttentionRoPE, MultiHeadCrossAttention

class AudioDecoders(nn.Module):
    def __init__(self, dim=288, innerDim=288, nHead=8, maxSeqLen=40, dimMult=4, nDecoder=6, vocabSize=32768):
        super(AudioDecoders, self).__init__()
        
        self.tokenEmbedding = nn.Embedding(vocabSize, dim)
        self.layers = nn.Sequential(
            *[DecoderBlock(dim=dim, innerDim=innerDim, nHead=nHead, maxSeqLen=maxSeqLen, dimMult=dimMult) 
                for _ in range(nDecoder)]
        )
        self.finalNorm = nn.LayerNorm(dim)
        self.reverseEmbedding = nn.Linear(dim, vocabSize, bias=False)

        # weight sharing between the embedding and reverse embedding layers
        self.reverseEmbedding.weight = self.tokenEmbedding.weight

    def forward(self, tokens, encoderHiddenState):
        # tokens: (B, T) where B is batch size and T is sequence length
        # encoderHiddenState: (B, T, C) where C is the dimension of the hidden state
        x = self.tokenEmbedding(tokens)  # (B, T, C)
        print(f"Embedded token size {x.shape}")
        for decoder in self.layers:
            x = decoder(x, encoderHiddenState)  # (B, T, C)
        x = self.finalNorm(x)
        logits = self.reverseEmbedding(x) # (B, T, vocabSize)
        return logits


class DecoderBlock(nn.Module):
    def __init__(self, dim=288, innerDim=288, nHead=8, maxSeqLen=40, dimMult=4):
        super(DecoderBlock, self).__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.selfAttn = MultiHeadSelfAttentionRoPE(dim, innerDim, nHead, bias=False, maxSeqLen=maxSeqLen, isCausal=True)
        self.norm2 = nn.LayerNorm(dim)
        self.crossAttn = MultiHeadCrossAttention(dim, innerDim, nHead, bias=False)
        self.norm3 = nn.LayerNorm(dim)
        self.ffSwiglu = nn.Sequential(
            nn.Linear(dim, dim * dimMult * 2),
            nn.SiLU(),
            nn.Linear(dim * dimMult * 2, dim)
        )

    def forward(self, x, encoderHiddenState):
        xSkip = x
        
        x = self.norm1(x)
        x = self.selfAttn(x)
        x = x + xSkip
        
        xSkip = x
        x = self.norm2(x)
        x = self.crossAttn(x, encoderHiddenState, encoderHiddenState)
        x = x + xSkip
        
        xSkip = x
        x = self.norm3(x)
        x = self.ffSwiglu(x)
        
        return x + xSkip

    def loadWeights(self, weights):
        # Implement the logic to load weights from a given source
        pass