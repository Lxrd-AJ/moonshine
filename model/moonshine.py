import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
from . import AudioPreprocessor, AudioEncoders, AudioDecoders, SAMPLE_RATE, DECODER_START_TOKEN, END_SPEECH_TOKEN

class MoonShine(nn.Module):
    def __init__(self, dim=288, innerDim=288, nHead=8, nEncoder=6, nDecoder=6, encMlpMult=4):
        super(MoonShine, self).__init__()
        self.preprocessor = AudioPreprocessor(dim=dim)
        maxSeqLen = 40
        self.encoders = AudioEncoders(dim=dim, innerDim=innerDim, nHead=nHead, maxSeqLen=maxSeqLen, dimMult=encMlpMult, nEncoder=nEncoder)
        self.decoders = AudioDecoders(dim=dim, innerDim=innerDim, nHead=nHead, maxSeqLen=maxSeqLen, dimMult=encMlpMult, nDecoder=nDecoder)

    def forward(self, audioIn):
        processedAudio = self.preprocessor(audioIn) # (B, C, T)
        processedAudio = processedAudio.transpose(1, 2) # (B, T, C)
        audioFeatures = self.encoders(processedAudio) # (B, T, C)

        numAudioSeconds = audioIn.shape[-1]
        maxTokensPerSecond = numAudioSeconds * 6
        
        tokens = torch.LongTensor([[DECODER_START_TOKEN]])
        print(tokens.shape)
        print(f"Encoder out (audio features) with shape {audioFeatures.shape}")
        logits = self.decoders(tokens, audioFeatures)
        
        return logits

    def loadWeights(self, preprocessorWeights, encoderWeights, decoderWeights):
        with h5py.File(preprocessorWeights, 'r') as f:
            self.preprocessor.loadWeights(f)

        with h5py.File(encoderWeights, 'r') as f:
            self.encoders.loadWeights(f)