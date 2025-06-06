from .preprocessor import AudioPreprocessor
from .encoders import AudioEncoders, EncoderBlock
from .transformer import MultiHeadSelfAttentionRoPE
from .moonshine import MoonShine
import torch

def diskSizeOf(model):
    numBytesPerParameter = torch.tensor([], dtype=torch.float32).element_size() # A 32-bit float takes 4 bytes
    numParams = sum(p.numel() for p in model.parameters())
    tensorSizeBytes = numParams * numBytesPerParameter
    # human readable size
    if tensorSizeBytes < 1024:
        return f"{tensorSizeBytes} bytes", numParams
    elif tensorSizeBytes < 1024**2:
        return f"{tensorSizeBytes / 1024:.2f} KB", numParams
    elif tensorSizeBytes < 1024**3:
        return f"{tensorSizeBytes / 1024**2:.2f} MB", numParams
    else:
        return f"{tensorSizeBytes / 1024**3:.2f} GB", numParams

def downloadPretrainedWeights(variant = "tiny"):
    from huggingface_hub import hf_hub_download
    repo = "UsefulSensors/moonshine"
    return (
        hf_hub_download(repo, f"{x}.weights.h5", subfolder=variant)
        for x in ("preprocessor", "encoder", "decoder")
    )