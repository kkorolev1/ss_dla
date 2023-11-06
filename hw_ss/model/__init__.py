import torchaudio.transforms as T
from hw_ss.model.spex_plus import SpexPlus


def normalize_audio(extracted_audio, gain=-23):
    norm = T.Vol(gain=gain, gain_type="db")
    return norm(extracted_audio)

__all__ = [
    "SpexPlus",
    "normalize_audio"
]
