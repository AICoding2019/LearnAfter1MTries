
from scipy.io import wavfile
import numpy as np


def get_wav(file_name, nsamples=16000):
    wav = wavfile.read(file_name)[1]
    if wav.size < nsamples:
        d = np.pad(wav, (nsamples - wav.size, 0), mode='constant')
    else:
        d = wav[0:nsamples]
    return d
