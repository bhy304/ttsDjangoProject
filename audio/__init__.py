import math
import tensorflow as tf
import numpy as np
from scipy import signal
from hparams import hparams

import librosa
import librosa.filters

def load_audio(path, pre_silence_length=0, post_silence_length=0):
    audio = librosa.core.load(path, sr=hparams.sample_rate)[0]
    if pre_silence_length > 0 or post_silence_length > 0:
        audio = np.concatenate([
                get_silence(pre_silence_length),
                audio,
                get_silence(post_silence_length),
        ])
    return audio

def save_audio(audio, path, sample_rate=None):
    audio *= 32767 / max(0.01, np.max(np.abs(audio)))
    librosa.output.write_wav(path, audio.astype(np.int16),
            hparams.sample_rate if sample_rate is None else sample_rate)

    print(" [*] Audio saved: {}".format(path))

def get_duration(audio):
    return librosa.core.get_duration(audio, sr=hparams.sample_rate)

def get_silence(sec):
    return np.zeros(hparams.sample_rate * sec)

def resample_audio(audio, target_sample_rate):
    return librosa.core.resample(
            audio, hparams.sample_rate, target_sample_rate)