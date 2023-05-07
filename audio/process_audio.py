import numpy as np
import librosa as lb
import librosa.display as lbd
import soundfile as sf
from  soundfile import SoundFile
import pandas as pd
from IPython.display import Audio
from pathlib import Path

from matplotlib import pyplot as plt

from tqdm import tqdm
import joblib, json, re
import os
from os import path

from sklearn.model_selection import StratifiedKFold


class Config:
    sampling_rate = 32000
    train_duration = 15
    val_duration = 5
    fmin = 50
    fmax = 14000


def get_audio_info(filepath):
    """Get some properties from  an audio file"""
    with SoundFile(filepath) as f:
        sr = f.samplerate
        frames = f.frames
        duration = float(frames) / sr
    return {"frames": frames, "sr": sr, "duration": duration}


def add_path_df(df):
    df["path"] = [str(Config.audios_path / filename) for filename in df.filename]
    df = df.reset_index(drop=True)
    pool = joblib.Parallel(2)
    mapper = joblib.delayed(get_audio_info)
    tasks = [mapper(filepath) for filepath in df.path]
    df2 = pd.DataFrame(pool(tqdm(tasks))).reset_index(drop=True)
    df = pd.concat([df, df2], axis=1).reset_index(drop=True)

    return df


def compute_melspec(y, sr, n_mels, fmin, fmax):
    """
    Computes a mel-spectrogram and puts it at decibel scale
    Arguments:
        y {np array} -- signal
        params {AudioParams} -- Parameters to use for the spectrogram. Expected to have the attributes sr, n_mels, f_min, f_max
    Returns:
        np array -- Mel-spectrogram
    """
    melspec = lb.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, fmin=fmin, fmax=fmax,
    )

    melspec = lb.power_to_db(melspec).astype(np.float32)
    return melspec


def mono_to_color(X, eps=1e-6, mean=None, std=None):
    mean = mean or X.mean()
    std = std or X.std()
    X = (X - mean) / (std + eps)

    _min, _max = X.min(), X.max()

    if (_max - _min) > eps:
        V = np.clip(X, _min, _max)
        V = 255 * (V - _min) / (_max - _min)
        V = V.astype(np.uint8)
    else:
        V = np.zeros_like(X, dtype=np.uint8)

    return V


def crop_or_pad(y, length, is_train=True, start=None):
    # y -> audio clip
    # length -> audio length
    # print("shape:", y.shape)
    if len(y) < length:
        # y = np.pad(y, pad_width=((length - len(y),0), (0,0)), mode='constant', constant_values=0)
        y = np.concatenate([y, np.zeros(length - len(y))])
        n_repeats = length // len(y)
        epsilon = length % len(y)

        y = np.concatenate([y] * n_repeats + [y[:epsilon]])

    elif len(y) > length:
        if not is_train:
            start = start or 0
        else:
            start = start or np.random.randint(len(y) - length)

        y = y[start:start + length]

    return y

class AudioToImage:
    def __init__(self, sr=Config.sampling_rate, n_mels=128, fmin=Config.fmin, fmax=Config.fmax,
                 res_type="kaiser_fast", resample=True, dataset_dir="/kaggle/input"):

        self.sr = sr
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax or self.sr//2


        self.step = None

        self.res_type = res_type
        self.resample = resample

        self.dataset_dir = dataset_dir
    def audio_to_image(self, audio):
        melspec = compute_melspec(audio, self.sr, self.n_mels, self.fmin, self.fmax )
        image = mono_to_color(melspec)
        #         compute_melspec(y, sr, n_mels, fmin, fmax)
        return image

    def __call__(self, row, save=True):

        is_train = (row.fold != 0)

        self.step = int(Config.train_duration * 0.666 * Config.sampling_rate) if is_train \
               else int(Config.val_duration * Config.sampling_rate)

        duration = Config.train_duration if is_train else Config.val_duration
        audio_length = duration * self.sr

        data_year = row['data_year']
        filename = os.path.join(self.dataset_dir, f"birdclef-{data_year}",
                                "train_audio" if data_year != 2021 else "train_short_audio", row["primary_label"],
                                row['filename'].split("/")[-1])
        output_filename = filename.replace(".ogg", f"_{duration}.npy")

        if os.path.exists(output_filename):
            print("skip: ", output_filename)
            return

        # print("F:", filename, "   ", is_train)

        audio, orig_sr = sf.read(filename, dtype="float32")

        # print("orig_sr:", orig_sr, "  ", filename)

        if self.resample and orig_sr != self.sr:
            #print("resample")
            audio = lb.resample(audio, orig_sr, self.sr, res_type=self.res_type)

        # 双声道
        if audio.ndim == 2 and audio.shape[1] == 2:
            audio = np.mean(audio, axis=1)

        audios = [audio[i: i + audio_length] for i in range(0, max(1, len(audio) - audio_length + 1), self.step)]
        audios[-1] = crop_or_pad(audios[-1] , length=audio_length)
        images = [self.audio_to_image(audio) for audio in audios]
        images = np.stack(images)

        if save:
            temp_file = "/tmp/temp.npy"
            np.save(temp_file, images)
            # Use the sudo command to copy the temporary file to the destination location with root permission
            os.system("sudo cp {} {}".format(temp_file, output_filename))

            # np.save(str(output_filename), images)
        else:
            return row.filename, images


def get_audios_as_images(df):
    pool = joblib.Parallel(4)
    converter = AudioToImage()
    mapper = joblib.delayed(converter)
    tasks = [mapper(row) for idx, row in df.iterrows()]
    pool(tqdm(tasks))