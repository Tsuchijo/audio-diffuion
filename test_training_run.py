# !/usr/bin/env python3
import numpy as np 
import torch
import os 
import re
import spectrogram
import torchaudio 
from scipy.io import wavfile
from Models.models import Mel_Convolv
from DDPM import DDPM_Scheduler


def main():
    data_path = 'data/'
    ## Only Diffusiing from a single sample (TODO: Load many samples)
    sample_name = os.listdir(data_path)[0]
    # Specify the path to the .wav file
    wav_file = os.path.join(data_path, sample_name)

    # Load the .wav file as a numpy array
    sample_rate, audio_data = wavfile.read(wav_file)
    # Avg together Left and Right channels to get Mono Signal
    audio_data = np.mean(audio_data, axis=1)
    ## Transform to torch tensor
    audio_data = torch.from_numpy(audio_data).float().unsqueeze(0)
    spectrogram_transforms = spectrogram.Spectrogram(
        n_fft=1024,
        win_length=1024,
        hop_length=512,
        n_mels=80,
        sample_rate=sample_rate,
    )
    mel_scale = spectrogram_transforms.mel_transform(audio_data)
    model = Mel_Convolv(80, 1470)

    # Reshape from 1x80xn to (n//1470)x80x1470
    # Calculate the desired size after reshaping
    N = mel_scale.size(2)
    new_size = (N // 1480, 1480, 80)
    remaining = N % 1480
    ## cut the remaining part
    mel_scale = mel_scale[:, :, :-remaining]
    # Reshape the padded tensor
    mel_scale = mel_scale.swapdims(1,2).reshape(new_size).swapdims(1,2)

    ## Start training loop
    ddpm = DDPM_Scheduler(
        t_total=1000,
        beta_min=1e-3,
        beta_max=0.1,
        model=model,
    )
    cumulative_loss = ddpm.train(mel_scale, 1000)
    ## Save the model
    torch.save(model.state_dict(), 'model.pt')
    
