# !/usr/bin/env python3
import numpy as np 
import torch
import os 
import re
import spectrogram
import torchaudio 
from scipy.io import wavfile
from Models.models import Mel_MLP
from DDPM import DDPM_Scheduler
import wandb
import audio_dataloader
from torch.utils.data import DataLoader

def main():
    print('Loading Data')
    data_path = 'data/'
    loader = audio_dataloader.AudioDataset(data_path, 512*500, 'cpu')
    dataloader = DataLoader(loader, batch_size=10, shuffle=True)
    model = Mel_MLP(80, 501)

   
    wandb.init(project='ddpm-audio')
    ## Start training loop
    ddpm = DDPM_Scheduler(
        t_total=1000,
        beta_min=1e-4,
        beta_max=0.02,
        model=model,
        device='cuda',
    )
    ddpm.train(dataloader, 10000)
    ## Save the model
    torch.save(model.state_dict(), 'model.pt')
    
main()