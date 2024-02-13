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
import torch.nn.init as init


def main():
    print('Loading Data')
    # Initialize all layers of a neural network with random values
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            m.weight.data.normal_(0, 1.0)

    data_path = 'data/'
    loader = audio_dataloader.AudioDataset(data_path, 512*100, 'cpu')
    dataloader = DataLoader(loader, batch_size=1, shuffle=True)
    model = Mel_MLP(80, 101, embedding_dim=4096, num_hidden=3)
    #model.apply(init_weights)	
    wandb.init(project='ddpm-audio')
    ## Start training loop
    ddpm = DDPM_Scheduler(
        t_total=1000,
        beta_min=0.0001,
        beta_max=0.02,
        model=model,
        device='cuda',
    )
    ddpm.train(dataloader, 10000)
    ## Save the model
    torch.save(model.state_dict(), 'model.pt')
    
main()
