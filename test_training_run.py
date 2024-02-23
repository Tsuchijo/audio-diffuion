# !/usr/bin/env python3
import numpy as np 
import torch
import os 
import re
import spectrogram
import torchaudio 
from scipy.io import wavfile
from Models.models import Latent_MLP
from DDPM import DDPM_Scheduler
import wandb
import audio_dataloader
from torch.utils.data import DataLoader
import torch.nn.init as init
from AudioLDM_Models.variational_autoencoder.autoencoder import AutoencoderKL
import yaml


def main():
    print('Loading Data')

    ## Load Autoencoder from weights 
    VAE_config = yaml.load(open('checkpoints/16k_64.yaml', 'r'), Loader=yaml.FullLoader)
    Autoencoder = AutoencoderKL(**VAE_config['model']['params'])
    state_dict = torch.load('checkpoints/vae_mel_16k_64bins.ckpt')['state_dict']
    state_dict = {k: v for k, v in state_dict.items() if not re.match('loss', k)}
    Autoencoder.load_state_dict(state_dict)
    Autoencoder.encoder.to('cpu')
    data_path = 'data/'
    hop_length = 160
    target_sample_rate = 16000
    num_frames = 255
    input_length = hop_length*num_frames*44100//target_sample_rate

    loader = audio_dataloader.AudioDataset(data_path, input_length, target_sample_rate,'cpu')
    dataloader = DataLoader(loader, batch_size=1000, shuffle=True, num_workers=8)
    model = Latent_MLP((8,num_frames//4,16), embedding_dim=1024, num_hidden=2)
    wandb.init(project='ddpm-audio')
    ## Start training loop
    ddpm = DDPM_Scheduler(
        t_total=1000,
        beta_min=0.0001,
        beta_max=0.02,
        model=model,
        autoencoder=Autoencoder,
        device='cuda',
    )
    ddpm.train(dataloader, 600000)
    ## Save the model
    torch.save(model.state_dict(), 'model.pt')
    
main()
