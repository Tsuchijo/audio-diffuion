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
from torchaudio import transforms
import sqlite3
 
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
n_mels = 64
batch_size = 100

## Load the dataloader
# TODO: Test if cpu or gpu loading is better
loader = audio_dataloader.AudioDataset(data_path, input_length, target_sample_rate,'cpu')
dataloader = DataLoader(loader, batch_size=batch_size, shuffle=False, num_workers=8)

## Open connection to SQL databse locally
con  = sqlite3.connect('embedding.db')
cur = con.cursor()
cur.execute("CREATE TABLE embedding(index, embedding)")

iter_loader = iter(dataloader)
for i, data in enumerate(iter_loader):
    ## save the data
    indices = np.arange(i * batch_size,(i+1)*batch_size)
    data_pairs  = [(index, embedding) for index, embedding in zip(indices, data)]
    cur.executemany("INSERT INTO embedding VALUES(?, ?)", data)
    con.commit()  # Remember to commit the transaction after executing INSERT.

