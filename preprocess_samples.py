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
print('Loading Autoencoder')
VAE_config = yaml.load(open('checkpoints/16k_64.yaml', 'r'), Loader=yaml.FullLoader)
Autoencoder = AutoencoderKL(**VAE_config['model']['params'])
state_dict = torch.load('checkpoints/vae_mel_16k_64bins.ckpt')['state_dict']
state_dict = {k: v for k, v in state_dict.items() if not re.match('loss', k)}
Autoencoder.load_state_dict(state_dict)
Autoencoder.to('cuda')

print('Loading Data')
data_path = 'data/'
hop_length = 160
target_sample_rate = 16000
num_frames = 255
input_length = hop_length*num_frames*44100//target_sample_rate
n_mels = 64
batch_size = 1

## Load the dataloader
# TODO: Test if cpu or gpu loading is better
loader = audio_dataloader.AudioDataset(data_path, input_length, target_sample_rate,'cuda')
dataloader = DataLoader(loader, batch_size=batch_size, shuffle=False)

## Open connection to SQL databse locally
con  = sqlite3.connect(data_path + 'embedding.db')
cur = con.cursor()
cur.execute("CREATE TABLE IF NOT EXISTS embedding(ID INTEGER PRIMARY KEY, Data BLOB)")

iter_loader = iter(dataloader)
for i, data in enumerate(iter_loader):
    ## save the data
    print("Loading slice: ", i)
    tensor_data = Autoencoder.encode(data.to('cuda')).sample().to('cpu')
    for data_pair in tensor_data:
        tensor_datum = data_pair  # Assuming data_pair is your Torch tensor or data pair
        tensor_bytes = bytes(tensor_datum.cpu().detach().numpy())
        cur.execute("INSERT INTO embedding (Data) VALUES (?)", (tensor_bytes,))
        con.commit()  # Remember to commit the transaction after executing INSERT.
con.close()
