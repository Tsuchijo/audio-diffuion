import torch
import math
import numpy as np

def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".

    Code taken from audioldm source code 
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

class Mel_MLP(torch.nn.Module):
    def __init__(self, in_mels, in_width, num_timesteps=1000, embedding_dim=4096, num_hidden = 5):
        super(Mel_MLP, self).__init__()
        ## Fully Connected model 
        self.fc_in = torch.nn.Linear(in_mels*in_width, embedding_dim)
        for i in range(num_hidden):
            setattr(self, f'fc{i}', torch.nn.Linear(embedding_dim*2, embedding_dim))
        self.fc_out = torch.nn.Linear(embedding_dim, in_mels*in_width)
        self.relu = torch.nn.ReLU()
        self.flatten = torch.nn.Flatten()
        self.unflatten = torch.nn.Unflatten(1, (in_mels, in_width))
        self.num_hidden = num_hidden
        ## Use Sinusoidal Embeddings for Timesteps injected into the embeddings dims
        # convert to device
        self.timestep_embedding = get_timestep_embedding(torch.arange(num_timesteps), embedding_dim)
        self.timestep_embedding = torch.nn.Parameter(self.timestep_embedding, requires_grad=False) 
    
    ## Forward Pass
    def forward(self, x, timesteps):
        x_flat = self.flatten(x)
        x1 = self.relu(self.fc_in(x_flat))
        for i in range(self.num_hidden):
            x1 = self.relu(getattr(self, f'fc{i}')(torch.cat((x1, self.timestep_embedding[timesteps]), dim=1)))
        x3 = self.fc_out(x1)
        return self.unflatten(x3)
    
class Latent_MLP(torch.nn.Module):
    def __init__(self, in_shape, num_timesteps=1000, embedding_dim=1024, num_hidden=3):
        super(Latent_MLP, self).__init__()
        ## Flatten input from input shape to 1D
        self.flatten = torch.nn.Flatten()
        self.unflatten = torch.nn.Unflatten(1, in_shape)
        in_length = np.prod(in_shape)
        self.fc_in = torch.nn.Linear(in_length, embedding_dim)
        for i in range(num_hidden):
            setattr(self, f'fc{i}', torch.nn.Linear(embedding_dim*2, embedding_dim))
        self.fc_out = torch.nn.Linear(embedding_dim, in_length)
        self.relu = torch.nn.ReLU()
        self.num_hidden = num_hidden
        ## Use Sinusoidal Embeddings for Timesteps injected into the embeddings dims
        # convert to device
        self.timestep_embedding = get_timestep_embedding(torch.arange(num_timesteps), embedding_dim)
        self.timestep_embedding = torch.nn.Parameter(self.timestep_embedding, requires_grad=False)
    
    ## Forward Pass
    def forward(self, x, timesteps):
        x_flat = self.flatten(x)
        x1 = self.relu(self.fc_in(x_flat))
        for i in range(self.num_hidden):
            x1 = self.relu(getattr(self, f'fc{i}')(torch.cat((x1, self.timestep_embedding[timesteps]), dim=1)))
        x3 = self.fc_out(x1)
        return self.unflatten(x3)


## Create U-Net architecture with Featurewose Linear Modulation to embed timesteps
class encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(encoder, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.maxpool(x)
        return x
    
class decoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(decoder, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')
        
    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return x


## Create a U-Net model, which projects down the data then projects it up
class Unet(torch.nn.Module):
    def __init__(self, num_timesteps=1000, embedding_dim=1024):
        self.enc1 = encoder(1, 64)
        self.enc2 = encoder(64, 128)
        self.enc3 = encoder(128, 256)

        self.midpoint = torch.nn.Conv2d(256, 256, 3, 1, 1)

        self.dec1 = decoder(256, 128)
        self.dec2 = decoder(128, 64)
        self.dec3 = decoder(64, 1)

        self.timestep_embedding = get_timestep_embedding(torch.arange(num_timesteps), embedding_dim)
        self.timestep_embedding = torch.nn.Parameter(self.timestep_embedding, requires_grad=False)

        ## Create linear FiLM networks which project timestep to two 6 dimensional vectors
        self.FiLMa = torch.nn.Linear(embedding_dim, 6)
        self. FiLMb = torch.nn.Linear(embedding_dim, 6)

    def forward(self, x, timesteps):
        a_embeddings = self.FiLMa(self.timestep_embedding[timesteps])
        b_embeddings = self.FiLMb(self.timestep_embedding[timesteps])

        e_x1 = self.enc1(x)
        e_x2 = self.enc2(a_embeddings[:, 0]*(e_x1)+b_embeddings[:, 0])
        e_x3 = self.enc2(a_embeddings[:, 1]*(e_x2)+b_embeddings[:, 1])

        d_x3 = self.midpoint(a_embeddings[:, 2]*(e_x3)+b_embeddings[:, 2])
        d_x2 = self.dec1(a_embeddings[:, 3]*(d_x3 + e_x3) + b_embeddings[:, 3])
        d_x1 = self.dec2(a_embeddings[:, 4]*(d_x2 + e_x2) + b_embeddings[:, 4])
        y = self.dec3(a_embeddings[:, 5]*(d_x1 + e_x1) + b_embeddings[:, 5])
        return y
