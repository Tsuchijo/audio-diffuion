import torch
import math

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
    
