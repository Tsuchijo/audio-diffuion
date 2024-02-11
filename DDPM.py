import torch
import numpy as np
## wandb logging ##
import wandb

class DDPM_Scheduler:

    def __init__(self, t_total, beta_min, beta_max, model, device):
        self.device = device
        self.betas = torch.linspace(beta_min, beta_max, t_total).to(device).unsqueeze(1)
        self.alphas = 1.0 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)
        self.model = model
    


    ## Forward pass ##
    def forward(self, x0, timestep):
        alphas_t = self.alphas_bar[timestep]
        noise = torch.randn_like(x0)
        x_t = torch.sqrt(alphas_t) * x0 + torch.sqrt(1.0 - alphas_t) * noise
        return x_t, noise


    ## Training Loops, For now only works with 1 batch of input samples ##
    def train(self, x0, iters):
        cumulative_loss = []
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        loss_fn = torch.nn.MSELoss().to(self.device)
        for iteration in range(iters):
            timestep = (torch.randint(0, self.t_total-1, (x0.shape[0],)) ).to(self.device)
            x_t, eps = self.forward(x0, self.alpha_bar, timestep)

            ## Zero out gradients
            optimizer.zero_grad()
            eps_pred = self.model(torch.cat((x_t, (timestep.float() / self.t_total).unsqueeze(1)), dim=1))
            ## MSE loss between eps and eps predicted
            loss = loss_fn(eps_pred, eps)
            loss.backward()
            optimizer.step()
            cumulative_loss.append(loss.item())
        return cumulative_loss
        

    ## Inference Pass ##
    def inference(self, x0):
        with torch.no_grad():
            x_T = torch.randn_like(x0)
            x_T_half = None
            x_T_start = x_T
            for t in reversed(range(self.t_total)):
                self.model.zero_grad()
                alpha_bar_t = self.alphas_bar[t]
                alpha_t = self.alphas[t]
                timestep = (torch.ones((x0.shape[0],)) * float(t) / self.t_total).to(self.device).unsqueeze(1).float()
                epsilon = self.model(torch.cat((x_T, timestep), dim=1))

                noise = torch.randn_like(x0).to(self.device)
                print('Iteration: ', t, end='\r')
                if t > 0:
                    beta_bar_t = (1.0 - self.alphas_bar[t-1]) / (1.0 - alpha_bar_t) * (1 - alpha_t)
                    x_T = (1.0 / torch.sqrt(alpha_t)) * (x_T - (((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * epsilon)) + torch.sqrt(1 - alpha_t) * noise
                else:
                    x_T = (1.0 / torch.sqrt(alpha_t)) * (x_T - (((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * epsilon))
            return x_T, x_T_half, x_T_start