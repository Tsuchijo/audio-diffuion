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
        self.model = model.to(device)
        self.t_total = t_total

    ## Forward pass ##
    def forward(self, x0, timestep):
        alphas_t = self.alphas_bar[timestep].unsqueeze(1)
        noise = torch.randn_like(x0)
        x_t = torch.sqrt(alphas_t) * x0 + torch.sqrt(1.0 - alphas_t) * noise
        return x_t, noise


    ## Training Loops, For now only works with 1 batch of input samples ##
    def train(self, dataloader, iters):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-2)
        loss_fn = torch.nn.MSELoss().to(self.device)
        x0 = next(iter(dataloader)).to(self.device)
        ## randomly initialize weights
        for m in self.model.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        # Tile the original Sample 10000 times
        x0 = x0.repeat(60000, 1, 1)

        for iteration in range(iters):
            timestep = (torch.randint(0, self.t_total-1, (x0.shape[0],)) ).to(self.device)
            x_t, eps = self.forward(x0, timestep)
            eps_pred = self.model(x_t.unsqueeze(1), timestep.to(self.device))
            ## MSE loss between eps and eps predicted
            loss = loss_fn(eps_pred, eps)
            loss.backward()
            optimizer.step()
            # Zero out gradients
            optimizer.zero_grad()

            ## Log to wandb
            wandb.log({'Loss': loss.item()})
            if iteration % 250 == 0:
                mel_image = x_t[0] - (((1 - self.alphas[timestep[0]]) / torch.sqrt(1 - self.alphas_bar[timestep[0]])) * eps[0])
                # convert to numpy and add channel
                mel_image = np.log(np.abs(np.expand_dims(mel_image.to('cpu').numpy(), axis=-1)))
                noise_diff = np.expand_dims((eps_pred[0] - eps[0]).detach().to('cpu').numpy(), axis=-1)
                wandb.log({"Denoised Spectrograph": wandb.Image(mel_image)})
                wandb.log({"Noise Difference": wandb.Image(noise_diff)})
            if iteration % 10000 == 0:
                ## Run 1 sampling pass
                mel_image = self.inference(x0[0].unsqueeze(0))
                mel_image = np.log(np.abs(np.expand_dims(mel_image.to('cpu').numpy(), axis=-1)))
                wandb.log({"Sampled Spectrograph": wandb.Image(mel_image)})
            print('Iteration: ', iteration, 'Loss: ', loss.item(), end='\r')

        

    ## Inference Pass ##
    def inference(self, x0):
        with torch.no_grad():
            x_T = torch.randn_like(x0).to(self.device)
            for t in reversed(range(self.t_total)):
                self.model.zero_grad()
                alpha_bar_t = self.alphas_bar[t]
                alpha_t = self.alphas[t]
                timestep = (torch.ones((x0.shape[0],)) * t).long().to(self.device)
                epsilon = self.model(x_T.unsqueeze(1), timestep)
                noise = torch.randn_like(x0).to(self.device)
                print('Iteration: ', t, end='\r')
                if t > 0:
                    x_T = (1.0 / torch.sqrt(alpha_t)) * (x_T - (((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * epsilon)) + torch.sqrt(1 - alpha_t) * noise
                else:
                    x_T = (1.0 / torch.sqrt(alpha_t)) * (x_T - (((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * epsilon))
            return x_T
