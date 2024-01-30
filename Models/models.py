import torch

## Define Model Archtechture for processing Mel Spectrograph
# Takes in a simple 2D image of some parametrized L x W
# Returns a 2D image of the same size

class Mel_Convolv(torch.nn.Module):
    def __init__(self, in_mels, in_width):
        super(Mel_Convolv, self).__init__()
        ## Simple Model For now 5 skip connections going from 80x1470 back to 80x1470 with bottleneck in the middle
        ## 80 x 1470 x 1 -> 19 x 367 x 64
        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(9,9), stride=(4,4), padding=(4,4))
        ## 19 x 367 x 64 -> 10 x 184 x 128
        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=(3,3), stride=(2,2), padding=(1,1))
        ## Flatten out to 1D linear layer 10 x 184 x 128 -> 1 x 1 x 1470
        self.fc1 = torch.nn.Linear(10*184*128, 1470)
        ## unflatten to 10x184x128
        self.fc2 = torch.nn.Linear(1470, 10*184*128)
        ## 10 x 184 x 128 -> 19 x 367 x 64
        self.conv3 = torch.nn.ConvTranspose2d(128, 64, kernel_size=(3,3), stride=(2,2), padding=(1,1))
        ## 19 x 367 x 64 -> 80 x 1470 x 1
        self.conv4 = torch.nn.ConvTranspose2d(64, 1, kernel_size=(9,9), stride=(4,4), padding=(4,4))
        self.relu = torch.nn.ReLU()
    
    ## Forward Pass using U-Net like architecture
    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.fc1(x2.view(-1, 10*184*128)))
        x4 = self.relu(self.fc2(x3))
        x5 = self.relu(self.conv3(x4.view(-1, 128, 10, 184) + x2))
        x6 = self.conv4(x5 + x1)
        return x6
