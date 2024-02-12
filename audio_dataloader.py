import os 
from torchaudio import load, info
import spectrogram
import torch
import numpy as np

## Dataset for any arbitrary audio data
# Data should be split up into sections of samples a set length
# Assumes all audio is wav format 441000 sample rate
class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, frame_length, device='cpu'):
        self.data_path = data_path
        self.data = os.listdir(data_path)
        self.data = [os.path.join(data_path, file) for file in self.data]
        self.spectrogram_transforms = spectrogram.Spectrogram(
            n_fft=1024,
            win_length=1024,
            hop_length=512,
            n_mels=128,
            sample_rate=44100,
            device=device,
        )
        ## Iterate through all available audio files and get their metadata,
        ## Then record how many sections of audio of a specified length can be made
        self.frame_length = frame_length
        files_sizes = [metadata.num_frames for metadata in [info(file) for file in self.data]]
        self.num_frames = np.array([size//frame_length for size in files_sizes])
        self.cumulative_frames = np.cumsum(self.num_frames)
        self.device = device

    def __len__(self):
        return sum(self.num_frames)
    
    def __getitem__(self, idx):
        ## Find the files that index i falls into 
        file_idx = np.argmax(self.cumulative_frames > idx)

        frame_start = (idx - self.cumulative_frames[file_idx-1] if file_idx > 0 else idx) * self.frame_length
        ## Load the audio the from the file
        audio_data, _ = load(self.data[file_idx], frame_offset=frame_start, num_frames=self.frame_length)
        ## Transform audio into mono
        audio_data = torch.mean(audio_data, dim=0)
        ## Transform the audio data into a mel spectrogram
        mel_scale = self.spectrogram_transforms.mel_transform(audio_data.to(self.device))
        return mel_scale