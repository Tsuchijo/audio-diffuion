import os 
from torchaudio import load, info
import spectrogram
import torch
import numpy as np
import torchaudio
import sqlite3

## Dataset for any arbitrary audio data
# Data should be split up into sections of samples a set length
# Assumes all audio is wav format 441000 sample rate
class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, frame_length, sample_rate=16000, device='cpu'):
        self.data_path = data_path
        self.data = os.listdir(data_path)
        self.data = [os.path.join(data_path, file) for file in self.data if '.wav' in file]
        self.spectrogram_transforms = spectrogram.Spectrogram(
            n_fft=1024,
            win_length=1024,
            hop_length=160,
            n_mels=64,
            sample_rate=sample_rate,
            device=device,
        )
        ## Iterate through all available audio files and get their metadata,
        ## Then record how many sections of audio of a specified length can be made
        self.frame_length = frame_length
        files_sizes = [metadata.num_frames for metadata in [info(file) for file in self.data]]
        self.num_frames = np.array([size//frame_length for size in files_sizes])
        self.cumulative_frames = np.cumsum(self.num_frames)
        self.device = device
        self.sample_rate = sample_rate
        self.resampler = torchaudio.transforms.Resample(44100, sample_rate, rolloff=0.9, lowpass_filter_width=1, dtype=torch.float32)

    def __len__(self):
        return sum(self.num_frames)
    
    def __getitem__(self, idx):
        ## Find the files that index i falls into 
        file_idx = np.argmax(self.cumulative_frames > idx)

        frame_start = (idx - self.cumulative_frames[file_idx-1] if file_idx > 0 else idx) * self.frame_length
        ## Load the audio the from the file
        audio_data, source_sample_rate = load(self.data[file_idx], frame_offset=frame_start, num_frames=self.frame_length, normalize=True)
        ## Downsample data from source sample rate to target sample rate
        ## Filter audio data with cutoff at 8000 hz
        audio_data = torchaudio.functional.lowpass_biquad(audio_data, source_sample_rate, cutoff_freq=8000)
        audio_data = torchaudio.functional.resample(audio_data, source_sample_rate, self.sample_rate)

        ## Transform audio into mono
        audio_data = torch.mean(audio_data, dim=0)
        ## Transform the audio data into a mel spectrogram
        mel_scale = self.spectrogram_transforms.mel_transform(audio_data.to(self.device))
        mel_scale =  torch.log(torch.clamp(mel_scale, min=1e-5)).swapaxes(0,1).unsqueeze(0)
        return mel_scale
    
class SQLiteDataset(torch.utils.data.Dataset):
    def __init__(self, db_path, table_name, shape):
        self.db_path = db_path
        self.table_name = table_name
        self.shape = shape

    def __getitem__(self, idx):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(f"SELECT * FROM {self.table_name} WHERE ID=?", (idx+1,))
        data_blob = cursor.fetchone()[1]
        conn.close()
        tensor_data = torch.from_numpy(np.reshape(np.frombuffer(data_blob, dtype=np.float32).copy(), self.shape))
        return tensor_data

    def __len__(self):
       conn = sqlite3.connect(self.db_path)
       cursor = conn.cursor()
       cursor.execute(f"SELECT COUNT(*) FROM {self.table_name}")
       length = cursor.fetchone()[0]
       conn.close()
       return length


