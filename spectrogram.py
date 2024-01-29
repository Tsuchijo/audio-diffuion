import torch
import torchaudio.transforms as transforms

## Spectrogram Class to wrap the torch audio transforms
class Spectrogram():
    def __init__(
        self, 
        n_fft=1024,
        win_length=1024,
        hop_length=512,
        normalized=False,
        n_mels=128,
        sample_rate=44100
    ): 
        self.window_fn = torch.hann_window
        self.melspectrogram = transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min = 0.0,
             f_max = sample_rate // 2,
            normalized=normalized,
            window_fn=self.window_fn,
        )

        self.inverse_mel = transforms.InverseMelScale(
            sample_rate=sample_rate,
            n_stft=(n_fft//2 + 1),
            n_mels=n_mels,
            f_min = 0.0,
            f_max = sample_rate // 2,
        )

        self.griffin_lim = transforms.GriffinLim(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            window_fn=self.window_fn,
        )

    def mel_transform(self, input_data):
        return self.melspectrogram(input_data)
    
    def inverse_transform(self, input_data):
        return self.griffin_lim(self.inverse_mel(input_data))

