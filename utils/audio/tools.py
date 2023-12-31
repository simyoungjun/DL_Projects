"""
	from NVIDIA's preprocessing

	reference)
		https://github.com/NVIDIA/tacotron2
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

from scipy.io.wavfile import read
from scipy.io.wavfile import write
import soundfile as sf 
import librosa

import scipy.signal as sps

import librosa
import os

from . import stft as stft
from .audio_preprocessing import griffin_lim
from config import Arguments as args


_stft = stft.TacotronSTFT(
    args.filter_length, args.hop_length, args.win_length,
    args.n_mels, args.sr, args.mel_fmin, args.mel_fmax)


def load_wav_to_numpy(full_path):
    # sampling_rate, data = read(full_path)
    data, sampling_rate = sf.read(full_path)
    # data, sampling_rate = librosa.load(full_path, sr= args.sr, mono=True)
    
    return data.astype(np.float32), sampling_rate


def get_mel(filename, trim_silence=False, frame_length=1024, hop_length=256, top_db=10):
    audio, sampling_rate = load_wav_to_numpy(filename)

    if sampling_rate != _stft.sampling_rate:
        raise ValueError("{} SR doesn't match target SR {}".format(
            sampling_rate, _stft.sampling_rate))
    audio_norm = audio / args.max_wav_value
    if trim_silence:
        audio_norm = audio_norm[200:-200]
        audio_norm, idx = librosa.effects.trim(audio_norm, top_db=top_db, frame_length=frame_length, hop_length=hop_length)
    
    
    audio_norm = torch.FloatTensor(audio_norm)
    audio_norm = audio_norm.unsqueeze(0)
    audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
    
    
    # print(filename)
    # print('Mel: ', torch.min(audio_norm, dim=1)[0], torch.max(audio_norm, dim=1)[0])
    # print('H: ', torch.min(harmonic_audio, dim=1)[0], torch.max(harmonic_audio, dim=1)[0])
    # print('p: ', torch.min(percussive_audio, dim=1)[0], torch.max(percussive_audio, dim=1)[0])

    
    melspec, energy  = _stft.mel_spectrogram(audio_norm)
    
    
    
    melspec = torch.squeeze(melspec, 0).detach().cpu().numpy().T
    energy = torch.squeeze(energy, 0).detach().cpu().numpy()

    
    # mel_plot = melspec.T
    # plt.figure()  # Define the figure size (optional)
    # plt.imshow(mel_plot, cmap='viridis')  # Plot the 2D array
    # plt.colorbar()  # Add a colorbar (optional)
    # plt.title('2D NumPy Array Plot')  # Add a title (optional)
    # plt.show()
    # plt.gca().invert_yaxis()
    # plt.savefig('mel[0].png')
    
    return melspec, energy

def get_mel_from_wav(audio):
    sampling_rate = args.sr
    if sampling_rate != _stft.sampling_rate:
        raise ValueError("{} {} SR doesn't match target {} SR".format(
            sampling_rate, _stft.sampling_rate))
    audio_norm = audio / args.max_wav_value
    audio_norm = audio_norm.unsqueeze(0)
    audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
    melspec, energy = _stft.mel_spectrogram(audio_norm)
    melspec = torch.squeeze(melspec, 0).detach().cpu().numpy().T
    energy = torch.squeeze(energy, 0).detach().cpu().numpy()


    return melspec, energy


def inv_mel_spec(mel, out_filename, griffin_iters=60):
    mel = torch.stack([mel])
    mel_decompress = _stft.spectral_de_normalize(mel)
    mel_decompress = mel_decompress.transpose(1, 2).data.cpu()
    spec_from_mel_scaling = 1000
    spec_from_mel = torch.mm(mel_decompress[0], _stft.mel_basis)
    spec_from_mel = spec_from_mel.transpose(0, 1).unsqueeze(0)
    spec_from_mel = spec_from_mel * spec_from_mel_scaling

    audio = griffin_lim(torch.autograd.Variable(
        spec_from_mel[:, :, :-1]), _stft.stft_fn, griffin_iters)

    audio = audio.squeeze()
    audio = audio.cpu().numpy()
    audio_path = out_filename
    write(audio_path, args.sr, audio)
